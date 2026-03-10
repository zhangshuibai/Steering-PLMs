"""
Train a thermostability (Tm) predictor using ESM2-650M features + RobertaLMHead-style head.
Architecture: Linear(1280,1280) -> GELU -> LayerNorm -> Linear(1280,1)
Regression task with MSE loss; evaluated by Spearman correlation (target ρ=0.76).

Paper-consistent data processing:
1. Meltome Atlas data, 100-900 AA, Tm 30-98°C
2. 90/10 random train/test split
3. CD-HIT 90% identity within training set
4. Remove training sequences with >=30% identity to any test sequence
5. Median Tm across species per protein

Supports multi-GPU parallel feature extraction.
"""

import argparse
import os
import subprocess
import sys
import tempfile

# Ensure conda env bin is in PATH
CONDA_BIN = os.path.join(os.path.dirname(sys.executable), "")
os.environ["PATH"] = CONDA_BIN + os.pathsep + os.environ.get("PATH", "")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import esm


# ========================= Model =========================

class ThermPredictor(nn.Module):
    """Same architecture as ESM2's RobertaLMHead, but output_dim=1 for regression."""
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x.squeeze(-1)


class LinearPredictor(nn.Module):
    """Simple linear probe."""
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# ========================= CD-HIT Utilities =========================

def write_fasta(seqs, path, prefix="seq"):
    """Write sequences to a FASTA file."""
    with open(path, 'w') as f:
        for i, seq in enumerate(seqs):
            f.write(f">{prefix}_{i}\n{seq}\n")


def parse_cdhit_clusters(clstr_file):
    """Parse CD-HIT .clstr file, return set of representative sequence indices."""
    reps = set()
    with open(clstr_file) as f:
        for line in f:
            if line.startswith('>'):
                continue
            if '*' in line:
                # This is the representative
                name = line.split('>')[1].split('...')[0]
                idx = int(name.split('_')[1])
                reps.add(idx)
    return reps


def cdhit_filter(seqs, identity=0.9, word_size=5):
    """Run CD-HIT to cluster sequences and return indices of representatives."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_fa = os.path.join(tmpdir, "input.fasta")
        output_fa = os.path.join(tmpdir, "output")
        write_fasta(seqs, input_fa)

        cmd = [
            "cd-hit", "-i", input_fa, "-o", output_fa,
            "-c", str(identity), "-n", str(word_size),
            "-M", "16000", "-T", "8", "-d", "0"
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, check=True)

        reps = parse_cdhit_clusters(output_fa + ".clstr")
    return sorted(reps)


def cdhit_2d_filter(db_seqs, query_seqs, identity=0.3, word_size=2):
    """Run CD-HIT-2D to find db sequences similar to query sequences.
    Returns indices of db sequences that are NOT similar to any query sequence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_fa = os.path.join(tmpdir, "db.fasta")
        query_fa = os.path.join(tmpdir, "query.fasta")
        output_fa = os.path.join(tmpdir, "output")

        write_fasta(db_seqs, db_fa, prefix="db")
        write_fasta(query_seqs, query_fa, prefix="query")

        cmd = [
            "cd-hit-2d", "-i", query_fa, "-i2", db_fa, "-o", output_fa,
            "-c", str(identity), "-n", str(word_size),
            "-M", "16000", "-T", "8", "-d", "0"
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, check=True)

        # Parse output: sequences in output_fa are those from db NOT matching query
        surviving = set()
        with open(output_fa) as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().lstrip('>')
                    idx = int(name.split('_')[1])
                    surviving.add(idx)
    return sorted(surviving)


# ========================= Feature Extraction =========================

def extract_features_single_gpu(seqs, model, alphabet, device, batch_size=8,
                                  max_len=1022, last_n_layers=1, desc="Extracting"):
    """Extract mean-pooled representations from ESM2."""
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    repr_layers = list(range(n_layers - last_n_layers + 1, n_layers + 1))
    all_features = []

    n_batches = (len(seqs) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(seqs), batch_size), total=n_batches, desc=desc):
        batch_seqs = seqs[start:start + batch_size]
        batch_seqs = [s[:max_len] for s in batch_seqs]
        data = [("protein", s) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=repr_layers)

        for i, seq_len in enumerate(batch_lens):
            layer_reps = []
            for layer in repr_layers:
                token_reps = results["representations"][layer]
                rep = token_reps[i, 1:seq_len - 1].mean(0)
                layer_reps.append(rep)
            rep = torch.stack(layer_reps).mean(0).cpu()
            all_features.append(rep)

    return torch.stack(all_features)


def _worker_extract(rank, gpu_ids, seqs_chunk, batch_size, max_len, last_n_layers, output_path):
    """Worker function for multi-GPU feature extraction."""
    device = f"cuda:{gpu_ids[rank]}"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    features = extract_features_single_gpu(
        seqs_chunk, model, alphabet, device, batch_size, max_len,
        last_n_layers=last_n_layers, desc=f"GPU {gpu_ids[rank]}"
    )
    torch.save(features, output_path)
    del model
    torch.cuda.empty_cache()


def extract_features_multi_gpu(seqs, gpu_ids, batch_size=8, max_len=1022,
                                last_n_layers=1, cache_dir="/tmp/esm2_feat_cache"):
    """Extract features in parallel across multiple GPUs."""
    os.makedirs(cache_dir, exist_ok=True)
    n_gpus = len(gpu_ids)

    chunk_size = (len(seqs) + n_gpus - 1) // n_gpus
    chunks = [seqs[i * chunk_size: (i + 1) * chunk_size] for i in range(n_gpus)]
    chunks = [c for c in chunks if len(c) > 0]
    actual_n = len(chunks)

    output_paths = [os.path.join(cache_dir, f"chunk_{i}.pt") for i in range(actual_n)]

    processes = []
    mp.set_start_method("spawn", force=True)
    for rank in range(actual_n):
        p = mp.Process(
            target=_worker_extract,
            args=(rank, gpu_ids, chunks[rank], batch_size, max_len, last_n_layers, output_paths[rank])
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_features = []
    for path in output_paths:
        all_features.append(torch.load(path))
        os.remove(path)

    return torch.cat(all_features, dim=0)


# ========================= Training =========================

def train_predictor(features_train, labels_train, features_val, labels_val,
                    embed_dim=1280, epochs=50, lr=1e-4, weight_decay=1e-2,
                    batch_size=256, patience=10, head='lm_head', device='cuda'):
    if head == 'lm_head':
        model = ThermPredictor(embed_dim).to(device)
    else:
        model = LinearPredictor(embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(features_train, labels_train.float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_spearman = -1
    best_state = None
    no_improve = 0

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()
        total_loss = 0
        for feat, lab in train_loader:
            feat, lab = feat.to(device), lab.to(device)
            preds = model(feat)
            loss = criterion(preds, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.size(0)

        avg_loss = total_loss / len(train_dataset)

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(features_val.to(device)).cpu().numpy()
            val_labels = labels_val.numpy()

        rho, pval = spearmanr(val_labels, val_preds)
        mae = np.mean(np.abs(val_labels - val_preds))

        if rho > best_spearman:
            best_spearman = rho
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        pbar.set_postfix(loss=f"{avg_loss:.4f}", rho=f"{rho:.4f}", mae=f"{mae:.2f}", best_rho=f"{best_spearman:.4f}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# ========================= Main =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/meltome/dataset_esm_temp.csv')
    parser.add_argument('--save_path', type=str, default='saved_predictors/therm_predictor_final.pt')
    parser.add_argument('--features_dir', type=str, default='saved_predictors/therm_features_v2',
                        help='Directory to cache extracted ESM2 features')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[4, 5, 6, 7],
                        help='GPU IDs for parallel feature extraction')
    parser.add_argument('--batch_size_extract', type=int, default=8)
    parser.add_argument('--batch_size_train', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--head', type=str, default='lm_head', choices=['lm_head', 'linear'])
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Fraction of data for test set')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--last_n_layers', type=int, default=1,
                        help='Number of last layers to average for features')
    parser.add_argument('--skip_cdhit', action='store_true',
                        help='Skip CD-HIT filtering (use random split only)')
    parser.add_argument('--no_val', action='store_true',
                        help='Do not split val from train; train on all non-test data')
    args = parser.parse_args()

    train_device = f"cuda:{args.gpu_ids[0]}"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    # --- Load data ---
    print("=" * 60)
    print("Loading data...")
    df = pd.read_csv(args.data_path)
    seqs = df['protein_sequence'].tolist()
    tms = df['tm'].tolist()
    print(f"Total samples: {len(seqs)}")

    # --- Check for cached split ---
    split_path = os.path.join(args.features_dir, "split_data.pt")
    if os.path.exists(split_path):
        print(f"Loading cached split from {split_path}")
        split_data = torch.load(split_path)
        train_seqs = split_data['train_seqs']
        val_seqs = split_data['val_seqs']
        test_seqs = split_data['test_seqs']
        train_tms = split_data['train_tms']
        val_tms = split_data['val_tms']
        test_tms = split_data['test_tms']
    else:
        # --- Step 1: 90/10 random split ---
        print("\nStep 1: 90/10 random train/test split...")
        indices = list(range(len(seqs)))
        train_idx, test_idx = train_test_split(indices, test_size=args.test_ratio, random_state=args.seed)
        print(f"  Before filtering: Train={len(train_idx)}, Test={len(test_idx)}")

        train_seqs_raw = [seqs[i] for i in train_idx]
        test_seqs = [seqs[i] for i in test_idx]
        train_tms_raw = [tms[i] for i in train_idx]
        test_tms = [tms[i] for i in test_idx]

        if not args.skip_cdhit:
            # --- Step 2: CD-HIT 90% within training set ---
            print("\nStep 2: CD-HIT 90% identity clustering within training set...")
            rep_indices = cdhit_filter(train_seqs_raw, identity=0.9, word_size=5)
            train_seqs_dedup = [train_seqs_raw[i] for i in rep_indices]
            train_tms_dedup = [train_tms_raw[i] for i in rep_indices]
            print(f"  After CD-HIT 90%: {len(train_seqs_raw)} -> {len(train_seqs_dedup)} train sequences")

            # --- Step 3: Remove train sequences with >=40% identity to test ---
            # Note: CD-HIT minimum threshold is 40% (word_size=2). Paper says 30% but
            # CD-HIT cannot go below 40%. Using 40% as closest achievable threshold.
            print("\nStep 3: Removing train sequences with >=40% identity to test set...")
            surviving = cdhit_2d_filter(train_seqs_dedup, test_seqs, identity=0.4, word_size=2)
            train_seqs_clean = [train_seqs_dedup[i] for i in surviving]
            train_tms_clean = [train_tms_dedup[i] for i in surviving]
            print(f"  After identity filter: {len(train_seqs_dedup)} -> {len(train_seqs_clean)} train sequences")
        else:
            print("\nSkipping CD-HIT filtering (--skip_cdhit)")
            train_seqs_clean = train_seqs_raw
            train_tms_clean = train_tms_raw

        # --- Step 4: Split remaining train into train/val ---
        if args.no_val:
            print("\nStep 4: Using all training data (no val split)...")
            train_seqs = train_seqs_clean
            val_seqs = train_seqs_clean[:500]  # use first 500 as dummy val for monitoring
            train_tms = train_tms_clean
            val_tms = train_tms_clean[:500]
        else:
            print("\nStep 4: Splitting train into train/val (90/10)...")
            n_train = len(train_seqs_clean)
            tv_indices = list(range(n_train))
            tr_idx, vl_idx = train_test_split(tv_indices, test_size=0.1, random_state=args.seed)
            train_seqs = [train_seqs_clean[i] for i in tr_idx]
            val_seqs = [train_seqs_clean[i] for i in vl_idx]
            train_tms = [train_tms_clean[i] for i in tr_idx]
            val_tms = [train_tms_clean[i] for i in vl_idx]

        # Cache the split
        torch.save({
            'train_seqs': train_seqs, 'val_seqs': val_seqs, 'test_seqs': test_seqs,
            'train_tms': train_tms, 'val_tms': val_tms, 'test_tms': test_tms,
        }, split_path)

    print(f"\nFinal split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")
    print(f"  Train Tm: mean={np.mean(train_tms):.1f}, std={np.std(train_tms):.1f}")
    print(f"  Test  Tm: mean={np.mean(test_tms):.1f}, std={np.std(test_tms):.1f}")

    # --- Extract or load cached features ---
    suffix = f"_last{args.last_n_layers}" if args.last_n_layers > 1 else ""
    train_feat_path = f"{args.features_dir}/train_features{suffix}.pt"
    val_feat_path = f"{args.features_dir}/val_features{suffix}.pt"
    test_feat_path = f"{args.features_dir}/test_features{suffix}.pt"

    if os.path.exists(train_feat_path) and os.path.exists(val_feat_path) and os.path.exists(test_feat_path):
        print(f"\nLoading cached features (last_n_layers={args.last_n_layers})...")
        train_features = torch.load(train_feat_path)
        val_features = torch.load(val_feat_path)
        test_features = torch.load(test_feat_path)
    else:
        print(f"\nExtracting features using GPUs: {args.gpu_ids}, last_n_layers={args.last_n_layers}")

        print("Extracting train features...")
        train_features = extract_features_multi_gpu(
            train_seqs, args.gpu_ids, args.batch_size_extract,
            last_n_layers=args.last_n_layers,
            cache_dir=os.path.join(args.features_dir, "tmp"))
        torch.save(train_features, train_feat_path)
        print(f"  Saved train features: {train_features.shape}")

        print("Extracting val features...")
        val_features = extract_features_multi_gpu(
            val_seqs, args.gpu_ids, args.batch_size_extract,
            last_n_layers=args.last_n_layers,
            cache_dir=os.path.join(args.features_dir, "tmp"))
        torch.save(val_features, val_feat_path)
        print(f"  Saved val features: {val_features.shape}")

        print("Extracting test features...")
        test_features = extract_features_multi_gpu(
            test_seqs, args.gpu_ids, args.batch_size_extract,
            last_n_layers=args.last_n_layers,
            cache_dir=os.path.join(args.features_dir, "tmp"))
        torch.save(test_features, test_feat_path)
        print(f"  Saved test features: {test_features.shape}")

    train_labels_t = torch.tensor(train_tms)
    val_labels_t = torch.tensor(val_tms)
    test_labels_t = torch.tensor(test_tms)

    # --- Train ---
    print("\n" + "=" * 60)
    print("Training predictor...")
    print(f"  Head: {args.head}, LR: {args.lr}, WD: {args.weight_decay}, BS: {args.batch_size_train}")
    predictor = train_predictor(
        train_features, train_labels_t,
        val_features, val_labels_t,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size_train, patience=args.patience,
        head=args.head, device=train_device
    )

    # --- Evaluate on val ---
    print("\nVal set evaluation:")
    predictor.eval()
    with torch.no_grad():
        val_preds = predictor(val_features.to(train_device)).cpu().numpy()
        val_labels_np = val_labels_t.numpy()
    rho_val, _ = spearmanr(val_labels_np, val_preds)
    mae_val = np.mean(np.abs(val_labels_np - val_preds))
    print(f"  Spearman ρ={rho_val:.4f}  MAE={mae_val:.2f}°C")

    # --- Evaluate on test ---
    print("\nTest set evaluation:")
    with torch.no_grad():
        test_preds = predictor(test_features.to(train_device)).cpu().numpy()
        test_labels_np = test_labels_t.numpy()

    rho, pval = spearmanr(test_labels_np, test_preds)
    mae = np.mean(np.abs(test_labels_np - test_preds))
    rmse = np.sqrt(np.mean((test_labels_np - test_preds) ** 2))
    print(f"  Spearman ρ={rho:.4f} (p={pval:.2e})  MAE={mae:.2f}°C  RMSE={rmse:.2f}°C")
    print(f"  Target: Spearman ρ >= 0.76")

    # --- Save ---
    torch.save(predictor.state_dict(), args.save_path)
    print(f"\nPredictor saved to {args.save_path}")
    print("=" * 60)
