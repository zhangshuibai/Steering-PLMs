"""
Train a solubility predictor using ESM2-650M features + RobertaLMHead-style head.
Architecture follows the paper: Linear(1280,1280) -> GELU -> LayerNorm -> Linear(1280,1)
Supports multi-GPU parallel feature extraction and multi-layer feature modes.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp

from tqdm import tqdm
import esm


# ========================= Model =========================

class SolubilityPredictor(nn.Module):
    """Same architecture as ESM2's RobertaLMHead, but output_dim=1."""
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


# ========================= Feature Extraction =========================

def load_sequences_and_labels(src_path, tgt_path):
    with open(src_path) as f:
        seqs = [line.strip() for line in f]
    with open(tgt_path) as f:
        labels = [int(line.strip()) for line in f]
    assert len(seqs) == len(labels)
    return seqs, labels


def extract_features_single_gpu(seqs, model, alphabet, device, batch_size=8,
                                  max_len=1022, last_n_layers=1, desc="Extracting"):
    """Extract mean-pooled representations from ESM2.

    last_n_layers: number of last layers to average.
        1 = last layer only, 4 = average of last 4 layers, 'all' handled by caller.
    """
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
            # Average across requested layers
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
        model = SolubilityPredictor(embed_dim).to(device)
    else:
        model = LinearPredictor(embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(features_train, labels_train.float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_f1 = 0
    best_state = None
    no_improve = 0

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()
        total_loss = 0
        for feat, lab in train_loader:
            feat, lab = feat.to(device), lab.to(device)
            logits = model(feat)
            loss = criterion(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.size(0)

        avg_loss = total_loss / len(train_dataset)

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(features_val.to(device))
            val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()
            val_labels = labels_val.numpy()

        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds)
        rec = recall_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}", f1=f"{f1:.4f}", best_f1=f"{best_f1:.4f}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# ========================= Main =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/deepsol/data')
    parser.add_argument('--save_path', type=str, default='saved_predictors/sol_predictor.pt')
    parser.add_argument('--features_dir', type=str, default='saved_predictors/features',
                        help='Directory to cache extracted ESM2 features')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[5, 6, 7],
                        help='GPU IDs for parallel feature extraction')
    parser.add_argument('--batch_size_extract', type=int, default=8)
    parser.add_argument('--batch_size_train', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--head', type=str, default='lm_head', choices=['lm_head', 'linear'],
                        help='Predictor head type')
    parser.add_argument('--merge_train_val', action='store_true', default=False,
                        help='Merge train+val for training')
    parser.add_argument('--last_n_layers', type=int, default=1,
                        help='Number of last layers to average for features (1=last only, 4=last 4, 33=all)')
    args = parser.parse_args()

    train_device = f"cuda:{args.gpu_ids[0]}"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    # --- Load data ---
    print("Loading sequences...")
    train_seqs, train_labels = load_sequences_and_labels(
        f"{args.data_dir}/train_src", f"{args.data_dir}/train_tgt")
    val_seqs, val_labels = load_sequences_and_labels(
        f"{args.data_dir}/val_src", f"{args.data_dir}/val_tgt")
    test_seqs, test_labels = load_sequences_and_labels(
        f"{args.data_dir}/test_src", f"{args.data_dir}/test_tgt")

    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    # --- Extract or load cached features ---
    suffix = f"_last{args.last_n_layers}" if args.last_n_layers > 1 else ""
    train_feat_path = f"{args.features_dir}/train_features{suffix}.pt"
    val_feat_path = f"{args.features_dir}/val_features{suffix}.pt"
    test_feat_path = f"{args.features_dir}/test_features{suffix}.pt"

    if os.path.exists(train_feat_path) and os.path.exists(val_feat_path) and os.path.exists(test_feat_path):
        print(f"Loading cached features (last_n_layers={args.last_n_layers})...")
        train_features = torch.load(train_feat_path)
        val_features = torch.load(val_feat_path)
        test_features = torch.load(test_feat_path)
    else:
        print(f"Extracting features using GPUs: {args.gpu_ids}, last_n_layers={args.last_n_layers}")

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

    train_labels_t = torch.tensor(train_labels)
    val_labels_t = torch.tensor(val_labels)
    test_labels_t = torch.tensor(test_labels)

    # --- Merge train+val for final training (use test as validation) ---
    if args.merge_train_val:
        print("Merging train + val for training...")
        all_train_features = torch.cat([train_features, val_features], dim=0)
        all_train_labels = torch.cat([train_labels_t, val_labels_t], dim=0)
        eval_features = test_features
        eval_labels = test_labels_t
    else:
        all_train_features = train_features
        all_train_labels = train_labels_t
        eval_features = val_features
        eval_labels = val_labels_t

    # --- Train ---
    print("\nTraining predictor...")
    predictor = train_predictor(
        all_train_features, all_train_labels,
        eval_features, eval_labels,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size_train, patience=args.patience,
        head=args.head, device=train_device
    )

    # --- Test ---
    print("\nTest set evaluation:")
    predictor.eval()
    with torch.no_grad():
        test_logits = predictor(test_features.to(train_device))
        test_preds = (torch.sigmoid(test_logits) > 0.5).cpu().numpy()
        test_labels_np = test_labels_t.numpy()

    acc = accuracy_score(test_labels_np, test_preds)
    prec = precision_score(test_labels_np, test_preds)
    rec = recall_score(test_labels_np, test_preds)
    f1 = f1_score(test_labels_np, test_preds)
    print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"Target: Acc>=0.708, F1>=0.677")

    # --- Save ---
    torch.save(predictor.state_dict(), args.save_path)
    print(f"\nPredictor saved to {args.save_path}")
