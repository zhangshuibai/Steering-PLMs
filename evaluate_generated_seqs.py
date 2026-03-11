"""
Evaluate generated protein sequences using a trained oracle predictor.
Extracts ESM2-650M features and predicts property scores.
Supports solubility (binary, sigmoid output) and thermostability (regression).
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import esm


# ========================= Models (same as training scripts) =========================

class PropertyPredictor(nn.Module):
    """RobertaLMHead-style: Linear -> GELU -> LayerNorm -> Linear(1)."""
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


# ========================= Feature Extraction =========================

def extract_features(seqs, model, alphabet, device, batch_size=8, max_len=1022):
    """Extract mean-pooled last-layer representations from ESM2."""
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    all_features = []

    n_batches = (len(seqs) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(seqs), batch_size), total=n_batches, desc="Extracting features"):
        batch_seqs = seqs[start:start + batch_size]
        batch_seqs = [s[:max_len] for s in batch_seqs]
        data = [("protein", s) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[n_layers])

        for i, seq_len in enumerate(batch_lens):
            rep = results["representations"][n_layers][i, 1:seq_len - 1].mean(0).cpu()
            all_features.append(rep)

    return torch.stack(all_features)


# ========================= Main =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True,
                        help='CSV with generated sequences (column: sequence)')
    parser.add_argument('--predictor_path', type=str, required=True,
                        help='Path to trained predictor checkpoint (.pt)')
    parser.add_argument('--property', type=str, required=True, choices=['sol', 'therm'],
                        help='Property type: sol (sigmoid+binary) or therm (regression)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV with predictions added. Default: input with _scored suffix')
    parser.add_argument('--ref_csv', type=str, default=None,
                        help='Reference sequences CSV for comparison (optional)')
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    if args.output_csv is None:
        base = os.path.splitext(args.input_csv)[0]
        args.output_csv = f"{base}_scored.csv"

    # --- Load generated sequences ---
    print(f"Loading generated sequences from {args.input_csv}")
    gen_df = pd.read_csv(args.input_csv)
    gen_seqs = gen_df['sequence'].tolist()
    print(f"  {len(gen_seqs)} sequences loaded")

    # --- Load ESM2 model ---
    print("Loading ESM2-650M...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(args.device)
    esm_model.eval()

    # --- Extract features for generated sequences ---
    print("Extracting features for generated sequences...")
    gen_features = extract_features(gen_seqs, esm_model, alphabet, args.device, args.batch_size)

    # --- Load and run predictor ---
    print(f"Loading predictor from {args.predictor_path}")
    predictor = PropertyPredictor(embed_dim=1280)
    ckpt = torch.load(args.predictor_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['model_state_dict'])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(args.device)
    predictor.eval()

    with torch.no_grad():
        gen_scores = predictor(gen_features.to(args.device)).cpu().numpy()

    if args.property == 'sol':
        gen_probs = torch.sigmoid(torch.tensor(gen_scores)).numpy()
        gen_labels = (gen_probs >= 0.5).astype(int)
        gen_df['pred_score'] = gen_scores
        gen_df['pred_prob'] = gen_probs
        gen_df['pred_label'] = gen_labels
    else:
        gen_df['pred_tm'] = gen_scores

    # --- Print statistics ---
    print("\n" + "=" * 60)
    print(f"Generated Sequences Evaluation ({args.property})")
    print("=" * 60)
    print(f"  N sequences: {len(gen_seqs)}")
    avg_len = np.mean([len(s) for s in gen_seqs])
    print(f"  Avg length:  {avg_len:.1f}")

    if args.property == 'sol':
        print(f"  Pred prob (mean±std): {gen_probs.mean():.4f} ± {gen_probs.std():.4f}")
        print(f"  Pred prob (median):   {np.median(gen_probs):.4f}")
        print(f"  Soluble (prob≥0.5):   {gen_labels.sum()}/{len(gen_labels)} ({gen_labels.mean()*100:.1f}%)")
    else:
        print(f"  Pred Tm (mean±std): {gen_scores.mean():.2f} ± {gen_scores.std():.2f} °C")
        print(f"  Pred Tm (median):   {np.median(gen_scores):.2f} °C")

    # --- Compare with reference sequences if provided ---
    if args.ref_csv is not None:
        print(f"\nReference sequences from {args.ref_csv}")
        ref_df = pd.read_csv(args.ref_csv)
        ref_seqs = ref_df['sequence'].tolist()
        print(f"  {len(ref_seqs)} reference sequences")

        print("Extracting features for reference sequences...")
        ref_features = extract_features(ref_seqs, esm_model, alphabet, args.device, args.batch_size)

        with torch.no_grad():
            ref_scores = predictor(ref_features.to(args.device)).cpu().numpy()

        if args.property == 'sol':
            ref_probs = torch.sigmoid(torch.tensor(ref_scores)).numpy()
            ref_labels = (ref_probs >= 0.5).astype(int)
            print(f"  Ref prob (mean±std):  {ref_probs.mean():.4f} ± {ref_probs.std():.4f}")
            print(f"  Ref soluble (≥0.5):   {ref_labels.sum()}/{len(ref_labels)} ({ref_labels.mean()*100:.1f}%)")
            print(f"\n  Δ prob (gen - ref):   {gen_probs.mean() - ref_probs.mean():+.4f}")
        else:
            print(f"  Ref Tm (mean±std):  {ref_scores.mean():.2f} ± {ref_scores.std():.2f} °C")
            print(f"\n  Δ Tm (gen - ref):   {gen_scores.mean() - ref_scores.mean():+.2f} °C")

    # --- Save results ---
    gen_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    print("=" * 60)
