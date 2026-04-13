"""
Evaluate structural quality of generated protein sequences using ESMFold.

Metrics:
  - pLDDT (predicted Local Distance Difference Test): per-residue confidence, 0-1 (higher = better)
  - pTM (predicted TM-score): global structural confidence, 0-1 (higher = better)

Requirements:
  - conda env: esmfold (torch >= 2.6, transformers, pandas, tqdm)
  - GPU: ~15GB VRAM for ESMFold (3.5B params)
  - Max sequence length: 512 (longer sequences are truncated)

Usage:
    conda activate esmfold
    # Default (4 recycles, accurate)
    python evaluation/esmfold/evaluate_esmfold.py \
        --input_csvs results/steering_sol_easy.csv results/no_steering_sol_easy.csv \
        --labels "steering" "no_steering" \
        --output_csv results/esmfold_eval.csv

    # Fast mode (1 recycle, ~3.7x faster, pLDDT corr=0.97 vs full)
    python evaluation/esmfold/evaluate_esmfold.py \
        --input_csvs results/steering_sol_easy.csv \
        --max_recycles 1
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def load_esmfold(device='cuda:0', max_recycles=4, chunk_size=None):
    """Load ESMFold model from HuggingFace.

    Args:
        device: CUDA device
        max_recycles: Number of recycling iterations (default=4, use 1 for ~3.7x speedup)
        chunk_size: Chunk size for structure module attention (None=full, 128=chunked)
    """
    from transformers import EsmForProteinFolding, AutoTokenizer
    print(f"Loading ESMFold on {device} (max_recycles={max_recycles}, chunk_size={chunk_size})...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
    model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')

    if max_recycles != 4:
        model.config.esmfold_config.trunk.max_recycles = max_recycles
    if chunk_size is not None:
        model.config.esmfold_config.trunk.chunk_size = chunk_size

    model = model.to(device).eval().half()
    print(f"  ESMFold loaded ({sum(p.numel() for p in model.parameters()):,} params, fp16)")
    return model, tokenizer


def predict_structure(seq, model, tokenizer, device, max_len=512):
    """Predict structure for a single sequence. Returns (pLDDT, pTM)."""
    seq = seq[:max_len]
    inputs = tokenizer([seq], return_tensors='pt', add_special_tokens=False).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        output = model(**inputs)

    plddt = output['plddt'][0, :len(seq)].float().mean().item()
    ptm = output['ptm'].float().item()
    return plddt, ptm


def evaluate_sequences(seqs, model, tokenizer, device, max_len=512, desc="Folding"):
    """Evaluate a list of sequences. Returns list of dicts with pLDDT, pTM."""
    results = []
    for seq in tqdm(seqs, desc=desc):
        try:
            plddt, ptm = predict_structure(seq, model, tokenizer, device, max_len)
            results.append({'plddt': plddt, 'ptm': ptm, 'length': len(seq), 'error': None})
        except Exception as e:
            print(f"  Error on seq len={len(seq)}: {e}")
            results.append({'plddt': float('nan'), 'ptm': float('nan'), 'length': len(seq), 'error': str(e)})
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESMFold structure quality evaluation")
    parser.add_argument('--input_csvs', type=str, nargs='+', required=True,
                        help='CSV files with sequences (column: sequence)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each CSV. Default: filenames')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Save per-sequence results to CSV')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Max sequence length (longer seqs truncated)')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Max sequences to evaluate per CSV')
    parser.add_argument('--max_recycles', type=int, default=4,
                        help='ESMFold recycling iterations (default=4, use 1 for ~3.7x speedup with corr>0.97)')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk size for structure module (default=None, set 128 for long seqs)')
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [os.path.splitext(os.path.basename(f))[0] for f in args.input_csvs]
    assert len(args.labels) == len(args.input_csvs)

    model, tokenizer = load_esmfold(args.device, args.max_recycles, args.chunk_size)

    all_results = []

    for csv_path, label in zip(args.input_csvs, args.labels):
        print(f"\n{'='*60}")
        print(f"Evaluating: {label} ({csv_path})")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)
        seqs = df['sequence'].tolist()
        if args.max_seqs is not None:
            seqs = seqs[:args.max_seqs]
        print(f"  {len(seqs)} sequences")

        results = evaluate_sequences(seqs, model, tokenizer, args.device, args.max_len, desc=label)

        plddts = [r['plddt'] for r in results if not np.isnan(r['plddt'])]
        ptms = [r['ptm'] for r in results if not np.isnan(r['ptm'])]

        print(f"\n  Results for [{label}]:")
        print(f"    pLDDT  mean ± std:  {np.mean(plddts):.4f} ± {np.std(plddts):.4f}")
        print(f"    pLDDT  median:      {np.median(plddts):.4f}")
        print(f"    pTM    mean ± std:  {np.mean(ptms):.4f} ± {np.std(ptms):.4f}")
        print(f"    pTM    median:      {np.median(ptms):.4f}")
        print(f"    Avg seq length:     {np.mean([len(s) for s in seqs]):.1f}")
        if len(plddts) < len(results):
            print(f"    Errors: {len(results) - len(plddts)}/{len(results)}")

        for i, (seq, r) in enumerate(zip(seqs, results)):
            all_results.append({
                'group': label,
                'seq_idx': i,
                'sequence': seq,
                'length': r['length'],
                'plddt': r['plddt'],
                'ptm': r['ptm'],
                'error': r['error'],
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary Comparison (ESMFold, recycles={args.max_recycles})")
    print(f"{'='*60}")
    print(f"{'Group':<25} {'N':>5} {'pLDDT mean':>11} {'pLDDT med':>10} {'pTM mean':>9} {'pTM med':>8}")
    print(f"{'-'*70}")
    results_df = pd.DataFrame(all_results)
    for label in args.labels:
        grp = results_df[results_df['group'] == label].dropna(subset=['plddt'])
        print(f"{label:<25} {len(grp):>5} {grp['plddt'].mean():>11.4f} {grp['plddt'].median():>10.4f} {grp['ptm'].mean():>9.4f} {grp['ptm'].median():>8.4f}")

    if args.output_csv is None:
        args.output_csv = "new-results/esmfold_eval.csv"
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nPer-sequence results saved to {args.output_csv}")
    print(f"{'='*60}")
