"""
Evaluate generated protein sequences using a trained oracle predictor.
Supports solubility (binary) and thermostability (regression).

Usage:
    # Solubility
    python evaluation/oracle/evaluate_oracle.py \
        --input_csv results/ESM2_gen_steering_sol_easy.csv \
        --property sol --ref_csv data/sol_easy.csv

    # Thermostability
    python evaluation/oracle/evaluate_oracle.py \
        --input_csv results/ESM2_gen_steering_therm_easy.csv \
        --property therm --ref_csv data/therm_easy.csv
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.common import load_esm2_model, extract_features, load_predictor

# Default predictor paths relative to project root
DEFAULT_PREDICTORS = {
    'sol': 'evaluation/oracle/solubility/sol_predictor_final.pt',
    'therm': 'evaluation/oracle/thermostability/therm_predictor_nocdhit.pt',
}


def evaluate_sequences(seqs, predictor, esm_model, alphabet, device, property_type):
    """Extract features and predict property scores."""
    features = extract_features(seqs, esm_model, alphabet, device)
    with torch.no_grad():
        scores = predictor(features.to(device)).cpu().numpy()

    if property_type == 'sol':
        probs = torch.sigmoid(torch.tensor(scores)).numpy()
        labels = (probs >= 0.5).astype(int)
        return scores, probs, labels
    else:
        return scores, None, None


def print_results(seqs, scores, probs, labels, property_type, tag="Generated"):
    """Print evaluation statistics."""
    print(f"\n{'='*60}")
    print(f"{tag} Sequences Evaluation ({property_type})")
    print(f"{'='*60}")
    print(f"  N sequences: {len(seqs)}")
    print(f"  Avg length:  {np.mean([len(s) for s in seqs]):.1f}")

    if property_type == 'sol':
        print(f"  Pred prob (mean±std): {probs.mean():.4f} ± {probs.std():.4f}")
        print(f"  Pred prob (median):   {np.median(probs):.4f}")
        print(f"  Soluble (prob≥0.5):   {labels.sum()}/{len(labels)} ({labels.mean()*100:.1f}%)")
    else:
        print(f"  Pred Tm (mean±std): {scores.mean():.2f} ± {scores.std():.2f} °C")
        print(f"  Pred Tm (median):   {np.median(scores):.2f} °C")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle evaluation for generated protein sequences")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='CSV with generated sequences (column: sequence)')
    parser.add_argument('--property', type=str, required=True, choices=['sol', 'therm'],
                        help='Property type: sol (solubility) or therm (thermostability)')
    parser.add_argument('--predictor_path', type=str, default=None,
                        help='Path to trained predictor checkpoint. Default: auto-detect by property')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV path. Default: input with _scored suffix')
    parser.add_argument('--ref_csv', type=str, default=None,
                        help='Reference sequences CSV for comparison (optional)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    # Resolve predictor path
    if args.predictor_path is None:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        args.predictor_path = os.path.join(project_root, DEFAULT_PREDICTORS[args.property])
    if not os.path.exists(args.predictor_path):
        # Fallback to old location
        fallback = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_predictors',
                                'sol_predictor_final.pt' if args.property == 'sol' else 'therm_predictor_nocdhit.pt')
        if os.path.exists(fallback):
            args.predictor_path = fallback

    if args.output_csv is None:
        base = os.path.splitext(args.input_csv)[0]
        args.output_csv = f"{base}_scored.csv"

    # Load sequences
    print(f"Loading generated sequences from {args.input_csv}")
    gen_df = pd.read_csv(args.input_csv)
    gen_seqs = gen_df['sequence'].tolist()
    print(f"  {len(gen_seqs)} sequences loaded")

    # Load models
    esm_model, alphabet = load_esm2_model("650M", args.device)
    print(f"Loading predictor from {args.predictor_path}")
    predictor = load_predictor(args.predictor_path, device=args.device)

    # Evaluate generated sequences
    gen_scores, gen_probs, gen_labels = evaluate_sequences(
        gen_seqs, predictor, esm_model, alphabet, args.device, args.property)

    if args.property == 'sol':
        gen_df['pred_score'] = gen_scores
        gen_df['pred_prob'] = gen_probs
        gen_df['pred_label'] = gen_labels
    else:
        gen_df['pred_tm'] = gen_scores

    print_results(gen_seqs, gen_scores, gen_probs, gen_labels, args.property, "Generated")

    # Compare with reference
    if args.ref_csv is not None:
        print(f"\nReference sequences from {args.ref_csv}")
        ref_df = pd.read_csv(args.ref_csv)
        ref_seqs = ref_df['sequence'].tolist()
        print(f"  {len(ref_seqs)} reference sequences")

        ref_scores, ref_probs, ref_labels = evaluate_sequences(
            ref_seqs, predictor, esm_model, alphabet, args.device, args.property)

        print_results(ref_seqs, ref_scores, ref_probs, ref_labels, args.property, "Reference")

        if args.property == 'sol':
            print(f"\n  Δ prob (gen - ref):   {gen_probs.mean() - ref_probs.mean():+.4f}")
        else:
            print(f"\n  Δ Tm (gen - ref):     {gen_scores.mean() - ref_scores.mean():+.2f} °C")

    gen_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    print("=" * 60)
