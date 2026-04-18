"""
Score generated sequences for a fitness benchmark group.

For TrpB: lookup ground-truth fitness from the 160k combinatorial library.
          For off-library sequences (not 4-site-only variants), mark as NaN.
For CreiLOV / GFP: use the trained oracle (ESM2-650M features + PropertyPredictor head).
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.oracle.fitness.common import (
    extract_features_multi_device, inverse_target_transform, load_predictor_checkpoint,
)


def lookup_trpb_fitness(gen_seqs, ground_truth_csv):
    """Look up TrpB fitness from ground-truth 160k library.
    Match by the 4-site signature (positions 183/184/227/228 -> 0-indexed 182/183/226/227).
    Returns array of fitness (NaN if not in library or off-target mutations)."""
    gt = pd.read_csv(ground_truth_csv)
    wt = gt['wt_sequence'].iloc[0]
    gt_sigs = gt['sequence'].map(lambda s: (s[182], s[183], s[226], s[227]))
    fitness_by_sig = dict(zip(gt_sigs, gt['fitness']))

    results = []
    for seq in gen_seqs:
        if len(seq) != len(wt):
            results.append(np.nan)
            continue
        sig = (seq[182], seq[183], seq[226], seq[227])
        # Check off-target mutations (positions other than 4 sites must match WT)
        ok = True
        for i, (a, b) in enumerate(zip(seq, wt)):
            if i in (182, 183, 226, 227):
                continue
            if a != b:
                ok = False
                break
        if not ok:
            results.append(np.nan)
        else:
            results.append(fitness_by_sig.get(sig, np.nan))
    return np.array(results)


def score_with_oracle(seqs, predictor_path, device='cuda:0'):
    predictor, config = load_predictor_checkpoint(predictor_path, device=device)
    model_size = config.get('esm_model', '650M')
    last_n_layers = int(config.get('args', {}).get('last_n_layers', 1))
    transform = config.get('target_transform', {'name': 'none'})
    features = extract_features_multi_device(
        seqs, model_size=model_size, gpu_ids=[int(device.split(':')[-1])] if 'cuda' in device else [],
        batch_size=8, last_n_layers=last_n_layers,
        cache_dir=os.path.join('saved_predictors', 'eval_cache'),
    )
    with torch.no_grad():
        preds = predictor(features.to(next(predictor.parameters()).device)).cpu().numpy()
    preds_raw = inverse_target_transform(preds, transform)
    return preds_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['trpb', 'creilov', 'gfp'])
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--predictor_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--summary_json', type=str, default=None)
    args = parser.parse_args()

    if args.output_csv is None:
        base = os.path.splitext(args.input_csv)[0]
        args.output_csv = f'{base}_scored.csv'

    df = pd.read_csv(args.input_csv)
    seqs = df['sequence'].astype(str).tolist()
    print(f'Scoring {len(seqs)} sequences from {args.input_csv}')

    in_library_frac = None
    if args.dataset == 'trpb':
        lookup = lookup_trpb_fitness(seqs, 'data/benchmarks/processed/trpb/trpb_processed.csv')
        df['lookup_fitness'] = lookup
        in_library_frac = float(np.mean(~np.isnan(lookup)))
        print(f'  TrpB in-library fraction: {in_library_frac:.3f}')
        if args.predictor_path:
            oracle_preds = score_with_oracle(seqs, args.predictor_path, args.device)
            df['oracle_fitness'] = oracle_preds
    else:
        if args.predictor_path is None:
            args.predictor_path = {
                'creilov': 'evaluation/oracle/creilov/creilov_predictor_final.pt',
                'gfp': 'evaluation/oracle/gfp_sarkisyan/gfp_sarkisyan_predictor_final.pt',
            }[args.dataset]
        preds = score_with_oracle(seqs, args.predictor_path, args.device)
        df['oracle_fitness'] = preds

    df.to_csv(args.output_csv, index=False)
    print(f'Saved -> {args.output_csv}')

    # Summary
    summary = {'dataset': args.dataset, 'input': args.input_csv, 'n_seqs': len(seqs)}
    if 'oracle_fitness' in df.columns:
        vals = df['oracle_fitness'].dropna().values
        summary['oracle_mean'] = float(np.mean(vals))
        summary['oracle_median'] = float(np.median(vals))
        summary['oracle_std'] = float(np.std(vals))
    if 'lookup_fitness' in df.columns:
        vals = df['lookup_fitness'].dropna().values
        summary['lookup_mean'] = float(np.mean(vals)) if len(vals) else None
        summary['lookup_median'] = float(np.median(vals)) if len(vals) else None
        summary['in_library_fraction'] = in_library_frac

    print(json.dumps(summary, indent=2))
    if args.summary_json:
        with open(args.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
