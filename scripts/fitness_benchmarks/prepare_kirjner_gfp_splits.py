"""
Reproduce Kirjner et al. 2023 (GGS) GFP difficulty splits.

Filter rules (from data/benchmarks/raw/kirjner_ggs/GGS/configs/experiment/train/):
  - Easy:   fitness percentile [0.0, 1.0], mutation gap = 0  (no filter)
  - Medium: fitness percentile [0.2, 0.4], mutation gap >= 6 (Levenshtein)
  - Hard:   fitness percentile [0.0, 0.3], mutation gap >= 7 (Levenshtein)

Top reference set = fitness >= 99th percentile.
Mutation gap = min Levenshtein distance to any top-1% sequence.

Input:  data/benchmarks/raw/kirjner_ggs/GGS/data/GFP/ground_truth.csv
Output: data/benchmarks/processed/gfp_kirjner/{easy,medium,hard}.csv + top_reference.csv
"""

import argparse
import os
import sys
import pandas as pd
from Levenshtein import distance as levenshtein
from tqdm import tqdm


def filter_split(df, lower_q, upper_q, min_gap, top_seqs):
    lower_val = df['score'].quantile(lower_q)
    upper_val = df['score'].quantile(upper_q)
    filtered = df[df['score'].between(lower_val, upper_val)].copy()
    if min_gap == 0:
        return filtered.reset_index(drop=True)
    tqdm.pandas(desc=f'Levenshtein [{lower_q},{upper_q}] gap>={min_gap}')
    filtered['mutation_gap'] = filtered['sequence'].progress_map(
        lambda s: min(levenshtein(s.strip(), t.strip()) for t in top_seqs)
    )
    return filtered[filtered['mutation_gap'] >= min_gap].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth', type=str,
                        default='data/benchmarks/raw/kirjner_ggs/GGS/data/GFP/ground_truth.csv')
    parser.add_argument('--output_dir', type=str, default='data/benchmarks/processed/gfp_kirjner')
    parser.add_argument('--top_quantile', type=float, default=0.99)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Loading {args.ground_truth}')
    df = pd.read_csv(args.ground_truth)
    print(f'  {len(df)} variants, fitness [{df["score"].min():.3f}, {df["score"].max():.3f}]')

    top_threshold = df['score'].quantile(args.top_quantile)
    top_df = df[df['score'] >= top_threshold].reset_index(drop=True)
    print(f'  Top-{(1-args.top_quantile)*100:.0f}% (score >= {top_threshold:.3f}): {len(top_df)} sequences')
    top_df.to_csv(os.path.join(args.output_dir, 'top_reference.csv'), index=False)

    top_seqs = top_df['sequence'].tolist()

    splits = {
        'easy':   {'lower': 0.5, 'upper': 0.6, 'min_gap': 0},
        'medium': {'lower': 0.2, 'upper': 0.4, 'min_gap': 6},
        'hard':   {'lower': 0.0, 'upper': 0.3, 'min_gap': 7},
    }

    for name, cfg in splits.items():
        print(f'\n>>> {name}: percentile [{cfg["lower"]}, {cfg["upper"]}], gap >= {cfg["min_gap"]}')
        split_df = filter_split(df, cfg['lower'], cfg['upper'], cfg['min_gap'], top_seqs)
        out_path = os.path.join(args.output_dir, f'{name}.csv')
        split_df.to_csv(out_path, index=False)
        print(f'  {name}: {len(split_df)} sequences saved to {out_path}')
        print(f'    fitness range: [{split_df["score"].min():.3f}, {split_df["score"].max():.3f}]')
        if 'mutation_gap' in split_df.columns:
            print(f'    mutation gap: min={split_df["mutation_gap"].min()}, mean={split_df["mutation_gap"].mean():.1f}')


if __name__ == '__main__':
    main()
