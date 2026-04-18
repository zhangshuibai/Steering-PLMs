"""
Extract steering vectors for fitness benchmarks (TrpB / CreiLOV / GFP).

For each dataset, selects:
  - positive set: 100 sequences following a spec (top-100 by fitness OR random-100 from 'easy')
  - negative set: 100 sequences following a spec (bottom-100 by fitness OR random-100 from 'hard')

Then runs ESM2-650M to extract per-layer mean representations and saves
(pos_mean, neg_mean) tuple to saved_steering_vectors/.
"""

import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.esm2_utils import load_esm2_model, extract_esm2_features, get_esm2_layer_and_feature_dim


def select_top_bottom(df, n=100, seed=42):
    df_sorted = df.sort_values('fitness', ascending=False).reset_index(drop=True)
    pos = df_sorted.head(n)['sequence'].tolist()
    neg = df_sorted.tail(n)['sequence'].tolist()
    return pos, neg


def select_random_from_splits(easy_csv, hard_csv, n=100, seed=42, score_col='score'):
    rng = np.random.RandomState(seed)
    easy = pd.read_csv(easy_csv)
    hard = pd.read_csv(hard_csv)
    pos_idx = rng.choice(len(easy), size=min(n, len(easy)), replace=False)
    neg_idx = rng.choice(len(hard), size=min(n, len(hard)), replace=False)
    return easy.iloc[pos_idx]['sequence'].tolist(), hard.iloc[neg_idx]['sequence'].tolist()


def build_steering_vectors(pos_seqs, neg_seqs, model, alphabet, n_layers):
    print(f'  Extracting features: pos={len(pos_seqs)}, neg={len(neg_seqs)}')
    pos_reps = extract_esm2_features(pos_seqs, model, alphabet, n_layers)  # (L, N, D)
    neg_reps = extract_esm2_features(neg_seqs, model, alphabet, n_layers)
    pos_mean = torch.stack([pos_reps[i].mean(dim=0) for i in range(n_layers)]).detach().cpu()
    neg_mean = torch.stack([neg_reps[i].mean(dim=0) for i in range(n_layers)]).detach().cpu()
    return pos_mean, neg_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['trpb', 'creilov', 'gfp'])
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='650M')
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f'saved_steering_vectors/650M_{args.dataset}_fitness_steering_vectors.pt'
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Select positive/negative sequences per dataset
    if args.dataset == 'trpb':
        df = pd.read_csv('data/benchmarks/processed/trpb/trpb_processed.csv')
        pos, neg = select_top_bottom(df, n=args.n, seed=args.seed)
        print(f'TrpB: pos={args.n} top-fitness, neg={args.n} bottom-fitness')
    elif args.dataset == 'creilov':
        df = pd.read_csv('data/benchmarks/processed/creilov/creilov_processed.csv')
        pos, neg = select_top_bottom(df, n=args.n, seed=args.seed)
        print(f'CreiLOV: pos={args.n} top-fitness, neg={args.n} bottom-fitness')
    elif args.dataset == 'gfp':
        # Follow Steering PLMs paper: random 100 from easy as pos, random 100 from hard as neg
        pos, neg = select_random_from_splits(
            'data/benchmarks/processed/gfp_kirjner/easy.csv',
            'data/benchmarks/processed/gfp_kirjner/hard.csv',
            n=args.n, seed=args.seed,
        )
        print(f'GFP: pos={args.n} from easy (Kirjner), neg={args.n} from hard (Kirjner)')

    model, alphabet = load_esm2_model(args.model)
    n_layers, _ = get_esm2_layer_and_feature_dim(args.model)

    pos_mean, neg_mean = build_steering_vectors(pos, neg, model, alphabet, n_layers)
    torch.save((pos_mean, neg_mean), args.output_path)
    print(f'Saved steering vectors -> {args.output_path}')
    print(f'  pos_mean shape: {pos_mean.shape}, neg_mean shape: {neg_mean.shape}')


if __name__ == '__main__':
    main()
