"""
Unified generation script for fitness benchmarks (TrpB / CreiLOV / GFP).

Supports 9 settings per dataset:
  - reference            (unchanged input sequences, no generation)
  - no_steering          (ESM2 mask-predict, no steering vector)
  - all_layer            (steering vector at all 33 layers)
  - L17                  (steering vector at layer 17 only)
  - L17_GLP_u{0.1,0.5,0.9,1.0}
  - random               (uniform random amino acid sampling, no model)

Generation strategies (dataset-specific):
  - trpb:    single-round, mask fixed 4 sites (183,184,227,228), temperature top-p
  - creilov: single-round, mask fixed 15 preselected sites (loaded from processed CSV)
  - gfp:     either R=4 rounds x T=2 positions OR R=1 x T=8 positions (arg --gen_protocol)
             positions selected randomly across full 237-residue sequence

Each output group is saved as a CSV with the full generated sequences.
"""

import argparse
import json
import math
import os
import sys
import types
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.esm2_utils import load_esm2_model, decode
from utils.gen_utils import sample_top_p
from module.steerable_esm2 import steering_forward
from scripts.glp.steering_with_glp import (
    load_glp, build_glp_projection_fn, steering_forward_with_glp,
)
from evaluation.common import load_predictor, extract_features


AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')  # 20 canonical


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ==================== Mask helpers ====================

def choose_mask_positions(seq_len, mode, num_positions, fixed_positions=None, rng=None):
    """Return sorted list of 0-indexed positions inside the protein (0..seq_len-1)."""
    if mode == 'fixed':
        return sorted(fixed_positions)
    if mode == 'random':
        assert rng is not None
        k = min(num_positions, seq_len)
        return sorted(rng.choice(seq_len, size=k, replace=False).tolist())
    raise ValueError(f'Unknown mask mode {mode}')


# ==================== Generation ====================

def generate_one(ref_seq, mask_positions_per_round, model, alphabet, device,
                 steering_vectors=None, glp_project_fn=None, glp_layer=16,
                 temperature=1.0, top_p=0.9, use_glp_fwd=False):
    """Generate a single sequence given pre-determined mask positions for each round.

    mask_positions_per_round: list of lists. Each inner list holds 0-indexed protein positions
                              (already chosen) to mask simultaneously in that round.
    """
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    _, _, tokens = batch_converter([('protein', ref_seq)])
    tokens = tokens.to(device).clone()

    for round_positions in mask_positions_per_round:
        if not round_positions:
            continue
        token_positions = torch.tensor([p + 1 for p in round_positions], device=device)  # +1 for BOS
        seq_token = tokens.clone()
        seq_token[0, token_positions] = mask_idx

        with torch.no_grad():
            if use_glp_fwd:
                outputs = model.steering_forward_glp(
                    tokens=seq_token, steering_vectors=steering_vectors,
                    glp_project_fn=glp_project_fn, glp_layer=glp_layer,
                )
            elif steering_vectors is not None:
                outputs = model.steering_forward(tokens=seq_token, steering_vectors=steering_vectors)
            else:
                outputs = model(tokens=seq_token)

        logits = outputs['logits'][0, :, 4:24]  # 20 amino acids
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            pred = sample_top_p(probs, top_p)
        else:
            pred = torch.argmax(logits, dim=-1)
        pred = pred + 4
        pred[0] = tokens[0, 0]
        pred[-1] = tokens[0, -1]
        tokens[0, token_positions] = pred[token_positions]

    return decode(alphabet, tokens[:, 1:-1], onehot=False)[0]


# ==================== Dataset Loaders ====================

def load_dataset(dataset, difficulty=None, n_refs=1000, seed=42):
    """Return (ref_sequences, wt_sequence, fixed_mask_positions_or_None, dataset_info)."""
    rng = np.random.RandomState(seed)

    if dataset == 'trpb':
        df = pd.read_csv('data/benchmarks/processed/trpb/trpb_processed.csv')
        wt = df['wt_sequence'].iloc[0]
        # Select references: low-fitness variants
        low_df = df[df['fitness'] < df['fitness'].quantile(0.1)].reset_index(drop=True)
        idx = rng.choice(len(low_df), size=min(n_refs, len(low_df)), replace=False)
        refs = low_df.iloc[idx]['sequence'].tolist()
        fixed_positions = [182, 183, 226, 227]  # 0-indexed (paper uses 1-indexed 183/184/227/228)
        info = dict(wt_len=len(wt), fitness_col='fitness', score_col='fitness',
                    oracle_path='evaluation/oracle/trpb/trpb_predictor_final.pt',
                    fixed_sites=fixed_positions, lookup_csv='data/benchmarks/processed/trpb/trpb_processed.csv')
        return refs, wt, fixed_positions, info

    if dataset == 'creilov':
        df = pd.read_csv('data/benchmarks/processed/creilov/creilov_processed.csv')
        wt = df['wt_sequence'].iloc[0]
        low_df = df[df['fitness'] < df['fitness'].quantile(0.1)].reset_index(drop=True)
        idx = rng.choice(len(low_df), size=min(n_refs, len(low_df)), replace=False)
        refs = low_df.iloc[idx]['sequence'].tolist()
        # Collect preselected positions from the union of mutated_positions in dataset
        all_positions = set()
        for mp in df['mutated_positions'].dropna():
            for p in str(mp).split(','):
                p = p.strip()
                if p and p != 'WT':
                    all_positions.add(int(p) - 1)  # 1-indexed -> 0-indexed
        fixed_positions = sorted(all_positions)
        info = dict(wt_len=len(wt), fitness_col='fitness', score_col='fitness',
                    oracle_path='evaluation/oracle/creilov/creilov_predictor_final.pt',
                    fixed_sites=fixed_positions, lookup_csv='data/benchmarks/processed/creilov/creilov_processed.csv')
        return refs, wt, fixed_positions, info

    if dataset == 'gfp':
        assert difficulty in ('easy', 'medium', 'hard')
        split_df = pd.read_csv(f'data/benchmarks/processed/gfp_kirjner/{difficulty}.csv')
        wt_df = pd.read_csv('data/benchmarks/processed/gfp_sarkisyan/gfp_sarkisyan_processed.csv')
        wt = wt_df['wt_sequence'].iloc[0]
        idx = rng.choice(len(split_df), size=min(n_refs, len(split_df)), replace=False)
        refs = split_df.iloc[idx]['sequence'].tolist()
        info = dict(wt_len=len(wt), fitness_col='score', score_col='score',
                    oracle_path='evaluation/oracle/gfp_sarkisyan/gfp_sarkisyan_predictor_final.pt',
                    fixed_sites=None, lookup_csv=None, difficulty=difficulty)
        return refs, wt, None, info

    raise ValueError(f'Unknown dataset {dataset}')


# ==================== Random baseline ====================

def generate_random_baseline(refs, mask_mode, fixed_positions, n_rounds, n_positions_per_round, rng):
    gen = []
    for seq in refs:
        L = len(seq)
        seq_list = list(seq)
        already = set()
        for _ in range(n_rounds):
            if mask_mode == 'fixed':
                positions = [p for p in fixed_positions if p < L]
            else:
                avail = [p for p in range(L) if p not in already]
                k = min(n_positions_per_round, len(avail))
                positions = sorted(rng.choice(avail, size=k, replace=False).tolist())
            for p in positions:
                seq_list[p] = rng.choice(AMINO_ACIDS)
                already.add(p)
        gen.append(''.join(seq_list))
    return gen


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['trpb', 'creilov', 'gfp'])
    parser.add_argument('--difficulty', type=str, default=None, choices=[None, 'easy', 'medium', 'hard'])
    parser.add_argument('--gen_protocol', type=str, default='default',
                        choices=['default', 'R4T2', 'R1T8'],
                        help='GFP-only: R4T2 = 4 rounds x 2 positions; R1T8 = 1 round x 8 positions')
    parser.add_argument('--setting', type=str, required=True,
                        choices=['reference', 'no_steering', 'all_layer', 'L17',
                                 'L17_GLP_u0.1', 'L17_GLP_u0.5', 'L17_GLP_u0.9', 'L17_GLP_u1.0',
                                 'random'])
    parser.add_argument('--steering_vec_path', type=str, default=None)
    parser.add_argument('--glp_path', type=str,
                        default='generative_latent_prior/runs/glp-esm2-650m-layer17-d6')
    parser.add_argument('--num_timesteps', type=int, default=25)
    parser.add_argument('--n_refs', type=int, default=1000)
    parser.add_argument('--n_gen', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    refs, wt, fixed_positions, info = load_dataset(
        args.dataset, difficulty=args.difficulty, n_refs=args.n_refs, seed=args.seed)
    print(f'Dataset: {args.dataset} {args.difficulty or ""}, refs={len(refs)}, wt_len={info["wt_len"]}')

    # Determine generation protocol
    if args.dataset == 'trpb':
        mask_mode = 'fixed'
        n_rounds, n_pos = 1, 4
    elif args.dataset == 'creilov':
        mask_mode = 'fixed'
        n_rounds = 1
        n_pos = len(fixed_positions)
    elif args.dataset == 'gfp':
        mask_mode = 'random'
        if args.gen_protocol == 'R4T2':
            n_rounds, n_pos = 4, 2
        elif args.gen_protocol == 'R1T8':
            n_rounds, n_pos = 1, 8
        else:
            n_rounds, n_pos = 4, 2  # default

    print(f'Protocol: mask_mode={mask_mode}, rounds={n_rounds}, positions_per_round={n_pos}')

    # Reference setting: just save inputs
    if args.setting == 'reference':
        gen_seqs = refs[:args.n_gen]
        out_path = os.path.join(args.output_dir, 'reference.csv')
        pd.DataFrame({'sequence': gen_seqs}).to_csv(out_path, index=False)
        print(f'Saved {len(gen_seqs)} reference sequences -> {out_path}')
        return

    # Random baseline
    if args.setting == 'random':
        rng = np.random.RandomState(args.seed)
        use_refs = refs[:args.n_gen] + refs[:max(0, args.n_gen - len(refs))]
        use_refs = (refs * ((args.n_gen // len(refs)) + 1))[:args.n_gen]
        gen_seqs = generate_random_baseline(use_refs, mask_mode, fixed_positions, n_rounds, n_pos, rng)
        out_path = os.path.join(args.output_dir, 'random.csv')
        pd.DataFrame({'sequence': gen_seqs}).to_csv(out_path, index=False)
        print(f'Saved {len(gen_seqs)} random sequences -> {out_path}')
        return

    # Load ESM2
    print(f'Loading ESM2-650M on {args.device}')
    model, alphabet = load_esm2_model('650M', device=args.device)
    model.steering_forward = types.MethodType(steering_forward, model)

    # Steering vectors
    steering_vectors = None
    if args.setting != 'no_steering':
        sv_path = args.steering_vec_path or f'saved_steering_vectors/650M_{args.dataset}_fitness_steering_vectors.pt'
        pos_sv, neg_sv = torch.load(sv_path, weights_only=False)
        diff = (pos_sv - neg_sv).to(args.device)
        if args.setting == 'all_layer':
            steering_vectors = diff
        elif args.setting == 'L17' or args.setting.startswith('L17_GLP'):
            # "L17" = 17th transformer block = self.layers[16] (0-indexed).
            # GLP was trained on self.layers[16] output (ESM2 API repr_layers=[17]).
            # All L17 experiments use the same layer for fair comparison.
            sv = torch.zeros_like(diff)
            sv[16] = diff[16]
            steering_vectors = sv

    # GLP
    glp_fn = None
    use_glp_fwd = False
    if args.setting.startswith('L17_GLP'):
        model.steering_forward_glp = types.MethodType(steering_forward_with_glp, model)
        u = float(args.setting.split('_u')[-1])
        glp_model = load_glp(args.glp_path, args.device)
        glp_fn = build_glp_projection_fn(glp_model, u=u, num_timesteps=args.num_timesteps)
        use_glp_fwd = True

    # Generate
    rng = np.random.RandomState(args.seed)
    use_refs = (refs * ((args.n_gen // len(refs)) + 1))[:args.n_gen]
    gen_seqs = []
    for i, ref_seq in enumerate(tqdm(use_refs, desc=f'{args.setting}')):
        # Pre-compute mask positions per round
        positions_per_round = []
        already = set()
        for r in range(n_rounds):
            if mask_mode == 'fixed':
                positions = [p for p in fixed_positions if p < len(ref_seq)]
            else:
                avail = [p for p in range(len(ref_seq)) if p not in already]
                k = min(n_pos, len(avail))
                positions = sorted(rng.choice(avail, size=k, replace=False).tolist())
                already.update(positions)
            positions_per_round.append(positions)
        new_seq = generate_one(
            ref_seq, positions_per_round, model, alphabet, args.device,
            steering_vectors=steering_vectors, glp_project_fn=glp_fn, glp_layer=16,
            temperature=args.temperature, top_p=args.top_p, use_glp_fwd=use_glp_fwd,
        )
        gen_seqs.append(new_seq)

    out_path = os.path.join(args.output_dir, f'{args.setting}.csv')
    pd.DataFrame({'sequence': gen_seqs}).to_csv(out_path, index=False)
    print(f'Saved {len(gen_seqs)} sequences -> {out_path}')


if __name__ == '__main__':
    main()
