"""
Measure how far L17 activations deviate from the GLP training manifold
under different steering configurations.

Approach:
- Extract L17 activations (self.layers[16] output) for sequences generated under
  each setting: reference, L17 alpha={1,2,3,5,10}, all_layer alpha={1,2,3,5}
- Use GLP's normalizer rep_statistics.pt (mean, var) to compute z-score
- Metric 1: mean |z-score| per token (how many std-devs off from mean)
- Metric 2: L2 distance to GLP training mean
- Metric 3: log-likelihood under isotropic Gaussian (approx of manifold density)

Output: JSON with deviation metrics per setting.
"""

import argparse
import json
import os
import sys
import types
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'generative_latent_prior'))

from utils.esm2_utils import load_esm2_model, decode
from utils.gen_utils import sample_top_p
from module.steerable_esm2 import steering_forward


def get_l17_activations(seqs, model, alphabet, device, max_seqs=50):
    """Extract L17 output activations for a batch of sequences."""
    bc = alphabet.get_batch_converter()
    activations = []
    for seq in seqs[:max_seqs]:
        _, _, tokens = bc([('protein', seq[:1022])])
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[17])
        # repr_layers[17] = self.layers[16] output (where GLP was trained)
        rep = results['representations'][17][0, 1:-1]  # exclude BOS/EOS
        activations.append(rep.cpu())
    return torch.cat(activations, dim=0)  # (total_tokens, 1280)


def get_steered_l17_activations(seqs, model, alphabet, device, steering_vectors, max_seqs=50):
    """Extract L17 output activations AFTER steering + rescale is applied.
    Uses the same steering_forward mechanism but captures L17 output.
    """
    bc = alphabet.get_batch_converter()
    activations = []

    # We need to hook into the forward to capture L17 output after steering
    for seq in seqs[:max_seqs]:
        _, _, tokens = bc([('protein', seq[:1022])])
        tokens = tokens.to(device)

        # Manually replicate steering_forward but capture L17 after steering
        padding_mask = tokens.eq(model.padding_idx)
        x = model.embed_scale * model.embed_tokens(tokens)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None

        captured = None
        with torch.no_grad():
            for layer_idx, layer in enumerate(model.layers):
                x, _ = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=False)
                if steering_vectors is not None:
                    add_x = steering_vectors[layer_idx]
                    new_x = x + add_x
                    new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True)
                    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
                    x = new_x * (x_norm / new_x_norm)
                if layer_idx == 16:  # After self.layers[16] = GLP training layer
                    captured = x.transpose(0, 1)[0, 1:-1].clone().cpu()
                    break  # Stop early
        activations.append(captured)
    return torch.cat(activations, dim=0)


def compute_deviation_metrics(acts, rep_mean, rep_var):
    """Compute how far activations are from GLP training distribution.

    acts: (N, D) tensor of L17 activations
    rep_mean: (D,) or (1, 1, D) GLP training mean
    rep_var: (D,) or (1, 1, D) GLP training variance
    """
    if rep_mean.ndim > 1:
        rep_mean = rep_mean.squeeze()
    if rep_var.ndim > 1:
        rep_var = rep_var.squeeze()
    rep_std = torch.sqrt(rep_var)

    # Z-score per dimension
    z = (acts - rep_mean) / rep_std  # (N, D)
    abs_z = z.abs()

    # Metric 1: Mean |z-score| across all dims (average standard deviations from mean)
    mean_abs_z = abs_z.mean().item()

    # Metric 2: Fraction of dimensions where |z| > 3 (significant deviation)
    frac_z_gt_3 = (abs_z > 3.0).float().mean().item()

    # Metric 3: Squared Mahalanobis-like distance per token (sum of z^2)
    mahal_per_token = (z ** 2).sum(dim=-1)  # (N,)
    mean_mahal = mahal_per_token.mean().item()
    median_mahal = mahal_per_token.median().item()

    # Metric 4: Average log-likelihood under isotropic standardized Gaussian
    # log p(z) = -0.5 * ||z||^2 - 0.5 * D * log(2*pi)
    D = acts.shape[-1]
    log_likelihood_per_token = -0.5 * mahal_per_token - 0.5 * D * math.log(2 * math.pi)
    mean_loglik = log_likelihood_per_token.mean().item()

    return {
        'mean_abs_z': mean_abs_z,
        'frac_z_gt_3': frac_z_gt_3,
        'mean_mahal_dist': mean_mahal,
        'median_mahal_dist': median_mahal,
        'mean_log_likelihood': mean_loglik,
        'n_tokens': acts.shape[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_data', type=str, default='data/sol_hard.csv')
    parser.add_argument('--n_seqs', type=int, default=30)
    parser.add_argument('--sv_path', type=str, default='saved_steering_vectors/650M_sol_steering_vectors.pt')
    parser.add_argument('--glp_stats', type=str,
                        default='generative_latent_prior/runs/glp-esm2-650m-layer17-d6/rep_statistics.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading GLP stats from {args.glp_stats}')
    rep_stats = torch.load(args.glp_stats, map_location='cpu', weights_only=False)
    rep_mean = rep_stats['mean'].float()
    rep_var = rep_stats['var'].float()
    print(f'  GLP training stats: mean shape {rep_mean.shape}, var shape {rep_var.shape}')

    print(f'Loading ESM2-650M on {args.device}')
    model, alphabet = load_esm2_model('650M', device=args.device)
    model.steering_forward = types.MethodType(steering_forward, model)

    # Load reference sequences
    df = pd.read_csv(args.ref_data)
    ref_seqs = df['sequence'].tolist()[:args.n_seqs]
    print(f'Using {len(ref_seqs)} reference sequences from {args.ref_data}')

    # Load steering vectors
    pos_sv, neg_sv = torch.load(args.sv_path, weights_only=False)
    diff = (pos_sv - neg_sv).to(args.device)

    results = {}

    # === No steering baseline ===
    print('\n>>> Reference (no steering)')
    acts_ref = get_l17_activations(ref_seqs, model, alphabet, args.device)
    results['reference_no_steering'] = compute_deviation_metrics(acts_ref, rep_mean, rep_var)
    print(f'  {results["reference_no_steering"]}')

    # === L17 single-layer with different alphas ===
    for alpha in [1, 2, 3, 5, 10]:
        print(f'\n>>> L17 single-layer, alpha={alpha}')
        sv = torch.zeros_like(diff)
        sv[16] = diff[16] * alpha  # index 16 = self.layers[16] = L17
        acts = get_steered_l17_activations(ref_seqs, model, alphabet, args.device, sv)
        results[f'L17_alpha{alpha}'] = compute_deviation_metrics(acts, rep_mean, rep_var)
        print(f'  {results[f"L17_alpha{alpha}"]}')

    # === All-layer with different alphas ===
    for alpha in [1, 2, 3]:
        print(f'\n>>> All-layer, alpha={alpha}')
        sv = diff * alpha
        acts = get_steered_l17_activations(ref_seqs, model, alphabet, args.device, sv)
        results[f'all_layer_alpha{alpha}'] = compute_deviation_metrics(acts, rep_mean, rep_var)
        print(f'  {results[f"all_layer_alpha{alpha}"]}')

    # Save
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {args.output_json}')

    # Summary table
    print('\n=== Summary ===')
    print(f'{"Setting":<25} {"mean|z|":>9} {"%z>3":>7} {"mahal":>9} {"loglik":>10}')
    print('-'*65)
    for k, v in results.items():
        print(f'{k:<25} {v["mean_abs_z"]:>9.3f} {v["frac_z_gt_3"]*100:>6.1f}% {v["mean_mahal_dist"]:>9.0f} {v["mean_log_likelihood"]:>10.1f}')


if __name__ == '__main__':
    main()
