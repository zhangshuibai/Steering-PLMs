"""
Single-Round Mask Ratio Experiment (V4)
========================================
Single round of mask-predict with varying mask_ratio from 0.1 to 1.0.
Only 1 forward pass per sequence — no iterative accumulation.

This isolates how mask_ratio affects pPPL and sol in a single step,
for each steering/GLP configuration.

Methods:
  - No Steering
  - L17 no GLP
  - L17+GLP with selected u values (steps=100)

mask_ratio: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python run_single_round_mask_ratio.py [--temperature 0]
"""

import os
import sys
import types
import math
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generative_latent_prior'))

from steering_with_glp import (
    build_glp_projection_fn,
    steering_forward_with_glp,
    evaluate_sol,
    PropertyPredictor,
)
from module.steerable_esm2 import steering_forward
from evaluate_ppl import compute_pseudo_perplexity_multi_gpu
from utils.esm2_utils import load_esm2_model, decode
from utils.gen_utils import sample_top_p
from generative_latent_prior.glp.denoiser import GLP

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ========================= Config =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_gen', type=int, default=100)
    return parser.parse_args()

ARGS = parse_args()

N_GEN = ARGS.n_gen
TEMPERATURE = ARGS.temperature
TOP_P = ARGS.top_p
GLP_LAYER = 17
GLP_PATH = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6'
GLP_CHECKPOINT = 'final'
SV_PATH = 'saved_steering_vectors/650M_sol_steering_vectors.pt'
PREDICTOR_PATH = 'saved_predictors/sol_predictor_final.pt'
REF_DATA = 'data/sol_easy.csv'
PPL_MODEL = '3B'
PPL_GPU_IDS = [0, 1, 2, 3]
BATCH_MASKS = 32

if ARGS.output_dir is not None:
    OUTPUT_DIR = ARGS.output_dir
elif TEMPERATURE == 0.0:
    OUTPUT_DIR = 'results/single_round_mask_ratio_greedy'
else:
    OUTPUT_DIR = 'results/single_round_mask_ratio'

MASK_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

U_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
GLP_STEPS = 100

# Methods: (name, sv_type, u, steps)
METHODS = [
    ('No Steering', None, None, None),
    ('L17 no GLP',  'single', None, None),
]
for u in U_VALUES:
    METHODS.append((f'L17+GLP u={u}', 'all', u, GLP_STEPS))


# ========================= Core Function =========================

def generate_single_round(ref_seqs, model, alphabet, device, n_gen,
                          mask_ratio, steering_vectors=None,
                          glp_project_fn=None, glp_layer=17,
                          temperature=1.0, top_p=0.9):
    """
    Single round of mask-predict: mask mask_ratio fraction of positions,
    one forward pass, fill back. Returns n_gen sequences.
    """
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    gen_seqs = []

    for i in tqdm(range(n_gen), desc=f"mask_ratio={mask_ratio:.1f}"):
        seq = ref_seqs[i % len(ref_seqs)]
        _, _, tokens = batch_converter([("protein", seq)])
        tokens = tokens.to(device).clone()
        length = tokens.size(1) - 2

        # Select positions to mask
        mask_size = math.ceil(length * mask_ratio)
        indices = torch.randperm(length)[:mask_size]
        mask_positions = indices + 1  # +1 for BOS

        seq_token = tokens.clone()
        seq_token[0, mask_positions] = mask_idx

        # Forward
        with torch.no_grad():
            if steering_vectors is None:
                outputs = model(seq_token)
            elif glp_project_fn is None:
                outputs = model.steering_forward(
                    tokens=seq_token,
                    steering_vectors=steering_vectors,
                )
            else:
                outputs = model.steering_forward_glp(
                    tokens=seq_token,
                    steering_vectors=steering_vectors,
                    glp_project_fn=glp_project_fn,
                    glp_layer=glp_layer,
                )

        logits = outputs['logits'][0, :, 4:24]

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            pred_seq = sample_top_p(probs, top_p)
        else:
            pred_seq = torch.argmax(logits, dim=-1)
        pred_seq = pred_seq + 4
        pred_seq[0] = tokens[0, 0]
        pred_seq[-1] = tokens[0, -1]

        tokens[0, mask_positions] = pred_seq[mask_positions]

        gen_seq = decode(alphabet, tokens[:, 1:-1], onehot=False)[0]
        gen_seqs.append(gen_seq)

    return gen_seqs


# ========================= Main =========================

def main():
    mp.set_start_method('spawn', force=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda:0'

    ref_df = pd.read_csv(REF_DATA)
    ref_seqs = ref_df['sequence'].tolist()
    print(f"Reference sequences: {len(ref_seqs)}")
    print(f"N_GEN={N_GEN}, temperature={TEMPERATURE}, top_p={TOP_P}")
    print(f"Mask ratios: {MASK_RATIOS}")
    print(f"Methods: {len(METHODS)}")
    print(f"Output: {OUTPUT_DIR}")

    # ==================== Phase 1: Generation + Sol Eval ====================
    print("=" * 70)
    print("Phase 1: Single-Round Generation + Sol Evaluation")
    print("=" * 70)

    # Load models
    print(f"Loading ESM2-650M on {device}...")
    model_650m, alphabet = load_esm2_model("650M", device=device)
    model_650m.steering_forward = types.MethodType(steering_forward, model_650m)
    model_650m.steering_forward_glp = types.MethodType(
        steering_forward_with_glp, model_650m
    )

    print(f"Loading steering vectors from {SV_PATH}...")
    pos_sv, neg_sv = torch.load(SV_PATH)
    sv_all = (pos_sv - neg_sv).to(device)
    sv_single = torch.zeros_like(sv_all)
    sv_single[GLP_LAYER] = sv_all[GLP_LAYER]

    print(f"Loading GLP from {GLP_PATH}...")
    from omegaconf import OmegaConf
    glp_config = OmegaConf.load(os.path.join(GLP_PATH, "config.yaml"))
    OmegaConf.resolve(glp_config)
    glp_config.glp_kwargs.normalizer_config.rep_statistic = os.path.join(
        GLP_PATH, "rep_statistics.pt"
    )
    glp_model = GLP(**glp_config.glp_kwargs)
    glp_model.to(device)
    glp_model.load_pretrained(GLP_PATH, name=GLP_CHECKPOINT)
    glp_model.eval()
    print(f"  GLP loaded. Params: {sum(p.numel() for p in glp_model.parameters()):,}")

    print(f"Loading sol predictor from {PREDICTOR_PATH}...")
    predictor = PropertyPredictor(embed_dim=1280)
    ckpt = torch.load(PREDICTOR_PATH, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['model_state_dict'])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(device)
    predictor.eval()

    # Reference sol
    ref_mean_prob, ref_sol_ratio, _ = evaluate_sol(
        ref_seqs, model_650m, alphabet, predictor, device
    )
    print(f"Reference: sol_ratio={ref_sol_ratio*100:.1f}%, mean_prob={ref_mean_prob:.4f}")

    # all_results: {method_name: {mask_ratio: {sol_mean_prob, sol_ratio, csv_path}}}
    all_results = {}

    for method_name, sv_type, u, steps in METHODS:
        print(f"\n{'='*50}")
        print(f"Method: {method_name}")
        print(f"{'='*50}")

        if sv_type is None:
            sv = None
        elif sv_type == 'single':
            sv = sv_single
        else:
            sv = sv_all

        glp_fn = None
        if u is not None:
            glp_fn = build_glp_projection_fn(glp_model, u=u, num_timesteps=steps)

        method_key = method_name.replace(' ', '_').replace('+', '_').replace('=', '')
        method_dir = os.path.join(OUTPUT_DIR, method_key)
        os.makedirs(method_dir, exist_ok=True)

        method_results = {}

        for mr in MASK_RATIOS:
            csv_path = os.path.join(method_dir, f'mr{mr:.1f}.csv')

            # Check cache
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if len(df) >= N_GEN:
                    gen_seqs = df['sequence'].tolist()
                    print(f"  mask_ratio={mr:.1f}: CACHED ({len(gen_seqs)} seqs)")
                    mean_prob, sol_ratio, _ = evaluate_sol(
                        gen_seqs, model_650m, alphabet, predictor, device
                    )
                    method_results[mr] = {
                        'sol_mean_prob': float(mean_prob),
                        'sol_ratio': float(sol_ratio),
                        'csv_path': csv_path,
                        'n_seqs': len(gen_seqs),
                    }
                    print(f"    sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")
                    continue

            gen_seqs = generate_single_round(
                ref_seqs, model_650m, alphabet, device, N_GEN,
                mask_ratio=mr, steering_vectors=sv,
                glp_project_fn=glp_fn, glp_layer=GLP_LAYER,
                temperature=TEMPERATURE, top_p=TOP_P,
            )
            pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

            mean_prob, sol_ratio, _ = evaluate_sol(
                gen_seqs, model_650m, alphabet, predictor, device
            )
            method_results[mr] = {
                'sol_mean_prob': float(mean_prob),
                'sol_ratio': float(sol_ratio),
                'csv_path': csv_path,
                'n_seqs': len(gen_seqs),
            }
            print(f"  mask_ratio={mr:.1f}: sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")

        all_results[method_name] = method_results
        sys.stdout.flush()

    # Free GPU
    del model_650m, predictor, glp_model, sv_all, sv_single
    torch.cuda.empty_cache()
    print(f"\nPhase 1 complete.")

    # ==================== Phase 2: pPPL Evaluation ====================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: pPPL Evaluation with ESM2-{PPL_MODEL} (GPUs {PPL_GPU_IDS})")
    print(f"{'=' * 70}")

    # Reference pPPL
    print(f"\nEvaluating Reference pPPL...")
    ref_ppls = compute_pseudo_perplexity_multi_gpu(
        ref_seqs[:N_GEN], PPL_MODEL, PPL_GPU_IDS, BATCH_MASKS
    )
    ref_ppl_mean = float(np.mean(ref_ppls))
    ref_ppl_median = float(np.median(ref_ppls))
    print(f"  Reference pPPL: mean={ref_ppl_mean:.4f}, median={ref_ppl_median:.4f}")

    for method_name in all_results:
        method_results = all_results[method_name]
        print(f"\n--- {method_name} ---")

        for mr in MASK_RATIOS:
            if mr not in method_results:
                continue
            res = method_results[mr]
            df = pd.read_csv(res['csv_path'])
            seqs = df['sequence'].tolist()

            ppls = compute_pseudo_perplexity_multi_gpu(
                seqs, PPL_MODEL, PPL_GPU_IDS, BATCH_MASKS
            )
            ppls_arr = np.array(ppls)
            res['ppl_mean'] = float(ppls_arr.mean())
            res['ppl_median'] = float(np.median(ppls_arr))
            res['ppl_std'] = float(ppls_arr.std())
            print(f"  mr={mr:.1f}: pPPL={res['ppl_mean']:.4f}")

        # Save intermediate
        with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
            json.dump({
                'reference': {
                    'sol_mean_prob': float(ref_mean_prob),
                    'sol_ratio': float(ref_sol_ratio),
                    'ppl_mean': ref_ppl_mean,
                    'ppl_median': ref_ppl_median,
                },
                'methods': {k: {str(mr): v for mr, v in mv.items()}
                            for k, mv in all_results.items()},
            }, f, indent=2)

    # ==================== Summary + Plot ====================
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")

    for method_name in all_results:
        method_results = all_results[method_name]
        print(f"\n{method_name}:")
        print(f"  {'MR':<5} | {'Sol Prob':>9} | {'pPPL mean':>10} | {'pPPL med':>10}")
        print(f"  {'-' * 45}")
        for mr in MASK_RATIOS:
            if mr not in method_results:
                continue
            r = method_results[mr]
            ppl = r.get('ppl_mean', float('nan'))
            ppl_med = r.get('ppl_median', float('nan'))
            print(f"  {mr:<5.1f} | {r['sol_mean_prob']:>9.4f} | {ppl:>10.4f} | {ppl_med:>10.4f}")

    # Save final
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({
            'reference': {
                'sol_mean_prob': float(ref_mean_prob),
                'sol_ratio': float(ref_sol_ratio),
                'ppl_mean': ref_ppl_mean,
                'ppl_median': ref_ppl_median,
            },
            'methods': {k: {str(mr): v for mr, v in mv.items()}
                        for k, mv in all_results.items()},
        }, f, indent=2)

    # Plot
    print(f"\nGenerating plots...")
    plot_results(all_results, ref_mean_prob, ref_ppl_mean)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("=" * 70)


def plot_results(all_results, ref_sol_prob, ref_ppl_mean):
    """Plot pPPL and sol vs mask_ratio."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sampling = "Greedy" if TEMPERATURE == 0.0 else f"Nucleus (T={TEMPERATURE})"
    fig.suptitle(f'Single Round, {sampling}', fontsize=14, fontweight='bold')

    colors = {
        'No Steering': '#1f77b4',
        'L17 no GLP': '#2ca02c',
    }
    u_colors = {0.1: '#d62728', 0.3: '#ff7f0e', 0.5: '#9467bd',
                0.7: '#8c564b', 0.9: '#e377c2', 1.0: '#7f7f7f'}
    for u in U_VALUES:
        colors[f'L17+GLP u={u}'] = u_colors[u]

    for method_name, method_results in all_results.items():
        mrs = sorted(method_results.keys())
        ppl_vals = [method_results[mr].get('ppl_mean', float('nan')) for mr in mrs]
        sol_vals = [method_results[mr]['sol_mean_prob'] for mr in mrs]

        color = colors.get(method_name, None)
        label = method_name

        ax1.plot(mrs, ppl_vals, 'o-', label=label, color=color, markersize=4)
        ax2.plot(mrs, sol_vals, 'o-', label=label, color=color, markersize=4)

    ax1.axhline(y=ref_ppl_mean, color='black', linestyle='--', alpha=0.5,
                label=f'Reference ({ref_ppl_mean:.2f})')
    ax2.axhline(y=ref_sol_prob, color='black', linestyle='--', alpha=0.5,
                label=f'Reference ({ref_sol_prob:.4f})')

    ax1.set_xlabel('Mask Ratio')
    ax1.set_ylabel('pPPL (ESM2-3B)')
    ax1.set_title('pPPL vs Mask Ratio')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Mask Ratio')
    ax2.set_ylabel('Sol Mean Prob')
    ax2.set_title('Solubility vs Mask Ratio')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mask_ratio_plot.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mask_ratio_plot.pdf'), bbox_inches='tight')
    print(f"  Plots saved to {OUTPUT_DIR}/mask_ratio_plot.png/pdf")
    plt.close()


if __name__ == "__main__":
    main()
