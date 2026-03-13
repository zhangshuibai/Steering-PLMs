"""
Step-wise Evaluation (V3): Track pPPL and Sol at Each Decoding Step
====================================================================
In the standard iterative mask-predict pipeline (mask_ratio=0.1, 10 rounds),
record intermediate sequences after each round and evaluate pPPL + sol.

This produces a trajectory of (pPPL, sol) vs decoding step for each method,
revealing how error accumulates across iterations.

Methods:
  1. No Steering
  2. L17 no GLP
  3. L17+GLP (selected u values with steps=100)

Output: line plots of pPPL and sol vs decoding step.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python run_stepwise_eval.py
"""

import os
import sys
import types
import math
import json
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

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0=greedy, default=1.0)')
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto based on temperature)')
    parser.add_argument('--n_gen', type=int, default=100)
    return parser.parse_args()

ARGS = parse_args()

N_GEN = ARGS.n_gen
MASK_RATIO = 0.1
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
    OUTPUT_DIR = 'results/stepwise_eval_greedy'
else:
    OUTPUT_DIR = 'results/stepwise_eval'

# Methods to evaluate: (name, sv_type, u, steps) — u=None means no GLP
U_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
STEPS_VALUES = [25, 50, 100, 200, 400]

METHODS = [
    ('No Steering', None, None, None),
    ('L17 no GLP',  'single', None, None),
]
for steps in STEPS_VALUES:
    for u in U_VALUES:
        METHODS.append((f'L17+GLP u={u} s={steps}', 'all', u, steps))


# ========================= Core Function =========================

def generate_iterative_with_snapshots(
    ref_seqs, model, alphabet, device, n_gen,
    steering_vectors=None, glp_project_fn=None, glp_layer=17,
    mask_ratio=0.1, temperature=1.0, top_p=0.9,
):
    """
    Iterative mask-predict generation with snapshots after each round.

    Returns:
        snapshots: dict mapping round_idx -> list of n_gen sequences
                   round 0 = after first mask-predict round
                   round (rounds-1) = final sequences
    """
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    # Determine number of rounds
    rounds = math.ceil(1.0 / mask_ratio)

    # Initialize snapshots: one list per round
    snapshots = {r: [] for r in range(rounds)}

    for i in tqdm(range(n_gen), desc="Generating"):
        seq = ref_seqs[i % len(ref_seqs)]
        _, _, tokens = batch_converter([("protein", seq)])
        tokens = tokens.to(device).clone()
        length = tokens.size(1) - 2
        candidate_sites = list(range(length))

        for r in range(rounds):
            mask_size = min(math.ceil(length * mask_ratio), len(candidate_sites))
            if mask_size == 0:
                # No more sites to mask, copy current state to remaining rounds
                cur_seq = decode(alphabet, tokens[:, 1:-1], onehot=False)[0]
                for rr in range(r, rounds):
                    snapshots[rr].append(cur_seq)
                break

            indices = torch.randperm(len(candidate_sites))[:mask_size]
            mask_positions = torch.tensor([candidate_sites[idx] for idx in indices]) + 1
            candidate_sites = [site for idx, site in enumerate(candidate_sites) if idx not in indices]

            seq_token = tokens.clone()
            seq_token[0, mask_positions] = mask_idx

            with torch.no_grad():
                if steering_vectors is None:
                    # No steering
                    outputs = model(seq_token)
                elif glp_project_fn is None:
                    # Steering only
                    outputs = model.steering_forward(
                        tokens=seq_token,
                        steering_vectors=steering_vectors,
                    )
                else:
                    # Steering + GLP
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

            # Snapshot after this round
            cur_seq = decode(alphabet, tokens[:, 1:-1], onehot=False)[0]
            snapshots[r].append(cur_seq)

    return snapshots, rounds


# ========================= Main =========================

def main():
    mp.set_start_method('spawn', force=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda:0'

    # Load reference data
    ref_df = pd.read_csv(REF_DATA)
    ref_seqs = ref_df['sequence'].tolist()
    print(f"Reference sequences: {len(ref_seqs)}")
    print(f"Generating {N_GEN} sequences per method, mask_ratio={MASK_RATIO}")
    print(f"Sampling: temperature={TEMPERATURE}, top_p={TOP_P}")
    print(f"Output: {OUTPUT_DIR}")

    # ==================== Phase 1: Generation + Sol Eval ====================
    print("=" * 70)
    print("Phase 1: Step-wise Generation + Sol Evaluation")
    print("=" * 70)

    # Load ESM2-650M
    print(f"Loading ESM2-650M on {device}...")
    model_650m, alphabet = load_esm2_model("650M", device=device)
    model_650m.steering_forward = types.MethodType(steering_forward, model_650m)
    model_650m.steering_forward_glp = types.MethodType(
        steering_forward_with_glp, model_650m
    )

    # Load steering vectors
    print(f"Loading steering vectors from {SV_PATH}...")
    pos_sv, neg_sv = torch.load(SV_PATH)
    sv_all = (pos_sv - neg_sv).to(device)
    sv_single = torch.zeros_like(sv_all)
    sv_single[GLP_LAYER] = sv_all[GLP_LAYER]

    # Load GLP
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

    # Load sol predictor
    print(f"Loading sol predictor from {PREDICTOR_PATH}...")
    predictor = PropertyPredictor(embed_dim=1280)
    ckpt = torch.load(PREDICTOR_PATH, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['model_state_dict'])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(device)
    predictor.eval()

    # Reference sol (step 0 baseline)
    ref_mean_prob, ref_sol_ratio, _ = evaluate_sol(
        ref_seqs, model_650m, alphabet, predictor, device
    )
    print(f"Reference: sol_ratio={ref_sol_ratio*100:.1f}%, mean_prob={ref_mean_prob:.4f}")

    # Store all trajectories: {method_name: {round_idx: {seqs, sol_mean_prob, sol_ratio}}}
    all_trajectories = {}

    for method_name, sv_type, u, steps in METHODS:
        print(f"\n--- {method_name} ---")

        # Select steering vectors
        if sv_type is None:
            sv = None
        elif sv_type == 'single':
            sv = sv_single
        else:
            sv = sv_all

        # Build GLP projection if needed
        glp_fn = None
        if u is not None:
            glp_fn = build_glp_projection_fn(glp_model, u=u, num_timesteps=steps)

        # Check cache
        method_dir = os.path.join(OUTPUT_DIR, method_name.replace(' ', '_').replace('+', '_').replace('=', ''))
        os.makedirs(method_dir, exist_ok=True)
        cache_file = os.path.join(method_dir, 'snapshots.json')

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            if len(cached.get('0', [])) >= N_GEN:
                print(f"  CACHED: {len(cached['0'])} seqs × {len(cached)} rounds")
                snapshots = {int(k): v for k, v in cached.items()}
                n_rounds = len(snapshots)
            else:
                snapshots, n_rounds = generate_iterative_with_snapshots(
                    ref_seqs, model_650m, alphabet, device, N_GEN,
                    steering_vectors=sv, glp_project_fn=glp_fn, glp_layer=GLP_LAYER,
                    mask_ratio=MASK_RATIO, temperature=TEMPERATURE, top_p=TOP_P,
                )
                with open(cache_file, 'w') as f:
                    json.dump({str(k): v for k, v in snapshots.items()}, f)
        else:
            snapshots, n_rounds = generate_iterative_with_snapshots(
                ref_seqs, model_650m, alphabet, device, N_GEN,
                steering_vectors=sv, glp_project_fn=glp_fn, glp_layer=GLP_LAYER,
                mask_ratio=MASK_RATIO, temperature=TEMPERATURE, top_p=TOP_P,
            )
            with open(cache_file, 'w') as f:
                json.dump({str(k): v for k, v in snapshots.items()}, f)

        # Evaluate sol at each round
        trajectory = {}
        for r in range(n_rounds):
            seqs = snapshots[r]
            mean_prob, sol_ratio, _ = evaluate_sol(
                seqs, model_650m, alphabet, predictor, device
            )
            trajectory[r] = {
                'sol_mean_prob': float(mean_prob),
                'sol_ratio': float(sol_ratio),
            }
            # Save CSV for pPPL eval later
            csv_path = os.path.join(method_dir, f'round_{r}.csv')
            pd.DataFrame({'sequence': seqs}).to_csv(csv_path, index=False)

        print(f"  Rounds: {n_rounds}, sol trajectory:")
        for r in range(n_rounds):
            t = trajectory[r]
            print(f"    round {r+1}: sol_ratio={t['sol_ratio']*100:.1f}%, mean_prob={t['sol_mean_prob']:.4f}")

        all_trajectories[method_name] = trajectory
        sys.stdout.flush()

    # Free GPU memory
    del model_650m, predictor, glp_model, sv_all, sv_single
    torch.cuda.empty_cache()
    print(f"\nPhase 1 complete.")

    # ==================== Phase 2: pPPL Evaluation ====================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: pPPL Evaluation with ESM2-{PPL_MODEL} (GPUs {PPL_GPU_IDS})")
    print(f"{'=' * 70}")

    # Also evaluate reference pPPL
    print(f"\nEvaluating Reference pPPL...")
    ref_ppls = compute_pseudo_perplexity_multi_gpu(
        ref_seqs[:N_GEN], PPL_MODEL, PPL_GPU_IDS, BATCH_MASKS
    )
    ref_ppl_mean = float(np.mean(ref_ppls))
    ref_ppl_median = float(np.median(ref_ppls))
    print(f"  Reference pPPL: mean={ref_ppl_mean:.4f}, median={ref_ppl_median:.4f}")

    for method_name in all_trajectories:
        trajectory = all_trajectories[method_name]
        n_rounds = len(trajectory)
        method_dir = os.path.join(OUTPUT_DIR, method_name.replace(' ', '_').replace('+', '_').replace('=', ''))

        print(f"\n--- {method_name} ({n_rounds} rounds) ---")

        for r in range(n_rounds):
            csv_path = os.path.join(method_dir, f'round_{r}.csv')
            df = pd.read_csv(csv_path)
            seqs = df['sequence'].tolist()

            ppls = compute_pseudo_perplexity_multi_gpu(
                seqs, PPL_MODEL, PPL_GPU_IDS, BATCH_MASKS
            )
            ppls_arr = np.array(ppls)
            trajectory[r]['ppl_mean'] = float(ppls_arr.mean())
            trajectory[r]['ppl_median'] = float(np.median(ppls_arr))
            trajectory[r]['ppl_std'] = float(ppls_arr.std())
            print(f"  round {r+1}: pPPL={trajectory[r]['ppl_mean']:.4f}")

        # Save intermediate
        with open(os.path.join(OUTPUT_DIR, 'trajectories.json'), 'w') as f:
            json.dump({
                'reference': {
                    'sol_mean_prob': float(ref_mean_prob),
                    'sol_ratio': float(ref_sol_ratio),
                    'ppl_mean': ref_ppl_mean,
                    'ppl_median': ref_ppl_median,
                },
                'methods': all_trajectories,
            }, f, indent=2)

    # ==================== Summary + Plot ====================
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")

    # Print table
    for method_name in all_trajectories:
        trajectory = all_trajectories[method_name]
        n_rounds = len(trajectory)
        print(f"\n{method_name}:")
        print(f"  {'Round':<6} | {'Sol %':>7} | {'Sol Prob':>9} | {'pPPL mean':>10} | {'pPPL med':>10}")
        print(f"  {'-' * 55}")
        for r in range(n_rounds):
            t = trajectory[r]
            ppl_mean = t.get('ppl_mean', float('nan'))
            ppl_med = t.get('ppl_median', float('nan'))
            print(f"  {r+1:<6} | {t['sol_ratio']*100:>6.1f}% | {t['sol_mean_prob']:>9.4f} | {ppl_mean:>10.4f} | {ppl_med:>10.4f}")

    # Save final
    with open(os.path.join(OUTPUT_DIR, 'trajectories.json'), 'w') as f:
        json.dump({
            'reference': {
                'sol_mean_prob': float(ref_mean_prob),
                'sol_ratio': float(ref_sol_ratio),
                'ppl_mean': ref_ppl_mean,
                'ppl_median': ref_ppl_median,
            },
            'methods': all_trajectories,
        }, f, indent=2)

    # ==================== Plot ====================
    print(f"\nGenerating plots...")
    plot_trajectories(all_trajectories, ref_mean_prob, ref_sol_ratio, ref_ppl_mean)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("=" * 70)


def plot_trajectories(all_trajectories, ref_sol_prob, ref_sol_ratio, ref_ppl_mean):
    """Generate per-steps subplots: for each steps value, plot u lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    # --- Plot 1: One figure per steps value (5 figures × 2 subplots) ---
    u_colors = {0.1: '#d62728', 0.3: '#ff7f0e', 0.5: '#9467bd',
                0.7: '#8c564b', 0.9: '#e377c2', 1.0: '#7f7f7f'}

    for steps in STEPS_VALUES:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Steps = {steps}', fontsize=14, fontweight='bold')

        # Baselines
        for bname in ['No Steering', 'L17 no GLP']:
            if bname not in all_trajectories:
                continue
            traj = all_trajectories[bname]
            n_rounds = len(traj)
            x = list(range(1, n_rounds + 1))
            color = '#1f77b4' if bname == 'No Steering' else '#2ca02c'
            ax1.plot(x, [traj[r].get('ppl_mean', float('nan')) for r in range(n_rounds)],
                     'o-', label=bname, color=color, markersize=4, linewidth=2)
            ax2.plot(x, [traj[r]['sol_mean_prob'] for r in range(n_rounds)],
                     'o-', label=bname, color=color, markersize=4, linewidth=2)

        # GLP lines for this steps value
        for u in U_VALUES:
            mname = f'L17+GLP u={u} s={steps}'
            if mname not in all_trajectories:
                continue
            traj = all_trajectories[mname]
            n_rounds = len(traj)
            x = list(range(1, n_rounds + 1))
            ax1.plot(x, [traj[r].get('ppl_mean', float('nan')) for r in range(n_rounds)],
                     'o-', label=f'u={u}', color=u_colors[u], markersize=4)
            ax2.plot(x, [traj[r]['sol_mean_prob'] for r in range(n_rounds)],
                     'o-', label=f'u={u}', color=u_colors[u], markersize=4)

        # Reference lines
        ax1.axhline(y=ref_ppl_mean, color='black', linestyle='--', alpha=0.5, label=f'Reference ({ref_ppl_mean:.2f})')
        ax2.axhline(y=ref_sol_prob, color='black', linestyle='--', alpha=0.5, label=f'Reference ({ref_sol_prob:.4f})')

        ax1.set_xlabel('Decoding Step')
        ax1.set_ylabel('pPPL (ESM2-3B)')
        ax1.set_title('pPPL vs Decoding Step')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Decoding Step')
        ax2.set_ylabel('Sol Mean Prob')
        ax2.set_title('Solubility vs Decoding Step')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'trajectory_steps{steps}.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f'trajectory_steps{steps}.pdf'), bbox_inches='tight')
        plt.close()

    # --- Plot 2: Summary — final round pPPL heatmap (u × steps) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ppl_matrix = np.full((len(U_VALUES), len(STEPS_VALUES)), np.nan)
    sol_matrix = np.full((len(U_VALUES), len(STEPS_VALUES)), np.nan)

    for si, steps in enumerate(STEPS_VALUES):
        for ui, u in enumerate(U_VALUES):
            mname = f'L17+GLP u={u} s={steps}'
            if mname in all_trajectories:
                traj = all_trajectories[mname]
                last_r = len(traj) - 1
                ppl_matrix[ui, si] = traj[last_r].get('ppl_mean', np.nan)
                sol_matrix[ui, si] = traj[last_r].get('sol_mean_prob', np.nan)

    im1 = ax1.imshow(ppl_matrix, aspect='auto', cmap='RdYlGn_r')
    ax1.set_xticks(range(len(STEPS_VALUES)))
    ax1.set_xticklabels(STEPS_VALUES)
    ax1.set_yticks(range(len(U_VALUES)))
    ax1.set_yticklabels(U_VALUES)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('u')
    ax1.set_title('Final pPPL (round 10)')
    for i in range(len(U_VALUES)):
        for j in range(len(STEPS_VALUES)):
            ax1.text(j, i, f'{ppl_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(sol_matrix, aspect='auto', cmap='RdYlGn')
    ax2.set_xticks(range(len(STEPS_VALUES)))
    ax2.set_xticklabels(STEPS_VALUES)
    ax2.set_yticks(range(len(U_VALUES)))
    ax2.set_yticklabels(U_VALUES)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('u')
    ax2.set_title('Final Sol Mean Prob (round 10)')
    for i in range(len(U_VALUES)):
        for j in range(len(STEPS_VALUES)):
            ax2.text(j, i, f'{sol_matrix[i, j]:.4f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_round_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_round_heatmap.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  Plots saved to {OUTPUT_DIR}/trajectory_steps*.png/pdf and final_round_heatmap.png/pdf")


if __name__ == "__main__":
    main()
