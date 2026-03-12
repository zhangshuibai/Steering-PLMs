"""
Comprehensive Evaluation: GLP Steering vs Baselines
=====================================================
Systematic comparison across u ∈ {0.1, 0.3, 0.5, 0.7, 0.9} and
steps ∈ {25, 50, 100, 200, 400} for L17+GLP steering, plus 4 baselines.

Metrics: Sol Ratio (oracle predictor) + pPPL (ESM2-3B).

Phase 1: Generate sequences + sol eval (single GPU)
Phase 2: pPPL eval (multi-GPU with ESM2-3B)

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python run_comprehensive_eval.py
"""

import os
import sys
import types
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generative_latent_prior'))

from steering_with_glp import (
    build_glp_projection_fn,
    generate_with_glp,
    steering_forward_with_glp,
    evaluate_sol,
    PropertyPredictor,
    extract_features_650m,
)
from evaluate_ppl import compute_pseudo_perplexity_multi_gpu

from generative_latent_prior.glp.denoiser import GLP
from utils.esm2_utils import load_esm2_model

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ========================= Config =========================

U_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
STEPS_VALUES = [25, 50, 100, 200, 400]

BASELINE_CSVS = {
    'Reference': 'data/sol_easy.csv',
    'No Steering': 'results/ESM2_gen_no_steering_sol_easy.csv',
    'L17 Single (no GLP)': 'results/single_layer_steering/layer_17.csv',
    'All-Layer Steering': 'results/ESM2_gen_steering_sol_easy.csv',
}

GLP_PATH = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6'
GLP_CHECKPOINT = 'final'
GLP_LAYER = 17
SV_PATH = 'saved_steering_vectors/650M_sol_steering_vectors.pt'
PREDICTOR_PATH = 'saved_predictors/sol_predictor_final.pt'
REF_DATA = 'data/sol_easy.csv'
N_GEN = 100
PPL_MODEL = '3B'
PPL_GPU_IDS = [0, 1, 2, 3]
BATCH_MASKS = 32
OUTPUT_DIR = 'results/comprehensive_eval'
GEN_PARAMS = {'mask_ratio': 0.1, 'temperature': 1.0, 'top_p': 0.9}


def main():
    mp.set_start_method('spawn', force=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'glp'), exist_ok=True)

    device = 'cuda:0'
    all_results = []  # list of dicts: method, csv_path, sol_ratio, ...

    # ==================== Phase 1: Generation + Sol Eval ====================
    print("=" * 70)
    print("Phase 1: Sequence Generation + Sol Evaluation")
    print("=" * 70)

    # Load reference data
    ref_df = pd.read_csv(REF_DATA)
    ref_seqs = ref_df['sequence'].tolist()
    print(f"Reference sequences: {len(ref_seqs)}")

    # Load ESM2-650M
    print(f"Loading ESM2-650M on {device}...")
    model_650m, alphabet = load_esm2_model("650M", device=device)
    model_650m.steering_forward_glp = types.MethodType(
        steering_forward_with_glp, model_650m
    )

    # Load steering vectors
    print(f"Loading steering vectors from {SV_PATH}...")
    pos_sv, neg_sv = torch.load(SV_PATH)
    steering_vectors_all = (pos_sv - neg_sv).to(device)

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

    # ---------- Baselines: read existing CSVs ----------
    print(f"\n--- Evaluating Baselines ---")
    for method, csv_path in BASELINE_CSVS.items():
        df = pd.read_csv(csv_path)
        seqs = df['sequence'].tolist()
        mean_prob, sol_ratio, probs = evaluate_sol(
            seqs, model_650m, alphabet, predictor, device
        )
        print(f"  {method}: {len(seqs)} seqs, sol_ratio={sol_ratio*100:.1f}%")
        all_results.append({
            'method': method,
            'csv_path': csv_path,
            'n_seqs': len(seqs),
            'sol_mean_prob': float(mean_prob),
            'sol_ratio': float(sol_ratio),
        })

    # ---------- GLP: 5 u × 5 steps = 25 groups ----------
    print(f"\n--- Generating GLP sequences (5 u × 5 steps = 25 groups) ---")
    total_groups = len(U_VALUES) * len(STEPS_VALUES)
    group_idx = 0

    for steps in STEPS_VALUES:
        for u in U_VALUES:
            group_idx += 1
            csv_path = os.path.join(OUTPUT_DIR, 'glp', f'u{u}_steps{steps}.csv')
            method = f'L17+GLP(u={u},s={steps})'

            # Check if already generated
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if len(df) >= N_GEN:
                    seqs = df['sequence'].tolist()
                    mean_prob, sol_ratio, probs = evaluate_sol(
                        seqs, model_650m, alphabet, predictor, device
                    )
                    print(f"  [{group_idx}/{total_groups}] {method}: CACHED, sol={sol_ratio*100:.1f}%")
                    all_results.append({
                        'method': method,
                        'csv_path': csv_path,
                        'n_seqs': len(seqs),
                        'sol_mean_prob': float(mean_prob),
                        'sol_ratio': float(sol_ratio),
                        'u': u,
                        'steps': steps,
                    })
                    continue

            print(f"  [{group_idx}/{total_groups}] {method}: generating {N_GEN} sequences...")

            # Build projection function for this (u, steps)
            glp_project_fn = build_glp_projection_fn(glp_model, u=u, num_timesteps=steps)

            # Generate
            gen_seqs = generate_with_glp(
                ref_seqs, model_650m, alphabet, steering_vectors_all,
                glp_project_fn, GLP_LAYER, device, N_GEN, GEN_PARAMS
            )

            # Save CSV
            pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

            # Evaluate sol
            mean_prob, sol_ratio, probs = evaluate_sol(
                gen_seqs, model_650m, alphabet, predictor, device
            )
            print(f"    sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")

            all_results.append({
                'method': method,
                'csv_path': csv_path,
                'n_seqs': len(gen_seqs),
                'sol_mean_prob': float(mean_prob),
                'sol_ratio': float(sol_ratio),
                'u': u,
                'steps': steps,
            })
            sys.stdout.flush()

    # Free GPU memory for phase 2
    del model_650m, predictor, glp_model, steering_vectors_all
    torch.cuda.empty_cache()
    print(f"\nPhase 1 complete. {len(all_results)} groups evaluated for sol.")

    # ==================== Phase 2: pPPL Evaluation ====================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: pPPL Evaluation with ESM2-{PPL_MODEL} (GPUs {PPL_GPU_IDS})")
    print(f"{'=' * 70}")

    for i, res in enumerate(all_results):
        print(f"\n[{i+1}/{len(all_results)}] {res['method']}")
        df = pd.read_csv(res['csv_path'])
        seqs = df['sequence'].tolist()

        ppls = compute_pseudo_perplexity_multi_gpu(
            seqs, PPL_MODEL, PPL_GPU_IDS, BATCH_MASKS
        )

        ppls_arr = np.array(ppls)
        res['ppl_mean'] = float(ppls_arr.mean())
        res['ppl_median'] = float(np.median(ppls_arr))
        res['ppl_std'] = float(ppls_arr.std())
        print(f"  pPPL: mean={res['ppl_mean']:.4f}, median={res['ppl_median']:.4f}, std={res['ppl_std']:.4f}")

        # Save intermediate summary after each group
        with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # ==================== Summary ====================
    print(f"\n{'=' * 70}")
    print(f"{'Method':<30} | {'N':>4} | {'Sol %':>7} | {'pPPL mean':>10} | {'pPPL med':>10} | {'pPPL std':>10}")
    print(f"{'-' * 80}")

    for res in all_results:
        ppl_mean = res.get('ppl_mean', float('nan'))
        ppl_med = res.get('ppl_median', float('nan'))
        ppl_std = res.get('ppl_std', float('nan'))
        print(f"{res['method']:<30} | {res['n_seqs']:>4} | {res['sol_ratio']*100:>6.1f}% | {ppl_mean:>10.4f} | {ppl_med:>10.4f} | {ppl_std:>10.4f}")

    # Save final summary
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'summary.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
