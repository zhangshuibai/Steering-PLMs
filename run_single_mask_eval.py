"""
Single-Mask Experiment (V2): Isolate Single-Step GLP Error
===========================================================
V1 used 10 rounds of iterative mask-predict, each applying steering+GLP.
Error may accumulate across rounds. This experiment uses a single mask +
single forward to isolate single-step GLP error.

For each reference sequence, randomly sample n_positions positions.
For each position: mask it → one forward pass → nucleus sampling → fill back.
All groups share the same seed → same mask positions → fair comparison.

9 groups:
  0. Reference (original sequences, no generation)
  1. No Steering (vanilla ESM2 forward)
  2. L17 no GLP (steering at L17 only, no GLP)
  3-8. L17+GLP with u ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}, steps=100

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python run_single_mask_eval.py
"""

import os
import sys
import types
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

N_POSITIONS = 10
SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.9
GLP_LAYER = 17
GLP_PATH = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6'
GLP_CHECKPOINT = 'final'
SV_PATH = 'saved_steering_vectors/650M_sol_steering_vectors.pt'
PREDICTOR_PATH = 'saved_predictors/sol_predictor_final.pt'
REF_DATA = 'data/sol_easy.csv'
PPL_MODEL = '3B'
PPL_GPU_IDS = [0, 1, 2, 3]
BATCH_MASKS = 32
OUTPUT_DIR = 'results/single_mask_eval'

U_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
GLP_STEPS = 100


# ========================= Core Function =========================

def generate_single_mask(ref_seqs, model, alphabet, device,
                         steering_vectors=None, glp_project_fn=None,
                         glp_layer=17, n_positions=10, seed=42):
    """
    For each ref_seq, randomly sample n_positions positions. For each position:
      1. Mask that position
      2. One forward pass (choosing mode based on parameters)
      3. Nucleus sampling to fill back
    Returns len(ref_seqs) * n_positions sequences.
    All groups use the same seed → same mask positions → fair comparison.
    """
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    rng = np.random.RandomState(seed)
    gen_seqs = []

    for seq_idx, seq in enumerate(tqdm(ref_seqs, desc="Single-mask generation")):
        seq_len = len(seq)
        # Sample n_positions random positions (0-indexed within the sequence)
        positions = rng.choice(seq_len, size=min(n_positions, seq_len), replace=False)

        for pos in positions:
            # Tokenize
            _, _, tokens = batch_converter([("protein", seq)])
            tokens = tokens.to(device)
            token_pos = pos + 1  # +1 for BOS token

            # Mask
            masked_tokens = tokens.clone()
            masked_tokens[0, token_pos] = mask_idx

            # Forward pass
            with torch.no_grad():
                if steering_vectors is None and glp_project_fn is None:
                    # Mode 1: vanilla ESM2
                    outputs = model(masked_tokens)
                elif steering_vectors is not None and glp_project_fn is None:
                    # Mode 2: steering only (L17)
                    outputs = model.steering_forward(
                        tokens=masked_tokens,
                        steering_vectors=steering_vectors,
                    )
                else:
                    # Mode 3: steering + GLP
                    outputs = model.steering_forward_glp(
                        tokens=masked_tokens,
                        steering_vectors=steering_vectors,
                        glp_project_fn=glp_project_fn,
                        glp_layer=glp_layer,
                    )

            # Extract logits for standard amino acids (indices 4:24)
            logits = outputs['logits'][0, token_pos, 4:24]

            # Nucleus sampling
            if TEMPERATURE > 0.0:
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                # sample_top_p expects (seq_len, n_vocab), we have (n_vocab,)
                probs_2d = probs.unsqueeze(0)
                sampled = sample_top_p(probs_2d, TOP_P)
                pred_token = sampled[0].item() + 4
            else:
                pred_token = logits.argmax().item() + 4

            # Fill back
            result_tokens = tokens.clone()
            result_tokens[0, token_pos] = pred_token

            # Decode
            gen_seq = decode(alphabet, result_tokens[:, 1:-1], onehot=False)[0]
            gen_seqs.append(gen_seq)

    return gen_seqs


# ========================= Main =========================

def main():
    mp.set_start_method('spawn', force=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda:0'

    # Load reference data
    ref_df = pd.read_csv(REF_DATA)
    ref_seqs = ref_df['sequence'].tolist()
    n_ref = len(ref_seqs)
    n_expected = n_ref * N_POSITIONS
    print(f"Reference sequences: {n_ref}")
    print(f"Expected sequences per group: {n_expected}")

    # ==================== Phase 1: Generation + Sol Eval ====================
    print("=" * 70)
    print("Phase 1: Single-Mask Generation + Sol Evaluation")
    print("=" * 70)

    # Load ESM2-650M
    print(f"Loading ESM2-650M on {device}...")
    model_650m, alphabet = load_esm2_model("650M", device=device)

    # Bind steering forwards
    model_650m.steering_forward = types.MethodType(steering_forward, model_650m)
    model_650m.steering_forward_glp = types.MethodType(
        steering_forward_with_glp, model_650m
    )

    # Load steering vectors
    print(f"Loading steering vectors from {SV_PATH}...")
    pos_sv, neg_sv = torch.load(SV_PATH)
    sv_all = (pos_sv - neg_sv).to(device)

    # Construct sv_single: zeros everywhere, only L17 has value
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

    all_results = []

    # ---------- Group 0: Reference ----------
    print(f"\n--- [0/8] Reference (original sequences) ---")
    csv_path = os.path.join(OUTPUT_DIR, 'reference.csv')
    pd.DataFrame({'sequence': ref_seqs}).to_csv(csv_path, index=False)
    mean_prob, sol_ratio, probs = evaluate_sol(
        ref_seqs, model_650m, alphabet, predictor, device
    )
    print(f"  {n_ref} seqs, sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")
    all_results.append({
        'method': 'Reference',
        'csv_path': csv_path,
        'n_seqs': n_ref,
        'sol_mean_prob': float(mean_prob),
        'sol_ratio': float(sol_ratio),
    })

    # ---------- Group 1: No Steering ----------
    print(f"\n--- [1/8] No Steering ---")
    csv_path = os.path.join(OUTPUT_DIR, 'no_steering.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) >= n_expected:
            gen_seqs = df['sequence'].tolist()
            print(f"  CACHED: {len(gen_seqs)} seqs")
        else:
            gen_seqs = generate_single_mask(
                ref_seqs, model_650m, alphabet, device,
                steering_vectors=None, glp_project_fn=None,
                n_positions=N_POSITIONS, seed=SEED,
            )
            pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)
    else:
        gen_seqs = generate_single_mask(
            ref_seqs, model_650m, alphabet, device,
            steering_vectors=None, glp_project_fn=None,
            n_positions=N_POSITIONS, seed=SEED,
        )
        pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

    mean_prob, sol_ratio, probs = evaluate_sol(
        gen_seqs, model_650m, alphabet, predictor, device
    )
    print(f"  {len(gen_seqs)} seqs, sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")
    all_results.append({
        'method': 'No Steering',
        'csv_path': csv_path,
        'n_seqs': len(gen_seqs),
        'sol_mean_prob': float(mean_prob),
        'sol_ratio': float(sol_ratio),
    })

    # ---------- Group 2: L17 no GLP ----------
    print(f"\n--- [2/8] L17 Steering, no GLP ---")
    csv_path = os.path.join(OUTPUT_DIR, 'l17_no_glp.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) >= n_expected:
            gen_seqs = df['sequence'].tolist()
            print(f"  CACHED: {len(gen_seqs)} seqs")
        else:
            gen_seqs = generate_single_mask(
                ref_seqs, model_650m, alphabet, device,
                steering_vectors=sv_single, glp_project_fn=None,
                glp_layer=GLP_LAYER, n_positions=N_POSITIONS, seed=SEED,
            )
            pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)
    else:
        gen_seqs = generate_single_mask(
            ref_seqs, model_650m, alphabet, device,
            steering_vectors=sv_single, glp_project_fn=None,
            glp_layer=GLP_LAYER, n_positions=N_POSITIONS, seed=SEED,
        )
        pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

    mean_prob, sol_ratio, probs = evaluate_sol(
        gen_seqs, model_650m, alphabet, predictor, device
    )
    print(f"  {len(gen_seqs)} seqs, sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")
    all_results.append({
        'method': 'L17 no GLP',
        'csv_path': csv_path,
        'n_seqs': len(gen_seqs),
        'sol_mean_prob': float(mean_prob),
        'sol_ratio': float(sol_ratio),
    })

    # ---------- Groups 3-8: L17 + GLP with varying u ----------
    for i, u in enumerate(U_VALUES):
        group_num = 3 + i
        method = f'L17+GLP(u={u},s={GLP_STEPS})'
        print(f"\n--- [{group_num}/8] {method} ---")

        csv_path = os.path.join(OUTPUT_DIR, f'l17_glp_u{u}_s{GLP_STEPS}.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) >= n_expected:
                gen_seqs = df['sequence'].tolist()
                print(f"  CACHED: {len(gen_seqs)} seqs")
                mean_prob, sol_ratio, probs = evaluate_sol(
                    gen_seqs, model_650m, alphabet, predictor, device
                )
                print(f"  {len(gen_seqs)} seqs, sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")
                all_results.append({
                    'method': method,
                    'csv_path': csv_path,
                    'n_seqs': len(gen_seqs),
                    'sol_mean_prob': float(mean_prob),
                    'sol_ratio': float(sol_ratio),
                    'u': u,
                    'steps': GLP_STEPS,
                })
                continue

        # Build projection function for this u
        glp_project_fn = build_glp_projection_fn(glp_model, u=u, num_timesteps=GLP_STEPS)

        gen_seqs = generate_single_mask(
            ref_seqs, model_650m, alphabet, device,
            steering_vectors=sv_all, glp_project_fn=glp_project_fn,
            glp_layer=GLP_LAYER, n_positions=N_POSITIONS, seed=SEED,
        )
        pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

        mean_prob, sol_ratio, probs = evaluate_sol(
            gen_seqs, model_650m, alphabet, predictor, device
        )
        print(f"  {len(gen_seqs)} seqs, sol_ratio={sol_ratio*100:.1f}%, mean_prob={mean_prob:.4f}")
        all_results.append({
            'method': method,
            'csv_path': csv_path,
            'n_seqs': len(gen_seqs),
            'sol_mean_prob': float(mean_prob),
            'sol_ratio': float(sol_ratio),
            'u': u,
            'steps': GLP_STEPS,
        })
        sys.stdout.flush()

    # Free GPU memory for phase 2
    del model_650m, predictor, glp_model, sv_all, sv_single
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

        # Save intermediate summary
        with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # ==================== Summary ====================
    print(f"\n{'=' * 70}")
    print(f"{'Method':<30} | {'N':>5} | {'Sol %':>7} | {'pPPL mean':>10} | {'pPPL med':>10} | {'pPPL std':>10}")
    print(f"{'-' * 80}")

    for res in all_results:
        ppl_mean = res.get('ppl_mean', float('nan'))
        ppl_med = res.get('ppl_median', float('nan'))
        ppl_std = res.get('ppl_std', float('nan'))
        print(f"{res['method']:<30} | {res['n_seqs']:>5} | {res['sol_ratio']*100:>6.1f}% | {ppl_mean:>10.4f} | {ppl_med:>10.4f} | {ppl_std:>10.4f}")

    # Save final summary
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'summary.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
