"""
Single-Layer Steering Experiment
================================
对 ESM2-650M 的每一层 (0~32) 单独施加 steering vector,
生成序列后评估溶解度 (oracle predictor) 和自然度 (ESM2-3B pPPL).
目标: 找出哪一层 steering 效果最好, 并与 all-layer steering 对比.

Usage:
    python exp_single_layer_steering.py \
        --gpu_gen cuda:0 --gpu_ppl 0 1 4 5 \
        --n_gen 100 --output_dir results/single_layer_steering
"""

import argparse
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
import esm

# Force unbuffered output for background execution
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from module.steerable_esm2 import steering_forward
from utils.esm2_utils import load_esm2_model, generate_sequences
from evaluate_ppl import compute_pseudo_perplexity_multi_gpu, compute_pseudo_perplexity, load_esm2_model as load_ppl_model


# ========================= Oracle Predictor =========================
class PropertyPredictor(nn.Module):
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x.squeeze(-1)


def extract_features_650m(seqs, model, alphabet, device, batch_size=8):
    """Extract mean-pooled last-layer representations from ESM2-650M."""
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    all_features = []
    for start in range(0, len(seqs), batch_size):
        batch_seqs = seqs[start:start + batch_size]
        data = [("protein", s[:1022]) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[n_layers])
        for i, seq_len in enumerate(batch_lens):
            rep = results["representations"][n_layers][i, 1:seq_len - 1].mean(0).cpu()
            all_features.append(rep)
    return torch.stack(all_features)


def evaluate_sol(seqs, esm_model, alphabet, predictor, device):
    """Evaluate solubility: returns (mean_prob, soluble_ratio)."""
    features = extract_features_650m(seqs, esm_model, alphabet, device)
    with torch.no_grad():
        scores = predictor(features.to(device)).cpu()
    probs = torch.sigmoid(scores).numpy()
    labels = (probs >= 0.5).astype(int)
    return probs.mean(), labels.mean(), probs


def generate_single_layer_seqs(layer_idx, n_layers, steering_vectors_all, ref_seqs,
                                model, alphabet, device, n_gen, args_gen):
    """Generate sequences with steering applied only at one layer."""
    batch_converter = alphabet.get_batch_converter()

    # Create single-layer steering vectors: zero everywhere except target layer
    sv_single = torch.zeros_like(steering_vectors_all)
    sv_single[layer_idx] = steering_vectors_all[layer_idx]

    gen_seqs = []
    for i in tqdm(range(n_gen), desc=f"  Gen L{layer_idx:02d}", leave=False):
        seq = ref_seqs[i % len(ref_seqs)]
        _, _, seq_token = batch_converter([("protein", seq)])
        seq_token = seq_token.to(device)
        new_seq = generate_sequences(seq_token, model, sv_single,
                                      args_gen['mask_ratio'], alphabet,
                                      temperature=args_gen['temperature'],
                                      top_p=args_gen['top_p'])
        gen_seqs.append(new_seq)
    return gen_seqs


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_gen', type=str, default='cuda:0',
                        help='GPU for generation and sol evaluation')
    parser.add_argument('--gpu_ppl', type=int, nargs='+', default=[0, 1, 4, 5],
                        help='GPU IDs for pPPL evaluation')
    parser.add_argument('--n_gen', type=int, default=100,
                        help='Number of sequences to generate per layer')
    parser.add_argument('--output_dir', type=str, default='results/single_layer_steering')
    parser.add_argument('--ppl_model', type=str, default='3B',
                        help='ESM2 model for pPPL evaluation')
    parser.add_argument('--predictor_path', type=str,
                        default='saved_predictors/sol_predictor_final.pt')
    parser.add_argument('--sv_path', type=str,
                        default='saved_steering_vectors/650M_sol_steering_vectors.pt')
    parser.add_argument('--ref_data', type=str, default='data/sol_easy.csv')
    parser.add_argument('--batch_masks', type=int, default=32)
    # Layers to test - default all 33 layers of ESM2-650M
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers to test (default: all 0-32)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ===================== Load everything =====================
    print("=" * 60)
    print("Single-Layer Steering Experiment")
    print("=" * 60)

    # Load reference data
    ref_df = pd.read_csv(args.ref_data)
    ref_seqs = ref_df['sequence'].tolist()
    print(f"Reference sequences: {len(ref_seqs)} from {args.ref_data}")

    # Load ESM2-650M for generation
    print(f"Loading ESM2-650M on {args.gpu_gen}...")
    model_650m, alphabet = load_esm2_model("650M", device=args.gpu_gen)
    model_650m.steering_forward = types.MethodType(steering_forward, model_650m)
    batch_converter = alphabet.get_batch_converter()
    n_layers = model_650m.num_layers  # 33

    # Load steering vectors
    print(f"Loading steering vectors from {args.sv_path}...")
    pos_sv, neg_sv = torch.load(args.sv_path)
    steering_vectors_all = (pos_sv - neg_sv).to(args.gpu_gen)
    print(f"  Steering vectors shape: {steering_vectors_all.shape}")

    # Load sol predictor
    print(f"Loading sol predictor from {args.predictor_path}...")
    predictor = PropertyPredictor(embed_dim=1280)
    ckpt = torch.load(args.predictor_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['model_state_dict'])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(args.gpu_gen)
    predictor.eval()

    # Generation params (same as original pipeline)
    gen_params = {'mask_ratio': 0.1, 'temperature': 1.0, 'top_p': 0.9}

    # Determine layers to test
    layers_to_test = args.layers if args.layers else list(range(n_layers))
    print(f"Layers to test: {layers_to_test}")
    print(f"Sequences per layer: {args.n_gen}")

    # ===================== Generate & Evaluate Sol =====================
    results = []

    for li, layer_idx in enumerate(layers_to_test):
        print(f"\n[{li+1}/{len(layers_to_test)}] Layer {layer_idx}/{n_layers - 1}")

        # Generate sequences
        gen_seqs = generate_single_layer_seqs(
            layer_idx, n_layers, steering_vectors_all, ref_seqs,
            model_650m, alphabet, args.gpu_gen, args.n_gen, gen_params
        )

        # Save generated sequences
        csv_path = os.path.join(args.output_dir, f"layer_{layer_idx}.csv")
        pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

        # Evaluate solubility
        mean_prob, sol_ratio, probs = evaluate_sol(
            gen_seqs, model_650m, alphabet, predictor, args.gpu_gen
        )
        print(f"  Sol: mean_prob={mean_prob:.4f}, soluble_ratio={sol_ratio*100:.1f}%")
        sys.stdout.flush()

        results.append({
            'layer': layer_idx,
            'csv_path': csv_path,
            'sol_mean_prob': mean_prob,
            'sol_ratio': sol_ratio,
        })

    # Free 650M model GPU memory before pPPL evaluation
    del model_650m
    del predictor
    torch.cuda.empty_cache()

    # ===================== Evaluate pPPL =====================
    print(f"\n{'=' * 60}")
    print(f"Evaluating pPPL with ESM2-{args.ppl_model} on GPUs {args.gpu_ppl}")
    print(f"{'=' * 60}")

    for res in results:
        layer_idx = res['layer']
        csv_path = res['csv_path']
        print(f"\n  Layer {layer_idx}: computing pPPL...")
        df = pd.read_csv(csv_path)
        seqs = df['sequence'].tolist()

        if len(args.gpu_ppl) > 1:
            ppls = compute_pseudo_perplexity_multi_gpu(
                seqs, args.ppl_model, args.gpu_ppl, args.batch_masks
            )
        else:
            device = f"cuda:{args.gpu_ppl[0]}"
            ppl_model, ppl_alphabet = load_ppl_model(args.ppl_model, device)
            ppls = compute_pseudo_perplexity(seqs, ppl_model, ppl_alphabet, device, args.batch_masks)

        ppls_arr = np.array(ppls)
        res['ppl_mean'] = ppls_arr.mean()
        res['ppl_median'] = np.median(ppls_arr)
        res['ppl_std'] = ppls_arr.std()
        print(f"  Layer {layer_idx}: pPPL={res['ppl_mean']:.4f} ± {res['ppl_std']:.4f}")

    # ===================== Summary =====================
    print(f"\n{'=' * 70}")
    print(f"{'Layer':>6} | {'Sol Prob':>10} | {'Sol Ratio':>10} | {'pPPL mean':>10} | {'pPPL med':>10}")
    print(f"{'-' * 70}")
    for res in sorted(results, key=lambda x: x['layer']):
        print(f"{res['layer']:>6} | {res['sol_mean_prob']:>10.4f} | {res['sol_ratio']*100:>9.1f}% | {res['ppl_mean']:>10.4f} | {res['ppl_median']:>10.4f}")

    # Find best layer (highest sol_ratio, then lowest pPPL as tiebreaker)
    best_sol = max(results, key=lambda x: (x['sol_ratio'], -x['ppl_mean']))
    best_ppl = min(results, key=lambda x: x['ppl_mean'])
    best_trade = max(results, key=lambda x: x['sol_ratio'] / max(x['ppl_mean'], 1.0))

    print(f"\n  Best sol layer:      {best_sol['layer']} (sol={best_sol['sol_ratio']*100:.1f}%, pPPL={best_sol['ppl_mean']:.2f})")
    print(f"  Best pPPL layer:     {best_ppl['layer']} (sol={best_ppl['sol_ratio']*100:.1f}%, pPPL={best_ppl['ppl_mean']:.2f})")
    print(f"  Best trade-off:      {best_trade['layer']} (sol={best_trade['sol_ratio']*100:.1f}%, pPPL={best_trade['ppl_mean']:.2f})")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Save as JSON too for easy reference
    summary_json = {
        'experiment': 'single_layer_steering',
        'model': 'ESM2-650M',
        'ppl_model': f'ESM2-{args.ppl_model}',
        'n_gen': args.n_gen,
        'ref_data': args.ref_data,
        'best_sol_layer': best_sol['layer'],
        'best_ppl_layer': best_ppl['layer'],
        'best_tradeoff_layer': best_trade['layer'],
        'results': results,
    }
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"JSON summary saved to {json_path}")
    print("=" * 70)
