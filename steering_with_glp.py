"""
Steering with GLP On-Manifold Projection
==========================================
在 ESM2-650M 的 Layer 17 施加 steering vector, 然后用 GLP 将
steered activations 投影回蛋白质激活流形, 以保持序列自然度.

流程:
  1. ESM2-650M forward, 在 Layer 17 加 steering vector
  2. GLP SDEdit: normalize → 加噪到 timestep u → denoise → denormalize
  3. 继续后续 layer 的 forward
  4. mask-predict 迭代生成序列
  5. 评估 sol (oracle) + pPPL (ESM2-3B)

Usage:
    python steering_with_glp.py \
        --glp_path generative_latent_prior/runs/glp-esm2-650m-layer17-d6 \
        --gpu_gen cuda:0 --gpu_ppl 0 1 2 3 \
        --n_gen 100 --u 0.5
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generative_latent_prior'))

from utils.esm2_utils import load_esm2_model, generate_sequences
from evaluate_ppl import compute_pseudo_perplexity_multi_gpu, compute_pseudo_perplexity, load_esm2_model as load_ppl_model
from generative_latent_prior.glp.denoiser import GLP
from generative_latent_prior.glp import flow_matching

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ========================= GLP Projection =========================

def build_glp_projection_fn(glp_model, u=0.5, num_timesteps=20):
    """Build a function that projects activations on-manifold using GLP."""
    scheduler = glp_model.scheduler
    # 关键: 先设置 timesteps, 确保 fm_prepare 和 sample_on_manifold 用同一个 schedule
    scheduler.set_timesteps(num_timesteps)

    def project_on_manifold(acts):
        """
        acts: (T, B, D) - ESM2 internal format (seq_len, batch, hidden_dim)
        returns: (T, B, D) projected activations
        """
        # Reshape to (B*T, 1, D) for GLP
        T, B, D = acts.shape
        latents = acts.permute(1, 0, 2).reshape(B * T, 1, D)  # (B*T, 1, D)

        # Normalize
        latents = glp_model.normalizer.normalize(latents)

        # Add noise to timestep u
        noise = torch.randn_like(latents)
        noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
            scheduler,
            latents,
            noise,
            u=torch.ones(latents.shape[0], device=latents.device) * u,
        )

        # Denoise (SDEdit)
        latents = flow_matching.sample_on_manifold(
            glp_model,
            noisy_latents,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
        )

        # Denormalize
        latents = glp_model.normalizer.denormalize(latents)

        # Reshape back to (T, B, D)
        result = latents.reshape(B, T, D).permute(1, 0, 2)
        return result.to(dtype=acts.dtype)

    return project_on_manifold


def steering_forward_with_glp(
    self, tokens, repr_layers=[], need_head_weights=False,
    return_contacts=False, steering_vectors=None,
    glp_project_fn=None, glp_layer=17
):
    """
    Modified steering_forward that applies GLP projection at a specific layer.
    """
    if return_contacts:
        need_head_weights = True

    assert tokens.ndim == 2
    padding_mask = tokens.eq(self.padding_idx)

    x = self.embed_scale * self.embed_tokens(tokens)
    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x

    if need_head_weights:
        attn_weights = []

    x = x.transpose(0, 1)  # (B, T, E) => (T, B, E)

    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(self.layers):
        x, attn = layer(
            x,
            self_attn_padding_mask=padding_mask,
            need_head_weights=need_head_weights,
        )

        # Apply steering only at the target layer
        if steering_vectors is not None and layer_idx == glp_layer:
            add_x = steering_vectors[layer_idx]
            new_x = x + add_x
            new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
            x = new_x * (x_norm / new_x_norm)

            # Apply GLP on-manifold projection after steering
            if glp_project_fn is not None:
                x = glp_project_fn(x)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            attn_weights.append(attn.transpose(1, 0))

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)

    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    x = self.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
        attentions = torch.stack(attn_weights, 1)
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.type_as(attentions)
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
        result["attentions"] = attentions
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts

    return result


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
    features = extract_features_650m(seqs, esm_model, alphabet, device)
    with torch.no_grad():
        scores = predictor(features.to(device)).cpu()
    probs = torch.sigmoid(scores).numpy()
    labels = (probs >= 0.5).astype(int)
    return probs.mean(), labels.mean(), probs


# ========================= Generation =========================

def generate_with_glp(ref_seqs, model, alphabet, steering_vectors_all,
                       glp_project_fn, glp_layer, device, n_gen, gen_params):
    """Generate sequences with single-layer steering + GLP projection."""
    from utils.gen_utils import sample_top_p
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    gen_seqs = []
    for i in tqdm(range(n_gen), desc="Generating (L17+GLP)"):
        seq = ref_seqs[i % len(ref_seqs)]
        _, _, tokens = batch_converter([("protein", seq)])
        tokens = tokens.to(device).clone()
        length = tokens.size(1) - 2
        candidate_sites = list(range(length))
        rounds = math.ceil(1.0 / gen_params['mask_ratio'])

        for _ in range(rounds):
            mask_size = min(math.ceil(length * gen_params['mask_ratio']), len(candidate_sites))
            if mask_size == 0:
                break

            indices = torch.randperm(len(candidate_sites))[:mask_size]
            mask_positions = torch.tensor([candidate_sites[idx] for idx in indices]) + 1
            candidate_sites = [site for idx, site in enumerate(candidate_sites) if idx not in indices]

            seq_token = tokens.clone()
            seq_token[0, mask_positions] = mask_idx

            with torch.no_grad():
                outputs = model.steering_forward_glp(
                    tokens=seq_token,
                    steering_vectors=steering_vectors_all,
                    glp_project_fn=glp_project_fn,
                    glp_layer=glp_layer,
                )
            logits = outputs['logits'][0, :, 4:24]

            if gen_params['temperature'] > 0.0:
                probs = torch.softmax(logits / gen_params['temperature'], dim=-1)
                pred_seq = sample_top_p(probs, gen_params['top_p'])
            else:
                pred_seq = torch.argmax(logits, dim=-1)
            pred_seq = pred_seq + 4
            pred_seq[0] = tokens[0, 0]
            pred_seq[-1] = tokens[0, -1]

            tokens[0, mask_positions] = pred_seq[mask_positions]

        # Decode
        from utils.esm2_utils import decode
        gen_seq = decode(alphabet, tokens[:, 1:-1], onehot=False)[0]
        gen_seqs.append(gen_seq)

    return gen_seqs


# ========================= Main =========================

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--glp_path', type=str, required=True,
                        help='Path to trained GLP checkpoint dir')
    parser.add_argument('--glp_checkpoint', type=str, default='final',
                        help='Checkpoint name (default: final)')
    parser.add_argument('--u', type=float, default=0.5,
                        help='SDEdit noise level (0=no projection, 1=full noise)')
    parser.add_argument('--num_timesteps', type=int, default=20,
                        help='Number of denoising steps')
    parser.add_argument('--glp_layer', type=int, default=17,
                        help='Layer to apply steering + GLP')
    parser.add_argument('--gpu_gen', type=str, default='cuda:0')
    parser.add_argument('--gpu_ppl', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--n_gen', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='results/steering_with_glp')
    parser.add_argument('--sv_path', type=str,
                        default='saved_steering_vectors/650M_sol_steering_vectors.pt')
    parser.add_argument('--ref_data', type=str, default='data/sol_easy.csv')
    parser.add_argument('--predictor_path', type=str,
                        default='saved_predictors/sol_predictor_final.pt')
    parser.add_argument('--ppl_model', type=str, default='3B')
    parser.add_argument('--batch_masks', type=int, default=32)
    # Also run baselines for comparison
    parser.add_argument('--run_baselines', action='store_true',
                        help='Also run single-layer-no-GLP and all-layer baselines')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Steering with GLP On-Manifold Projection")
    print("=" * 60)

    # Load reference data
    ref_df = pd.read_csv(args.ref_data)
    ref_seqs = ref_df['sequence'].tolist()
    print(f"Reference sequences: {len(ref_seqs)}")

    # Load ESM2-650M
    print(f"Loading ESM2-650M on {args.gpu_gen}...")
    model_650m, alphabet = load_esm2_model("650M", device=args.gpu_gen)

    # Bind the GLP-aware steering forward
    model_650m.steering_forward_glp = types.MethodType(
        steering_forward_with_glp, model_650m
    )

    # Load steering vectors
    print(f"Loading steering vectors from {args.sv_path}...")
    pos_sv, neg_sv = torch.load(args.sv_path)
    steering_vectors_all = (pos_sv - neg_sv).to(args.gpu_gen)

    # Load GLP
    print(f"Loading GLP from {args.glp_path}...")
    from omegaconf import OmegaConf
    glp_config = OmegaConf.load(os.path.join(args.glp_path, "config.yaml"))
    OmegaConf.resolve(glp_config)
    # Update rep_statistic path to be relative to the checkpoint
    glp_config.glp_kwargs.normalizer_config.rep_statistic = os.path.join(
        args.glp_path, "rep_statistics.pt"
    )
    glp_model = GLP(**glp_config.glp_kwargs)
    glp_model.to(args.gpu_gen)
    glp_model.load_pretrained(args.glp_path, name=args.glp_checkpoint)
    glp_model.eval()
    print(f"  GLP loaded. Denoiser params: {sum(p.numel() for p in glp_model.parameters()):,}")

    # Build projection function
    glp_project_fn = build_glp_projection_fn(
        glp_model, u=args.u, num_timesteps=args.num_timesteps
    )

    # Load sol predictor
    print(f"Loading sol predictor...")
    predictor = PropertyPredictor(embed_dim=1280)
    ckpt = torch.load(args.predictor_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['model_state_dict'])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(args.gpu_gen)
    predictor.eval()

    gen_params = {'mask_ratio': 0.1, 'temperature': 1.0, 'top_p': 0.9}
    results = []

    # ==================== Generate with GLP ====================
    print(f"\n--- Generating with Layer {args.glp_layer} steering + GLP (u={args.u}) ---")
    gen_seqs = generate_with_glp(
        ref_seqs, model_650m, alphabet, steering_vectors_all,
        glp_project_fn, args.glp_layer, args.gpu_gen, args.n_gen, gen_params
    )

    csv_path = os.path.join(args.output_dir, f"L{args.glp_layer}_glp_u{args.u}.csv")
    pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

    mean_prob, sol_ratio, probs = evaluate_sol(
        gen_seqs, model_650m, alphabet, predictor, args.gpu_gen
    )
    print(f"  Sol: mean_prob={mean_prob:.4f}, soluble_ratio={sol_ratio*100:.1f}%")
    results.append({
        'method': f'L{args.glp_layer}+GLP(u={args.u})',
        'csv_path': csv_path,
        'sol_mean_prob': float(mean_prob),
        'sol_ratio': float(sol_ratio),
    })
    sys.stdout.flush()

    # Free 650M
    del model_650m, predictor, glp_model
    torch.cuda.empty_cache()

    # ==================== pPPL Evaluation ====================
    print(f"\nEvaluating pPPL with ESM2-{args.ppl_model}...")
    for res in results:
        df = pd.read_csv(res['csv_path'])
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
        res['ppl_mean'] = float(ppls_arr.mean())
        res['ppl_median'] = float(np.median(ppls_arr))
        res['ppl_std'] = float(ppls_arr.std())
        print(f"  {res['method']}: pPPL={res['ppl_mean']:.4f} ± {res['ppl_std']:.4f}")

    # ==================== Summary ====================
    print(f"\n{'=' * 70}")
    print(f"{'Method':<30} | {'Sol Ratio':>10} | {'pPPL mean':>10} | {'pPPL med':>10}")
    print(f"{'-' * 70}")

    # Add baselines from previous experiments for comparison
    baselines = [
        {'method': 'Reference', 'sol_ratio': 0.056, 'ppl_mean': 5.47},
        {'method': 'No Steering', 'sol_ratio': 0.11, 'ppl_mean': 7.19},
        {'method': 'L17 Single (no GLP)', 'sol_ratio': 0.32, 'ppl_mean': 7.01},
        {'method': 'All-Layer Steering', 'sol_ratio': 0.32, 'ppl_mean': 15.23},
    ]
    for b in baselines:
        print(f"{b['method']:<30} | {b['sol_ratio']*100:>9.1f}% | {b['ppl_mean']:>10.2f} | {'—':>10}")

    for res in results:
        print(f"{res['method']:<30} | {res['sol_ratio']*100:>9.1f}% | {res['ppl_mean']:>10.4f} | {res['ppl_median']:>10.4f}")

    # Save results
    summary = {
        'experiment': 'steering_with_glp',
        'glp_path': args.glp_path,
        'u': args.u,
        'num_timesteps': args.num_timesteps,
        'glp_layer': args.glp_layer,
        'n_gen': args.n_gen,
        'results': results,
    }
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {json_path}")
    print("=" * 70)
