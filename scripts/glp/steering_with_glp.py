"""
Steering with GLP On-Manifold Projection
==========================================
Layer 17 steering + GLP SDEdit projection for ESM2-650M.

Flow:
  1. ESM2-650M forward, inject steering vector at Layer 17
  2. GLP SDEdit: normalize -> add noise to level u -> denoise -> denormalize
  3. Continue remaining layers
  4. Iterative mask-predict generation (10 rounds, mask_ratio=0.1)

Usage:
    python scripts/glp/steering_with_glp.py \
        --glp_path generative_latent_prior/runs/glp-esm2-650m-layer17-d6 \
        --u 0.5 --n_gen 100 \
        --output_dir new-results/glp
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

# Project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'generative_latent_prior'))

from utils.esm2_utils import load_esm2_model, decode
from utils.gen_utils import sample_top_p
from evaluation.common import load_predictor, extract_features, evaluate_sol
from generative_latent_prior.glp.denoiser import GLP
from generative_latent_prior.glp import flow_matching

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ========================= GLP Loading =========================

def load_glp(glp_path, device, checkpoint='final'):
    """Load GLP model from checkpoint directory."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(os.path.join(glp_path, 'config.yaml'))
    OmegaConf.resolve(config)
    config.glp_kwargs.normalizer_config.rep_statistic = os.path.join(
        glp_path, 'rep_statistics.pt'
    )
    glp_model = GLP(**config.glp_kwargs)
    glp_model.to(device)
    glp_model.load_pretrained(glp_path, name=checkpoint)
    glp_model.eval()
    print(f"  GLP loaded from {glp_path} ({sum(p.numel() for p in glp_model.parameters()):,} params)")
    return glp_model


# ========================= GLP Projection =========================

def build_glp_projection_fn(glp_model, u=0.5, num_timesteps=20):
    """Build a function that projects activations on-manifold using GLP SDEdit."""
    scheduler = glp_model.scheduler
    scheduler.set_timesteps(num_timesteps)

    def project_on_manifold(acts):
        """
        acts: (T, B, D) - ESM2 internal format (seq_len, batch, hidden_dim)
        returns: (T, B, D) projected activations
        """
        T, B, D = acts.shape
        latents = acts.permute(1, 0, 2).reshape(B * T, 1, D)  # (B*T, 1, D)

        # Normalize -> add noise -> denoise -> denormalize
        latents = glp_model.normalizer.normalize(latents)
        noise = torch.randn_like(latents)
        noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
            scheduler, latents, noise,
            u=torch.ones(latents.shape[0], device=latents.device) * u,
        )
        latents = flow_matching.sample_on_manifold(
            glp_model, noisy_latents,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
        )
        latents = glp_model.normalizer.denormalize(latents)

        result = latents.reshape(B, T, D).permute(1, 0, 2)
        return result.to(dtype=acts.dtype)

    return project_on_manifold


# ========================= Steering Forward with GLP =========================

def steering_forward_with_glp(
    self, tokens, repr_layers=[], need_head_weights=False,
    return_contacts=False, steering_vectors=None,
    glp_project_fn=None, glp_layer=17
):
    """ESM2 forward with steering at glp_layer + GLP projection."""
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
        x, attn = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights)

        # Steering + GLP only at target layer
        if steering_vectors is not None and layer_idx == glp_layer:
            add_x = steering_vectors[layer_idx]
            new_x = x + add_x
            new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
            x = new_x * (x_norm / new_x_norm)

            if glp_project_fn is not None:
                x = glp_project_fn(x)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            attn_weights.append(attn.transpose(1, 0))

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
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


# ========================= Generation =========================

def generate_with_glp(ref_seqs, model, alphabet, steering_vectors,
                      glp_project_fn, glp_layer, device, n_gen,
                      mask_ratio=0.1, temperature=1.0, top_p=0.9):
    """Generate sequences with single-layer steering + GLP projection."""
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    gen_seqs = []
    for i in tqdm(range(n_gen), desc=f"Generating (L{glp_layer}+GLP)"):
        seq = ref_seqs[i % len(ref_seqs)]
        _, _, tokens = batch_converter([("protein", seq)])
        tokens = tokens.to(device).clone()
        length = tokens.size(1) - 2
        candidate_sites = list(range(length))
        rounds = math.ceil(1.0 / mask_ratio)

        for _ in range(rounds):
            mask_size = min(math.ceil(length * mask_ratio), len(candidate_sites))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L17 Steering + GLP On-Manifold Projection")
    parser.add_argument('--glp_path', type=str,
                        default='generative_latent_prior/runs/glp-esm2-650m-layer17-d6')
    parser.add_argument('--u', type=float, default=0.5)
    parser.add_argument('--num_timesteps', type=int, default=20)
    parser.add_argument('--glp_layer', type=int, default=17)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_gen', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='new-results/glp')
    parser.add_argument('--sv_path', type=str,
                        default='saved_steering_vectors/650M_sol_steering_vectors.pt')
    parser.add_argument('--ref_data', type=str, default='data/sol_easy.csv')
    parser.add_argument('--predictor_path', type=str,
                        default='evaluation/oracle/solubility/sol_predictor_final.pt')
    parser.add_argument('--property', type=str, default='sol', choices=['sol', 'therm'],
                        help='Property type for oracle evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"L{args.glp_layer} Steering + GLP (u={args.u}, steps={args.num_timesteps})")
    print("=" * 60)

    # Load reference data
    ref_df = pd.read_csv(args.ref_data)
    ref_seqs = ref_df['sequence'].tolist()
    print(f"Reference: {len(ref_seqs)} sequences from {args.ref_data}")

    # Load ESM2-650M
    model, alphabet = load_esm2_model("650M", device=args.device)
    model.steering_forward_glp = types.MethodType(steering_forward_with_glp, model)

    # Load steering vectors
    pos_sv, neg_sv = torch.load(args.sv_path, weights_only=False)
    steering_vectors = (pos_sv - neg_sv).to(args.device)
    print(f"Steering vectors: {steering_vectors.shape}")

    # Load GLP
    glp_model = load_glp(args.glp_path, args.device)
    glp_project_fn = build_glp_projection_fn(glp_model, u=args.u, num_timesteps=args.num_timesteps)

    # Load predictor
    predictor = load_predictor(args.predictor_path, device=args.device)

    # Generate
    gen_seqs = generate_with_glp(
        ref_seqs, model, alphabet, steering_vectors,
        glp_project_fn, args.glp_layer, args.device, args.n_gen,
    )

    # Save
    csv_path = os.path.join(args.output_dir, f"L{args.glp_layer}_glp_u{args.u}.csv")
    pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)
    print(f"Generated {len(gen_seqs)} sequences -> {csv_path}")

    # Oracle eval
    features = extract_features(gen_seqs, model, alphabet, args.device)
    with torch.no_grad():
        scores = predictor(features.to(args.device)).cpu()

    if args.property == 'sol':
        probs = torch.sigmoid(scores).numpy()
        labels = (probs >= 0.5).astype(int)
        mean_val = float(probs.mean())
        ratio = float(labels.mean())
        print(f"Sol: mean_prob={mean_val:.4f}, ratio={ratio*100:.1f}%")
        scored_df = pd.DataFrame({
            'sequence': gen_seqs,
            'pred_prob': probs.tolist(),
            'pred_label': labels.tolist(),
        })
    else:
        tm_scores = scores.numpy()
        mean_val = float(tm_scores.mean())
        ratio = None
        print(f"Therm: mean_Tm={mean_val:.2f}°C, std={tm_scores.std():.2f}°C")
        scored_df = pd.DataFrame({
            'sequence': gen_seqs,
            'pred_tm': tm_scores.tolist(),
        })

    scored_path = csv_path.replace('.csv', '_scored.csv')
    scored_df.to_csv(scored_path, index=False)

    # Summary
    summary = {
        'method': f'L{args.glp_layer}+GLP(u={args.u})',
        'property': args.property,
        'glp_path': args.glp_path,
        'u': args.u,
        'num_timesteps': args.num_timesteps,
        'n_gen': args.n_gen,
        'mean_val': mean_val,
        'ratio': ratio,
        'csv_path': csv_path,
    }
    json_path = os.path.join(args.output_dir, f"L{args.glp_layer}_glp_u{args.u}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {json_path}")
    print("=" * 60)
