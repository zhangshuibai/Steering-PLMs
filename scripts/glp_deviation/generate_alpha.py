"""
Unified generation script for alpha scaling + GLP variants on sol/therm.

Settings:
  - "L17_a{a}"              : L17 steering with alpha, no GLP
  - "L17_a{a}_GLP_u{u}"     : L17 steering + GLP projection at L17
  - "allL_a{a}"             : All-layer steering, no GLP
  - "allL_a{a}_L17GLP_u{u}" : All-layer steering + GLP projection at L17 only

All use: 10 rounds mask-predict (same as sol/therm default).

RNG control (to make GLP vs non-GLP comparable):
  - Mask positions are pre-computed with a dedicated NumPy RNG seeded by
    (seed * 10000 + seq_idx).
    Same seq_idx produces identical mask positions across all settings.
  - Token sampling uses a dedicated torch.Generator seeded by
    (seed * 10000 + seq_idx * 2 + 1).
  - GLP noise uses its own torch.Generator seeded by
    (seed * 10000 + seq_idx * 2 + 2).
  - These three sources are fully independent, so adding GLP does NOT shift
    the mask / sampling RNG stream (the multiplier 10000 also prevents seeds
    from colliding across different seq_idx values).

BOS/EOS exclusion:
  - GLP projection only operates on interior residues [1:-1], matching the GLP
    training data which excluded BOS/EOS tokens. BOS/EOS are passed through.
"""

import argparse
import math
import os
import re
import sys
import types
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'generative_latent_prior'))

from utils.esm2_utils import load_esm2_model, decode
from module.steerable_esm2 import steering_forward
from scripts.glp.steering_with_glp import (
    load_glp, steering_forward_with_glp,
)
from generative_latent_prior.glp import flow_matching
from scripts.glp_deviation.steering_variants import steering_forward_all_layer_with_glp


# ==================== Setting parser (strict) ====================

SETTING_RE = re.compile(
    r'^(L17|L17post|allL)_a(\d+(?:\.\d+)?)'  # scope + alpha
    r'(?:_(GLP|L17GLP)(mask)?_u(\d+(?:\.\d+)?))?$'  # optional GLP (+mask suffix) + u
)


def parse_setting(setting):
    m = SETTING_RE.match(setting)
    if not m:
        raise ValueError(
            f'Invalid setting: {setting!r}. '
            'Expected: L17_a{a}[_GLP[mask]_u{u}] | '
            'L17post_a{a}[_GLP[mask]_u{u}] | '
            'allL_a{a}[_L17GLP[mask]_u{u}]'
        )
    scope_tag, alpha_str, glp_tag, mask_suffix, u_str = m.groups()
    # scope_map: 'L17' → 'L17' (single layer), 'L17post' → 'L17post' (layers 16..32),
    #            'allL' → 'all' (every layer)
    scope_map = {'L17': 'L17', 'L17post': 'L17post', 'allL': 'all'}
    scope = scope_map[scope_tag]
    alpha = float(alpha_str)
    use_glp = glp_tag is not None
    glp_mask_only = mask_suffix == 'mask'
    u = float(u_str) if u_str is not None else None

    # Validate GLP tag matches scope. L17 and L17post both use _GLP[mask]_u (single-layer GLP at L17).
    # Only allL uses _L17GLP[mask]_u to emphasize that GLP is at L17 but steering is everywhere.
    if use_glp:
        if scope in ('L17', 'L17post') and glp_tag != 'GLP':
            raise ValueError(f'{scope_tag} scope must use _GLP[mask]_u tag, got _{glp_tag}_')
        if scope == 'all' and glp_tag != 'L17GLP':
            raise ValueError(f'all-layer scope must use _L17GLP[mask]_u tag, got _{glp_tag}_')
    return scope, alpha, use_glp, u, glp_mask_only


# ==================== Steering vector construction ====================

def build_steering_vectors(diff, scope, alpha):
    if scope == 'L17':
        sv = torch.zeros_like(diff)
        sv[16] = diff[16] * alpha  # self.layers[16] = L17 = where GLP was trained
        return sv
    elif scope == 'L17post':
        # Zero on layers 0..15 (L1..L16), steer at layers 16..32 (L17..L33).
        # Purpose: test whether post-L17 steering acts as a coherence corrector after GLP.
        sv = torch.zeros_like(diff)
        sv[16:] = diff[16:] * alpha
        return sv
    elif scope == 'all':
        return diff * alpha
    raise ValueError(scope)


# ==================== GLP projection that skips BOS/EOS ====================

def build_glp_projection_fn_interior(glp_model, u=0.5, num_timesteps=25, noise_generator=None):
    """Like build_glp_projection_fn but only projects interior residues [1:-1].

    Uses the provided torch.Generator for noise sampling (so RNG state is isolated
    from other randomness sources).
    """
    scheduler = glp_model.scheduler
    scheduler.set_timesteps(num_timesteps)

    def project_on_manifold(acts):
        """
        acts: (T, B, D) - ESM2 internal format (seq_len, batch, hidden_dim).
              T includes BOS (pos 0) and EOS (pos T-1).
        returns: (T, B, D) with BOS/EOS unchanged, interior residues projected.
        """
        T, B, D = acts.shape
        if T < 3:
            return acts  # nothing interior to project

        interior = acts[1:-1]  # (T-2, B, D)
        T_int = T - 2
        latents = interior.permute(1, 0, 2).reshape(B * T_int, 1, D)

        # Normalize
        latents = glp_model.normalizer.normalize(latents)

        # Add noise using dedicated generator (isolate RNG state)
        if noise_generator is None:
            noise = torch.randn_like(latents)
        else:
            noise = torch.randn(
                latents.shape, generator=noise_generator,
                dtype=latents.dtype, device=latents.device,
            )

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

        result = latents.reshape(B, T_int, D).permute(1, 0, 2).to(dtype=acts.dtype)
        # Put interior back into the full tensor, leaving BOS/EOS unchanged
        projected = acts.clone()
        projected[1:-1] = result
        return projected

    return project_on_manifold


# ==================== Top-p sampling with dedicated generator ====================

def sample_top_p_with_gen(probs, p, generator=None):
    """Same as utils.gen_utils.sample_top_p but accepts a torch.Generator."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token[:, 0]


# ==================== Mask position pre-computation ====================

def precompute_mask_positions(seq_len, mask_ratio, seed, n_rounds=None):
    """Pre-compute mask positions per round using a dedicated NumPy RNG.

    Returns: list of lists of 0-indexed positions (relative to protein sequence,
    not to tokens; caller adds +1 for BOS offset).
    """
    rng = np.random.RandomState(seed)
    rounds = n_rounds if n_rounds is not None else math.ceil(1.0 / mask_ratio)
    candidate = list(range(seq_len))
    positions_per_round = []
    for _ in range(rounds):
        mask_size = min(math.ceil(seq_len * mask_ratio), len(candidate))
        if mask_size == 0:
            positions_per_round.append([])
            continue
        idx = rng.choice(len(candidate), size=mask_size, replace=False)
        chosen = sorted(candidate[i] for i in idx)
        positions_per_round.append(chosen)
        candidate = [c for i, c in enumerate(candidate) if i not in set(idx.tolist())]
    return positions_per_round


def precompute_mask_positions_fixed(fixed_positions, n_rounds=1):
    """For datasets with fixed mask positions (TrpB, CreiLOV)."""
    return [list(fixed_positions) for _ in range(n_rounds)]


def precompute_mask_positions_nk(seq_len, n_positions, n_rounds, seed):
    """Protocol: pre-select n_positions positions per round, random, no overlap across rounds."""
    rng = np.random.RandomState(seed)
    candidate = list(range(seq_len))
    positions_per_round = []
    for _ in range(n_rounds):
        k = min(n_positions, len(candidate))
        if k == 0:
            positions_per_round.append([])
            continue
        idx = rng.choice(len(candidate), size=k, replace=False)
        chosen = sorted(candidate[i] for i in idx)
        positions_per_round.append(chosen)
        candidate = [c for i, c in enumerate(candidate) if i not in set(idx.tolist())]
    return positions_per_round


# ==================== Generation ====================

def generate_one(ref_seq, model, alphabet, device, sv, use_glp, glp_fn,
                 needs_multilayer_forward,
                 mask_positions_per_round, sample_generator,
                 glp_mask_only=False,
                 temperature=1.0, top_p=0.9):
    """needs_multilayer_forward: True when steering vectors are non-zero at layers
    beyond L17 (scope ∈ {'all', 'L17post'}). Selects the multilayer-steering GLP
    forward path. For pure L17 single-layer steering (zero everywhere else), use
    the single-layer path which only touches L17.

    glp_mask_only: if True, the GLP projection is only applied at the residue
    positions masked in the CURRENT round (the ones being re-predicted).
    Unmasked positions keep their ESM2-computed L17 intact, which preserves
    inter-position coherence for the 90% of residues that aren't being
    regenerated. Only meaningful when use_glp=True."""
    bc = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    _, _, tokens = bc([('protein', ref_seq)])
    tokens = tokens.to(device).clone()

    for round_positions in mask_positions_per_round:
        if not round_positions:
            continue
        # +1 for BOS offset
        token_positions = torch.tensor([p + 1 for p in round_positions], device=device)
        seq_token = tokens.clone()
        seq_token[0, token_positions] = mask_idx

        # For mask-only GLP, wrap glp_fn so it only projects the residues
        # currently being masked and re-predicted. Since GLP is per-position
        # independent (MLP denoiser, no inter-position attention), we build a
        # minimal [BOS, mask_pos_1, ..., mask_pos_k, EOS] tensor and only
        # GLP-project that — mathematically equivalent to running GLP on the
        # full sequence and discarding non-mask rows, but ~10x faster.
        round_glp_fn = glp_fn
        if use_glp and glp_mask_only and glp_fn is not None:
            _mask_tok_pos = token_positions  # (n_mask,) tensor on device
            def _mask_only_project(acts, _base_fn=glp_fn, _mp=_mask_tok_pos):
                # acts: (T, B, D). Build a minimal tensor with just BOS + mask
                # residues + EOS, project that, then write back into result.
                bos = acts[:1]                               # (1, B, D)
                eos = acts[-1:]                              # (1, B, D)
                masked_slice = acts[_mp]                     # (n_mask, B, D)
                mini = torch.cat([bos, masked_slice, eos], dim=0)  # (n_mask+2, B, D)
                mini_projected = _base_fn(mini)              # BOS/EOS unchanged,
                                                             #   interior projected
                projected_masked = mini_projected[1:-1]      # (n_mask, B, D)
                result = acts.clone()
                result[_mp] = projected_masked
                return result
            round_glp_fn = _mask_only_project

        with torch.no_grad():
            if use_glp and needs_multilayer_forward:
                # Steer at every layer where sv is non-zero, GLP at L17
                out = model.steering_forward_all_l17glp(
                    tokens=seq_token, steering_vectors=sv,
                    glp_project_fn=round_glp_fn, glp_layer=16,
                )
            elif use_glp:
                # Steer + GLP only at L17 (pure single-layer)
                out = model.steering_forward_glp(
                    tokens=seq_token, steering_vectors=sv,
                    glp_project_fn=round_glp_fn, glp_layer=16,
                )
            elif sv is not None:
                # No GLP: steering_forward iterates all layers and applies sv
                # wherever it's non-zero (zeros are no-ops after rescale)
                out = model.steering_forward(tokens=seq_token, steering_vectors=sv)
            else:
                out = model(tokens=seq_token)

        logits = out['logits'][0, :, 4:24]
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            pred = sample_top_p_with_gen(probs, top_p, generator=sample_generator)
        else:
            pred = torch.argmax(logits, dim=-1)
        pred = pred + 4
        pred[0] = tokens[0, 0]
        pred[-1] = tokens[0, -1]
        tokens[0, token_positions] = pred[token_positions]

    return decode(alphabet, tokens[:, 1:-1], onehot=False)[0]


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, required=True)
    parser.add_argument('--ref_data', type=str, required=True)
    parser.add_argument('--sv_path', type=str, required=True)
    parser.add_argument('--glp_path', type=str,
                        default='generative_latent_prior/runs/glp-esm2-650m-layer17-d6')
    parser.add_argument('--num_timesteps', type=int, default=25)
    parser.add_argument('--n_gen', type=int, default=100)
    parser.add_argument('--mask_ratio', type=float, default=0.1)
    parser.add_argument('--protocol', type=str, default='whole_seq',
                        choices=['whole_seq', 'fixed_sites', 'n_per_round'],
                        help='whole_seq: 10 rounds × mask_ratio (sol/therm); '
                             'fixed_sites: 1 round with fixed positions (TrpB/CreiLOV); '
                             'n_per_round: n_rounds × n_positions (GFP)')
    parser.add_argument('--n_rounds', type=int, default=None)
    parser.add_argument('--n_positions', type=int, default=None,
                        help='For n_per_round protocol')
    parser.add_argument('--fixed_positions', type=int, nargs='*', default=None,
                        help='For fixed_sites protocol (0-indexed)')
    parser.add_argument('--fixed_positions_from', type=str, default=None,
                        help='CSV with mutated_positions column (for CreiLOV union)')
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    scope, alpha, use_glp, u, glp_mask_only = parse_setting(args.setting)
    print(f'Setting: {args.setting} → scope={scope}, alpha={alpha}, GLP={use_glp}, u={u}, '
          f'mask_only={glp_mask_only}')

    model, alphabet = load_esm2_model('650M', device=args.device)
    model.steering_forward = types.MethodType(steering_forward, model)
    model.steering_forward_glp = types.MethodType(steering_forward_with_glp, model)
    model.steering_forward_all_l17glp = types.MethodType(steering_forward_all_layer_with_glp, model)

    pos_sv, neg_sv = torch.load(args.sv_path, weights_only=False)
    diff = (pos_sv - neg_sv).to(args.device)
    sv = build_steering_vectors(diff, scope, alpha)

    glp_model = None
    if use_glp:
        glp_model = load_glp(args.glp_path, args.device)

    ref_seqs = pd.read_csv(args.ref_data)['sequence'].tolist()

    # Ensure output dir exists (handle bare filename edge case)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Resolve fixed positions if loading from a CSV with mutated_positions column
    fixed_positions_resolved = args.fixed_positions
    if args.fixed_positions_from:
        df_fp = pd.read_csv(args.fixed_positions_from)
        pos_set = set()
        for mp in df_fp['mutated_positions'].dropna():
            for p in str(mp).split(','):
                p = p.strip()
                if p and p != 'WT':
                    pos_set.add(int(p) - 1)  # 1-indexed -> 0-indexed
        fixed_positions_resolved = sorted(pos_set)
        print(f'Loaded {len(fixed_positions_resolved)} fixed positions from {args.fixed_positions_from}')

    gen_seqs = []
    for i in tqdm(range(args.n_gen), desc=args.setting):
        ref_seq = ref_seqs[i % len(ref_seqs)]
        seq_len = len(ref_seq)

        # --- Pre-compute mask positions (identical across settings for same seq_idx) ---
        if args.protocol == 'whole_seq':
            mask_positions_per_round = precompute_mask_positions(
                seq_len, args.mask_ratio, seed=args.seed * 10000 + i,
                n_rounds=args.n_rounds,  # None -> default to ceil(1/mask_ratio)
            )
        elif args.protocol == 'fixed_sites':
            mask_positions_per_round = precompute_mask_positions_fixed(
                fixed_positions_resolved,
                n_rounds=args.n_rounds if args.n_rounds else 1,
            )
        elif args.protocol == 'n_per_round':
            mask_positions_per_round = precompute_mask_positions_nk(
                seq_len,
                n_positions=args.n_positions,
                n_rounds=args.n_rounds,
                seed=args.seed * 10000 + i,
            )
        else:
            raise ValueError(f'Unknown protocol {args.protocol}')

        # --- Separate torch.Generators for sampling and GLP noise ---
        sample_gen = torch.Generator(device=args.device)
        sample_gen.manual_seed(args.seed * 10000 + i * 2 + 1)

        glp_fn = None
        if use_glp:
            glp_noise_gen = torch.Generator(device=args.device)
            glp_noise_gen.manual_seed(args.seed * 10000 + i * 2 + 2)
            glp_fn = build_glp_projection_fn_interior(
                glp_model, u=u, num_timesteps=args.num_timesteps,
                noise_generator=glp_noise_gen,
            )

        new_seq = generate_one(
            ref_seq, model, alphabet, args.device, sv,
            use_glp=use_glp, glp_fn=glp_fn,
            needs_multilayer_forward=(scope in ('all', 'L17post')),
            glp_mask_only=glp_mask_only,
            mask_positions_per_round=mask_positions_per_round,
            sample_generator=sample_gen,
        )
        gen_seqs.append(new_seq)

    pd.DataFrame({'sequence': gen_seqs}).to_csv(args.output_csv, index=False)
    print(f'Saved {len(gen_seqs)} sequences -> {args.output_csv}')


if __name__ == '__main__':
    main()
