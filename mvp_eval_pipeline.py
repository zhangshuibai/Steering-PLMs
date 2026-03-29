"""
MVP ESM2 steering + editing evaluation pipeline.

This script is designed for controlled-family editing experiments where we:
1. build steering vectors from the top/bottom quantiles of a family pool,
2. edit test sequences with either random masking or ASPO-style masking,
3. compare no steering / naive steering / alignment steering,
4. evaluate primary fitness, auxiliary fitness, edit size, diversity, and pPPL.

Notes
-----
- "alignment steering" is implemented as a projection hook. For MVP the default
  projector is identity, so the mode is runnable now and can later be replaced
  by a diffusion prior without changing the pipeline shape.
- "ASPO-style" site selection here is an ESM2 approximation: we compute a token
  relatedness score from cosine similarity between token representations and
  steering vectors at the selected steering layers, then iteratively edit the
  lowest-score sites.
"""

import argparse
import json
import math
import os
import types
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from evaluate_ppl import compute_pseudo_perplexity, compute_pseudo_perplexity_multi_gpu
from module.steerable_esm2 import steering_forward
from utils.esm2_utils import (
    extract_esm2_features,
    get_esm2_layer_and_feature_dim,
    load_esm2_model,
)


AMINO_ACID_SLICE = slice(4, 24)


class PropertyPredictor(torch.nn.Module):
    def __init__(self, embed_dim: int = 1280):
        super().__init__()
        self.dense = torch.nn.Linear(embed_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x.squeeze(-1)


@dataclass
class PredictorBundle:
    name: str
    predictor_type: str
    model: PropertyPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESM2 MVP editing + evaluation pipeline")
    parser.add_argument("--property", choices=["sol", "therm"], required=True,
                        help="Primary steering objective.")
    parser.add_argument("--input_csv", required=True,
                        help="Controlled-family test set (for example N_sol_test / N_therm_test).")
    parser.add_argument("--family_csv", required=True,
                        help="Pool used to construct positive/negative steering sets.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save generations and summaries.")
    parser.add_argument("--input_score_col", default=None,
                        help="Score column in input_csv. Defaults to 'score' when present.")
    parser.add_argument("--family_score_col", default="score",
                        help="Score column used in family_csv for steering-vector construction.")
    parser.add_argument("--sequence_col", default="sequence")
    parser.add_argument("--model", default="650M", choices=["150M", "650M", "3B"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=None,
                        help="Maximum number of test sequences to edit.")
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument(
        "--mask_strategy",
        choices=["random", "aspo", "targeted"],
        default="random",
        help="Editing task. 'targeted' is kept as a backward-compatible alias for 'aspo'.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Number of edit rounds. Defaults to ceil(1 / mask_ratio).",
    )
    parser.add_argument("--modes", nargs="+",
                        default=["no_steering", "naive_steering", "alignment_steering"],
                        choices=["no_steering", "naive_steering", "alignment_steering"])
    parser.add_argument("--top_quantile", type=float, default=0.2)
    parser.add_argument("--bottom_quantile", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[17, 18],
                        help="Layers used for ASPO token-relatedness scoring and steering.")
    parser.add_argument(
        "--avoid_original_token",
        action="store_true",
        default=False,
        help="Disallow sampling the current residue at masked sites. Default is to allow it.",
    )
    parser.add_argument("--allow_original_token", dest="avoid_original_token", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sol_predictor_path", default="saved_predictors/sol_predictor_final.pt")
    parser.add_argument("--therm_predictor_path", default="saved_predictors/therm_predictor_nocdhit.pt")
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--ppl_model", default="3B", choices=["150M", "650M", "3B"])
    parser.add_argument("--ppl_gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--ppl_batch_masks", type=int, default=32)
    parser.add_argument("--diversity_pairs", type=int, default=2048,
                        help="Maximum number of pair samples for diversity estimation.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mask_strategy == "targeted":
        args.mask_strategy = "aspo"

    if args.num_rounds is not None and args.num_rounds <= 0:
        raise ValueError("--num_rounds must be positive when provided.")

    if args.mask_strategy == "aspo" and "no_steering" in args.modes:
        raise ValueError("ASPO only supports steering modes. Remove 'no_steering' from --modes.")

    return args


def resolve_score_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    if "score" in df.columns:
        return "score"
    return None


def build_steering_vectors(
    family_df: pd.DataFrame,
    sequence_col: str,
    score_col: str,
    model,
    alphabet,
    model_name: str,
    device: str,
    top_quantile: float,
    bottom_quantile: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if score_col not in family_df.columns:
        raise ValueError(f"family_csv is missing score column '{score_col}'")

    n_layers, _ = get_esm2_layer_and_feature_dim(model_name)
    scores = family_df[score_col].to_numpy()
    pos_threshold = np.quantile(scores, 1.0 - top_quantile)
    neg_threshold = np.quantile(scores, bottom_quantile)

    pos_df = family_df[family_df[score_col] >= pos_threshold]
    neg_df = family_df[family_df[score_col] <= neg_threshold]
    pos_seqs = pos_df[sequence_col].tolist()
    neg_seqs = neg_df[sequence_col].tolist()

    if not pos_seqs or not neg_seqs:
        raise ValueError("Positive or negative steering set is empty after quantile split.")

    pos_repr = extract_esm2_features(pos_seqs, model, alphabet, n_layers, batch_size=1, device=device)
    neg_repr = extract_esm2_features(neg_seqs, model, alphabet, n_layers, batch_size=1, device=device)
    steering_vectors = pos_repr.mean(dim=1) - neg_repr.mean(dim=1)

    meta = {
        "pos_threshold": float(pos_threshold),
        "neg_threshold": float(neg_threshold),
        "n_pos": int(len(pos_seqs)),
        "n_neg": int(len(neg_seqs)),
    }
    return steering_vectors.to(device), meta


def select_single_layer_vectors(
    steering_vectors: torch.Tensor, target_layers: Sequence[int]
) -> torch.Tensor:
    selected = torch.zeros_like(steering_vectors)
    for layer in target_layers:
        selected[layer] = steering_vectors[layer]
    return selected


def build_alignment_projector(projector_name: str = "identity"):
    if projector_name == "identity":
        return lambda x, _layer_idx: x
    raise ValueError(f"Unknown projector '{projector_name}'")


def forward_with_mode(
    model,
    tokens: torch.Tensor,
    steering_vectors: Optional[torch.Tensor],
    mode: str,
    alignment_projector=None,
):
    if mode == "no_steering" or steering_vectors is None:
        return model(tokens=tokens)

    if mode == "naive_steering":
        return model.steering_forward(tokens=tokens, steering_vectors=steering_vectors)

    if mode == "alignment_steering":
        return steering_forward_with_alignment(
            model,
            tokens=tokens,
            steering_vectors=steering_vectors,
            alignment_projector=alignment_projector,
        )

    raise ValueError(f"Unknown mode: {mode}")


def steering_forward_with_alignment(
    model,
    tokens: torch.Tensor,
    steering_vectors: torch.Tensor,
    alignment_projector,
):
    assert tokens.ndim == 2
    padding_mask = tokens.eq(model.padding_idx)
    x = model.embed_scale * model.embed_tokens(tokens)
    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
    x = x.transpose(0, 1)

    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(model.layers):
        x, _ = layer(
            x,
            self_attn_padding_mask=padding_mask,
            need_head_weights=False,
        )
        add_x = steering_vectors[layer_idx]
        new_x = x + add_x
        new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
        x = new_x * (x_norm / new_x_norm.clamp_min(1e-8))
        x = alignment_projector(x, layer_idx)

    x = model.emb_layer_norm_after(x)
    x = x.transpose(0, 1)
    logits = model.lm_head(x)
    return {"logits": logits, "representations": {}}


def sample_predictions(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    aa_logits = logits[..., AMINO_ACID_SLICE]
    if temperature > 0.0:
        probs = torch.softmax(aa_logits / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort = probs_sort.masked_fill(mask, 0.0)
        probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sampled = torch.multinomial(probs_sort, num_samples=1)
        pred = probs_idx.gather(-1, sampled).squeeze(-1)
    else:
        pred = aa_logits.argmax(dim=-1)
    return pred + AMINO_ACID_SLICE.start


def decode_tokens(tokens: torch.Tensor, alphabet) -> str:
    seq = tokens[0, 1:-1].detach().cpu()
    return "".join(alphabet.get_tok(tok.item()) for tok in seq)


def predict_masked_positions(
    model,
    seq_token: torch.Tensor,
    original_tokens: torch.Tensor,
    mask_positions: torch.Tensor,
    mode: str,
    steering_vectors: Optional[torch.Tensor],
    temperature: float,
    top_p: float,
    avoid_original_token: bool,
    alignment_projector,
) -> torch.Tensor:
    outputs = forward_with_mode(
        model=model,
        tokens=seq_token,
        steering_vectors=steering_vectors,
        mode=mode,
        alignment_projector=alignment_projector,
    )
    logits = outputs["logits"][0]
    if avoid_original_token:
        original = original_tokens[0].clone()
        mask_one_hot = F.one_hot(original, logits.size(-1)).float()
        logits = logits + mask_one_hot * -1e8
    pred_tokens = sample_predictions(logits, temperature, top_p)
    return pred_tokens[mask_positions]


def get_token_relatedness_scores(
    tokens: torch.Tensor,
    model,
    target_layers: Sequence[int],
    steering_vectors: torch.Tensor,
    center_features: bool = False,
) -> np.ndarray:
    repr_layers = sorted(set(layer + 1 for layer in target_layers))
    with torch.no_grad():
        outputs = model(tokens=tokens, repr_layers=repr_layers)

    per_layer_scores = []
    seq_len = tokens.size(1) - 2
    for layer in target_layers:
        reps = outputs["representations"][layer + 1][0, 1:seq_len + 1]
        if center_features:
            reps = reps - reps.mean(dim=0, keepdim=True)
        sv = steering_vectors[layer].unsqueeze(0).expand_as(reps)
        scores = F.cosine_similarity(reps, sv, dim=-1)
        per_layer_scores.append(scores)

    stacked = torch.stack(per_layer_scores, dim=0)
    return stacked.mean(dim=0).detach().cpu().numpy()


def choose_random_mask_positions(
    candidate_sites: Sequence[int],
    mask_count: int,
    rng: np.random.Generator,
) -> List[int]:
    sampled = rng.choice(np.array(candidate_sites), size=mask_count, replace=False)
    return sampled.tolist()


def choose_aspo_mask_positions(
    tokens: torch.Tensor,
    model,
    candidate_sites: Sequence[int],
    mask_count: int,
    steering_vectors: torch.Tensor,
    target_layers: Sequence[int],
    property_name: str,
) -> List[int]:
    if mask_count <= 0:
        return []

    center_features = property_name == "sol"
    relatedness_scores = get_token_relatedness_scores(
        tokens=tokens,
        model=model,
        target_layers=target_layers,
        steering_vectors=steering_vectors,
        center_features=center_features,
    )
    ranked = sorted(candidate_sites, key=lambda idx: relatedness_scores[idx])
    return ranked[:mask_count]


def compute_rounds(mask_ratio: float, num_rounds: Optional[int]) -> int:
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("--mask_ratio must be in (0, 1].")
    if num_rounds is not None:
        return num_rounds
    return math.ceil(1.0 / mask_ratio)


def edit_sequence_random(
    seq: str,
    model,
    alphabet,
    mode: str,
    steering_vectors: Optional[torch.Tensor],
    mask_ratio: float,
    num_rounds: Optional[int],
    target_layers: Sequence[int],
    temperature: float,
    top_p: float,
    avoid_original_token: bool,
    rng: np.random.Generator,
    alignment_projector,
) -> Dict[str, object]:
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    _, _, tokens = batch_converter([("protein", seq)])
    tokens = tokens.to(next(model.parameters()).device)
    edited_tokens = tokens.clone()
    length = edited_tokens.size(1) - 2
    candidate_sites = list(range(length))
    rounds = compute_rounds(mask_ratio, num_rounds)
    edited_sites: List[int] = []

    for _round_idx in range(rounds):
        if not candidate_sites:
            break
        mask_count = min(math.ceil(length * mask_ratio), len(candidate_sites))
        chosen_sites = choose_random_mask_positions(candidate_sites, mask_count, rng)
        if not chosen_sites:
            break

        mask_positions = torch.tensor(chosen_sites, device=edited_tokens.device) + 1
        masked_tokens = edited_tokens.clone()
        masked_tokens[0, mask_positions] = mask_idx
        new_values = predict_masked_positions(
            model=model,
            seq_token=masked_tokens,
            original_tokens=edited_tokens,
            mask_positions=mask_positions,
            mode=mode,
            steering_vectors=steering_vectors,
            temperature=temperature,
            top_p=top_p,
            avoid_original_token=avoid_original_token,
            alignment_projector=alignment_projector,
        )
        edited_tokens[0, mask_positions] = new_values
        candidate_sites = [idx for idx in candidate_sites if idx not in chosen_sites]
        edited_sites.extend(chosen_sites)

    edited_seq = decode_tokens(edited_tokens, alphabet)
    return {
        "edited_sequence": edited_seq,
        "edited_sites": sorted(set(edited_sites)),
        "rounds": rounds,
    }


def edit_sequence_aspo(
    seq: str,
    model,
    alphabet,
    mode: str,
    steering_vectors: torch.Tensor,
    mask_ratio: float,
    num_rounds: Optional[int],
    target_layers: Sequence[int],
    temperature: float,
    top_p: float,
    avoid_original_token: bool,
    property_name: str,
    alignment_projector,
) -> Dict[str, object]:
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    _, _, tokens = batch_converter([("protein", seq)])
    tokens = tokens.to(next(model.parameters()).device)
    edited_tokens = tokens.clone()
    length = edited_tokens.size(1) - 2
    candidate_sites = list(range(length))
    rounds = compute_rounds(mask_ratio, num_rounds)
    edited_sites: List[int] = []

    for _round_idx in range(rounds):
        if not candidate_sites:
            break
        mask_count = min(math.ceil(length * mask_ratio), len(candidate_sites))
        chosen_sites = choose_aspo_mask_positions(
            tokens=edited_tokens,
            model=model,
            candidate_sites=candidate_sites,
            mask_count=mask_count,
            steering_vectors=steering_vectors,
            target_layers=target_layers,
            property_name=property_name,
        )
        if not chosen_sites:
            break

        mask_positions = torch.tensor(chosen_sites, device=edited_tokens.device) + 1
        masked_tokens = edited_tokens.clone()
        masked_tokens[0, mask_positions] = mask_idx
        new_values = predict_masked_positions(
            model=model,
            seq_token=masked_tokens,
            original_tokens=edited_tokens,
            mask_positions=mask_positions,
            mode=mode,
            steering_vectors=steering_vectors,
            temperature=temperature,
            top_p=top_p,
            avoid_original_token=avoid_original_token,
            alignment_projector=alignment_projector,
        )
        edited_tokens[0, mask_positions] = new_values
        candidate_sites = [idx for idx in candidate_sites if idx not in chosen_sites]
        edited_sites.extend(chosen_sites)

    edited_seq = decode_tokens(edited_tokens, alphabet)
    return {
        "edited_sequence": edited_seq,
        "edited_sites": sorted(set(edited_sites)),
        "rounds": rounds,
    }


def edit_sequence(
    seq: str,
    model,
    alphabet,
    mode: str,
    mask_strategy: str,
    steering_vectors: Optional[torch.Tensor],
    mask_ratio: float,
    num_rounds: Optional[int],
    target_layers: Sequence[int],
    temperature: float,
    top_p: float,
    avoid_original_token: bool,
    rng: np.random.Generator,
    property_name: str,
    alignment_projector,
) -> Dict[str, object]:
    if mask_strategy == "random":
        return edit_sequence_random(
            seq=seq,
            model=model,
            alphabet=alphabet,
            mode=mode,
            steering_vectors=steering_vectors,
            mask_ratio=mask_ratio,
            num_rounds=num_rounds,
            target_layers=target_layers,
            temperature=temperature,
            top_p=top_p,
            avoid_original_token=avoid_original_token,
            rng=rng,
            alignment_projector=alignment_projector,
        )

    if mask_strategy == "aspo":
        if steering_vectors is None:
            raise ValueError("ASPO editing requires steering vectors.")
        return edit_sequence_aspo(
            seq=seq,
            model=model,
            alphabet=alphabet,
            mode=mode,
            steering_vectors=steering_vectors,
            mask_ratio=mask_ratio,
            num_rounds=num_rounds,
            target_layers=target_layers,
            temperature=temperature,
            top_p=top_p,
            avoid_original_token=avoid_original_token,
            property_name=property_name,
            alignment_projector=alignment_projector,
        )

    raise ValueError(f"Unknown mask strategy: {mask_strategy}")


def load_predictor(path: str, device: str, name: str, predictor_type: str) -> PredictorBundle:
    predictor = PropertyPredictor(embed_dim=1280)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        predictor.load_state_dict(ckpt["model_state_dict"])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(device)
    predictor.eval()
    return PredictorBundle(name=name, predictor_type=predictor_type, model=predictor)


def extract_last_layer_features(
    seqs: Sequence[str],
    model,
    alphabet,
    device: str,
    batch_size: int = 8,
) -> torch.Tensor:
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    features = []
    for start in range(0, len(seqs), batch_size):
        batch = [("protein", seq[:1022]) for seq in seqs[start:start + batch_size]]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            outputs = model(batch_tokens, repr_layers=[n_layers])
        reps = outputs["representations"][n_layers]
        for idx, seq_len in enumerate(batch_lens):
            features.append(reps[idx, 1:seq_len - 1].mean(0).cpu())
    return torch.stack(features, dim=0)


def evaluate_predictors(
    seqs: Sequence[str],
    model,
    alphabet,
    device: str,
    predictors: Sequence[PredictorBundle],
) -> Dict[str, np.ndarray]:
    features = extract_last_layer_features(seqs, model, alphabet, device)
    outputs: Dict[str, np.ndarray] = {}
    for bundle in predictors:
        with torch.no_grad():
            preds = bundle.model(features.to(device)).cpu().numpy()
        if bundle.predictor_type == "sol":
            preds = 1.0 / (1.0 + np.exp(-preds))
        outputs[bundle.name] = preds
    return outputs


def hamming_distance(seq_a: str, seq_b: str) -> int:
    if len(seq_a) != len(seq_b):
        raise ValueError("Hamming distance requires equal-length sequences.")
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq_a, seq_b))


def summarize_diversity(seqs: Sequence[str], max_pairs: int, seed: int) -> Dict[str, float]:
    unique_fraction = len(set(seqs)) / max(len(seqs), 1)
    if len(seqs) < 2:
        return {"unique_fraction": unique_fraction, "pairwise_hamming_mean": 0.0}

    rng = np.random.default_rng(seed)
    pairs = []
    total_possible = len(seqs) * (len(seqs) - 1) // 2
    if total_possible <= max_pairs:
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                pairs.append((i, j))
    else:
        seen = set()
        while len(pairs) < max_pairs:
            i = int(rng.integers(0, len(seqs)))
            j = int(rng.integers(0, len(seqs)))
            if i >= j:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)

    distances = []
    for i, j in pairs:
        if len(seqs[i]) != len(seqs[j]):
            continue
        distances.append(hamming_distance(seqs[i], seqs[j]) / max(len(seqs[i]), 1))

    pairwise_mean = float(np.mean(distances)) if distances else 0.0
    return {
        "unique_fraction": float(unique_fraction),
        "pairwise_hamming_mean": pairwise_mean,
    }


def summarize_primary_metrics(
    results_df: pd.DataFrame,
    primary_col: str,
) -> Dict[str, float]:
    deltas = results_df[f"delta_{primary_col}"].to_numpy()
    hamming = results_df["hamming_distance"].to_numpy()
    top_k_values = {}
    for k in [1, 5, 10]:
        actual_k = min(k, len(results_df))
        top_k_values[f"top_{k}_{primary_col}_mean"] = float(
            results_df.nlargest(actual_k, primary_col)[primary_col].mean()
        )

    gain_per_mut = np.divide(
        deltas,
        np.maximum(hamming, 1),
        out=np.zeros_like(deltas, dtype=float),
        where=np.maximum(hamming, 1) > 0,
    )

    return {
        f"{primary_col}_mean": float(results_df[primary_col].mean()),
        f"source_{primary_col}_mean": float(results_df[f"source_{primary_col}"].mean()),
        f"delta_{primary_col}_mean": float(deltas.mean()),
        f"delta_{primary_col}_median": float(np.median(deltas)),
        "success_rate": float((deltas > 0).mean()),
        "edit_success_rate": float(((deltas > 0) & (hamming > 0)).mean()),
        "hamming_mean": float(results_df["hamming_distance"].mean()),
        "percent_identity_mean": float(results_df["percent_identity_to_source"].mean()),
        "fitness_gain_per_mutation_mean": float(gain_per_mut.mean()),
        **top_k_values,
    }


def maybe_compute_ppl(
    seqs: Sequence[str],
    args: argparse.Namespace,
) -> Optional[Dict[str, float]]:
    if args.skip_ppl:
        return None
    if len(args.ppl_gpu_ids) > 1:
        ppls = compute_pseudo_perplexity_multi_gpu(
            list(seqs),
            args.ppl_model,
            args.ppl_gpu_ids,
            args.ppl_batch_masks,
        )
    else:
        device = f"cuda:{args.ppl_gpu_ids[0]}" if str(args.device).startswith("cuda") else args.device
        model, alphabet = load_esm2_model(args.ppl_model, device=device)
        ppls = compute_pseudo_perplexity(list(seqs), model, alphabet, device, args.ppl_batch_masks)
    arr = np.asarray(ppls, dtype=float)
    return {
        "ppl_mean": float(arr.mean()),
        "ppl_median": float(np.median(arr)),
        "ppl_std": float(arr.std()),
    }


def main() -> None:
    args = normalize_args(parse_args())
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    input_df = pd.read_csv(args.input_csv)
    family_df = pd.read_csv(args.family_csv)

    score_col = resolve_score_column(input_df, args.input_score_col)
    if args.n is not None:
        input_df = input_df.head(args.n).copy()

    if args.sequence_col not in input_df.columns:
        raise ValueError(f"input_csv is missing sequence column '{args.sequence_col}'")
    if args.sequence_col not in family_df.columns:
        raise ValueError(f"family_csv is missing sequence column '{args.sequence_col}'")

    model, alphabet = load_esm2_model(args.model, device=args.device)
    model.steering_forward = types.MethodType(steering_forward, model)

    steering_vectors_all, steering_meta = build_steering_vectors(
        family_df=family_df,
        sequence_col=args.sequence_col,
        score_col=args.family_score_col,
        model=model,
        alphabet=alphabet,
        model_name=args.model,
        device=args.device,
        top_quantile=args.top_quantile,
        bottom_quantile=args.bottom_quantile,
    )
    steering_vectors_all = steering_vectors_all * args.alpha
    steering_vectors_selected = select_single_layer_vectors(
        steering_vectors_all,
        args.target_layers,
    )

    predictors = [
        load_predictor(args.sol_predictor_path, args.device, "sol", "sol"),
        load_predictor(args.therm_predictor_path, args.device, "therm", "therm"),
    ]

    source_seqs = input_df[args.sequence_col].tolist()
    source_scores = evaluate_predictors(source_seqs, model, alphabet, args.device, predictors)

    alignment_projector = build_alignment_projector("identity")
    primary_metric = args.property
    run_summary = {
        "args": vars(args),
        "steering_meta": steering_meta,
        "results": {},
        "notes": {
            "alignment_steering": "MVP uses identity projector hook; replace with diffusion prior later.",
            "aspo_masking": "ASPO-style approximation using token-relatedness scores at selected ESM2 layers and editing the lowest-score sites.",
            "optional_metrics_not_in_mvp": ["pLDDT", "nearest_neighbor_identity_to_natural_db"],
        },
    }

    for mode in args.modes:
        mode_dir = os.path.join(args.output_dir, mode)
        ensure_dir(mode_dir)
        rng = np.random.default_rng(args.seed)
        edited_rows = []
        edited_seqs = []

        for row_idx, source_seq in enumerate(source_seqs):
            steering_vectors = None if mode == "no_steering" else steering_vectors_selected
            edit_result = edit_sequence(
                seq=source_seq,
                model=model,
                alphabet=alphabet,
                mode=mode,
                mask_strategy=args.mask_strategy,
                steering_vectors=steering_vectors,
                mask_ratio=args.mask_ratio,
                num_rounds=args.num_rounds,
                target_layers=args.target_layers,
                temperature=args.temperature,
                top_p=args.top_p,
                avoid_original_token=args.avoid_original_token,
                rng=rng,
                property_name=args.property,
                alignment_projector=alignment_projector,
            )
            edited_seq = edit_result["edited_sequence"]
            edited_seqs.append(edited_seq)
            hd = hamming_distance(source_seq, edited_seq)
            edited_rows.append({
                "source_index": row_idx,
                "source_sequence": source_seq,
                "sequence": edited_seq,
                "mask_strategy": args.mask_strategy,
                "mode": mode,
                "edited_sites": json.dumps(edit_result["edited_sites"]),
                "edited_site_count": len(edit_result["edited_sites"]),
                "hamming_distance": hd,
                "percent_identity_to_source": 1.0 - (hd / max(len(source_seq), 1)),
            })

        edited_scores = evaluate_predictors(edited_seqs, model, alphabet, args.device, predictors)
        result_df = pd.DataFrame(edited_rows)
        for predictor_name in ["sol", "therm"]:
            result_df[predictor_name] = edited_scores[predictor_name]
            result_df[f"source_{predictor_name}"] = source_scores[predictor_name]
            result_df[f"delta_{predictor_name}"] = (
                result_df[predictor_name] - result_df[f"source_{predictor_name}"]
            )

        if score_col is not None:
            result_df["input_score"] = input_df[score_col].to_numpy()

        ppl_summary = maybe_compute_ppl(edited_seqs, args)
        diversity_summary = summarize_diversity(edited_seqs, args.diversity_pairs, args.seed)
        primary_summary = summarize_primary_metrics(result_df, primary_metric)
        sol_threshold_cross = (
            (result_df["source_sol"] < 0.5) & (result_df["sol"] >= 0.5)
        ).mean()

        summary = {
            **primary_summary,
            **diversity_summary,
            "sol_threshold_cross_rate": float(sol_threshold_cross),
        }
        if ppl_summary is not None:
            summary.update(ppl_summary)

        csv_path = os.path.join(mode_dir, "per_sequence_results.csv")
        result_df.to_csv(csv_path, index=False)
        run_summary["results"][mode] = {
            "summary": summary,
            "csv_path": csv_path,
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(json.dumps(run_summary, indent=2))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
