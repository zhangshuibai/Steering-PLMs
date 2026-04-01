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
import hashlib
import json
import math
import os
import shlex
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path
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
    parser.add_argument("--mutation_entropy_bins", type=int, default=10)
    parser.add_argument("--natural_db_path", default=None,
                        help="Optional natural-sequence database for nearest-neighbor identity (csv/fasta/txt).")
    parser.add_argument("--natural_db_sequence_col", default="sequence")
    parser.add_argument("--natural_db_max_seqs", type=int, default=None)
    parser.add_argument("--compute_plddt", action="store_true",
                        help="Optionally compute and report pLDDT metrics.")
    parser.add_argument(
        "--plddt_backend",
        default="esmfold",
        choices=["esmfold", "colabfold"],
        help="Backend used for pLDDT. 'colabfold' shells out to an external colabfold_batch command.",
    )
    parser.add_argument(
        "--plddt_cache_csv",
        default=None,
        help="Optional shared CSV cache for pLDDT results across runs.",
    )
    parser.add_argument(
        "--colabfold_batch_cmd",
        default="colabfold_batch",
        help="Executable used when --plddt_backend colabfold.",
    )
    parser.add_argument(
        "--colabfold_msa_mode",
        default="mmseqs2_uniref_env",
        choices=[
            "mmseqs2_uniref_env",
            "mmseqs2_uniref_env_envpair",
            "mmseqs2_uniref",
            "single_sequence",
        ],
        help="MSA mode passed to colabfold_batch. Defaults to the official MMseqs2 server-backed mode.",
    )
    parser.add_argument(
        "--colabfold_model_type",
        default="alphafold2_ptm",
        choices=[
            "auto",
            "alphafold2",
            "alphafold2_ptm",
            "alphafold2_multimer_v1",
            "alphafold2_multimer_v2",
            "alphafold2_multimer_v3",
            "deepfold_v1",
        ],
        help="AlphaFold model type passed to colabfold_batch.",
    )
    parser.add_argument(
        "--colabfold_rank",
        default="plddt",
        choices=["auto", "plddt", "ptm", "iptm", "multimer"],
        help="Ranking metric passed to colabfold_batch.",
    )
    parser.add_argument(
        "--colabfold_num_models",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Number of AlphaFold models to evaluate in ColabFold.",
    )
    parser.add_argument(
        "--colabfold_num_seeds",
        type=int,
        default=1,
        help="Number of seeds passed to colabfold_batch.",
    )
    parser.add_argument(
        "--colabfold_data_dir",
        default=None,
        help="Optional AlphaFold weights directory passed via --data to colabfold_batch.",
    )
    parser.add_argument(
        "--colabfold_host_url",
        default=None,
        help="Optional MSA server URL passed to colabfold_batch.",
    )
    parser.add_argument(
        "--colabfold_templates",
        action="store_true",
        help="Request templates in the ColabFold/MMseqs2 workflow.",
    )
    parser.add_argument(
        "--colabfold_overwrite_existing_results",
        action="store_true",
        help="Pass --overwrite-existing-results to colabfold_batch.",
    )
    parser.add_argument(
        "--colabfold_batch_args",
        default="",
        help="Extra arguments passed to colabfold_batch before the input/output paths.",
    )
    parser.add_argument(
        "--colabfold_output_dir",
        default=None,
        help="Optional root directory for ColabFold intermediate/output files.",
    )
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

    if args.mutation_entropy_bins <= 1:
        raise ValueError("--mutation_entropy_bins must be > 1.")

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


def parse_fasta_sequences(path: str) -> List[str]:
    seqs: List[str] = []
    chunks: List[str] = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    seqs.append("".join(chunks).upper())
                    chunks = []
            else:
                chunks.append(line)
    if chunks:
        seqs.append("".join(chunks).upper())
    return seqs


def load_sequence_db(path: str, sequence_col: str, max_seqs: Optional[int]) -> List[str]:
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if sequence_col not in df.columns:
            raise ValueError(f"Natural DB file is missing sequence column '{sequence_col}'")
        seqs = df[sequence_col].dropna().astype(str).str.upper().tolist()
    elif suffix in {".fasta", ".fa", ".faa"}:
        seqs = parse_fasta_sequences(path)
    else:
        with open(path) as f:
            seqs = [line.strip().upper() for line in f if line.strip()]
    if max_seqs is not None:
        seqs = seqs[:max_seqs]
    return seqs


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
    round_sequences: List[str] = []

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
        round_sequences.append(decode_tokens(edited_tokens, alphabet))

    edited_seq = decode_tokens(edited_tokens, alphabet)
    return {
        "edited_sequence": edited_seq,
        "edited_sites": sorted(set(edited_sites)),
        "rounds": rounds,
        "round_sequences": round_sequences,
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
    round_sequences: List[str] = []

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
        round_sequences.append(decode_tokens(edited_tokens, alphabet))

    edited_seq = decode_tokens(edited_tokens, alphabet)
    return {
        "edited_sequence": edited_seq,
        "edited_sites": sorted(set(edited_sites)),
        "rounds": rounds,
        "round_sequences": round_sequences,
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
    return predict_from_features(features, predictors, device), features


def predict_from_features(
    features: torch.Tensor,
    predictors: Sequence[PredictorBundle],
    device: str,
) -> Dict[str, np.ndarray]:
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


def levenshtein_distance(seq_a: str, seq_b: str) -> int:
    if seq_a == seq_b:
        return 0
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, ch_a in enumerate(seq_a, start=1):
        curr = [i]
        for j, ch_b in enumerate(seq_b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def normalized_edit_similarity(seq_a: str, seq_b: str) -> float:
    max_len = max(len(seq_a), len(seq_b), 1)
    return 1.0 - (levenshtein_distance(seq_a, seq_b) / max_len)


def mutation_positions(seq_a: str, seq_b: str) -> List[int]:
    return [idx for idx, (ch_a, ch_b) in enumerate(zip(seq_a, seq_b)) if ch_a != ch_b]


def contiguous_runs(positions: Sequence[int]) -> List[Tuple[int, int]]:
    if not positions:
        return []
    runs = []
    start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev))
        start = pos
        prev = pos
    runs.append((start, prev))
    return runs


def position_entropy(positions: Sequence[int], seq_len: int, n_bins: int) -> float:
    if not positions or seq_len <= 0:
        return 0.0
    counts = np.zeros(n_bins, dtype=float)
    for pos in positions:
        rel = min(max(pos / max(seq_len - 1, 1), 0.0), 1.0)
        bin_idx = min(n_bins - 1, int(rel * n_bins))
        counts[bin_idx] += 1.0
    probs = counts[counts > 0] / counts.sum()
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy / np.log(n_bins))


def mutation_pattern_metrics(seq_a: str, seq_b: str, entropy_bins: int) -> Dict[str, float]:
    positions = mutation_positions(seq_a, seq_b)
    n_mut = len(positions)
    if n_mut == 0:
        return {
            "mutation_count": 0.0,
            "mutation_fraction": 0.0,
            "mutation_run_count": 0.0,
            "mutation_run_mean_length": 0.0,
            "mutation_run_max_length": 0.0,
            "mutation_span_fraction": 0.0,
            "mutation_position_entropy": 0.0,
        }
    runs = contiguous_runs(positions)
    run_lengths = [end - start + 1 for start, end in runs]
    span_fraction = (positions[-1] - positions[0] + 1) / max(len(seq_a), 1)
    return {
        "mutation_count": float(n_mut),
        "mutation_fraction": float(n_mut / max(len(seq_a), 1)),
        "mutation_run_count": float(len(runs)),
        "mutation_run_mean_length": float(np.mean(run_lengths)),
        "mutation_run_max_length": float(max(run_lengths)),
        "mutation_span_fraction": float(span_fraction),
        "mutation_position_entropy": position_entropy(positions, len(seq_a), entropy_bins),
    }


def compute_representation_drift(
    source_features: torch.Tensor,
    edited_features: torch.Tensor,
) -> Dict[str, np.ndarray]:
    deltas = edited_features - source_features
    l2 = torch.norm(deltas, p=2, dim=1).cpu().numpy()
    cos = F.cosine_similarity(edited_features, source_features, dim=1).cpu().numpy()
    return {
        "representation_l2": l2,
        "representation_cosine": cos,
    }


def summarize_representation_drift(
    drift: Dict[str, np.ndarray],
) -> Dict[str, float]:
    l2 = drift["representation_l2"]
    cos = drift["representation_cosine"]
    return {
        "representation_l2_mean": float(np.mean(l2)),
        "representation_l2_median": float(np.median(l2)),
        "representation_cosine_mean": float(np.mean(cos)),
        "representation_cosine_median": float(np.median(cos)),
    }


def summarize_mutation_patterns(
    results_df: pd.DataFrame,
    source_seqs: Sequence[str],
    edited_seqs: Sequence[str],
    entropy_bins: int,
) -> Dict[str, float]:
    per_seq = [mutation_pattern_metrics(src, edt, entropy_bins) for src, edt in zip(source_seqs, edited_seqs)]
    metrics = pd.DataFrame(per_seq)

    all_relative_positions = []
    for src, edt in zip(source_seqs, edited_seqs):
        for pos in mutation_positions(src, edt):
            all_relative_positions.append((pos, len(src)))
    if all_relative_positions:
        counts = np.zeros(entropy_bins, dtype=float)
        for pos, seq_len in all_relative_positions:
            rel = min(max(pos / max(seq_len - 1, 1), 0.0), 1.0)
            bin_idx = min(entropy_bins - 1, int(rel * entropy_bins))
            counts[bin_idx] += 1.0
        probs = counts[counts > 0] / counts.sum()
        global_entropy = float((-(probs * np.log(probs)).sum()) / np.log(entropy_bins))
    else:
        global_entropy = 0.0

    summary = {f"{col}_mean": float(metrics[col].mean()) for col in metrics.columns}
    summary["global_mutation_position_entropy"] = global_entropy
    return summary


def maybe_compute_round_metrics(
    round_sequences_by_source: Sequence[Sequence[str]],
    source_features: torch.Tensor,
    source_scores: Dict[str, np.ndarray],
    source_seqs: Sequence[str],
    model,
    alphabet,
    device: str,
    predictors: Sequence[PredictorBundle],
    entropy_bins: int,
) -> List[Dict[str, float]]:
    if not round_sequences_by_source:
        return []

    max_rounds = max(len(seq_rounds) for seq_rounds in round_sequences_by_source)
    round_metrics = []
    prev_round_seqs = list(source_seqs)
    prev_round_features = source_features
    for round_idx in range(max_rounds):
        round_seqs = []
        for seq_rounds in round_sequences_by_source:
            chosen_idx = min(round_idx, len(seq_rounds) - 1)
            round_seqs.append(seq_rounds[chosen_idx])
        round_scores, round_features = evaluate_predictors(round_seqs, model, alphabet, device, predictors)
        drift = compute_representation_drift(source_features, round_features)
        step_drift = compute_representation_drift(prev_round_features, round_features)
        round_hamming = np.asarray(
            [hamming_distance(src, cur) for src, cur in zip(source_seqs, round_seqs)],
            dtype=float,
        )
        step_hamming = np.asarray(
            [hamming_distance(prev, cur) for prev, cur in zip(prev_round_seqs, round_seqs)],
            dtype=float,
        )
        round_identity = np.asarray(
            [1.0 - (hd / max(len(src), 1)) for src, hd in zip(source_seqs, round_hamming)],
            dtype=float,
        )
        pattern_df = pd.DataFrame(
            [mutation_pattern_metrics(src, cur, entropy_bins) for src, cur in zip(source_seqs, round_seqs)]
        )
        row = {
            "round": int(round_idx + 1),
            "representation_l2_mean": float(np.mean(drift["representation_l2"])),
            "representation_cosine_mean": float(np.mean(drift["representation_cosine"])),
            "step_representation_l2_mean": float(np.mean(step_drift["representation_l2"])),
            "step_representation_cosine_mean": float(np.mean(step_drift["representation_cosine"])),
            "hamming_mean": float(round_hamming.mean()),
            "percent_identity_mean": float(round_identity.mean()),
            "step_hamming_mean": float(step_hamming.mean()),
        }
        for col in pattern_df.columns:
            row[f"{col}_mean"] = float(pattern_df[col].mean())
        for predictor_name, scores in round_scores.items():
            row[f"{predictor_name}_mean"] = float(np.mean(scores))
            row[f"delta_{predictor_name}_mean"] = float(np.mean(scores - source_scores[predictor_name]))
        round_metrics.append(row)
        prev_round_seqs = round_seqs
        prev_round_features = round_features
    return round_metrics


def maybe_compute_nearest_natural_identity(
    seqs: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    if not args.natural_db_path:
        return None, {"nearest_natural_status": "not_requested"}

    db_seqs = load_sequence_db(args.natural_db_path, args.natural_db_sequence_col, args.natural_db_max_seqs)
    if not db_seqs:
        return None, {"nearest_natural_status": "empty_db"}

    identities = []
    for seq in seqs:
        best = max(normalized_edit_similarity(seq, db_seq) for db_seq in db_seqs)
        identities.append(best)

    arr = np.asarray(identities, dtype=float)
    return arr, {
        "nearest_natural_status": "ok",
        "nearest_natural_db_size": int(len(db_seqs)),
        "nearest_natural_identity_mean": float(arr.mean()),
        "nearest_natural_identity_median": float(np.median(arr)),
        "nearest_natural_identity_max": float(arr.max()),
    }


def maybe_compute_plddt(
    seqs: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    if not args.compute_plddt:
        return None, {"plddt_status": "not_requested"}

    if args.plddt_backend == "colabfold":
        return maybe_compute_plddt_colabfold(seqs, args)

    return maybe_compute_plddt_esmfold(seqs, args)


def maybe_compute_plddt_esmfold(
    seqs: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    try:
        import esm
        model = esm.pretrained.esmfold_v1()
        model = model.to(args.device)
        model.eval()
    except Exception as exc:
        return None, {"plddt_status": f"unavailable: {exc}"}

    mean_plddts = []
    for seq in seqs:
        output = model.infer(seq)
        mean_plddts.append(float(output["mean_plddt"][0].detach().cpu().item()))
    arr = np.asarray(mean_plddts, dtype=float)
    return arr, {
        "plddt_backend": "esmfold",
        "plddt_status": "ok",
        "mean_plddt_mean": float(arr.mean()),
        "mean_plddt_median": float(np.median(arr)),
        "mean_plddt_min": float(arr.min()),
        "mean_plddt_max": float(arr.max()),
    }


def resolve_plddt_cache_path(args: argparse.Namespace) -> Path:
    if args.plddt_cache_csv:
        return Path(args.plddt_cache_csv)
    return Path(args.output_dir) / f"plddt_cache_{args.plddt_backend}.csv"


def load_plddt_cache(cache_path: Path) -> Dict[str, Dict[str, object]]:
    if not cache_path.exists():
        return {}
    df = pd.read_csv(cache_path)
    required = {"sequence", "mean_plddt"}
    if not required.issubset(df.columns):
        return {}

    cache = {}
    for row in df.to_dict(orient="records"):
        seq = row["sequence"]
        if not isinstance(seq, str) or not seq:
            continue
        try:
            mean_plddt = float(row["mean_plddt"])
        except (TypeError, ValueError):
            continue
        cache[seq] = {
            "mean_plddt": mean_plddt,
            "artifact_path": row.get("artifact_path"),
            "backend": row.get("backend", "colabfold"),
        }
    return cache


def write_plddt_cache(cache_path: Path, cache: Dict[str, Dict[str, object]]) -> None:
    ensure_dir(str(cache_path.parent))
    rows = []
    for seq, payload in cache.items():
        rows.append({
            "sequence_sha1": hashlib.sha1(seq.encode()).hexdigest(),
            "sequence": seq,
            "mean_plddt": payload["mean_plddt"],
            "artifact_path": payload.get("artifact_path"),
            "backend": payload.get("backend", "colabfold"),
        })
    df = pd.DataFrame(rows).sort_values("sequence_sha1") if rows else pd.DataFrame(
        columns=["sequence_sha1", "sequence", "mean_plddt", "artifact_path", "backend"]
    )
    df.to_csv(cache_path, index=False)


def parse_mean_plddt_from_pdb(pdb_path: Path) -> float:
    ca_values = []
    atom_values = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            try:
                b_factor = float(line[60:66].strip())
            except ValueError:
                continue
            atom_values.append(b_factor)
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                ca_values.append(b_factor)

    values = ca_values if ca_values else atom_values
    if not values:
        raise ValueError(f"No ATOM records with B-factors found in {pdb_path}")
    return float(np.mean(values))


def pick_colabfold_prediction_file(output_dir: Path, seq_id: str) -> Optional[Path]:
    candidates = sorted(output_dir.rglob(f"{seq_id}*.pdb"))
    if not candidates:
        return None

    def sort_key(path: Path) -> Tuple[int, int, str]:
        name = path.name
        rank_score = 0 if "rank_001" in name else 1
        relaxed_score = 0 if "relaxed" in name else 1
        return (rank_score, relaxed_score, name)

    return sorted(candidates, key=sort_key)[0]


def run_colabfold_batch(
    fasta_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    cmd = build_colabfold_batch_command(fasta_path, output_dir, args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        return False, str(exc)
    combined_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, combined_output.strip()


def build_colabfold_batch_command(
    fasta_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [args.colabfold_batch_cmd]
    if args.colabfold_msa_mode:
        cmd.extend(["--msa-mode", args.colabfold_msa_mode])
    if args.colabfold_model_type:
        cmd.extend(["--model-type", args.colabfold_model_type])
    if args.colabfold_rank:
        cmd.extend(["--rank", args.colabfold_rank])
    if args.colabfold_num_models:
        cmd.extend(["--num-models", str(args.colabfold_num_models)])
    if args.colabfold_num_seeds:
        cmd.extend(["--num-seeds", str(args.colabfold_num_seeds)])
    if args.colabfold_data_dir:
        cmd.extend(["--data", args.colabfold_data_dir])
    if args.colabfold_host_url:
        cmd.extend(["--host-url", args.colabfold_host_url])
    if args.colabfold_templates:
        cmd.append("--templates")
    if args.colabfold_overwrite_existing_results:
        cmd.append("--overwrite-existing-results")
    if args.colabfold_batch_args:
        cmd.extend(shlex.split(args.colabfold_batch_args))
    cmd.extend([str(fasta_path), str(output_dir)])
    return cmd


def maybe_compute_plddt_colabfold(
    seqs: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    cache_path = resolve_plddt_cache_path(args)
    cache = load_plddt_cache(cache_path)
    missing = []
    for seq in seqs:
        if seq not in cache:
            missing.append(seq)

    missing = list(dict.fromkeys(missing))
    colabfold_output_dir = Path(args.colabfold_output_dir) if args.colabfold_output_dir else Path(args.output_dir) / "plddt_colabfold"
    batch_output = ""

    if missing:
        ensure_dir(str(colabfold_output_dir))
        batch_digest = hashlib.sha1("".join(missing).encode()).hexdigest()[:12]
        batch_dir = colabfold_output_dir / f"batch_{batch_digest}"
        ensure_dir(str(batch_dir))
        fasta_path = batch_dir / "queries.fa"

        seq_id_to_seq = {}
        with fasta_path.open("w") as handle:
            for idx, seq in enumerate(missing):
                seq_digest = hashlib.sha1(seq.encode()).hexdigest()[:12]
                seq_id = f"seq_{idx:05d}_{seq_digest}"
                seq_id_to_seq[seq_id] = seq
                handle.write(f">{seq_id}\n{seq}\n")

        ok, batch_output = run_colabfold_batch(fasta_path, batch_dir, args)
        if not ok:
            return None, {
                "plddt_backend": "colabfold",
                "plddt_status": f"unavailable: {batch_output or 'colabfold_batch failed'}",
                "plddt_cache_csv": str(cache_path),
            }

        for seq_id, seq in seq_id_to_seq.items():
            pdb_path = pick_colabfold_prediction_file(batch_dir, seq_id)
            if pdb_path is None:
                return None, {
                    "plddt_backend": "colabfold",
                    "plddt_status": f"unavailable: no prediction PDB found for {seq_id}",
                    "plddt_cache_csv": str(cache_path),
                    "colabfold_output_dir": str(batch_dir),
                }
            cache[seq] = {
                "mean_plddt": parse_mean_plddt_from_pdb(pdb_path),
                "artifact_path": str(pdb_path),
                "backend": "colabfold",
            }

        write_plddt_cache(cache_path, cache)

    values = [cache[seq]["mean_plddt"] for seq in seqs if seq in cache]
    if len(values) != len(seqs):
        return None, {
            "plddt_backend": "colabfold",
            "plddt_status": "unavailable: missing cache entries after ColabFold run",
            "plddt_cache_csv": str(cache_path),
        }

    arr = np.asarray(values, dtype=float)
    summary = {
        "plddt_backend": "colabfold",
        "plddt_status": "ok",
        "plddt_cache_csv": str(cache_path),
        "mean_plddt_mean": float(arr.mean()),
        "mean_plddt_median": float(np.median(arr)),
        "mean_plddt_min": float(arr.min()),
        "mean_plddt_max": float(arr.max()),
    }
    if batch_output:
        summary["plddt_backend_message"] = batch_output[-500:]
    return arr, summary


def summarize_diversity(seqs: Sequence[str], max_pairs: int, seed: int) -> Dict[str, float]:
    unique_fraction = len(set(seqs)) / max(len(seqs), 1)
    if len(seqs) < 2:
        return {
            "unique_fraction": unique_fraction,
            "pairwise_hamming_mean": 0.0,
            "pairwise_edit_similarity_mean": 1.0,
        }

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
    edit_similarities = []
    for i, j in pairs:
        if len(seqs[i]) != len(seqs[j]):
            edit_similarities.append(normalized_edit_similarity(seqs[i], seqs[j]))
            continue
        distances.append(hamming_distance(seqs[i], seqs[j]) / max(len(seqs[i]), 1))
        edit_similarities.append(normalized_edit_similarity(seqs[i], seqs[j]))

    pairwise_mean = float(np.mean(distances)) if distances else 0.0
    pairwise_edit_similarity_mean = float(np.mean(edit_similarities)) if edit_similarities else 1.0
    return {
        "unique_fraction": float(unique_fraction),
        "pairwise_hamming_mean": pairwise_mean,
        "pairwise_edit_similarity_mean": pairwise_edit_similarity_mean,
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
        edited_top = results_df.nlargest(actual_k, primary_col)
        source_top = results_df.nlargest(actual_k, f"source_{primary_col}")
        top_k_values[f"top_{k}_{primary_col}_mean"] = float(edited_top[primary_col].mean())
        top_k_values[f"source_top_{k}_{primary_col}_mean"] = float(source_top[f"source_{primary_col}"].mean())
        top_k_values[f"delta_top_{k}_{primary_col}_mean"] = float(
            edited_top[primary_col].mean() - source_top[f"source_{primary_col}"].mean()
        )
        top_k_values[f"top_{k}_success_rate"] = float((edited_top[f"delta_{primary_col}"] > 0).mean())

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
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    if args.skip_ppl:
        return None, {"ppl_status": "skipped"}
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
    return arr, {
        "ppl_status": "ok",
        "ppl_mean": float(arr.mean()),
        "ppl_median": float(np.median(arr)),
        "ppl_std": float(arr.std()),
    }


def summarize_round_trajectories(
    round_metrics: Sequence[Dict[str, float]],
    primary_metric: str,
) -> Dict[str, float]:
    if not round_metrics:
        return {}

    round_df = pd.DataFrame(round_metrics).sort_values("round")
    summary = {
        "round_count": int(len(round_df)),
        f"delta_{primary_metric}_auc": float(np.trapz(round_df[f"delta_{primary_metric}_mean"], round_df["round"])),
        "representation_l2_auc": float(np.trapz(round_df["representation_l2_mean"], round_df["round"])),
        "hamming_auc": float(np.trapz(round_df["hamming_mean"], round_df["round"])),
    }

    best_idx = int(round_df[f"{primary_metric}_mean"].idxmax())
    best_row = round_df.loc[best_idx]
    summary[f"best_round_by_{primary_metric}"] = int(best_row["round"])
    summary[f"best_round_{primary_metric}_mean"] = float(best_row[f"{primary_metric}_mean"])
    summary[f"best_round_delta_{primary_metric}_mean"] = float(best_row[f"delta_{primary_metric}_mean"])
    return summary


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
    source_scores, source_features = evaluate_predictors(source_seqs, model, alphabet, args.device, predictors)

    alignment_projector = build_alignment_projector("identity")
    primary_metric = args.property
    run_summary = {
        "args": vars(args),
        "steering_meta": steering_meta,
        "results": {},
        "notes": {
            "alignment_steering": "MVP uses identity projector hook; replace with diffusion prior later.",
            "aspo_masking": "ASPO-style approximation using token-relatedness scores at selected ESM2 layers and editing the lowest-score sites.",
            "optional_metrics_not_in_mvp": [],
        },
    }

    for mode in args.modes:
        mode_dir = os.path.join(args.output_dir, mode)
        ensure_dir(mode_dir)
        rng = np.random.default_rng(args.seed)
        edited_rows = []
        edited_seqs = []
        round_sequences_by_source = []

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
            round_sequences_by_source.append(edit_result["round_sequences"])
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

        edited_scores, edited_features = evaluate_predictors(edited_seqs, model, alphabet, args.device, predictors)
        result_df = pd.DataFrame(edited_rows)
        for predictor_name in ["sol", "therm"]:
            result_df[predictor_name] = edited_scores[predictor_name]
            result_df[f"source_{predictor_name}"] = source_scores[predictor_name]
            result_df[f"delta_{predictor_name}"] = (
                result_df[predictor_name] - result_df[f"source_{predictor_name}"]
            )

        rep_drift = compute_representation_drift(source_features, edited_features)
        result_df["representation_l2"] = rep_drift["representation_l2"]
        result_df["representation_cosine"] = rep_drift["representation_cosine"]

        pattern_df = pd.DataFrame(
            [mutation_pattern_metrics(src, edt, args.mutation_entropy_bins) for src, edt in zip(source_seqs, edited_seqs)]
        )
        for col in pattern_df.columns:
            result_df[col] = pattern_df[col].to_numpy()

        if score_col is not None:
            result_df["input_score"] = input_df[score_col].to_numpy()

        source_nearest_natural_identity, source_natural_summary = maybe_compute_nearest_natural_identity(source_seqs, args)
        nearest_natural_identity, natural_summary = maybe_compute_nearest_natural_identity(edited_seqs, args)
        if source_nearest_natural_identity is not None and nearest_natural_identity is not None:
            result_df["source_nearest_natural_identity"] = source_nearest_natural_identity
            result_df["nearest_natural_identity"] = nearest_natural_identity
            result_df["delta_nearest_natural_identity"] = (
                result_df["nearest_natural_identity"] - result_df["source_nearest_natural_identity"]
            )

        source_plddt_values, source_plddt_summary = maybe_compute_plddt(source_seqs, args)
        plddt_values, plddt_summary = maybe_compute_plddt(edited_seqs, args)
        if source_plddt_values is not None and plddt_values is not None:
            result_df["source_mean_plddt"] = source_plddt_values
            result_df["mean_plddt"] = plddt_values
            result_df["delta_mean_plddt"] = result_df["mean_plddt"] - result_df["source_mean_plddt"]

        source_ppl_values, source_ppl_summary = maybe_compute_ppl(source_seqs, args)
        ppl_values, ppl_summary = maybe_compute_ppl(edited_seqs, args)
        if source_ppl_values is not None and ppl_values is not None:
            result_df["source_ppl"] = source_ppl_values
            result_df["ppl"] = ppl_values
            result_df["delta_ppl"] = result_df["ppl"] - result_df["source_ppl"]
        diversity_summary = summarize_diversity(edited_seqs, args.diversity_pairs, args.seed)
        primary_summary = summarize_primary_metrics(result_df, primary_metric)
        rep_summary = summarize_representation_drift(rep_drift)
        mutation_summary = summarize_mutation_patterns(
            result_df,
            source_seqs,
            edited_seqs,
            args.mutation_entropy_bins,
        )
        round_metrics = maybe_compute_round_metrics(
            round_sequences_by_source,
            source_features,
            source_scores,
            source_seqs,
            model,
            alphabet,
            args.device,
            predictors,
            args.mutation_entropy_bins,
        )
        round_summary = summarize_round_trajectories(round_metrics, primary_metric)
        sol_threshold_cross = (
            (result_df["source_sol"] < 0.5) & (result_df["sol"] >= 0.5)
        ).mean()

        summary = {
            **primary_summary,
            **rep_summary,
            **mutation_summary,
            **diversity_summary,
            **round_summary,
            "sol_threshold_cross_rate": float(sol_threshold_cross),
            **natural_summary,
            **plddt_summary,
        }
        if source_natural_summary.get("nearest_natural_status") == "ok" and natural_summary.get("nearest_natural_status") == "ok":
            summary.update({
                "source_nearest_natural_identity_mean": float(source_nearest_natural_identity.mean()),
                "source_nearest_natural_identity_median": float(np.median(source_nearest_natural_identity)),
                "delta_nearest_natural_identity_mean": float(
                    (nearest_natural_identity - source_nearest_natural_identity).mean()
                ),
                "delta_nearest_natural_identity_median": float(
                    np.median(nearest_natural_identity - source_nearest_natural_identity)
                ),
            })
        if source_plddt_summary.get("plddt_status") == "ok" and plddt_summary.get("plddt_status") == "ok":
            summary.update({
                "source_mean_plddt_mean": float(source_plddt_values.mean()),
                "source_mean_plddt_median": float(np.median(source_plddt_values)),
                "delta_mean_plddt_mean": float((plddt_values - source_plddt_values).mean()),
                "delta_mean_plddt_median": float(np.median(plddt_values - source_plddt_values)),
            })
        if source_ppl_summary.get("ppl_status") == "ok" and ppl_summary.get("ppl_status") == "ok":
            summary.update({
                "source_ppl_mean": float(source_ppl_values.mean()),
                "source_ppl_median": float(np.median(source_ppl_values)),
                "delta_ppl_mean": float((ppl_values - source_ppl_values).mean()),
                "delta_ppl_median": float(np.median(ppl_values - source_ppl_values)),
            })
        if ppl_summary is not None:
            summary.update(ppl_summary)

        csv_path = os.path.join(mode_dir, "per_sequence_results.csv")
        result_df.to_csv(csv_path, index=False)
        run_summary["results"][mode] = {
            "summary": summary,
            "csv_path": csv_path,
            "round_metrics": round_metrics,
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(json.dumps(run_summary, indent=2))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
