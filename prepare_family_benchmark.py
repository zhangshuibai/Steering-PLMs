"""
Prepare a controlled-family benchmark for steering-vector construction and optimization.

Pipeline stages
---------------
A. Build a clean family pool from FASTA/CSV input.
B. Score the pool with solubility and thermostability predictors.
C. Split the scored family pool into train/test.
D. Build positive/negative train subsets for each property.
E. Build low-scoring test subsets for optimization benchmarks.

The script is intentionally config-driven so data prep is decoupled from the
editing/evaluation pipeline. See configs/lysozyme_preprocess.example.json.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_FILES = {
    "pool": "lysozyme_pool.csv",
    "scored": "lysozyme_scored.csv",
    "train": "lysozyme_train.csv",
    "test": "lysozyme_test.csv",
    "p_sol_train": "P_sol_train.csv",
    "n_sol_train": "N_sol_train.csv",
    "p_therm_train": "P_therm_train.csv",
    "n_therm_train": "N_therm_train.csv",
    "n_sol_test": "N_sol_test.csv",
    "n_therm_test": "N_therm_test.csv",
    "reference_pool": "reference_generation_pool.csv",
    "manifest": "preprocess_manifest.json",
}

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class LoadedInput:
    df: pd.DataFrame
    metadata: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a controlled-family benchmark dataset.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional override for config.output_dir.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, object]:
    with open(path) as f:
        config = json.load(f)
    return config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_fasta(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seq_id: Optional[str] = None
    description = ""
    chunks: List[str] = []

    def flush_current() -> None:
        nonlocal seq_id, description, chunks
        if seq_id is None:
            return
        sequence = "".join(chunks).strip().upper()
        records.append(
            {
                "seq_id": seq_id,
                "sequence": sequence,
                "description": description,
            }
        )
        seq_id = None
        description = ""
        chunks = []

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_current()
                header = line[1:].strip()
                parts = header.split(maxsplit=1)
                seq_id = parts[0]
                description = parts[1] if len(parts) > 1 else ""
            else:
                chunks.append(line)
    flush_current()
    return records


def load_family_input(config: Dict[str, object]) -> LoadedInput:
    input_cfg = config["input"]
    input_path = Path(input_cfg["path"])
    input_format = str(input_cfg.get("format", input_path.suffix.lstrip(".").lower())).lower()

    metadata = {
        "input_path": str(input_path),
        "input_format": input_format,
    }

    if input_format in {"fasta", "fa", "faa"}:
        records = parse_fasta(input_path)
        df = pd.DataFrame(records)
    elif input_format in {"csv", "tsv"}:
        sep = "\t" if input_format == "tsv" else ","
        df = pd.read_csv(input_path, sep=sep)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    return LoadedInput(df=df, metadata=metadata)


def normalize_pool_columns(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    input_cfg = config["input"]
    family_cfg = config.get("family", {})

    seq_col = input_cfg.get("sequence_col", "sequence")
    id_col = input_cfg.get("id_col", "seq_id")
    description_col = input_cfg.get("description_col")
    source_col = input_cfg.get("source_db_col")
    family_tag_col = input_cfg.get("family_tag_col")

    if seq_col not in df.columns:
        raise ValueError(f"Input is missing sequence column '{seq_col}'")

    normalized = pd.DataFrame()

    if id_col in df.columns:
        normalized["seq_id"] = df[id_col].astype(str)
    else:
        prefix = str(family_cfg.get("seq_id_prefix", "seq"))
        normalized["seq_id"] = [f"{prefix}_{i:06d}" for i in range(len(df))]

    normalized["sequence"] = df[seq_col].astype(str).str.strip().str.upper()
    normalized["length"] = normalized["sequence"].str.len()

    if description_col and description_col in df.columns:
        normalized["description"] = df[description_col].fillna("").astype(str)
    else:
        normalized["description"] = ""

    if source_col and source_col in df.columns:
        normalized["source_db"] = df[source_col].fillna("").astype(str)
    else:
        normalized["source_db"] = str(family_cfg.get("source_db", ""))

    if family_tag_col and family_tag_col in df.columns:
        normalized["family_tag"] = df[family_tag_col].fillna("").astype(str)
    else:
        normalized["family_tag"] = str(family_cfg.get("family_tag", ""))

    passthrough_cols = [
        input_cfg.get("sol_logit_col"),
        input_cfg.get("sol_prob_col"),
        input_cfg.get("therm_score_col"),
    ]
    for col in passthrough_cols:
        if col and col in df.columns:
            normalized[col] = df[col]

    return normalized


def filter_pool(df: pd.DataFrame, config: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    family_cfg = config.get("family", {})
    min_length = int(family_cfg.get("min_length", 1))
    max_length = int(family_cfg.get("max_length", 1022))
    deduplicate_by_sequence = bool(family_cfg.get("deduplicate_by_sequence", True))
    drop_non_canonical = bool(family_cfg.get("drop_non_canonical", True))

    counts = {"input_rows": int(len(df))}

    filtered = df.copy()
    filtered = filtered[filtered["sequence"].notna()]
    filtered = filtered[filtered["sequence"] != ""]
    counts["non_empty_rows"] = int(len(filtered))

    filtered = filtered[(filtered["length"] >= min_length) & (filtered["length"] <= max_length)]
    counts["length_filtered_rows"] = int(len(filtered))

    if drop_non_canonical:
        filtered = filtered[filtered["sequence"].map(lambda seq: set(seq).issubset(AMINO_ACIDS))]
    counts["canonical_rows"] = int(len(filtered))

    if deduplicate_by_sequence:
        filtered = filtered.drop_duplicates(subset=["sequence"], keep="first")
    counts["deduplicated_rows"] = int(len(filtered))

    filtered = filtered.reset_index(drop=True)
    return filtered, counts


def load_predictor(path: str, embed_dim: int, device: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PropertyPredictor(nn.Module):
        def __init__(self, embed_dim: int = 1280):
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

    predictor = PropertyPredictor(embed_dim=embed_dim)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        predictor.load_state_dict(ckpt["model_state_dict"])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(device)
    predictor.eval()
    return predictor


def extract_last_layer_features(
    seqs: Sequence[str],
    esm_model,
    alphabet,
    device: str,
    batch_size: int,
    max_length: int,
):
    import torch

    batch_converter = alphabet.get_batch_converter()
    n_layers = esm_model.num_layers
    features = []

    for start in range(0, len(seqs), batch_size):
        batch_seqs = [seq[:max_length] for seq in seqs[start:start + batch_size]]
        data = [("protein", seq) for seq in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[n_layers])

        reps = results["representations"][n_layers]
        for i, seq_len in enumerate(batch_lens):
            features.append(reps[i, 1:seq_len - 1].mean(0).cpu())

    return torch.stack(features)


def score_pool(df: pd.DataFrame, config: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    scoring_cfg = config.get("scoring", {})
    if not scoring_cfg.get("enabled", True):
        scored = df.copy()
        input_cfg = config["input"]
        sol_logit_col = input_cfg.get("sol_logit_col", "sol_logit")
        sol_prob_col = input_cfg.get("sol_prob_col", "sol_prob")
        therm_score_col = input_cfg.get("therm_score_col", "therm_score")
        required = [sol_prob_col, therm_score_col]
        missing = [col for col in required if col not in scored.columns]
        if missing:
            raise ValueError(
                "Scoring is disabled, but required score columns are missing: "
                + ", ".join(missing)
            )
        if sol_logit_col not in scored.columns:
            probs = scored[sol_prob_col].astype(float).to_numpy()
            eps = 1e-6
            probs = np.clip(probs, eps, 1.0 - eps)
            scored["sol_logit"] = np.log(probs / (1.0 - probs))
        elif sol_logit_col != "sol_logit":
            scored["sol_logit"] = scored[sol_logit_col]
        if sol_prob_col != "sol_prob":
            scored["sol_prob"] = scored[sol_prob_col]
        if therm_score_col != "therm_score":
            scored["therm_score"] = scored[therm_score_col]
        return scored, {"scoring_enabled": False, "mode": "passthrough"}

    import torch
    from utils.esm2_utils import load_esm2_model

    device = str(scoring_cfg.get("device", "cuda"))
    batch_size = int(scoring_cfg.get("batch_size", 8))
    esm_model_size = str(scoring_cfg.get("esm_model", "650M"))
    max_length = int(config.get("family", {}).get("max_length", 1022))

    esm_model, alphabet = load_esm2_model(esm_model_size, device=device)
    embed_dim = esm_model.embed_dim

    sol_predictor = load_predictor(scoring_cfg["sol_predictor_path"], embed_dim, device)
    therm_predictor = load_predictor(scoring_cfg["therm_predictor_path"], embed_dim, device)

    features = extract_last_layer_features(
        df["sequence"].tolist(),
        esm_model=esm_model,
        alphabet=alphabet,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    with torch.no_grad():
        features_device = features.to(device)
        sol_logits = sol_predictor(features_device).cpu()
        sol_probs = torch.sigmoid(sol_logits)
        therm_scores = therm_predictor(features_device).cpu()

    scored = df.copy()
    scored["sol_logit"] = sol_logits.numpy()
    scored["sol_prob"] = sol_probs.numpy()
    scored["therm_score"] = therm_scores.numpy()
    scored["length"] = scored["sequence"].str.len()

    return scored, {
        "scoring_enabled": True,
        "device": device,
        "batch_size": batch_size,
        "esm_model": esm_model_size,
        "embed_dim": embed_dim,
        "sol_predictor_path": scoring_cfg["sol_predictor_path"],
        "therm_predictor_path": scoring_cfg["therm_predictor_path"],
    }


def random_split(df: pd.DataFrame, train_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("split.train_fraction must be between 0 and 1.")

    indices = list(range(len(df)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_size = int(round(len(indices) * train_fraction))
    train_size = min(max(train_size, 1), len(indices) - 1)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def quantile_selection(
    df: pd.DataFrame,
    score_col: str,
    fraction: float,
    descending: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Selection fraction must be in (0, 1].")
    count = max(1, int(math.ceil(len(df) * fraction)))
    selected = df.sort_values(score_col, ascending=not descending).head(count).reset_index(drop=True)
    threshold = float(selected[score_col].min() if descending else selected[score_col].max())
    return selected, {"count": int(len(selected)), "fraction": float(fraction), "threshold": threshold}


def bottom_n_selection(
    df: pd.DataFrame,
    score_col: str,
    target_size: int,
    base_fraction: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if len(df) == 0:
        raise ValueError("Cannot build benchmark input from an empty test split.")

    if target_size <= 0:
        raise ValueError("selection.test_target_size must be positive.")

    effective_fraction = min(1.0, base_fraction)
    count = min(len(df), max(1, int(math.ceil(len(df) * effective_fraction))))
    selected = df.sort_values(score_col, ascending=True).head(count).reset_index(drop=True)
    if len(selected) > target_size:
        selected = selected.head(target_size).reset_index(drop=True)

    threshold = float(selected[score_col].max())
    return selected, {
        "count": int(len(selected)),
        "requested_size": int(target_size),
        "effective_fraction": float(effective_fraction),
        "available_tail_count": int(count),
        "requested_size_exceeds_tail": bool(target_size > count),
        "threshold": threshold,
    }


def build_reference_pool(
    scored_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, object],
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, object]]]:
    ref_cfg = config.get("reference_pool", {})
    if not ref_cfg.get("enabled", False):
        return None, None

    source = str(ref_cfg.get("source", "test"))
    if source == "all":
        source_df = scored_df
    elif source == "train":
        source_df = train_df
    elif source == "test":
        source_df = test_df
    else:
        raise ValueError(f"Unsupported reference_pool.source: {source}")

    size = int(ref_cfg.get("size", min(100, len(source_df))))
    if size <= 0:
        raise ValueError("reference_pool.size must be positive.")

    shuffle = bool(ref_cfg.get("shuffle", True))
    seed = int(config.get("split", {}).get("seed", 42))

    if shuffle:
        reference_df = source_df.sample(n=min(size, len(source_df)), random_state=seed).reset_index(drop=True)
    else:
        reference_df = source_df.head(min(size, len(source_df))).reset_index(drop=True)

    metadata = {
        "source": source,
        "requested_size": size,
        "actual_size": int(len(reference_df)),
        "shuffle": shuffle,
    }
    return reference_df, metadata


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir or config["output_dir"])
    ensure_dir(output_dir)

    loaded = load_family_input(config)
    pool_df = normalize_pool_columns(loaded.df, config)
    pool_df, pool_counts = filter_pool(pool_df, config)

    if len(pool_df) < 2:
        raise ValueError("Family pool is too small after filtering; need at least 2 sequences.")

    output_paths = {name: output_dir / filename for name, filename in DEFAULT_OUTPUT_FILES.items()}
    write_csv(pool_df, output_paths["pool"])

    scored_df, scoring_meta = score_pool(pool_df, config)
    write_csv(scored_df, output_paths["scored"])

    split_cfg = config.get("split", {})
    train_fraction = float(split_cfg.get("train_fraction", 0.8))
    seed = int(split_cfg.get("seed", 42))
    train_df, test_df = random_split(scored_df, train_fraction=train_fraction, seed=seed)
    write_csv(train_df, output_paths["train"])
    write_csv(test_df, output_paths["test"])

    selection_cfg = config.get("selection", {})
    pos_fraction = float(selection_cfg.get("positive_fraction", 0.2))
    neg_fraction = float(selection_cfg.get("negative_fraction", 0.2))
    test_target_size = int(selection_cfg.get("test_target_size", 100))

    p_sol_train, p_sol_meta = quantile_selection(train_df, "sol_prob", pos_fraction, descending=True)
    n_sol_train, n_sol_meta = quantile_selection(train_df, "sol_prob", neg_fraction, descending=False)
    p_therm_train, p_therm_meta = quantile_selection(train_df, "therm_score", pos_fraction, descending=True)
    n_therm_train, n_therm_meta = quantile_selection(train_df, "therm_score", neg_fraction, descending=False)

    write_csv(p_sol_train, output_paths["p_sol_train"])
    write_csv(n_sol_train, output_paths["n_sol_train"])
    write_csv(p_therm_train, output_paths["p_therm_train"])
    write_csv(n_therm_train, output_paths["n_therm_train"])

    n_sol_test, n_sol_test_meta = bottom_n_selection(
        test_df, "sol_prob", target_size=test_target_size, base_fraction=neg_fraction
    )
    n_therm_test, n_therm_test_meta = bottom_n_selection(
        test_df, "therm_score", target_size=test_target_size, base_fraction=neg_fraction
    )

    write_csv(n_sol_test, output_paths["n_sol_test"])
    write_csv(n_therm_test, output_paths["n_therm_test"])

    reference_df, reference_meta = build_reference_pool(scored_df, train_df, test_df, config)
    if reference_df is not None:
        write_csv(reference_df, output_paths["reference_pool"])

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "input": loaded.metadata,
        "pool": {
            "counts": pool_counts,
            "final_size": int(len(pool_df)),
        },
        "scoring": scoring_meta,
        "split": {
            "train_fraction": train_fraction,
            "seed": seed,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        },
        "selection": {
            "positive_fraction": pos_fraction,
            "negative_fraction": neg_fraction,
            "test_target_size": test_target_size,
            "p_sol_train": p_sol_meta,
            "n_sol_train": n_sol_meta,
            "p_therm_train": p_therm_meta,
            "n_therm_train": n_therm_meta,
            "n_sol_test": n_sol_test_meta,
            "n_therm_test": n_therm_test_meta,
        },
        "files": {
            "lysozyme_pool_csv": str(output_paths["pool"].resolve()),
            "lysozyme_scored_csv": str(output_paths["scored"].resolve()),
            "lysozyme_train_csv": str(output_paths["train"].resolve()),
            "lysozyme_test_csv": str(output_paths["test"].resolve()),
            "p_sol_train_csv": str(output_paths["p_sol_train"].resolve()),
            "n_sol_train_csv": str(output_paths["n_sol_train"].resolve()),
            "p_therm_train_csv": str(output_paths["p_therm_train"].resolve()),
            "n_therm_train_csv": str(output_paths["n_therm_train"].resolve()),
            "n_sol_test_csv": str(output_paths["n_sol_test"].resolve()),
            "n_therm_test_csv": str(output_paths["n_therm_test"].resolve()),
            "manifest_json": str(output_paths["manifest"].resolve()),
        },
    }

    if reference_meta is not None:
        manifest["reference_pool"] = reference_meta
        manifest["files"]["reference_generation_pool_csv"] = str(output_paths["reference_pool"].resolve())

    with open(output_paths["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
