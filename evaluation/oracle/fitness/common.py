"""
Shared model, feature-extraction, and metric helpers for fitness oracle training.
"""

import math
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


MODEL_NAMES = {
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
    "3B": "esm2_t36_3B_UR50D",
}
MODEL_EMBED_DIMS = {
    "150M": 640,
    "650M": 1280,
    "3B": 2560,
}


class FitnessPredictor(nn.Module):
    """RobertaLMHead-style scalar regressor on frozen ESM2 features."""

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


class LinearFitnessPredictor(nn.Module):
    """Simple linear probe on frozen ESM2 features."""

    def __init__(self, embed_dim=1280):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def build_predictor(head="lm_head", embed_dim=1280):
    if head == "lm_head":
        return FitnessPredictor(embed_dim=embed_dim)
    if head == "linear":
        return LinearFitnessPredictor(embed_dim=embed_dim)
    raise ValueError(f"Unsupported head {head!r}")


def load_esm2_model(model_size, device):
    try:
        import esm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'esm'. Install the main repo environment first "
            "(see README.md: pip install fair-esm ...)."
        ) from exc

    model_name = MODEL_NAMES[model_size]
    print(f"Loading {model_name} on {device}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    return model, alphabet


def extract_features_single_device(
    seqs,
    model,
    alphabet,
    device,
    batch_size=8,
    max_len=1022,
    last_n_layers=1,
    desc="Extracting features",
):
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    repr_layers = list(range(n_layers - last_n_layers + 1, n_layers + 1))
    all_features = []

    n_batches = (len(seqs) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(seqs), batch_size), total=n_batches, desc=desc):
        batch_seqs = seqs[start:start + batch_size]
        batch_seqs = [seq[:max_len] for seq in batch_seqs]
        data = [("protein", seq) for seq in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=repr_layers)

        for seq_index, seq_len in enumerate(batch_lens):
            layer_reps = []
            for layer in repr_layers:
                token_reps = results["representations"][layer]
                rep = token_reps[seq_index, 1:seq_len - 1].mean(0)
                layer_reps.append(rep)
            all_features.append(torch.stack(layer_reps).mean(0).cpu())

    return torch.stack(all_features)


def _worker_extract(rank, model_size, gpu_ids, seqs_chunk, batch_size, max_len, last_n_layers, output_path):
    device = f"cuda:{gpu_ids[rank]}"
    model, alphabet = load_esm2_model(model_size, device)
    features = extract_features_single_device(
        seqs_chunk,
        model,
        alphabet,
        device,
        batch_size=batch_size,
        max_len=max_len,
        last_n_layers=last_n_layers,
        desc=f"GPU {gpu_ids[rank]}",
    )
    torch.save(features, output_path)
    del model
    torch.cuda.empty_cache()


def extract_features_multi_device(
    seqs,
    model_size,
    gpu_ids,
    batch_size=8,
    max_len=1022,
    last_n_layers=1,
    cache_dir="/tmp/esm2_feat_cache",
):
    if len(seqs) == 0:
        raise ValueError("Cannot extract features for an empty sequence list")

    if gpu_ids and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU feature extraction")
        gpu_ids = []

    if not gpu_ids:
        model, alphabet = load_esm2_model(model_size, "cpu")
        return extract_features_single_device(
            seqs,
            model,
            alphabet,
            "cpu",
            batch_size=batch_size,
            max_len=max_len,
            last_n_layers=last_n_layers,
            desc="CPU",
        )

    if len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"
        model, alphabet = load_esm2_model(model_size, device)
        return extract_features_single_device(
            seqs,
            model,
            alphabet,
            device,
            batch_size=batch_size,
            max_len=max_len,
            last_n_layers=last_n_layers,
            desc=f"GPU {gpu_ids[0]}",
        )

    os.makedirs(cache_dir, exist_ok=True)
    n_gpus = len(gpu_ids)
    chunk_size = (len(seqs) + n_gpus - 1) // n_gpus
    chunks = [seqs[index * chunk_size:(index + 1) * chunk_size] for index in range(n_gpus)]
    chunks = [chunk for chunk in chunks if chunk]
    output_paths = [os.path.join(cache_dir, f"chunk_{index}.pt") for index in range(len(chunks))]

    processes = []
    mp.set_start_method("spawn", force=True)
    for rank, seqs_chunk in enumerate(chunks):
        process = mp.Process(
            target=_worker_extract,
            args=(rank, model_size, gpu_ids, seqs_chunk, batch_size, max_len, last_n_layers, output_paths[rank]),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Feature extraction worker exited with code {process.exitcode}")

    all_features = []
    for output_path in output_paths:
        all_features.append(torch.load(output_path))
        os.remove(output_path)
    return torch.cat(all_features, dim=0)


def fit_target_transform(train_values, transform_name):
    train_values = np.asarray(train_values, dtype=np.float32)
    if transform_name == "none":
        return {"name": "none"}
    if transform_name == "zscore":
        mean = float(train_values.mean())
        std = float(train_values.std())
        if std == 0.0:
            std = 1.0
        return {"name": "zscore", "mean": mean, "std": std}
    raise ValueError(f"Unsupported target transform {transform_name!r}")


def apply_target_transform(values, transform):
    values = np.asarray(values, dtype=np.float32)
    if transform["name"] == "none":
        return values
    if transform["name"] == "zscore":
        return (values - transform["mean"]) / transform["std"]
    raise ValueError(f"Unsupported target transform {transform['name']!r}")


def inverse_target_transform(values, transform):
    values = np.asarray(values, dtype=np.float32)
    if transform["name"] == "none":
        return values
    if transform["name"] == "zscore":
        return values * transform["std"] + transform["mean"]
    raise ValueError(f"Unsupported target transform {transform['name']!r}")


def top_k_enrichment(y_true, y_pred, fraction=0.05):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if len(y_true) == 0:
        return float("nan")

    k = max(1, int(math.ceil(len(y_true) * fraction)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    hit_rate = len(true_top & pred_top) / k
    baseline = k / len(y_true)
    return hit_rate / baseline


def average_ranks(values):
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    cursor = 0
    while cursor < len(values):
        next_cursor = cursor + 1
        while next_cursor < len(values) and values[order[next_cursor]] == values[order[cursor]]:
            next_cursor += 1
        avg_rank = 0.5 * (cursor + next_cursor - 1) + 1.0
        ranks[order[cursor:next_cursor]] = avg_rank
        cursor = next_cursor
    return ranks


def pearson_correlation(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = math.sqrt(float(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def regression_metrics(y_true, y_pred, top_k_fraction=0.05):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    residual = y_true - y_pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    metrics = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if len(y_true) > 1 and ss_tot > 0 else float("nan"),
        "top_k_fraction": float(top_k_fraction),
        "top_k_enrichment": float(top_k_enrichment(y_true, y_pred, fraction=top_k_fraction)),
    }

    metrics["pearson_r"] = pearson_correlation(y_true, y_pred)
    metrics["spearman_rho"] = pearson_correlation(average_ranks(y_true), average_ranks(y_pred))

    return metrics


def load_predictor_checkpoint(predictor_path, device="cpu"):
    checkpoint = torch.load(predictor_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    head = config.get("architecture", "lm_head")
    embed_dim = config.get("embed_dim", 1280)
    predictor = build_predictor(head=head, embed_dim=embed_dim)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor = predictor.to(device)
    predictor.eval()
    return predictor, config
