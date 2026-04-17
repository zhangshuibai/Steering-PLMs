"""
Train a regression oracle for wet-lab benchmark fitness datasets.

Expected processed CSV columns:
    sequence
    fitness
    split  (train / val / test)
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.oracle.fitness.common import (
    MODEL_EMBED_DIMS,
    apply_target_transform,
    build_predictor,
    extract_features_multi_device,
    fit_target_transform,
    inverse_target_transform,
    regression_metrics,
)


def load_processed_splits(processed_csv):
    df = pd.read_csv(processed_csv)
    required_columns = {"sequence", "fitness", "split"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{processed_csv} is missing required columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    valid_splits = {"train", "val", "test"}
    unknown = sorted(set(df["split"].tolist()).difference(valid_splits))
    if unknown:
        raise ValueError(f"Unsupported split labels in {processed_csv}: {unknown}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Processed dataset must contain non-empty train/val/test splits; "
            f"got train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    overlaps = []
    train_set = set(train_df["sequence"].tolist())
    val_set = set(val_df["sequence"].tolist())
    test_set = set(test_df["sequence"].tolist())
    if train_set & val_set:
        overlaps.append("train/val")
    if train_set & test_set:
        overlaps.append("train/test")
    if val_set & test_set:
        overlaps.append("val/test")
    if overlaps:
        raise ValueError(f"Detected exact sequence overlap across splits: {', '.join(overlaps)}")

    return train_df, val_df, test_df


def sequence_fingerprint(sequences):
    digest = hashlib.sha256()
    for sequence in sequences:
        encoded = str(sequence).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()[:12]


def load_or_compute_features(df, split_name, args):
    fingerprint = sequence_fingerprint(df["sequence"].tolist())
    suffix = f"{split_name}_{fingerprint}_features_{args.model_size}_last{args.last_n_layers}.pt"
    feature_path = os.path.join(args.features_dir, suffix)
    if os.path.exists(feature_path):
        print(f"Loading cached {split_name} features from {feature_path}")
        return torch.load(feature_path)

    print(f"Extracting {split_name} features using GPUs: {args.gpu_ids or 'cpu'}")
    features = extract_features_multi_device(
        df["sequence"].tolist(),
        model_size=args.model_size,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size_extract,
        last_n_layers=args.last_n_layers,
        cache_dir=os.path.join(args.features_dir, "tmp"),
    )
    torch.save(features, feature_path)
    print(f"Saved {split_name} features: {features.shape} -> {feature_path}")
    return features


def evaluate_split(predictor, features, labels_raw, transform, device, top_k_fraction):
    predictor.eval()
    with torch.no_grad():
        preds_transformed = predictor(features.to(device)).cpu().numpy()
    preds_raw = inverse_target_transform(preds_transformed, transform)
    metrics = regression_metrics(labels_raw, preds_raw, top_k_fraction=top_k_fraction)
    return metrics, preds_raw


def train_predictor(
    train_features,
    train_labels,
    val_features,
    val_labels,
    val_labels_raw,
    transform,
    args,
    embed_dim,
    device,
):
    predictor = build_predictor(head=args.head, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)

    best_score = float("-inf")
    best_state = None
    no_improve = 0

    progress = tqdm(range(args.epochs), desc="Training")
    for epoch in progress:
        predictor.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            preds = predictor(batch_features)
            loss = criterion(preds, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_features.size(0)

        avg_loss = total_loss / len(train_dataset)
        predictor.eval()
        with torch.no_grad():
            val_preds_transformed = predictor(val_features.to(device)).cpu().numpy()
        val_preds_raw = inverse_target_transform(val_preds_transformed, transform)
        val_metrics = regression_metrics(val_labels_raw, val_preds_raw, top_k_fraction=args.top_k_fraction)
        rho = float(np.nan_to_num(val_metrics["spearman_rho"], nan=-1.0))

        if rho > best_score:
            best_score = rho
            best_state = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        progress.set_postfix(loss=f"{avg_loss:.4f}", val_spearman=f"{rho:.4f}", best=f"{best_score:.4f}")
        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    predictor.load_state_dict(best_state)
    return predictor


def build_arg_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Train a wet-lab fitness oracle from processed benchmark CSV")
    parser.add_argument("--dataset_name", type=str, default=defaults.get("dataset_name", "fitness"))
    parser.add_argument("--processed_csv", type=str, default=defaults.get("processed_csv"))
    parser.add_argument("--save_path", type=str, default=defaults.get("save_path"))
    parser.add_argument("--features_dir", type=str, default=defaults.get("features_dir"))
    parser.add_argument("--model_size", type=str, choices=["150M", "650M", "3B"], default=defaults.get("model_size", "650M"))
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=defaults.get("gpu_ids", [0]))
    parser.add_argument("--batch_size_extract", type=int, default=defaults.get("batch_size_extract", 8))
    parser.add_argument("--batch_size_train", type=int, default=defaults.get("batch_size_train", 256))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 200))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-2))
    parser.add_argument("--patience", type=int, default=defaults.get("patience", 20))
    parser.add_argument("--head", type=str, choices=["lm_head", "linear"], default=defaults.get("head", "lm_head"))
    parser.add_argument("--last_n_layers", type=int, default=defaults.get("last_n_layers", 1))
    parser.add_argument("--target_transform", type=str, choices=["none", "zscore"], default=defaults.get("target_transform", "zscore"))
    parser.add_argument("--top_k_fraction", type=float, default=defaults.get("top_k_fraction", 0.05))
    return parser


def main(defaults=None):
    parser = build_arg_parser(defaults=defaults)
    args = parser.parse_args()

    if args.gpu_ids and not torch.cuda.is_available():
        print("CUDA is not available; training will run on CPU")
        args.gpu_ids = []

    if args.processed_csv is None:
        raise ValueError("--processed_csv is required")

    if args.save_path is None:
        args.save_path = os.path.join(
            "evaluation",
            "oracle",
            args.dataset_name,
            f"{args.dataset_name}_predictor_final.pt",
        )
    if args.features_dir is None:
        args.features_dir = os.path.join("saved_predictors", f"{args.dataset_name}_features")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    train_df, val_df, test_df = load_processed_splits(args.processed_csv)
    print(
        f"Loaded {args.dataset_name}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    train_features = load_or_compute_features(train_df, "train", args)
    val_features = load_or_compute_features(val_df, "val", args)
    test_features = load_or_compute_features(test_df, "test", args)

    transform = fit_target_transform(train_df["fitness"].values, args.target_transform)
    train_labels = torch.tensor(apply_target_transform(train_df["fitness"].values, transform), dtype=torch.float32)
    val_labels = torch.tensor(apply_target_transform(val_df["fitness"].values, transform), dtype=torch.float32)

    device = "cpu"
    if args.gpu_ids:
        device = f"cuda:{args.gpu_ids[0]}"

    embed_dim = MODEL_EMBED_DIMS[args.model_size]
    predictor = train_predictor(
        train_features,
        train_labels,
            val_features,
            val_labels,
            val_df["fitness"].values,
            transform,
            args,
            embed_dim=embed_dim,
            device=device,
        )

    val_metrics, _ = evaluate_split(
        predictor,
        val_features,
        val_df["fitness"].values,
        transform,
        device,
        args.top_k_fraction,
    )
    test_metrics, test_preds = evaluate_split(
        predictor,
        test_features,
        test_df["fitness"].values,
        transform,
        device,
        args.top_k_fraction,
    )

    print(f"Validation metrics: {json.dumps(val_metrics, indent=2)}")
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

    checkpoint = {
        "model_state_dict": predictor.state_dict(),
        "config": {
            "args": vars(args),
            "dataset_name": args.dataset_name,
            "architecture": args.head,
            "embed_dim": embed_dim,
            "esm_model": args.model_size,
            "feature_layer": "last" if args.last_n_layers == 1 else f"last_{args.last_n_layers}_avg",
            "target_transform": transform,
            "loss": "MSELoss",
            "data": {
                "processed_csv": args.processed_csv,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    }
    torch.save(checkpoint, args.save_path)

    config_path = args.save_path.replace(".pt", "_config.json")
    with open(config_path, "w") as handle:
        json.dump(checkpoint["config"], handle, indent=2)

    preds_path = args.save_path.replace(".pt", "_test_predictions.csv")
    pred_df = test_df.copy()
    pred_df["pred_fitness"] = test_preds
    pred_df.to_csv(preds_path, index=False)

    print(f"Saved predictor to {args.save_path}")
    print(f"Saved config to {config_path}")
    print(f"Saved test predictions to {preds_path}")


if __name__ == "__main__":
    main()
