"""
Evaluate a trained fitness oracle on arbitrary sequence tables.
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.oracle.fitness.common import (
    extract_features_multi_device,
    inverse_target_transform,
    load_predictor_checkpoint,
    regression_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Run a fitness oracle on a sequence table")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--predictor_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--label_col", type=str, default=None, help="Optional ground-truth fitness column")
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=[0])
    parser.add_argument("--batch_size_extract", type=int, default=8)
    parser.add_argument("--top_k_fraction", type=float, default=0.05)
    args = parser.parse_args()

    if args.gpu_ids and not torch.cuda.is_available():
        print("CUDA is not available; evaluation will run on CPU")
        args.gpu_ids = []

    if args.output_csv is None:
        base, _ = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_fitness_scored.csv"

    df = pd.read_csv(args.input_csv)
    if args.sequence_col not in df.columns:
        raise ValueError(f"Sequence column {args.sequence_col!r} not found in {args.input_csv}")

    predictor, config = load_predictor_checkpoint(
        args.predictor_path,
        device=f"cuda:{args.gpu_ids[0]}" if args.gpu_ids else "cpu",
    )
    model_size = config.get("esm_model", "650M")
    last_n_layers = int(config.get("args", {}).get("last_n_layers", 1))
    transform = config.get("target_transform", {"name": "none"})

    features = extract_features_multi_device(
        df[args.sequence_col].astype(str).tolist(),
        model_size=model_size,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size_extract,
        last_n_layers=last_n_layers,
        cache_dir=os.path.join("saved_predictors", "eval_cache"),
    )

    with torch.no_grad():
        preds_transformed = predictor(features.to(next(predictor.parameters()).device)).cpu().numpy()
    preds_raw = inverse_target_transform(preds_transformed, transform)

    output_df = df.copy()
    output_df["pred_fitness"] = preds_raw
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

    if args.label_col is not None:
        if args.label_col not in df.columns:
            raise ValueError(f"Label column {args.label_col!r} not found in {args.input_csv}")
        metrics = regression_metrics(df[args.label_col].values, preds_raw, top_k_fraction=args.top_k_fraction)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
