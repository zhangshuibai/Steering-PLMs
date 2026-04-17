"""
Normalize wet-lab fitness benchmark tables into a shared CSV schema.

Output columns:
    sequence
    fitness
    split
    dataset
    wt_sequence
    num_mutations
    mutated_positions

The script prefers official train/val/test splits when present. If no split
column is available, it can derive deterministic mutation-position-group splits
using the wild-type sequence to reduce near-neighbor leakage.
"""

import argparse
import hashlib
import math
import os

import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_ALIASES = [
    "sequence",
    "mutant_sequence",
    "mutated_sequence",
    "seq",
    "aa_sequence",
    "amino_acid_sequence",
]
DEFAULT_FITNESS_ALIASES = [
    "fitness",
    "score",
    "activity",
    "activity_score",
    "fluorescence",
    "brightness",
    "log_fitness",
]
DEFAULT_SPLIT_ALIASES = [
    "split",
    "set",
    "partition",
    "fold",
]
DEFAULT_WT_ALIASES = [
    "wt_sequence",
    "wildtype_sequence",
    "wild_type_sequence",
    "reference_sequence",
]
SPLIT_NORMALIZATION = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "te": "test",
    "holdout": "test",
    "eval": "test",
}


def parse_aliases(explicit, defaults):
    if explicit is None:
        return defaults
    return [item.strip() for item in explicit.split(",") if item.strip()]


def read_table(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format for {path!r}; expected .csv or .tsv")


def choose_column(df, explicit_name, aliases, label):
    if explicit_name is not None:
        if explicit_name not in df.columns:
            raise ValueError(f"{label} column {explicit_name!r} not found; columns={list(df.columns)}")
        return explicit_name

    lookup = {column.lower(): column for column in df.columns}
    for alias in aliases:
        column = lookup.get(alias.lower())
        if column is not None:
            return column

    raise ValueError(
        f"Could not infer {label} column from aliases {aliases}; available columns={list(df.columns)}"
    )


def normalize_split_values(series):
    normalized = []
    for value in series:
        key = str(value).strip().lower()
        mapped = SPLIT_NORMALIZATION.get(key)
        if mapped is None:
            raise ValueError(f"Unsupported split label {value!r}")
        normalized.append(mapped)
    return pd.Series(normalized, index=series.index)


def infer_wt_sequence(df, sequence_col, wt_col=None, wt_sequence=None):
    if wt_sequence:
        return wt_sequence.strip()

    if wt_col is not None:
        unique = [
            str(value).strip()
            for value in df[wt_col].dropna().unique().tolist()
            if str(value).strip()
        ]
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            raise ValueError(f"WT column {wt_col!r} has multiple sequences; pass --wt-sequence explicitly")

    if "num_mutations" in df.columns:
        numeric = pd.to_numeric(df["num_mutations"], errors="coerce")
        wt_rows = df[numeric == 0]
        if len(wt_rows) >= 1:
            return str(wt_rows.iloc[0][sequence_col]).strip()

    for candidate in ("variant", "mutation", "mutant"):
        if candidate in df.columns:
            text = df[candidate].astype(str).str.strip().str.lower()
            wt_rows = df[text.isin({"wt", "wildtype", "wild_type"})]
            if len(wt_rows) >= 1:
                return str(wt_rows.iloc[0][sequence_col]).strip()

    return None


def mutation_positions(sequence, wt_sequence):
    if len(sequence) != len(wt_sequence):
        raise ValueError(
            "Mutation-aware split requires constant-length sequences. "
            f"Observed sequence length {len(sequence)} vs WT length {len(wt_sequence)}."
        )
    return tuple(index + 1 for index, (aa, wt_aa) in enumerate(zip(sequence, wt_sequence)) if aa != wt_aa)


def stable_hash(text, seed):
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def split_counts(n_items, train_frac, val_frac, test_frac):
    if n_items <= 0:
        return 0, 0, 0

    raw = np.array([train_frac, val_frac, test_frac], dtype=float)
    if not math.isclose(raw.sum(), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {raw.tolist()}")

    counts = np.floor(raw * n_items).astype(int)
    remainder = n_items - counts.sum()
    if remainder > 0:
        fractions = raw * n_items - counts
        for index in np.argsort(-fractions)[:remainder]:
            counts[index] += 1

    if n_items >= 3:
        for index in range(3):
            if raw[index] > 0 and counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def assign_ordered_splits(items, train_frac, val_frac, test_frac):
    n_train, n_val, n_test = split_counts(len(items), train_frac, val_frac, test_frac)
    assignments = {}
    cursor = 0
    for item in items[cursor:cursor + n_train]:
        assignments[item] = "train"
    cursor += n_train
    for item in items[cursor:cursor + n_val]:
        assignments[item] = "val"
    cursor += n_val
    for item in items[cursor:cursor + n_test]:
        assignments[item] = "test"
    return assignments


def build_group_keys(df):
    if df["mutated_positions"].notna().all() and df["num_mutations"].notna().all():
        return (
            df["num_mutations"].astype(int).astype(str)
            + "|"
            + df["mutated_positions"].astype(str)
        )
    if df["mutated_positions"].notna().all():
        return df["mutated_positions"].astype(str)
    return df.index.astype(str)


def ensure_required_group_splits(df, required_splits, seed, group_key_col="_group_key"):
    group_to_split = df.groupby(group_key_col)["split"].first().to_dict()

    for required in required_splits:
        if required in set(group_to_split.values()):
            continue

        donor = None
        donor_candidates = sorted(
            {split for split in group_to_split.values() if split != required},
            key=lambda split: (0 if split == "train" else 1, split),
        )
        for candidate in donor_candidates:
            n_groups = sum(1 for split in group_to_split.values() if split == candidate)
            if n_groups > 1:
                donor = candidate
                break

        if donor is None:
            continue

        donor_groups = [group for group, split in group_to_split.items() if split == donor]
        chosen_group = sorted(
            donor_groups,
            key=lambda group: stable_hash(f"{donor}->{required}:{group}", seed),
        )[0]
        group_to_split[chosen_group] = required

    df["split"] = df[group_key_col].map(group_to_split)
    return df


def assign_group_splits(df, train_frac, val_frac, test_frac, seed):
    df = df.copy()
    df["_group_key"] = build_group_keys(df)
    split_map = {}
    buckets = sorted(df["num_mutations"].astype(int).unique().tolist())
    for bucket in buckets:
        bucket_df = df[df["num_mutations"].astype(int) == bucket]
        groups = sorted(bucket_df["_group_key"].unique().tolist(), key=lambda value: stable_hash(value, seed))
        bucket_splits = assign_ordered_splits(groups, train_frac, val_frac, test_frac)
        split_map.update(bucket_splits)
    df["split"] = df["_group_key"].map(split_map)

    required_splits = []
    if train_frac > 0:
        required_splits.append("train")
    if val_frac > 0:
        required_splits.append("val")
    if test_frac > 0:
        required_splits.append("test")

    df = ensure_required_group_splits(df, required_splits, seed, group_key_col="_group_key")
    return df.drop(columns=["_group_key"])


def assign_random_splits(df, train_frac, val_frac, test_frac, seed):
    ordered = sorted(df.index.tolist(), key=lambda idx: stable_hash(str(df.loc[idx, "sequence"]), seed))
    split_map = assign_ordered_splits(ordered, train_frac, val_frac, test_frac)
    df["split"] = df.index.map(split_map)
    return df


def hold_out_validation_from_train(df, strategy, val_fraction, seed):
    train_df = df[df["split"] == "train"].copy()
    remaining_df = df[df["split"] != "train"].copy()
    if len(train_df) == 0:
        raise ValueError("Cannot carve validation split because no train rows are present")

    if strategy == "position_group" and train_df["mutated_positions"].isna().any():
        strategy = "random"

    if strategy == "position_group":
        rel_train = max(0.0, 1.0 - val_fraction)
        rel_val = min(1.0, val_fraction)
        train_df = assign_group_splits(train_df, rel_train, rel_val, 0.0, seed)
    else:
        train_df = assign_random_splits(train_df, 1.0 - val_fraction, val_fraction, 0.0, seed)

    combined = pd.concat([train_df, remaining_df], ignore_index=True)
    return combined


def assert_required_splits(df, required_splits, context):
    present = set(df["split"].tolist())
    missing = [split for split in required_splits if split not in present]
    if missing:
        raise ValueError(
            f"{context} could not produce required splits {missing}. "
            "Use official train/val/test labels, or provide enough mutation groups to carve the missing split(s) "
            "without leaking near-neighbor variants."
        )


def build_processed_dataframe(args):
    df = read_table(args.input_path)
    sequence_col = choose_column(
        df,
        args.sequence_col,
        parse_aliases(args.sequence_aliases, DEFAULT_SEQUENCE_ALIASES),
        "sequence",
    )
    fitness_col = choose_column(
        df,
        args.fitness_col,
        parse_aliases(args.fitness_aliases, DEFAULT_FITNESS_ALIASES),
        "fitness",
    )

    split_col = None
    if args.split_col is not None or args.split_strategy in {"official", "auto"}:
        try:
            split_col = choose_column(
                df,
                args.split_col,
                parse_aliases(args.split_aliases, DEFAULT_SPLIT_ALIASES),
                "split",
            )
        except ValueError:
            split_col = None

    wt_col = None
    if args.wt_col is not None:
        wt_col = choose_column(
            df,
            args.wt_col,
            parse_aliases(args.wt_aliases, DEFAULT_WT_ALIASES),
            "wild-type sequence",
        )
    else:
        try:
            wt_col = choose_column(
                df,
                None,
                parse_aliases(args.wt_aliases, DEFAULT_WT_ALIASES),
                "wild-type sequence",
            )
        except ValueError:
            wt_col = None

    processed = pd.DataFrame(
        {
            "source_row": df.index,
            "sequence": df[sequence_col].astype(str).str.strip(),
            "fitness": pd.to_numeric(df[fitness_col], errors="coerce"),
            "dataset": args.dataset_name,
        }
    )
    processed = processed[(processed["sequence"] != "") & processed["fitness"].notna()].copy()
    processed = processed.drop_duplicates(subset=["sequence"], keep=args.dedup_keep).reset_index(drop=True)

    wt_sequence = infer_wt_sequence(df, sequence_col, wt_col=wt_col, wt_sequence=args.wt_sequence)
    if wt_sequence is not None:
        processed["wt_sequence"] = wt_sequence
        processed["num_mutations"] = processed["sequence"].map(lambda value: len(mutation_positions(value, wt_sequence)))
        processed["mutated_positions"] = processed["sequence"].map(
            lambda value: ",".join(map(str, mutation_positions(value, wt_sequence))) or "WT"
        )
    else:
        processed["wt_sequence"] = np.nan
        processed["num_mutations"] = np.nan
        processed["mutated_positions"] = np.nan

    if split_col is not None and args.split_strategy in {"auto", "official"}:
        split_values = normalize_split_values(df.loc[processed["source_row"], split_col])
        processed["split"] = split_values.values
        if "val" not in set(processed["split"].tolist()):
            strategy = "position_group" if processed["mutated_positions"].notna().all() else "random"
            processed = hold_out_validation_from_train(processed, strategy, args.val_fraction, args.seed)
        assert_required_splits(processed, ["train", "val", "test"], "Official/auto split handling")
        return processed.drop(columns=["source_row"])

    split_strategy = args.split_strategy
    if split_strategy == "auto":
        split_strategy = "position_group"

    if split_strategy == "position_group":
        if processed["mutated_positions"].isna().any():
            raise ValueError(
                "Mutation-position split requested but WT sequence could not be inferred. "
                "Pass --wt-sequence or provide a WT column."
            )
        processed = assign_group_splits(
            processed,
            args.train_fraction,
            args.val_fraction,
            args.test_fraction,
            args.seed,
        )
        required = []
        if args.train_fraction > 0:
            required.append("train")
        if args.val_fraction > 0:
            required.append("val")
        if args.test_fraction > 0:
            required.append("test")
        assert_required_splits(processed, required, "Mutation-position split handling")
        return processed.drop(columns=["source_row"])

    if split_strategy == "random":
        if not args.allow_random_split:
            raise ValueError(
                "Random splitting is disabled by default because it risks leakage on benchmark libraries. "
                "Re-run with --allow-random-split if you explicitly want that behavior."
            )
        processed = assign_random_splits(
            processed,
            args.train_fraction,
            args.val_fraction,
            args.test_fraction,
            args.seed,
        )
        required = []
        if args.train_fraction > 0:
            required.append("train")
        if args.val_fraction > 0:
            required.append("val")
        if args.test_fraction > 0:
            required.append("test")
        assert_required_splits(processed, required, "Random split handling")
        return processed.drop(columns=["source_row"])

    raise ValueError(f"Unsupported split strategy {args.split_strategy!r}")


def build_arg_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Normalize a wet-lab fitness dataset into repo-standard CSV")
    parser.add_argument("--input_path", type=str, required=True, help="Raw CSV/TSV file")
    parser.add_argument("--output_csv", type=str, default=defaults.get("output_csv"))
    parser.add_argument("--dataset_name", type=str, default=defaults.get("dataset_name", "fitness"))
    parser.add_argument("--sequence_col", type=str, default=defaults.get("sequence_col"))
    parser.add_argument("--fitness_col", type=str, default=defaults.get("fitness_col"))
    parser.add_argument("--split_col", type=str, default=defaults.get("split_col"))
    parser.add_argument("--wt_col", type=str, default=defaults.get("wt_col"))
    parser.add_argument("--wt_sequence", type=str, default=defaults.get("wt_sequence"))
    parser.add_argument("--sequence_aliases", type=str, default=defaults.get("sequence_aliases"))
    parser.add_argument("--fitness_aliases", type=str, default=defaults.get("fitness_aliases"))
    parser.add_argument("--split_aliases", type=str, default=defaults.get("split_aliases"))
    parser.add_argument("--wt_aliases", type=str, default=defaults.get("wt_aliases"))
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=["auto", "official", "position_group", "random"],
        default=defaults.get("split_strategy", "auto"),
    )
    parser.add_argument("--allow_random_split", action="store_true", default=defaults.get("allow_random_split", False))
    parser.add_argument("--train_fraction", type=float, default=defaults.get("train_fraction", 0.8))
    parser.add_argument("--val_fraction", type=float, default=defaults.get("val_fraction", 0.1))
    parser.add_argument("--test_fraction", type=float, default=defaults.get("test_fraction", 0.1))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--dedup_keep", type=str, choices=["first", "last"], default=defaults.get("dedup_keep", "first"))
    return parser


def main(defaults=None):
    parser = build_arg_parser(defaults=defaults)
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(
            "data",
            "benchmarks",
            "processed",
            args.dataset_name,
            f"{args.dataset_name}_processed.csv",
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    processed = build_processed_dataframe(args)
    processed = processed.sort_values(["split", "sequence"]).reset_index(drop=True)

    split_sizes = processed["split"].value_counts().to_dict()
    print(f"Prepared {len(processed)} sequences for dataset={args.dataset_name}")
    print(f"Split sizes: {split_sizes}")
    if processed["num_mutations"].notna().any():
        print(
            "Mutation count summary: "
            f"min={processed['num_mutations'].min():.0f}, "
            f"median={processed['num_mutations'].median():.0f}, "
            f"max={processed['num_mutations'].max():.0f}"
        )

    processed.to_csv(args.output_csv, index=False)
    print(f"Saved processed dataset to {args.output_csv}")


if __name__ == "__main__":
    main()
