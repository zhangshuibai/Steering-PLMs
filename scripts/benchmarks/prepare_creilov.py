import argparse
import os
import re

import pandas as pd

from prepare_fitness_dataset import assign_group_splits, mutation_positions


DEFAULT_INPUT = "data/benchmarks/raw/creilov/sb2c00662_si_002.xlsx"
DEFAULT_OUTPUT = "data/benchmarks/processed/creilov/creilov_processed.csv"
DEFAULT_WT_SEQUENCE = (
    "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEA"
    "CSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
)
DEFAULT_FITNESS_ALIASES = ["mean_log", "log_mean", "mean", "fitness"]
VARIANT_PATTERN = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter)$")
THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
}


def choose_fitness_column(df, explicit_name):
    if explicit_name is not None:
        if explicit_name not in df.columns:
            raise ValueError(f"Fitness column {explicit_name!r} not found; columns={list(df.columns)}")
        return explicit_name

    lowered = {column.lower(): column for column in df.columns}
    for alias in DEFAULT_FITNESS_ALIASES:
        column = lowered.get(alias.lower())
        if column is not None:
            return column

    raise ValueError(
        f"Could not infer fitness column from aliases {DEFAULT_FITNESS_ALIASES}; columns={list(df.columns)}"
    )


def split_variant_tokens(variant_text):
    text = str(variant_text).strip()
    if text.lower() in {"wt", "wildtype", "wild_type"}:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def apply_variant(wt_sequence, variant_text):
    seq = list(wt_sequence)
    for token in split_variant_tokens(variant_text):
        match = VARIANT_PATTERN.match(token)
        if match is None:
            raise ValueError(f"Unsupported CreiLOV variant token {token!r}")
        wt_three, position_text, mut_three = match.groups()
        wt_aa = THREE_TO_ONE[wt_three]
        mut_aa = THREE_TO_ONE[mut_three]
        position = int(position_text)
        if not 1 <= position <= len(seq):
            raise ValueError(
                f"Variant {token!r} points outside reference sequence of length {len(seq)}"
            )
        if seq[position - 1] != wt_aa:
            raise ValueError(
                f"Variant {token!r} expects WT residue {wt_aa} at position {position}, "
                f"found {seq[position - 1]}"
            )
        seq[position - 1] = mut_aa
    return "".join(seq)


def build_processed_dataframe(input_path, wt_sequence, variant_col, fitness_col, seed, train_fraction, val_fraction, test_fraction):
    raw = pd.read_excel(input_path)
    if variant_col not in raw.columns:
        raise ValueError(f"Variant column {variant_col!r} not found; columns={list(raw.columns)}")

    resolved_fitness_col = choose_fitness_column(raw, fitness_col)
    processed = pd.DataFrame(
        {
            "sequence": raw[variant_col].map(lambda value: apply_variant(wt_sequence, value)),
            "fitness": pd.to_numeric(raw[resolved_fitness_col], errors="coerce"),
            "dataset": "creilov",
            "wt_sequence": wt_sequence,
        }
    )
    processed = processed[processed["fitness"].notna()].copy()
    processed = processed[~processed["sequence"].str.contains(r"\*", regex=True)].copy()
    processed = processed.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    processed["num_mutations"] = processed["sequence"].map(lambda seq: len(mutation_positions(seq, wt_sequence)))
    processed["mutated_positions"] = processed["sequence"].map(
        lambda seq: ",".join(map(str, mutation_positions(seq, wt_sequence))) or "WT"
    )
    processed = assign_group_splits(processed, train_fraction, val_fraction, test_fraction, seed)
    return processed.sort_values(["split", "sequence"]).reset_index(drop=True), resolved_fitness_col


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Prepare the CreiLOV combinatorial benchmark for fitness-oracle training")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant_col", type=str, default="Unnamed: 0")
    parser.add_argument("--fitness_col", type=str, default=None)
    parser.add_argument("--wt_sequence", type=str, default=DEFAULT_WT_SEQUENCE)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    processed, resolved_fitness_col = build_processed_dataframe(
        input_path=args.input_path,
        wt_sequence=args.wt_sequence,
        variant_col=args.variant_col,
        fitness_col=args.fitness_col,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    processed.to_csv(args.output_csv, index=False)
    print(f"Prepared {len(processed)} CreiLOV sequences")
    print(f"Fitness column: {resolved_fitness_col}")
    print(f"Split sizes: {processed['split'].value_counts().to_dict()}")
    print(f"Saved processed dataset to {args.output_csv}")


if __name__ == "__main__":
    main()
