import argparse
import os
import zipfile

import pandas as pd

from prepare_fitness_dataset import assign_group_splits, mutation_positions


DEFAULT_INPUT = "data/benchmarks/raw/trpb/data.zip"
DEFAULT_CSV_MEMBER = "data/figure_data/4-site_merged_replicates/20230827/four-site_simplified_AA_data.csv"
DEFAULT_FASTA_MEMBER = "data/ftmlde_data/tm9d8s_AAs.fasta"
DEFAULT_OUTPUT = "data/benchmarks/processed/trpb/trpb_processed.csv"
DEFAULT_POSITIONS = [183, 184, 227, 228]


def read_zip_csv(zip_path, member):
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle)


def read_zip_fasta(zip_path, member):
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            text = handle.read().decode("utf-8")
    return "".join(line.strip() for line in text.splitlines() if not line.startswith(">")).rstrip("*")


def apply_four_site_mutations(wt_sequence, mutant_aas, positions):
    if len(mutant_aas) != len(positions):
        raise ValueError(
            f"Expected {len(positions)} amino acids for positions {positions}, got sequence {mutant_aas!r}"
        )
    seq = list(wt_sequence)
    for aa, position in zip(mutant_aas, positions):
        seq[position - 1] = aa
    return "".join(seq)


def build_processed_dataframe(zip_path, csv_member, fasta_member, positions, seed, train_fraction, val_fraction, test_fraction):
    raw = read_zip_csv(zip_path, csv_member)
    wt_sequence = read_zip_fasta(zip_path, fasta_member)
    wt_signature = "".join(wt_sequence[position - 1] for position in positions)

    processed = pd.DataFrame(
        {
            "sequence": raw["AAs"].astype(str).map(lambda aas: apply_four_site_mutations(wt_sequence, aas, positions)),
            "fitness": pd.to_numeric(raw["fitness"], errors="coerce"),
            "dataset": "trpb",
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
    print(f"WT 4-site signature at positions {positions}: {wt_signature}")
    return processed.sort_values(["split", "sequence"]).reset_index(drop=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Prepare the Johnston TrpB 4-site benchmark for fitness-oracle training")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--csv_member", type=str, default=DEFAULT_CSV_MEMBER)
    parser.add_argument("--fasta_member", type=str, default=DEFAULT_FASTA_MEMBER)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--positions", type=int, nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    processed = build_processed_dataframe(
        zip_path=args.input_path,
        csv_member=args.csv_member,
        fasta_member=args.fasta_member,
        positions=args.positions,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    processed.to_csv(args.output_csv, index=False)
    print(f"Prepared {len(processed)} TrpB sequences")
    print(f"Split sizes: {processed['split'].value_counts().to_dict()}")
    print(f"Saved processed dataset to {args.output_csv}")


if __name__ == "__main__":
    main()
