import argparse
import os
import re

import pandas as pd

from prepare_fitness_dataset import assign_group_splits, mutation_positions


DEFAULT_INPUT = "data/benchmarks/raw/gfp_sarkisyan/amino_acid_genotypes_to_brightness.tsv"
DEFAULT_FASTA = "data/benchmarks/raw/gfp_sarkisyan/avGFP_reference_sequence.fa"
DEFAULT_OUTPUT = "data/benchmarks/processed/gfp_sarkisyan/gfp_sarkisyan_processed.csv"
DEFAULT_POSITION_OFFSET = 1
MUTATION_PATTERN = re.compile(r"^S([A-Z])(\d+)([A-Z*])$")


CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def read_reference_aa_sequence(path):
    with open(path, "r") as handle:
        raw = "".join(line.strip() for line in handle if not line.startswith(">"))
    raw = raw.upper()
    if set(raw).issubset({"A", "C", "G", "T", "N"}):
        if len(raw) % 3 != 0:
            raise ValueError(f"Reference FASTA {path} has nucleotide length {len(raw)}; expected multiple of 3")
        aa = []
        for index in range(0, len(raw), 3):
            codon = raw[index:index + 3]
            aa.append(CODON_TABLE.get(codon, "X"))
        return "".join(aa).rstrip("*")
    return raw.rstrip("*")


def apply_mutations(wt_sequence, aa_mutations, position_offset):
    if pd.isna(aa_mutations):
        return wt_sequence

    seq = list(wt_sequence)
    for token in str(aa_mutations).split(":"):
        token = token.strip()
        if not token:
            continue
        match = MUTATION_PATTERN.match(token)
        if match is None:
            raise ValueError(f"Unsupported Sarkisyan mutation token {token!r}")
        wt_aa, position_text, mut_aa = match.groups()
        position = int(position_text) + position_offset
        if not 1 <= position <= len(seq):
            raise ValueError(
                f"Mutation {token!r} points outside reference sequence of length {len(seq)} "
                f"after applying offset {position_offset}"
            )
        if seq[position - 1] != wt_aa:
            raise ValueError(
                f"Mutation {token!r} expects WT residue {wt_aa} at position {position}, "
                f"found {seq[position - 1]}"
            )
        seq[position - 1] = mut_aa
    return "".join(seq)


def build_processed_dataframe(
    input_path,
    reference_fasta,
    position_offset,
    seed,
    train_fraction,
    val_fraction,
    test_fraction,
):
    raw = pd.read_csv(input_path, sep="\t")
    wt_sequence = read_reference_aa_sequence(reference_fasta)

    processed = pd.DataFrame(
        {
            "sequence": raw["aaMutations"].map(lambda value: apply_mutations(wt_sequence, value, position_offset)),
            "fitness": pd.to_numeric(raw["medianBrightness"], errors="coerce"),
            "dataset": "gfp_sarkisyan",
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
    return processed.sort_values(["split", "sequence"]).reset_index(drop=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Prepare the Sarkisyan avGFP benchmark for fitness-oracle training")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--reference_fasta", type=str, default=DEFAULT_FASTA)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--position_offset", type=int, default=DEFAULT_POSITION_OFFSET)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    processed = build_processed_dataframe(
        input_path=args.input_path,
        reference_fasta=args.reference_fasta,
        position_offset=args.position_offset,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    processed.to_csv(args.output_csv, index=False)
    print(f"Prepared {len(processed)} GFP sequences")
    print(f"Split sizes: {processed['split'].value_counts().to_dict()}")
    print(f"Saved processed dataset to {args.output_csv}")


if __name__ == "__main__":
    main()
