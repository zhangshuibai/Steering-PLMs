"""
Fetch a controlled protein family pool from UniProtKB and normalize it for
prepare_family_benchmark.py.

Default query targets a relatively homogeneous lysozyme-C family:
    reviewed:true AND protein_name:lysozyme AND fragment:false AND length:[80 TO 250]
    with a few explicit exclusions to avoid obvious non-family outliers.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_FIELDS = [
    "accession",
    "id",
    "protein_name",
    "organism_name",
    "length",
    "sequence",
    "gene_names",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a normalized family pool from UniProtKB.")
    parser.add_argument(
        "--query",
        default='(reviewed:true) AND (protein_name:lysozyme) AND (fragment:false) AND (length:[80 TO 250]) NOT (protein_name:"lysozyme-like") NOT (protein_name:"alpha-lactalbumin") NOT (protein_name:inhibitor) NOT (protein_name:endolysin)',
        help="UniProt query string.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/lysozyme/lysozyme_family_input.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_metadata_json",
        default="data/lysozyme/lysozyme_family_input.metadata.json",
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--family_tag",
        default="lysozyme_general",
        help="family_tag value written to the output CSV.",
    )
    parser.add_argument(
        "--source_db",
        default="UniProtKB/Swiss-Prot",
        help="source_db value written to the output CSV.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Page size for UniProt fetch. 500 is enough for the current lysozyme-C pool.",
    )
    return parser.parse_args()


def fetch_tsv(query: str, size: int) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    params = {
        "query": query,
        "format": "tsv",
        "size": size,
        "fields": ",".join(DEFAULT_FIELDS),
    }
    url = f"{UNIPROT_SEARCH_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Codex/lysozyme-pipeline"})
    with urlopen(request) as response:
        payload = response.read().decode("utf-8")
        headers = dict(response.headers.items())

    lines = payload.splitlines()
    reader = csv.DictReader(lines, delimiter="\t")
    rows = [dict(row) for row in reader]
    return rows, headers


def normalize_rows(rows: List[Dict[str, str]], family_tag: str, source_db: str) -> List[Dict[str, str]]:
    normalized = []
    for row in rows:
        sequence = (row.get("Sequence") or "").strip().upper()
        if not sequence:
            continue
        normalized.append(
            {
                "seq_id": (row.get("Entry") or "").strip(),
                "sequence": sequence,
                "length": int(row.get("Length") or len(sequence)),
                "source_db": source_db,
                "family_tag": family_tag,
                "description": (row.get("Protein names") or "").strip(),
                "entry_name": (row.get("Entry Name") or "").strip(),
                "organism": (row.get("Organism") or "").strip(),
                "gene_names": (row.get("Gene Names") or "").strip(),
            }
        )
    return normalized


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_id",
        "sequence",
        "length",
        "source_db",
        "family_tag",
        "description",
        "entry_name",
        "organism",
        "gene_names",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, query: str, headers: Dict[str, str], row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    metadata = {
        "query": query,
        "row_count": row_count,
        "uniprot_release": lower_headers.get("x-uniprot-release"),
        "uniprot_release_date": lower_headers.get("x-uniprot-release-date"),
        "api_deployment_date": lower_headers.get("x-api-deployment-date"),
        "x_total_results": lower_headers.get("x-total-results"),
        "source_url": UNIPROT_SEARCH_URL,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    rows, headers = fetch_tsv(args.query, args.size)
    normalized = normalize_rows(rows, family_tag=args.family_tag, source_db=args.source_db)

    output_csv = Path(args.output_csv)
    output_metadata_json = Path(args.output_metadata_json)

    write_csv(normalized, output_csv)
    write_metadata(output_metadata_json, args.query, headers, len(normalized))

    print(
        json.dumps(
            {
                "output_csv": str(output_csv.resolve()),
                "output_metadata_json": str(output_metadata_json.resolve()),
                "rows_written": len(normalized),
                "uniprot_release": {str(k).lower(): v for k, v in headers.items()}.get("x-uniprot-release"),
                "x_total_results": {str(k).lower(): v for k, v in headers.items()}.get("x-total-results"),
                "query": args.query,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
