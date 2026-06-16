"""
Fetch a broad protein family pool from UniRef and normalize it for
prepare_family_benchmark.py.

This is intended for larger, less tightly controlled pools than the
Swiss-Prot-only UniProt fetcher. The default query targets a broad UniRef50
lysozyme-like space:

    identity:0.5 AND lysozyme AND length:[60 TO 400]

That query is intentionally broad enough to yield a large pool for downstream
scoring and low-tail benchmark construction.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen


UNIREF_SEARCH_URL = "https://rest.uniprot.org/uniref/search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a normalized family pool from UniRef.")
    parser.add_argument(
        "--query",
        default="identity:0.5 AND lysozyme AND length:[60 TO 400]",
        help="UniRef query string.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/lysozyme_uniref50/lysozyme_uniref50_family_input.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_metadata_json",
        default="data/lysozyme_uniref50/lysozyme_uniref50_family_input.metadata.json",
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--family_tag",
        default="lysozyme_uniref50_broad",
        help="family_tag value written to the output CSV.",
    )
    parser.add_argument(
        "--source_db",
        default="UniRef50",
        help="source_db value written to the output CSV.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Page size for UniRef fetch.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap on the number of rows to fetch. Useful for smoke tests.",
    )
    return parser.parse_args()


def extract_next_cursor(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
    if not match:
        return None
    next_url = match.group(1)
    query = parse_qs(urlparse(next_url).query)
    cursor_values = query.get("cursor")
    return cursor_values[0] if cursor_values else None


def fetch_page(query: str, size: int, cursor: Optional[str]) -> Tuple[Dict[str, object], Dict[str, str]]:
    params = {
        "query": query,
        "format": "json",
        "size": size,
    }
    if cursor:
        params["cursor"] = cursor
    url = f"{UNIREF_SEARCH_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Codex/lysozyme-uniref50-pipeline"})
    with urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = dict(response.headers.items())
    return payload, headers


def fetch_all(query: str, size: int, max_records: Optional[int]) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    results: List[Dict[str, object]] = []
    cursor: Optional[str] = None
    final_headers: Dict[str, str] = {}

    while True:
        payload, headers = fetch_page(query=query, size=size, cursor=cursor)
        final_headers = headers
        page_results = payload.get("results", [])
        if not page_results:
            break

        results.extend(page_results)
        if max_records is not None and len(results) >= max_records:
            results = results[:max_records]
            break

        cursor = extract_next_cursor(headers.get("Link") or headers.get("link"))
        if not cursor:
            break

    return results, final_headers


def normalize_rows(rows: Iterable[Dict[str, object]], family_tag: str, source_db: str) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for row in rows:
        rep = row.get("representativeMember") or {}
        seq = ((rep.get("sequence") or {}).get("value") or "").strip().upper()
        if not seq:
            continue

        accessions = rep.get("accessions") or []
        accession = accessions[0] if accessions else ""
        normalized.append(
            {
                "seq_id": str(row.get("id") or "").strip(),
                "sequence": seq,
                "length": int((rep.get("sequence") or {}).get("length") or len(seq)),
                "source_db": source_db,
                "family_tag": family_tag,
                "description": str(rep.get("proteinName") or row.get("name") or "").strip(),
                "entry_name": str(rep.get("memberId") or "").strip(),
                "organism": str(rep.get("organismName") or "").strip(),
                "gene_names": "",
                "cluster_name": str(row.get("name") or "").strip(),
                "entry_type": str(row.get("entryType") or "").strip(),
                "member_count": int(row.get("memberCount") or 0),
                "organism_count": int(row.get("organismCount") or 0),
                "representative_accession": accession,
                "seed_id": str(row.get("seedId") or "").strip(),
                "uniref90_id": str(rep.get("uniref90Id") or "").strip(),
                "uniparc_id": str(rep.get("uniparcId") or "").strip(),
            }
        )
    return normalized


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
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
        "cluster_name",
        "entry_type",
        "member_count",
        "organism_count",
        "representative_accession",
        "seed_id",
        "uniref90_id",
        "uniparc_id",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    path: Path,
    query: str,
    headers: Dict[str, str],
    row_count: int,
    size: int,
    max_records: Optional[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    metadata = {
        "query": query,
        "row_count": row_count,
        "page_size": size,
        "max_records": max_records,
        "uniprot_release": lower_headers.get("x-uniprot-release"),
        "uniprot_release_date": lower_headers.get("x-uniprot-release-date"),
        "api_deployment_date": lower_headers.get("x-api-deployment-date"),
        "x_total_results": lower_headers.get("x-total-results"),
        "source_url": UNIREF_SEARCH_URL,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    rows, headers = fetch_all(query=args.query, size=args.size, max_records=args.max_records)
    normalized = normalize_rows(rows, family_tag=args.family_tag, source_db=args.source_db)

    output_csv = Path(args.output_csv)
    output_metadata_json = Path(args.output_metadata_json)
    write_csv(normalized, output_csv)
    write_metadata(
        output_metadata_json,
        query=args.query,
        headers=headers,
        row_count=len(normalized),
        size=args.size,
        max_records=args.max_records,
    )

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
