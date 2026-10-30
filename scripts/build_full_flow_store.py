"""Build a bounded-memory, indexed fact store for all verified benchmark flows."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cyberlab.expanded_training import input_inventory
from cyberlab.flow_store import build_flow_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.getenv("CYBERLAB_FLOW_STORE", str(ROOT / "artifacts/expanded_preparation/full_flows.sqlite")))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    entries = input_inventory(ROOT)
    print(f"Verified {len(entries)} manifest-listed source CSVs; processing all benchmark inputs.", flush=True)
    result = build_flow_store(entries, args.output, chunk_size=args.chunk_size)
    print(json.dumps({key: result[key] for key in (
        "status", "input_rows", "valid_flows", "invalid_rows", "duplicate_rows", "blank_rows",
        "database_record_count", "dataset_counts", "database_bytes", "integrity_check", "timestamp_policy",
    )}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
