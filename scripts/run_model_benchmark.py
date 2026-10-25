from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cyberlab.benchmark import run_model_benchmark


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train and compare ML/DL detectors with grouped validation and charts")
    parser.add_argument("--input", default="data/raw/Thursday-WorkingHours.pcap_ISCX.csv")
    parser.add_argument("--output", default="artifacts/model_benchmark")
    parser.add_argument("--max-rows", type=int, default=None, help="Development-only truncation; omit to use all data")
    parser.add_argument("--group-minutes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        result = run_model_benchmark(args.input, args.output, max_rows=args.max_rows,
                                     group_minutes=args.group_minutes, random_state=args.seed)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"model benchmark: {exc}", file=sys.stderr)
        return 1

    selected = result["selection"]["model"]
    summary = {
        "output": str(Path(args.output).resolve()),
        "flows": result["dataset"]["valid_flows"],
        "windows": result["dataset"]["window_count"],
        "positive_windows": result["dataset"]["positive_windows"],
        "selected_model": selected,
        "mean_grouped_cv_average_precision": result["selection"]["selected_mean_cv_average_precision"],
        "held_out_test": result["held_out_test"]["metrics"],
        "report": str((Path(args.output) / result["report_file"]).resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
