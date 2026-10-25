"""CLI: run training/evaluation, score fresh flows, and report labeled predictions."""
import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import evaluate_predictions, predict_pipeline, run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="CICIDS2017 source-IP minute-window threat detection")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="train detectors, evaluate chronological splits, and write artifacts")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--max-rows", type=int, default=None)
    run.add_argument("--publish", action="store_true")
    run.add_argument("--es-url", default=os.environ.get("ES_URL", os.environ.get("ELASTICSEARCH_URL", "http://localhost:19200")))

    predict = sub.add_parser("predict", help="apply a saved model to new labeled or unlabeled flow CSVs")
    predict.add_argument("--input", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--max-rows", type=int, default=None)

    evaluate = sub.add_parser("evaluate", help="evaluate prediction NDJSON against its reference labels")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_pipeline(args.input, args.output, args.max_rows)
            if args.publish:
                from .publish import publish_artifacts
                manifest_path = Path(args.output) / "manifest.json"
                try:
                    manifest["publication"] = publish_artifacts(args.output, args.es_url)
                except Exception as exc:
                    manifest["publication"] = {"status": "failed", "error": str(exc)}
                    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
                    raise
                manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({"output": args.output, "counts": manifest["counts"], "detectors": manifest["detector_status"], "publication": manifest["publication"]}, indent=2))
        elif args.command == "predict":
            result = predict_pipeline(args.input, args.model, args.output, args.max_rows)
            print(json.dumps(result, indent=2))
        else:
            report = evaluate_predictions(args.predictions)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({"output": str(output_path), "detectors": sorted(report) if isinstance(report, dict) else []}, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"cyberlab: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
