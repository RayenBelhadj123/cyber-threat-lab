from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

INDEX_NAME = "cyber-benchmark-v1"


def _field_types():
    return {
        "@timestamp": "date", "doc_type": "keyword", "run_id": "keyword",
        "model_name": "keyword", "selected_model_name": "keyword", "is_selected": "boolean",
        "dataset_input_rows": "long", "dataset_valid_flows": "long",
        "dataset_window_count": "integer", "dataset_positive_windows": "integer",
        "dataset_negative_windows": "integer", "candidate_model_count": "integer",
        "test_windows": "integer", "test_positive_windows": "integer",
        "test_negative_windows": "integer", "test_prevalence": "float",
        "mean_cv_average_precision": "float", "std_cv_average_precision": "float",
        "mean_train_average_precision": "float", "train_cv_ap_gap": "float",
        "dev_oof_average_precision": "float", "test_average_precision": "float",
        "test_roc_auc": "float", "test_f1": "float", "test_precision": "float",
        "test_recall": "float", "test_threshold": "float",
        "threshold_from_dev_oof_max_f1": "float", "test_tn": "integer",
        "test_fp": "integer", "test_fn": "integer", "test_tp": "integer",
        "selected_mean_cv_average_precision": "float",
        "selected_test_average_precision": "float", "selected_test_roc_auc": "float",
        "selected_test_f1": "float", "selected_test_precision": "float",
        "selected_test_recall": "float", "selected_test_threshold": "float",
        "selected_test_tn": "integer", "selected_test_fp": "integer",
        "selected_test_fn": "integer", "selected_test_tp": "integer",
        "training_fraction": "float", "training_fraction_percent": "integer",
        "mean_train_windows": "integer", "mean_train_positive_windows": "integer",
        "train_ap": "float", "cv_ap": "float", "cv_ap_std": "float",
    }


INDEX_MAPPING = {
    "mappings": {
        "dynamic": "strict",
        "properties": {name: {"type": field_type} for name, field_type in _field_types().items()},
    }
}


def _dataset_fields(dataset):
    return {
        "dataset_input_rows": int(dataset["input_rows"]),
        "dataset_valid_flows": int(dataset["valid_flows"]),
        "dataset_window_count": int(dataset["window_count"]),
        "dataset_positive_windows": int(dataset["positive_windows"]),
        "dataset_negative_windows": int(dataset["negative_windows"]),
    }


def build_documents(report, run_id: str, timestamp: str):
    """Build deterministic ES document IDs and strict-mapping-safe payloads."""
    dataset = report["dataset"]
    selection = report["selection"]
    selected_name = selection["model"]
    models = report["models"]
    candidate_count = len(models)
    common = {
        "@timestamp": timestamp,
        "run_id": run_id,
        "candidate_model_count": candidate_count,
        **_dataset_fields(dataset),
    }
    documents = []
    for name, record in sorted(models.items()):
        test = record["test_metrics"]
        confusion = test["confusion_matrix"]
        body = {
            **common,
            "doc_type": "benchmark_model",
            "model_name": name,
            "selected_model_name": selected_name,
            "is_selected": name == selected_name,
            "test_windows": int(test["windows"]),
            "test_positive_windows": int(test["positives"]),
            "test_negative_windows": int(test["negatives"]),
            "test_prevalence": float(test["prevalence"]),
            "mean_cv_average_precision": float(record["mean_cv_average_precision"]),
            "std_cv_average_precision": float(record["std_cv_average_precision"]),
            "mean_train_average_precision": float(record["mean_train_average_precision"]),
            "train_cv_ap_gap": float(record["train_cv_ap_gap"]),
            "dev_oof_average_precision": float(record["dev_oof_metrics"]["average_precision"]),
            "test_average_precision": float(test["average_precision"]),
            "test_roc_auc": float(test["roc_auc"]),
            "test_f1": float(test["f1"]),
            "test_precision": float(test["precision"]),
            "test_recall": float(test["recall"]),
            "test_threshold": float(test["threshold"]),
            "threshold_from_dev_oof_max_f1": float(record["threshold_from_dev_oof_max_f1"]),
            "test_tn": int(confusion["tn"]), "test_fp": int(confusion["fp"]),
            "test_fn": int(confusion["fn"]), "test_tp": int(confusion["tp"]),
        }
        documents.append({"id": f"model-{name}", "body": body})

    selected_record = models[selected_name]
    selected_test = selected_record["test_metrics"]
    selected_cm = selected_test["confusion_matrix"]
    summary = {
        **common,
        "doc_type": "benchmark_summary",
        "selected_model_name": selected_name,
        "selected_mean_cv_average_precision": float(selection["selected_mean_cv_average_precision"]),
        "selected_test_average_precision": float(selected_test["average_precision"]),
        "selected_test_roc_auc": float(selected_test["roc_auc"]),
        "selected_test_f1": float(selected_test["f1"]),
        "selected_test_precision": float(selected_test["precision"]),
        "selected_test_recall": float(selected_test["recall"]),
        "selected_test_threshold": float(selected_test["threshold"]),
        "test_windows": int(selected_test["windows"]),
        "test_positive_windows": int(selected_test["positives"]),
        "test_negative_windows": int(selected_test["negatives"]),
        "test_prevalence": float(selected_test["prevalence"]),
        "selected_test_tn": int(selected_cm["tn"]),
        "selected_test_fp": int(selected_cm["fp"]),
        "selected_test_fn": int(selected_cm["fn"]),
        "selected_test_tp": int(selected_cm["tp"]),
    }
    documents.append({"id": "summary", "body": summary})

    for row in report.get("learning_curve", []):
        fraction = float(row["fraction_of_training_groups"])
        percent = int(round(100 * fraction))
        body = {
            "@timestamp": timestamp,
            "doc_type": "learning_curve",
            "run_id": run_id,
            "selected_model_name": selected_name,
            "training_fraction": fraction,
            "training_fraction_percent": percent,
            "mean_train_windows": int(row["mean_train_windows"]),
            "mean_train_positive_windows": int(row["mean_train_positive_windows"]),
            "train_ap": float(row["mean_train_average_precision"]),
            "cv_ap": float(row["mean_validation_average_precision"]),
            "cv_ap_std": float(row["std_validation_average_precision"]),
            "dataset_window_count": int(dataset["window_count"]),
            "dataset_positive_windows": int(dataset["positive_windows"]),
        }
        documents.append({"id": f"learning-{percent:03d}", "body": body})
    return documents


def _ensure_index(session, base_url: str):
    index_url = f"{base_url}/{INDEX_NAME}"
    response = session.get(index_url, timeout=30)
    if response.status_code == 404:
        response = session.put(index_url, json=INDEX_MAPPING, timeout=60)
        response.raise_for_status()
        return
    response.raise_for_status()
    current = response.json()[INDEX_NAME]["mappings"].get("properties", {})
    for name, expected in INDEX_MAPPING["mappings"]["properties"].items():
        actual_type = current.get(name, {}).get("type")
        if actual_type != expected["type"]:
            raise RuntimeError(f"Index mapping mismatch for {name}: expected {expected['type']}, got {actual_type}")


def publish(metrics_path, elasticsearch_url: str):
    metrics_path = Path(metrics_path)
    raw = metrics_path.read_bytes()
    report = json.loads(raw)
    run_id = hashlib.sha256(raw).hexdigest()[:16]
    timestamp = datetime.fromtimestamp(metrics_path.stat().st_mtime, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    documents = build_documents(report, run_id, timestamp)
    base_url = elasticsearch_url.rstrip("/")
    with requests.Session() as session:
        _ensure_index(session, base_url)
        for item in documents:
            doc_url = f"{base_url}/{INDEX_NAME}/_doc/{quote(item['id'], safe='')}"
            response = session.put(doc_url, json=item["body"], timeout=60)
            response.raise_for_status()
        session.post(f"{base_url}/{INDEX_NAME}/_refresh", timeout=60).raise_for_status()
        for item in documents:
            response = session.get(f"{base_url}/{INDEX_NAME}/_doc/{quote(item['id'], safe='')}", timeout=30)
            response.raise_for_status()
            if response.json().get("_source") != item["body"]:
                raise RuntimeError(f"Elasticsearch read-back mismatch for {item['id']}")
    return {"index": INDEX_NAME, "documents": len(documents), "run_id": run_id,
            "selected_model": report["selection"]["model"], "timestamp": timestamp}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Publish benchmark metrics to Elasticsearch for Kibana dashboards")
    parser.add_argument("--metrics", default="artifacts/model_benchmark/metrics.json")
    parser.add_argument("--elasticsearch-url", default=os.environ.get("ELASTICSEARCH_URL", "http://localhost:19200"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = publish(args.metrics, args.elasticsearch_url)
    except (OSError, ValueError, requests.RequestException, RuntimeError) as exc:
        print(f"benchmark publish: {exc}", file=sys.stderr)
        return 1
    output = Path(args.metrics).parent / "elasticsearch_publish.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
