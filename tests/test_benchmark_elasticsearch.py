from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_publisher():
    path = ROOT / "scripts" / "publish_benchmark_metrics.py"
    assert path.exists()
    spec = importlib.util.spec_from_file_location("publish_benchmark_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_report():
    test = {
        "windows": 5, "positives": 1, "negatives": 4, "prevalence": 0.2,
        "threshold": 0.5, "average_precision": 0.9, "roc_auc": 0.8,
        "f1": 2 / 3, "precision": 0.5, "recall": 1.0,
        "confusion_matrix": {"tn": 3, "fp": 1, "fn": 0, "tp": 1},
    }
    return {
        "dataset": {"input_rows": 100, "valid_flows": 99, "window_count": 10,
                    "positive_windows": 2, "negative_windows": 8},
        "selection": {"model": "extra_trees", "selected_mean_cv_average_precision": 0.91},
        "models": {
            "extra_trees": {"mean_cv_average_precision": 0.91, "std_cv_average_precision": 0.02,
                            "mean_train_average_precision": 0.98, "train_cv_ap_gap": 0.07,
                            "threshold_from_dev_oof_max_f1": 0.5, "dev_oof_metrics": {"average_precision": 0.89},
                            "test_metrics": test},
            "prior_baseline": {"mean_cv_average_precision": 0.01, "std_cv_average_precision": 0.0,
                               "mean_train_average_precision": 0.01, "train_cv_ap_gap": 0.0,
                               "threshold_from_dev_oof_max_f1": 0.01, "dev_oof_metrics": {"average_precision": 0.01},
                               "test_metrics": test},
        },
        "learning_curve": [
            {"fraction_of_training_groups": 0.5, "mean_train_windows": 5,
             "mean_train_positive_windows": 1, "mean_train_average_precision": 0.8,
             "mean_validation_average_precision": 0.7, "std_validation_average_precision": 0.1},
            {"fraction_of_training_groups": 1.0, "mean_train_windows": 8,
             "mean_train_positive_windows": 2, "mean_train_average_precision": 0.98,
             "mean_validation_average_precision": 0.91, "std_validation_average_precision": 0.02},
        ],
    }


def test_build_documents_flattens_model_summary_learning_curve_and_confusion_metrics():
    publisher = load_publisher()
    docs = publisher.build_documents(fixture_report(), "abc123", "2026-10-25T09:00:00Z")
    by_id = {item["id"]: item["body"] for item in docs}
    assert set(by_id) == {
        "model-extra_trees", "model-prior_baseline", "summary",
        "learning-050", "learning-100",
    }
    assert by_id["model-extra_trees"]["is_selected"] is True
    assert by_id["model-extra_trees"]["test_fp"] == 1
    assert by_id["model-extra_trees"]["test_fn"] == 0
    assert by_id["summary"]["selected_model_name"] == "extra_trees"
    assert by_id["summary"]["selected_test_tn"] == 3
    assert by_id["summary"]["selected_test_tp"] == 1
    assert by_id["learning-050"]["training_fraction"] == 0.5
    assert by_id["learning-100"]["cv_ap"] == 0.91


def test_mapping_keeps_kibana_aggregation_fields_keyword_and_numeric():
    properties = load_publisher().INDEX_MAPPING["mappings"]["properties"]
    assert properties["@timestamp"]["type"] == "date"
    assert properties["model_name"]["type"] == "keyword"
    assert properties["doc_type"]["type"] == "keyword"
    assert properties["mean_cv_average_precision"]["type"] == "float"
    assert properties["test_fp"]["type"] == "integer"
    assert properties["training_fraction"]["type"] == "float"
