from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest


def synthetic_windows(group_count: int = 25) -> pd.DataFrame:
    rows = []
    positive_groups = {1, 3, 5, 8, 11, 14, 17, 20, 23, 24}
    origin = pd.Timestamp("2017-07-06T12:00:00Z")
    for group in range(group_count):
        start = origin + timedelta(minutes=group * 5)
        for offset in range(4):
            rows.append({
                "timestamp": start + timedelta(seconds=offset * 60),
                "source_ip": f"192.0.2.{group + 1}",
                "window_id": f"{group}-{offset}",
                "malicious": group in positive_groups and offset == 0,
                "reference_label": "Web Attack" if group in positive_groups and offset == 0 else "BENIGN",
                "flow_count": float(1 + group % 3),
            })
    return pd.DataFrame(rows)


def test_grouped_benchmark_split_keeps_time_buckets_disjoint_and_has_both_classes():
    from cyberlab.benchmark import create_grouped_splits

    windows = synthetic_windows()
    split = create_grouped_splits(windows, group_minutes=5, random_state=42)
    groups = split["groups"]
    y = windows["malicious"].to_numpy()

    assert not set(groups[split["dev_indices"]]) & set(groups[split["test_indices"]])
    assert y[split["test_indices"]].any() and (~y[split["test_indices"]]).any()
    assert len(split["inner_folds"]) == 4
    for fold in split["inner_folds"]:
        assert not set(groups[fold["train_indices"]]) & set(groups[fold["validation_indices"]])
        assert y[fold["validation_indices"]].any()
        assert (~y[fold["validation_indices"]]).any()


def test_grouped_split_declares_nonchronological_generalization_scope():
    from cyberlab.benchmark import create_grouped_splits

    split = create_grouped_splits(synthetic_windows(), group_minutes=5, random_state=42)
    assert split["temporal_policy"] == "shuffled_group_holdout_not_forward_in_time"


def test_learning_curve_subsampling_keeps_groups_intact_and_preserves_classes():
    import numpy as np
    from cyberlab.benchmark import sample_group_indices

    windows = synthetic_windows()
    y = windows["malicious"].to_numpy(dtype=bool)
    groups = windows["timestamp"].dt.floor("5min").astype(str).to_numpy()
    indices = np.arange(len(windows))
    sample = sample_group_indices(indices, groups, y, fraction=0.5, random_state=7)
    assert y[sample].any() and (~y[sample]).any()
    for group in set(groups[sample]):
        assert set(np.flatnonzero(groups == group)) <= set(sample)
    assert len(sample_group_indices(indices, groups, y, fraction=1.0, random_state=7)) == len(indices)


def test_validation_threshold_uses_f1_and_is_deterministic():
    from cyberlab.benchmark import choose_f1_threshold

    threshold = choose_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert threshold == pytest.approx(0.8)


def test_classification_metrics_include_imbalance_aware_and_threshold_metrics():
    from cyberlab.benchmark import evaluate_scores

    result = evaluate_scores([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], threshold=0.5)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["average_precision"] == pytest.approx(0.8333333333)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_html_report_contains_metrics_and_svg_charts():
    from cyberlab.benchmark import render_html_report

    cm = {"tn": 1, "fp": 0, "fn": 1, "tp": 1}
    metric = {"average_precision": 0.83, "roc_auc": 0.75, "f1": 0.67,
              "precision": 1.0, "recall": 0.5, "balanced_accuracy": 0.75,
              "threshold": 0.5, "confusion_matrix": cm}
    model = {"mean_cv_average_precision": 0.8, "std_cv_average_precision": 0.05,
             "mean_train_average_precision": 0.9, "train_cv_ap_gap": 0.1,
             "dev_oof_metrics": metric,
             "test_metrics": metric, "cv_folds": []}
    report = {"dataset": {"window_count": 4, "positive_windows": 2, "negative_windows": 2,
                          "input_rows": 20, "valid_flows": 18, "positive_window_labels": {"Attack": 2}},
              "split": {"method": "grouped", "group_minutes": 5,
                        "dev_windows": 3, "dev_positives": 1,
                        "dev_groups": 2, "test_windows": 1, "test_positives": 1,
                        "test_groups": 1, "group_overlap_test_development": False,
                        "inner_folds": []},
              "selection": {"model": "logistic_regression", "primary_metric": "mean grouped CV AP",
                            "selected_mean_cv_average_precision": 0.8},
              "models": {"logistic_regression": model},
              "learning_curve": [
                  {"fraction_of_training_groups": 0.25, "mean_train_windows": 10, "mean_train_positive_windows": 2,
                   "mean_train_average_precision": 0.8, "mean_validation_average_precision": 0.7,
                   "std_validation_average_precision": 0.05},
                  {"fraction_of_training_groups": 1.0, "mean_train_windows": 40, "mean_train_positive_windows": 8,
                   "mean_train_average_precision": 0.9, "mean_validation_average_precision": 0.8,
                   "std_validation_average_precision": 0.03},
              ],
              "held_out_test": {"selected_model": "logistic_regression", "metrics": metric,
                                "model_test_metrics": {"logistic_regression": metric}},
              "caveats": ["Exploratory only"]}
    html = render_html_report(report, [0, 1, 0], {"logistic_regression": [0.1, 0.9, 0.4]},
                              [1], {"logistic_regression": [0.9]})
    assert "logistic regression" in html
    assert "Exploratory only" in html
    assert "Learning-size curve" in html
    assert 'id="live-model-select"' in html
    assert 'id="live-metric-select"' in html
    assert 'id="dynamic-metrics-chart"' in html
    assert 'fetch("metrics.json"' in html
    assert "setInterval(refreshLiveMetrics" in html
    assert html.count("<svg") >= 3


def test_run_model_benchmark_exports_artifacts_and_a_renderable_report(tmp_path):
    from datetime import datetime, timedelta
    from cyberlab.benchmark import run_model_benchmark

    rows = []
    origin = datetime(2017, 7, 6, 9, 0)
    positive_groups = {1, 3, 5, 8, 11, 14, 17, 20, 23, 24}
    for group in range(25):
        for minute in range(4):
            stamp = origin + timedelta(minutes=group * 5 + minute)
            rows.append({
                "Source IP": "172.16.0.1", "Destination IP": f"192.0.2.{group % 5 + 1}",
                "Source Port": 1000 + minute, "Destination Port": 80 + group % 5,
                "Protocol": 6, "Timestamp": stamp.strftime("%d/%m/%Y %H:%M:%S"),
                "Flow Duration": 1000 + group, "Total Fwd Packets": 2,
                "Total Backward Packets": 1, "Total Length of Fwd Packets": 200 + group,
                "Total Length of Bwd Packets": 100, "Flow Bytes/s": 300 + group,
                "Label": "Web Attack - XSS" if group in positive_groups and minute == 0 else "BENIGN",
            })
    source = tmp_path / "flows.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "benchmark"
    report = run_model_benchmark(source, output)

    assert report["dataset"]["valid_flows"] == 100
    assert report["dataset"]["window_count"] == 100
    assert report["dataset"]["positive_windows"] == len(positive_groups)
    assert report["split"]["group_overlap_test_development"] is False
    assert report["held_out_test"]["metrics"]["positives"] > 0
    assert report["selection"]["model"] in report["models"]
    assert [row["fraction_of_training_groups"] for row in report["learning_curve"]] == [0.25, 0.5, 0.75, 1.0]
    for name in ["metrics.json", "report.html", "cv_results.csv", "development_oof_predictions.csv",
                 "held_out_test_predictions.csv", "models.joblib"]:
        assert (output / name).is_file()
    assert "<svg" in (output / "report.html").read_text(encoding="utf-8")


def test_best_model_is_selected_by_mean_grouped_validation_average_precision():
    from cyberlab.benchmark import select_best_model

    records = {
        "forest": {"cv_folds": [{"average_precision": 0.65}, {"average_precision": 0.75}]},
        "mlp": {"cv_folds": [{"average_precision": 0.70}, {"average_precision": 0.72}]},
    }
    assert select_best_model(records) == "mlp"
