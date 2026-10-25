"""Read-only benchmark reporting. Never deserialize saved estimators."""
from dataclasses import dataclass
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


class ArtifactError(ValueError):
    """Missing, invalid, or inconsistent reporting artifacts."""


@dataclass
class Benchmark:
    metrics: dict
    development: pd.DataFrame
    test: pd.DataFrame


def resolve_artifact_directory(project_root):
    """Prefer local experiment outputs; use the real public snapshot in a clean clone."""
    root = Path(project_root)
    local = root / "artifacts" / "model_benchmark"
    return local if (local / "metrics.json").is_file() else root / "report_data"


def load_benchmark(directory):
    directory = Path(directory)
    filenames = ("metrics.json", "development_oof_predictions.csv", "held_out_test_predictions.csv")
    for filename in filenames:
        if not (directory / filename).is_file():
            raise ArtifactError(f"Missing benchmark artifact: {directory / filename}")
    try:
        metrics = json.loads((directory / filenames[0]).read_text(encoding="utf-8"))
        development, test = (pd.read_csv(directory / f) for f in filenames[1:])
        if metrics["schema_version"] != 1:
            raise ValueError("Unsupported metrics schema")
        selected = metrics["selection"]["model"]
        if selected not in metrics["models"]:
            raise ValueError("Selected model absent from metrics")
        for frame, prefix in ((development, "dev"), (test, "test")):
            if len(frame) != metrics["split"][f"{prefix}_windows"]:
                raise ValueError(f"{prefix} prediction row count disagrees with metrics")
            required = {"malicious", "timestamp", "time_group"}
            required.update(f"score_{model}" for model in metrics["models"])
            if prefix == "test":
                required.update(f"flagged_{model}" for model in metrics["models"])
            if not required.issubset(frame.columns):
                raise ValueError(f"{prefix} prediction columns missing: {sorted(required - set(frame.columns))}")
            if not frame.malicious.isin([0, 1]).all() or frame.malicious.sum() != metrics["split"][f"{prefix}_positives"]:
                raise ValueError(f"{prefix} labels disagree with metrics")
            for model in metrics["models"]:
                if not np.isfinite(frame[f"score_{model}"]).all():
                    raise ValueError(f"Non-finite scores for {model}")
                if prefix == "test" and not frame[f"flagged_{model}"].isin([0, 1]).all():
                    raise ValueError(f"Invalid saved flags for {model}")
        return Benchmark(metrics, development, test)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"Cannot read benchmark artifacts: {exc}") from exc


def comparison_table(metrics):
    rows = []
    for model, values in metrics["models"].items():
        row = {"model": model}
        for key in ("mean_cv_average_precision", "std_cv_average_precision", "mean_train_average_precision", "train_cv_ap_gap"):
            row[key] = values[key]
        row.update({f"test_{k}": values["test_metrics"][k] for k in ("average_precision", "roc_auc", "precision", "recall", "f1")})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_cv_average_precision", ascending=False).reset_index(drop=True)


def diagnostics(bundle, model, split):
    """Recompute diagnostics from scores, using only the saved OOF threshold."""
    from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

    if split not in ("Development OOF", "Held-out test"):
        raise ValueError(f"Unknown split: {split}")
    frame = bundle.development if split == "Development OOF" else bundle.test
    threshold = bundle.metrics["models"][model]["threshold_from_dev_oof_max_f1"]
    labels, scores = frame.malicious, frame[f"score_{model}"]
    precision, recall, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "pr": pd.DataFrame({"recall": recall, "precision": precision}),
        "roc": pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr}),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion": confusion_matrix(labels, scores >= threshold, labels=[0, 1]).tolist(),
        "threshold": threshold,
        "prevalence": float(labels.mean()),
    }


def governance_lines(metrics):
    """Describe only recorded scope/policies, including extensible provenance fields."""
    dataset, split = metrics["dataset"], metrics["split"]
    lines = [
        f"Dataset: {dataset['description']}",
        f"Input flow rows: {dataset['input_rows']}; valid flows: {dataset['valid_flows']}.",
        f"Windows: {dataset['window_count']}; positive windows: {dataset['positive_windows']}.",
        f"Split: {split['method']}.",
        f"Temporal policy: {split['temporal_policy']}.",
        f"Recorded inner folds: {len(split['inner_folds'])}.",
        f"Selection: {metrics['selection']['primary_metric']}.",
        f"Threshold policy: {metrics['selection']['threshold_policy']}",
        "No retraining is performed by this report.",
    ]
    # Keep new source manifests/capture boundaries visible without assuming a
    # specific producer's provenance schema or silently omitting saved evidence.
    for label, record, displayed in (
        ("Dataset evidence / provenance", dataset, {"description", "input_rows", "valid_flows", "window_count", "positive_windows"}),
        ("Split evidence", split, {"method", "temporal_policy", "inner_folds"}),
    ):
        lines.extend(f"{label} — {key}: {json.dumps(value, ensure_ascii=False)}"
                     for key, value in record.items() if key not in displayed)
    lines.extend(f"- {caveat}" for caveat in metrics["caveats"])
    return lines


def build_report(bundle, include_comparison=True, include_learning=True, title="Cyber Threat Lab | Benchmark report"):
    """Portable Markdown and escaped standalone HTML with mandatory limitations."""
    metrics = bundle.metrics
    selected = metrics["selection"]["model"]
    model = metrics["models"][selected]
    test = model["test_metrics"]
    cm = test["confusion_matrix"]
    lines = [f"# {title}", "", "## Executive summary",
             f"Selected model: {selected}. Selection: {metrics['selection']['primary_metric']}; not selected using test performance.",
             f"Mean CV average precision: {model['mean_cv_average_precision']:.6f} (+/- {model['std_cv_average_precision']:.6f} fold SD).",
             f"Held-out test AP: {test['average_precision']:.6f}; precision: {test['precision']:.6f}; recall: {test['recall']:.6f}; F1: {test['f1']:.6f}.",
             f"Held-out counts: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.",
             f"Frozen development-OOF threshold: {model['threshold_from_dev_oof_max_f1']:.9f}.",
             "", "## Governance"]
    lines.extend(governance_lines(metrics))
    if include_comparison:
        lines.extend(["", "## Model comparison", "Ranked by mean grouped CV average precision; test is descriptive only.", "", "| Model | CV AP | Fold SD | Test AP |", "|---|---:|---:|---:|"])
        for row in comparison_table(metrics).to_dict("records"):
            lines.append(f"| {row['model']} | {row['mean_cv_average_precision']:.6f} | {row['std_cv_average_precision']:.6f} | {row['test_average_precision']:.6f} |")
    if include_learning:
        lines.extend(["", "## Learning curve", "Saved grouped training-size experiment; fold SD is not a confidence interval.", "", "| Training fraction | Mean train windows | Train AP | Validation AP | Validation SD |", "|---:|---:|---:|---:|---:|"])
        for row in metrics["learning_curve"]:
            lines.append(f"| {row['fraction_of_training_groups']:.2f} | {row['mean_train_windows']} | {row['mean_train_average_precision']:.6f} | {row['mean_validation_average_precision']:.6f} | {row['std_validation_average_precision']:.6f} |")
    markdown = "\n".join(lines)
    blocks = []
    in_table = False
    for line in lines:
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if not in_table:
                blocks.append("<table><thead><tr>" + "".join(f"<th>{escape(cell)}</th>" for cell in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                blocks.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            blocks.append("</tbody></table>")
            in_table = False
        if line.startswith("# "):
            blocks.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{escape(line[3:])}</h2>")
        else:
            blocks.append(f"<p>{escape(line)}</p>")
    if in_table:
        blocks.append("</tbody></table>")
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>{escape(title)}</title><style>body{{background:#ffffff;color:#0f172a;font:16px system-ui;max-width:1050px;margin:48px auto;padding:24px}}h1,h2{{color:#0f172a}}p{{white-space:pre-wrap;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #e2e8f0}}th{{background:#f8fafc;color:#334155}}</style></head><body>'
            + "\n".join(blocks) + "</body></html>")
    return markdown, html
