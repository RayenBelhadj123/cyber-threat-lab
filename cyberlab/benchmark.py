"""Grouped, leakage-aware model benchmark for the CICIDS2017 window data."""
from __future__ import annotations

import statistics
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve


def create_grouped_splits(
    windows,
    *,
    group_minutes: int = 5,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Hold out whole time buckets; return one outer test fold and inner CV folds."""
    if group_minutes < 1:
        raise ValueError("group_minutes must be at least 1")
    if outer_splits < 3 or inner_splits < 2:
        raise ValueError("Need at least 3 outer folds and 2 inner folds")
    y = windows["malicious"].to_numpy(dtype=bool)
    if np.unique(y).size != 2:
        raise ValueError("Grouped benchmark requires positive and negative windows")
    groups = windows["timestamp"].dt.floor(f"{group_minutes}min").astype(str).to_numpy()
    if len(np.unique(groups)) < outer_splits:
        raise ValueError("Not enough distinct time buckets for the requested outer folds")

    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    outer_folds = list(outer.split(np.zeros(len(windows)), y, groups))
    dev_indices, test_indices = outer_folds[0]
    if np.unique(y[test_indices]).size != 2:
        raise ValueError("Outer test fold is missing a class; use more attack time buckets")

    dev_y = y[dev_indices]
    dev_groups = groups[dev_indices]
    if len(np.unique(dev_groups)) < inner_splits:
        raise ValueError("Not enough development time buckets for the requested inner folds")
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + 1)
    inner_folds = []
    for train_local, validation_local in inner.split(np.zeros(len(dev_indices)), dev_y, dev_groups):
        train_indices = dev_indices[train_local]
        validation_indices = dev_indices[validation_local]
        if np.unique(y[validation_indices]).size != 2:
            raise ValueError("An inner validation fold is missing a class; use more attack time buckets")
        inner_folds.append({"train_indices": train_indices, "validation_indices": validation_indices})

    return {
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "inner_folds": inner_folds,
        "groups": groups,
        "group_minutes": group_minutes,
        "random_state": random_state,
        "outer_test_fold": 0,
        "temporal_policy": "shuffled_group_holdout_not_forward_in_time",
    }


def choose_f1_threshold(y_true, scores) -> float:
    """Choose a score threshold by maximum F1 on development OOF scores."""
    y = np.asarray(y_true, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if len(y) != len(values) or not len(y):
        raise ValueError("y_true and scores must be nonempty arrays with equal length")
    if np.unique(y).size != 2:
        raise ValueError("Threshold selection requires positive and negative outcomes")
    precision, recall, thresholds = precision_recall_curve(y, values)
    if not len(thresholds):
        return float(np.median(values))
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[int(np.argmax(f1))])


def evaluate_scores(y_true, scores, threshold: float) -> dict[str, Any]:
    """Report both threshold-free ranking and thresholded detection metrics."""
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    y = np.asarray(y_true, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if len(y) != len(values) or not len(y):
        raise ValueError("y_true and scores must be nonempty arrays with equal length")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    predicted = values >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[False, True]).ravel()
    has_both = bool(y.any() and (~y).any())
    return {
        "windows": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((~y).sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if has_both else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "average_precision": float(average_precision_score(y, values)) if y.any() else None,
        "roc_auc": float(roc_auc_score(y, values)) if has_both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_best_model(records: dict[str, dict[str, Any]]) -> str:
    """Select by mean inner-CV average precision, with a stable name tie-break."""
    candidates = []
    for name, record in records.items():
        folds = record.get("cv_folds", [])
        values = [float(fold["average_precision"]) for fold in folds
                  if fold.get("average_precision") is not None]
        if values:
            candidates.append((statistics.mean(values), name))
    if not candidates:
        raise ValueError("No model has available cross-validation average precision")
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]


MODEL_NAMES = (
    "prior_baseline",
    "logistic_regression",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "isolation_forest",
    "pytorch_mlp",
)


def _score_model(name: str, model, x: np.ndarray) -> np.ndarray:
    if name == "prior_baseline":
        return np.full(len(x), float(model["probability"]), dtype=float)
    if name == "isolation_forest":
        return -model.score_samples(x)
    if name == "pytorch_mlp":
        import torch
        bundle = model
        transformed = np.log1p(np.clip(x, 0, None))
        scaled = bundle["scaler"].transform(transformed).astype(np.float32)
        network = bundle["network"]
        network.eval()
        with torch.no_grad():
            return torch.sigmoid(network(torch.from_numpy(scaled))).cpu().numpy().reshape(-1)
    return model.predict_proba(x)[:, 1]


def _fit_mlp(x_train, y_train, x_validation, y_validation, seed, fixed_epochs=None):
    import copy
    import torch
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    x_train_log = np.log1p(np.clip(x_train, 0, None))
    scaler = StandardScaler().fit(x_train_log)
    x_train_scaled = scaler.transform(x_train_log).astype(np.float32)
    tx = torch.from_numpy(x_train_scaled)
    ty = torch.from_numpy(np.asarray(y_train, dtype=np.float32).reshape(-1, 1))

    network = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1),
    )
    positives = max(float(np.sum(y_train)), 1.0)
    negatives = max(float(len(y_train) - np.sum(y_train)), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.001)
    batch_size = min(512, len(tx))
    max_epochs = int(fixed_epochs) if fixed_epochs is not None else 80
    best_state = None
    best_ap = -1.0
    best_epoch = max_epochs
    stale_epochs = 0
    patience = 10

    for epoch in range(max_epochs):
        network.train()
        order = torch.randperm(len(tx))
        for batch in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(tx[batch]), ty[batch])
            loss.backward()
            optimizer.step()

        if fixed_epochs is not None or x_validation is None:
            continue
        network.eval()
        x_val_log = np.log1p(np.clip(x_validation, 0, None))
        x_val_scaled = scaler.transform(x_val_log).astype(np.float32)
        with torch.no_grad():
            val_scores = torch.sigmoid(network(torch.from_numpy(x_val_scaled))).cpu().numpy().reshape(-1)
        val_ap = float(average_precision_score(y_validation, val_scores))
        if val_ap > best_ap + 1e-6:
            best_ap = val_ap
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    bundle = {
        "kind": "pytorch_mlp",
        "network": network,
        "scaler": scaler,
        "features": list(__import__("cyberlab.pipeline", fromlist=["FEATURES"]).FEATURES),
        "input_transform": "log1p_nonnegative_then_standard_scale",
        "epochs": int(best_epoch),
        "seed": int(seed),
    }
    train_scores = _score_model("pytorch_mlp", bundle, x_train)
    val_scores = None if x_validation is None else _score_model("pytorch_mlp", bundle, x_validation)
    return bundle, train_scores, val_scores, {"epochs": int(best_epoch), "best_validation_ap": None if best_ap < 0 else float(best_ap)}


def fit_candidate(name, x_train, y_train, x_validation, y_validation=None, *, seed=42, fixed_epochs=None):
    """Fit one candidate and return in-sample and validation scores plus fit metadata."""
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  IsolationForest, RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=bool)
    x_validation = None if x_validation is None else np.asarray(x_validation, dtype=float)
    if name == "pytorch_mlp":
        return _fit_mlp(x_train, y_train, x_validation, y_validation, seed, fixed_epochs)
    if name == "prior_baseline":
        model = {"kind": name, "probability": float(y_train.mean())}
    elif name == "logistic_regression":
        model = make_pipeline(
            FunctionTransformer(np.log1p, validate=False),
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000,
                               solver="liblinear", random_state=seed),
        )
        model.fit(x_train, y_train)
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=120, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=2.0, random_state=seed,
        )
        model.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=1,
        ).fit(x_train, y_train)
    elif name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=1,
        ).fit(x_train, y_train)
    elif name == "isolation_forest":
        benign = ~y_train
        if int(benign.sum()) < 2:
            raise ValueError("Isolation Forest requires at least two benign training windows")
        model = IsolationForest(n_estimators=150, contamination="auto", random_state=seed, n_jobs=1)
        model.fit(x_train[benign])
    else:
        raise ValueError(f"Unknown model: {name}")

    train_scores = _score_model(name, model, x_train)
    val_scores = None if x_validation is None else _score_model(name, model, x_validation)
    return model, train_scores, val_scores, {}


def sample_group_indices(indices, groups, y, fraction: float, random_state: int = 42) -> np.ndarray:
    """Subsample whole time groups while retaining at least one group of each class."""
    indices = np.asarray(indices, dtype=int)
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y, dtype=bool)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return np.sort(indices)

    group_values = np.unique(groups[indices])
    positive_groups = [group for group in group_values if y[indices[groups[indices] == group]].any()]
    negative_groups = [group for group in group_values if (~y[indices[groups[indices] == group]]).any()]
    if not positive_groups or not negative_groups:
        raise ValueError("Training groups must contain both classes")
    rng = np.random.default_rng(random_state)
    n_positive = min(len(positive_groups), max(1, int(np.ceil(fraction * len(positive_groups)))))
    n_negative = min(len(negative_groups), max(1, int(np.ceil(fraction * len(negative_groups)))))
    chosen = set(rng.choice(positive_groups, size=n_positive, replace=False).tolist())
    chosen.update(rng.choice(negative_groups, size=n_negative, replace=False).tolist())
    sampled = indices[np.isin(groups[indices], list(chosen))]

    # Add a whole group if overlap between positive/negative groups left one class absent.
    for target_is_positive in (True, False):
        if np.any(y[sampled] == target_is_positive):
            continue
        candidates = [group for group in group_values if np.any(y[indices[groups[indices] == group]] == target_is_positive)]
        if not candidates:
            raise ValueError("Could not preserve both classes in the group sample")
        chosen.add(candidates[0])
        sampled = indices[np.isin(groups[indices], list(chosen))]
    return np.sort(sampled)


def run_model_benchmark(input_path, output_dir, *, max_rows=None, group_minutes=5, random_state=42):
    """Train candidates with grouped inner CV, select by CV AP, and test once on held-out groups."""
    import importlib.metadata
    import json
    import platform
    import joblib
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    from .pipeline import FEATURES, aggregate_windows, load_flows

    flows, lineage = load_flows(input_path, max_rows=max_rows)
    windows = aggregate_windows(flows)
    split = create_grouped_splits(windows, group_minutes=group_minutes, random_state=random_state)
    x = windows[FEATURES].astype(float).to_numpy()
    y = windows["malicious"].to_numpy(dtype=bool)
    dev = split["dev_indices"]
    test = split["test_indices"]
    records = {name: {"cv_folds": [], "fold_epochs": [], "oof_scores": np.full(len(windows), np.nan)} for name in MODEL_NAMES}

    for fold_number, fold in enumerate(split["inner_folds"]):
        train_idx = fold["train_indices"]
        val_idx = fold["validation_indices"]
        for name in MODEL_NAMES:
            model, train_scores, val_scores, fit_info = fit_candidate(
                name, x[train_idx], y[train_idx], x[val_idx], y[val_idx],
                seed=random_state + fold_number,
            )
            train_ap = float(average_precision_score(y[train_idx], train_scores))
            val_ap = float(average_precision_score(y[val_idx], val_scores))
            val_auc = float(roc_auc_score(y[val_idx], val_scores))
            records[name]["oof_scores"][val_idx] = val_scores
            records[name]["cv_folds"].append({
                "fold": fold_number,
                "train_windows": int(len(train_idx)),
                "train_positives": int(y[train_idx].sum()),
                "validation_windows": int(len(val_idx)),
                "validation_positives": int(y[val_idx].sum()),
                "train_average_precision": train_ap,
                "average_precision": val_ap,
                "roc_auc": val_auc,
                "best_epoch": fit_info.get("epochs"),
            })
            if fit_info.get("epochs") is not None:
                records[name]["fold_epochs"].append(int(fit_info["epochs"]))

    for name, record in records.items():
        oof_scores = record.pop("oof_scores")
        if not np.isfinite(oof_scores[dev]).all():
            raise RuntimeError(f"Missing out-of-fold predictions for {name}")
        fold_ap = [row["average_precision"] for row in record["cv_folds"]]
        fold_train_ap = [row["train_average_precision"] for row in record["cv_folds"]]
        threshold = choose_f1_threshold(y[dev], oof_scores[dev])
        record.update({
            "mean_cv_average_precision": float(statistics.mean(fold_ap)),
            "std_cv_average_precision": float(statistics.pstdev(fold_ap)),
            "mean_train_average_precision": float(statistics.mean(fold_train_ap)),
            "train_cv_ap_gap": float(statistics.mean(fold_train_ap) - statistics.mean(fold_ap)),
            "threshold_from_dev_oof_max_f1": threshold,
            "dev_oof_metrics": evaluate_scores(y[dev], oof_scores[dev], threshold),
            "fold_epochs": record["fold_epochs"],
        })
        record["_oof_scores"] = oof_scores

    selected_name = select_best_model(records)
    selected_fixed_epochs = None
    if selected_name == "pytorch_mlp":
        selected_fixed_epochs = max(1, int(round(statistics.median(records[selected_name]["fold_epochs"]))))
    learning_curve = []
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        fraction_train_ap, fraction_validation_ap = [], []
        fraction_sizes, fraction_positive = [], []
        for fold_number, fold in enumerate(split["inner_folds"]):
            train_idx = sample_group_indices(
                fold["train_indices"], split["groups"], y, fraction,
                random_state=random_state + 500 + fold_number,
            )
            validation_idx = fold["validation_indices"]
            _, train_scores, validation_scores, _ = fit_candidate(
                selected_name, x[train_idx], y[train_idx], x[validation_idx], y[validation_idx],
                seed=random_state + 1500 + fold_number,
                fixed_epochs=selected_fixed_epochs,
            )
            fraction_train_ap.append(float(average_precision_score(y[train_idx], train_scores)))
            fraction_validation_ap.append(float(average_precision_score(y[validation_idx], validation_scores)))
            fraction_sizes.append(len(train_idx))
            fraction_positive.append(int(y[train_idx].sum()))
        learning_curve.append({
            "fraction_of_training_groups": fraction,
            "mean_train_windows": int(round(statistics.mean(fraction_sizes))),
            "mean_train_positive_windows": int(round(statistics.mean(fraction_positive))),
            "mean_train_average_precision": float(statistics.mean(fraction_train_ap)),
            "mean_validation_average_precision": float(statistics.mean(fraction_validation_ap)),
            "std_validation_average_precision": float(statistics.pstdev(fraction_validation_ap)),
        })

    final_models = {}
    for name, record in records.items():
        fixed_epochs = None
        if name == "pytorch_mlp":
            fixed_epochs = max(1, int(round(statistics.median(record["fold_epochs"]))))
        model, train_scores, test_scores, fit_info = fit_candidate(
            name, x[dev], y[dev], x[test], None, seed=random_state + 1000,
            fixed_epochs=fixed_epochs,
        )
        threshold = float(record["threshold_from_dev_oof_max_f1"])
        record["test_metrics"] = evaluate_scores(y[test], test_scores, threshold)
        record["final_fit"] = {"dev_windows": int(len(dev)), "dev_positives": int(y[dev].sum()),
                                "epochs": fit_info.get("epochs", fixed_epochs)}
        final_models[name] = model
        record["_test_scores"] = test_scores

    out = __import__("pathlib").Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_record = records[selected_name]
    selected_test = selected_record["test_metrics"]
    window_label_counts = windows.loc[windows["malicious"], "reference_label"].value_counts().to_dict()
    group_meta = {
        "method": f"StratifiedGroupKFold on {group_minutes}-minute UTC window-time buckets; outer fold 0 held out as test; 4 inner folds on the remaining development groups",
        "group_minutes": group_minutes,
        "outer_test_fold": 0,
        "temporal_policy": "shuffled_group_holdout_not_forward_in_time",
        "dev_windows": int(len(dev)),
        "dev_positives": int(y[dev].sum()),
        "dev_groups": int(len(set(split["groups"][dev]))),
        "test_windows": int(len(test)),
        "test_positives": int(y[test].sum()),
        "test_groups": int(len(set(split["groups"][test]))),
        "inner_folds": [{
            "train_windows": int(len(fold["train_indices"])),
            "train_positives": int(y[fold["train_indices"]].sum()),
            "validation_windows": int(len(fold["validation_indices"])),
            "validation_positives": int(y[fold["validation_indices"]].sum()),
            "train_groups": int(len(set(split["groups"][fold["train_indices"]]))),
            "validation_groups": int(len(set(split["groups"][fold["validation_indices"]]))),
        } for fold in split["inner_folds"]],
        "group_overlap_test_development": bool(set(split["groups"][dev]) & set(split["groups"][test])),
    }
    report = {
        "schema_version": 1,
        "dataset": {
            "description": "Full available CICIDS2017 Thursday slice; no arbitrary row subsampling.",
            "input_rows": lineage["input_rows"],
            "valid_flows": int(len(flows)),
            "window_count": int(len(windows)),
            "positive_windows": int(y.sum()),
            "negative_windows": int((~y).sum()),
            "positive_window_labels": {str(k): int(v) for k, v in window_label_counts.items()},
            "row_labels": {str(k): int(v) for k, v in flows["label"].value_counts().to_dict().items()},
            "positive_source_ips": sorted(windows.loc[windows["malicious"], "source_ip"].unique().tolist()),
            "positive_window_time_range_utc": [windows.loc[windows["malicious"], "timestamp"].min().isoformat(),
                                                windows.loc[windows["malicious"], "timestamp"].max().isoformat()],
            "model_features": list(FEATURES),
        },
        "split": group_meta,
        "learning_curve": learning_curve,
        "selection": {
            "model": selected_name,
            "primary_metric": "mean 4-fold grouped cross-validation average precision",
            "selected_mean_cv_average_precision": selected_record["mean_cv_average_precision"],
            "test_used_for_selection": False,
            "threshold_policy": "Per-model threshold maximizes F1 on development out-of-fold scores; threshold is frozen before held-out test evaluation.",
        },
        "models": {name: {key: value for key, value in record.items() if not key.startswith("_")} for name, record in records.items()},
        "held_out_test": {
            "selected_model": selected_name,
            "metrics": selected_test,
            "model_test_metrics": {name: record["test_metrics"] for name, record in records.items()},
        },
        "caveats": [
            "Only 68 positive source-IP/minute windows are available; all come from one source IP and a single continuous web-attack period.",
            "Five-minute grouped splits prevent windows from the same short time bucket crossing folds, but do not prove generalization to independent attack campaigns, hosts, or networks.",
            "The held-out test contains a small number of positive windows; test metrics are exploratory and have high uncertainty.",
            "Use a larger dataset with multiple attack campaigns and hosts before claiming deployment readiness.",
            "The selected model is the best candidate under this grouped experiment, not a production-validated detector.",
        ],
        "runtime": {"python": platform.python_version(), "scikit_learn": importlib.metadata.version("scikit-learn"),
                    "torch": importlib.metadata.version("torch"), "numpy": importlib.metadata.version("numpy"),
                    "pandas": importlib.metadata.version("pandas")},
    }

    # Flatten fold/model scores into auditable CSV files.
    cv_rows = []
    for name, record in records.items():
        for fold_row in record["cv_folds"]:
            cv_rows.append({"model": name, **fold_row})
    pd.DataFrame(cv_rows).to_csv(out / "cv_results.csv", index=False)
    oof_rows = []
    for idx in dev:
        base = {"window_id": windows.iloc[idx]["window_id"], "timestamp": windows.iloc[idx]["timestamp"].isoformat(),
                "source_ip": windows.iloc[idx]["source_ip"], "malicious": bool(y[idx]),
                "reference_label": windows.iloc[idx]["reference_label"], "time_group": split["groups"][idx]}
        for name, record in records.items():
            base[f"score_{name}"] = float(record["_oof_scores"][idx])
        oof_rows.append(base)
    pd.DataFrame(oof_rows).to_csv(out / "development_oof_predictions.csv", index=False)
    test_rows = []
    test_position = {int(index): pos for pos, index in enumerate(test)}
    for idx in test:
        base = {"window_id": windows.iloc[idx]["window_id"], "timestamp": windows.iloc[idx]["timestamp"].isoformat(),
                "source_ip": windows.iloc[idx]["source_ip"], "malicious": bool(y[idx]),
                "reference_label": windows.iloc[idx]["reference_label"], "time_group": split["groups"][idx]}
        for name, record in records.items():
            score = float(record["_test_scores"][test_position[int(idx)]])
            base[f"score_{name}"] = score
            base[f"flagged_{name}"] = bool(score >= record["threshold_from_dev_oof_max_f1"])
        test_rows.append(base)
    pd.DataFrame(test_rows).to_csv(out / "held_out_test_predictions.csv", index=False)
    joblib.dump({"selected_model": selected_name, "models": final_models, "features": list(FEATURES),
                 "thresholds": {name: float(record["threshold_from_dev_oof_max_f1"]) for name, record in records.items()},
                 "split_policy": group_meta["method"]}, out / "models.joblib")
    report["report_file"] = "report.html"
    development_scores = {name: record["_oof_scores"][dev] for name, record in records.items()}
    test_scores = {name: record["_test_scores"] for name, record in records.items()}
    rendered = render_html_report(report, y[dev], development_scores, y[test], test_scores)
    (out / "report.html").write_text(rendered, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _svg_bar_chart(title, labels, values, *, errors=None, baseline=None, y_label="Score", y_max=1.0):
    import html
    labels = list(labels)
    values = [float(value) for value in values]
    errors = [0.0] * len(values) if errors is None else [float(value) for value in errors]
    width, height = 850, 390
    left, right, top, bottom = 64, 24, 46, 122
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ["#38bdf8", "#22c55e", "#f59e0b", "#a78bfa", "#fb7185", "#2dd4bf", "#94a3b8"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{left}" y="28" class="chart-title">{html.escape(title)}</text>']
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_h * (1 - value / y_max)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" class="tick" text-anchor="end">{value:.1f}</text>')
    if baseline is not None and 0 <= baseline <= y_max:
        y = top + plot_h * (1 - baseline / y_max)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#f97316" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{left+plot_w-4}" y="{y-6:.1f}" class="tick" text-anchor="end">baseline {baseline:.3f}</text>')
    slot = plot_w / max(len(labels), 1)
    bar_w = min(56, slot * 0.58)
    for idx, (label, value, error) in enumerate(zip(labels, values, errors)):
        safe_value = min(max(value, 0.0), y_max)
        x = left + idx * slot + (slot - bar_w) / 2
        y = top + plot_h * (1 - safe_value / y_max)
        color = colors[idx % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y:.1f}" rx="4" fill="{color}"/>')
        if error > 0:
            ey1 = top + plot_h * (1 - min(y_max, safe_value + error) / y_max)
            ey2 = top + plot_h * (1 - max(0, safe_value - error) / y_max)
            cx = x + bar_w / 2
            parts.append(f'<line x1="{cx:.1f}" y1="{ey1:.1f}" x2="{cx:.1f}" y2="{ey2:.1f}" stroke="#334155" stroke-width="2"/>')
            parts.append(f'<line x1="{cx-6:.1f}" y1="{ey1:.1f}" x2="{cx+6:.1f}" y2="{ey1:.1f}" stroke="#334155" stroke-width="2"/>')
            parts.append(f'<line x1="{cx-6:.1f}" y1="{ey2:.1f}" x2="{cx+6:.1f}" y2="{ey2:.1f}" stroke="#334155" stroke-width="2"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{max(top+14,y-7):.1f}" class="value" text-anchor="middle">{value:.3f}</text>')
        label_x = x + bar_w / 2
        label_y = top + plot_h + 17
        parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="tick" text-anchor="end" transform="rotate(-34 {label_x:.1f} {label_y:.1f})">{html.escape(label.replace("_", " "))}</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" class="tick" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle">{html.escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_dual_bar_chart(title, labels, first_values, second_values, *, first_name="Train AP", second_name="CV AP"):
    import html
    width, height = 850, 390
    left, right, top, bottom = 64, 24, 46, 122
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{left}" y="28" class="chart-title">{html.escape(title)}</text>']
    for tick in range(6):
        value = tick / 5
        y = top + plot_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" class="tick" text-anchor="end">{value:.1f}</text>')
    slot = plot_w / max(len(labels), 1)
    bar_w = min(28, slot * 0.28)
    for idx, label in enumerate(labels):
        center = left + idx * slot + slot / 2
        for offset, value, color in [(-bar_w/2, first_values[idx], "#a78bfa"), (bar_w/2, second_values[idx], "#38bdf8")]:
            value = min(max(float(value), 0.0), 1.0)
            x = center + offset - bar_w/2
            y = top + plot_h * (1 - value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y:.1f}" rx="3" fill="{color}"/>')
        label_x = center
        label_y = top + plot_h + 17
        parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="tick" text-anchor="end" transform="rotate(-34 {label_x:.1f} {label_y:.1f})">{html.escape(label.replace("_", " "))}</text>')
    parts.append(f'<rect x="{left+10}" y="{height-22}" width="12" height="12" fill="#a78bfa"/><text x="{left+28}" y="{height-12}" class="tick">{html.escape(first_name)}</text>')
    parts.append(f'<rect x="{left+135}" y="{height-22}" width="12" height="12" fill="#38bdf8"/><text x="{left+153}" y="{height-12}" class="tick">{html.escape(second_name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_curve_chart(title, curves, *, x_label, y_label, reference=None):
    import html
    import math
    width, height = 780, 390
    left, right, top, bottom = 62, 24, 48, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    palette = ["#0284c7", "#16a34a", "#ea580c", "#7c3aed", "#db2777", "#0f766e", "#475569"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{left}" y="28" class="chart-title">{html.escape(title)}</text>']
    for tick in range(6):
        value = tick / 5
        x = left + plot_w * value
        y = top + plot_h * (1 - value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+19}" class="tick" text-anchor="middle">{value:.1f}</text>')
        parts.append(f'<text x="{left-9}" y="{y+4:.1f}" class="tick" text-anchor="end">{value:.1f}</text>')
    if reference is not None and 0 < reference < 1:
        y = top + plot_h * (1 - reference)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#f97316" stroke-dasharray="5 5"/>')
    for idx, (name, xs, ys) in enumerate(curves):
        color = palette[idx % len(palette)]
        points = []
        step = max(1, int(math.ceil(len(xs) / 160)))
        for xval, yval in zip(xs[::step], ys[::step]):
            px = left + plot_w * min(max(float(xval), 0), 1)
            py = top + plot_h * (1 - min(max(float(yval), 0), 1))
            points.append(f"{px:.1f},{py:.1f}")
        if len(xs) and (len(xs) - 1) % step:
            px = left + plot_w * min(max(float(xs[-1]), 0), 1)
            py = top + plot_h * (1 - min(max(float(ys[-1]), 0), 1))
            points.append(f"{px:.1f},{py:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        legend_x = left + (idx % 3) * 235
        legend_y = height - 25 + (idx // 3) * 16
        parts.append(f'<line x1="{legend_x}" y1="{legend_y-4}" x2="{legend_x+18}" y2="{legend_y-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+24}" y="{legend_y}" class="tick">{html.escape(name.replace("_", " "))}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{top+plot_h+43}" class="tick" text-anchor="middle">{html.escape(x_label)}</text>')
    parts.append(f'<text x="16" y="{top+plot_h/2}" class="tick" transform="rotate(-90 16 {top+plot_h/2})" text-anchor="middle">{html.escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_confusion_matrix(title, confusion):
    import html
    width, height = 430, 330
    labels = [["TN", "FP"], ["FN", "TP"]]
    keys = [["tn", "fp"], ["fn", "tp"]]
    values = [[int(confusion[key]) for key in row] for row in keys]
    maximum = max(max(row) for row in values) or 1
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="20" y="28" class="chart-title">{html.escape(title)}</text>',
             '<text x="230" y="58" class="tick" text-anchor="middle">Predicted</text>',
             '<text x="17" y="185" class="tick" transform="rotate(-90 17 185)" text-anchor="middle">Actual</text>']
    x0, y0, size = 95, 82, 125
    for row in range(2):
        for col in range(2):
            value = values[row][col]
            alpha = 0.15 + 0.75 * value / maximum
            color = "34,197,94" if row == col else "248,113,113"
            fill = f"rgba({color},{alpha:.2f})"
            x, y = x0 + col * size, y0 + row * size
            parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{fill}" stroke="#cbd5e1"/>')
            parts.append(f'<text x="{x+size/2}" y="{y+55}" class="value" text-anchor="middle">{value}</text>')
            parts.append(f'<text x="{x+size/2}" y="{y+80}" class="tick" text-anchor="middle">{labels[row][col]}</text>')
    parts.append('<text x="220" y="322" class="tick" text-anchor="middle">Benign                 Malicious</text>')
    parts.append("</svg>")
    return "".join(parts)


def _interactive_panel(report) -> str:
    import json
    data = {"selection": report["selection"], "models": report["models"], "dataset": report["dataset"]}
    embedded_json = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return """
<section class="card interactive-panel">
  <h2>Interactive model explorer</h2>
  <div class="interactive-controls">
    <label>Rank models by
      <select id="live-metric-select">
        <option value="mean_cv_average_precision">Grouped CV AP</option>
        <option value="test_average_precision">Held-out test AP</option>
        <option value="test_f1">Held-out test F1</option>
        <option value="test_precision">Held-out test precision</option>
        <option value="test_recall">Held-out test recall</option>
      </select>
    </label>
    <label>Focus model <select id="live-model-select"></select></label>
    <button id="refresh-metrics" type="button">Refresh latest metrics</button>
    <span id="refresh-status" class="muted">Loading run metrics…</span>
  </div>
  <div id="live-model-details" class="grid"></div>
  <div id="dynamic-metrics-chart" class="chart" aria-live="polite"></div>
  <p class="muted">When served locally, this panel fetches <code>metrics.json</code> on open and refreshes every 30 seconds. Reload the page to refresh the embedded SVG report charts after a new run.</p>
</section>
<script id="report-metrics" type="application/json">__REPORT_JSON__</script>
<script>
(() => {
  const node = document.getElementById("report-metrics");
  let current = JSON.parse(node.textContent);
  const metricSelect = document.getElementById("live-metric-select");
  const modelSelect = document.getElementById("live-model-select");
  const details = document.getElementById("live-model-details");
  const chart = document.getElementById("dynamic-metrics-chart");
  const status = document.getElementById("refresh-status");
  const refreshButton = document.getElementById("refresh-metrics");
  const metricLabels = {
    mean_cv_average_precision: "Grouped CV AP",
    test_average_precision: "Held-out test AP",
    test_f1: "Held-out test F1",
    test_precision: "Held-out test precision",
    test_recall: "Held-out test recall"
  };
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, ch => {
    if (ch === "&") return "&amp;";
    if (ch === "<") return "&lt;";
    if (ch === ">") return "&gt;";
    if (ch === String.fromCharCode(34)) return "&quot;";
    return "&#39;";
  });
  const metricValue = (record, key) => key === "mean_cv_average_precision"
    ? Number(record.mean_cv_average_precision || 0)
    : Number((record.test_metrics || {})[key.replace("test_", "")] || 0);

  function fillModels() {
    const names = Object.keys(current.models || {}).sort();
    const previous = modelSelect.value || (current.selection || {}).model;
    modelSelect.innerHTML = names.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name.replaceAll("_", " "))}</option>`).join("");
    if (names.includes(previous)) modelSelect.value = previous;
    else if (names.length) modelSelect.value = names[0];
  }

  function drawDetails() {
    const name = modelSelect.value;
    const record = (current.models || {})[name];
    if (!record) { details.innerHTML = ""; return; }
    const test = record.test_metrics || {};
    const cm = test.confusion_matrix || {};
    const cards = [
      ["Mean grouped-CV AP", Number(record.mean_cv_average_precision || 0).toFixed(3)],
      ["Fold spread (SD)", Number(record.std_cv_average_precision || 0).toFixed(3)],
      ["Train − CV AP gap", Number(record.train_cv_ap_gap || 0).toFixed(3)],
      ["Held-out test F1", Number(test.f1 || 0).toFixed(3)],
      ["Test precision / recall", `${Number(test.precision || 0).toFixed(3)} / ${Number(test.recall || 0).toFixed(3)}`],
      ["Test confusion (TN / FP / FN / TP)", `${cm.tn || 0} / ${cm.fp || 0} / ${cm.fn || 0} / ${cm.tp || 0}`]
    ];
    details.innerHTML = cards.map(([label, value]) => `<div class="card"><div class="muted">${escapeHtml(label)}</div><strong>${escapeHtml(value)}</strong></div>`).join("");
  }

  function drawChart() {
    const key = metricSelect.value;
    const rows = Object.entries(current.models || {}).map(([name, record]) => ({name, value: metricValue(record, key)}))
      .sort((a, b) => b.value - a.value);
    const width = 820, height = 330, left = 56, right = 20, top = 44, bottom = 78;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const slot = plotWidth / Math.max(rows.length, 1), barWidth = Math.min(56, slot * 0.6);
    const color = "#38bdf8";
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Models ranked by ${escapeHtml(metricLabels[key])}">`;
    svg += `<text x="${left}" y="26" fill="var(--foreground,#172033)" font-size="15" font-weight="600">Ranked by ${escapeHtml(metricLabels[key])}</text>`;
    for (let tick = 0; tick <= 5; tick++) {
      const value = tick / 5, y = top + plotHeight * (1 - value);
      svg += `<line x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}" stroke="var(--border,#dbe2ea)"/>`;
      svg += `<text x="${left - 8}" y="${y + 4}" text-anchor="end" fill="var(--muted-foreground,#64748b)" font-size="10">${value.toFixed(1)}</text>`;
    }
    rows.forEach((row, index) => {
      const x = left + index * slot + (slot - barWidth) / 2;
      const value = Math.max(0, Math.min(1, row.value));
      const y = top + plotHeight * (1 - value);
      const selected = row.name === modelSelect.value;
      svg += `<g data-model="${escapeHtml(row.name)}" style="cursor:pointer">`;
      svg += `<rect x="${x}" y="${y}" width="${barWidth}" height="${top + plotHeight - y}" rx="4" fill="${selected ? "#22c55e" : color}"/>`;
      svg += `<text x="${x + barWidth / 2}" y="${Math.max(top + 12, y - 6)}" text-anchor="middle" fill="var(--foreground,#172033)" font-size="10">${value.toFixed(3)}</text>`;
      svg += `<text x="${x + barWidth / 2}" y="${top + plotHeight + 16}" text-anchor="end" transform="rotate(-32 ${x + barWidth / 2} ${top + plotHeight + 16})" fill="var(--muted-foreground,#64748b)" font-size="10">${escapeHtml(row.name.replaceAll("_", " "))}</text>`;
      svg += `</g>`;
    });
    svg += `</svg>`;
    chart.innerHTML = svg;
    chart.querySelectorAll("[data-model]").forEach(item => item.addEventListener("click", () => {
      modelSelect.value = item.getAttribute("data-model");
      drawDetails(); drawChart();
    }));
    chart.setAttribute("aria-label", metricLabels[key]);
  }

  function render() { fillModels(); drawDetails(); drawChart(); }
  async function refreshLiveMetrics() {
    try {
      const response = await fetch("metrics.json", {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const latest = await response.json();
      if (latest.models && latest.selection) {
        current = latest;
        render();
        status.textContent = "Live metrics loaded from metrics.json.";
      } else {
        throw new Error("metrics.json does not contain benchmark models");
      }
    } catch (error) {
      status.textContent = "Using embedded metrics; serve this report over HTTP to refresh from metrics.json.";
    }
  }
  metricSelect.addEventListener("change", drawChart);
  modelSelect.addEventListener("change", () => { drawDetails(); drawChart(); });
  refreshButton.addEventListener("click", refreshLiveMetrics);
  render();
  refreshLiveMetrics();
  setInterval(refreshLiveMetrics, 30000);
})();
</script>
""".replace("__REPORT_JSON__", embedded_json)


def render_html_report(report, development_truth, development_scores, test_truth, test_scores) -> str:
    """Render a self-contained HTML/SVG dashboard with metrics and diagnostics."""
    import html
    from sklearn.metrics import precision_recall_curve, roc_curve

    models = report["models"]
    names = sorted(models, key=lambda name: models[name]["mean_cv_average_precision"], reverse=True)
    means = [models[name]["mean_cv_average_precision"] for name in names]
    errors = [models[name]["std_cv_average_precision"] for name in names]
    train_ap = [models[name]["mean_train_average_precision"] for name in names]
    cv_ap = means
    prevalence = report["dataset"]["positive_windows"] / report["dataset"]["window_count"]
    pr_curves, roc_curves = [], []
    for name in names:
        scores = np.asarray(development_scores[name], dtype=float)
        precision, recall, _ = precision_recall_curve(development_truth, scores)
        pr_curves.append((name, recall, precision))
        fpr, tpr, _ = roc_curve(development_truth, scores)
        roc_curves.append((name, fpr, tpr))

    selected = report["selection"]["model"]
    selected_test = report["models"][selected]["test_metrics"]
    learning = report.get("learning_curve", [])
    learning_curve_chart = _svg_curve_chart(
        "Learning-size curve for selected model",
        [("Train AP", [row["fraction_of_training_groups"] for row in learning],
          [row["mean_train_average_precision"] for row in learning]),
         ("Grouped CV AP", [row["fraction_of_training_groups"] for row in learning],
          [row["mean_validation_average_precision"] for row in learning])],
        x_label="Fraction of training time groups", y_label="Average precision", reference=prevalence,
    ) if learning else ""
    learning_rows = "".join(
        f"<tr><td>{row['fraction_of_training_groups']:.0%}</td><td>{row['mean_train_windows']:,}</td>"
        f"<td>{row['mean_train_positive_windows']}</td><td>{row['mean_train_average_precision']:.3f}</td>"
        f"<td>{row['mean_validation_average_precision']:.3f} ± {row['std_validation_average_precision']:.3f}</td></tr>"
        for row in learning
    )
    table_rows = []
    for name in names:
        record = models[name]
        test = record["test_metrics"]
        table_rows.append(
            "<tr>"
            f"<th>{html.escape(name.replace('_', ' '))}{' <span class=best>selected</span>' if name == selected else ''}</th>"
            f"<td>{record['mean_cv_average_precision']:.3f} ± {record['std_cv_average_precision']:.3f}</td>"
            f"<td>{record['mean_train_average_precision']:.3f}</td>"
            f"<td>{record['train_cv_ap_gap']:+.3f}</td>"
            f"<td>{test['average_precision']:.3f}</td><td>{test['f1']:.3f}</td>"
            f"<td>{test['precision']:.3f}</td><td>{test['recall']:.3f}</td>"
            f"<td>{test['confusion_matrix']['fp']}</td>"
            "</tr>"
        )

    split_rows = []
    for index, fold in enumerate(report["split"]["inner_folds"], 1):
        split_rows.append(
            f"<tr><td>CV fold {index}</td><td>{fold['train_windows']:,} / {fold['train_positives']}</td>"
            f"<td>{fold['validation_windows']:,} / {fold['validation_positives']}</td>"
            f"<td>{fold['train_groups']} / {fold['validation_groups']}</td></tr>"
        )

    charts = [
        _svg_bar_chart("Grouped CV average precision (mean ± fold SD)", names, means, errors=errors,
                       baseline=prevalence, y_label="Average precision"),
        _svg_dual_bar_chart("Training vs grouped-CV AP (gap is an overfit diagnostic)", names, train_ap, cv_ap),
        *([learning_curve_chart] if learning_curve_chart else []),
        _svg_curve_chart("Development out-of-fold precision–recall curves", pr_curves,
                          x_label="Recall", y_label="Precision", reference=prevalence),
        _svg_curve_chart("Development out-of-fold ROC curves", roc_curves,
                          x_label="False-positive rate", y_label="True-positive rate", reference=0.5),
        _svg_bar_chart("Held-out test average precision by model", names,
                       [models[name]["test_metrics"]["average_precision"] for name in names],
                       baseline=prevalence, y_label="Average precision"),
        _svg_curve_chart("Held-out test precision–recall curve (selected model; small support)",
                          [(selected, *precision_recall_curve(test_truth, test_scores[selected])[1::-1])],
                          x_label="Recall", y_label="Precision", reference=prevalence),
        _svg_confusion_matrix(f"Selected model on held-out test: {selected.replace('_', ' ')}", selected_test["confusion_matrix"]),
    ]

    dataset = report["dataset"]
    split = report["split"]
    caveats = "".join(f"<li>{html.escape(item)}</li>" for item in report["caveats"])
    labels = "".join(f"<li>{html.escape(label)}: {count:,} positive windows</li>"
                     for label, count in dataset["positive_window_labels"].items())
    interactive_block = _interactive_panel(report)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cyber Threat Lab — Model Benchmark</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ color: var(--foreground,#172033); background: var(--card,#ffffff); font: 14px/1.5 Segoe UI,Arial,sans-serif; margin: 0; padding: 16px; }}
h1,h2,h3 {{ margin: 0 0 10px; }} h1 {{ font-size: 25px; }} h2 {{ font-size: 18px; margin-top: 24px; }}
.sub {{ color: var(--muted-foreground,#64748b); margin: 0 0 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 10px; }}
.card {{ border: 1px solid var(--border,#dbe2ea); border-radius: 12px; padding: 12px; }}
.kpi {{ font-size: 22px; font-weight: 700; }} .muted {{ color: var(--muted-foreground,#64748b); }}
.warn {{ border-left: 4px solid #f59e0b; padding: 10px 12px; margin: 12px 0; background: color-mix(in srgb,#f59e0b 10%,transparent); }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,600px),1fr)); gap: 12px; margin-top: 12px; }}
.chart {{ border: 1px solid var(--border,#dbe2ea); border-radius: 12px; padding: 8px; overflow: hidden; }} svg {{ display:block; width:100%; height:auto; }}
.interactive-panel {{ margin-top: 16px; }} .interactive-controls {{ display:flex; flex-wrap:wrap; align-items:end; gap:12px; margin:12px 0; }}
.interactive-controls label {{ display:grid; gap:4px; font-weight:600; }} select,button {{ color:var(--foreground,#172033); background:var(--card,#fff); border:1px solid var(--border,#cbd5e1); border-radius:6px; padding:7px 9px; }}
button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent,#38bdf8); }}
.chart-title {{ fill: var(--foreground,#172033); font: 600 16px Segoe UI,Arial,sans-serif; }}
.tick {{ fill: var(--muted-foreground,#64748b); font: 11px Segoe UI,Arial,sans-serif; }}
.value {{ fill: var(--foreground,#172033); font: 600 11px Segoe UI,Arial,sans-serif; }}
.grid {{ stroke: var(--border,#e2e8f0); stroke-width: 1; }}
table {{ width:100%; border-collapse: collapse; font-size:12px; }} th,td {{ text-align:left; padding:8px; border-bottom:1px solid var(--border,#e2e8f0); }}
th {{ color:var(--muted-foreground,#64748b); }} .best {{ color:#047857; font-size:10px; text-transform:uppercase; }}
code {{ overflow-wrap:anywhere; }}
@media(max-width:700px) {{ body {{padding:10px}} table {{display:block; overflow-x:auto}} }}
</style></head><body>
<h1>🛡️ Cyber Threat Lab — model benchmark</h1>
<p class="sub">Full available Thursday slice · grouped {split['group_minutes']}-minute window split · model selection by mean inner-CV average precision</p>
<div class="grid">
  <div class="card"><div class="muted">Raw labeled flow rows</div><div class="kpi">{dataset['input_rows']:,}</div></div>
  <div class="card"><div class="muted">Valid normalized flows</div><div class="kpi">{dataset['valid_flows']:,}</div></div>
  <div class="card"><div class="muted">Model samples (source/minute windows)</div><div class="kpi">{dataset['window_count']:,}</div></div>
  <div class="card"><div class="muted">Positive windows</div><div class="kpi">{dataset['positive_windows']:,} ({prevalence:.2%})</div></div>
  <div class="card"><div class="muted">Selected model</div><div class="kpi">{html.escape(selected.replace('_',' '))}</div></div>
  <div class="card"><div class="muted">Held-out test support</div><div class="kpi">{split['test_windows']:,} / {split['test_positives']} positive</div></div>
</div>
{interactive_block}
<div class="warn"><strong>Interpret cautiously:</strong> the full row set is used, but it contains only {dataset['positive_windows']} positive windows. Grouped CV keeps each {split['group_minutes']}-minute bucket intact, but group assignment is shuffled rather than a forward-in-time test. All positives still come from one source host and one continuous web-attack period, so this does not prove generalization to future or independent attack campaigns.</div>
<h2>Learning-size curve</h2>
<p class="muted">The selected candidate is retrained on increasing fractions of whole training groups inside each inner fold. Rising validation AP with more windows suggests more data may help; a large train–validation gap suggests overfit. This diagnostic does not use the held-out test fold.</p>
<div class="chart"><table><thead><tr><th>Training-group fraction</th><th>Mean train windows</th><th>Mean train positives</th><th>Train AP</th><th>Grouped-CV AP ± SD</th></tr></thead><tbody>{learning_rows}</tbody></table></div>
<h2>Model comparison</h2>
<p class="muted">Winner is chosen by inner grouped-CV average precision. Held-out test metrics are shown for transparency but were not used to pick the model.</p>
<div class="chart"><table><thead><tr><th>Model</th><th>Mean CV AP ± SD</th><th>Train AP</th><th>Train−CV gap</th><th>Held-out test AP</th><th>Test F1</th><th>Test precision</th><th>Test recall</th><th>Test false positives</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<div class="charts">{''.join(f'<div class="chart">{chart}</div>' for chart in charts)}</div>
<h2>Split evidence</h2>
<p class="muted">Outer development: {split['dev_windows']:,} windows / {split['dev_positives']} positives across {split['dev_groups']} groups. Held-out test: {split['test_windows']:,} windows / {split['test_positives']} positives across {split['test_groups']} groups. Group overlap: {split['group_overlap_test_development']}.</p>
<div class="chart"><table><thead><tr><th>Fold</th><th>Train windows / positives</th><th>Validation windows / positives</th><th>Train / validation time groups</th></tr></thead><tbody>{''.join(split_rows)}</tbody></table></div>
<h2>Attack label support</h2><ul>{labels}</ul>
<h2>Fit notes and limitations</h2><ul>{caveats}</ul>
<p class="muted">Primary ranking metric: average precision, the area under the precision–recall curve in the step-function sense. The saved JSON and CSV files contain fold-level scores, thresholds, predictions, and confusion matrices. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">Metric definition</a> · <a href="https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold">Grouped stratified CV</a>.</p>
</body></html>"""
