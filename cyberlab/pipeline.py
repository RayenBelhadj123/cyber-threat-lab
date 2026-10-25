"""CICIDS2017 flow ingestion. Labels and identities are never model inputs."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ALIASES = {
    "source_ip": ["Source IP", "Src IP"], "destination_ip": ["Destination IP", "Dst IP"],
    "source_port": ["Source Port", "Src Port"], "destination_port": ["Destination Port", "Dst Port"],
    "protocol": ["Protocol"], "timestamp": ["Timestamp"],
    "duration": ["Flow Duration"], "fwd_packets": ["Total Fwd Packets", "Tot Fwd Pkts"],
    "bwd_packets": ["Total Backward Packets", "Tot Bwd Pkts"],
    "fwd_bytes": ["Total Length of Fwd Packets", "TotLen Fwd Pkts"],
    "bwd_bytes": ["Total Length of Bwd Packets", "TotLen Bwd Pkts"],
    "rate": ["Flow Bytes/s", "Flow Byts/s"], "label": ["Label"],
}
NUMERIC = ["source_port", "destination_port", "protocol", "duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes", "rate"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def canonical_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def valid_ip(value):
    try:
        return str(ipaddress.ip_address(str(value).strip()))
    except ValueError:
        return None


def load_flows(input_path, max_rows=None, require_label=True):
    path = Path(input_path)
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    if not files or any(not p.is_file() for p in files):
        raise ValueError(f"No input CSV files: {path}")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    frames = []
    lineage = {"input_rows": 0, "invalid_rows": 0, "duplicate_rows": 0, "files": [], "labels_available": True, "timezone": "America/Moncton", "timestamp_policy": "dayfirst; ambiguous/nonexistent local times rejected", "failure_ratio": "Unavailable in CICIDS2017 flow schema; default 0 (not inferred from labels)."}
    for file in files:
        if max_rows is not None and lineage["input_rows"] >= max_rows:
            break
        file_info = {"path": str(file.resolve()), "size_bytes": file.stat().st_size}
        file_hash = hashlib.sha256()
        with file.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                file_hash.update(block)
        file_info["sha256"] = file_hash.hexdigest()
        encoding = "utf-8-sig"
        try:
            with file.open("r", encoding=encoding) as probe:
                while probe.read(1024 * 1024):
                    pass
        except UnicodeDecodeError:
            encoding = "cp1252"
        file_info["encoding"] = encoding
        lineage["files"].append(file_info)
        for frame in pd.read_csv(file, chunksize=100000, encoding=encoding, low_memory=False):
            if max_rows is not None:
                frame = frame.iloc[:max_rows-lineage["input_rows"]].copy()
            lineage["input_rows"] += len(frame)
            normalized = {canonical_header(c): c for c in frame.columns}
            rename = {}
            for target, options in ALIASES.items():
                match = next((normalized[canonical_header(a)] for a in options if canonical_header(a) in normalized), None)
                if match is None:
                    if target == "label" and not require_label:
                        lineage["labels_available"] = False
                        continue
                    raise ValueError(f"Missing required column {target} in {file}; full labeled flows with real identity/time are required")
                rename[match] = target
            frame = frame.rename(columns=rename)
            if "label" not in frame:
                frame["label"] = "UNKNOWN"
            frame = frame[list(ALIASES)].copy()
            for c in NUMERIC:
                frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
            frame["source_ip"] = frame["source_ip"].map(valid_ip)
            frame["destination_ip"] = frame["destination_ip"].map(valid_ip)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce").dt.tz_localize(ZoneInfo("America/Moncton"), ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
            frame["label"] = frame["label"].astype("string").str.strip()
            valid = frame.notna().all(axis=1) & np.isfinite(frame[NUMERIC]).all(axis=1) & frame["label"].ne("")
            valid &= frame[NUMERIC].ge(0).all(axis=1)
            for c, upper in [("source_port", 65535), ("destination_port", 65535), ("protocol", 255)]:
                valid &= frame[c].le(upper) & frame[c].eq(frame[c].round())
            lineage["invalid_rows"] += int((~valid).sum())
            frame = frame.loc[valid].copy()
            frame["malicious"] = frame["label"].str.upper().ne("BENIGN") & frame["label"].str.upper().ne("UNKNOWN")
            frame["bytes_total"] = frame["fwd_bytes"] + frame["bwd_bytes"]
            frame["packets_total"] = frame["fwd_packets"] + frame["bwd_packets"]
            frame["event_id"] = [digest({k: (v.isoformat() if k == "timestamp" else v) for k, v in row.items()}) for row in frame.to_dict("records")]
            frames.append(frame)
            if max_rows is not None and lineage["input_rows"] >= max_rows:
                break
    if not frames:
        raise ValueError("No input rows")
    flows = pd.concat(frames, ignore_index=True)
    before = len(flows)
    flows = flows.drop_duplicates("event_id").sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    lineage["duplicate_rows"] = before-len(flows)
    lineage["valid_rows"] = len(flows)
    if flows.empty:
        raise ValueError(f"No valid flows remain: {lineage}")
    return flows, lineage


FEATURES = ["flow_count", "unique_destinations", "unique_ports", "bytes_total", "packets_total", "duration_mean", "rate_mean", "failure_ratio"]


def aggregate_windows(flows):
    work = flows.assign(timestamp=flows["timestamp"].dt.floor("min") + pd.offsets.Minute(1))
    windows = work.groupby(["source_ip", "timestamp"], sort=True).agg(
        flow_count=("event_id", "size"), unique_destinations=("destination_ip", "nunique"),
        unique_ports=("destination_port", "nunique"), bytes_total=("bytes_total", "sum"),
        packets_total=("packets_total", "sum"), duration_mean=("duration", "mean"),
        rate_mean=("rate", "mean"), malicious=("malicious", "max"),
        reference_label=("label", lambda labels: "|".join(sorted(set(labels)))),
    ).reset_index().sort_values(["timestamp", "source_ip"]).reset_index(drop=True)
    windows["failure_ratio"] = 0.0
    windows["window_id"] = [digest([row.source_ip, row.timestamp.isoformat()]) for row in windows.itertuples()]
    return windows


def split_windows(windows):
    times = sorted(windows["timestamp"].unique())
    n = len(times)
    a, b = int(n * .6), int(n * .8)
    if not (0 < a < b < n):
        raise ValueError("Need enough unique window timestamps for nonempty 60/20/20 chronological splits (at least 3)")
    assignment = {t: "train" if i < a else "validation" if i < b else "test" for i, t in enumerate(times)}
    result = windows.copy()
    result["split"] = result["timestamp"].map(assignment)
    return result


def label_set(labels):
    return {label for value in labels for label in value.split("|") if label.upper() != "BENIGN"}


def evaluation(windows, scores, threshold):
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    if windows.empty:
        return {"status": "unavailable", "reason": "No eligible windows"}
    truth = windows["malicious"].to_numpy(dtype=bool)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, flagged, labels=[False, True]).ravel()
    host_days = windows.assign(day=windows["timestamp"].dt.floor("D")).drop_duplicates(["source_ip", "day"]).shape[0]
    both_classes = bool(truth.any() and (~truth).any())
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else None
    recalls = [value for value in [specificity, sensitivity] if value is not None]
    balanced = float(np.mean(recalls)) if recalls else None
    return {"status": "available", "windows": len(windows), "positives": int(truth.sum()),
            "negatives": int((~truth).sum()), "threshold": float(threshold),
            "precision": float(precision_score(truth, flagged, zero_division=0)),
            "recall": float(recall_score(truth, flagged, zero_division=0)),
            "f1": float(f1_score(truth, flagged, zero_division=0)),
            "balanced_accuracy": balanced,
            "balanced_accuracy_note": None if both_classes else "Only one reference class is present; this equals recall for the present class",
            "specificity": specificity, "sensitivity": sensitivity,
            "average_precision": float(average_precision_score(truth, scores)) if truth.any() else None,
            "average_precision_note": None if truth.any() else "No positive reference windows",
            "roc_auc": float(roc_auc_score(truth, scores)) if both_classes else None,
            "roc_auc_note": None if both_classes else "Requires both positive and negative reference windows",
            "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                                 "false_negative": int(fn), "true_positive": int(tp)},
            "false_positives": int(fp), "observed_host_days": host_days,
            "false_positives_per_host_day": float(fp / host_days) if host_days else None,
            "host_day_definition": "distinct observed source IP / UTC calendar day in evaluated subset"}


def train_evaluate(windows):
    from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier, RandomForestClassifier
    import torch
    from sklearn.preprocessing import StandardScaler

    train = windows["split"].eq("train")
    validation = windows["split"].eq("validation")
    benign_train = train & ~windows["malicious"]
    benign_validation = validation & ~windows["malicious"]
    x = windows[FEATURES].astype(float).to_numpy()
    y = windows["malicious"].to_numpy(dtype=bool)
    models = {"feature_names": FEATURES, "random_state": 42}
    metrics = {}
    predictions = {}
    if benign_train.sum() >= 2 and benign_validation.sum() >= 2:
        model = IsolationForest(n_estimators=100, random_state=42, n_jobs=1).fit(x[benign_train.to_numpy()])
        scores = -model.score_samples(x)
        threshold = float(np.percentile(scores[benign_validation.to_numpy()], 99))
        predictions["isolation_forest"] = (scores, threshold)
        models["isolation_forest"] = model
        metrics["isolation_forest"] = {"status": "available", "threshold": threshold, "threshold_source": "benign_validation_99th_percentile", "benign_train_windows": int(benign_train.sum()), "benign_validation_windows": int(benign_validation.sum())}
    else:
        metrics["isolation_forest"] = {"status": "unavailable", "reason": "Requires >=2 benign train and >=2 benign validation windows for independent threshold calibration"}

    train_y = y[train.to_numpy()]
    if np.unique(train_y).size == 2:
        model = HistGradientBoostingClassifier(random_state=42, max_iter=100).fit(x[train.to_numpy()], train_y)
        scores = model.predict_proba(x)[:, 1]
        predictions["supervised"] = (scores, 0.5)
        models["supervised"] = model
        metrics["supervised"] = {"status": "available", "threshold": 0.5, "threshold_source": "fixed_probability_0.5", "algorithm": "HistGradientBoostingClassifier"}

        rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2,
                                    class_weight="balanced_subsample", random_state=42, n_jobs=1)
        rf.fit(x[train.to_numpy()], train_y)
        rf_scores = rf.predict_proba(x)[:, list(rf.classes_).index(True)]
        predictions["random_forest"] = (rf_scores, 0.5)
        models["random_forest"] = rf
        metrics["random_forest"] = {"status": "available", "threshold": 0.5, "threshold_source": "fixed_probability_0.5", "algorithm": "RandomForestClassifier", "trees": 150, "class_weight": "balanced_subsample"}

        torch.manual_seed(42)
        torch.set_num_threads(1)
        transformed = np.log1p(np.clip(x, 0, None))
        scaler = StandardScaler().fit(transformed[train.to_numpy()])
        x_scaled = scaler.transform(transformed).astype(np.float32)
        net = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        y_float = y.astype(np.float32)
        positive = max(float(train_y.sum()), 1.0)
        negative = max(float(len(train_y) - train_y.sum()), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive]))
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
        tx = torch.from_numpy(x_scaled[train.to_numpy()])
        ty = torch.from_numpy(y_float[train.to_numpy()]).reshape(-1, 1)
        net.train()
        for _ in range(30):
            order = torch.randperm(len(tx))
            for batch in order.split(min(256, len(tx))):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(net(tx[batch]), ty[batch])
                loss.backward()
                optimizer.step()
        net.eval()
        with torch.no_grad():
            dl_scores = torch.sigmoid(net(torch.from_numpy(x_scaled))).cpu().numpy().reshape(-1)
        predictions["deep_learning"] = (dl_scores, 0.5)
        models["deep_learning"] = {"model": net, "scaler": scaler, "features": FEATURES,
                                   "input_transform": "log1p_nonnegative_then_standard_scale",
                                   "architecture": [64, 32, 16], "threshold": 0.5}
        metrics["deep_learning"] = {"status": "available", "threshold": 0.5, "threshold_source": "fixed_probability_0.5", "algorithm": "PyTorch feed-forward neural network", "hidden_layers": [64, 32, 16], "epochs": 30, "class_weighting": "positive BCE weight = train negatives / train positives"}
    else:
        reason = "Training windows do not contain both reference classes"
        for name in ["supervised", "random_forest", "deep_learning"]:
            metrics[name] = {"status": "unavailable", "reason": reason}

    predictions["scan_rule"] = (np.maximum(windows["unique_ports"].to_numpy()/20, windows["flow_count"].to_numpy()/100), 1.0)
    metrics["scan_rule"] = {"status": "available", "threshold": 1.0, "rule": "unique_ports >= 20 OR flow_count >= 100"}
    train_attacks = label_set(windows.loc[train, "reference_label"])
    test = windows["split"].eq("test")
    unseen = label_set(windows.loc[test, "reference_label"]) - train_attacks
    unseen_mask = test & windows["reference_label"].map(lambda value: bool(label_set([value]) & unseen))
    unseen_evaluation = test & (~windows["malicious"] | unseen_mask)
    metrics["unseen_test_attacks"] = {"labels": sorted(unseen), "attack_windows": int(unseen_mask.sum()), "evaluation_policy": "unseen-attack windows plus benign test windows; known-only attack windows excluded", "detectors": {}}
    for name, (scores, threshold) in predictions.items():
        metrics[name]["evaluation"] = {split: evaluation(windows.loc[windows["split"].eq(split)], scores[windows["split"].eq(split).to_numpy()], threshold) for split in ["train", "validation", "test"]}
        metrics["unseen_test_attacks"]["detectors"][name] = evaluation(windows.loc[unseen_evaluation], scores[unseen_evaluation.to_numpy()], threshold) if unseen else {"status": "unavailable", "reason": "No test attack labels unseen in training"}
    findings = []
    detector_names = ["isolation_forest", "supervised", "random_forest", "deep_learning", "scan_rule"]
    for name in detector_names:
        scores, threshold = predictions.get(name, (None, None))
        for i, row in enumerate(windows.to_dict("records")):
            score = float(scores[i]) if scores is not None else None
            findings.append({"@timestamp": row["timestamp"].isoformat(), "source": {"ip": row["source_ip"]}, "window_id": row["window_id"],
                             "detector": name, "score": score, "threshold": threshold, "flagged": bool(score >= threshold) if score is not None else None,
                             "status": metrics[name]["status"], "reason": metrics[name].get("reason"), "split": row["split"], "malicious": bool(row["malicious"]),
                             "reference_label": row["reference_label"], "features": {f: float(row[f]) for f in FEATURES}})
    models["thresholds"] = {name: threshold for name, (_, threshold) in predictions.items()}
    return models, metrics, findings


def score_model_bundle(models, windows):
    import torch
    x = windows[FEATURES].astype(float).to_numpy()
    scores = {}
    if "isolation_forest" in models:
        scores["isolation_forest"] = -models["isolation_forest"].score_samples(x)
    if "supervised" in models:
        scores["supervised"] = models["supervised"].predict_proba(x)[:, 1]
    if "random_forest" in models:
        forest = models["random_forest"]
        scores["random_forest"] = forest.predict_proba(x)[:, list(forest.classes_).index(True)]
    if "deep_learning" in models:
        bundle = models["deep_learning"]
        transformed = np.log1p(np.clip(x, 0, None))
        scaled = bundle["scaler"].transform(transformed).astype(np.float32)
        network = bundle["model"]
        network.eval()
        with torch.no_grad():
            scores["deep_learning"] = torch.sigmoid(network(torch.from_numpy(scaled))).cpu().numpy().reshape(-1)
    scores["scan_rule"] = np.maximum(windows["unique_ports"].to_numpy()/20,
                                     windows["flow_count"].to_numpy()/100)
    return scores


def predict_pipeline(input_path, model_path, output_path, max_rows=None):
    import joblib
    flows, lineage = load_flows(input_path, max_rows, require_label=False)
    windows = aggregate_windows(flows)
    models = joblib.load(model_path)
    scores_by_model = score_model_bundle(models, windows)
    thresholds = models.get("thresholds", {})
    documents = []
    for name, scores in scores_by_model.items():
        threshold = float(thresholds.get(name, 1.0 if name == "scan_rule" else 0.5))
        for index, row in enumerate(windows.to_dict("records")):
            score = float(scores[index])
            document = {"@timestamp": row["timestamp"].isoformat(), "source": {"ip": row["source_ip"]},
                        "window_id": row["window_id"], "detector": name, "score": score,
                        "threshold": threshold, "flagged": bool(score >= threshold)}
            if lineage["labels_available"]:
                document["malicious"] = bool(row["malicious"])
                document["reference_label"] = row["reference_label"]
            documents.append(document)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ndjson(output_path, documents)
    return {"predictions": len(documents), "windows": len(windows),
            "labels_available": lineage["labels_available"], "output": str(output_path)}


def evaluate_predictions(prediction_path):
    from collections import defaultdict
    grouped = defaultdict(list)
    with Path(prediction_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSON on line {line_number}: {exc}") from exc
            if "malicious" in row and row["malicious"] is not None:
                grouped[row["detector"]].append(row)
    report = {}
    for detector, rows in grouped.items():
        truth = np.asarray([bool(row["malicious"]) for row in rows], dtype=bool)
        scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
        thresholds = {float(row["threshold"]) for row in rows}
        if len(thresholds) != 1:
            raise ValueError(f"Detector {detector} has inconsistent thresholds")
        report[detector] = evaluation(pd.DataFrame({"timestamp": pd.to_datetime([row["@timestamp"] for row in rows], utc=True),
                                                     "source_ip": [row["source"]["ip"] for row in rows],
                                                     "malicious": truth}), scores, thresholds.pop())
        report[detector]["reference_labels"] = sorted({label for row in rows for label in row.get("reference_label", "").split("|") if label})
    if not report:
        return {"status": "unavailable", "reason": "Prediction rows have no labeled reference outcomes; include labels in input to evaluate"}
    return report


def flow_documents(flows):
    for row in flows.to_dict("records"):
        yield {"@timestamp": row["timestamp"].isoformat(), "event": {"id": row["event_id"]},
               "source": {"ip": row["source_ip"], "port": int(row["source_port"])},
               "destination": {"ip": row["destination_ip"], "port": int(row["destination_port"])},
               "network": {"bytes": float(row["bytes_total"]), "packets": float(row["packets_total"]), "transport": {6: "tcp", 17: "udp", 1: "icmp"}.get(int(row["protocol"]), "other"), "iana_number": int(row["protocol"])},
               "duration": float(row["duration"]), "rate": float(row["rate"]), "malicious": bool(row["malicious"]), "reference_label": row["label"]}


def feature_documents(windows):
    for row in windows.to_dict("records"):
        yield {"@timestamp": row["timestamp"].isoformat(), "window_id": row["window_id"], "source": {"ip": row["source_ip"]},
               "split": row["split"], "malicious": bool(row["malicious"]), "reference_label": row["reference_label"],
               **{f: float(row[f]) for f in FEATURES}}


def write_ndjson(path, documents):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=True, allow_nan=False) + "\n")


def run_pipeline(input_path, output_path, max_rows=None):
    import importlib.metadata
    import platform
    import joblib
    flows, lineage = load_flows(input_path, max_rows)
    windows = split_windows(aggregate_windows(flows))
    models, metrics, findings = train_evaluate(windows)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    write_ndjson(output / "flows.ndjson", flow_documents(flows))
    write_ndjson(output / "features.ndjson", feature_documents(windows))
    write_ndjson(output / "findings.ndjson", findings)
    write_ndjson(output / "predictions.ndjson", (row for row in findings if row["detector"] != "scan_rule" and row["score"] is not None))
    joblib.dump(models, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    split_ranges = {}
    for split, part in windows.groupby("split"):
        split_ranges[split] = {"start": part["timestamp"].min().isoformat(), "end": part["timestamp"].max().isoformat(), "unique_timestamps": int(part["timestamp"].nunique()), "windows": len(part), "malicious_windows": int(part["malicious"].sum()), "reference_labels": sorted(label_set(part["reference_label"]))}
    manifest = {"schema_version": 1, "dataset": "User-supplied CICIDS2017 full labeled flow CSV(s); results describe only the processed files/row cap, not the complete benchmark", "lineage": lineage,
                "counts": {"flows": len(flows), "windows": len(windows), "findings": len(findings), "predictions": sum(row["detector"] != "scan_rule" and row["score"] is not None for row in findings)}, "max_rows": max_rows,
                "window": "source-IP left-closed 1 minute; @timestamp is exclusive window end UTC", "split_policy": "sorted unique window timestamps; floor(60%) train, through floor(80%) validation, remaining test; no timestamp overlap",
                "split_ranges": split_ranges, "model_features": FEATURES, "label_policy": "Any non-BENIGN flow makes its source window malicious; labels retained only as reference, never model inputs",
                "detector_status": {name: metrics[name]["status"] for name in ["isolation_forest", "supervised", "random_forest", "deep_learning", "scan_rule"]},
                "limitations": ["failure_ratio defaults to zero because failure outcome is unavailable", "Window labels can mix benign and attack flows", "FP/host-day uses observed host UTC days, not continuous monitored exposure", "Models are exploratory and not production security controls"],
                "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib", "requests", "tzdata", "torch"]}},
                "artifacts": {}, "publication": {"status": "not_requested"}}
    for name in ["flows.ndjson", "features.ndjson", "findings.ndjson", "predictions.ndjson", "metrics.json", "model.joblib"]:
        hasher = hashlib.sha256()
        with (output / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        manifest["artifacts"][name] = {"sha256": hasher.hexdigest(), "size_bytes": (output / name).stat().st_size}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest
