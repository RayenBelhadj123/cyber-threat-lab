"""Synthetic fixtures only: these tests do not measure benchmark performance."""
import json
from pathlib import Path
import pandas as pd
import pytest


def synthetic_rows(n=15):
    return [{" Source IP": "192.0.2.1", "Destination IP": "198.51.100.2", "Source Port": 1234,
             "Destination Port": 80+i, "Protocol": 6, "Timestamp": f"03/07/2017 09:{i:02}:10",
             "Flow Duration": 1000, "Total Fwd Packets": 2, "Total Backward Packets": 1,
             "Total Length of Fwd Packets": 200, "Total Length of Bwd Packets": 100,
             "Flow Bytes/s": 300, " Label": "BENIGN" if i % 3 else "PortScan"} for i in range(n)]


def test_windows_end_timestamp_and_chronological_splits(tmp_path):
    from cyberlab.pipeline import load_flows, aggregate_windows, split_windows, FEATURES
    rows = synthetic_rows(15)
    rows += [dict(rows[0], **{"Destination Port": 443, "Destination IP": "203.0.113.5"})]
    p = tmp_path / "flows.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    flows, _ = load_flows(p)
    windows = split_windows(aggregate_windows(flows))
    first = windows.iloc[0]
    assert first["timestamp"].isoformat() == "2017-07-03T12:01:00+00:00"
    assert first["flow_count"] == 2
    assert first["bytes_total"] == 600
    assert first["unique_ports"] == 2
    assert first["failure_ratio"] == 0
    assert windows.groupby("timestamp")["split"].nunique().max() == 1
    assert windows.groupby("split").size().to_dict() == {"train": 9, "validation": 3, "test": 3}
    assert not set(FEATURES) & {"label", "reference_label", "malicious", "source_ip", "timestamp"}
    with pytest.raises(ValueError, match="timestamp"):
        split_windows(windows.iloc[:2])


def test_models_use_benign_validation_and_report_unseen_attacks(tmp_path):
    from cyberlab.pipeline import load_flows, aggregate_windows, split_windows, train_evaluate
    rows = synthetic_rows(20)
    for i, row in enumerate(rows):
        row[" Label"] = "BENIGN" if i % 2 else ("PortScan" if i < 16 else "NovelAttack")
    p = tmp_path / "flows.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    w = split_windows(aggregate_windows(load_flows(p)[0]))
    models, metrics, findings = train_evaluate(w)
    assert metrics["isolation_forest"]["status"] == "available"
    assert metrics["isolation_forest"]["threshold_source"] == "benign_validation_99th_percentile"
    assert metrics["supervised"]["status"] == "available"
    assert metrics["random_forest"]["status"] == "available"
    assert metrics["deep_learning"]["status"] == "available"
    assert set(metrics["deep_learning"]["evaluation"]) == {"train", "validation", "test"}
    assert metrics["deep_learning"]["evaluation"]["test"]["confusion_matrix"]
    assert metrics["unseen_test_attacks"]["labels"] == ["NovelAttack"]
    assert len(findings) == len(w) * 5
    assert set(findings[0]) >= {"@timestamp", "source", "window_id", "detector", "score", "threshold", "flagged", "split", "malicious", "reference_label", "features"}
    w["malicious"] = False
    w["reference_label"] = "BENIGN"
    _, unavailable, _ = train_evaluate(w)
    assert unavailable["supervised"]["status"] == "unavailable"
    w.loc[w["split"] == "validation", "malicious"] = True
    _, unavailable, missing = train_evaluate(w)
    assert unavailable["isolation_forest"]["status"] == "unavailable"
    assert all(f["score"] is None for f in missing if f["detector"] == "isolation_forest")


def test_evaluation_reports_confusion_matrix_and_auc_support():
    from cyberlab.pipeline import evaluation
    windows = pd.DataFrame({"timestamp": pd.to_datetime(["2017-07-06T00:01:00Z"] * 4),
                            "source_ip": ["192.0.2.1"] * 4,
                            "malicious": [False, False, True, True]})
    result = evaluation(windows, [0.1, 0.8, 0.4, 0.9], 0.5)
    assert result["confusion_matrix"] == {"true_negative": 1, "false_positive": 1,
                                           "false_negative": 1, "true_positive": 1}
    assert result["roc_auc"] == 0.75
    assert result["balanced_accuracy"] == 0.5


def test_evaluation_handles_single_class_without_metric_warnings():
    import warnings
    from cyberlab.pipeline import evaluation
    windows = pd.DataFrame({"timestamp": pd.to_datetime(["2017-07-06T00:01:00Z"] * 4),
                            "source_ip": ["192.0.2.1"] * 4, "malicious": [False] * 4})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluation(windows, [0.1, 0.8, 0.1, 0.1], 0.5)
    assert result["roc_auc"] is None
    assert result["confusion_matrix"]["false_positive"] == 1
    assert result["balanced_accuracy"] == 0.75


def test_predict_and_evaluate_saved_models_on_labeled_csv(tmp_path):
    import joblib
    from cyberlab.pipeline import (aggregate_windows, evaluate_predictions, load_flows,
                                   predict_pipeline, split_windows, train_evaluate)
    source = tmp_path / "flows.csv"
    pd.DataFrame(synthetic_rows(20)).to_csv(source, index=False)
    windows = split_windows(aggregate_windows(load_flows(source)[0]))
    models, _, _ = train_evaluate(windows)
    model_path = tmp_path / "model.joblib"
    joblib.dump(models, model_path)
    prediction_path = tmp_path / "predictions.ndjson"
    result = predict_pipeline(source, model_path, prediction_path)
    assert result["predictions"] > 0
    assert result["labels_available"] is True
    report = evaluate_predictions(prediction_path)
    assert report["deep_learning"]["status"] == "available"
    assert report["deep_learning"]["windows"] == len(windows)
    assert "confusion_matrix" in report["deep_learning"]


def test_predict_accepts_unlabeled_csv(tmp_path):
    import joblib
    from cyberlab.pipeline import (aggregate_windows, load_flows, predict_pipeline,
                                   split_windows, train_evaluate)
    labeled = tmp_path / "labeled.csv"
    pd.DataFrame(synthetic_rows(20)).to_csv(labeled, index=False)
    windows = split_windows(aggregate_windows(load_flows(labeled)[0]))
    models, _, _ = train_evaluate(windows)
    model_path = tmp_path / "model.joblib"
    joblib.dump(models, model_path)
    rows = synthetic_rows(5)
    for row in rows:
        row.pop(" Label")
    unlabeled = tmp_path / "unlabeled.csv"
    pd.DataFrame(rows).to_csv(unlabeled, index=False)
    result = predict_pipeline(unlabeled, model_path, tmp_path / "predictions.ndjson")
    assert result["labels_available"] is False
    assert result["predictions"] > 0


def test_windows_dash_labels_cp1252(tmp_path):
    from cyberlab.pipeline import load_flows
    row = synthetic_rows(1)[0]
    row[" Label"] = "Web Attack – Brute Force"
    p = tmp_path / "cp1252.csv"
    pd.DataFrame([row]).to_csv(p, index=False, encoding="cp1252")
    flows, manifest = load_flows(p)
    assert flows.iloc[0]["label"] == "Web Attack – Brute Force"
    assert manifest["files"][0]["encoding"] == "cp1252"


def test_cli_writes_complete_artifacts_and_lineage(tmp_path):
    import subprocess
    import sys
    import joblib
    p = tmp_path / "flows.csv"
    pd.DataFrame(synthetic_rows(20)).to_csv(p, index=False)
    out = tmp_path / "artifacts"
    result = subprocess.run([sys.executable, "-m", "cyberlab", "run", "--input", str(p), "--output", str(out), "--max-rows", "18"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["flows"] == 18
    assert manifest["counts"]["findings"] == manifest["counts"]["windows"] * 5
    assert manifest["counts"]["predictions"] > 0
    assert manifest["split_ranges"]["train"]["end"] < manifest["split_ranges"]["validation"]["start"]
    assert set(joblib.load(out / "model.joblib")["feature_names"]) == set(manifest["model_features"])
    flows = [json.loads(line) for line in (out / "flows.ndjson").read_text().splitlines()]
    assert flows[0]["source"]["ip"] == "192.0.2.1"
    assert flows[0]["network"]["bytes"] == 300
    assert flows[0]["network"]["transport"] == "tcp"
    assert "id" in flows[0]["event"]
    features = [json.loads(line) for line in (out / "features.ndjson").read_text().splitlines()]
    assert features[0]["flow_count"] == 1
    assert features[0]["source"]["ip"] == "192.0.2.1"
    assert (out / "metrics.json").is_file()
    assert (out / "predictions.ndjson").is_file()
    predicted = subprocess.run([sys.executable, "-m", "cyberlab", "predict", "--input", str(p), "--model", str(out / "model.joblib"), "--output", str(out / "fresh-predictions.ndjson")], capture_output=True, text=True)
    assert predicted.returncode == 0, predicted.stderr
    assert json.loads(predicted.stdout)["labels_available"] is True
    evaluated = subprocess.run([sys.executable, "-m", "cyberlab", "evaluate", "--predictions", str(out / "fresh-predictions.ndjson"), "--output", str(out / "evaluation.json")], capture_output=True, text=True)
    assert evaluated.returncode == 0, evaluated.stderr
    report = json.loads((out / "evaluation.json").read_text())
    assert report["deep_learning"]["status"] == "available"
    assert "confusion_matrix" in report["deep_learning"]


@pytest.fixture
def elastic_stub():
    """Local HTTP protocol double; not an Elasticsearch integration benchmark."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    state = {"indices": {}, "bulk_error": False, "refreshes": 0, "mappings": {}}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def answer(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_HEAD(self):
            self.send_response(200 if self.path.strip("/") in state["indices"] else 404)
            self.end_headers()
        def do_PUT(self):
            index = self.path.strip("/").split("/")[0]
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["indices"].setdefault(index, {})
            state["mappings"][index] = value
            self.answer(200, {"acknowledged": True})
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/_bulk":
                lines = [json.loads(line) for line in body.decode().splitlines()]
                items = []
                for action, doc in zip(lines[::2], lines[1::2]):
                    a = action["index"]
                    state["indices"][a["_index"]][a["_id"]] = doc
                    items.append({"index": {"status": 400 if state["bulk_error"] else 201, **({"error": {"reason": "test rejection"}} if state["bulk_error"] else {})}})
                self.answer(200, {"errors": state["bulk_error"], "items": items})
            elif self.path.endswith("/_refresh"):
                state["refreshes"] += 1
                self.answer(200, {"_shards": {"failed": 0}})
            elif self.path.endswith("/_mget"):
                index = self.path.strip("/").split("/")[0]
                ids = json.loads(body)["ids"]
                self.answer(200, {"docs": [{"_id": i, "found": i in state["indices"][index], "_source": state["indices"][index].get(i)} for i in ids]})
            else:
                self.answer(404, {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join()


def test_publish_batches_idempotent_checked_errors_and_explicit_mapping(tmp_path, elastic_stub):
    from cyberlab.pipeline import run_pipeline
    from cyberlab.publish import publish_artifacts
    p = tmp_path / "flows.csv"
    pd.DataFrame(synthetic_rows(20)).to_csv(p, index=False)
    out = tmp_path / "artifacts"
    manifest = run_pipeline(p, out)
    url, state = elastic_stub
    first = publish_artifacts(out, url, batch_size=7)
    second = publish_artifacts(out, url, batch_size=7)
    assert first == second
    assert len(state["indices"]["cyber-flows-v1"]) == manifest["counts"]["flows"]
    assert len(state["indices"]["cyber-findings-v1"]) == manifest["counts"]["findings"]
    props = state["mappings"]["cyber-flows-v1"].get("mappings", state["mappings"]["cyber-flows-v1"])["properties"]
    assert props["source"]["properties"]["ip"]["type"] == "ip"
    assert props["@timestamp"]["type"] == "date"
    assert state["refreshes"] == 6
    state["bulk_error"] = True
    with pytest.raises(RuntimeError, match="bulk"):
        publish_artifacts(out, url, batch_size=7)


def test_cli_publish_and_env_url(tmp_path, elastic_stub, monkeypatch, capsys):
    from cyberlab.__main__ import main
    p = tmp_path / "flows.csv"
    pd.DataFrame(synthetic_rows(15)).to_csv(p, index=False)
    out = tmp_path / "published"
    url, state = elastic_stub
    monkeypatch.setenv("ES_URL", url)
    assert main(["run", "--input", str(p), "--output", str(out), "--publish"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["publication"]["status"] == "verified"
    assert len(state["indices"]) == 3


def test_event_ids_do_not_depend_on_csv_numeric_inference(tmp_path):
    from cyberlab.pipeline import load_flows
    a, b = tmp_path / "integer.csv", tmp_path / "float.csv"
    rows = synthetic_rows(3)
    pd.DataFrame(rows).to_csv(a, index=False)
    for row in rows:
        row["Flow Bytes/s"] = float(row["Flow Bytes/s"])
    pd.DataFrame(rows).to_csv(b, index=False)
    assert load_flows(a)[0]["event_id"].tolist() == load_flows(b)[0]["event_id"].tolist()


def test_recursive_limit_invalid_time_missing_time_and_scan_rule(tmp_path):
    from cyberlab.pipeline import load_flows, aggregate_windows, split_windows, train_evaluate
    nested = tmp_path / "nested"
    nested.mkdir()
    p = nested / "a.csv"
    rows = synthetic_rows(15)
    rows[0]["Timestamp"] = "bad-time"
    pd.DataFrame(rows).to_csv(p, index=False)
    flows, lineage = load_flows(tmp_path, max_rows=10)
    assert lineage["input_rows"] == 10
    assert lineage["invalid_rows"] == 1
    windows = split_windows(aggregate_windows(flows))
    windows.loc[windows.index[0], "unique_ports"] = 20
    windows.loc[windows.index[1], "flow_count"] = 100
    _, _, findings = train_evaluate(windows)
    assert sum(f["flagged"] for f in findings if f["detector"] == "scan_rule") == 2
    pd.DataFrame(rows).drop(columns=["Timestamp"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="timestamp"):
        load_flows(p)


def test_load_cleans_and_keeps_real_identity_and_utc(tmp_path):
    from cyberlab.pipeline import load_flows
    rows = synthetic_rows(3)
    rows[1][" Source IP"] = "not-an-ip"
    rows[2]["Flow Bytes/s"] = "Infinity"
    p = tmp_path / "flows.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    flows, lineage = load_flows(p)
    assert len(flows) == 1
    assert flows.iloc[0]["timestamp"].isoformat() == "2017-07-03T12:00:10+00:00"
    assert flows.iloc[0]["source_ip"] == "192.0.2.1"
    assert lineage["invalid_rows"] == 2
    assert len(flows.iloc[0]["event_id"]) == 64
    assert load_flows(p)[0].iloc[0]["event_id"] == flows.iloc[0]["event_id"]
    pd.DataFrame(rows).drop(columns=[" Source IP"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="source_ip"):
        load_flows(p)
