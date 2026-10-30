"""Read-only SOC API over saved artifacts and optional local services.

Historical benchmark and event times are never described as current telemetry.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import socket
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from cyberlab.flow_store import flow_stats, flow_topology, query_flows

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.getenv("CYBERLAB_ARTIFACTS", ROOT / "artifacts"))
EXPANDED = Path(os.getenv("CYBERLAB_EXPANDED_BENCHMARK", ARTIFACTS / "model_benchmark_expanded"))
REPORT = Path(os.getenv("CYBERLAB_REPORT_DATA", ROOT / "report_data"))
FLOW_STORE = Path(os.getenv("CYBERLAB_FLOW_STORE", ARTIFACTS / "expanded_preparation" / "full_flows.sqlite"))
FLOW_STORE_MANIFEST = Path(os.getenv("CYBERLAB_FLOW_STORE_MANIFEST", FLOW_STORE.with_suffix(".manifest.json")))
PREPARATION = Path(os.getenv("CYBERLAB_EXPANDED_PREPARATION", ARTIFACTS / "expanded_preparation" / "preparation.json"))
ES_URL = os.getenv("CYBERLAB_ELASTICSEARCH_URL", "http://127.0.0.1:19200").rstrip("/")
KIBANA_URL = os.getenv("CYBERLAB_KIBANA_URL", "http://127.0.0.1:15601").rstrip("/")
STREAMLIT_URL = os.getenv("CYBERLAB_STREAMLIT_URL", "http://127.0.0.1:8502").rstrip("/")

app = FastAPI(title="Cyber Threat Lab SOC API", version="1.0.0", description="Read-only APIs for saved cybersecurity evidence and real service health.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CYBERLAB_API_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","), allow_methods=["GET"], allow_headers=["*"])


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def es_request(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(f"{ES_URL}/{path.lstrip('/')}", data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=2.0) as response:
        return json.loads(response.read())


def records(path: Path, *, offset: int, limit: int, query: str = "", flagged: bool | None = None, protocol: str = "", source_ip: str = "") -> tuple[list[dict], int]:
    """Memory-bounded full-file scan; pagination/filtering covers every saved line."""
    if not path.is_file():
        return [], 0
    q, ip, proto = query.casefold(), source_ip.casefold(), protocol.casefold()
    found: list[dict] = []
    matched = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                is_flagged = item.get("flagged", item.get("malicious"))
                if flagged is not None and bool(is_flagged) != flagged:
                    continue
                src = str(item.get("source", {}).get("ip", item.get("source_ip", "")))
                if ip and ip not in src.casefold():
                    continue
                transport = str(item.get("network", {}).get("transport", item.get("protocol", "")))
                if proto and transport.casefold() != proto:
                    continue
                haystack = json.dumps(item, ensure_ascii=False).casefold()
                if q and q not in haystack:
                    continue
                if matched >= offset and len(found) < limit:
                    found.append(item)
                matched += 1
    except OSError as exc:
        raise HTTPException(500, f"Could not read saved evidence: {exc}") from exc
    return found, matched


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(1024 * 1024), b""))


def probe_http(name: str, url: str) -> dict:
    started = time.monotonic()
    try:
        with urlopen(Request(url), timeout=2.0) as response:
            code = response.status
        return {"name": name, "status": "healthy" if code < 500 else "degraded", "http_status": code, "latency_ms": round((time.monotonic()-started)*1000, 1), "checked_at": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        return {"name": name, "status": "unavailable", "detail": str(exc)[:200], "checked_at": datetime.now(timezone.utc).isoformat()}


def probe_tcp(name: str, host: str, port: int) -> dict:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return {"name": name, "status": "healthy", "checked_at": datetime.now(timezone.utc).isoformat()}
    except OSError as exc:
        return {"name": name, "status": "unavailable", "detail": str(exc)[:160], "checked_at": datetime.now(timezone.utc).isoformat()}


def benchmark_data() -> tuple[dict | None, Path]:
    metrics = read_json(EXPANDED / "metrics.json")
    return metrics, EXPANDED


@app.get("/api/health")
def api_health():
    return {"status": "healthy", "service": "cyber-threat-lab-api", "mode": "read-only", "checked_at": datetime.now(timezone.utc).isoformat()}


@app.get("/api/summary")
def summary():
    metrics, source = benchmark_data()
    manifest = read_json(ROOT / "data" / "cicids2017_full_manifest.json") or {}
    ext = read_json(ROOT / "data" / "cicddos2019_extension" / "source_manifest.json") or {}
    if metrics:
        dataset = metrics.get("dataset", {})
        models = metrics.get("models", {})
        selected_name = metrics.get("selection", {}).get("model")
        test = models.get(selected_name, {}).get("test_metrics", {})
        return {"data_mode": "Historical benchmark", "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source), "benchmark_scope": dataset.get("description"), "raw_input_rows": dataset.get("input_rows"), "valid_flows": dataset.get("valid_flows"), "feature_windows": dataset.get("window_count"), "positive_windows": dataset.get("positive_windows"), "attack_window_labels": dataset.get("positive_window_labels", {}), "selected_model": selected_name, "held_out_test": test, "candidates": len(models), "time_range_utc": dataset.get("positive_window_time_range_utc"), "live_source_configured": bool(os.getenv("CYBERLAB_LIVE_ELASTICSEARCH_URL")), "downloaded_datasets": {"CICIDS2017_full_week": manifest.get("rows"), "CIC-DDoS2019_extension_rows": ext.get("total_rows") or ext.get("rows")}}
    base = read_json(ARTIFACTS / "metrics.json") or {}
    return {"data_mode": "Historical benchmark", "source": str(ARTIFACTS), "benchmark_scope": "Expanded benchmark is not available; legacy artifacts are separate and must not be represented as the full-week run.", "raw_input_rows": base.get("dataset", {}).get("input_rows"), "feature_windows": base.get("dataset", {}).get("window_count"), "positive_windows": base.get("dataset", {}).get("positive_windows"), "selected_model": None, "candidates": 0, "expanded_benchmark_available": False}


@app.get("/api/findings")
def findings(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200), q: str = "", flagged: bool | None = None, source_ip: str = ""):
    path = ARTIFACTS / "findings.ndjson"
    rows, total = records(path, offset=(page-1)*page_size, limit=page_size, query=q, flagged=flagged, source_ip=source_ip)
    return {"items": rows, "page": page, "page_size": page_size, "total": total, "source": "Historical saved findings (2017); not current alerts"}


@app.get("/api/findings/{window_id}")
def finding_detail(window_id: str):
    for filename in ("findings.ndjson", "features.ndjson"):
        path = ARTIFACTS / filename
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("window_id") == window_id or obj.get("event", {}).get("id") == window_id:
                        return {"record_type": filename.removesuffix(".ndjson"), "historical": True, "record": obj}
    raise HTTPException(404, "Finding/window not present in saved local artifacts")


@app.get("/api/traffic")
def traffic(page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=500), q: str = "", protocol: str = "", source_ip: str = ""):
    rows, total = records(ARTIFACTS / "flows.ndjson", offset=(page-1)*page_size, limit=page_size, query=q, protocol=protocol, source_ip=source_ip)
    return {"items": rows, "page": page, "page_size": page_size, "total": total, "source": "Local flow artifact; CICIDS2017 Thursday capture only"}


@app.get("/api/traffic/stats")
def traffic_stats():
    path = ARTIFACTS / "flows.ndjson"
    counts: dict[str, int] = {}
    labels: dict[str, int] = {}
    sources: dict[str, int] = {}
    ports: dict[str, int] = {}
    per_hour: dict[str, int] = {}
    total = 0
    malicious = 0
    packet_sum = 0.0
    duration_sum = 0.0
    for row in iter_records(path):
        total += 1
        malicious += int(bool(row.get("malicious")))
        network = row.get("network", {})
        proto = str(network.get("transport", "unknown")).lower()
        counts[proto] = counts.get(proto, 0) + 1
        labels[str(row.get("reference_label", "Unknown"))] = labels.get(str(row.get("reference_label", "Unknown")), 0) + 1
        src = str(row.get("source", {}).get("ip", "Unknown")); sources[src] = sources.get(src, 0) + 1
        port = str(row.get("destination", {}).get("port", "Unknown")); ports[port] = ports.get(port, 0) + 1
        stamp = str(row.get("@timestamp", ""))[:13]
        if stamp: per_hour[stamp] = per_hour.get(stamp, 0) + 1
        packet_sum += float(network.get("packets", 0) or 0)
        duration_sum += float(row.get("duration", 0) or 0)
    return {"mode": "Historical only", "source": "artifacts/flows.ndjson (legacy Thursday capture; NOT the expanded 4.6M-row benchmark)", "total_flows": total, "suspicious_flows": malicious, "protocols": counts, "labels": labels, "top_sources": sorted(({"source_ip": k, "flows": v} for k,v in sources.items()), key=lambda x:-x["flows"])[:12], "top_destination_ports": sorted(({"port": k, "flows": v} for k,v in ports.items()), key=lambda x:-x["flows"])[:12], "hourly": sorted(({"hour": k, "flows": v} for k,v in per_hour.items()), key=lambda x:x["hour"]), "mean_packets": packet_sum/total if total else None, "mean_duration_microseconds": duration_sum/total if total else None}


def full_flow_store_status() -> dict[str, Any]:
    """Prove the local raw-flow store matches the saved full-benchmark counts."""
    meta = read_json(FLOW_STORE_MANIFEST)
    if not FLOW_STORE.is_file() or not meta:
        return {"status": "unavailable", "ready": False, "path": str(FLOW_STORE),
                "detail": "Full-benchmark raw-flow store is not built; run python scripts/build_full_flow_store.py."}
    reasons = []
    try:
        with sqlite3.connect(FLOW_STORE) as db:
            actual = int(db.execute("SELECT COUNT(*) FROM flows").fetchone()[0])
            actual_by_dataset = {name: count for name, count in db.execute("SELECT dataset,COUNT(*) FROM flows GROUP BY dataset")}
            topology_rows, topology_malicious = db.execute("SELECT COALESCE(SUM(flow_count),0),COALESCE(SUM(malicious_count),0) FROM flow_topology").fetchone()
    except (OSError, sqlite3.Error) as exc:
        return {"status": "invalid", "ready": False, "path": str(FLOW_STORE), "detail": f"Could not verify flow store: {exc}"}
    metrics = read_json(EXPANDED / "metrics.json") or {}
    dataset = metrics.get("dataset", {})
    lineage = (read_json(PREPARATION) or {}).get("lineage", {})
    expected_input = dataset.get("input_rows")
    expected_valid = dataset.get("valid_flows")
    prep_files = lineage.get("files", [])
    lineage_by_dataset: dict[str, int] = {}
    lineage_windows = 0
    for item in prep_files:
        lineage_by_dataset[item.get("dataset", "")] = lineage_by_dataset.get(item.get("dataset", ""), 0) + int(item.get("valid_unique_flows", 0))
        lineage_windows += int(item.get("windows", 0))
    stored_labels: dict[str, int] = {}
    for label_map in meta.get("label_counts", {}).values():
        for label_name, label_count in label_map.items():
            stored_labels[label_name] = stored_labels.get(label_name, 0) + int(label_count)
    expected_labels = dataset.get("row_labels", {})
    expected_malicious = sum(count for label_name, count in expected_labels.items() if label_name.upper() not in {"BENIGN", "UNKNOWN"})
    if meta.get("status") != "verified":
        reasons.append("store manifest is not verified")
    if actual != meta.get("database_record_count") or actual != meta.get("valid_flows"):
        reasons.append("SQLite row count differs from its verified manifest")
    if sum(actual_by_dataset.values()) != actual or actual_by_dataset != meta.get("dataset_counts"):
        reasons.append("per-dataset counts differ from the verified manifest")
    if expected_input is None or expected_valid is None or actual != expected_valid or meta.get("input_rows") != expected_input:
        reasons.append("store counts do not match saved expanded benchmark metadata")
    if lineage.get("input_rows") != expected_input or lineage.get("valid_rows") != expected_valid or lineage_by_dataset != actual_by_dataset:
        reasons.append("preparation lineage does not match saved benchmark metadata")
    if not expected_labels or stored_labels != expected_labels:
        reasons.append("stored reference-label totals differ from benchmark metadata")
    if lineage_windows != dataset.get("window_count"):
        reasons.append("prepared feature-window aggregation count differs from benchmark metadata")
    if topology_rows != actual or topology_malicious != expected_malicious:
        reasons.append("topology aggregation does not conserve flow and reference-label counts")
    return {"status": "ready" if not reasons else "inconsistent", "ready": not reasons,
            "path": str(FLOW_STORE), "data_mode": "Historical benchmark",
            "dataset_scope": ["CICIDS2017", "CICDDoS2019"], "input_rows": meta.get("input_rows"),
            "valid_flows": actual, "feature_windows": lineage_windows,
            "invalid_rows": meta.get("invalid_rows"), "duplicate_rows": meta.get("duplicate_rows"),
            "blank_rows": meta.get("blank_rows"), "dataset_counts": actual_by_dataset,
            "indexed_records": actual, "topology_aggregated_flow_count": topology_rows,
            "topology_aggregated_suspicious_count": topology_malicious,
            "risk_score_available": False, "integrity_check": meta.get("integrity_check"),
            "database_bytes": FLOW_STORE.stat().st_size,
            "detail": "; ".join(reasons) if reasons else "Flow, label, dataset, topology and feature-window counts match saved benchmark metadata."}


def _require_full_flow_store() -> dict[str, Any]:
    status = full_flow_store_status()
    if not status["ready"]:
        raise HTTPException(503, status["detail"])
    return status


@app.get("/api/flows/status")
def flows_status():
    return full_flow_store_status()


@app.get("/api/flows")
def full_flows(
    dataset: str = Query("expanded", pattern="^(expanded|CICIDS2017|CICDDoS2019)$"),
    page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=500),
    sort: str = Query("timestamp"), order: str = Query("desc", pattern="^(asc|desc)$"),
    q: str = "", source_ip: str = "", destination_ip: str = "",
    source_port: int | None = Query(None, ge=0, le=65535),
    destination_port: int | None = Query(None, ge=0, le=65535),
    protocol: int | None = Query(None, ge=0, le=255), label: str = "",
    start: datetime | None = None, end: datetime | None = None,
):
    status = _require_full_flow_store()
    try:
        result = query_flows(FLOW_STORE, dataset=dataset,
                             filters={"query": q, "source_ip": source_ip, "destination_ip": destination_ip,
                                      "source_port": source_port, "destination_port": destination_port,
                                      "protocol": protocol, "label": label, "start": start, "end": end},
                             page=page, page_size=page_size, sort=sort, order=order)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result.update({"data_mode": "Historical benchmark", "source": "Verified full-benchmark SQLite flow store",
                   "indexed_record_count": status["indexed_records"]})
    return result


@app.get("/api/flows/stats")
def full_flow_stats(dataset: str = Query("expanded", pattern="^(expanded|CICIDS2017|CICDDoS2019)$")):
    _require_full_flow_store()
    result = flow_stats(FLOW_STORE, dataset=dataset)
    result["source"] = "Verified full-benchmark SQLite flow store"
    return result


@app.get("/api/flows/topology")
def full_flow_topology(dataset: str = Query("expanded", pattern="^(expanded|CICIDS2017|CICDDoS2019)$"), limit: int = Query(100, ge=1, le=500)):
    _require_full_flow_store()
    return flow_topology(FLOW_STORE, dataset=dataset, limit=limit)


def iter_records(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@app.get("/api/ml")
def ml_analytics():
    metrics, source = benchmark_data()
    if not metrics:
        raise HTTPException(503, "Expanded full-data benchmark metrics are unavailable")
    return {"source": str(source), "scope": metrics.get("dataset"), "split": metrics.get("split"), "selection": metrics.get("selection"), "models": metrics.get("models"), "learning_curve": metrics.get("learning_curve"), "evaluation_by_capture": metrics.get("evaluation_by_capture"), "held_out_attack_detection": metrics.get("held_out_attack_detection"), "caveats": metrics.get("caveats", [])}


@app.get("/api/datasets")
def datasets():
    outputs = []
    manifests = [("CICIDS2017 complete-week download", ROOT/"data/cicids2017_full_manifest.json", "Downloaded source files; not alone proof of model use"), ("CIC-DDoS2019 extension", ROOT/"data/cicddos2019_extension/source_manifest.json", "Downloaded extension; see expanded benchmark manifest for actual use"), ("Expanded benchmark", EXPANDED/"metrics.json", "Model evaluation input (CICIDS2017 + CIC-DDoS2019; no subsampling)"), ("Legacy pipeline artifacts", ARTIFACTS/"manifest.json", "Legacy Thursday-only pipeline; separate from expanded benchmark")]
    for name, path, description in manifests:
        meta = read_json(path)
        outputs.append({"name": name, "available": meta is not None, "description": description, "manifest": meta})
    return {"datasets": outputs, "full_benchmark": read_json(EXPANDED/"metrics.json"), "note": "Downloaded, normalized, and model-evaluated datasets are separate states. No claim that every downloaded file was used unless the benchmark provenance records it."}


@app.get("/api/health/components")
def components_health():
    checks = [
        lambda: probe_http("Elasticsearch", f"{ES_URL}/_cluster/health"),
        lambda: probe_http("Kibana", f"{KIBANA_URL}/api/status"),
        lambda: probe_tcp("Kafka broker 1", os.getenv("CYBERLAB_KAFKA_HOST", "127.0.0.1"), int(os.getenv("CYBERLAB_KAFKA_PORT", "19092"))),
        lambda: probe_http("Spark master UI", os.getenv("CYBERLAB_SPARK_URL", "http://127.0.0.1:18180")),
        lambda: probe_http("Streamlit report workspace", STREAMLIT_URL),
        lambda: {"name": "Expanded ML benchmark", "status": "healthy" if (EXPANDED/"metrics.json").is_file() else "unavailable", "checked_at": datetime.now(timezone.utc).isoformat()},
        lambda: {"name": "Frozen inference artifact", "status": "healthy" if (REPORT/"detector_metadata.json").is_file() else "unavailable", "checked_at": datetime.now(timezone.utc).isoformat()},
    ]
    with ThreadPoolExecutor(max_workers=len(checks)) as pool:
        result = list(pool.map(lambda check: check(), checks))
    # Report live observations only if a separately configured live Elasticsearch endpoint
    # responds; the historical replay index is never treated as live by default.
    live_url = os.getenv("CYBERLAB_LIVE_ELASTICSEARCH_URL")
    if live_url:
        live = probe_http("Configured live Elasticsearch", live_url.rstrip("/") + "/_cluster/health")
        live["classification"] = "live source connectivity only; no current incident claim"
        result.append(live)
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "components": result}


@app.get("/api/investigations/{window_id}")
def investigation(window_id: str):
    output = {}
    for filename, key in (("findings.ndjson", "finding"), ("features.ndjson", "feature_window"), ("flows.ndjson", "related_flows")):
        path = ARTIFACTS / filename
        matches = []
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try: record = json.loads(line)
                    except json.JSONDecodeError: continue
                    if record.get("window_id") == window_id or record.get("event", {}).get("id") == window_id:
                        matches.append(record)
        output[key] = matches if key == "related_flows" else (matches[0] if matches else None)
    output["mode"] = "Historical saved evidence; no external threat intelligence"
    if not any(output[k] for k in ("finding", "feature_window", "related_flows")):
        raise HTTPException(404, "No saved evidence for this identifier")
    return output
