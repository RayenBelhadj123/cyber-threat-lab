from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from api import main as api
from cyberlab.flow_store import build_flow_store, flow_stats


def _row(source: str, destination: str, dst_port: str, protocol: str, label: str, stamp: str):
    return {
        "Source IP": source,
        "Destination IP": destination,
        "Source Port": "51000",
        "Destination Port": dst_port,
        "Protocol": protocol,
        "Timestamp": stamp,
        "Flow Duration": "1000",
        "Total Fwd Packets": "2",
        "Total Backward Packets": "1",
        "Total Length of Fwd Packets": "120",
        "Total Length of Bwd Packets": "60",
        "Flow Bytes/s": "180",
        "Label": label,
    }


def _entry(tmp_path: Path, dataset: str, rows: list[dict[str, str]]):
    path = tmp_path / f"{dataset}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "dataset": dataset,
        "path": path,
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_rows": len(rows),
    }


def test_build_indexes_valid_rows_and_audits_invalids_and_duplicates(tmp_path):
    benign = _row("10.0.0.1", "10.0.0.2", "80", "6", "BENIGN", "07/03/2017 09:00:00 AM")
    entries = [
        _entry(tmp_path, "CICIDS2017", [benign, benign.copy(), _row("bad-ip", "10.0.0.3", "443", "6", "Bot", "07/03/2017 09:01:00 AM")]),
        _entry(tmp_path, "CICDDoS2019", [_row("10.0.0.4", "10.0.0.5", "53", "17", "DrDoS_NTP", "2018-12-01 13:18:00")]),
    ]

    result = build_flow_store(entries, tmp_path / "flows.sqlite", chunk_size=1)

    assert result["input_rows"] == 4
    assert result["valid_flows"] == 2
    assert result["invalid_rows"] == 1
    assert result["duplicate_rows"] == 1
    assert result["dataset_counts"] == {"CICDDoS2019": 1, "CICIDS2017": 1}
    with sqlite3.connect(tmp_path / "flows.sqlite") as connection:
        cached_stats = connection.execute("SELECT summary_json FROM flow_stats_summary WHERE dataset='expanded'").fetchone()
    assert cached_stats is not None
    assert json.loads(cached_stats[0])["total_flows"] == 2
    assert result["integrity_check"] == "ok"


def test_expanded_top_sources_and_ports_are_ranked_across_both_datasets(tmp_path):
    entries = []
    for dataset_index, dataset in enumerate(("CICIDS2017", "CICDDoS2019")):
        rows = []
        for candidate in range(12):
            source = f"10.{dataset_index + 1}.0.{candidate + 1}"
            port = str((1000 if dataset_index == 0 else 2000) + candidate + 1)
            for occurrence in range(101):
                row = _row(source, "198.51.100.1", port, "6", "BENIGN", "07/03/2017 09:00:00 AM")
                row["Source Port"] = str(40000 + occurrence)
                rows.append(row)
        for occurrence in range(100):
            row = _row("192.0.2.1", "198.51.100.1", "443", "6", "BENIGN", "07/03/2017 09:00:00 AM")
            row["Source Port"] = str(50000 + occurrence)
            rows.append(row)
        entries.append(_entry(tmp_path, dataset, rows))

    store = tmp_path / "flows.sqlite"
    build_flow_store(entries, store, chunk_size=500)
    expanded = flow_stats(store, dataset="expanded")

    assert expanded["top_sources"][0] == {"source_ip": "192.0.2.1", "flows": 200}
    assert expanded["top_destination_ports"][0] == {"port": 443, "flows": 200}


def test_outdated_flow_stats_summary_is_rebuilt(tmp_path):
    rows = [
        _row("10.0.0.1", "10.0.0.2", "80", "6", "BENIGN", "07/03/2017 09:00:00 AM"),
        _row("10.0.0.3", "10.0.0.4", "443", "6", "Bot", "07/03/2017 09:01:00 AM"),
    ]
    store = tmp_path / "flows.sqlite"
    build_flow_store([_entry(tmp_path, "CICIDS2017", rows)], store)
    with sqlite3.connect(store) as connection:
        summary = json.loads(connection.execute("SELECT summary_json FROM flow_stats_summary WHERE dataset='expanded'").fetchone()[0])
        summary["_summary_version"] = -1
        summary["total_flows"] = 999
        connection.execute("UPDATE flow_stats_summary SET summary_json=? WHERE dataset='expanded'", (json.dumps(summary),))

    assert flow_stats(store, dataset="expanded")["total_flows"] == 2


def test_full_flow_api_filters_paginates_sorts_and_keeps_dataset_scope(tmp_path, monkeypatch):
    benign = _row("10.0.0.1", "10.0.0.2", "80", "6", "BENIGN", "07/03/2017 09:00:00 AM")
    entries = [
        _entry(tmp_path, "CICIDS2017", [benign, _row("10.0.0.1", "10.0.0.3", "443", "6", "DoS Hulk", "07/03/2017 09:01:00 AM")]),
        _entry(tmp_path, "CICDDoS2019", [_row("10.0.0.4", "10.0.0.5", "53", "17", "DrDoS_NTP", "2018-12-01 13:18:00")]),
    ]
    store = tmp_path / "flows.sqlite"
    metadata = build_flow_store(entries, store, chunk_size=1)
    with sqlite3.connect(store) as connection:
        connection.execute("DROP TABLE flow_stats_summary")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.json").write_text(
        '{"dataset":{"input_rows":3,"valid_flows":3,"window_count":0,"row_labels":{"BENIGN":1,"DoS Hulk":1,"DrDoS_NTP":1}}}', encoding="utf-8"
    )
    prep = tmp_path / "preparation.json"
    prep.write_text(
        '{"lineage":{"input_rows":3,"valid_rows":3,"files":[{"dataset":"CICIDS2017","valid_unique_flows":2,"windows":0},{"dataset":"CICDDoS2019","valid_unique_flows":1,"windows":0}]}}', encoding="utf-8"
    )
    monkeypatch.setattr(api, "FLOW_STORE", store)
    monkeypatch.setattr(api, "FLOW_STORE_MANIFEST", store.with_suffix(".manifest.json"))
    monkeypatch.setattr(api, "EXPANDED", metrics_dir)
    monkeypatch.setattr(api, "PREPARATION", prep)
    client = TestClient(api.app)

    response = client.get("/api/flows", params={"dataset": "expanded", "page": 1, "page_size": 1, "sort": "timestamp", "order": "asc"})
    assert response.status_code == 200
    first = response.json()
    assert first["total"] == 3
    assert first["page"] == 1
    assert first["items"][0]["dataset"] == "CICIDS2017"
    assert first["items"][0]["timestamp"] < "2018"
    assert first["data_mode"] == "Historical benchmark"

    page_two = client.get("/api/flows", params={"dataset": "expanded", "page": 2, "page_size": 1, "sort": "timestamp", "order": "asc"}).json()
    assert page_two["items"][0]["event_id"] != first["items"][0]["event_id"]

    filtered = client.get("/api/flows", params={"dataset": "CICIDS2017", "destination_port": 443, "label": "DoS Hulk"}).json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["destination_port"] == 443
    assert filtered["items"][0]["label"] == "DoS Hulk"

    extension = client.get("/api/flows", params={"dataset": "CICDDoS2019"}).json()
    assert extension["total"] == 1
    assert extension["items"][0]["source_file"] == "CICDDoS2019.csv"

    invalid_sort = client.get("/api/flows", params={"sort": "timestamp; DROP TABLE flows"})
    assert invalid_sort.status_code == 422
    stats = client.get("/api/flows/stats", params={"dataset": "expanded"}).json()
    assert stats["total_flows"] == 3
    assert sum(stats["labels"].values()) == 3
    assert stats["protocols"] == {"TCP": 2, "UDP": 1}
    assert stats["suspicious_flows"] == 2
    assert sum(item["flows"] for item in stats["hourly"]) == 3
    assert stats["mean_packets"] == 3
    status = client.get("/api/flows/status").json()
    assert status["ready"] is True
    assert status["dashboard_statistics_cached"] is True
    with sqlite3.connect(store) as connection:
        summary = json.loads(connection.execute("SELECT summary_json FROM flow_stats_summary WHERE dataset='expanded'").fetchone()[0])
        summary["total_flows"] += 1
        connection.execute("UPDATE flow_stats_summary SET summary_json=? WHERE dataset='expanded'", (json.dumps(summary),))
    inconsistent = client.get("/api/flows/status").json()
    assert inconsistent["ready"] is False
    assert "cached dashboard statistics" in inconsistent["detail"]
    summary["total_flows"] -= 1
    with sqlite3.connect(store) as connection:
        connection.execute("UPDATE flow_stats_summary SET summary_json=? WHERE dataset='expanded'", (json.dumps(summary),))
    topology = client.get("/api/flows/topology", params={"dataset": "expanded", "limit": 10}).json()
    assert sum(edge["flow_count"] for edge in topology["edges"]) == 3
    assert all(edge["first_seen"] <= edge["last_seen"] for edge in topology["edges"])


def test_full_flow_api_does_not_fall_back_when_full_store_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FLOW_STORE", tmp_path / "missing.sqlite")
    client = TestClient(api.app)
    missing = client.get("/api/flows")
    assert missing.status_code == 503
    assert "build_full_flow_store.py" in missing.json()["detail"]
