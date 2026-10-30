from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api import main as api


pytestmark = pytest.mark.skipif(
    not api.FLOW_STORE.is_file() or not api.FLOW_STORE_MANIFEST.is_file(),
    reason="Full benchmark flow store has not been built locally",
)


def test_full_benchmark_store_matches_saved_metadata_and_queries_all_rows():
    client = TestClient(api.app)
    status_response = client.get("/api/flows/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["ready"] is True
    assert status["input_rows"] == 4_610_049
    assert status["valid_flows"] == 4_553_532
    assert status["indexed_records"] == status["valid_flows"]
    assert status["feature_windows"] == 197_941
    assert status["topology_aggregated_flow_count"] == status["valid_flows"]
    assert status["topology_aggregated_suspicious_count"] == 2_259_959

    manifest = json.loads(api.FLOW_STORE_MANIFEST.read_text(encoding="utf-8"))
    assert status["dataset_counts"] == manifest["dataset_counts"]

    first_response = client.get("/api/flows", params={"dataset": "expanded", "page": 1, "page_size": 20, "sort": "timestamp", "order": "asc"})
    assert first_response.status_code == 200
    first = first_response.json()
    assert first["total"] == status["valid_flows"]
    assert len(first["items"]) == 20
    assert {item["dataset"] for item in first["items"]} <= {"CICIDS2017", "CICDDoS2019"}
    assert all(item["source_file"] and item["label"] and item["timestamp"] for item in first["items"])

    second = client.get("/api/flows", params={"dataset": "expanded", "page": 2, "page_size": 20, "sort": "timestamp", "order": "asc"}).json()
    first_ids = {item["event_id"] for item in first["items"]}
    second_ids = {item["event_id"] for item in second["items"]}
    assert len(second["items"]) == 20
    assert first_ids.isdisjoint(second_ids)

    sample = first["items"][0]
    filtered = client.get(
        "/api/flows",
        params={
            "dataset": sample["dataset"],
            "source_ip": sample["source_ip"],
            "destination_port": sample["destination_port"],
            "start": sample["timestamp"],
            "end": sample["timestamp"],
            "page_size": 5,
        },
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] >= 1
    assert any(item["event_id"] == sample["event_id"] for item in filtered.json()["items"])

    stats_response = client.get("/api/flows/stats", params={"dataset": "expanded"})
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["total_flows"] == status["valid_flows"]
    assert sum(stats["labels"].values()) == status["valid_flows"]
    assert sum(stats["protocols"].values()) == status["valid_flows"]

    saved_metrics = json.loads((api.EXPANDED / "metrics.json").read_text(encoding="utf-8"))["dataset"]
    assert stats["labels"] == saved_metrics["row_labels"]

    topology_response = client.get("/api/flows/topology", params={"dataset": "expanded", "limit": 25})
    assert topology_response.status_code == 200
    edges = topology_response.json()["edges"]
    assert 1 <= len(edges) <= 25
    assert all(edge["source_ip"] and edge["destination_ip"] and edge["flow_count"] > 0 for edge in edges)
