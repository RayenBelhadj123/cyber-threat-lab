"""Contract tests for the optional distributed cyber lab profile."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_bigdata_compose_has_required_pipeline_and_replica_services():
    if not shutil.which("docker"):
        pytest.skip("Docker Compose CLI is required to validate the distributed profile")
    result = subprocess.run(["docker", "compose", "-f", str(ROOT / "compose.yaml"),
                             "--profile", "bigdata", "--profile", "tools", "config", "--format", "json"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    compose = json.loads(result.stdout)
    services = compose["services"]
    required = {
        "kafka-1", "kafka-2", "kafka-3", "kafka-topics",
        "spark-master", "spark-worker-1", "spark-worker-2", "spark-stream", "spark-batch",
        "mongo-1", "mongo-2", "mongo-3", "mongo-init",
        "cassandra-1", "cassandra-2", "cassandra-3",
    }
    assert required <= services.keys()
    for service in services.values():
        for port in service.get("ports", []):
            assert port["host_ip"] == "127.0.0.1"
    assert services["kafka-1"]["image"] == "apache/kafka:4.2.2"
    assert services["mongo-1"]["image"] == "mongo:9.0.4"
    assert services["cassandra-1"]["image"] == "cassandra:5.0.9"
    assert services["spark-master"]["image"].startswith("cyberlab-spark:")
    assert services["mongo-1"].get("ports", []) == []
    assert services["spark-master"]["ports"][0]["published"] == "18180"
    assert services["spark-worker-1"]["mem_limit"]
    assert services["kafka-1"]["environment"]["KAFKA_DEFAULT_REPLICATION_FACTOR"] == "3"
    assert services["kafka-1"]["environment"]["KAFKA_MIN_INSYNC_REPLICAS"] == "2"
    assert services["kafka-1"]["environment"]["KAFKA_HEAP_OPTS"] == "-Xms256M -Xmx256M"
    for node, host_port in ((1, 19092), (2, 19093), (3, 19094)):
        listeners = services[f"kafka-{node}"]["environment"]["KAFKA_ADVERTISED_LISTENERS"]
        assert listeners == f"PLAINTEXT://kafka-{node}:9092,HOST://127.0.0.1:{host_port}"
    assert services["kafka-1"]["environment"]["KAFKA_LOG_CLEANER_DEDUPE_BUFFER_SIZE"] == "16777216"
    assert services["cassandra-1"]["environment"]["MAX_DIRECT_MEMORY_SIZE"] == "128M"
    assert int(services["cassandra-1"]["mem_limit"]) >= 1024 * 1024 * 1024
    assert services["cassandra-2"]["depends_on"]["cassandra-1"]["condition"] == "service_healthy"
    assert services["cassandra-3"]["depends_on"]["cassandra-2"]["condition"] == "service_healthy"
    stream_command = services["spark-stream"]["command"]
    stream_env = services["spark-stream"]["environment"]
    assert stream_env["KAFKA_FLOW_TOPIC"] == "cyber-flows"
    assert stream_env["KAFKA_PREDICTION_TOPIC"] == "cyber-predictions"
    assert "/opt/spark/jobs/spark_jobs/streaming.py" in stream_command
    assert "/app/checkpoints/stream" in stream_command
    assert "spark.jars.ivy=/tmp/ivy" in stream_command
    assert any(volume["target"] == "/tmp/ivy" for volume in services["spark-stream"]["volumes"])
    assert "/opt/spark/jobs/spark_jobs/batch.py" in services["spark-batch"]["command"]
    for worker in ("spark-worker-1", "spark-worker-2"):
        targets = {volume["target"] for volume in services[worker]["volumes"]}
        assert "/app/artifacts" in targets
        assert "/app/checkpoints" in targets
        worker_env = services[worker]["environment"]
        assert worker_env["MONGO_URI"].startswith("mongodb://mongo-1:27017")
        assert worker_env["CASSANDRA_HOSTS"] == "cassandra-1,cassandra-2,cassandra-3"
        assert worker_env["ELASTICSEARCH_URL"] == "http://elasticsearch:9200"
    topic_command = " ".join(services["kafka-topics"]["command"])
    assert '--topic "$$topic"' in topic_command
    assert '--topic ""' not in topic_command


def test_spark_image_pins_runtime_and_database_clients():
    dockerfile = (ROOT / "Dockerfile.spark").read_text()
    assert "FROM apache/spark:4.2.0" in dockerfile
    assert "COPY spark_jobs/ /opt/spark/jobs/spark_jobs/" in dockerfile
    assert "ENV PYTHONPATH=/opt/spark/jobs" in dockerfile
    assert "COPY scripts/init_cassandra.py /opt/spark/jobs/init_cassandra.py" in dockerfile
    assert "mkdir -p /tmp/ivy" in dockerfile and "chown 185:0 /tmp/ivy" in dockerfile
    assert "spark_jobs.common" in (ROOT / "spark_jobs" / "batch.py").read_text()
    assert "cyber-features-spark-batch-v1" in (ROOT / "spark_jobs" / "batch.py").read_text()
    assert "cyber-findings-spark-batch-v1" in (ROOT / "spark_jobs" / "batch.py").read_text()
    streaming = (ROOT / "spark_jobs" / "streaming.py").read_text()
    assert "cyber-features-spark-stream-v1" in streaming
    assert "cyber-findings-spark-stream-v1" in streaming
    assert 'os.environ.get("KAFKA_FLOW_TOPIC", "cyber-flows")' in streaming
    assert 'os.environ.get("KAFKA_PREDICTION_TOPIC", "cyber-predictions")' in streaming
    requirements = (ROOT / "requirements-spark.txt").read_text().splitlines()
    assert requirements == ["pymongo==4.15.4", "cassandra-driver==3.29.2", "requests==2.32.5"]


def test_replay_client_is_pinned_in_base_requirements():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    assert "kafka-python==2.2.15" in requirements


def test_json_safe_normalizes_driver_values_for_document_stores():
    from datetime import date, datetime, timezone
    from spark_jobs.common import json_safe
    value = {"ts": datetime(2017, 7, 6, tzinfo=timezone.utc), "day": date(2017, 7, 6), "items": {"b", "a"}}
    assert json_safe(value) == {"ts": "2017-07-06T00:00:00+00:00", "day": "2017-07-06", "items": ["a", "b"]}


def test_sink_documents_keep_elasticsearch_metadata_outside_source():
    from spark_jobs.common import build_sink_documents
    original = {"window_id": "w-1", "source": {"ip": "192.0.2.1"}, "_id": "stale"}
    mongo_document, json_document = build_sink_documents(original, "stable-id")
    assert mongo_document["_id"] == "stable-id"
    assert "_id" not in json_document
    assert original["_id"] == "stale"
    assert json_document == {"window_id": "w-1", "source": {"ip": "192.0.2.1"}}


def test_empty_sink_partition_is_a_noop_without_opening_clients():
    from spark_jobs.common import sink_partition
    assert sink_partition(iter(()), "features", "cyber-test", "batch_features") is None


def test_replay_source_reads_flows_and_predictions_from_artifacts(tmp_path):
    from scripts.stream_replay import iter_artifact_events
    flow = {"event": {"id": "f-1"}, "source": {"ip": "192.0.2.1"}}
    prediction = {"window_id": "w-1", "detector": "deep_learning"}
    (tmp_path / "flows.ndjson").write_text(json.dumps(flow) + "\n", encoding="utf-8")
    (tmp_path / "predictions.ndjson").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    assert list(iter_artifact_events(tmp_path)) == [
        ("cyber-flows", flow), ("cyber-predictions", prediction)]


def test_replay_source_can_route_to_isolated_topics(tmp_path):
    from scripts.stream_replay import iter_artifact_events
    flow = {"event": {"id": "f-1"}, "source": {"ip": "192.0.2.1"}}
    prediction = {"window_id": "w-1", "detector": "deep_learning"}
    (tmp_path / "flows.ndjson").write_text(json.dumps(flow) + "\n", encoding="utf-8")
    (tmp_path / "predictions.ndjson").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    assert list(iter_artifact_events(tmp_path, "cyber-flows-parity", "cyber-predictions-parity")) == [
        ("cyber-flows-parity", flow), ("cyber-predictions-parity", prediction)]


def test_watermark_marker_is_a_valid_stream_flow():
    from scripts.stream_replay import build_watermark_event
    from spark_jobs.common import event_topic, stable_storage_id
    marker = build_watermark_event("2017-07-06T16:15:00+00:00")
    assert event_topic(marker) == "cyber-flows"
    assert stable_storage_id(marker) == "replay-watermark-2017-07-06T16:15:00+00:00"
    assert marker["source"]["ip"] == "192.0.2.254"
    assert marker["reference_label"] == "__REPLAY_WATERMARK__"


def test_event_routing_and_storage_ids_are_deterministic():
    from spark_jobs.common import event_topic, stable_storage_id
    flow = {"event": {"id": "flow-1"}, "source": {"ip": "192.0.2.1"}}
    prediction = {"window_id": "window-1", "detector": "deep_learning"}
    assert event_topic(flow) == "cyber-flows"
    assert event_topic(prediction) == "cyber-predictions"
    assert stable_storage_id(flow) == "flow-1"
    assert stable_storage_id(prediction) == "window-1:deep_learning"
    assert stable_storage_id(prediction) == stable_storage_id(dict(prediction))


def test_cassandra_ids_are_namespaced_by_event_kind():
    from spark_jobs.common import cassandra_storage_id
    document_id = "window-1:deep_learning"
    assert cassandra_storage_id("batch_prediction", document_id) == "batch_prediction:window-1:deep_learning"
    assert cassandra_storage_id("stream_prediction", document_id) == "stream_prediction:window-1:deep_learning"
    assert cassandra_storage_id("batch_prediction", document_id) != cassandra_storage_id("stream_prediction", document_id)


def test_cassandra_rows_use_bounded_single_statement_writes():
    from spark_jobs.common import write_cassandra_rows

    class Future:
        def __init__(self, session):
            self.session = session

        def result(self):
            self.session.pending -= 1
            self.session.completed += 1

    class Session:
        def __init__(self):
            self.calls = []
            self.pending = 0
            self.maximum_pending = 0
            self.completed = 0

        def execute_async(self, statement, values):
            self.calls.append((statement, values))
            self.pending += 1
            self.maximum_pending = max(self.maximum_pending, self.pending)
            return Future(self)

    session = Session()
    rows = [(index,) for index in range(7)]
    write_cassandra_rows(session, "prepared", rows, max_in_flight=2)
    assert session.calls == [("prepared", row) for row in rows]
    assert session.maximum_pending <= 2
    assert session.completed == len(rows)
