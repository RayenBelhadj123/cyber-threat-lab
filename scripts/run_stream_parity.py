"""Run a clean, isolated batch/stream parity check against the local lab."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ES_URL = "http://127.0.0.1:19200"
COMPOSE = ["docker", "compose", "--profile", "bigdata"]
KAFKA_CONTAINER = "cyber-threat-lab-kafka-1-1"
MONGO_CONTAINER = "cyber-threat-lab-mongo-1-1"
CASSANDRA_CONTAINER = "cyber-threat-lab-cassandra-1-1"
DEFAULT_STREAM_CONTAINER = "cyber-threat-lab-spark-stream-1"


def run_names(run_id: str) -> dict[str, str]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", run_id):
        raise ValueError("run_id must be 1-24 lowercase letters, digits, or hyphens")
    return {
        "flow_topic": f"cyber-flows-parity-{run_id}",
        "prediction_topic": f"cyber-predictions-parity-{run_id}",
        "checkpoint": run_id,
        "metrics": f"spark_stream_parity_{run_id}_metrics.json",
        "container": f"cyber-threat-lab-spark-stream-parity-{run_id}",
        "report": f"stream_parity_{run_id}.json",
    }


def replay_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    paths = [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path]
    project_root = str(ROOT)
    if project_root not in paths:
        paths.insert(0, project_root)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def validate_fresh_run(names: dict[str, str], checkpoint_root: Path, artifact_root: Path,
                       existing_topics: set[str]) -> None:
    topic_collisions = {names["flow_topic"], names["prediction_topic"]} & existing_topics
    checkpoint = checkpoint_root / names["checkpoint"]
    outputs = [artifact_root / names["metrics"], artifact_root / names["report"]]
    if topic_collisions:
        raise ValueError(f"Parity topics already exist; choose a fresh run ID: {sorted(topic_collisions)}")
    if checkpoint.exists() or any(path.exists() for path in outputs):
        raise ValueError(f"Checkpoint or report already exists for run ID {names['checkpoint']!r}")


def max_offsets_behind_latest(metrics: dict) -> int | None:
    if metrics.get("status") != "running":
        return None
    queries = metrics.get("queries") or []
    if not queries:
        return None
    lags = []
    for query in queries:
        if not query.get("active") or query.get("exception"):
            return None
        progress = query.get("last_progress")
        sources = progress.get("sources") if progress else None
        if not sources:
            return None
        for source in sources:
            raw_lag = (source.get("metrics") or {}).get("maxOffsetsBehindLatest")
            if raw_lag is None:
                return None
            lags.append(int(raw_lag))
    return max(lags) if lags else None


def default_stream_ready(metrics: dict | None, observed_after: float) -> bool:
    if not metrics or metrics.get("status") != "running":
        return False
    if float(metrics.get("observed_at_epoch_seconds", 0)) <= observed_after:
        return False
    queries = metrics.get("queries") or []
    return bool(queries) and all(q.get("active") and not q.get("exception") for q in queries)


def marker_timestamp(latest_flow_timestamp: str) -> str:
    normalized = latest_flow_timestamp.replace("Z", "+00:00")
    latest = datetime.fromisoformat(normalized)
    if latest.tzinfo is None:
        raise ValueError("Latest flow timestamp must include a timezone")
    return (latest.astimezone(timezone.utc) + timedelta(minutes=16)).isoformat()


def _run(command: list[str], *, check: bool = True, capture: bool = True, quiet: bool = False):
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    if not quiet:
        print(f"$ {printable}", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=capture, check=False)
    if result.stdout and not quiet:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr and not quiet:
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}: {printable}\n{result.stderr or result.stdout}")
    return result


def _container_state(name: str) -> tuple[str | None, str | None]:
    result = _run(["docker", "inspect", "--format", "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}", name], check=False, quiet=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None, None
    state, health = result.stdout.strip().split("|", 1)
    return state, health


def _require_running_containers() -> None:
    required = [
        "cyber-threat-lab-elasticsearch-1",
        "cyber-threat-lab-kafka-1-1", "cyber-threat-lab-kafka-2-1", "cyber-threat-lab-kafka-3-1",
        "cyber-threat-lab-mongo-1-1", "cyber-threat-lab-mongo-2-1", "cyber-threat-lab-mongo-3-1",
        "cyber-threat-lab-cassandra-1-1", "cyber-threat-lab-cassandra-2-1", "cyber-threat-lab-cassandra-3-1",
        "cyber-threat-lab-spark-master-1", "cyber-threat-lab-spark-worker-1-1", "cyber-threat-lab-spark-worker-2-1",
    ]
    for name in required:
        state, health = _container_state(name)
        if state != "running" or (health not in (None, "none", "healthy")):
            raise RuntimeError(f"Required dependency is not healthy: {name} state={state} health={health}")


def _wait_healthy(names: list[str], timeout: float = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = [_container_state(name) for name in names]
        if all(state == "running" and health == "healthy" for state, health in states):
            return
        time.sleep(3)
    raise TimeoutError(f"Spark services did not become healthy: {names}")


def _create_topic(topic: str) -> None:
    _run([
        "docker", "exec", KAFKA_CONTAINER, "/opt/kafka/bin/kafka-topics.sh",
        "--bootstrap-server", "kafka-1:9092", "--create", "--if-not-exists",
        "--topic", topic, "--partitions", "6", "--replication-factor", "3",
        "--config", "min.insync.replicas=2",
    ])


def _latest_flow_timestamp(path: Path) -> str:
    last_line = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if last_line is None:
        raise ValueError(f"No flow rows in {path}")
    return json.loads(last_line)["@timestamp"]


def _load_metrics(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _wait_for_catchup(metrics_path: Path, container: str, timeout: float, poll: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, _ = _container_state(container)
        if state != "running":
            raise RuntimeError(f"Parity stream stopped before catch-up: {container} state={state}")
        metrics = _load_metrics(metrics_path)
        if metrics:
            for query in metrics.get("queries", []):
                if query.get("exception"):
                    raise RuntimeError(f"Spark query {query.get('name')} failed: {query['exception']}")
            if max_offsets_behind_latest(metrics) == 0:
                return metrics
        time.sleep(poll)
    raise TimeoutError(f"Kafka offsets did not reach zero lag within {timeout} seconds")


def _wait_for_watermark(metrics_path: Path, container: str, marker_time: str, timeout: float, poll: float) -> dict:
    target = datetime.fromisoformat(marker_time) - timedelta(minutes=10)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, _ = _container_state(container)
        if state != "running":
            raise RuntimeError(f"Parity stream stopped before watermark flush: {container} state={state}")
        metrics = _load_metrics(metrics_path)
        if metrics:
            for query in metrics.get("queries", []):
                if query.get("exception"):
                    raise RuntimeError(f"Spark query {query.get('name')} failed: {query['exception']}")
            if max_offsets_behind_latest(metrics) == 0:
                progress = next((q.get("last_progress") for q in metrics.get("queries", [])
                                 if q.get("name") == "cyber-flow-windows"), None)
                watermark = (progress or {}).get("eventTime", {}).get("watermark")
                if watermark and datetime.fromisoformat(watermark.replace("Z", "+00:00")) >= target:
                    return metrics
        time.sleep(poll)
    raise TimeoutError(f"Event-time watermark did not reach {target.isoformat()} within {timeout} seconds")


def _es_json(path: str, payload: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"{ES_URL}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _es_count(index: str) -> int:
    return int(_es_json(f"/{index}/_count")["count"])


def _es_ids(index: str) -> set[str]:
    payload = {"size": 1000, "_source": False, "sort": ["_doc"], "query": {"match_all": {}}}
    page = _es_json(f"/{index}/_search?scroll=1m", payload, "POST")
    scroll_id = page["_scroll_id"]
    found: set[str] = set()
    while page["hits"]["hits"]:
        found.update(hit["_id"] for hit in page["hits"]["hits"])
        page = _es_json("/_search/scroll", {"scroll": "1m", "scroll_id": scroll_id}, "POST")
        scroll_id = page.get("_scroll_id", scroll_id)
    return found


def _mongo_counts() -> dict[str, int]:
    script = ('const d=db.getSiblingDB("cyberlab"); print(JSON.stringify({'
              'stream_features:d.stream_features.countDocuments({}),'
              'stream_predictions:d.stream_predictions.countDocuments({})}));')
    result = _run(["docker", "exec", MONGO_CONTAINER, "mongosh",
                   "mongodb://mongo-1:27017/?replicaSet=rs0", "--quiet", "--eval", script])
    return json.loads(result.stdout.strip().splitlines()[-1])


def _cassandra_status() -> dict:
    result = _run(["docker", "exec", CASSANDRA_CONTAINER, "nodetool", "status"])
    live = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("UN ")]
    if len(live) != 3:
        raise RuntimeError(f"Expected 3 Cassandra nodes in UN state, found {live}")
    return {"nodes_up_normal": live}


def _verify_cassandra_sample() -> dict:
    query = {"size": 1, "_source": ["@timestamp", "window_id", "source.ip"],
             "sort": [{"@timestamp": "asc"}], "query": {"match_all": {}}}
    hit = _es_json("/cyber-features-spark-batch-v1/_search", query, "POST")["hits"]["hits"][0]["_source"]
    timestamp = datetime.fromisoformat(hit["@timestamp"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    cql_timestamp = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    source_ip = hit["source"]["ip"]
    window_id = hit["window_id"]
    event_id = f"stream_feature:{window_id}"
    cql = ("SELECT event_id,event_kind FROM cyberlab.event_rows "
           f"WHERE event_day='{timestamp.date()}' AND source_ip='{source_ip}' "
           f"AND event_timestamp='{cql_timestamp}' AND event_id='{event_id}';")
    result = _run(["docker", "exec", CASSANDRA_CONTAINER, "cqlsh", "--request-timeout=20", "-e", cql])
    if event_id not in result.stdout or "stream_feature" not in result.stdout:
        raise RuntimeError("Cassandra did not return the sampled stream feature row")
    return {"event_id": event_id, "source_ip": source_ip, "event_timestamp": cql_timestamp}


def _verify_outputs(expected_features: int, expected_predictions: int) -> dict:
    _es_json("/cyber-features-spark-stream-v1/_refresh", method="POST")
    _es_json("/cyber-findings-spark-stream-v1/_refresh", method="POST")
    counts = {
        "elasticsearch": {
            "batch_features": _es_count("cyber-features-spark-batch-v1"),
            "stream_features": _es_count("cyber-features-spark-stream-v1"),
            "batch_predictions": _es_count("cyber-findings-spark-batch-v1"),
            "stream_predictions": _es_count("cyber-findings-spark-stream-v1"),
        },
        "mongodb": _mongo_counts(),
        "cassandra": _cassandra_status(),
    }
    expected = {
        "batch_features": expected_features,
        "stream_features": expected_features,
        "batch_predictions": expected_predictions,
        "stream_predictions": expected_predictions,
    }
    if any(counts["elasticsearch"][key] != value for key, value in expected.items()):
        raise RuntimeError(f"Elasticsearch count mismatch: {counts['elasticsearch']} expected={expected}")
    if counts["mongodb"] != {"stream_features": expected_features, "stream_predictions": expected_predictions}:
        raise RuntimeError(f"MongoDB count mismatch: {counts['mongodb']}")
    id_sets = {}
    for kind, batch_index, stream_index in (
        ("features", "cyber-features-spark-batch-v1", "cyber-features-spark-stream-v1"),
        ("predictions", "cyber-findings-spark-batch-v1", "cyber-findings-spark-stream-v1"),
    ):
        batch_ids, stream_ids = _es_ids(batch_index), _es_ids(stream_index)
        id_sets[kind] = {
            "batch_ids": len(batch_ids), "stream_ids": len(stream_ids),
            "missing_from_stream": len(batch_ids - stream_ids),
            "extra_in_stream": len(stream_ids - batch_ids),
        }
        if batch_ids != stream_ids:
            raise RuntimeError(f"Elasticsearch {kind} IDs differ: {id_sets[kind]}")
    counts["cassandra"]["sample_stream_feature"] = _verify_cassandra_sample()
    return {"counts": counts, "elasticsearch_id_sets": id_sets}


def _write_metrics_as_stopped(metrics_path: Path, exit_code: int | None) -> None:
    metrics = _load_metrics(metrics_path)
    if not metrics:
        return
    metrics["status"] = "stopped"
    metrics["shutdown_note"] = f"Parity check finished; temporary Spark container exit code={exit_code}."
    for query in metrics.get("queries", []):
        query["active"] = False
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _wait_container_started(container: str, metrics_path: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, _ = _container_state(container)
        if state != "running":
            raise RuntimeError(f"Parity stream container failed to start: {state}")
        metrics = _load_metrics(metrics_path)
        if metrics and metrics.get("status") == "running" and all(q.get("active") for q in metrics.get("queries", [])):
            return metrics
        time.sleep(2)
    raise TimeoutError("Parity stream queries did not start before timeout")


def _wait_default_stream_started(previous_observed_at: float, timeout: float = 180) -> None:
    metrics_path = ARTIFACTS / "spark_stream_metrics.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, _ = _container_state(DEFAULT_STREAM_CONTAINER)
        if state == "running" and default_stream_ready(_load_metrics(metrics_path), previous_observed_at):
            return
        time.sleep(2)
    raise TimeoutError("Default spark-stream did not restart with fresh active-query metrics")


def run_parity(rate: float = 1000, timeout: float = 1800, poll: float = 10, run_id: str | None = None) -> dict:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    names = run_names(run_id)
    _require_running_containers()
    batch_metrics = json.loads((ARTIFACTS / "spark_batch_metrics.json").read_text(encoding="utf-8"))
    expected_features = int(batch_metrics["feature_windows"])
    expected_predictions = int(batch_metrics["prediction_records"])
    latest = _latest_flow_timestamp(ARTIFACTS / "flows.ndjson")
    marker_time = marker_timestamp(latest)
    topic_listing = _run([
        "docker", "exec", KAFKA_CONTAINER, "/opt/kafka/bin/kafka-topics.sh",
        "--bootstrap-server", "kafka-1:9092", "--list",
    ], quiet=True)
    validate_fresh_run(names, ARTIFACTS / "spark-checkpoints", ARTIFACTS,
                       set(topic_listing.stdout.splitlines()))
    default_metrics = _load_metrics(ARTIFACTS / "spark_stream_metrics.json")
    previous_default_observed = float((default_metrics or {}).get("observed_at_epoch_seconds", 0))
    main_stream_was_running = _container_state(DEFAULT_STREAM_CONTAINER)[0] == "running"
    parity_started = False
    exit_code = None
    metrics_path = ARTIFACTS / names["metrics"]
    try:
        if main_stream_was_running:
            _run(COMPOSE + ["stop", "spark-stream"])
        _run(COMPOSE + ["build", "spark-master"])
        _run(COMPOSE + ["up", "-d", "--force-recreate", "spark-master", "spark-worker-1", "spark-worker-2"])
        _wait_healthy([
            "cyber-threat-lab-spark-master-1",
            "cyber-threat-lab-spark-worker-1-1",
            "cyber-threat-lab-spark-worker-2-1",
        ])
        _create_topic(names["flow_topic"])
        _create_topic(names["prediction_topic"])
        spark_args = [
            "--master", "spark://spark-master:7077",
            "--conf", "spark.jars.ivy=/tmp/ivy",
            "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0",
            "--conf", f"spark.driver.host={names['container']}",
            "--conf", "spark.driver.bindAddress=0.0.0.0",
            "--conf", "spark.driver.port=7082",
            "--conf", "spark.blockManager.port=7083",
            "--conf", "spark.executor.memory=512m",
            "--conf", "spark.cores.max=2",
            "--conf", "spark.sql.session.timeZone=UTC",
            "/opt/spark/jobs/spark_jobs/streaming.py",
            "--checkpoints", f"/app/checkpoints/{names['checkpoint']}",
            "--metrics", f"/app/artifacts/{names['metrics']}",
        ]
        _run(COMPOSE + [
            "run", "-d", "--no-deps", "--name", names["container"],
            "-e", f"KAFKA_FLOW_TOPIC={names['flow_topic']}",
            "-e", f"KAFKA_PREDICTION_TOPIC={names['prediction_topic']}",
            "spark-stream", *spark_args,
        ])
        parity_started = True
        _wait_container_started(names["container"], metrics_path, timeout=min(timeout, 180))
        replay = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "stream_replay.py"),
            "--artifacts", str(ARTIFACTS), "--rate", str(rate),
            "--flow-topic", names["flow_topic"], "--prediction-topic", names["prediction_topic"],
        ], cwd=ROOT, env=replay_environment(), text=True, capture_output=True, check=False)
        if replay.stdout:
            print(replay.stdout.rstrip(), flush=True)
        if replay.stderr:
            print(replay.stderr.rstrip(), file=sys.stderr, flush=True)
        if replay.returncode != 0:
            raise RuntimeError(f"Artifact replay failed with exit code {replay.returncode}")
        replay_result = json.loads(replay.stdout)
        progress = _wait_for_catchup(metrics_path, names["container"], timeout, poll)
        marker_result = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "stream_replay.py"),
            "--watermark-only", "--artifacts", str(ARTIFACTS),
            "--flow-topic", names["flow_topic"], "--prediction-topic", names["prediction_topic"],
            "--watermark-timestamp", marker_time,
        ], cwd=ROOT, env=replay_environment(), text=True, capture_output=True, check=False)
        if marker_result.stdout:
            print(marker_result.stdout.rstrip(), flush=True)
        if marker_result.stderr:
            print(marker_result.stderr.rstrip(), file=sys.stderr, flush=True)
        if marker_result.returncode != 0:
            raise RuntimeError(f"Watermark marker publish failed with exit code {marker_result.returncode}")
        marker_output = json.loads(marker_result.stdout)
        progress = _wait_for_watermark(metrics_path, names["container"], marker_time, timeout, poll)
        verified = _verify_outputs(expected_features, expected_predictions)
        return {
            "run_id": run_id,
            "topics": {"flows": names["flow_topic"], "predictions": names["prediction_topic"]},
            "replay": replay_result,
            "watermark_marker": {**marker_output, "timestamp": marker_time},
            "verification": verified,
            "progress": {
                "max_offsets_behind_latest": max_offsets_behind_latest(progress),
                "watermark": next((q.get("last_progress", {}).get("eventTime", {}).get("watermark")
                                    for q in progress.get("queries", []) if q.get("name") == "cyber-flow-windows"), None),
            },
        }
    finally:
        if parity_started:
            stopped = _run(["docker", "stop", "--time", "30", names["container"]], check=False)
            state, _ = _container_state(names["container"])
            if state == "exited":
                inspect = _run(["docker", "inspect", "--format", "{{.State.ExitCode}}", names["container"]], check=False)
                try:
                    exit_code = int(inspect.stdout.strip())
                except (ValueError, AttributeError):
                    exit_code = stopped.returncode
                _write_metrics_as_stopped(metrics_path, exit_code)
                _run(["docker", "rm", names["container"]], check=False)
        if main_stream_was_running:
            _run(COMPOSE + ["up", "-d", "--force-recreate", "spark-stream"])
            _wait_default_stream_started(previous_default_observed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="optional lowercase run suffix; a UTC timestamp is used by default")
    parser.add_argument("--rate", type=float, default=1000, help="maximum total Kafka messages per second")
    parser.add_argument("--timeout", type=float, default=1800, help="max seconds to await each catch-up stage")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    if args.rate <= 0 or args.timeout <= 0 or args.poll_seconds <= 0:
        parser.error("rate, timeout, and poll interval must be positive")
    try:
        report = run_parity(args.rate, args.timeout, args.poll_seconds, args.run_id)
        output = ARTIFACTS / f"stream_parity_{report['run_id']}.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "verified", "report": str(output), **report}, indent=2))
        return 0
    except Exception as exc:
        print(f"stream parity failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
