"""Replay saved CICIDS flows and model predictions into the Kafka demo topics."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from spark_jobs.common import event_topic, stable_storage_id


def iter_artifact_events(artifact_dir, flow_topic="cyber-flows", prediction_topic="cyber-predictions"):
    root = Path(artifact_dir)
    configured_topics = {
        "flows.ndjson": flow_topic,
        "predictions.ndjson": prediction_topic,
    }
    expected_types = {"flows.ndjson": "cyber-flows", "predictions.ndjson": "cyber-predictions"}
    for filename in ("flows.ndjson", "predictions.ndjson"):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required replay artifact is missing: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path.name} line {line_number}: {exc}") from exc
                if not isinstance(document, dict):
                    raise ValueError(f"Expected an object in {path.name} line {line_number}")
                if event_topic(document) != expected_types[filename]:
                    raise ValueError(f"Unexpected record type in {path.name} line {line_number}")
                stable_storage_id(document)
                yield configured_topics[filename], document


def build_watermark_event(timestamp):
    """Build a synthetic flow that advances event time without closing its own window."""
    from datetime import datetime
    normalized = timestamp.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("Watermark timestamp must include a timezone")
    normalized = parsed.isoformat()
    return {
        "@timestamp": normalized,
        "event": {"id": f"replay-watermark-{normalized}"},
        "source": {"ip": "192.0.2.254", "port": 0},
        "destination": {"ip": "192.0.2.253", "port": 0},
        "network": {"bytes": 0.0, "packets": 0.0, "transport": "TCP"},
        "duration": 0.0,
        "rate": 0.0,
        "malicious": False,
        "reference_label": "__REPLAY_WATERMARK__",
    }


def replay(artifact_dir, brokers, rate=0, max_records=None,
           flow_topic="cyber-flows", prediction_topic="cyber-predictions",
           watermark_only=False, watermark_timestamp="2017-07-06T16:15:00+00:00"):
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=[item.strip() for item in brokers.split(",") if item.strip()],
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8"),
        acks="all", retries=10, max_in_flight_requests_per_connection=1,
        compression_type="gzip", linger_ms=10,
    )
    counts = Counter()
    started = time.monotonic()
    next_flush = 1000
    try:
        if watermark_only:
            events = ((flow_topic, build_watermark_event(watermark_timestamp)),)
        else:
            events = iter_artifact_events(artifact_dir, flow_topic, prediction_topic)
        for topic, document in events:
            if max_records is not None and sum(counts.values()) >= max_records:
                break
            sent = sum(counts.values())
            if rate > 0:
                due = started + sent / rate
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            producer.send(topic, key=stable_storage_id(document), value=document)
            counts[topic] += 1
            if sum(counts.values()) >= next_flush:
                producer.flush()
                next_flush += 1000
        producer.flush()
    finally:
        producer.close()
    elapsed = max(time.monotonic() - started, 1e-9)
    return {"brokers": brokers, "topics": dict(sorted(counts.items())),
            "messages": sum(counts.values()), "elapsed_seconds": round(elapsed, 3),
            "messages_per_second": round(sum(counts.values()) / elapsed, 2)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--brokers", default="127.0.0.1:19092,127.0.0.1:19093,127.0.0.1:19094")
    parser.add_argument("--rate", type=float, default=0, help="maximum total messages per second; 0 sends as fast as possible")
    parser.add_argument("--max-records", type=int, default=None, help="development/testing cap")
    parser.add_argument("--flow-topic", default="cyber-flows", help="Kafka topic for flow records")
    parser.add_argument("--prediction-topic", default="cyber-predictions", help="Kafka topic for predictions")
    parser.add_argument("--watermark-only", action="store_true", help="send one synthetic event-time marker instead of replaying artifacts")
    parser.add_argument("--watermark-timestamp", default="2017-07-06T16:15:00+00:00",
                        help="timezone-aware timestamp for --watermark-only")
    args = parser.parse_args(argv)
    if args.rate < 0 or (args.max_records is not None and args.max_records < 1):
        parser.error("--rate must be nonnegative and --max-records must be positive")
    print(json.dumps(replay(args.artifacts, args.brokers, args.rate, args.max_records,
                            args.flow_topic, args.prediction_topic, args.watermark_only,
                            args.watermark_timestamp), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
