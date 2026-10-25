"""Shared utilities for Spark batch and streaming applications."""


def event_topic(document):
    """Route normalized raw-flow and model-prediction documents to separate topics."""
    return "cyber-predictions" if "detector" in document else "cyber-flows"


def stable_storage_id(document):
    """Use deterministic IDs so retries and replays are idempotent at the sinks."""
    event_id = (document.get("event") or {}).get("id")
    if event_id:
        return str(event_id)
    window_id = document.get("window_id")
    if window_id and document.get("detector"):
        return f"{window_id}:{document['detector']}"
    if window_id:
        return str(window_id)
    raise ValueError("A stored event must contain event.id or window_id")


def cassandra_storage_id(event_kind, document_id):
    """Keep batch/stream event kinds distinct in Cassandra's primary key."""
    return f"{event_kind}:{document_id}"


def write_cassandra_rows(session, prepared, rows, max_in_flight=8):
    """Write single-partition mutations concurrently without cross-partition batches."""
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    pending = []
    for values in rows:
        pending.append(session.execute_async(prepared, values))
        if len(pending) >= max_in_flight:
            _await_cassandra_futures(pending)
            pending = []
    if pending:
        _await_cassandra_futures(pending)


def _await_cassandra_futures(futures):
    first_error = None
    for future in futures:
        try:
            future.result()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def json_safe(value):
    """Convert Spark rows and driver scalar values into strict JSON-safe values."""
    from datetime import date, datetime
    from collections.abc import Mapping
    if hasattr(value, "asDict"):
        value = value.asDict(recursive=True)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        return json_safe(value.item())
    return value


def build_sink_documents(document, document_id):
    """Build a Mongo replacement with _id and JSON sources without reserved metadata."""
    source_document = {key: value for key, value in document.items() if key != "_id"}
    mongo_document = dict(source_document)
    mongo_document["_id"] = document_id
    return mongo_document, source_document


def _utc_datetime(value):
    from datetime import datetime, timezone
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError(f"Event @timestamp must be a datetime or ISO string, got {type(value).__name__}")
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def sink_partition(rows, collection_name, es_index, event_kind):
    """Idempotently write one Spark partition to MongoDB, Cassandra, and Elasticsearch."""
    from itertools import chain
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    rows = chain((first,), rows)
    import json
    import os
    import requests
    from pymongo import MongoClient, ReplaceOne
    from cassandra.cluster import Cluster
    from cassandra.query import ConsistencyLevel

    mongo_uri = os.environ["MONGO_URI"]
    cassandra_hosts = [host.strip() for host in os.environ.get("CASSANDRA_HOSTS", "cassandra-1,cassandra-2,cassandra-3").split(",") if host.strip()]
    es_url = os.environ.get("ELASTICSEARCH_URL", "http://elasticsearch:9200").rstrip("/")
    mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=15000, socketTimeoutMS=30000)
    collection = mongo["cyberlab"][collection_name]
    cluster = Cluster(contact_points=cassandra_hosts, connect_timeout=10)
    session = cluster.connect("cyberlab")
    prepared = session.prepare("INSERT INTO event_rows (event_day, source_ip, event_timestamp, event_id, event_kind, payload) VALUES (?, ?, ?, ?, ?, ?)")
    prepared.consistency_level = ConsistencyLevel.QUORUM
    mongo_ops = []
    es_lines = []
    cassandra_rows = []

    def flush():
        nonlocal mongo_ops, es_lines, cassandra_rows
        if mongo_ops:
            collection.bulk_write(mongo_ops, ordered=False)
            mongo_ops = []
        if cassandra_rows:
            write_cassandra_rows(session, prepared, cassandra_rows, max_in_flight=8)
            cassandra_rows = []
        if es_lines:
            response = requests.post(f"{es_url}/_bulk", data="\n".join(es_lines) + "\n",
                                     headers={"Content-Type": "application/x-ndjson"}, timeout=30)
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                failed = next((item for item in body.get("items", []) if next(iter(item.values())).get("error")), None)
                raise RuntimeError(f"Elasticsearch bulk write failed: {failed}")
            es_lines = []

    try:
        for row in rows:
            document = json_safe(row)
            document_id = stable_storage_id(document)
            mongo_document, json_document = build_sink_documents(document, document_id)
            source_ip = (json_document.get("source") or {}).get("ip")
            if not source_ip:
                raise ValueError(f"Event {document_id} has no source.ip")
            event_time = _utc_datetime(json_document.get("@timestamp"))
            payload = json.dumps(json_document, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            mongo_ops.append(ReplaceOne({"_id": document_id}, mongo_document, upsert=True))
            cassandra_rows.append((event_time.date(), source_ip, event_time, cassandra_storage_id(event_kind, document_id), event_kind, payload))
            es_lines.append(json.dumps({"index": {"_index": es_index, "_id": document_id}}, separators=(",", ":")))
            es_lines.append(payload)
            if len(mongo_ops) >= 50 or len(cassandra_rows) >= 50:
                flush()
        flush()
    finally:
        session.shutdown()
        cluster.shutdown()
        mongo.close()
