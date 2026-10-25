"""Non-destructive idempotent Elasticsearch bulk publishing with read-back verification."""
from pathlib import Path
import json
import requests

from .pipeline import FEATURES, digest


def index_mapping(kind):
    properties = {"@timestamp": {"type": "date"}, "source": {"properties": {"ip": {"type": "ip"}}},
                  "malicious": {"type": "boolean"}, "reference_label": {"type": "keyword"}}
    numeric_features = {f: {"type": "double"} for f in FEATURES}
    if kind == "flows":
        properties.update({"event": {"properties": {"id": {"type": "keyword"}}},
                           "destination": {"properties": {"ip": {"type": "ip"}, "port": {"type": "integer"}}},
                           "network": {"properties": {"bytes": {"type": "double"}, "packets": {"type": "double"}, "transport": {"type": "keyword"}, "iana_number": {"type": "integer"}}},
                           "duration": {"type": "double"}, "rate": {"type": "double"}})
        properties["source"]["properties"]["port"] = {"type": "integer"}
    else:
        properties.update({"window_id": {"type": "keyword"}, "split": {"type": "keyword"}})
        if kind == "features":
            properties.update(numeric_features)
        else:
            properties.update({"detector": {"type": "keyword"}, "score": {"type": "double"}, "threshold": {"type": "double"},
                               "flagged": {"type": "boolean"}, "status": {"type": "keyword"}, "reason": {"type": "keyword"},
                               "features": {"properties": numeric_features}})
    return {"dynamic": "strict", "properties": properties}


def document_id(kind, document):
    if kind == "flows":
        return document["event"]["id"]
    if kind == "features":
        return document["window_id"]
    return digest([document["window_id"], document["detector"]])


def publish_artifacts(output, es_url="http://localhost:19200", batch_size=500):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    base = es_url.rstrip("/")
    counts = {}
    with requests.Session() as session:
        def request(method, route, **kwargs):
            response = session.request(method, base + route, timeout=120, **kwargs)
            response.raise_for_status()
            return response.json()

        for kind in ["flows", "features", "findings"]:
            index = f"cyber-{kind}-v1"
            mapping = index_mapping(kind)
            head = session.head(f"{base}/{index}", timeout=30)
            if head.status_code == 404:
                result = request("PUT", f"/{index}", json={"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping})
            else:
                head.raise_for_status()
                result = request("PUT", f"/{index}/_mapping", json=mapping)
            if not result.get("acknowledged"):
                raise RuntimeError(f"Mapping was not acknowledged for {index}: {result}")
            count = 0

            def send(batch):
                lines = []
                expected = {}
                for document in batch:
                    identifier = document_id(kind, document)
                    expected[identifier] = document
                    lines.append(json.dumps({"index": {"_index": index, "_id": identifier}}))
                    lines.append(json.dumps(document, allow_nan=False))
                result = request("POST", "/_bulk", data=("\n".join(lines) + "\n").encode("utf-8"), headers={"Content-Type": "application/x-ndjson"})
                errors = [item for item in result.get("items", []) if any(info.get("status", 500) >= 300 or info.get("error") for info in item.values())]
                if result.get("errors") or errors or len(result.get("items", [])) != len(batch):
                    raise RuntimeError(f"Elasticsearch bulk failure for {index}: {str(errors or result)[:2000]}")
                verified = request("POST", f"/{index}/_mget", json={"ids": list(expected)})
                docs = verified.get("docs", [])
                if len(docs) != len(expected) or any(not doc.get("found") or doc.get("_source") != expected.get(doc.get("_id")) for doc in docs):
                    raise RuntimeError(f"Read-back verification failed for {index}")

            with (Path(output) / f"{kind}.ndjson").open(encoding="utf-8") as handle:
                batch = []
                for line in handle:
                    batch.append(json.loads(line))
                    if len(batch) == batch_size:
                        send(batch)
                        count += len(batch)
                        batch = []
                if batch:
                    send(batch)
                    count += len(batch)
            refreshed = request("POST", f"/{index}/_refresh")
            if refreshed.get("_shards", {}).get("failed", 0):
                raise RuntimeError(f"Refresh failed: {index}: {refreshed}")
            counts[index] = count
    return {"status": "verified", "url": base, "indexed_and_verified": counts, "idempotency": "deterministic IDs; index/upsert; no deletes; unrelated prior documents retained"}
