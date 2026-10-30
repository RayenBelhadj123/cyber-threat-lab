"""Disk-backed, full-benchmark flow store with bounded ingestion and queries.

The store contains normalized raw flow records, not aggregated feature windows.
Source files and labels remain attached to every record.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .pipeline import ALIASES, NUMERIC, canonical_header, digest, valid_ip

SCHEMA_VERSION = 1
FLOW_STATS_SUMMARY_VERSION = 2
DATASETS = {"CICIDS2017", "CICDDoS2019"}
SORT_COLUMNS = {
    "timestamp": "timestamp",
    "source_ip": "source_ip",
    "destination_ip": "destination_ip",
    "source_port": "source_port",
    "destination_port": "destination_port",
    "protocol": "protocol",
    "label": "label",
    "bytes": "bytes_total",
    "packets": "packets_total",
    "dataset": "dataset",
}
PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP", 41: "IPv6", 47: "GRE", 50: "ESP", 58: "ICMPv6", 132: "SCTP"}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _capture_group(dataset: str, name: str) -> str:
    capture = name.split("-")[0] if dataset == "CICIDS2017" else name.split("_")[0]
    return f"{dataset}:{capture}"


def _normalized_chunks(path: Path, chunk_size: int):
    """Yield (normalized-valid-frame, input rows, blank rows, invalid rows)."""
    encoding = "utf-8-sig"
    try:
        with path.open("r", encoding=encoding) as probe:
            while probe.read(1024 * 1024):
                pass
    except UnicodeDecodeError:
        encoding = "cp1252"
    for frame in pd.read_csv(path, chunksize=chunk_size, encoding=encoding, low_memory=False):
        blank = frame.isna().all(axis=1)
        blank_count = int(blank.sum())
        frame = frame.loc[~blank].copy()
        if frame.empty:
            yield frame, 0, blank_count, 0
            continue
        normalized = {canonical_header(c): c for c in frame.columns}
        rename = {}
        for target, options in ALIASES.items():
            match = next((normalized[canonical_header(alias)] for alias in options if canonical_header(alias) in normalized), None)
            if match is None:
                raise ValueError(f"Missing required column {target} in {path}; identity, timestamp and label are required")
            rename[match] = target
        frame = frame.rename(columns=rename)[list(ALIASES)].copy()
        for column in NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame["source_ip"] = frame["source_ip"].map(valid_ip)
        frame["destination_ip"] = frame["destination_ip"].map(valid_ip)
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce"
        ).dt.tz_localize(ZoneInfo("America/Moncton"), ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        frame["label"] = frame["label"].astype("string").str.strip()
        valid = frame.notna().all(axis=1) & np.isfinite(frame[NUMERIC]).all(axis=1) & frame["label"].ne("")
        valid &= frame[NUMERIC].ge(0).all(axis=1)
        for column, upper in (("source_port", 65535), ("destination_port", 65535), ("protocol", 255)):
            valid &= frame[column].le(upper) & frame[column].eq(frame[column].round())
        input_rows = len(frame)
        invalid_rows = int((~valid).sum())
        frame = frame.loc[valid].copy()
        frame["malicious"] = frame["label"].str.upper().ne("BENIGN") & frame["label"].str.upper().ne("UNKNOWN")
        frame["bytes_total"] = frame["fwd_bytes"] + frame["bwd_bytes"]
        frame["packets_total"] = frame["fwd_packets"] + frame["bwd_packets"]
        yield frame, input_rows, blank_count, invalid_rows


def _event_id(row: dict[str, Any]) -> str:
    return digest({key: (value.isoformat() if key == "timestamp" else value) for key, value in row.items()})


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE flows (
        dataset TEXT NOT NULL, event_id TEXT NOT NULL, timestamp TEXT NOT NULL,
        source_ip TEXT NOT NULL, source_port INTEGER NOT NULL,
        destination_ip TEXT NOT NULL, destination_port INTEGER NOT NULL,
        protocol INTEGER NOT NULL, duration REAL NOT NULL, bytes_total REAL NOT NULL,
        packets_total REAL NOT NULL, rate REAL NOT NULL, label TEXT NOT NULL,
        malicious INTEGER NOT NULL, source_file TEXT NOT NULL, capture_group TEXT NOT NULL,
        PRIMARY KEY (dataset, event_id)
    ) WITHOUT ROWID""")


def _add_indexes(connection: sqlite3.Connection) -> None:
    for name, columns in (
        ("idx_flows_timestamp", "timestamp"),
        ("idx_flows_source_ip", "source_ip"),
        ("idx_flows_destination_ip", "destination_ip"),
        ("idx_flows_source_port", "source_port"),
        ("idx_flows_destination_port", "destination_port"),
        ("idx_flows_protocol", "protocol"),
        ("idx_flows_label", "label"),
        ("idx_flows_dataset_time", "dataset, timestamp"),
    ):
        connection.execute(f"CREATE INDEX {name} ON flows ({columns})")
    connection.execute("""CREATE TABLE flow_topology (
        dataset TEXT NOT NULL, source_ip TEXT NOT NULL, destination_ip TEXT NOT NULL,
        destination_port INTEGER NOT NULL, protocol INTEGER NOT NULL,
        flow_count INTEGER NOT NULL, malicious_count INTEGER NOT NULL,
        bytes_total REAL NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
        PRIMARY KEY (dataset, source_ip, destination_ip, destination_port, protocol)
    ) WITHOUT ROWID""")
    connection.execute("""INSERT INTO flow_topology
        SELECT dataset, source_ip, destination_ip, destination_port, protocol,
               COUNT(*), SUM(malicious), SUM(bytes_total), MIN(timestamp), MAX(timestamp)
        FROM flows GROUP BY dataset, source_ip, destination_ip, destination_port, protocol""")
    connection.execute("CREATE INDEX idx_topology_count ON flow_topology(dataset, flow_count DESC)")


def _build_flow_stats_summary(connection: sqlite3.Connection) -> None:
    """Materialize bounded dashboard aggregates so API reads avoid rescanning flows."""
    connection.execute("CREATE TABLE IF NOT EXISTS flow_stats_summary (dataset TEXT PRIMARY KEY, summary_json TEXT NOT NULL) WITHOUT ROWID")
    summaries = {name: {"protocols": Counter(), "labels": Counter(), "sources": Counter(), "ports": Counter(), "hourly": Counter()}
                 for name in (*sorted(DATASETS), "expanded")}
    scalars = {row[0]: row[1:] for row in connection.execute(
        "SELECT dataset,COUNT(*),SUM(malicious),SUM(packets_total),SUM(duration),MIN(timestamp),MAX(timestamp) FROM flows GROUP BY dataset")}
    for dataset, protocol, count in connection.execute(
        "SELECT dataset,protocol,COUNT(*) FROM flows GROUP BY dataset,protocol"):
        summaries[dataset]["protocols"][PROTOCOL_NAMES.get(protocol, f"IP protocol {protocol}")] += count
    for dataset, label, count in connection.execute("SELECT dataset,label,COUNT(*) FROM flows GROUP BY dataset,label"):
        summaries[dataset]["labels"][label] += count
    for dataset, ip, count in connection.execute("""
        SELECT dataset,source_ip,n FROM (
            SELECT dataset,source_ip,COUNT(*) AS n,
                   ROW_NUMBER() OVER (PARTITION BY dataset ORDER BY COUNT(*) DESC,source_ip) AS position
            FROM flows GROUP BY dataset,source_ip
        ) WHERE position <= 12"""):
        summaries[dataset]["sources"][ip] += count
    for dataset, port, count in connection.execute("""
        SELECT dataset,destination_port,n FROM (
            SELECT dataset,destination_port,COUNT(*) AS n,
                   ROW_NUMBER() OVER (PARTITION BY dataset ORDER BY COUNT(*) DESC,destination_port) AS position
            FROM flows GROUP BY dataset,destination_port
        ) WHERE position <= 12"""):
        summaries[dataset]["ports"][int(port)] += count
    for dataset, hour, count in connection.execute(
        "SELECT dataset,substr(timestamp,1,13),COUNT(*) FROM flows GROUP BY dataset,substr(timestamp,1,13)"):
        summaries[dataset]["hourly"][hour] += count

    dataset_payloads = {}
    for dataset in sorted(DATASETS):
        count, suspicious, packet_sum, duration_sum, first, last = scalars.get(dataset, (0, 0, 0, 0, None, None))
        grouped = summaries[dataset]
        dataset_payloads[dataset] = {
            "data_mode": "Historical benchmark", "dataset": dataset,
            "total_flows": int(count or 0), "suspicious_flows": int(suspicious or 0),
            "protocols": dict(grouped["protocols"]), "labels": dict(grouped["labels"]),
            "top_sources": [{"source_ip": ip, "flows": n} for ip, n in sorted(grouped["sources"].items(), key=lambda item: (-item[1], item[0]))[:12]],
            "top_destination_ports": [{"port": port, "flows": n} for port, n in sorted(grouped["ports"].items(), key=lambda item: (-item[1], item[0]))[:12]],
            "hourly": [{"hour": hour, "flows": n} for hour, n in sorted(grouped["hourly"].items())],
            "mean_packets": (packet_sum / count) if count else None,
            "mean_duration_microseconds": (duration_sum / count) if count else None,
            "time_range_utc": [first, last] if first else None,
        }

    expanded = summaries["expanded"]
    total = suspicious = packet_sum = duration_sum = 0
    first_values, last_values = [], []
    for dataset in sorted(DATASETS):
        count, positive, packets, duration, first, last = scalars.get(dataset, (0, 0, 0, 0, None, None))
        total += int(count or 0)
        suspicious += int(positive or 0)
        packet_sum += float(packets or 0)
        duration_sum += float(duration or 0)
        if first: first_values.append(first)
        if last: last_values.append(last)
        grouped = summaries[dataset]
        for name in ("protocols", "labels", "sources", "ports", "hourly"):
            expanded[name].update(grouped[name])
    # Top-N rankings are not additive: a value outside each dataset's top 12
    # can become the global leader after the datasets are combined.
    expanded["sources"] = Counter(dict(connection.execute(
        "SELECT source_ip,COUNT(*) FROM flows GROUP BY source_ip ORDER BY COUNT(*) DESC,source_ip LIMIT 12")))
    expanded["ports"] = Counter(dict(connection.execute(
        "SELECT destination_port,COUNT(*) FROM flows GROUP BY destination_port ORDER BY COUNT(*) DESC,destination_port LIMIT 12")))
    expanded_payload = {
        "data_mode": "Historical benchmark", "dataset": "expanded", "total_flows": total,
        "suspicious_flows": suspicious, "protocols": dict(expanded["protocols"]),
        "labels": dict(expanded["labels"]),
        "top_sources": [{"source_ip": ip, "flows": n} for ip, n in sorted(expanded["sources"].items(), key=lambda item: (-item[1], item[0]))[:12]],
        "top_destination_ports": [{"port": port, "flows": n} for port, n in sorted(expanded["ports"].items(), key=lambda item: (-item[1], item[0]))[:12]],
        "hourly": [{"hour": hour, "flows": n} for hour, n in sorted(expanded["hourly"].items())],
        "mean_packets": packet_sum / total if total else None,
        "mean_duration_microseconds": duration_sum / total if total else None,
        "time_range_utc": [min(first_values), max(last_values)] if first_values else None,
    }
    summaries = {**dataset_payloads, "expanded": expanded_payload}
    for summary in summaries.values():
        summary["_summary_version"] = FLOW_STATS_SUMMARY_VERSION
    connection.execute("DELETE FROM flow_stats_summary")
    connection.executemany("INSERT INTO flow_stats_summary(dataset,summary_json) VALUES (?,?)",
                           [(name, json.dumps(summary, separators=(",", ":"))) for name, summary in summaries.items()])


def _ensure_flow_stats_summary(connection: sqlite3.Connection, dataset: str) -> str:
    connection.execute("CREATE TABLE IF NOT EXISTS flow_stats_summary (dataset TEXT PRIMARY KEY, summary_json TEXT NOT NULL) WITHOUT ROWID")
    row = connection.execute("SELECT summary_json FROM flow_stats_summary WHERE dataset=?", (dataset,)).fetchone()
    try:
        cached = json.loads(row[0]) if row is not None else None
    except (TypeError, json.JSONDecodeError):
        cached = None
    if not isinstance(cached, dict) or cached.get("_summary_version") != FLOW_STATS_SUMMARY_VERSION:
        _build_flow_stats_summary(connection)
        row = connection.execute("SELECT summary_json FROM flow_stats_summary WHERE dataset=?", (dataset,)).fetchone()
    if row is None:
        raise RuntimeError(f"Cached flow statistics are unavailable for dataset {dataset}")
    return str(row[0])


def build_flow_store(entries: list[dict[str, Any]], output: str | Path, *, chunk_size: int = 50_000) -> dict[str, Any]:
    """Build a verified SQLite fact store in bounded memory from manifest entries."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not entries:
        raise ValueError("No verified input entries were provided")
    for entry in entries:
        if entry["dataset"] not in DATASETS:
            raise ValueError(f"Unsupported dataset scope: {entry['dataset']}")
        if _hash_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Input checksum mismatch: {entry['name']}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    building = output.with_name(output.name + ".building")
    manifest_path = output.with_suffix(".manifest.json")
    for leftover in (building, Path(str(building) + "-journal")):
        if leftover.exists():
            leftover.unlink()

    signature = digest([(entry["dataset"], entry["name"], entry["sha256"], entry["expected_rows"]) for entry in entries])
    totals = Counter()
    file_stats = []
    label_counts: dict[str, Counter] = {}
    connection = sqlite3.connect(building)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-65536")
        _create_schema(connection)
        insert_sql = "INSERT OR IGNORE INTO flows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        for entry in entries:
            dataset, name = entry["dataset"], entry["name"]
            file_input = file_blank = file_invalid = file_valid = 0
            file_labels = Counter()
            for frame, input_rows, blank_rows, invalid_rows in _normalized_chunks(Path(entry["path"]), chunk_size):
                file_input += input_rows
                file_blank += blank_rows
                file_invalid += invalid_rows
                if frame.empty:
                    continue
                rows = []
                for record in frame.to_dict("records"):
                    identifier = _event_id(record)
                    label = str(record["label"])
                    file_labels[label] += 1
                    rows.append((
                        dataset, identifier, record["timestamp"].isoformat(), record["source_ip"], int(record["source_port"]),
                        record["destination_ip"], int(record["destination_port"]), int(record["protocol"]),
                        float(record["duration"]), float(record["bytes_total"]), float(record["packets_total"]),
                        float(record["rate"]), label, int(bool(record["malicious"])), name, _capture_group(dataset, name),
                    ))
                before = connection.total_changes
                connection.executemany(insert_sql, rows)
                inserted = connection.total_changes - before
                file_valid += inserted
                totals["duplicate_rows"] += len(rows) - inserted
                connection.commit()
            if file_input != int(entry["expected_rows"]):
                raise ValueError(f"Manifest row count mismatch for {name}: expected {entry['expected_rows']}, got {file_input}")
            totals["input_rows"] += file_input
            totals["blank_rows"] += file_blank
            totals["invalid_rows"] += file_invalid
            totals["valid_flows"] += file_valid
            label_counts.setdefault(dataset, Counter()).update(file_labels)
            file_stats.append({
                "dataset": dataset, "file": name, "sha256": entry["sha256"], "input_rows": file_input,
                "blank_rows": file_blank, "invalid_rows": file_invalid, "valid_unique_flows": file_valid,
                "duplicate_rows": sum(1 for _ in ()),
            })
            file_stats[-1]["duplicate_rows"] = max(0, sum(file_labels.values()) - file_valid)

        _add_indexes(connection)
        _build_flow_stats_summary(connection)
        connection.commit()
        counts = {row[0]: row[1] for row in connection.execute("SELECT dataset, COUNT(*) FROM flows GROUP BY dataset")}
        stored_labels: dict[str, Counter] = {}
        for source_dataset, label, label_count in connection.execute("SELECT dataset,label,COUNT(*) FROM flows GROUP BY dataset,label"):
            stored_labels.setdefault(source_dataset, Counter())[label] = label_count
        actual = int(connection.execute("SELECT COUNT(*) FROM flows").fetchone()[0])
        if actual != totals["valid_flows"]:
            raise RuntimeError(f"Indexed flow count mismatch: expected {totals['valid_flows']}, stored {actual}")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        if totals["input_rows"] != totals["valid_flows"] + totals["invalid_rows"] + totals["duplicate_rows"]:
            raise RuntimeError("Flow accounting invariant failed: input != valid + invalid + duplicates")
        index_names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")]
        meta = {
            "schema_version": SCHEMA_VERSION, "status": "verified", "source_signature": signature,
            "generated_at": datetime.now(timezone.utc).isoformat(), "input_rows": totals["input_rows"],
            "valid_flows": totals["valid_flows"], "invalid_rows": totals["invalid_rows"],
            "duplicate_rows": totals["duplicate_rows"], "blank_rows": totals["blank_rows"],
            "database_record_count": actual, "dataset_counts": dict(sorted(counts.items())),
            "label_counts": {source_dataset: dict(sorted(counter.items())) for source_dataset, counter in sorted(stored_labels.items())},
            "files": file_stats, "index_names": index_names, "integrity_check": integrity,
            "timestamp_policy": "Same parser as cyberlab.pipeline.load_flows: day-first; America/Moncton local clock converted to UTC. CICDDoS2019 mirror timestamp authenticity is not established.",
            "risk_score_available": False,
            **dict(totals),
        }
    except Exception:
        connection.close()
        if building.exists():
            building.unlink()
        raise
    else:
        connection.close()

    os.replace(building, output)
    temp_manifest = manifest_path.with_name(manifest_path.name + ".building")
    temp_manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    meta["database_bytes"] = output.stat().st_size
    return meta


def _dataset_clause(dataset: str) -> tuple[str, list[Any]]:
    if dataset == "expanded":
        return "dataset IN (?,?)", ["CICIDS2017", "CICDDoS2019"]
    if dataset in DATASETS:
        return "dataset = ?", [dataset]
    raise ValueError("dataset must be 'expanded', 'CICIDS2017' or 'CICDDoS2019'")


def _utc_time(value: str | datetime | None) -> str | None:
    if value is None or value == "":
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp filters must include a timezone, for example 2017-07-03T00:00:00Z")
    return parsed.astimezone(timezone.utc).isoformat()


def _where(dataset: str, filters: dict[str, Any]) -> tuple[str, list[Any]]:
    scope_sql, values = _dataset_clause(dataset)
    clauses = [scope_sql]
    for key in ("source_ip", "destination_ip", "source_port", "destination_port", "protocol"):
        if filters.get(key) not in (None, ""):
            clauses.append(f"{key} = ?")
            values.append(filters[key])
    if filters.get("label"):
        clauses.append("instr(lower(label), lower(?)) > 0")
        values.append(str(filters["label"]))
    if filters.get("query"):
        clauses.append("(instr(lower(source_ip), lower(?)) > 0 OR instr(lower(destination_ip), lower(?)) > 0 OR instr(lower(label), lower(?)) > 0)")
        values.extend([str(filters["query"])] * 3)
    start, end = _utc_time(filters.get("start")), _utc_time(filters.get("end"))
    if start:
        clauses.append("timestamp >= ?")
        values.append(start)
    if end:
        clauses.append("timestamp <= ?")
        values.append(end)
    return " AND ".join(clauses), values


def _public_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["protocol_name"] = PROTOCOL_NAMES.get(result["protocol"], f"IP protocol {result['protocol']}")
    result["malicious"] = bool(result["malicious"])
    result["risk_score"] = None
    return result


def query_flows(database: str | Path, *, dataset: str = "expanded", filters: dict[str, Any] | None = None,
                page: int = 1, page_size: int = 100, sort: str = "timestamp", order: str = "desc") -> dict[str, Any]:
    if page < 1 or not 1 <= page_size <= 500:
        raise ValueError("page must be >= 1 and page_size must be between 1 and 500")
    if sort not in SORT_COLUMNS or order.lower() not in {"asc", "desc"}:
        raise ValueError("Unsupported sort or sort order")
    where, values = _where(dataset, filters or {})
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        total = int(connection.execute(f"SELECT COUNT(*) FROM flows WHERE {where}", values).fetchone()[0])
        rows = connection.execute(
            f"SELECT dataset,event_id,timestamp,source_ip,source_port,destination_ip,destination_port,protocol,duration,bytes_total,packets_total,rate,label,malicious,source_file,capture_group FROM flows WHERE {where} ORDER BY {SORT_COLUMNS[sort]} {order.upper()}, event_id ASC LIMIT ? OFFSET ?",
            [*values, page_size, (page - 1) * page_size],
        ).fetchall()
        return {"items": [_public_row(row) for row in rows], "total": total, "page": page, "page_size": page_size,
                "sort": sort, "order": order.lower(), "dataset": dataset, "risk_score_available": False}
    finally:
        connection.close()


def flow_stats(database: str | Path, *, dataset: str = "expanded") -> dict[str, Any]:
    _dataset_clause(dataset)
    connection = sqlite3.connect(database)
    try:
        summary = json.loads(_ensure_flow_stats_summary(connection, dataset))
        connection.commit()
        summary.pop("_summary_version", None)
        summary["risk_score_available"] = False
        return summary
    finally:
        connection.close()


def flow_topology(database: str | Path, *, dataset: str = "expanded", limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    where, values = _dataset_clause(dataset)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"SELECT dataset,source_ip,destination_ip,destination_port,protocol,flow_count,malicious_count,bytes_total,first_seen,last_seen FROM flow_topology WHERE {where} ORDER BY flow_count DESC, source_ip, destination_ip LIMIT ?",
            [*values, limit],
        ).fetchall()
        return {"data_mode": "Historical benchmark", "dataset": dataset, "edge_limit": limit,
                "edges": [{**dict(row), "protocol_name": PROTOCOL_NAMES.get(row["protocol"], f"IP protocol {row['protocol']}")} for row in rows],
                "scope_note": "Observed source-to-destination/service aggregates from normalized raw flows; not a live topology."}
    finally:
        connection.close()
