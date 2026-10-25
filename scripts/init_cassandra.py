"""Wait for the three-node Cassandra ring and create its lab event table."""
import json
import os
import sys
import time


def main():
    from cassandra.cluster import Cluster
    hosts = [host.strip() for host in os.environ.get("CASSANDRA_HOSTS", "cassandra-1,cassandra-2,cassandra-3").split(",") if host.strip()]
    if len(hosts) != 3:
        raise ValueError(f"Expected exactly three Cassandra nodes; got {hosts!r}")
    deadline = time.monotonic() + 240
    last_error = None
    cluster = None
    session = None
    while time.monotonic() < deadline:
        try:
            cluster = Cluster(contact_points=hosts, connect_timeout=5)
            session = cluster.connect()
            host_rows = list(session.execute("SELECT peer FROM system.peers"))
            live_hosts = [host for host in cluster.metadata.all_hosts() if host.is_up]
            if len(host_rows) >= 2 and len(live_hosts) >= 3:
                break
            cluster.shutdown()
            cluster = None
            session = None
            time.sleep(3)
        except Exception as exc:
            last_error = exc
            if cluster is not None:
                cluster.shutdown()
            cluster = None
            session = None
            time.sleep(3)
    if session is None:
        raise RuntimeError(f"Cassandra three-node ring did not become ready: {last_error}")
    try:
        session.execute("CREATE KEYSPACE IF NOT EXISTS cyberlab WITH replication = {'class':'SimpleStrategy','replication_factor':3}")
        session.execute("""
            CREATE TABLE IF NOT EXISTS cyberlab.event_rows (
                event_day date,
                source_ip text,
                event_timestamp timestamp,
                event_id text,
                event_kind text,
                payload text,
                PRIMARY KEY ((event_day, source_ip), event_timestamp, event_id)
            ) WITH CLUSTERING ORDER BY (event_timestamp DESC, event_id ASC)
        """)
        print(json.dumps({"keyspace": "cyberlab", "replication_factor": 3,
                          "live_nodes": len([host for host in cluster.metadata.all_hosts() if host.is_up]),
                          "table": "event_rows"}))
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"cassandra-init: {exc}", file=sys.stderr)
        raise
