# SOC platform verification report

**Verification time:** 2026-10-30 17:01 (UTC+01:00)
**Repository:** `RayenBelhadj123/cyber-threat-detection-lab`, local clone `C:\Users\Dell\cyber-threat-lab`, branch `main`. The requested Desktop path `C:\Users\Dell\Desktop\cyber-threat-detection-lab` does not exist.
**Publication:** baseline commit `6064c0b` was previously pushed; the follow-up changes described here are committed locally but have not been pushed.

## Full-dataset evidence

The local store `artifacts/expanded_preparation/full_flows.sqlite` is present (4,772,331,520 bytes after statistics-cache materialization) and passed the API's live consistency comparison with preparation lineage, saved benchmark counts, labels, source dataset totals, feature-window count, and topology aggregate.

| Measure | Verified |
| --- | ---: |
| Input flow rows accounted | 4,610,049 |
| Valid raw flows indexed and available to paginated queries | 4,553,532 |
| Invalid rows | 56,231 |
| Duplicate rows | 286 |
| Blank rows (tracked separately) | 288,602 |
| CICIDS2017 indexed flows | 2,827,475 |
| Separate CIC-DDoS2019 extension flows | 1,726,057 |
| Aggregated feature windows | 197,941 |
| Label aggregate flows | 4,553,532 |
| Protocol aggregate flows | 4,553,532 |
| Hourly aggregate flows | 4,553,532 |
| Topology aggregate flows | 4,553,532 |
| Suspicious reference-label aggregate | 2,259,959 |

Live HTTP calls to `/api/flows` returned exactly 500 rows for a 500-row request; the reported full total matched 4,553,532. Source-IP filtering and ascending timestamp ordering were checked against the returned records. An additional live check summed label, protocol and hourly aggregates to 4,553,532 each and matched the 2,259,959 suspicious-reference-label count to the topology aggregate. The 197,941 feature windows are not raw flows. The legacy `artifacts/flows.ndjson` has 170,217 Thursday-only historical records and remains a separate, explicitly limited view. No model metrics were converted into network events.

The saved learning-size curve contains train AP and validation AP fields, but no validation-loss field/series. The UI reports that absence instead of inventing a loss chart. Full-benchmark flow queries use SQLite; a full expanded Elasticsearch index was not built.

## Checks run

| Check | Result |
| --- | --- |
| `env -u PYTHONPATH ./.venv/Scripts/python.exe -m pytest -q` | **294 passed** in 126.64 seconds; full current suite |
| `npm run build` | **Passed**; TypeScript + Vite, 2,324 modules transformed, 5.34 seconds |
| PowerShell parser on `scripts/launch_soc.ps1` | **Passed** |
| `git diff --check` | **Passed**; only Git LF-to-CRLF working-copy notices |
| Local launcher | **Passed** after stopping the identified stale project API; API, React and Streamlit HTTP checks succeeded, expanded statistics cache warmed |
| Real API `/api/flows/status` | **Ready**; build ID `78c2c6be5378`, indexed-record count matches expanded statistics |
| Real full-store pagination/filter/sort | **Passed**; fetched 500 rows, ascending timestamps verified, source-IP filter returned matching rows |
| Query bounds | **Passed**; `page_size=501` returned HTTP 422; maximum accepted page size is 500 |
| Dataset partition check | **Passed**; CICIDS2017 2,827,475 + CIC-DDoS2019 1,726,057 = expanded 4,553,532 |
| Real API `/api/datasets` | **Passed**; all four dataset/benchmark manifests available; expanded benchmark metrics present |
| Full Playwright/browser suite | **Not run**: prior browser automation was blocked pending approval; not retried |
| Responsive/browser performance verification | **Not verified** without the browser suite |

The local API process was separately restarted from current source and verified over HTTP; this is distinct from the Python in-process test client.

## Actual local service state

Component probes were collected on 2026-10-30 around 16:36 UTC+01:00. These are point-in-time reachability checks.

| Component | Observed state |
| --- | --- |
| Elasticsearch | Unavailable (connection timed out) |
| Kibana | Unavailable (connection timed out) |
| Kafka broker 1 | Unavailable (connection timed out) |
| Spark master UI | Unavailable (connection timed out) |
| Streamlit reporting workspace | Healthy (HTTP reachability) |
| Expanded ML benchmark artifact | Healthy |
| Frozen inference artifact | Healthy |

These are reachability/artifact checks, not claims about ingestion or freshness. No simulated health data was used.

The launcher detected the stale API by its absent/old source `build_id`, refused to kill it automatically, and completed startup after the known project API was stopped explicitly. A fresh API now answers `/api/flows/status` with ready=true. The local full-dataset query path returned 500 rows in a 3.4-second round trip including status/integrity checks; cached statistics returned the correct total in an 843 ms round trip including the same status check. These timings are local point measurements, not latency guarantees.

## Implemented

- Built and verified a chunk-ingested full-flow SQLite index over the expanded benchmark; kept CICIDS2017 and CIC-DDoS2019 separate.
- Added fail-closed status, paginated query, filtering, server-side sorting, dataset statistics and observed-flow topology API routes.
- Added the React Flow investigation view, expanded dataset/index status, provenance labels, saved model comparison and held-out confusion matrix.
- Added the Streamlit `SOC dashboard` workflow embedding the existing React UI, honest frontend/API/index availability guidance, and the one-command local launcher.
- Added API source build-ID checking and statistics-cache validation; inconsistent cached totals now fail closed.
- Versioned the aggregate cache and fixed expanded top-source/top-port rankings to aggregate over both datasets before selecting the global top 12.
- Added an updated architecture diagram, launch instructions and explicit implementation/limitation documentation.
- Inspected the Housing Streamlit source directly; REA was checked in the available MCP/plugin catalog, with no installable/available match found.

## Remaining work / limits

1. Browser-rendered E2E, responsive behavior and browser performance are unverified; the Playwright run remains approval-blocked. Current browser window was not used to interact with the user's separate hosted Streamlit app.
2. Full-benchmark search is SQLite-backed, not Elasticsearch-backed. Elasticsearch/Kibana were unreachable in the latest point-in-time probe.
3. Kafka/Spark were unreachable; streaming freshness, consumer lag, processing throughput and end-to-end live detections are not integrated. Streamlit itself is reachable locally.
4. Grounded AI, analyst-reviewed ATT&CK mapping, Sigma rule execution and case management are not implemented. The companion plan documents evidence/privacy/validation gates; per-flow risk scores remain unavailable.
5. MongoDB/Cassandra are preserved but are not individual SOC health cards; the API has no authentication and is intended for loopback-only use.

## Git state

The follow-up project changes are committed locally and not pushed. `facebook_profile_180.png` is unrelated and remains untracked. No local benchmark database or generated build output is part of the commit.
