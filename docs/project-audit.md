# Cyber Threat Lab — local project audit and restructure

Date: 2026-10-25. Scope: current local working tree, Streamlit workflows, existing local Elasticsearch/Kibana/HTML services, and downloaded-data integrity. This is not a penetration test, full branch-coverage certification, public deployment or proof of production detection.

## Confirmed issues corrected

| Root cause | Correction | Evidence |
| --- | --- | --- |
| Run/Live detection tests existed without an implementation | Added independent `live_detection.py`, sidebar Data mode, bounded CSV inference, exact frozen model export, read-only Elasticsearch snapshots and explicit historical/paused states | Real browser upload scored all 16,174 feature rows and downloaded actual results; unit tests include unlabeled-flow ingestion, observed IP preservation, unsupported-type/tamper rejection and no querying while paused |
| Batch prediction helper was unfinished | Centralized `predict_csv` and formula-safe `csv_bytes`, and routed the actual batch button through that helper | Contract, numeric/schema/size rejection and formula-defense tests pass |
| Three declared guidance pages had no renderer implementation | Split Domain Review, Code Lab and Assistant into a dedicated guidance module | Real app navigation, explicit save/generate/answer actions and exports pass |
| Page ownership was duplicated and could silently advertise unsupported routes | Added central ownership validation; each route has exactly one renderer | All 32 workspace routes checked through actual entrypoint and browser |
| Local indexed findings lagged current local artifact outputs | Idempotently published trusted existing artifacts without deleting indices; exact document readback and count checks | 170,217 flows, 16,174 feature windows and 80,870 detector findings match local artifacts; original dashboards read back successfully |

The baseline had **5 failing / 209 passing tests**. Missing renderer ownership and the stale index were found by the additional integration/service checks rather than by weakening the original tests.

## Verified results

- **231 tests passed**, full suite; latest run 87.45 seconds.
- Browser: **32 workspace routes + 6 benchmark routes**, no recorded JavaScript page errors, real code download, real feature upload/scoring/download, no fabricated source IPs, empty recent-event state and explicitly labeled historical replay.
- The packaged detector reproduces **3,229 saved held-out scores**, with maximum absolute difference `9.71445146547012e-17`; this verifies serialization/inference parity, not a new model evaluation.
- Native benchmark dashboard: **13 panels** read back. The published benchmark itself and its selected winner were not retrained or reselected.
- Streamlit, Elasticsearch, Kibana and standalone HTML report returned **HTTP 200**. The three running Compose services were healthy.
- **11 downloaded CSVs** passed size/SHA256 checks against their pinned-mirror manifests. This verifies mirror integrity, not independent original-publisher authenticity.
- Clean working-tree copy with **no ignored artifacts or raw data** rendered all benchmark/workspace pages and scored packaged-model inputs using only committed-candidate snapshots.
- Python dependency compatibility check passed; Linux/Python 3.12 dependency resolution passed. No live public-cloud deployment was performed.
- Source/document pattern scan: **66 files**, no matches for the checked credential/private-key signatures. This is not an exhaustive security audit.

## Structure and operating instructions

See [project-structure.md](project-structure.md) for ownership and verification commands, [analytics-feature-parity.md](analytics-feature-parity.md) for functional adaptation boundaries, and the README for Run data / Live data usage.

Local app: `http://127.0.0.1:8502/`.

Machine-readable evidence:

- `artifacts/project-audit/summary.json`
- `artifacts/project-audit/data-integrity.json`
- `artifacts/project-audit/frozen-detector-readback.json`
- `artifacts/project-audit/clean-checkout.json`
- `artifacts/project-audit/source-secret-scan.json`
- `artifacts/workspace-verification/verification.json` and screenshots/downloads
- `artifacts/verification.json` for original index/dashboard readback

## Remaining boundaries — not claimed complete

- The optional Kafka/Spark/Mongo/Cassandra profile is **not running**. Live mode can consume its indexed feature windows, but the audit did not generate fresh capture traffic or verify real-time attacks. Existing stream index contents are historical. No automatic blocking or packet sniffing is provided by the UI.
- The detector remains Thursday-only. Downloaded full-week/DDoS data is intact but **has not been incorporated into a newly validated model**. Source-host attribution and future generalization remain limited; flagged sources require analyst investigation.
- Workspace models, reviews, registry candidates and experiments are session-local, not deployed production state. Some native tree estimators intentionally cannot be imported/exported through the strict uploaded-bundle path; the separate packaged detector has a narrow, hash-pinned loading policy.
- Public Streamlit Cloud deployment and GitHub push were **not performed** in this audit. Changes are local working-tree changes.
- Distributed sink parity, whole-table Cassandra totals, SQL Server integration and a new independent full-week benchmark were not implemented or reverified by this pass.
