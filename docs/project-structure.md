# Project structure and operating boundaries

## Three independent application surfaces

| Surface | Entry / implementation | Evidence and state |
| --- | --- | --- |
| Saved benchmark reports | `streamlit_app.py`, `cyberlab/reporting.py` | Read-only metrics and saved prediction CSVs; original CV-selected winner remains fixed |
| Interactive analytics | `cyberlab/workspace.py` and page modules below | Explicit-button, bounded session experiments; no benchmark replacement or service deployment |
| Run / Live detection | `cyberlab/live_detection.py` | Packaged hash/version-checked Extra Trees inference, frozen threshold; uploaded windows/flows or bounded existing Elasticsearch stream snapshots |

The sidebar **Data mode** switches Reports / Run data / Live data. Within Reports, **Full analytics workspace** opens the full workflow. Turning it off returns to the six benchmark views.

## Workspace module ownership

`workspace.py` owns input validation, project/context setup, fingerprints, navigation, central route ownership validation and the core Start Here / Executive Summary / Report Generator pages. Its page registry requires exactly one renderer per advertised page and rejects missing or duplicate ownership.

- `workspace_data.py`: project management, smart setup, audited cleaning, overview, exploration, quality, comparisons and dictionary.
- `workspace_ml.py`: supervised fitting/CV/holdout evidence, predictions, explainability, scenarios, fit diagnostics, safe inference bundles, experiment tracking and review registry.
- `workspace_advanced.py`: unsupervised analytics, educational offline reinforcement learning, ordered forecasting, OLAP/export, readiness/pipeline/big-data checks, business-cost scenarios and conclusions.
- `workspace_guidance.py`: analyst domain review, whitelisted code previews and local evidence-based assistance. No arbitrary code execution or automatic external LLM call.

Shared session contracts include `workspace_cleaned`, `workspace_cleaning_audit`, and `workspace_model_result`; data/feature/target fingerprints prevent stale current models from masquerading as evidence for another setup. Inputs are bounded; source files are not overwritten by UI cleaning.

## Existing pipeline and services remain separate

- `cyberlab/pipeline.py`: established flow identity/time validation and host/minute features; offline model artifacts.
- `cyberlab/benchmark.py`: fixed grouped-CV comparison and report generation, separate from exploratory workspace fits.
- `cyberlab/publish.py`: deterministic, idempotent local Elasticsearch publishing with exact document readback; no index deletion.
- `spark_jobs/`: optional distributed batch/stream consumers and sink adapters.
- `scripts/stream_replay.py`, `scripts/run_stream_parity.py`: explicit replay and isolated parity verification.
- `scripts/provision_kibana.py`, `scripts/publish_benchmark_metrics.py`: native dashboards and benchmark documents.
- `compose.yaml`, `deploy/report/`: loopback-bound local services and standalone HTML serving.
- `scripts/download_cicids2017_full.py`, `scripts/download_attack_extension.py`: provenance/checksummed acquisition, not automatic model retraining.
- `data/`, `artifacts/`, `dist/`: local datasets, generated evidence/build outputs and launcher; keep raw data/identities out of public reporting snapshots.
- `report_data/`: real identifier-reduced report/input snapshots, lineage, and the inference-only packaged detector. The detector version is pinned in root requirements; its trust checks do not apply to arbitrary uploaded files.

## Reproducible checks

From the project directory on Windows (use `.venv/bin/python` on Linux/macOS):

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/verify_lab.py --artifacts artifacts
# Install Playwright only in the development/test environment, not as a Cloud runtime dependency.
uv pip install --python .venv/Scripts/python.exe playwright
.venv/Scripts/python.exe scripts/verify_workspace_browser.py
```

Browser verification requires the local Streamlit server and Chrome. It checks all six benchmark routes, all advertised workspace routes, real feature upload/scoring/download, no invented source IPs, recent empty-event handling and explicitly labeled historical Elasticsearch replay. Browser screenshots/downloads and machine-readable results are written under `artifacts/workspace-verification/`. It does not generate real attack traffic or prove fresh packet ingestion.

`verify_lab.py` checks local source/artifact versus indexed counts and the original four dashboard objects. The benchmark dashboard is checked separately in the project audit. A count mismatch is a publication discrepancy to investigate, not a reason to weaken assertions or delete indices. Republish trusted existing artifacts explicitly, then read the exact targets back.

## Known operating limits

- A running Elasticsearch/Kibana pair does not imply the optional Kafka/Spark/Mongo/Cassandra profile is running.
- The Live page does not start packet capture or distributed consumers; historical records keep historical timestamps.
- Polling every five seconds is not five-second detection latency. Window completion, ingestion, processing and indexing contribute latency.
- The detector remains Thursday-only; downloaded full-week/DDoS extensions are separate and require new independent validation before deployment or improvement claims.
- Uploaded workspace bundles deny all unknown skops types; some native tree models intentionally remain session-local. The separate packaged detector accepts only its hash-pinned estimator and a narrow static sklearn tree representation.
- Session candidate promotion is a review record, not automatic deployment.
- Local success and dependency resolution do not prove a public Streamlit Cloud deployment or an updated GitHub commit.
