# SOC platform: architecture review and staged implementation

## Repository findings

The repository is a Python-first, offline-capable cybersecurity lab. `cyberlab.pipeline` normalizes and aggregates the identity/time-preserving CIC flow data; `cyberlab.benchmark` and the current expanded benchmark workflow write reproducible metrics/predictions; `cyberlab.reporting` validates saved evidence; `cyberlab.live_detection` handles bounded inference and optional Elasticsearch stream reads. The existing Streamlit report, Kibana dashboards, Docker Compose services (Elasticsearch, Kibana, Kafka, Spark, MongoDB, Cassandra), replay scripts, and full-data experiments are retained. The expanded benchmark artifacts record 4,610,049 input flow rows and 197,941 source-file/host/minute windows; this is separate from the legacy Thursday-only flow/index artifacts.

No SOC service should call a historical artifact “live.” The React dashboard therefore carries an always-visible historical evidence label, and the legacy traffic view explicitly states its narrower Thursday-only provenance. Service health is probed from the API, not hard-coded as healthy. The full-week expanded benchmark is the source for model/dataset metrics; the 170k-row legacy flow artifact is not substituted for it.

## Three highest-impact opportunities

| Rank | Improvement | Feasibility / dependencies | Complexity | Why it matters |
|---|---|---|---|---|
| 1 | **Full-data model evidence + provenance dashboard (implemented first)** | High: saved expanded `metrics.json`, split/capture metadata, existing Python evaluation code; FastAPI + React/TypeScript + Vite + Tailwind + Recharts. No retraining or new datastore. | Medium | Reuses verified evaluation outputs and makes the full 4.61M input scope, held-out evaluation, model selection policy, and label/window distinction inspectable. Avoids silently treating the Thursday-only 170k artifact as the whole benchmark. |
| 2 | **Read-only investigation and topology explorer (implemented on SQLite)** | Full-flow builder, verified 4.55M-row local store, FastAPI page/filter/sort/statistics/topology endpoints, and React Flow drill-down. The current query backend is SQLite; Elasticsearch remains a separately probed service and does not index the expanded full-flow store. | Medium–high | Gives analysts paginated access to the full verified historical flow set without browser bulk loading. Nodes and edges derive only from observed indexed records; the feature-window and legacy Thursday artifacts remain separate. |
| 3 | **ATT&CK/Sigma context and analyst-assisted explanations** | Medium: curated Sigma rules and MITRE ATT&CK Navigator layer format provide a useful vocabulary; add a versioned, reviewed mapping catalogue linking only supported dataset labels/detector evidence to candidate techniques. An LLM explanation layer would need a separately configured model/provider, strict evidence citations, privacy controls and abstention tests. | High | Helps analysts communicate why evidence is relevant, but technique attribution from broad CIC labels is not authoritative. Sigma rules must be parsed/validated against observed fields; AI must summarize only supplied evidence and must not create alerts or change scores. |

Security Onion's useful product pattern is a unified analyst workflow across network/security telemetry and case review, rather than another independent pipeline. React Flow is appropriate for a graph workspace with inspectable nodes/edges; ATT&CK Navigator is a useful coverage/layer representation, while Sigma supplies portable rule-as-code ideas. Open Lovable is most useful as a design/iteration reference, not as a runtime dependency or a reason to replace the existing stack. This project borrows workflow patterns only; it does not copy branding or assets.

## Implemented slice: verified full-flow store, SOC API and React investigation UI

The full-flow layer is now built from all eleven manifest-listed expanded benchmark CSVs, with a disk-backed local SQLite index. It preserves the source datasets separately, the raw-flow versus feature-window distinction, and historical provenance. The API does not fall back to the old Thursday-only flow file.

The matching repository checkout in this environment is `C:\Users\Dell\cyber-threat-lab`; the requested `C:\Users\Dell\Desktop\cyber-threat-detection-lab` folder is absent. Its configured Git remote is the requested `RayenBelhadj123/cyber-threat-detection-lab` repository.

### REA and reference project

REA was searched in the available plugin/MCP catalog; no installed/available match was found. The local Housing Streamlit source was inspected directly. Reusable source patterns include `clean_data`, `data_cleaning_report`, `dataset_health`, `evaluate_all_models`, `model_feature_importance`, `executive_report_markdown`, and `report_generator_markdown`. The adapted feature matrix and safety differences are documented in [analytics-feature-parity.md](analytics-feature-parity.md). No Housing files or data were changed.

### Verified local data

The manifest-listed expanded benchmark comprises eight CICIDS2017 CSVs and three separate CIC-DDoS2019 extension CSVs. Live inspection of the built store and its preparation/metrics metadata verified:

| Count/evidence | Verified value |
| --- | ---: |
| Input flow rows accounted | 4,610,049 |
| Valid raw flows indexed and explorable | 4,553,532 |
| Invalid / duplicate rows | 56,231 / 286 |
| Blank CSV rows tracked separately | 288,602 |
| CICIDS2017 / CIC-DDoS2019 valid flows | 2,827,475 / 1,726,057 |
| Aggregated feature windows (not raw flows) | 197,941 |
| SQLite store size | 4,772,331,520 bytes (about 4.77 GB, including cached dashboard statistics) |
| Topology aggregate flow total | 4,553,532 |

The input accounting invariant is `input_rows = valid_flows + invalid_rows + duplicate_rows`; blank rows are separate. The topology total and suspicious-reference-label aggregate match the saved evidence. `artifacts/flows.ndjson` remains a separate 170,217-record Thursday-only historical capture. Model metrics are never converted into synthetic event records. Risk scores/per-flow predictions are unavailable in the full-flow store.

### API

`api/main.py` exposes a read-only local FastAPI API:

- `GET /api/health`, `/api/summary`, `/api/ml`, `/api/datasets` — API state and saved benchmark, split, model and source-lineage evidence.
- `GET /api/flows/status` — verifies SQLite counts, source lineage, dataset/label totals, feature windows and topology conservation against saved benchmark evidence.
- `GET /api/flows` — dataset-scoped, server-side search/filter, pagination (up to 500 rows), allowlisted sorting, source/destination IP and port, protocol, label, and timestamp range.
- `GET /api/flows/stats` and `/api/flows/topology` — aggregations over actual indexed records.
- `GET /api/traffic` and `/api/traffic/stats` — clearly labeled legacy Thursday-only flow evidence; never a fallback for expanded flows.
- `GET /api/health/components` — real HTTP/TCP reachability checks for Elasticsearch, Kibana, Kafka broker 1, Spark UI and Streamlit, plus saved ML/inference artifacts.
- Existing `/api/findings` and `/api/investigations/{window_id}` remain bounded to saved local pipeline artifacts.

If the full store or verification manifest is absent or inconsistent, full-flow routes return 503 rather than silently serving a smaller dataset. The full-flow query backend is SQLite; this is not a full-benchmark Elasticsearch index.

### React SOC UI and model evidence

The existing React/TypeScript application retains its navigation and adds:

- Overview with separate expanded input, valid-flow, feature-window and legacy-capture counts.
- Dataset/source provenance and readiness for the actual full-flow index.
- Paginated, filterable, server-sorted full-flow search and React Flow topology; node selection drills into actual stored flow evidence.
- Model comparison with training AP, grouped-CV AP, and held-out test metrics; saved precision, recall, F1, average precision, confusion matrix and learning-size validation-AP curve.
- System-health cards from point-in-time probes; external links to the existing Streamlit reports and Kibana.

The saved experiment did not contain a validation-loss series, so the UI identifies that omission and shows saved validation AP instead. Historical event data is not presented as live traffic. The flow explorer does not show risk scoring where no per-flow scores exist.

### Launch (Windows PowerShell)

**Primary local entry:** run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\\launch_soc.ps1` from the repository root, then open Streamlit at `http://127.0.0.1:8502` and select **Workflow → SOC dashboard**. The launcher starts or reuses the local API, React frontend and Streamlit services; it verifies the API source build ID, refuses stale/unknown listeners instead of terminating them, waits for actual HTTP readiness, and prepares the expanded statistics cache. The React interface is still directly available at `http://127.0.0.1:5173`.

Install API dependencies and start it from the repository root:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-soc.txt
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

If the full-flow database is absent, build it locally from the verified CSV sources (a long-running multi-GB operation):

```powershell
.venv\Scripts\python.exe scripts\build_full_flow_store.py --chunk-size 50000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

For manual startup, open Streamlit at `http://127.0.0.1:8502` and choose **SOC dashboard**; the React frontend remains directly available at `http://127.0.0.1:5173`. The default API base is `http://127.0.0.1:8000/api`; Streamlit is linked/probed at port 8502 by default and can be changed with `CYBERLAB_STREAMLIT_URL`. The local API is unauthenticated and must remain loopback-bound unless authentication and deployment hardening are added.

### Verification and live-service limits

The current 294-test Python suite exercises full-flow status, filtering, pagination, sorting, cached-stat integrity and dataset separation. Live API checks against the local SQLite index returned all 4,553,532 flows, confirmed 500-row query bounds, timestamp sorting and source-IP filters. The API's cached statistics are checked against indexed counts and topology; the launcher prepares that cache before opening Streamlit. Full data stays server-side; the browser only requests bounded pages and aggregates.

At the 2026-10-30 16:36 UTC+01:00 reachability snapshot, Streamlit, saved expanded benchmark and frozen inference artifact were healthy. Elasticsearch, Kibana, Kafka broker 1 and Spark master UI timed out. These checks report connection state only, not ingestion, freshness, consumer lag, or Spark throughput.

On 2026-10-30, the launcher detected stale API processes by comparing the `/api/health` source build ID to the current API and flow-store source hashes; it refused to terminate unknown listeners. After the identified stale project API was stopped, the launcher started the current API and verified `/api/flows/status` HTTP 200, full-index readiness and a rebuilt version-2 statistics cache. Current source build ID: `78c2c6be5378`.

### Still incomplete

- Full-benchmark Elasticsearch search/indexing; the implemented full-flow backend is SQLite.
- Live service integration beyond real reachability probes; no Kafka lag, Spark throughput, freshness or live-alert evidence is shown.
- Grounded AI analyst explanations, evidence-reviewed MITRE ATT&CK technique mapping, Sigma rule execution, and case management.
- Per-flow model predictions/risk scores (not present in source artifacts).
- Individual SOC health probes for MongoDB and Cassandra.
- Production authentication/authorization and multi-user hardening.
- Browser-rendered E2E, responsive behavior and browser performance remain unverified; the previously approval-blocked Playwright run was not retried. Current Python suite and production-build evidence is in [the verification report](soc-verification-report.md).

### Security-workflow scope and evidence gates

- **ATT&CK:** not implemented. Add only a versioned, analyst-reviewed mapping from exact dataset labels or detector evidence to candidate techniques; every mapping must carry source evidence, scope and uncertainty. Broad benchmark labels alone must not be displayed as confirmed technique attribution.
- **Sigma:** not implemented. Before executing rules, validate supported log sources and fields against the actual stored event schema, pin the Sigma ruleset/parser version, and test matches/non-matches on observed records. No alert count may be inferred from ML metrics.
- **AI explanations:** not implemented. Any later assistant must receive bounded saved evidence, cite each claim to evidence identifiers, abstain when evidence is insufficient, avoid changing scores/alerts, and pass prompt-injection/privacy tests. Provider access and data-retention controls must be explicit.
- **Cases:** not implemented. A later case workflow needs evidence references, analyst-authored notes, explicit state transitions and an audit trail; authentication/authorization is a prerequisite for multi-user use.

## Reference research

- Security Onion project: https://github.com/Security-Onion-Solutions/securityonion — analyst workflow and integrated telemetry are useful architectural inspiration; not a dependency for this lab.
- React Flow / xyflow: https://reactflow.dev/ and https://github.com/xyflow/xyflow — suitable future foundation for a graph-based investigation view.
- MITRE ATT&CK Navigator: https://github.com/mitre-attack/attack-navigator — coverage/layer model; map only evidence-backed techniques and preserve uncertainty.
- SigmaHQ: https://github.com/SigmaHQ/sigma — portable detection-rule format; adoption requires validating logsource/field compatibility and rule execution.
- Open Lovable: https://github.com/firecrawl/open-lovable — a Firecrawl-supported website recreation/code-generation project; useful for rapid visual exploration only, not a runtime dependency. Keep final project code original and in its existing repo.

These projects inform future design choices only. No code, detections, labels, scores, assets or branding were imported from them.
