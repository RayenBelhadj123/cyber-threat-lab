# Behavioral Threat Detection Lab

A local, reproducible cybersecurity data-science lab using **Elasticsearch, Kibana, Python, Docker Compose, Kafka, Spark, MongoDB, and Cassandra**. It combines flow normalization, source-host/minute windows, ML/deep learning, event replay, Spark batch and streaming, and idempotent multi-store persistence. This is an educational offline lab, not a production detection service.

## What the first run actually uses

The included downloader retrieves **170,366 real CICIDS2017 Thursday morning flow records** from the public WSU-AI-CyberSecurity mirror, not generated benchmark data. Labels inspected in the downloaded file:

| Label | Flow records |
|---|---:|
| BENIGN | 168,186 |
| Web Attack – Brute Force | 1,507 |
| Web Attack – XSS | 652 |
| Web Attack – Sql Injection | 21 |

This is **not the complete CICIDS2017 benchmark**. The file does not establish port-scan, DoS/DDoS, beaconing, or exfiltration detection performance. Those require additional labeled input. The pipeline accepts additional full-flow CSVs under `data/raw/`. Feature-only exports without IP addresses or timestamps are rejected rather than supplied with invented identities.

Original publisher: https://www.unb.ca/cic/datasets/ids-2017.html

Public mirror: https://github.com/WSU-AI-CyberSecurity/data

The direct original archive URL redirected to the publisher's dataset index during setup. The mirror is explicitly attributed; its labels have not been independently verified or corrected. `data/source_manifest.json` records the resolved URL, download date, byte count, and SHA256. Respect the original dataset's attribution and use conditions.

## Measured end-to-end result

The reproducible artifact run contains **170,366 input rows**, 146 rejected rows, 3 duplicates, **170,217 normalized flows**, **16,174 one-minute source-host windows**, **80,870 detector findings**, and **64,696 learned-model prediction records**. The available models are Isolation Forest, HistGradientBoosting, Random Forest, and a CPU PyTorch feed-forward network; the transparent scan/volume rule is also recorded. Current counts and runtime versions are in `artifacts/manifest.json`.

**Critical evaluation limit:** the chronological split put all 68 positive attack windows in training. Validation (3,571 windows) and test (1,569) contain zero positive attack windows. Therefore held-out attack recall, AP, and ROC AUC are undefined; none of these results demonstrate attack detection generalization. Do not present perfect training scores as test performance. This Thursday-only slice contains web attacks, not labeled port-scan or DoS/DDoS events.

On benign-only periods, Isolation Forest produced 36 false alerts across 1,399 observed host-days in validation (about 0.026/host-day) and 10 across 632 in test (about 0.016). The scan/volume rule produced 84/1,399 (about 0.060) and 28/632 (about 0.044). These are slice-specific false-alert observations, not estimates for production traffic. Full metrics, class support, undefined values, and reasons are in `artifacts/metrics.json`.

The four Kibana dashboards each contain eight panels (32 linked visualizations total), including KPI cards, time series, donut breakdowns, and ranked bars. The optional Spark jobs write feature and finding indices that match the existing Kibana data-view wildcards (`cyber-features*`, `cyber-findings*`). Re-run `scripts/provision_kibana.py` after changing saved-object definitions.

The editable architecture diagram is `diagrams/cyberlab-architecture.drawio`; it was generated with the requested Next AI Draw.io workflow and can also be opened in diagrams.net.

## Architecture


```text
CICIDS2017 Thursday CSV
        |
        v
Python validation / normalization / deterministic source-host windows
        |                              |
        v                              v
NDJSON artifacts + ML/DL       Kibana / Elasticsearch base indices
(4 models + scan rule)
        |                              |
        +---- Spark batch -------------+
        |
        +---- replay -> Kafka (3-node KRaft)
                       |
                       v
              Spark Structured Streaming
                       |
            +----------+----------+
            v          v          v
      Elasticsearch  MongoDB   Cassandra
       + Kibana      3-node RS   3-node ring
```

Batch and streaming Spark jobs both persist feature/prediction documents to Elasticsearch, MongoDB, and Cassandra. Kafka topics use three replicas with minimum in-sync replicas of two. All services are in the local Compose network; published ports are bound to `127.0.0.1`. See `diagrams/cyberlab-architecture.drawio` for the detailed editable diagram.

## Quick start (Docker)

Requirements: Docker Engine with Compose, enough free RAM for an additional Elasticsearch/Kibana pair, internet for initial image/data retrieval. Existing Elasticsearch/Kibana services on default ports are left alone.

```bash
docker compose up -d elasticsearch kibana
docker compose build pipeline
docker compose --profile tools run --rm pipeline python scripts/download_data.py
docker compose --profile tools run --rm pipeline
docker compose --profile tools run --rm pipeline python scripts/provision_kibana.py --output artifacts/kibana.ndjson
```

Open **http://localhost:15601/app/dashboards#/view/cyber-overview**.

Elasticsearch is at **http://localhost:19200**. The historical data is from July 6, 2017, not today: use the saved dashboard's historical time range or set an absolute range covering July 6–7, 2017 UTC. A default 'last 15 minutes' filter will show no data.

## Full-week CICIDS2017 training data (optional)

Download the original labeled-flow CSV distribution from a pinned, unofficial Hugging Face mirror:

```bash
.venv/Scripts/python.exe scripts/download_cicids2017_full.py
```

This verifies the archive against the pinned mirror SHA256, validates all eight identity/time/label CSV schemas, counts every row and label, and writes `data/cicids2017_full_manifest.json`. Files are saved separately under `data/cicids2017_full/`; the existing Thursday input, reports and model artifacts are not overwritten. The archive is 283,876,488 bytes and the eight CSVs total 1,203,105,382 bytes, containing 2,830,743 raw flow rows (2,273,097 benign and 557,646 attack-labeled). Counts precede normalization/deduplication.

The mirror is `bencorn/CICIDS2017`, revision `811c007f08d693a8a8b4226c197c138391806ce6`, file `csvs/GeneratedLabelledFlows.zip`. Its checksum is a mirror-integrity check, not independent confirmation from the original publisher. Attribute CIC/UNB and consult their CICIDS2017 dataset terms and paper. Do not substitute MachineLearningCSV: that version lacks the identity/time fields required by the host-window pipeline.

**Do not combine this directory with `data/raw`**: the original Thursday web-attack rows overlap. Use `data/cicids2017_full` alone for a future full-week experiment. Downloading does not retrain models or refresh the Thursday-only reporting snapshot; evaluation design and report caveats must be updated before a full-week benchmark is presented. The downloader refuses to overwrite an existing dataset directory.

## Additional attack captures (optional)

```bash
.venv/Scripts/python.exe scripts/download_attack_extension.py
```

This acquires three complete, checksum-pinned CIC-DDoS2019 mirror CSVs (NTP amplification, UDP-lag and Portmap captures) into `data/cicddos2019_extension/`. Every raw row is retained, including benign examples. A manifest records observed labels/counts and provenance. Downloads resume in a `.staging` directory after network interruptions; a final directory appears only after all files pass size, SHA256 and schema checks. Existing complete datasets are never overwritten.

Keep this extension separate from CICIDS2017. It is attack-heavy and must not determine a production-like evaluation class balance. Hold out an independent capture/domain before tuning, deduplicate/group flows, and explicitly align feature definitions. The mirror also contains 2018-looking timestamp strings despite its CIC-DDoS2019 label; raw values are preserved and the ambiguity must be resolved before temporal integration. No current model is retrained and no improvement is claimed merely because additional attack rows were downloaded. The downloader uses a pinned, unofficial Hugging Face mirror attributed to CIC/UNB and does not publish raw data to GitHub.

## Expanded benchmark status (local, not activated)

The separately saved full-week plus attack-extension experiment has completed: **4,610,049 raw data rows**, **197,941 feature windows**. Whole-capture development CV selected PyTorch MLP, but its frozen threshold detected only **61 of 778 held-out attack windows**, missed **717**, and produced **39 false alerts**. It has **not replaced the active Thursday-only Extra Trees detector**. More data did not establish reliable generalization.

To reproduce the offline experiment without altering the active reports or detector:

```bash
.venv/Scripts/python.exe scripts/run_expanded_benchmark.py
```

This uses all eleven checksum-verified full-week/extension CSVs and excludes the overlapping `data/raw` Thursday input. Results go to `artifacts/model_benchmark_expanded/`, with reusable preparation in `artifacts/expanded_preparation/`. The command refuses an existing nonempty output directory before verifying inputs or fitting models; use `--output` with a fresh directory for another experiment. Do not tune on the now-inspected holdout or automatically promote the selected candidate.

See [the October 29 integrity follow-up](docs/maintenance-2026-10-29.md) for overwrite protection, label/metric validation and the final verification results. [The October 28 maintenance review](docs/maintenance-2026-10-28.md) records the earlier release-validation fixes and browser checks. Each review describes its own service state.

## Windows one-click Kibana launcher

Build the shield-icon launcher from the repository root:

```powershell
dotnet publish launcher/CyberLabLauncher.csproj -c Release -r win-x64 `
  --self-contained false -p:PublishSingleFile=true `
  -p:IncludeNativeLibrariesForSelfExtract=true -o dist
```

Run `dist\CyberThreatLabLauncher.exe`. It opens the Cyber Threat Overview dashboard; if Kibana is not responding, it launches Docker Desktop when needed, waits for the Docker engine, starts only this project's Elasticsearch and Kibana services, waits for readiness, then opens the dashboard. Keep the executable in the project's `dist` folder. Docker Desktop and the .NET 10 Windows Desktop runtime are required; the launcher does not start the optional Kafka/Spark/MongoDB/Cassandra profile.

## Quick start (host Python)

Use Python 3.12 and `uv`, or an equivalent virtual environment. On Windows substitute `.venv/Scripts/python.exe` for `python` after creating the environment.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/download_data.py
.venv/Scripts/python.exe -m cyberlab run --input data/raw --output artifacts --publish
.venv/Scripts/python.exe scripts/provision_kibana.py --output artifacts/kibana.ndjson
.venv/Scripts/python.exe -m pytest tests -q
```

Without `--publish`, the full experiment writes local artifacts and does not require Elasticsearch. On Linux/macOS use `.venv/bin/python` instead. Optional `--max-rows` is for development only and changes dataset coverage; do not present a truncated run as a full benchmark.

## Model prediction and evaluation

The lab supports prediction from either labeled or unlabeled CICIDS-style flow CSVs, using the saved `model.joblib` bundle. Metrics explicitly leave attack metrics undefined when a partition has no positive labels.

```bash
# Optional for host-Python users; the Docker pipeline image includes CPU PyTorch.
uv pip install --python .venv/Scripts/python.exe -r requirements-dl.txt

.venv/Scripts/python.exe -m cyberlab predict \
  --input data/raw/Thursday-WorkingHours.pcap_ISCX.csv \
  --model artifacts/model.joblib --output artifacts/predictions-new.ndjson
.venv/Scripts/python.exe -m cyberlab evaluate \
  --predictions artifacts/predictions-new.ndjson --output artifacts/evaluation-new.json
```

On Linux/macOS, use `.venv/bin/python` in place of `.venv/Scripts/python.exe`. Review each output's labels and support before interpreting precision/recall. Prediction on unlabeled files is for scoring only and cannot establish efficacy.

## Comparative model benchmark and charts

Run the full-data benchmark (no row cap/downsampling) with:

```bash
.venv/Scripts/python.exe scripts/run_model_benchmark.py \
  --input data/raw/Thursday-WorkingHours.pcap_ISCX.csv \
  --output artifacts/model_benchmark
```

It compares a prior baseline, Logistic Regression, HistGradientBoosting, Random Forest, Extra Trees, Isolation Forest, and a PyTorch MLP. It selects by mean grouped-CV average precision, tunes the decision threshold only on development out-of-fold predictions, and reports a separate held-out test fold. Whole 5-minute UTC buckets stay within one fold, but the groups are shuffled—not split forward in time. A 25/50/75/100% whole-group learning-size curve and train-versus-CV AP gap are included as overfit/underfit diagnostics; class weighting and regularization are used for the rare attack class.

Open `artifacts/model_benchmark/report.html` for the SVG charts, metric table, and interactive model explorer. The explorer ranks models by a selected metric, lets you focus on any candidate, refreshes `metrics.json` every 30 seconds when served, and keeps an embedded snapshot for file viewing. `report_preview.png` is a static image preview. Machine-readable results and reproducibility artifacts are saved alongside them: `metrics.json`, `cv_results.csv`, `development_oof_predictions.csv`, `held_out_test_predictions.csv`, and `models.joblib`.

### Streamlit reporting workspace

An additional reporting workspace adapts the DataIQ housing project's workflow. The original six views are read-only benchmark evidence. Turn on **Full analytics workspace** in the sidebar for the expanded CSV-to-model workflow, without replacing Kibana or the published benchmark.

The full workspace has project templates and four app modes, CSV upload, smart role/feature setup, audited cleaning, exploration, supervised and unsupervised labs, prediction and model bundles, explainability and scenarios, forecasting, OLAP, governance, experiment tracking and report exports. Search pages/columns or filter workflow areas; **Show advanced pages** controls the complete navigation. The bundled training/exploration table is `report_data/workspace_features.csv`, derived from actual saved feature windows, with provenance in `workspace_provenance.json`. Uploads are bounded to 25 MB, 50,000 rows and 200 columns; larger capture processing stays in the offline pipeline. Interactive experiments live in the current browser session and never silently retrain or promote the fixed benchmark. See `docs/analytics-feature-parity.md` for the reference mapping and safety differences.

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://127.0.0.1:8502`. Local serving uses explicit loopback-only flags; shared Streamlit config does not force a host-specific bind or port, so cloud hosting can manage them. A clean clone uses the real, identifier-free snapshot in `report_data/`. If local benchmark artifacts exist, they take precedence; `CYBERLAB_BENCHMARK_DIR` can select another trusted directory. On Linux/macOS substitute `.venv/bin/python`. See `docs/reporting-reference.md` for the adapted patterns and deliberate differences. No reference-project files or data are modified.

### Run data and Live data detection

Use **Data mode** at the top of the sidebar:

- **Reports** preserves the six benchmark views and optional full workspace.
- **Run data** scores an uploaded feature-window CSV or CICIDS flow CSV with the published Extra Trees model at its frozen threshold. Feature uploads require the eight exact model columns; `source_ip` and `timestamp` are optional observed metadata, never invented. Flow uploads use the existing ingestion/aggregation rules, allow absent labels and report rejected/duplicate rows. Download the template or scored results. Inputs are bounded to 25 MB / 50,000 rows / 200 columns.
- **Live data** scores the latest bounded windows already present in `cyber-features-spark-stream-v1`. Explicitly fetch once or start 5–60 second refresh; paused views make no source query. The default last-15-minute event filter prevents 2017 data from appearing as current attacks. **Include historical / replayed events** is clearly labeled, opt-in replay inspection, not evidence of attacks now.

The server-side `CYBERLAB_LIVE_ELASTICSEARCH_URL` defaults to `http://127.0.0.1:19200`; visitors cannot supply arbitrary URLs. Start the existing Kafka/Spark pipeline separately to supply recent windows. This view is **not packet capture**, does not automatically start distributed services, and does not block hosts. Polling interval is not end-to-end latency: completed host/minute aggregation and pipeline delay also apply. A hosted Streamlit server cannot reach your PC through its own localhost.

Results identify **suspicious source IPs for analyst review**, not confirmed attackers. Identifier-reduced uploads can be scored but cannot identify hosts. The current detector remains Thursday-only and has not been retrained on the downloaded extensions. The packaged `.skops` model is hash/version checked; arbitrary model uploads are not accepted in detection mode.

See `docs/project-structure.md` for module ownership and reproducible checks, `docs/project-audit.md` for verified coverage and remaining operating limits, and `docs/soc-platform-plan.md` for the SOC roadmap and React/FastAPI implementation.

### React SOC evidence dashboard

The React/TypeScript SOC surface adds full historical-flow exploration without replacing the existing Streamlit, Kibana, or pipeline work. It serves the verified expanded raw-flow index from a local SQLite store with server-side pagination, filters, sorting, statistics, and topology; it does **not** claim full-benchmark Elasticsearch indexing. The overview, saved ML evidence, dataset provenance, React Flow investigations, and point-in-time service health remain distinct from the legacy Thursday-only flow capture. The live health checks report reachability, not Kafka lag, Spark throughput, or data freshness.

The full store is built locally from the 11 manifest-listed benchmark CSVs in bounded chunks; the roughly 4.77 GB database is ignored by Git and must not be copied into the repository. If absent, create it with `.venv\Scripts\python.exe scripts\build_full_flow_store.py --chunk-size 50000`. The API fails closed when it is absent or inconsistent instead of substituting the smaller legacy capture.

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-soc.txt
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The default API base is `http://127.0.0.1:8000/api`; the Streamlit report link defaults to `http://127.0.0.1:8502` and its health probe can be configured with `CYBERLAB_STREAMLIT_URL`. See [the SOC implementation and limitations](docs/soc-platform-plan.md), [the feature-parity review](docs/analytics-feature-parity.md), [the architecture diagram](docs/soc-architecture.html), and [the verification report](docs/soc-verification-report.md). To run the browser suite, start the API and frontend, then run `npm run test:e2e` from `frontend`; this verification pass did not run it.

### Streamlit Community Cloud

Create an app with these settings:

- Repository: `RayenBelhadj123/cyber-threat-detection-lab`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python: `3.12`

Root `requirements.txt` includes the reporting dependencies. The app loads `report_data/` on a clean clone, so it does not need local Docker services, raw training data or PyTorch. Its default benchmark views are read-only; the optional full workspace runs bounded, explicit-button experiments in session memory and supports validated model bundles, not arbitrary pickle uploads. This is not a cloud deployment of Elasticsearch/Kibana. Local training artifacts and network identities remain excluded from Git. The committed snapshot is derived from the actual benchmark; it is not synthetic.

### Native Kibana report

To publish the benchmark results as a **native Kibana dashboard** (not an HTML iframe), run:

```bash
.venv/Scripts/python.exe scripts/publish_benchmark_metrics.py
.venv/Scripts/python.exe scripts/provision_kibana.py --output artifacts/kibana-with-benchmark.ndjson
```

Open `http://localhost:15601/app/dashboards#/view/cyber-benchmark`. This creates/updates `cyber-benchmark-v1` with the latest run's model scores, selected-model confusion counts, dataset KPIs, and learning-size diagnostics, then provisions the dashboard and data view. Model document IDs are stable, so republishing updates the current benchmark snapshot rather than building run history. The dashboard uses the same exploratory-data caveats as the HTML report.

To deploy the standalone report locally in Docker (bound only to `127.0.0.1`):

```bash
docker compose --profile report up -d model-report
```

Open `http://127.0.0.1:18090/`. New benchmark runs update the mounted artifacts; the interactive panel refreshes metrics automatically, while reloading the page refreshes all embedded charts. Stop the report service with `docker compose --profile report stop model-report`. This is a local development deployment, not public hosting.

**Important limitation:** this Thursday slice contains only 68 positive source-host/minute windows, all from one source IP and one continuous web-attack period; the held-out fold contains only a small number of positives. The grouped split blocks short time-bucket leakage but cannot establish generalization to independent campaigns, hosts, or networks. Treat these scores as exploratory, not as proof of a deployment-ready detector. Add independently captured attack campaigns before making that claim.

## Optional distributed profile: Kafka + Spark + MongoDB + Cassandra

Generate the `artifacts/` files first, then start the isolated big-data services. This optional profile is resource-intensive; this Compose file binds published ports only to localhost. It does not use or stop containers from other Compose projects.

```bash
# Starts the three-node Kafka KRaft cluster, three-node MongoDB replica set,
# three-node Cassandra ring, Spark master/workers, streaming, and batch job.
docker compose --profile bigdata up -d --build

# Replay normalized flows and learned-model predictions into Kafka.
# Rate limit is an overall cap; omit --rate or set it to 0 for unlimited replay.
.venv/Scripts/python.exe scripts/stream_replay.py --artifacts artifacts --rate 1000

# Re-run the one-shot Spark batch job after artifacts change.
docker compose --profile bigdata run --rm spark-batch

# Inspect the long-running stream and service states.
docker compose --profile bigdata logs --tail=100 spark-stream
docker compose --profile bigdata ps --all
```

The replay prints its actual topic counts, elapsed time, and messages/second. Spark writes `artifacts/spark_batch_metrics.json` and periodically refreshes `artifacts/spark_stream_metrics.json`. Inspect the terminal logs and these files; a running container alone is not proof that data reached each sink. Kafka topics are `cyber-flows` and `cyber-predictions`, each with six partitions, RF=3, and min ISR=2. MongoDB writes deterministic-ID upserts to separate batch/stream collections. Cassandra uses keyspace `cyberlab`, RF=3, bounded individual prepared writes at `QUORUM`, and event IDs include the event kind so batch and stream versions of the same event do not overwrite one another. Spark indices use the existing Kibana wildcards `cyber-features*` and `cyber-findings*`.

**Streaming replay caveat:** Spark streaming is stateful and uses a 10-minute event-time watermark plus persistent checkpoints. Replaying into topics/checkpoints that already contain smoke or later-timestamp events can skip earlier windows, and trailing windows remain pending until event time advances. Do not treat reused-checkpoint count differences as parity results. For a clean full-data comparison, run `docker compose --profile bigdata up -d --build`, then `.venv/Scripts/python.exe scripts/run_stream_parity.py`. The harness creates disposable topics and a fresh checkpoint, replays data, advances the watermark, verifies sink counts and Elasticsearch ID sets, and restarts the normal stream. It leaves per-run topics/checkpoint and `artifacts/stream_parity_<run-id>.json` for audit; use `--run-id` to choose a suffix.

For manual topic routing and watermark control instead:

```bash
.venv/Scripts/python.exe scripts/stream_replay.py --artifacts artifacts --rate 1000 \
  --flow-topic cyber-flows-parity --prediction-topic cyber-predictions-parity
```

After the stream metrics report zero Kafka lag, advance the watermark past the last flow time to flush final real windows:

```bash
.venv/Scripts/python.exe scripts/stream_replay.py --watermark-only \
  --flow-topic cyber-flows-parity --prediction-topic cyber-predictions-parity \
  --watermark-timestamp 2017-07-06T16:15:00+00:00
```

The watermark marker is synthetic; it advances stream time but its own final window remains buffered unless later events arrive. Verify sink counts and event-ID sets before stopping the temporary consumer.

Useful read-only checks after startup:

```bash
docker compose --profile bigdata exec kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka-1:9092 --describe
docker compose --profile bigdata exec mongo-1 mongosh --quiet --eval 'rs.status().members.map(m => ({name:m.name,state:m.stateStr}))'
docker compose --profile bigdata exec cassandra-1 nodetool status
curl -fsS 'http://localhost:19200/_cat/indices/cyber-features-spark-*,cyber-findings-spark-*?v'
```

Stop only this project's containers with `docker compose --profile bigdata down`; omit `-v` to preserve its data volumes. Do not stop or delete unrelated Compose projects.

## Experiment design and limitations

- **Unit of evaluation:** source-host / one-minute window, not individual network flows. A malicious window contains at least one flow with a non-BENIGN reference label.
- **Reference labels are not model inputs.** IP identifiers, absolute timestamps, flow IDs, labels and train/test membership are excluded from feature vectors. Source IP remains investigation metadata and the grouping key.
- **Chronological splitting:** unique window times are allocated 60/20/20 to training, validation and test. No timestamp is shared between partitions. Results depend strongly on the attack schedule: a partition may contain no malicious windows, or a test attack class may not have appeared in training. Such cases must be reported, not repaired using random stratification.
- **Historical timestamp assumption:** CIC timestamps are interpreted day-first in `America/Moncton` and converted to UTC. This timezone assignment is a documented lab assumption, not independently verified capture metadata.
- **Availability caveat:** flow features are extracted from completed flows. The window timestamp represents the host aggregation boundary, not proof that every flow-derived feature was available in real time at that instant. This is an offline benchmark, not a validated low-latency streaming detector.
- **Calibration:** Isolation Forest is trained only on benign training windows; validation data is used for its alert threshold. The supervised benchmark is unavailable if training lacks both classes. Do not treat missing AP or unavailable models as successful evaluations.
- **Feature caution:** the optional failure-ratio feature defaults to zero when the exporter lacks a defensible failure indicator; connection failures must not be inferred from missing labels or arbitrary TCP flag combinations.
- **Baselines:** the scan/volume rule is a transparent behavioral heuristic, not a web-attack signature. Its performance on this first data slice may be poor.
- **Model claims:** inspect `artifacts/metrics.json`, class support, and split ranges before reporting recall. A benign-only test partition can estimate false alerts, but cannot estimate attack recall.
- **Generalization:** this controlled 2017 dataset cannot demonstrate performance against present-day production threats.

## Artifacts

The pipeline writes normalized flow, feature and finding NDJSON files, metrics, a fitted model bundle, and a run manifest. The big-data profile adds `spark_batch_metrics.json` and `spark_stream_metrics.json`; `scripts/stream_replay.py` reports measured Kafka counts and throughput. The provisioner can export Kibana saved objects. SHA256 provenance is in `data/source_manifest.json`. Outputs are reproducible experiment artifacts, not hand-written sample results.

Elasticsearch document IDs are deterministic so replaying the same file does not add duplicate events. To avoid mixing experiments, use the same complete input set when rerunning; changing or reducing input does not delete previously indexed documents. A different experiment should use a fresh dedicated Compose volume or explicitly managed versioned indices after saving prior results. Never delete unrelated indices.

## Safety and operational scope

This is a **localhost-only development lab**. Authentication is disabled deliberately and ports bind to `127.0.0.1`. Do not expose these containers or ports to a network, deploy this configuration to production, or put sensitive production logs here. The installed lab image version is pinned for reproducibility, not claimed to be latest or production-hardened.

The project reads historical data; it does not scan hosts, generate attack traffic, or modify existing Docker stacks.

```bash
docker compose stop                 # stop this lab, preserve data
docker compose down                 # remove this lab's containers, preserve named volumes
# docker compose down -v            # destructive: removes this lab's indexed data
```

## Suggested extensions

1. Add the full Wednesday and Friday captures only from a verified source; record checksums and publisher attribution.
2. Evaluate known-attack and unseen-attack test subsets with explicit class support.
3. Compare class weighting, temporal stability, and alert budgets before tuning model complexity.
4. Test the Spark streaming watermark with controlled late/out-of-order events and record accepted/dropped counts.
5. Design authenticated, TLS-enabled deployment separately; do not expose this lab Compose file to networks.
