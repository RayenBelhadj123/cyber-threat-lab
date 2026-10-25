# Real benchmark reporting snapshot

These files contain the actual saved CICIDS2017 Thursday benchmark results, not generated/example scores. They let the reporting app run from a clean checkout and on Streamlit Community Cloud without raw data, training, Docker, or model deserialization.

The snapshot retains metrics, binary labels, historical timestamps/time groups and per-model scores/frozen flags. Source IPs, per-host window IDs and row reference-label strings are excluded. Aggregate attack-label counts remain available in metrics.json. The default reporting views never deserialize estimators. Separately, `detector.skops` is a packaged inference-only copy of the existing CV-selected Extra Trees classifier, with its frozen OOF threshold, version and SHA256 recorded in `detector_metadata.json`. Run/Live detection checks the hash and exact scikit-learn version, denies unknown Python types except the narrowly vetted sklearn internal tree representation, and does not accept uploaded estimator files. This does not retrain or improve the benchmark.

The source metrics SHA256 is recorded in metrics.json under reporting_snapshot. `workspace_features.csv` additionally contains the actual eight per-window numeric input features, timestamp/time_group and malicious target from the existing feature pipeline, with independent file/source hashes in `workspace_provenance.json`. It omits IPs, IDs and the older pipeline's split column. These are genuine Thursday feature windows, not synthetic rows and not a merger of newly downloaded attack data. The optional workspace can use this table for bounded, session-only experiments.

The selected benchmark model is chosen by grouped development CV, not test performance. All dataset/split limitations still apply: only 68 positive windows from one source host/continuous attack period and shuffled grouped—not forward temporal—holdout. This snapshot is reporting evidence, not a production detector.

Local artifacts/model_benchmark takes precedence when present. Set CYBERLAB_BENCHMARK_DIR explicitly to override the source.
