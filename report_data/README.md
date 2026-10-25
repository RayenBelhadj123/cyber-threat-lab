# Real benchmark reporting snapshot

These files contain the actual saved CICIDS2017 Thursday benchmark results, not generated/example scores. They let the reporting app run from a clean checkout and on Streamlit Community Cloud without raw data, training, Docker, or model deserialization.

The snapshot retains metrics, binary labels, historical timestamps/time groups and per-model scores/frozen flags. Source IPs, per-host window IDs and row reference-label strings are excluded. Aggregate attack-label counts remain available in metrics.json. No trained model bundles are included.

The source metrics SHA256 is recorded in metrics.json under reporting_snapshot. The selected model is chosen by grouped development CV, not test performance. All dataset/split limitations still apply: only 68 positive windows from one source host/continuous attack period and shuffled grouped—not forward temporal—holdout. This snapshot is reporting evidence, not a production detector.

Local artifacts/model_benchmark takes precedence when present. Set CYBERLAB_BENCHMARK_DIR explicitly to override the source.
