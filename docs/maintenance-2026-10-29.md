# Local integrity maintenance — October 29, 2026

Scope: finish a bounded safety review of the existing uncommitted expanded-training and reporting work. Existing edits were preserved. No commit, push, deployment, distributed-service startup, full-data retraining, threshold tuning or active-model replacement was performed. Tests train only small synthetic fixtures.

## Changes

- `scripts/run_expanded_benchmark.py` now refuses an existing nonempty output directory or file before input verification/preparation. Reruns require a fresh or empty `--output`; README instructions match this behavior.
- `cyberlab/reporting_release.py` joins both prediction partitions to prepared windows by `window_id` and rejects disagreeing binary labels. Preserving positive totals alone is no longer sufficient.
- The release builder recomputes every candidate's aggregate held-out metrics from saved scores at its frozen threshold. It checks the per-model results, duplicated all-model held-out results, and selected-model held-out summary. Counts must match exactly; floating metrics use `rtol=atol=1e-12`. This is consistency verification, not threshold selection or retraining. It does not independently reconstruct training/CV history or all optional per-capture/per-attack summaries.

## Verified

- Baseline: **268 passed**. Final full suite: **280 passed**.
- Overwrite regression tests failed before the CLI guard and passed afterward. Eight release regression cases reproduced label swaps and incorrect precision/confusion summaries before the fix and passed afterward.
- The real default expanded CLI now exits with argument error code 2 without overwriting its saved experiment. SHA256 comparisons confirmed unchanged files in `report_data/` and `artifacts/model_benchmark_expanded/`.
- Prepared-window checksum and capture split validated on **197,941 windows**.
- Active Extra Trees scored all **16,174** bundled feature rows, yielding **65 suspicious windows**. This is saved-data scoring, not a live-network detection test.
- The stricter builder successfully created a separate inactive staging directory at `artifacts/verification-2026-10-29/release/`. All **35,955** held-out decisions matched the saved candidate, with maximum score delta **1.1920928966180355e-07** against tolerance `1e-6`. Its preview contains **50,000 development-only rows**.
- `uv pip check --python .venv/Scripts/python.exe`: all **61** installed packages compatible.
- `git diff --check` passed; only Windows line-ending notices were emitted. Python compilation passed before the release changes; the full suite exercised the final code.

Machine-readable evidence and protected-file hashes: `artifacts/verification-2026-10-29/evidence.json`. Candidate release verification: `artifacts/verification-2026-10-29/release/release-verification.json`. These generated artifacts remain outside version control.

## Remaining boundaries

Streamlit's loopback health endpoint on port 8502 returned HTTP 200. Docker Desktop's Linux engine was unavailable, and Elasticsearch/Kibana endpoints refused connections. Those services were not restarted; distributed/live end-to-end behavior and browser rendering were not reverified in this pass.

The expanded candidate still detects only **61 of 778** held-out attack windows and misses **717**, with **39** false alerts. It remains inactive. Neither it nor the narrow Thursday-only active detector is established as production-ready. Future model improvement requires development-only diagnosis and fresh untouched evaluation captures—not retuning on this inspected holdout.
