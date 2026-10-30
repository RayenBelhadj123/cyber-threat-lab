# Local maintenance and expanded-benchmark review

Date: 2026-10-28. Scope: local working tree, existing benchmark artifacts, safe staging, and the Streamlit app. Existing uncommitted expanded-training work was preserved. No commit, push, public deployment, raw-data deletion, model reselection, threshold tuning, or active-detector replacement was performed.

## Fixes verified with regression tests

The snapshot builder previously accepted an existing staging directory and could overwrite a verified release. It now requires a fresh or empty directory.

The builder also accepted corrupted development window identities, potentially including held-out samples in the supposedly development-only interactive preview. It now checks:

- Non-null, unique identities in the prepared windows and both prediction tables.
- Disjoint development and held-out window identities.
- Exact coverage of all prepared windows by the two prediction tables.
- Disjoint actual capture groups, looked up from prepared windows rather than trusting prediction metadata.

Tests reproduced overwrite, held-out overlap, unknown identity, duplicate identity and cross-capture identity-swap failures before the fixes. Additional tests confirm rejection of non-finite saved scores by the existing report validator.

Independent review found two more fail-open paths, also reproduced and corrected: cached preparation now compares each manifest's expected row count with cached file lineage, and benchmark fitting now validates split index bounds/uniqueness, window-group alignment, development/test isolation, CV training/validation isolation, and exactly-once development OOF coverage before fitting any model. The real 197,941-window preparation passes these checks; the findings do not imply leakage in the saved experiment. Evidence: `artifacts/maintenance-verification/evidence-check.json`.

## Real expanded evidence — not promoted

The existing expanded benchmark processed **4,610,049 data rows** from **11 CSVs**, retaining **4,553,532 valid unique flows** and producing **197,941 feature windows**. Separately audited: 288,602 blank placeholder records, 56,231 invalid data rows, and 286 duplicate data rows. All eleven inputs passed size and SHA256 checks against the pinned-mirror manifests during this maintenance pass; the prepared-window checksum also matched. This verifies mirror integrity, not independent publisher authenticity.

Development CV selected **PyTorch MLP**. On **35,955 held-out windows**, the frozen threshold detected **61 of 778 attack windows**, missed **717**, and generated **39 false alerts**. All **648 Bot-labeled held-out windows were missed**. These results do not justify replacing the active detector or claiming broad attack detection. The active Thursday-only Extra Trees model is also limited; retaining it is not an endorsement of production readiness.

A separate privacy-reduced candidate was built at `artifacts/expanded_release_review/`. Its NumPy-only packaged inference matched all **35,955 held-out decisions**, with maximum absolute score difference **1.1920928966180355e-07** (allowed tolerance `1e-6`). The interactive preview contains **50,000 development-only windows**; this preview cap does not cap training data. Serialization parity is not an accuracy or deployment-readiness certification. The candidate remains inactive.

Evidence: `artifacts/expanded_release_review/release-verification.json`.

## Verification

- Baseline: 252 tests passed.
- After release and training validation hardening: 266 tests passed; `git diff --check` passed (Windows line-ending notices only).
- `uv pip check --python .venv/Scripts/python.exe`: all 61 installed packages compatible. This uv-created environment does not contain pip, so `python -m pip check` is not the correct local command.
- Actual browser: all six saved-report routes rendered without JavaScript page errors.
- Actual browser upload/download: all 16,174 bundled feature rows scored with the unchanged Extra Trees detector; 65 suspicious windows; no source IPs fabricated.
- Live mode's paused page rendered. No Elasticsearch fetch, replay, packet capture, or live attack detection was verified.
- Local Streamlit started on loopback port 8502 and returned HTTP 200.

Browser evidence: `artifacts/maintenance-verification/browser-smoke.json`, `browser-scored-results.csv`, and `run-data.png`.

## Operating boundaries and next work

Docker Desktop's engine was unavailable at inspection; Elasticsearch, Kibana and the standalone Docker HTML report were stopped. They were not restarted, and no distributed sink parity or service-health claim is made. The local Streamlit app works without those services for saved reports and uploaded-data scoring.

Do not select a replacement model or tune its threshold using the now-inspected held-out captures. Treat these captures as diagnostic evidence for subsequent development; establish new untouched capture/host/domain evaluation data before making a fresh generalization claim. Diagnose missed Bot windows and feature/aggregation limitations in development data first. Preserve the documented mixed-source clock ambiguity and capture-versus-host isolation limits.

The older `project-audit.md` describes its own historical run, not today's service state. Raw datasets and generated candidate artifacts remain outside version control.
