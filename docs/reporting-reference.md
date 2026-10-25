# Reporting approach

The cyber benchmark reporting interface follows the local DataIQ / U.S. Housing Streamlit project's reporting patterns. That reference project was inspected read-only; its source and data were not changed or bundled here.

## Patterns adapted

| Reference pattern | Cyber benchmark adaptation |
| --- | --- |
| Wide Streamlit layout and workflow navigation | Focused sidebar sections for benchmark reporting |
| Executive Summary | Dataset coverage, CV-selected model, held-out evidence, limitations |
| Plotly charts with hover, legend toggles and zoom | Interactive candidate comparisons, PR/ROC and learning-size diagnostics |
| Interpretation beside diagnostics | Plain-language explanation of false alerts, missed windows, precision/recall and fit gaps |
| Report Generator options | Configurable evidence sections and downloadable Markdown/HTML/CSV |
| Evaluation table and data dictionary | Artifact-backed classification metrics and feature/split descriptions |
| `streamlit run streamlit_app.py` and `runOnSave` | Same local serving approach, bound to 127.0.0.1:8502 |

## Deliberate differences

- The default benchmark surface reports existing, reproducible artifacts. The optional full analytics workspace fits bounded models only on explicit user actions; its candidates do not replace the published benchmark.
- Candidate selection remains based on mean grouped development cross-validation AP. The housing app's generic test-score sorting is not reused as a model-selection policy.
- Accuracy and housing regression metrics are not promoted as cyber model-selection measures.
- Uploaded pickle/joblib and arbitrary code execution are not exposed. Workspace model import is format- and trust-validated; explicit training is resource-bounded and session-only.
- Rare-attack and split caveats remain visible. The 68 positive windows represent one source host and one continuous attack period; the group assignment is shuffled, not a forward-in-time test.
- Kibana and the existing HTML report remain available. The Streamlit interface is an additional reporting surface, not an Elasticsearch replacement.
- The reference documents local Streamlit serving, not a verified public/cloud deployment. This project's matching deployment is also local-only.
