# DataIQ feature adaptation and verification scope

Reference: the local `US_Housing_Kaggle_Advanced_Streamlit_Project`, inspected read-only. This is functional workflow adaptation for cyber/tabular analytics, not a byte-for-byte copy of housing-specific conclusions or its 9,349-line implementation.

Enable **Full analytics workspace** in the sidebar. The original six saved-benchmark pages remain available with that switch off.

## Shared workspace features

- Upload CSV with explicit memory/row/column limits; no silent dataset truncation.
- Project name and profile, categorized dataset templates, four audience modes.
- Target/date/feature selection and smart defaults that exclude obvious labels, IDs, stored scores and flags from model inputs.
- Page/column search, workflow-area filtering and focused/advanced navigation.
- Real, identifier-free feature-window snapshot for a clean checkout; original data is unchanged.
- Dataset/model fingerprinting so changing data or feature/target setup invalidates a stale current model.

## Reference-page mapping

| Reference feature | Cyber workspace route |
| --- | --- |
| Start Here | Start Here |
| Project Manager | Project Manager |
| Smart Auto-Setup | Smart Auto-Setup |
| Data Cleaning Studio | Data Cleaning Studio |
| Overview and correlation | Overview |
| Line/scatter/histogram/box/violin exploration | Explore |
| Missing/duplicate/numeric quality | Data Quality |
| Period/segment comparison | Compare |
| Supervised classification and regression | ML Lab |
| Model comparison and metrics | Evaluation |
| Feature importance/coefficients | Model Explainability |
| Clustering/PCA/anomaly analysis | Unsupervised Lab |
| Educational reinforcement learning | Reinforcement Lab |
| Lagged future-series scenarios | Forecast |
| Final evidence narrative | Conclusion |
| Domain context and sources | Domain Review |
| Code generation helper | Code Lab |
| Pivot/heatmap/3D segmentation and exports | OLAP & Export |
| One-page evidence story | Executive Summary |
| Column roles/types/missingness | Data Dictionary |
| Multi-feature what-if sensitivity | Scenario Simulator |
| Validation/drift/model-card/checklist | Production Readiness |
| Session experiment evidence | Experiment Tracker |
| Candidate promotion record | Model Registry |
| Dataset-to-serving lifecycle | Data Pipeline |
| Local-versus-distributed capability review | Big Data Readiness |
| Alert-cost/decision interpretation | Business Impact |
| Train/validation gaps and recommendations | Fit Diagnostics |
| Single/batch scoring | Prediction Page |
| Configurable evidence exports | Report Generator |
| Save and reload model artifacts | Model Save / Load |
| Data-aware assistance | Assistant |

## Intentional safety and scientific differences

- Generic model exploration is separate from the published cyber benchmark. An exploratory winner never overwrites the saved benchmark winner, thresholds or reporting artifacts.
- Model selection must use development CV, not held-out test F1; holdout scores are descriptive. Feature cleaning and fitted preprocessing require careful leakage review.
- Model import is restricted to validated supported bundles; arbitrary pickle/joblib files and blanket trust of user-declared Python types are not accepted.
- Code Lab produces previews/downloads and does not execute generated code. The assistant works locally without a hosted-model API key; no private data is sent to an external LLM automatically. Hosted LLM generation is not enabled by this adaptation.
- Reinforcement learning is an offline educational decision simulation, not an agent that attacks hosts or autonomously changes security controls.
- Forecasts and what-if responses are scenarios, not causal or guaranteed predictions. Housing trading/macroeconomic conclusions are replaced with cyber/data-specific interpretation.
- Interactive limits are deliberate for local/Cloud resource safety; processing all millions of raw flows belongs in the existing offline/distributed pipeline.
- Bundled feature data remains Thursday-only, with 68 positive windows from one host/episode. Downloaded full-week and DDoS extension data is not silently merged; timestamp alignment and independent validation remain prerequisites.

Verification must cover each route plus substantive actions (clean, train, predict, export/import, forecast/cluster/OLAP/RL) and preserve existing benchmark tests. A passing default page alone is not feature parity. The final verification artifacts describe the actual tested coverage; do not infer verified public-cloud deployment from local browser success.
