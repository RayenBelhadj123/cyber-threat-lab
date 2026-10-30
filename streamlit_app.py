"""Local, read-only Cyber Threat Lab benchmark explorer.

Run locally: .venv/Scripts/python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502
Reporting mode reads only JSON/CSV evidence. Explicit Run/Live modes load the
packaged hash-checked inference detector; workspace experiments remain separate.
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cyberlab.reporting import ArtifactError, build_report, comparison_table, diagnostics, governance_lines, load_benchmark, resolve_artifact_directory

st.set_page_config(page_title="Cyber Threat Lab | Evidence workspace", page_icon="🛡️", layout="wide")
from cyberlab.ui_theme import apply_theme, style_chart
apply_theme()

SECTIONS = ("Executive summary", "Model comparison", "Model diagnostics", "Learning curve", "Dataset & governance", "Report generator", "SOC dashboard")
st.sidebar.markdown("## 🛡️ Cyber Threat Lab")
st.sidebar.caption("Data · Detection · Evidence")
data_mode = st.sidebar.segmented_control("Data mode", ["Reports", "Run data", "Live data"], default="Reports", key="cyber_detection_mode", width="stretch", wrap=False)
if data_mode in ("Run data", "Live data"):
    from cyberlab.live_detection import render_detection
    render_detection(Path(__file__).resolve().parent, data_mode)
    st.stop()
if st.sidebar.toggle("Full analytics workspace", value=False, key="full_analytics_workspace"):
    from cyberlab.workspace import render_workspace
    render_workspace(Path(__file__).resolve().parent)
    st.stop()
section = st.sidebar.radio("Workflow", SECTIONS)
if section == "SOC dashboard":
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen

    soc_url = os.environ.get("CYBERLAB_SOC_FRONTEND_URL", "http://127.0.0.1:5173").rstrip("/")
    api_base = os.environ.get("CYBERLAB_API_BASE_URL", "http://127.0.0.1:8000/api").rstrip("/")

    def probe(url):
        try:
            with urlopen(url, timeout=0.75) as response:
                return f"available (HTTP {response.status})"
        except HTTPError as exc:
            return f"unavailable (HTTP {exc.code})"
        except (URLError, TimeoutError, OSError) as exc:
            return f"unavailable ({type(exc).__name__})"

    frontend_state = probe(soc_url)
    api_state = probe(f"{api_base}/health")
    index_state = probe(f"{api_base}/flows/status")

    st.title("Embedded SOC dashboard")
    st.caption("React SOC interface embedded here; it uses the local FastAPI service for full-flow investigations.")
    st.caption(f"Embedded frontend: {soc_url}")
    st.subheader("Connection status")
    st.write(f"Frontend: {frontend_state}")
    st.write(f"API: {api_state}")
    st.write(f"Full-flow index: {index_state}")
    for label, value in (("Frontend", frontend_state), ("API", api_state), ("Full-flow index", index_state)):
        if value.startswith("unavailable"):
            st.warning(f"{label} unavailable — check that its service is running and the configured URL is correct.")
    st.link_button("Open SOC dashboard in a new tab", soc_url)
    st.iframe(soc_url, height=920)
    st.info("If the embedded view is blank, start the FastAPI service on 127.0.0.1:8000 and the React frontend on 127.0.0.1:5173. The expanded flow database is local and is not downloaded from GitHub. Set CYBERLAB_SOC_FRONTEND_URL or CYBERLAB_API_BASE_URL if you use different addresses.")
    st.stop()
st.sidebar.divider()
st.sidebar.caption("Read-only reporting • no retraining\n\nLocal experiment, not production validation")
artifact_dir = Path(os.environ.get("CYBERLAB_BENCHMARK_DIR", str(resolve_artifact_directory(Path(__file__).resolve().parent))))
try:
    bundle = load_benchmark(artifact_dir)
except ArtifactError as exc:
    st.title("Benchmark artifacts unavailable")
    st.error(str(exc))
    st.info("Provide metrics.json, development_oof_predictions.csv and held_out_test_predictions.csv in artifacts/model_benchmark or report_data, or set CYBERLAB_BENCHMARK_DIR. No synthetic fallback is used.")
    st.stop()

m = bundle.metrics
selected = m["selection"]["model"]
selected_model = m["models"][selected]
test = selected_model["test_metrics"]
table = comparison_table(m)
st.caption("CYBERSECURITY ANALYTICS / SAVED BENCHMARK")
st.title(section)
st.caption(m["dataset"]["description"])
with st.expander("Dataset scope & validation limits", expanded=False):
    st.caption(f"Split: {m['split']['method']} • Temporal policy: {m['split']['temporal_policy']}")
    st.warning("Exploratory evidence only; saved benchmark limitations:\n\n" + "\n\n".join(m["caveats"]))


def show_chart(fig):
    style_chart(fig)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


if section == "Executive summary":
    st.markdown("### Benchmark overview")
    a, b, c, d = st.columns(4)
    a.metric("Selected by grouped CV", selected)
    b.metric("Mean CV average precision", f"{selected_model['mean_cv_average_precision']:.3f}")
    c.metric("Held-out test AP", f"{test['average_precision']:.3f}")
    d.metric("Positive test windows", f"{test['positives']} / {test['windows']:,}")
    left, right = st.columns([1.35, 1])
    with left:
        cm = test["confusion_matrix"]
        outcomes = pd.DataFrame({"Outcome": ["Detected attacks", "Missed attacks", "False alerts"], "Windows": [cm["tp"], cm["fn"], cm["fp"]]})
        show_chart(px.bar(outcomes, x="Outcome", y="Windows", text="Windows", title="Held-out alert outcomes · frozen threshold", color="Outcome", color_discrete_sequence=["#0f766e", "#b45309", "#2563eb"]))
    with right:
        st.subheader("What the evidence says")
        st.write(f"{selected} was selected using {m['selection']['primary_metric']}. Test results did not choose the winner.")
        st.write(f"At the frozen OOF threshold ({test['threshold']:.4f}), the candidate detects {cm['tp']} of {test['positives']} attack windows, misses {cm['fn']}, and raises {cm['fp']} false alerts. Held-out precision is {test['precision']:.1%}; recall is {test['recall']:.1%}.")
        st.info("Next decision: collect independent campaigns, hosts and networks; run forward temporal validation and evaluate operational alert costs. These results do not establish deployment readiness.")
    st.caption(f"Provenance: {artifact_dir} • metrics.json and saved prediction CSVs only • existing HTML/Kibana reports remain unchanged")

elif section == "Model comparison":
    st.write(f"{len(m['models'])} saved candidates. Ordering stays fixed by mean CV average precision even when a test metric is displayed.")
    metric = st.selectbox("Comparison metric", ["mean_cv_average_precision", "mean_train_average_precision", "train_cv_ap_gap", "test_average_precision", "test_precision", "test_recall", "test_f1", "test_roc_auc"])
    fig = px.bar(table, x="model", y=metric, title=f"Candidate comparison · {metric}", hover_data=["mean_cv_average_precision", "std_cv_average_precision", "test_f1"])
    fig.update_xaxes(categoryorder="array", categoryarray=table.model.tolist(), tickangle=-25)
    if metric == "mean_cv_average_precision":
        fig.update_traces(error_y=dict(type="data", array=table.std_cv_average_precision))
    show_chart(fig)
    st.info("Average precision emphasizes rare attacks. CV error bars are fold standard deviations, not confidence intervals. A high ROC AUC can coexist with poor alert precision; test F1 is descriptive and must not drive model reselection.")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download model comparison CSV", table.to_csv(index=False), "cyberlab_model_comparison.csv", "text/csv")

elif section == "Model diagnostics":
    a, b = st.columns(2)
    model = a.selectbox("Diagnostic model", list(m["models"]), index=list(m["models"]).index(selected))
    split = b.selectbox("Prediction split", ["Held-out test", "Development OOF"])
    result = diagnostics(bundle, model, split)
    st.caption(f"Artifact-selected winner remains {selected}. Inspecting another model does not reselect it. Threshold frozen from development OOF: {result['threshold']:.9f}.")
    left, right = st.columns(2)
    with left:
        fig = px.line(result["pr"], x="recall", y="precision", title=f"Precision–recall · AP {result['average_precision']:.4f}")
        fig.add_hline(y=result["prevalence"], line_dash="dash", annotation_text="Prevalence baseline")
        fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1.05])
        show_chart(fig)
        st.caption("Read precision at the recall level your analysts need. The dashed baseline is attack prevalence; AP summarizes the ranking, not performance at the frozen operating threshold.")
    with right:
        fig = px.line(result["roc"], x="false_positive_rate", y="true_positive_rate", title=f"ROC · AUC {result['roc_auc']:.4f}")
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="#8b9cb5"))
        fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1.05])
        show_chart(fig)
        st.caption("ROC compares sensitivity to false-positive rate. With overwhelmingly benign traffic, a tiny false-positive rate may still create many alerts. Use the confusion counts alongside ROC.")
    left, right = st.columns([1, 1])
    with left:
        show_chart(px.imshow(result["confusion"], x=["Predicted benign", "Predicted attack"], y=["Actual benign", "Actual attack"], text_auto=True, color_continuous_scale="Teal", title="Confusion matrix · frozen OOF threshold", aspect="auto"))
    with right:
        (tn, fp), (fn, tp) = result["confusion"]
        st.subheader("Operational reading")
        st.write(f"True negatives: {tn:,} · false alerts: {fp:,} · missed attacks: {fn:,} · detected attacks: {tp:,}.")
        st.write("No test threshold optimization is available. Development OOF thresholds were chosen by maximum F1 and are frozen; OOF operating-point metrics may be optimistic because those same OOF labels selected the threshold.")
        frame = bundle.test if split == "Held-out test" else bundle.development
        st.download_button("Download saved prediction CSV", frame.to_csv(index=False), "cyberlab_saved_predictions.csv", "text/csv")

elif section == "Learning curve":
    curve = pd.DataFrame(m["learning_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.mean_train_windows, y=curve.mean_train_average_precision, mode="lines+markers", name="Training AP"))
    fig.add_trace(go.Scatter(x=curve.mean_train_windows, y=curve.mean_validation_average_precision, mode="lines+markers", name="Grouped validation AP", error_y=dict(type="data", array=curve.std_validation_average_precision), customdata=curve[["fraction_of_training_groups", "mean_train_positive_windows"]], hovertemplate="Windows: %{x}<br>Validation AP: %{y:.4f}<br>Group fraction: %{customdata[0]}<br>Train positives: %{customdata[1]}<extra></extra>"))
    fig.update_layout(title=f"{selected} · saved grouped training-size experiment", xaxis_title="Mean training windows", yaxis_title="Average precision")
    fig.update_yaxes(range=[0, 1.05])
    show_chart(fig)
    last = curve.iloc[-1]
    st.info(f"At the largest saved size, train AP minus validation AP is {last.mean_train_average_precision - last.mean_validation_average_precision:.4f}. Validation SD is {last.std_validation_average_precision:.4f}. This does not prove absence of overfitting; interpret it within the recorded dataset, split and caveats. Error bars show fold SD, not confidence intervals.")
    st.caption("Learning-curve validation AP differs from the headline model-selection CV estimate because it is a separate saved training-size experiment. No models are fitted here.")
    st.dataframe(curve, hide_index=True, width="stretch")

elif section == "Dataset & governance":
    dataset, split = m["dataset"], m["split"]
    a, b, c, d = st.columns(4)
    a.metric("Input flow rows", f"{dataset['input_rows']:,}")
    b.metric("Valid flows", f"{dataset['valid_flows']:,}")
    c.metric("Source-IP/minute windows", f"{dataset['window_count']:,}")
    d.metric("Attack-window prevalence", f"{dataset['positive_windows'] / dataset['window_count']:.2%}")
    left, right = st.columns(2)
    with left:
        labels = pd.DataFrame(dataset["positive_window_labels"].items(), columns=["Attack type", "Positive windows"])
        show_chart(px.bar(labels, x="Positive windows", y="Attack type", orientation="h", title="Attack composition · positive windows"))
        st.caption("These are positive source-IP/minute windows, not independent attacks or flow counts. Coverage limitations are recorded below.")
    with right:
        split_table = pd.DataFrame({"Partition": ["Development", "Held-out test"], "Windows": [split["dev_windows"], split["test_windows"]], "Positives": [split["dev_positives"], split["test_positives"]], "Recorded groups": [split["dev_groups"], split["test_groups"]]})
        st.subheader("Split contract")
        st.dataframe(split_table, hide_index=True, width="stretch")
        st.write(split["method"])
        st.write(f"Recorded test/development group overlap: {split['group_overlap_test_development']}. Temporal policy: {split['temporal_policy']}.")
        st.caption(f"Recorded inner folds: {len(split['inner_folds'])}. Group meaning and capture boundaries follow the saved split contract, not an assumed time bucket.")
    st.subheader("Guardrails & limitations")
    for caveat in m["caveats"]:
        st.write(f"• {caveat}")
    st.subheader("Recorded dataset, split & provenance")
    for line in governance_lines(m):
        if not line.startswith("- "):
            st.write(line)
    with st.expander("Recorded inner folds and runtime"):
        st.dataframe(pd.DataFrame(split["inner_folds"]), hide_index=True)
        st.json(m["runtime"])

elif section == "Report generator":
    st.write("Build a portable evidence pack from the saved benchmark. Executive summary and governance caveats are always included.")
    left, right = st.columns(2)
    include_comparison = left.checkbox("Include model comparison", value=True)
    include_learning = right.checkbox("Include learning curve table", value=True)
    markdown, html = build_report(bundle, include_comparison=include_comparison, include_learning=include_learning)
    a, b, c = st.columns(3)
    a.download_button("Download Markdown report", markdown, "cyberlab_report.md", "text/markdown")
    b.download_button("Download HTML report", html, "cyberlab_report.html", "text/html")
    c.download_button("Download summary CSV", table.to_csv(index=False), "cyberlab_summary.csv", "text/csv")
    a, b = st.columns(2)
    a.download_button("Download development OOF CSV", bundle.development.to_csv(index=False), "development_oof_predictions.csv", "text/csv")
    b.download_button("Download held-out test CSV", bundle.test.to_csv(index=False), "held_out_test_predictions.csv", "text/csv")
    st.caption("Prediction exports preserve exactly the saved artifact columns. They may contain source-IP identifiers; review sharing permissions before distributing. No uploads, estimator deserialization, or retraining.")
    st.divider()
    st.markdown(markdown)
