"""Artifact-backed reporting acceptance tests (no model deserialization)."""
from pathlib import Path
import importlib
import json
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "model_benchmark"


def reporting():
    assert (ROOT / "cyberlab" / "reporting.py").exists(), "artifact reporting helper is not implemented"
    return importlib.import_module("cyberlab.reporting")


@pytest.fixture
def expanded_scope_artifacts(tmp_path):
    """Controlled scope metadata with unchanged real saved prediction evidence."""
    for name in ("metrics.json", "development_oof_predictions.csv", "held_out_test_predictions.csv"):
        shutil.copyfile(ARTIFACTS / name, tmp_path / name)
    path = tmp_path / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["dataset"]["description"] = "Controlled full-week CICIDS2017 + three DDoS CSV scope fixture"
    metrics["dataset"]["input_rows"] = 4600123
    metrics["dataset"]["valid_flows"] = 4599987
    metrics["dataset"]["provenance"] = {"manifest": "controlled-sources.json", "sources": ["CICIDS2017 full week", "DDoS 03-11 Portmap"]}
    metrics["split"].update(
        method="Capture-disjoint outer holdout with three inner capture folds",
        temporal_policy="capture_disjoint_mixed_source_clocks_not_global_forward",
        inner_folds=metrics["split"]["inner_folds"][:3],
        test_captures=["CICIDS2017 Friday", "DDoS 03-11 Portmap"],
    )
    metrics["split"].pop("group_minutes", None)
    metrics["selection"]["primary_metric"] = "mean 3-fold capture-disjoint cross-validation average precision"
    metrics["caveats"] = ["Controlled expanded-scope fixture, not a new training result.", "Capture disjointness does not guarantee host disjointness."]
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return tmp_path


def test_report_uses_saved_scope_counts_policy_and_provenance(expanded_scope_artifacts):
    r = reporting()
    bundle = r.load_benchmark(expanded_scope_artifacts)
    md, html = r.build_report(bundle)
    for text in (md, html):
        assert bundle.metrics["dataset"]["description"] in text
        assert "4600123" in text
        assert "4599987" in text
        assert "capture_disjoint_mixed_source_clocks_not_global_forward" in text
        assert "controlled-sources.json" in text
        assert "CICIDS2017 Friday" in text
        for caveat in bundle.metrics["caveats"]:
            assert caveat in text
        for obsolete in ("Thursday", "one host/episode", "Shuffled grouped holdout", "5-minute groups"):
            assert obsolete not in text


def test_app_expanded_scope_all_sections_use_saved_evidence(monkeypatch, expanded_scope_artifacts):
    from streamlit.testing.v1 import AppTest
    import joblib
    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not deserialize estimator artifacts")
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setenv("CYBERLAB_BENCHMARK_DIR", str(expanded_scope_artifacts))
    metrics = json.loads((expanded_scope_artifacts / "metrics.json").read_text())
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.sidebar.segmented_control[0].options == ["Reports", "Run data", "Live data"]
    assert at.sidebar.toggle[0].label == "Full analytics workspace"
    for section in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(section).run()
        assert not at.exception, section
        assert not at.error
        text = "\n".join(str(element.value) for kind in ("markdown", "caption", "warning", "info", "json") for element in at.get(kind))
        assert metrics["dataset"]["description"] in text
        assert metrics["split"]["method"] in text
        assert metrics["split"]["temporal_policy"] in text
        for caveat in metrics["caveats"]:
            assert caveat in text
        for obsolete in ("Thursday", "one attack source", "one episode", "one host/episode", "Shuffled grouped holdout", "5-minute groups", "continuous episode"):
            assert obsolete not in text, (section, obsolete)
        if section == "Dataset & governance":
            assert at.metric[0].value == "4,600,123"
            assert at.metric[1].value == "4,599,987"
            assert at.metric[2].value == f"{metrics['dataset']['window_count']:,}"
            assert "controlled-sources.json" in text
            assert "CICIDS2017 Friday" in text
            assert "5-minute groups" not in at.dataframe[-2].value.columns


def test_legacy_thursday_scope_is_preserved_in_app_and_exports(monkeypatch, expanded_scope_artifacts):
    """Legacy scope stays explicit even after the actual artifacts are promoted."""
    from streamlit.testing.v1 import AppTest
    path = expanded_scope_artifacts / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["dataset"].update(
        description="Full available CICIDS2017 Thursday slice; no arbitrary row subsampling.",
        input_rows=170366, valid_flows=170217, window_count=16174, positive_windows=68,
    )
    metrics["dataset"].pop("provenance")
    metrics["split"].update(
        method="StratifiedGroupKFold on 5-minute UTC window-time buckets",
        temporal_policy="shuffled_group_holdout_not_forward_in_time", group_minutes=5,
    )
    metrics["split"].pop("test_captures")
    metrics["caveats"] = ["Only 68 positive source-IP/minute windows are available; all come from one source IP and a single continuous web-attack period."]
    path.write_text(json.dumps(metrics), encoding="utf-8")
    r = reporting()
    bundle = r.load_benchmark(expanded_scope_artifacts)
    for text in r.build_report(bundle):
        assert metrics["dataset"]["description"] in text
        assert metrics["split"]["temporal_policy"] in text
        assert metrics["caveats"][0] in text
        assert "170366" in text and "170217" in text
    monkeypatch.setenv("CYBERLAB_BENCHMARK_DIR", str(expanded_scope_artifacts))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    assert metrics["caveats"][0] in at.warning[0].value
    at.sidebar.radio[0].set_value("Dataset & governance").run()
    assert not at.exception
    assert at.metric[0].value == "170,366"
    assert at.metric[1].value == "170,217"
    assert at.metric[2].value == "16,174"


def test_load_real_artifacts_and_rank_by_cv_not_test():
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    metrics = bundle.metrics
    winner = metrics["selection"]["model"]
    table = r.comparison_table(bundle.metrics)
    assert len(table) == len(metrics["models"]) == 7
    assert table.iloc[0]["model"] == winner
    assert table.mean_cv_average_precision.is_monotonic_decreasing
    assert table.iloc[0]["mean_cv_average_precision"] == metrics["selection"]["selected_mean_cv_average_precision"]
    assert len(bundle.test) == metrics["split"]["test_windows"]
    assert len(bundle.development) == metrics["split"]["dev_windows"]
    assert bundle.test.malicious.sum() == metrics["split"]["test_positives"]


def test_diagnostics_recompute_curves_and_frozen_threshold_counts():
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    assert hasattr(r, "diagnostics"), "prediction diagnostics are not implemented"
    selected = bundle.metrics["selection"]["model"]
    model = bundle.metrics["models"][selected]
    result = r.diagnostics(bundle, selected, "Held-out test")
    cm = model["test_metrics"]["confusion_matrix"]
    assert result["confusion"] == [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    assert result["average_precision"] == pytest.approx(model["test_metrics"]["average_precision"])
    assert result["roc_auc"] == pytest.approx(model["test_metrics"]["roc_auc"])
    assert list(result["pr"].columns) == ["recall", "precision"]
    assert list(result["roc"].columns) == ["false_positive_rate", "true_positive_rate"]
    from sklearn.metrics import confusion_matrix
    dev = r.diagnostics(bundle, selected, "Development OOF")
    assert dev["threshold"] == model["threshold_from_dev_oof_max_f1"]
    assert dev["confusion"] == confusion_matrix(bundle.development.malicious, bundle.development[f"score_{selected}"] >= dev["threshold"], labels=[0, 1]).tolist()
    with pytest.raises(ValueError, match="Unknown split"):
        r.diagnostics(bundle, selected, "invalid")


def test_reports_preserve_caveats_allow_inclusion_and_escape_html():
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    assert hasattr(r, "build_report"), "downloadable reports are not implemented"
    md, html = r.build_report(bundle, include_comparison=True, include_learning=True, title="<script>alert(1)</script>")
    for text in (md, html):
        assert bundle.metrics["selection"]["model"] in text
        assert str(bundle.metrics["dataset"]["positive_windows"]) in text
        assert bundle.metrics["split"]["temporal_policy"] in text
        for caveat in bundle.metrics["caveats"]:
            from html import escape
            assert (escape(caveat) if text == html else caveat) in text
        assert "Learning curve" in text
        assert "Model comparison" in text
    assert "<script>" not in html
    assert html.count("<table") == 2
    assert "| Model | CV AP | Fold SD | Test AP |\n|---|---:|---:|---:|" in md
    md_small, _ = r.build_report(bundle, include_comparison=False, include_learning=False)
    assert "Model comparison" not in md_small
    assert "Learning curve" not in md_small
    assert "## Governance" in md_small


def test_app_all_sections_selectors_and_downloads(monkeypatch):
    from streamlit.testing.v1 import AppTest
    assert (ROOT / "streamlit_app.py").exists(), "Streamlit reporting app is not implemented"
    monkeypatch.setenv("CYBERLAB_BENCHMARK_DIR", str(ARTIFACTS))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    sections = list(at.sidebar.radio[0].options)
    assert len(sections) == 6
    assert len(at.metric) >= 4
    for section in sections:
        at.sidebar.radio[0].set_value(section).run()
        assert not at.exception, f"{section}: {at.exception}"
        assert not at.error
        if section == "Model comparison":
            assert len(at.selectbox) >= 1
            at.selectbox[0].select("test_f1").run()
            assert not at.exception
            assert len(at.get("plotly_chart")) >= 1
        elif section == "Model diagnostics":
            assert len(at.selectbox) == 2
            assert len(at.get("plotly_chart")) >= 3
            at.selectbox[0].select("logistic_regression").run()
            at.selectbox[1].select("Development OOF").run()
            assert not at.exception
        elif section == "Learning curve":
            assert len(at.get("plotly_chart")) >= 1
        elif section == "Report generator":
            assert len(at.checkbox) >= 2
            assert len(at.get("download_button")) >= 5
            at.checkbox[0].uncheck().run()
            assert not at.exception
            assert "## Model comparison" not in at.markdown[-1].value


def test_app_missing_artifacts_is_friendly(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CYBERLAB_BENCHMARK_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    assert len(at.error) == 1
    assert "metrics.json" in at.error[0].value
    assert "No synthetic fallback" in at.info[0].value
    assert not at.get("plotly_chart")


def test_app_default_artifacts_are_independent_of_cwd(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("CYBERLAB_BENCHMARK_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    assert not at.error
    r = reporting()
    assert at.metric[0].value == r.load_benchmark(r.resolve_artifact_directory(ROOT)).metrics["selection"]["model"]


def test_reporting_never_loads_estimators(monkeypatch):
    import joblib
    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not deserialize estimator artifacts")
    monkeypatch.setattr(joblib, "load", forbidden)
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    r.diagnostics(bundle, bundle.metrics["selection"]["model"], "Held-out test")
    r.build_report(bundle)


def test_missing_artifacts():
    r = reporting()
    with pytest.raises(r.ArtifactError, match="metrics.json"):
        r.load_benchmark(ARTIFACTS / "missing")
