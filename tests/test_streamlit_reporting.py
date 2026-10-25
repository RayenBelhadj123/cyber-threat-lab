"""Artifact-backed reporting acceptance tests (no model deserialization)."""
from pathlib import Path
import importlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "model_benchmark"


def reporting():
    assert (ROOT / "cyberlab" / "reporting.py").exists(), "artifact reporting helper is not implemented"
    return importlib.import_module("cyberlab.reporting")


def test_load_real_artifacts_and_rank_by_cv_not_test():
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    assert bundle.metrics["selection"]["model"] == "extra_trees"
    table = r.comparison_table(bundle.metrics)
    assert len(table) == 7
    assert table.iloc[0]["model"] == "extra_trees"
    assert table.iloc[0]["mean_cv_average_precision"] == pytest.approx(0.9710381054)
    assert len(bundle.test) == 3229
    assert len(bundle.development) == 12945
    assert bundle.test.malicious.sum() == 12


def test_diagnostics_recompute_curves_and_frozen_threshold_counts():
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    assert hasattr(r, "diagnostics"), "prediction diagnostics are not implemented"
    result = r.diagnostics(bundle, "extra_trees", "Held-out test")
    assert result["confusion"] == [[3215, 2], [1, 11]]
    assert result["average_precision"] == pytest.approx(0.9722222222)
    assert result["roc_auc"] == pytest.approx(0.99984457569)
    assert list(result["pr"].columns) == ["recall", "precision"]
    assert list(result["roc"].columns) == ["false_positive_rate", "true_positive_rate"]
    dev = r.diagnostics(bundle, "extra_trees", "Development OOF")
    assert dev["confusion"] == [[12889, 0], [8, 48]]
    with pytest.raises(ValueError, match="Unknown split"):
        r.diagnostics(bundle, "extra_trees", "invalid")


def test_reports_preserve_caveats_allow_inclusion_and_escape_html():
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    assert hasattr(r, "build_report"), "downloadable reports are not implemented"
    md, html = r.build_report(bundle, include_comparison=True, include_learning=True, title="<script>alert(1)</script>")
    for text in (md, html):
        assert "extra_trees" in text
        assert "68" in text
        assert "not forward" in text
        assert "not a production-validated detector" in text
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
    assert at.metric[0].value == "extra_trees"


def test_reporting_never_loads_estimators(monkeypatch):
    import joblib
    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not deserialize estimator artifacts")
    monkeypatch.setattr(joblib, "load", forbidden)
    r = reporting()
    bundle = r.load_benchmark(ARTIFACTS)
    r.diagnostics(bundle, "extra_trees", "Held-out test")
    r.build_report(bundle)


def test_missing_artifacts():
    r = reporting()
    with pytest.raises(r.ArtifactError, match="metrics.json"):
        r.load_benchmark(ARTIFACTS / "missing")
