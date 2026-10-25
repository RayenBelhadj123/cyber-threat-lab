from pathlib import Path
import json
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_can_load_real_snapshot_without_local_artifacts(monkeypatch):
    from streamlit.testing.v1 import AppTest
    snapshot = ROOT / 'report_data'
    assert (snapshot / 'metrics.json').is_file(), 'Cloud reporting snapshot is missing'
    monkeypatch.setenv('CYBERLAB_BENCHMARK_DIR', str(snapshot))
    at = AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=30).run()
    assert not at.exception
    assert not at.error
    assert at.metric[0].value == json.loads((snapshot / 'metrics.json').read_text())['selection']['model']
    for section in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(section).run()
        assert not at.exception
        assert not at.error


def test_cloud_snapshot_has_no_host_identity_or_model_bundles():
    from cyberlab.reporting import load_benchmark, diagnostics
    snapshot = ROOT / 'report_data'
    assert (snapshot / 'metrics.json').exists(), 'Cloud reporting snapshot is missing'
    bundle = load_benchmark(snapshot)
    assert 'positive_source_ips' not in bundle.metrics['dataset']
    for frame in (bundle.development, bundle.test):
        assert {'source_ip', 'window_id', 'reference_label'}.isdisjoint(frame.columns)
    assert not list(snapshot.glob('*.joblib'))
    selected = bundle.metrics['selection']['model']
    cm = bundle.metrics['models'][selected]['test_metrics']['confusion_matrix']
    assert diagnostics(bundle, selected, 'Held-out test')['confusion'] == [[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]]


def test_default_directory_falls_back_to_bundled_snapshot(tmp_path):
    from cyberlab import reporting
    assert hasattr(reporting,'resolve_artifact_directory'), 'Default cloud data resolution missing'
    assert reporting.resolve_artifact_directory(tmp_path) == tmp_path / 'report_data'
    (tmp_path / 'artifacts' / 'model_benchmark').mkdir(parents=True)
    (tmp_path / 'artifacts' / 'model_benchmark' / 'metrics.json').write_text('{}')
    assert reporting.resolve_artifact_directory(tmp_path) == tmp_path / 'artifacts' / 'model_benchmark'


def test_cloud_dependencies_and_config_do_not_require_local_bindings():
    requirements = (ROOT / 'requirements.txt').read_text()
    assert 'streamlit==' in requirements
    assert 'plotly==' in requirements
    config = tomllib.loads((ROOT / '.streamlit' / 'config.toml').read_text())
    assert 'address' not in config['server']
    assert 'port' not in config['server']
