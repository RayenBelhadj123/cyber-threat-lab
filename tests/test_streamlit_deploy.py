from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_deployment_uses_reference_run_mode_but_local_only():
    config_path = ROOT / '.streamlit' / 'config.toml'
    assert config_path.exists(), 'Streamlit deployment config is missing'
    config = tomllib.loads(config_path.read_text(encoding='utf-8'))
    assert 'address' not in config['server']
    assert 'port' not in config['server']
    assert '--server.address 127.0.0.1 --server.port 8502' in (ROOT / 'README.md').read_text(encoding='utf-8')
    assert config['server']['runOnSave'] is True
    assert config['server']['headless'] is True
    assert config['browser']['gatherUsageStats'] is False
    assert config['theme']['base'] == 'light'
    requirements = (ROOT / 'requirements-report.txt').read_text(encoding='utf-8')
    assert 'streamlit==' in requirements
    assert 'plotly==' in requirements
