import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from cyberlab import flow_store
from api.main import app

client = TestClient(app)


def test_summary_uses_expanded_full_benchmark_not_thursday_only():
    response = client.get('/api/summary')
    assert response.status_code == 200
    body = response.json()
    assert body['raw_input_rows'] == 4_610_049
    assert body['feature_windows'] == 197_941
    assert 'Complete CICIDS2017 full-week' in body['benchmark_scope']
    assert body['data_mode'] == 'Historical benchmark'


def test_ml_api_preserves_saved_candidate_and_evaluation_splits():
    response = client.get('/api/ml')
    assert response.status_code == 200
    body = response.json()
    assert body['selection']['model'] in body['models']
    assert body['split']['test_windows'] > 0
    assert body['caveats']
    assert 'pytorch_mlp' in body['models']


def test_health_endpoint_reports_source_build_id():
    from api import main as api
    api_hash = hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()
    store_hash = hashlib.sha256(Path(flow_store.__file__).read_bytes()).hexdigest()
    expected = hashlib.sha256(f"{api_hash}:{store_hash}".encode()).hexdigest()[:12]
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['build_id'] == expected


def test_health_endpoint_reports_api_but_not_fake_live_data():
    response = client.get('/api/health/components')
    assert response.status_code == 200
    body = response.json()
    assert any(row['name'] == 'Elasticsearch' for row in body['components'])
    assert not any(row['name'] == 'Live detection feed' for row in body['components'])


def test_health_probe_includes_streamlit_workspace(monkeypatch):
    from api import main as api

    calls = []
    def fake_probe(name, url):
        calls.append((name, url))
        return {'name': name, 'status': 'unavailable', 'detail': 'test probe'}

    monkeypatch.setattr(api, 'probe_http', fake_probe)
    monkeypatch.setattr(api, 'probe_tcp', lambda name, host, port: {'name': name, 'status': 'unavailable'})
    result = api.components_health()
    streamlit = next(row for row in result['components'] if row['name'] == 'Streamlit report workspace')
    assert streamlit['status'] == 'unavailable'
    assert ('Streamlit report workspace', 'http://127.0.0.1:8502') in calls


def test_flow_pagination_clearly_marks_legacy_historical_scope():
    response = client.get('/api/traffic', params={'page': 1, 'page_size': 2})
    assert response.status_code == 200
    body = response.json()
    assert len(body['items']) == 2
    assert body['total'] > 2
    assert 'Thursday capture only' in body['source']


def test_missing_finding_returns_404_not_synthetic_evidence():
    response = client.get('/api/findings/not-a-real-window')
    assert response.status_code == 404
