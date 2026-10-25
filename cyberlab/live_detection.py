"""Bounded inference from uploaded flows/windows or the existing Elasticsearch stream.

Never sniffs interfaces, writes detections back, changes the frozen benchmark, or
attributes a real attacker. Packaged estimator loading is hash/version checked;
user model uploads are not accepted on this path.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

INDEX = 'cyber-features-spark-stream-v1'
MAX_ROWS = 50_000


def load_detector(root):
    import skops.io as sio
    import sklearn
    from sklearn.ensemble import ExtraTreesClassifier
    base = Path(root) / 'report_data'
    metadata = json.loads((base / 'detector_metadata.json').read_text(encoding='utf-8'))
    payload = (base / 'detector.skops').read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata['artifact_sha256']:
        raise ValueError('Packaged detector checksum mismatch; restore the trusted release.')
    if sklearn.__version__ != metadata['sklearn_version']:
        raise ValueError(f"Detector requires scikit-learn {metadata['sklearn_version']}; installed {sklearn.__version__}.")
    # Static, narrowly vetted sklearn internal type, never the uploaded file's trust list.
    allowed = {'sklearn.tree._tree.Tree'}
    if set(sio.get_untrusted_types(data=payload)) - allowed:
        raise ValueError('Packaged detector contains unsupported Python types.')
    model = sio.loads(payload, trusted=list(allowed))
    if not isinstance(model, ExtraTreesClassifier) or list(model.classes_) != [False, True]:
        raise ValueError('Unsupported detector family or positive class.')
    from cyberlab.pipeline import FEATURES
    if metadata['features'] != FEATURES or not 0 <= metadata['threshold'] <= 1:
        raise ValueError('Invalid packaged detector feature/threshold contract.')
    return model, metadata


def score_windows(frame, model, metadata):
    from cyberlab.pipeline import valid_ip
    names = metadata['features']
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError('Missing model features: ' + ', '.join(missing))
    if not 1 <= len(frame) <= MAX_ROWS or len(frame.columns) > 200:
        raise ValueError('Detection accepts 1–50,000 rows and at most 200 columns.')
    features = frame[names].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(features.to_numpy(dtype=float)).all() or (features < 0).any().any():
        raise ValueError('Model features must be finite nonnegative numbers; invalid values are not silently imputed.')
    if (features.failure_ratio > 1).any():
        raise ValueError('failure_ratio must lie between 0 and 1.')
    out = pd.DataFrame(index=frame.index)
    source = frame['source_ip'] if 'source_ip' in frame else pd.Series(None, index=frame.index, dtype='object')
    out['source_ip'] = source.map(valid_ip)
    for column in ('timestamp', '@timestamp', 'event_id', 'window_id'):
        if column in frame:
            out[column] = frame[column]
    X = features if hasattr(model, 'feature_names_in_') else features.to_numpy(dtype=float)
    out['detector_score'] = model.predict_proba(X)[:, 1]
    out['suspicious'] = out.detector_score >= metadata['threshold']
    out['detector'] = metadata['model_name']
    return out.reset_index(drop=True)


def prepare_input(payload, schema='Feature windows'):
    from cyberlab.workspace import read_workspace_csv
    frame = read_workspace_csv(payload)
    if schema == 'Feature windows':
        return frame, {'input_rows': len(frame), 'mode': 'already aggregated host/minute windows'}
    if schema != 'CICIDS flow CSV':
        raise ValueError('Unsupported input schema.')
    import tempfile
    from cyberlab.pipeline import load_flows, aggregate_windows
    # Reuse the established ingestion contract instead of a second timestamp/feature implementation.
    with tempfile.TemporaryDirectory(prefix='cyberlab-detect-', dir=os.environ.get('TMPDIR')) as directory:
        path = Path(directory) / 'uploaded.csv'
        frame.to_csv(path, index=False)
        flows, lineage = load_flows(path, require_label=False)
        windows = aggregate_windows(flows)
    lineage.pop('files', None)
    return windows, lineage


def build_live_query(limit=500, lookback_minutes=15, include_history=False):
    if not 1 <= int(limit) <= 5000 or not 1 <= int(lookback_minutes) <= 1440:
        raise ValueError('Live view supports 1–5,000 events and 1–1,440 minutes.')
    query = {'match_all': {}} if include_history else {'range': {'@timestamp': {'gte': f'now-{int(lookback_minutes)}m'}}}
    return {'size': int(limit), 'query': query, 'sort': [{'@timestamp': 'desc'}], 'track_total_hits': False}


def flatten_live_events(hits):
    records = []
    for hit in hits:
        source = dict(hit['_source'])
        address = source.pop('source', {})
        source['source_ip'] = address.get('ip') if isinstance(address, dict) else None
        source['event_id'] = hit.get('_id')
        records.append(source)
    return pd.DataFrame(records)


def fetch_live_windows(limit=500, lookback_minutes=15, include_history=False):
    import requests
    # Administrator configuration only; never an arbitrary URL supplied by a public app visitor.
    url = os.environ.get('CYBERLAB_LIVE_ELASTICSEARCH_URL', 'http://127.0.0.1:19200').rstrip('/')
    response = requests.post(f'{url}/{INDEX}/_search', json=build_live_query(limit, lookback_minutes, include_history), timeout=(3, 10))
    response.raise_for_status()
    data = response.json()
    if data.get('timed_out') or data.get('_shards', {}).get('failed', 0):
        raise ValueError('Elasticsearch returned a partial result; retry after the stream is healthy.')
    return flatten_live_events(data['hits']['hits'])


def suspicious_hosts(scored):
    alerts = scored.loc[scored.suspicious & scored.source_ip.notna()]
    if alerts.empty:
        return pd.DataFrame(columns=['source_ip', 'suspicious_windows', 'maximum_model_score'])
    return alerts.groupby('source_ip', as_index=False).agg(
        suspicious_windows=('suspicious', 'size'), maximum_model_score=('detector_score', 'max')
    ).sort_values('maximum_model_score', ascending=False)


def _display(st, scored, metadata):
    from cyberlab.workspace_ml import csv_bytes
    a, b, c = st.columns(3)
    a.metric('Windows scored in this snapshot', len(scored))
    b.metric('Suspicious windows', int(scored.suspicious.sum()))
    hosts = suspicious_hosts(scored)
    c.metric('Observed suspicious source IPs', len(hosts))
    if scored.source_ip.isna().all():
        st.warning('No valid source IPs are present. Scores are available, but this input cannot identify a suspicious host.')
    st.subheader('Suspicious source IPs — analyst review required')
    st.dataframe(hosts, hide_index=True, width='stretch')
    st.subheader('Window-level scores')
    st.dataframe(scored, hide_index=True, width='stretch')
    st.caption(f"Frozen operating threshold: {metadata['threshold']:.9f}. The score is not a calibrated probability or proof of compromise.")
    st.download_button('Download detection results CSV', csv_bytes(scored), 'cyber_detection_results.csv', 'text/csv', key='detection_download')


def render_detection(root, mode):
    import streamlit as st
    if mode not in ('Run data', 'Live data'):
        raise ValueError('Unknown detection mode.')
    st.title('Run data — threat scoring' if mode == 'Run data' else 'Live data — threat monitoring')
    st.warning('Flagged sources are suspicious, not confirmed attackers. A model score is not attacker attribution; investigate with independent evidence.')

    @st.cache_resource(show_spinner=False)
    def detector(path):
        return load_detector(path)
    try:
        model, metadata = detector(str(root))
    except Exception as exc:
        st.error(f'Detector unavailable: {exc}')
        return
    st.caption(f"Published detector: {metadata['model_name']} · frozen benchmark threshold · no retraining")
    st.info(metadata['scope'])
    if mode == 'Run data':
        schema = st.selectbox('Input schema', ['Feature windows', 'CICIDS flow CSV'], key='detection_schema')
        st.caption('Feature windows require the eight exact model features. Add source_ip and timestamp to retain observed identities. Flow CSV uses the established CICIDS columns and America/Moncton timestamp policy; labels are optional. Uploading a CSV is not live packet capture.')
        columns = [*metadata['features'], 'source_ip', 'timestamp']
        st.download_button('Download empty input template', pd.DataFrame(columns=columns).to_csv(index=False), 'detection_feature_template.csv', 'text/csv')
        uploaded = st.file_uploader('Run data CSV (25 MB / 50,000 rows)', type=['csv'], key='detection_upload')
        if st.button('Run detection now', key='detection_run', disabled=uploaded is None, type='primary'):
            try:
                windows, audit = prepare_input(uploaded.getvalue(), schema)
                scored = score_windows(windows, model, metadata)
                st.session_state['detection_batch_result'] = {'hash': hashlib.sha256(uploaded.getvalue()).hexdigest(), 'schema': schema, 'scores': scored, 'audit': audit}
            except Exception as exc:
                st.error(f'Input rejected: {exc}')
        result = st.session_state.get('detection_batch_result')
        if uploaded is not None and result and result['hash'] == hashlib.sha256(uploaded.getvalue()).hexdigest() and result['schema'] == schema:
            st.json(result['audit'])
            _display(st, result['scores'], metadata)
        return
    st.caption(f'Read-only source: {INDEX}. Only completed feature windows already written by the Kafka/Spark pipeline are scored. This view does not start capture or the pipeline. Refresh interval is not detection latency; host/minute aggregation adds latency.')
    interval = st.slider('Refresh interval (seconds)', 5, 60, 5, key='detection_interval')
    lookback = st.slider('Recent event lookback (minutes)', 1, 1440, 15, key='detection_lookback')
    limit = st.slider('Maximum latest windows per snapshot', 50, 5000, 500, step=50, key='detection_limit')
    history = st.checkbox('Include historical / replayed events', key='detection_history')
    if history:
        st.warning('HISTORICAL REPLAY VIEW: event timestamps are retained. These results do not establish attacks happening now.')
    active = st.toggle('Start live refresh', value=False, key='detection_live_active')
    refresh = st.button('Fetch snapshot now', key='detection_fetch')

    @st.fragment(run_every=f'{interval}s' if active else None)
    def snapshot():
        if not (active or refresh):
            st.info('Start live refresh or fetch one snapshot. No network query runs while paused.')
            return
        try:
            windows = fetch_live_windows(limit, lookback, history)
            if windows.empty:
                st.info('No recent feature windows. Start the real ingestion pipeline, or explicitly enable historical replay. No current attack verdict can be made.')
                return
            scored = score_windows(windows, model, metadata)
            if '@timestamp' in windows:
                st.caption('Event timestamps in this bounded snapshot: ' + str(windows['@timestamp'].min()) + ' → ' + str(windows['@timestamp'].max()))
            _display(st, scored, metadata)
            st.caption('Latest bounded snapshot only, not a durable alert log or a total across the entire stream. No automatic blocking is performed.')
        except Exception:
            # Do not leak server URLs, credentials or upstream response bodies to Cloud visitors.
            st.error('Live source unavailable or invalid. Check Elasticsearch, the Spark stream and server-side CYBERLAB_LIVE_ELASTICSEARCH_URL. A Cloud app cannot reach your PC through its own localhost.')
    snapshot()
