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


class NumpyMLPDetector:
    """Fixed 8→64→32→16→1 ReLU network; arrays only, no deserialized code."""
    def __init__(self, arrays):
        shapes={'mean':(8,),'scale':(8,),'w0':(64,8),'b0':(64,),
                'w1':(32,64),'b1':(32,),'w2':(16,32),'b2':(16,),
                'w3':(1,16),'b3':(1,)}
        if set(arrays)!=set(shapes):
            raise ValueError('Unsupported neural array inventory.')
        for name,shape in shapes.items():
            value=np.asarray(arrays[name])
            if value.shape!=shape or value.dtype.kind!='f' or not np.isfinite(value).all():
                raise ValueError('Invalid neural array shape, dtype or value.')
        if not (arrays['scale']>0).all():
            raise ValueError('Neural input scales must be positive.')
        self.arrays=arrays

    def scores(self, x):
        a=self.arrays
        z=((np.log1p(np.clip(np.asarray(x,dtype=float),0,None))-a['mean'])/a['scale']).astype(np.float32)
        for index in range(4):
            z=z@a[f'w{index}'].T+a[f'b{index}']
            if index<3:
                z=np.maximum(z,0)
        return (1/(1+np.exp(-np.clip(z,-80,80)))).reshape(-1)


SAFE_INTERNAL_TYPES = {'sklearn.tree._tree.Tree', 'sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor'}


def _validate_detector(model, name):
    if name == 'pytorch_mlp':
        if type(model) is not NumpyMLPDetector:
            raise ValueError('Unsupported neural inference object.')
        return
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    families = {'extra_trees': ExtraTreesClassifier, 'random_forest': RandomForestClassifier,
                'hist_gradient_boosting': HistGradientBoostingClassifier, 'isolation_forest': IsolationForest}
    if name == 'prior_baseline':
        if type(model) is not dict or set(model) != {'kind', 'probability'} or model['kind'] != name or not 0 <= model['probability'] <= 1:
            raise ValueError('Invalid prior-baseline artifact.')
        return
    if name == 'logistic_regression':
        if type(model) is not Pipeline or [type(step) for _, step in model.steps] != [FunctionTransformer, StandardScaler, LogisticRegression] or model.steps[0][1].func is not np.log1p:
            raise ValueError('Unsupported inference preprocessing pipeline.')
    elif name not in families or type(model) is not families[name]:
        raise ValueError('Unsupported detector family.')
    if name != 'isolation_forest' and list(model.classes_) != [False, True]:
        raise ValueError('Unsupported detector positive class.')


def export_detector(model, name, threshold, root, *, scope, source_bundle_sha256=None):
    import skops.io as sio
    import sklearn
    from cyberlab.pipeline import FEATURES
    artifact_file='detector.skops'
    artifact_format='skops'
    if name == 'pytorch_mlp':
        import torch
        if type(model) is not dict or model.get('kind')!='pytorch_mlp' or type(model['network']) is not torch.nn.Sequential:
            raise ValueError('Unsupported training-network artifact.')
        layers=list(model['network'].children())
        expected=[torch.nn.Linear,torch.nn.ReLU,torch.nn.Dropout,torch.nn.Linear,torch.nn.ReLU,torch.nn.Dropout,torch.nn.Linear,torch.nn.ReLU,torch.nn.Linear]
        if [type(layer) for layer in layers]!=expected:
            raise ValueError('Unsupported training-network architecture.')
        state=model['network'].state_dict()
        arrays={'mean':model['scaler'].mean_.copy(),'scale':model['scaler'].scale_.copy()}
        for target,index in enumerate((0,3,6,8)):
            arrays[f'w{target}']=state[f'{index}.weight'].detach().cpu().numpy().copy()
            arrays[f'b{target}']=state[f'{index}.bias'].detach().cpu().numpy().copy()
        model=NumpyMLPDetector(arrays)
        buffer=io.BytesIO()
        np.savez_compressed(buffer,**arrays)
        payload=buffer.getvalue()
        artifact_file='detector.npz'
        artifact_format='numpy_mlp'
    else:
        payload=sio.dumps(model)
        if set(sio.get_untrusted_types(data=payload)) - SAFE_INTERNAL_TYPES:
            raise ValueError('Detector export contains unsupported Python types.')
    _validate_detector(model, name)
    if not np.isfinite(threshold):
        raise ValueError('Detector threshold must be finite.')
    base=Path(root)/'report_data'
    base.mkdir(parents=True,exist_ok=True)
    metadata={'model_name':name,'features':list(FEATURES),'threshold':float(threshold),
              'artifact_sha256':hashlib.sha256(payload).hexdigest(),'format':artifact_format,'artifact_file':artifact_file,
              'sklearn_version':sklearn.__version__,'scope':scope,
              'threshold_policy':'Frozen development OOF maximum-F1 threshold; held-out captures excluded.'}
    if source_bundle_sha256 is not None:
        metadata['source_bundle_sha256']=source_bundle_sha256
    (base/artifact_file).write_bytes(payload)
    (base/'detector_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def load_detector(root):
    import skops.io as sio
    import sklearn
    base = Path(root) / 'report_data'
    metadata = json.loads((base / 'detector_metadata.json').read_text(encoding='utf-8'))
    artifact_file='detector.npz' if metadata['format']=='numpy_mlp' else 'detector.skops'
    payload = (base / artifact_file).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata['artifact_sha256']:
        raise ValueError('Packaged detector checksum mismatch; restore the trusted release.')
    if sklearn.__version__ != metadata['sklearn_version']:
        raise ValueError(f"Detector requires scikit-learn {metadata['sklearn_version']}; installed {sklearn.__version__}.")
    if metadata['format']=='numpy_mlp':
        import zipfile
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist())>2_000_000:
                raise ValueError('Neural array archive exceeds the safety limit.')
        with np.load(io.BytesIO(payload),allow_pickle=False) as data:
            model=NumpyMLPDetector({key:data[key] for key in data.files})
    elif metadata['format']=='skops':
        if set(sio.get_untrusted_types(data=payload)) - SAFE_INTERNAL_TYPES:
            raise ValueError('Packaged detector contains unsupported Python types.')
        model = sio.loads(payload, trusted=list(SAFE_INTERNAL_TYPES))
    else:
        raise ValueError('Unsupported inference artifact format.')
    _validate_detector(model, metadata['model_name'])
    from cyberlab.pipeline import FEATURES
    if metadata['features'] != FEATURES or not np.isfinite(metadata['threshold']):
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
    from cyberlab.benchmark import _score_model
    out['detector_score'] = model.scores(X) if metadata['format']=='numpy_mlp' else _score_model(metadata['model_name'], model, X)
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
