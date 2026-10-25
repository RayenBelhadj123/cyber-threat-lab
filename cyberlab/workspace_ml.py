"""Isolated exploratory supervised workspace. Never changes benchmark artifacts.

All UI state is workspace_-prefixed. Selection uses development CV only; only
that CV-selected model is evaluated on the untouched holdout.
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, GroupKFold,
                                     StratifiedKFold, KFold, cross_val_score)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import f1_score, r2_score, mean_squared_error, accuracy_score, mean_absolute_error

PAGES = ['ML Lab', 'Evaluation', 'Prediction Page', 'Model Save / Load',
         'Model Explainability', 'Scenario Simulator', 'Fit Diagnostics',
         'Experiment Tracker', 'Model Registry']
SEED = 42
MAX_ROWS = 50_000
MAX_FEATURES = 40


BUNDLE_LIMIT = 25 * 1024 * 1024
UNPACKED_LIMIT = 100 * 1024 * 1024


def export_bundle(result):
    import io, json, zipfile
    import sklearn
    import skops, skops.io as sio
    # Inference-only copy: these two training-time attributes are not needed
    # in a persisted predictor. dtype is reconstructed locally after validation;
    # AdamOptimizer is deliberately omitted, never added to a trust allowlist.
    from copy import deepcopy
    frozen = deepcopy(result['pipeline'])
    frozen.named_steps['imputer'].__dict__.pop('_fit_dtype', None)
    frozen.named_steps['imputer'].__dict__.pop('_fill_dtype', None)
    frozen.named_steps['model'].__dict__.pop('_optimizer', None)
    model_data = sio.dumps(frozen)
    unknown = sio.get_untrusted_types(data=model_data)
    if unknown:
        raise ValueError('Untrusted serialization types denied: ' + ', '.join(unknown))
    metadata = {k: v for k, v in result.items() if k not in {'pipeline', 'comparison'}}
    metadata.update(schema_version=1, sklearn_version=sklearn.__version__, skops_version=skops.__version__,
                    comparison=result['comparison'].to_dict(orient='records'))
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported metadata type: {type(value).__name__}')
    meta = json.dumps(metadata, default=encode, allow_nan=False).encode('utf-8')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('metadata.json', meta)
        z.writestr('model.skops', model_data)
    data = output.getvalue()
    if len(data) > BUNDLE_LIMIT or len(model_data) + len(meta) > UNPACKED_LIMIT:
        raise ValueError('Bundle exceeds size limit.')
    return data


def load_bundle(data, ctx):
    import io, json, zipfile
    import sklearn
    import skops, skops.io as sio
    if not isinstance(data, bytes) or len(data) > BUNDLE_LIMIT:
        raise ValueError('Bundle must be ZIP bytes under 25 MiB.')
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            entries = z.infolist()
            if len(entries) != 2 or {e.filename for e in entries} != {'metadata.json', 'model.skops'}:
                raise ValueError('Only metadata.json and model.skops are permitted; paths/duplicates denied.')
            if sum(e.file_size for e in entries) > UNPACKED_LIMIT or any(e.flag_bits & 1 for e in entries):
                raise ValueError('Bundle expanded size/encryption refused.')
            if z.getinfo('metadata.json').file_size > 2 * 1024 * 1024:
                raise ValueError('Metadata exceeds size limit.')
            meta = json.loads(z.read('metadata.json'))
            model_data = z.read('model.skops')
        if not isinstance(meta, dict) or meta.get('schema_version') != 1:
            raise ValueError('Unsupported bundle schema version.')
        if meta.get('sklearn_version') != sklearn.__version__ or meta.get('skops_version') != skops.__version__:
            raise ValueError('Library version mismatch; exact sklearn/skops versions required.')
        require_current(meta, ctx)
        if meta.get('features') != list(ctx['features']) or meta.get('target') != ctx['target']:
            raise ValueError('Bundle feature/target schema does not match active context.')
        if meta.get('task') not in {'classification', 'regression'} or meta.get('winner') not in model_catalog(meta.get('task')):
            raise ValueError('Unsupported model/task metadata.')
        finite = lambda v: type(v) in (int, float) and np.isfinite(v)
        required_metrics = {'train_score', 'test_score', 'generalization_gap'}
        required_metrics |= {'accuracy'} if meta['task'] == 'classification' else {'rmse', 'mae', 'r2'}
        if not isinstance(meta.get('metrics'), dict) or not required_metrics <= set(meta['metrics']) or not all(finite(v) for v in meta['metrics'].values()):
            raise ValueError('Invalid/missing metrics schema.')
        expected_scoring = 'f1_macro' if meta['task'] == 'classification' else 'neg_root_mean_squared_error'
        if meta.get('scoring') != expected_scoring or not isinstance(meta.get('cv_policy'), str):
            raise ValueError('Invalid evaluation policy metadata.')
        if not isinstance(meta.get('created_at'), str):
            raise ValueError('Missing experiment creation time.')
        datetime.fromisoformat(meta['created_at'])
        for key in ('project_name', 'source_name'):
            if not isinstance(meta.get(key), str):
                raise ValueError('Invalid project/source metadata.')
        comparison = meta.get('comparison')
        if not isinstance(comparison, list) or not 1 <= len(comparison) <= 4:
            raise ValueError('Invalid model comparison schema.')
        if any(not isinstance(r, dict) or r.get('model') not in model_catalog(meta['task']) or not finite(r.get('cv_score')) or not finite(r.get('cv_std')) for r in comparison):
            raise ValueError('Invalid comparison scores.')
        if comparison[0]['model'] != meta['winner'] or comparison[0]['cv_score'] != max(r['cv_score'] for r in comparison):
            raise ValueError('Winner must match development CV selection.')
        for key in ('feature_defaults', 'feature_ranges'):
            if not isinstance(meta.get(key), dict) or set(meta[key]) != set(meta['features']):
                raise ValueError('Missing feature statistics.')
        if not all(finite(v) for v in meta['feature_defaults'].values()):
            raise ValueError('Invalid numeric feature defaults.')
        if any(not isinstance(v, list) or len(v) != 2 or not all(finite(n) for n in v) or v[0] > v[1] for v in meta['feature_ranges'].values()):
            raise ValueError('Invalid numeric feature ranges.')
        for key in ('y_test', 'predictions'):
            if not isinstance(meta.get(key), list) or not 1 <= len(meta[key]) <= MAX_ROWS or any(type(v) not in (str, bool, int, float) for v in meta[key]):
                raise ValueError('Invalid holdout output schema.')
        if len(meta['y_test']) != len(meta['predictions']) or len(meta['predictions']) != len(meta.get('test_indices', [])):
            raise ValueError('Holdout output length mismatch.')
        # A skops document is itself a ZIP: guard its expansion before inspection.
        with zipfile.ZipFile(io.BytesIO(model_data)) as inner:
            infos = inner.infolist()
            if len(infos) > 5000 or sum(e.file_size for e in infos) > UNPACKED_LIMIT:
                raise ValueError('Expanded model exceeds size limit.')
            if any(e.filename.startswith(('/', '\\')) or '..' in e.filename.replace('\\', '/').split('/') for e in infos):
                raise ValueError('Unsafe model member path.')
        unknown = sio.get_untrusted_types(data=model_data)
        if unknown:
            raise ValueError('Untrusted serialization types denied: ' + ', '.join(unknown))
        pipeline = sio.loads(model_data, trusted=[])
        expected = model_catalog(meta['task'])[meta['winner']]
        if type(pipeline) is not Pipeline or list(pipeline.named_steps) != ['imputer', 'scaler', 'model']:
            raise ValueError('Only workspace sklearn pipelines are accepted.')
        if type(pipeline.named_steps['model']) is not type(expected) or type(pipeline.named_steps['imputer']) is not SimpleImputer or type(pipeline.named_steps['scaler']) is not StandardScaler:
            raise ValueError('Unexpected pipeline step type.')
        pipeline.named_steps['imputer']._fit_dtype = np.dtype('float64')
        pipeline.named_steps['imputer']._fill_dtype = np.dtype('float64')
        if list(pipeline.feature_names_in_) != meta['features']:
            raise ValueError('Pipeline features disagree with metadata.')
        for key in ('train_indices', 'test_indices'):
            if not isinstance(meta.get(key), list) or not meta[key] or any(type(i) is not int or not 0 <= i < len(ctx['df']) for i in meta[key]):
                raise ValueError('Invalid training/test indices.')
        if set(meta['train_indices']) & set(meta['test_indices']):
            raise ValueError('Overlapping training/test indices.')
        for key in ('feature_defaults', 'feature_ranges'):
            if set(meta.get(key, {})) != set(meta['features']):
                raise ValueError('Missing feature statistics.')
        probe = pd.DataFrame([meta['feature_defaults']], columns=meta['features'])
        pipeline.predict(probe)
        meta['pipeline'] = pipeline
        meta['comparison'] = pd.DataFrame(meta['comparison'])
        meta['y_test'] = np.asarray(meta['y_test'])
        meta['predictions'] = np.asarray(meta['predictions'])
        return meta
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'Invalid safe bundle: {exc}') from exc


def record_experiment(state, result):
    from copy import deepcopy
    records = state.setdefault('workspace_experiments', [])
    run_id = result['created_at']
    if not any(r['created_at'] == run_id for r in records):
        records.append(deepcopy(result))
        del records[:-10]
    state['workspace_model_result'] = result


def promote_candidate(state, result, note=''):
    candidate = dict(run_id=result['created_at'], winner=result['winner'], fingerprint=result['fingerprint'],
                     target=result['target'], task=result['task'], metrics=dict(result['metrics']),
                     status='candidate — not deployed', note=str(note)[:2000],
                     registered_at=datetime.now(timezone.utc).isoformat())
    records = state.setdefault('workspace_registry', [])
    if not any(r['run_id'] == candidate['run_id'] for r in records):
        records.append(candidate)
        del records[:-20]
    return next(r for r in records if r['run_id'] == candidate['run_id'])


def diagnose(result):
    recommendations = []
    gap = result['metrics']['generalization_gap']
    if gap > (.1 if result['task'] == 'classification' else .2 * max(abs(result['metrics'].get('test_score', 0)), .001)):
        recommendations.append('Large train/holdout gap: reduce tree depth or MLP width, increase regularization, and obtain new independent groups. Change settings using development CV only.')
    else:
        recommendations.append('A small fit gap is not proof against overfitting. Validate on new independent time groups before promotion.')
    if result['task'] == 'classification':
        recommendations.append('For rare attacks inspect per-class recall and support, not accuracy alone; compare balanced class weights in development CV.')
    else:
        recommendations.append('Inspect residual scale and extreme errors; test transformations using development CV, never the holdout.')
    recommendations.append('Repeated experiments reuse this holdout: do not choose configurations or thresholds from its score. Reserve fresh external data for final validation.')
    return recommendations


def require_current(result, ctx):
    if not result or result.get('fingerprint') != ctx['fingerprint']:
        raise ValueError('Dataset/target/features changed; train or load a matching model first.')
    return result


def predict_frame(result, frame):
    missing = [c for c in result['features'] if c not in frame]
    if missing:
        raise ValueError('Missing required fields: ' + ', '.join(missing))
    if len(frame) < 1 or len(frame) > MAX_ROWS:
        raise ValueError('Prediction requires 1–50,000 rows.')
    X = frame[result['features']].copy()
    for name in X:
        if not pd.api.types.is_numeric_dtype(X[name]):
            raise ValueError(f'Prediction field must be numeric: {name}')
    X = X.replace([np.inf, -np.inf], np.nan)
    out = X.copy()
    out['prediction'] = result['pipeline'].predict(X)
    if hasattr(result['pipeline'], 'predict_proba'):
        probabilities = result['pipeline'].predict_proba(X)
        for index, label in enumerate(result['pipeline'].classes_):
            out[f'probability_{label}'] = probabilities[:, index]
    return out


def sensitivity(result, baseline, feature, points=21):
    if feature not in result['features']:
        raise ValueError('Select a trained feature.')
    lo, hi = result['feature_ranges'][feature]
    values = np.linspace(lo, hi, max(2, min(101, int(points))))
    frame = pd.DataFrame([baseline] * len(values), columns=result['features'])
    frame[feature] = values
    return predict_frame(result, frame)


def feature_importance(result, ctx):
    from sklearn.inspection import permutation_importance
    require_current(result, ctx)
    # Development-only diagnostic: never use the holdout for iterative exploration.
    frame = ctx['df'].iloc[result['train_indices']].head(2000)
    importance = permutation_importance(result['pipeline'], frame[result['features']].replace([np.inf, -np.inf], np.nan),
                                        frame[result['target']], scoring=result['scoring'],
                                        n_repeats=3, random_state=SEED, n_jobs=1)
    return pd.DataFrame({'feature': result['features'], 'importance': importance.importances_mean,
                         'std': importance.importances_std, 'method': 'development permutation'}).sort_values('importance', ascending=False)


def infer_task(y):
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return 'classification'
    return 'regression'


def model_catalog(task, class_weight=False):
    if task == 'classification':
        weight = 'balanced' if class_weight else None
        return {'Logistic Regression': LogisticRegression(max_iter=400, class_weight=weight, random_state=SEED),
                'Random Forest': RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=2, class_weight=weight, n_jobs=1, random_state=SEED),
                'Histogram Gradient Boosting': HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, class_weight=weight, random_state=SEED),
                'MLP': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, early_stopping=True, random_state=SEED)}
    return {'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=SEED),
            'Histogram Gradient Boosting': HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, random_state=SEED),
            'MLP': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=150, early_stopping=True, random_state=SEED)}


def train_models(ctx, models=None, folds=3, class_weight=False, task='auto', params=None):
    df, target = ctx['df'], ctx['target']
    features = list(ctx['features'])
    if not 1 <= len(features) <= MAX_FEATURES or len(df) > MAX_ROWS:
        raise ValueError('Training requires 1–40 features and at most 50,000 rows; filter data first.')
    if target not in df or not features or len(set(features)) != len(features):
        raise ValueError('Select a valid target and unique numeric features.')
    for name in features:
        low = str(name).lower()
        if name == target or low.startswith(('score_', 'flagged_', 'reference_label')) or low in {'time_group', 'id', 'row_id', 'flow_id', 'label', 'target'} or low.endswith('_id') or name == ctx.get('date_col'):
            raise ValueError(f'Leakage/identifier feature refused: {name}')
        if name not in df or not pd.api.types.is_numeric_dtype(df[name]):
            raise ValueError(f'Feature must be numeric: {name}')
    valid = df[target].notna().to_numpy()
    positions = np.flatnonzero(valid)
    if len(positions) < 20:
        raise ValueError('At least 20 labeled rows are required.')
    X = df.iloc[positions][features].replace([np.inf, -np.inf], np.nan)
    if X.isna().all().any():
        raise ValueError('All-missing features are not trainable.')
    y = df.iloc[positions][target]
    task = infer_task(y) if task == 'auto' else task
    if task not in {'classification', 'regression'}:
        raise ValueError('Unknown task.')
    if task == 'classification' and (y.nunique() < 2 or y.nunique() > 100):
        raise ValueError('Classification needs 2–100 classes.')
    folds = max(2, min(5, int(folds)))
    groups = None
    if 'time_group' in df:
        groups = df.iloc[positions]['time_group'].astype('string').fillna('<missing>').to_numpy()
        if len(np.unique(groups)) < 4:
            raise ValueError('At least four time groups required for independent holdout and CV.')
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(X, y, groups))
        folds = min(folds, len(np.unique(groups[train])))
        cv = list(GroupKFold(n_splits=folds).split(X.iloc[train], y.iloc[train], groups[train]))
        policy = 'Group-disjoint time_group holdout + GroupKFold; not a forward-time guarantee.'
    else:
        stratify = y if task == 'classification' else None
        train, test = train_test_split(np.arange(len(X)), test_size=.25, random_state=SEED, stratify=stratify)
        if task == 'classification':
            folds = min(folds, int(y.iloc[train].value_counts().min()))
            if folds < 2:
                raise ValueError('Too few development examples per class for CV.')
            cv = list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X.iloc[train], y.iloc[train]))
        else:
            cv = list(KFold(folds, shuffle=True, random_state=SEED).split(X.iloc[train]))
        policy = 'Shuffled stratified holdout/CV' if task == 'classification' else 'Shuffled holdout/KFold'
        policy += '; no temporal/group independence proof.'
    if X.iloc[train].isna().all().any():
        raise ValueError('An all-missing development feature cannot be fitted safely.')
    if task == 'classification':
        expected = set(y.unique())
        if set(y.iloc[train].unique()) != expected or any(set(y.iloc[train].iloc[a].unique()) != expected for a, _ in cv):
            raise ValueError('Class missing from a development training fold; revise groups/data.')
    catalog = model_catalog(task, class_weight)
    controls = dict(n_estimators=80, max_depth=12, regularization=1., mlp_width=32, mlp_iterations=150)
    params = params or {}
    bounds = dict(n_estimators=(10, 100), max_depth=(2, 20), regularization=(.001, 100.), mlp_width=(4, 64), mlp_iterations=(50, 200))
    if set(params) - set(bounds) or any(not isinstance(v, (int, float)) or not bounds[k][0] <= v <= bounds[k][1] for k, v in params.items()):
        raise ValueError('Hyperparameters outside supported bounds.')
    controls.update(params)
    for key in ('n_estimators', 'max_depth', 'mlp_width', 'mlp_iterations'):
        if int(controls[key]) != controls[key]:
            raise ValueError('Integer hyperparameters outside supported bounds.')
        controls[key] = int(controls[key])
    catalog['Random Forest'].set_params(n_estimators=controls['n_estimators'], max_depth=controls['max_depth'])
    catalog['MLP'].set_params(hidden_layer_sizes=(controls['mlp_width'], max(4, controls['mlp_width'] // 2)),
                              max_iter=controls['mlp_iterations'], alpha=controls['regularization'] * .001)
    if task == 'classification':
        catalog['Logistic Regression'].set_params(C=1. / controls['regularization'])
    else:
        catalog['Ridge'].set_params(alpha=controls['regularization'])
    models = list(models) if models is not None else list(catalog)[:2]
    if not models or any(m not in catalog for m in models):
        raise ValueError('Choose at least one supported model.')
    scoring = 'f1_macro' if task == 'classification' else 'neg_root_mean_squared_error'
    import warnings
    from threadpoolctl import threadpool_limits
    rows, pipelines, fit_warnings = [], {}, []
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name in dict.fromkeys(models):
            pipe = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('scaler', StandardScaler()), ('model', clone(catalog[name]))])
            scores = cross_val_score(pipe, X.iloc[train], y.iloc[train], cv=cv, scoring=scoring, n_jobs=1, error_score='raise')
            if not np.isfinite(scores).all():
                raise ValueError(f'Non-finite CV scores for {name}.')
            rows.append(dict(model=name, cv_score=float(scores.mean()), cv_std=float(scores.std())))
            pipelines[name] = pipe
        comparison = pd.DataFrame(rows).sort_values('cv_score', ascending=False, kind='stable').reset_index(drop=True)
        winner = comparison.iloc[0]['model']
        pipeline = pipelines[winner].fit(X.iloc[train], y.iloc[train])
        pred_train, pred_test = pipeline.predict(X.iloc[train]), pipeline.predict(X.iloc[test])
        fit_warnings = list(dict.fromkeys(str(w.message) for w in caught))
    score = (lambda a, b: f1_score(a, b, average='macro', zero_division=0)) if task == 'classification' else (lambda a, b: -float(np.sqrt(mean_squared_error(a, b))))
    train_score, test_score = score(y.iloc[train], pred_train), score(y.iloc[test], pred_test)
    metrics = dict(train_score=float(train_score), test_score=float(test_score), generalization_gap=float(train_score - test_score))
    if task == 'classification':
        metrics['accuracy'] = float(accuracy_score(y.iloc[test], pred_test))
    else:
        metrics.update(rmse=-float(test_score), mae=float(mean_absolute_error(y.iloc[test], pred_test)), r2=float(r2_score(y.iloc[test], pred_test)))
    return dict(winner=winner, task=task, features=features, target=target, pipeline=pipeline,
                metrics=metrics, comparison=comparison, cv_policy=policy, scoring=scoring,
                train_indices=positions[train].tolist(), test_indices=positions[test].tolist(),
                cv_splits=[(positions[train][a].tolist(), positions[train][b].tolist()) for a,b in cv],
                fingerprint=ctx['fingerprint'], created_at=datetime.now(timezone.utc).isoformat(),
                controls=controls, class_weight=bool(class_weight), warnings=fit_warnings,
                project_name=ctx.get('project_name', ''), source_name=ctx.get('source_name', ''),
                y_test=y.iloc[test].to_numpy(), predictions=pred_test,
                feature_defaults=X.iloc[train].median().fillna(0).to_dict(),
                feature_ranges={c: [float(X.iloc[train][c].min()), float(X.iloc[train][c].max())] for c in features})


def predict_csv(result, payload):
    import io
    if len(payload) > 10 * 1024 * 1024:
        raise ValueError('CSV exceeds 10 MiB limit.')
    frame = pd.read_csv(io.BytesIO(payload), nrows=MAX_ROWS + 1)
    if len(frame.columns) > 200:
        raise ValueError('CSV exceeds 200 columns.')
    return predict_frame(result, frame)


def csv_bytes(frame):
    # Formula defense applies to metadata/class labels, not numeric features.
    safe = frame.copy()
    for col in safe.select_dtypes(include=['object', 'string']):
        safe[col] = safe[col].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v)
    return safe.to_csv(index=False).encode('utf-8')


def _download_csv(st, frame, label, key, filename):
    st.download_button(label, csv_bytes(frame), filename, mime='text/csv', key=key)


def _manual_inputs(st, result, prefix):
    values = {}
    cols = st.columns(2)
    for i, name in enumerate(result['features']):
        with cols[i % 2]:
            values[name] = st.number_input(str(name), value=float(result['feature_defaults'][name]),
                                           key=f'workspace_{prefix}_{result["created_at"]}_{i}')
    return values


def _show_output(st, result, key, label, filename):
    output = st.session_state.get(key)
    if output and output['run_id'] == result['created_at']:
        st.dataframe(output['data'], hide_index=True, width='stretch')
        _download_csv(st, output['data'], label, key + '_download', filename)
        return output['data']
    return None


def _evaluation(st, result):
    import plotly.express as px
    from sklearn.metrics import classification_report, confusion_matrix
    st.caption('Untouched holdout of the CV-selected winner only. No threshold search, no holdout-based winner selection.')
    cols = st.columns(3)
    cols[0].metric('Development train score', f"{result['metrics']['train_score']:.4f}")
    cols[1].metric('Holdout score', f"{result['metrics']['test_score']:.4f}")
    cols[2].metric('Train − holdout gap', f"{result['metrics']['generalization_gap']:.4f}")
    st.caption('Score: macro F1 (classification) or negative RMSE (regression); higher is better.')
    y, pred = result['y_test'], result['predictions']
    if result['task'] == 'classification':
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T, width='stretch')
        labels = np.unique(np.concatenate([np.asarray(y), np.asarray(pred)]))
        matrix = confusion_matrix(y, pred, labels=labels)
        fig = px.imshow(matrix, x=[str(v) for v in labels], y=[str(v) for v in labels],
                        text_auto=True, labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
                        title='Holdout confusion matrix', color_continuous_scale='Blues')
        st.plotly_chart(fig, width='stretch')
    else:
        residual = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
        frame = pd.DataFrame({'actual': y, 'predicted': pred, 'residual': residual})
        left, right = st.columns(2)
        left.plotly_chart(px.scatter(frame, x='actual', y='predicted', title='Actual versus predicted'), width='stretch')
        right.plotly_chart(px.histogram(frame, x='residual', nbins=30, title='Holdout residual distribution'), width='stretch')
        st.json({k: result['metrics'][k] for k in ('rmse', 'mae', 'r2')})
    _download_csv(st, pd.DataFrame({'row_position': result['test_indices'], 'actual': y, 'prediction': pred}),
                  'Download holdout predictions', 'workspace_holdout_download', 'holdout_predictions.csv')


def render(page, ctx):
    """Render one PAGES entry. All persistent state keys start with workspace_."""
    import io, json
    import streamlit as st
    import plotly.express as px
    if page not in PAGES:
        raise ValueError(f'Unknown modeling workspace page: {page}')
    st.header(page)
    st.caption('Exploratory workspace • fixed benchmark and winner remain unchanged • session-local models')
    state = st.session_state
    result = state.get('workspace_model_result')
    if page == 'ML Lab':
        if not ctx.get('target') or ctx['target'] not in ctx['df']:
            st.warning('Choose a supervised target in the workspace sidebar.')
            return
        inferred = infer_task(ctx['df'][ctx['target']].dropna())
        st.write(f"{len(ctx['df']):,} rows · {len(ctx['features'])} numeric features · inferred task: {inferred}")
        st.info('Winner is selected by development CV, then evaluated once on a separate holdout. Imputation/scaling are fitted within each training fold. Never choose settings from holdout scores.')
        task_choice = st.selectbox('Task', ['Auto', 'Classification', 'Regression'], key='workspace_task')
        task = inferred if task_choice == 'Auto' else task_choice.lower()
        catalog = model_catalog(task)
        selected = st.multiselect('Candidate models', list(catalog), default=list(catalog)[:2], key=f'workspace_models_{task}')
        folds = st.slider('Development CV folds', 2, 5, 3, key='workspace_folds')
        balanced = st.checkbox('Balanced class weights (linear, forest, histogram)', value=True, disabled=task != 'classification', key='workspace_class_weight')
        with st.expander('Bounded hyperparameters'):
            st.caption('Regularization: linear strength (C = 1/strength), Ridge alpha; MLP alpha = strength × 0.001. Histogram has a fixed 80 iterations / 15 leaves.')
            trees = st.slider('Forest trees', 10, 100, 80, key='workspace_trees')
            depth = st.slider('Forest maximum depth', 2, 20, 12, key='workspace_depth')
            regularization = st.number_input('Regularization strength', .001, 100., 1., key='workspace_regularization')
            width = st.slider('MLP first hidden layer width', 4, 64, 32, key='workspace_mlp_width')
            iterations = st.slider('MLP maximum iterations', 50, 200, 150, key='workspace_mlp_iterations')
        st.caption('Limits: 50,000 rows, 40 numeric features, one CV worker, ≤100 trees; identifiers/labels/scores are refused. Neural training is bounded, not guaranteed converged.')
        if st.button('Train selected models', key='workspace_train', type='primary'):
            try:
                with st.spinner('Fitting development CV; evaluating the selected winner…'):
                    result = train_models(ctx, models=selected, folds=folds, class_weight=balanced,
                                          task=task, params=dict(n_estimators=trees, max_depth=depth,
                                          regularization=regularization, mlp_width=width, mlp_iterations=iterations))
                    record_experiment(state, result)
                st.success(f"CV-selected winner: {result['winner']}")
            except Exception as exc:
                st.error(f'Training refused/failed: {exc}')
        if result and result.get('fingerprint') == ctx['fingerprint']:
            st.subheader(f"Current winner: {result['winner']}")
            st.dataframe(result['comparison'], hide_index=True, width='stretch')
            st.caption(result['cv_policy'])
        return
    if page == 'Model Save / Load':
        st.info('ZIP format: metadata.json + model.skops. Exact sklearn/skops versions and active data fingerprint must match. No pickle/joblib. All skops unknown types are denied, with no trust override.')
        st.caption('Safe inference export: Logistic Regression, Ridge, MLPClassifier/MLPRegressor. Native Random Forest / Histogram tree objects are deliberately denied by skops 0.16; those models remain session-local. MLP optimizer state is omitted; bundles are for inference, not continued training.')
        uploaded = st.file_uploader('Load matching safe bundle (.zip, ≤25 MiB)', type=['zip'], key='workspace_model_upload')
        if st.button('Load safe bundle', key='workspace_load', disabled=uploaded is None):
            try:
                result = load_bundle(uploaded.getvalue(), ctx)
                record_experiment(state, result)
                st.success('Matching inference bundle loaded. Uploaded metrics are declarations, not independently authenticated evidence.')
            except Exception as exc:
                st.error(f'Bundle rejected: {exc}')
        if result and result.get('fingerprint') == ctx['fingerprint']:
            if st.button('Prepare safe model bundle', key='workspace_export'):
                try:
                    state['workspace_bundle_output'] = dict(run_id=result['created_at'], data=export_bundle(result))
                except Exception as exc:
                    st.warning(f'Export denied: {exc}')
            output = state.get('workspace_bundle_output')
            if output and output['run_id'] == result['created_at']:
                st.download_button('Download safe model bundle', output['data'], 'workspace_model.zip',
                                   mime='application/zip', key='workspace_bundle_download')
        else:
            st.warning('Train a model or load a matching bundle first; stale results cannot be exported.')
        return
    if page == 'Experiment Tracker':
        records = state.get('workspace_experiments', [])
        st.info('Last 10 session experiments only; download a bundle to preserve an exportable model. Chronological history is not a holdout leaderboard. Repeated experiments reuse the same holdout.')
        if not records:
            st.warning('No experiments yet. Train explicitly in ML Lab.')
            return
        table = pd.DataFrame([dict(run_id=r['created_at'], model=r['winner'], task=r['task'],
                                   cv_score=float(r['comparison'].iloc[0]['cv_score']),
                                   fingerprint=r['fingerprint'], current_data=r['fingerprint'] == ctx['fingerprint']) for r in records])
        st.dataframe(table, hide_index=True, width='stretch')
        _download_csv(st, table, 'Download experiment log', 'workspace_experiment_download', 'workspace_experiments.csv')
        compatible = [r for r in records if r['fingerprint'] == ctx['fingerprint']]
        if compatible:
            run = st.selectbox('Matching saved experiment', [r['created_at'] for r in compatible], key='workspace_restore_run')
            if st.button('Activate selected experiment', key='workspace_activate'):
                selected = next(r for r in compatible if r['created_at'] == run)
                require_current(selected, ctx)
                state['workspace_model_result'] = selected
                st.success('Matching experiment activated; no refit.')
        return
    if page == 'Model Registry':
        st.info('Explicit promotion creates a review candidate only. Nothing is deployed, no benchmark file is replaced, and no filesystem writes occur. Registry is session-local (last 20 candidates).')
        current = bool(result and result.get('fingerprint') == ctx['fingerprint'])
        if not current:
            st.warning('Train or load a matching model before promotion; existing registry entries are historical.')
        note = st.text_area('Review notes / intended use', key='workspace_registry_note')
        if st.button('Promote current model to review candidate', key='workspace_promote', disabled=not current):
            require_current(result, ctx)
            promote_candidate(state, result, note)
            st.success('Candidate recorded — not deployed.')
        records = state.get('workspace_registry', [])
        if records:
            table = pd.DataFrame([{k: v for k,v in r.items() if k != 'metrics'} for r in records])
            st.dataframe(table, hide_index=True, width='stretch')
            st.download_button('Download registry metadata', json.dumps(records, indent=2), 'workspace_registry.json',
                               mime='application/json', key='workspace_registry_download')
        return
    try:
        result = require_current(result, ctx)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.caption(f"Frozen winner: {result['winner']} · {result['task']} · {result['created_at']}")
    st.caption(result['cv_policy'])
    if page == 'Evaluation':
        _evaluation(st, result)
    elif page == 'Prediction Page':
        st.subheader('Single-row prediction')
        values = _manual_inputs(st, result, 'single')
        if st.button('Predict single row', key='workspace_predict_single'):
            try:
                output = predict_frame(result, pd.DataFrame([values]))
                state['workspace_single_output'] = dict(run_id=result['created_at'], data=output)
            except Exception as exc:
                st.error(str(exc))
        _show_output(st, result, 'workspace_single_output', 'Download single prediction', 'single_prediction.csv')
        st.subheader('Batch prediction')
        st.caption('Only the trained features, in training order, are used. Extra fields are ignored; missing or non-numeric fields are refused. NaN/Infinity use the fitted training imputer.')
        template = pd.DataFrame([result['feature_defaults']])[result['features']]
        _download_csv(st, template, 'Download feature template', 'workspace_template_download', 'prediction_template.csv')
        uploaded = st.file_uploader('Batch CSV (≤10 MiB, ≤50,000 rows)', type=['csv'], key='workspace_batch_upload')
        if st.button('Predict uploaded batch', key='workspace_predict_batch', disabled=uploaded is None):
            try:
                output = predict_csv(result, uploaded.getvalue())
                state['workspace_batch_output'] = dict(run_id=result['created_at'], data=output)
            except Exception as exc:
                st.error(f'Batch rejected: {exc}')
        _show_output(st, result, 'workspace_batch_output', 'Download batch predictions', 'batch_predictions.csv')
    elif page == 'Model Explainability':
        st.info('Permutation importance uses up to 2,000 development rows and 3 repeats, never the held-out test set. It is in-sample diagnostic association, not causal explanation; correlated features can share importance.')
        model = result['pipeline'].named_steps['model']
        native = None
        if hasattr(model, 'feature_importances_'):
            native = np.asarray(model.feature_importances_)
            label = 'Native impurity importance (can favor high-cardinality features)'
        elif hasattr(model, 'coef_'):
            native = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
            label = 'Mean absolute standardized coefficient'
        if native is not None:
            table = pd.DataFrame({'feature': result['features'], 'importance': native}).sort_values('importance')
            st.plotly_chart(px.bar(table, x='importance', y='feature', orientation='h', title=label), width='stretch')
        if st.button('Compute development permutation importance', key='workspace_importance'):
            try:
                with st.spinner('Permuting development features…'):
                    output = feature_importance(result, ctx)
                state['workspace_importance_output'] = dict(run_id=result['created_at'], data=output)
            except Exception as exc:
                st.error(str(exc))
        output = _show_output(st, result, 'workspace_importance_output', 'Download importance', 'feature_importance.csv')
        if output is not None:
            st.plotly_chart(px.bar(output.sort_values('importance'), x='importance', y='feature', error_x='std', orientation='h', title='Development permutation importance'), width='stretch')
    elif page == 'Scenario Simulator':
        st.info('Frozen-model what-if: vary one feature over its development range while holding the others at your baseline. No retraining; not causal or a promise of real-world outcomes.')
        baseline = _manual_inputs(st, result, 'scenario')
        feature = st.selectbox('Sensitivity feature', result['features'], key='workspace_scenario_feature')
        if st.button('Run frozen-model scenario', key='workspace_simulate'):
            try:
                output = sensitivity(result, baseline, feature)
                state['workspace_scenario_output'] = dict(run_id=result['created_at'], data=output, feature=feature)
            except Exception as exc:
                st.error(str(exc))
        output = _show_output(st, result, 'workspace_scenario_output', 'Download scenario', 'scenario_predictions.csv')
        if output is not None:
            x = state['workspace_scenario_output']['feature']
            probabilities = [c for c in output if c.startswith('probability_')]
            if probabilities:
                fig = px.line(output, x=x, y=probabilities, title='Frozen class probability sensitivity')
            else:
                fig = px.scatter(output, x=x, y='prediction', title='Frozen prediction sensitivity')
            st.plotly_chart(fig, width='stretch')
    elif page == 'Fit Diagnostics':
        st.subheader('Fit gap and recommended next checks')
        st.dataframe(pd.DataFrame([result['metrics']]), hide_index=True, width='stretch')
        st.write(f"Development rows: {len(result['train_indices']):,} · holdout rows: {len(result['test_indices']):,}")
        for recommendation in diagnose(result):
            st.warning(recommendation)
        for message in result.get('warnings', []):
            st.warning('Training warning: ' + message)
        model = result['pipeline'].named_steps['model']
        if hasattr(model, 'loss_curve_'):
            st.plotly_chart(px.line(y=model.loss_curve_, labels={'index': 'Iteration', 'y': 'Training loss'}, title='Winner training loss'), width='stretch')
        if hasattr(model, 'n_iter_'):
            st.write('Fitted optimizer iterations:', np.asarray(model.n_iter_).tolist())
        _evaluation(st, result)
