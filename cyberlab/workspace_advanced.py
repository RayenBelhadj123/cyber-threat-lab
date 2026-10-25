"""Offline advanced analytics for the active workspace. No service writes or code execution.

Algorithms accept dataframes independently of Streamlit. UI results are scoped to both
active data fingerprints and widget settings, so a prior dataset is never evidence.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

PAGES = ['Unsupervised Lab', 'Reinforcement Lab', 'Forecast', 'OLAP & Export',
         'Production Readiness', 'Data Pipeline', 'Big Data Readiness',
         'Business Impact', 'Conclusion']
SAMPLE_LIMIT = 5000


def run_unsupervised(df, features, method='KMeans', clusters=3, eps=0.6, contamination=0.05):
    """Median-impute and standardize a reproducible bounded sample (no labels used)."""
    features = [c for c in dict.fromkeys(features) if c in df and pd.api.types.is_numeric_dtype(df[c])]
    sampled = df.sample(SAMPLE_LIMIT, random_state=42) if len(df) > SAMPLE_LIMIT else df.copy()
    matrix = sampled[features].replace([np.inf, -np.inf], np.nan).astype(float)
    matrix = matrix.dropna(axis=1, how='all')
    if len(matrix) < 3 or matrix.empty or not (matrix.nunique() > 1).any():
        raise ValueError('Need at least three rows and one usable varying finite numeric feature.')
    values = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(matrix))
    if not np.isfinite(values).all():
        raise ValueError('Inputs could not be transformed to finite values.')
    projection = PCA(n_components=min(2, values.shape[1]), random_state=42)
    coords = projection.fit_transform(values)
    rows = pd.DataFrame({'source_row': sampled.index.astype(str), 'PC1': coords[:, 0],
                         'PC2': coords[:, 1] if coords.shape[1] > 1 else 0})
    metrics = {'rows': len(rows), 'features': len(matrix.columns)}
    if method == 'KMeans':
        k = min(max(2, int(clusters)), len(np.unique(values, axis=0)), len(values) - 1)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(values)
    elif method == 'DBSCAN':
        labels = DBSCAN(eps=float(eps), min_samples=5).fit_predict(values)
    elif method == 'IsolationForest':
        model = IsolationForest(contamination=float(contamination), random_state=42, n_jobs=1)
        labels = model.fit_predict(values)
        rows['anomaly_score'] = -model.score_samples(values)
        metrics['anomalies'] = int((labels == -1).sum())
    elif method == 'PCA':
        labels = np.zeros(len(rows), dtype=int)
        metrics['explained_variance'] = float(projection.explained_variance_ratio_.sum())
    else:
        raise ValueError('Unknown unsupervised method.')
    rows['group'] = labels.astype(str)
    if method in ('KMeans', 'DBSCAN'):
        valid = labels != -1
        n = len(set(labels[valid]))
        metrics.update(clusters=n, noise=int((labels == -1).sum()), silhouette=None)
        if 1 < n < int(valid.sum()):
            metrics['silhouette'] = float(silhouette_score(values[valid], labels[valid],
                                                          sample_size=min(1500, int(valid.sum())), random_state=42))
    return dict(rows=rows, metrics=metrics, sampled=len(df) > SAMPLE_LIMIT,
                input_rows=len(df), features=list(matrix.columns))


def _features(ctx):
    df = ctx['df']
    return [c for c in ctx.get('features', []) if c in df and c != ctx.get('target')
            and pd.api.types.is_numeric_dtype(df[c])]


def _csv(label, df, name):
    st.download_button(label, df.to_csv(index=False).encode('utf-8'), name, 'text/csv', key='workspace_' + name)


def _saved(ctx, page, settings, compute):
    """Only run on the explicit button; never display stale results after edits."""
    key = 'workspace_advanced_' + page
    # Include content hash even when a caller supplies a stale/absent fingerprint.
    df = ctx['df']
    signature = (ctx.get('fingerprint'), tuple(df.columns), tuple(map(str, df.dtypes)),
                 int(pd.util.hash_pandas_object(df, index=True).sum()), repr(settings))
    if st.button('Run ' + page, key=key + '_run', type='primary'):
        try:
            st.session_state[key] = (signature, compute())
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.session_state.pop(key, None)
            st.warning(str(exc))
    saved = st.session_state.get(key)
    if saved and saved[0] == signature:
        return saved[1]
    if saved:
        st.info('Data or settings changed. Run again to refresh the evidence.')
    return None


def _unsupervised(ctx):
    features = st.multiselect('Numeric features (target excluded)', _features(ctx), default=_features(ctx)[:8], key='workspace_unsup_features')
    method = st.selectbox('Method', ['KMeans', 'DBSCAN', 'PCA', 'IsolationForest'], key='workspace_unsup_method')
    k = st.slider('Clusters', 2, 10, 3, key='workspace_unsup_k')
    eps = st.slider('DBSCAN radius (standardized units)', 0.1, 3.0, 0.6, key='workspace_unsup_eps')
    contamination = st.slider('Expected anomaly fraction', 0.01, 0.3, 0.05, key='workspace_unsup_contamination')
    st.caption('Median imputation + standardization. Reproducible sample capped at 5,000 rows; silhouette capped at 1,500. Original row identifiers exported.')
    result = _saved(ctx, 'Unsupervised Lab', (features, method, k, eps, contamination),
                    lambda: run_unsupervised(ctx['df'], features, method, k, eps, contamination))
    if result is not None:
        st.json(result['metrics'])
        st.plotly_chart(px.scatter(result['rows'], x='PC1', y='PC2', color='group', hover_data=['source_row'], title=method + ' — PCA view'), width='stretch')
        st.write(f"Analyzed {len(result['rows']):,} of {result['input_rows']:,} observed rows. Groups and anomaly scores are exploratory, not validated attacks.")
        st.dataframe(result['rows'].head(100), hide_index=True)
        _csv('Download unsupervised results', result['rows'], 'unsupervised.csv')


def _classification_labels(series, positive_label=None):
    valid = series.dropna()
    if valid.empty:
        raise ValueError('Observed classification labels are required.')
    if pd.api.types.is_bool_dtype(valid) or set(valid.unique()) <= {0, 1}:
        return series.astype('Float64')
    if pd.api.types.is_numeric_dtype(valid) or valid.nunique() > 30:
        raise ValueError('Choose discrete classification labels, not a continuous target.')
    if positive_label is not None and valid.nunique() == 2:
        return series.map(lambda x: np.nan if pd.isna(x) else float(x == positive_label))
    benign = valid.astype(str).str.strip().str.lower().isin(['benign', 'normal', 'false', '0'])
    if benign.any():
        return series.map(lambda x: np.nan if pd.isna(x) else float(str(x).strip().lower() not in ['benign', 'normal', 'false', '0']))
    raise ValueError('Select the positive class for binary classification labels; attack semantics cannot be inferred.')


def run_bandit(df, target, signal, date_col=None, miss_cost=20., review_cost=1., epochs=20, positive_label=None):
    """Seeded contextual bandit with a train-only state definition and ordered holdout.

    Reward is a user-defined simulated cost, not an observed analyst outcome.
    Review is assumed to perfectly resolve a label and has a fixed cost.
    """
    if target not in df or signal not in df or target == signal:
        raise ValueError('A classification target and separate numeric signal are required.')
    labels = _classification_labels(df[target], positive_label)
    data = pd.DataFrame({'signal': pd.to_numeric(df[signal], errors='coerce'), 'label': labels,
                         'source_row': df.index.astype(str)})
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['label'])
    if date_col and date_col in df:
        data['time'] = pd.to_datetime(df.loc[data.index, date_col], errors='coerce', utc=True)
        data = data.dropna(subset=['time']).sort_values('time', kind='stable')
    data = data.iloc[:SAMPLE_LIMIT].reset_index(drop=True)
    if len(data) < 20:
        raise ValueError('Need at least 20 observed labeled rows for training and holdout.')
    split = int(len(data) * .75)
    train = data.iloc[:split]
    median = train['signal'].median()
    if not np.isfinite(median):
        raise ValueError('Signal needs finite observed training values.')
    cuts = np.unique(np.quantile(train['signal'].fillna(median), [.33, .67]))
    states = np.digitize(data['signal'].fillna(median), cuts)
    q = np.zeros((len(cuts) + 1, 2))
    counts = np.zeros_like(q)
    rng = np.random.default_rng(42)
    y = data['label'].to_numpy(dtype=float)
    for _ in range(min(100, max(1, int(epochs)))):
        for i in rng.permutation(split):
            state = states[i]
            action = int(rng.integers(2)) if rng.random() < .2 else int(np.argmax(q[state]))
            reward = -float(review_cost) if action else -float(miss_cost) * y[i]
            counts[state, action] += 1
            q[state, action] += (reward - q[state, action]) / counts[state, action]
    actions = np.argmax(q[states[split:]], axis=1)
    evaluation = data.iloc[split:][['source_row', 'label']].copy()
    evaluation['action'] = np.where(actions, 'review', 'ignore')
    evaluation['policy_reward'] = np.where(actions, -float(review_cost), -float(miss_cost) * y[split:])
    evaluation['ignore_reward'] = -float(miss_cost) * y[split:]
    evaluation['review_reward'] = -float(review_cost)
    for col in ['policy_reward', 'ignore_reward', 'review_reward']:
        evaluation['cumulative_' + col] = evaluation[col].cumsum()
    policy = pd.DataFrame({'state': np.arange(len(q)), 'ignore_Q': q[:, 0], 'review_Q': q[:, 1],
                           'action': np.where(q.argmax(axis=1), 'review', 'ignore'),
                           'training_visits': counts.sum(axis=1).astype(int)})
    return dict(policy=policy, evaluation=evaluation, train_rows=split, cuts=cuts.tolist(), input_rows=len(df))


def _reinforcement(ctx):
    df = ctx['df']
    target = ctx.get('target')
    if target not in df or (pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 2):
        target = 'malicious' if 'malicious' in df else None
    st.info('Educational offline alert-triage contextual bandit. No live network actions and no deployed IDS. Rewards are cost assumptions, not measured operational outcomes.')
    if target is None or not _features(ctx):
        st.warning('Eligibility: need classification labels (a malicious boolean is supported) and a numeric signal.')
        return
    positive = None
    if not pd.api.types.is_numeric_dtype(df[target]) and not pd.api.types.is_bool_dtype(df[target]) and df[target].nunique() == 2:
        positive = st.selectbox('Positive (attack) class', list(df[target].dropna().unique()), key='workspace_rl_positive')
    signal = st.selectbox('Observed signal', _features(ctx), key='workspace_rl_signal')
    miss = st.slider('Missed attack cost', 1., 100., 20., key='workspace_rl_miss')
    review = st.slider('Review cost per alert', .1, 10., 1., key='workspace_rl_review')
    epochs = st.slider('Learning passes', 1, 50, 20, key='workspace_rl_epochs')
    st.caption(f'Label: {target}. Ordered 75% training / 25% holdout; at most 5,000 rows. Train-only signal bins. Review assumes perfect resolution; no causal policy value is established.')
    result = _saved(ctx, 'Reinforcement Lab', (target, signal, miss, review, epochs, positive),
                    lambda: run_bandit(df, target, signal, ctx.get('date_col'), miss, review, epochs, positive))
    if result is not None:
        st.dataframe(result['policy'], hide_index=True)
        ev = result['evaluation']
        st.plotly_chart(px.line(ev.reset_index(drop=True), y=['cumulative_policy_reward', 'cumulative_ignore_reward', 'cumulative_review_reward'], title='Held-out cumulative simulated reward vs fixed baselines'), width='stretch')
        st.write(f"Learned on {result['train_rows']} observed rows; evaluated on {len(ev)} subsequent rows. Policy reward: {ev.policy_reward.sum():.2f}; ignore-all baseline: {ev.ignore_reward.sum():.2f}; review-all baseline: {ev.review_reward.sum():.2f}.")
        _csv('Download offline policy', result['policy'], 'triage_policy.csv')
        _csv('Download held-out rewards', ev, 'triage_rewards.csv')


def run_forecast(df, date_col, measure=None, frequency='h', aggregation='mean', lags=4, horizon=12):
    """Aggregate actual observations; evaluate lagged Ridge on an ordered holdout."""
    if date_col not in df:
        raise ValueError('Select an observed date column to aggregate a time series.')
    if frequency not in ['min', 'h', 'D', 'W'] or aggregation not in ['mean', 'sum']:
        raise ValueError('Unsupported time frequency or aggregation.')
    dates = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    data = pd.DataFrame({'timestamp': dates, 'value': 1. if measure is None else pd.to_numeric(df[measure], errors='coerce')})
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    if data.empty:
        raise ValueError('No dated observations are available.')
    seconds = {'min': 60, 'h': 3600, 'D': 86400, 'W': 604800}[frequency]
    if (data.index.max() - data.index.min()).total_seconds() / seconds > 10000:
        raise ValueError('More than 10,000 periods. Choose a coarser frequency.')
    if measure is None:
        series = data.resample(frequency).size().astype(float)
    elif aggregation == 'sum':
        series = data['value'].resample(frequency).sum(min_count=1)
    else:
        series = data['value'].resample(frequency).mean()
    lags = max(1, min(24, int(lags)))
    frame = pd.DataFrame({'actual': series})
    for lag in range(1, lags + 1):
        frame['lag_' + str(lag)] = series.shift(lag)
    usable = frame.dropna()
    if len(usable) < max(16, lags * 2):
        raise ValueError('Need at least 16 complete lagged periods; choose a finer frequency or more observations. Missing measure bins are not fabricated.')
    train_count = int(len(usable) * .8)
    train, test = usable.iloc[:train_count], usable.iloc[train_count:]
    columns = ['lag_' + str(i) for i in range(1, lags + 1)]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.))
    model.fit(train[columns], train['actual'])
    predictions = model.predict(test[columns])
    holdout = pd.DataFrame({'timestamp': test.index, 'actual': test['actual'].to_numpy(),
                            'ridge': predictions, 'persistence': test['lag_1'].to_numpy()})
    metrics = dict(mae=float(mean_absolute_error(holdout.actual, holdout.ridge)),
                   baseline_mae=float(mean_absolute_error(holdout.actual, holdout.persistence)),
                   training_periods=len(train), holdout_periods=len(test), missing_bins=int(series.isna().sum()))
    # Refit only after evaluation, then recursively forecast beyond the real series.
    model.fit(usable[columns], usable.actual)
    history = series.iloc[-lags:].tolist()
    if not np.isfinite(history).all():
        raise ValueError('Latest lag window has missing values; recursive forecast is not eligible.')
    future_values = []
    for _ in range(max(1, min(72, int(horizon)))):
        row = pd.DataFrame([[history[-i] for i in range(1, lags + 1)]], columns=columns)
        value = float(model.predict(row)[0])
        if measure is None or (series.dropna() >= 0).all():
            value = max(0., value)
        history.append(value)
        future_values.append(value)
    times = pd.date_range(series.index[-1], periods=len(future_values) + 1, freq=frequency)[1:]
    return dict(series=series.rename('value').rename_axis('timestamp').reset_index(), holdout=holdout,
                future=pd.DataFrame({'timestamp': times, 'forecast': future_values}), metrics=metrics,
                train_end=train.index.max())


def _forecast(ctx):
    df = ctx['df']
    st.caption('Actual time aggregation → lagged Ridge → ordered 80/20 holdout with a persistence baseline. Holdout is rolling one-step (previous actuals available), not a fixed-origin multi-step test. Future horizon is recursive and has no known future truth.')
    if ctx.get('date_col') not in df:
        st.info('Choose a date column in the workspace first. Forecast requires genuine timestamps.')
        return
    numerics = list(df.select_dtypes(include='number').columns)
    measure = st.selectbox('Series measure', ['Traffic count'] + numerics, index=1 if numerics else 0, key='workspace_forecast_measure')
    freq = st.selectbox('Time bucket', ['h', 'min', 'D', 'W'], key='workspace_forecast_freq')
    agg = st.selectbox('Measure aggregation', ['mean', 'sum'], key='workspace_forecast_agg')
    lags = st.slider('Lag periods', 1, 24, 4, key='workspace_forecast_lags')
    horizon = st.slider('Recursive horizon (periods)', 1, 72, 12, key='workspace_forecast_horizon')
    st.info('Count bins with no records are zero observed rows, not proof of zero network traffic. Numeric measure gaps remain missing; no interpolation or synthetic future labels.')
    result = _saved(ctx, 'Forecast', (measure, freq, agg, lags, horizon),
                    lambda: run_forecast(df, ctx['date_col'], None if measure == 'Traffic count' else measure, freq, agg, lags, horizon))
    if result is not None:
        st.json(result['metrics'])
        st.plotly_chart(px.line(result['holdout'], x='timestamp', y=['actual', 'ridge', 'persistence'], title='Ordered held-out one-step evaluation'), width='stretch')
        fig = go.Figure()
        fig.add_scatter(x=result['series'].timestamp, y=result['series'].value, name='observed aggregate')
        fig.add_scatter(x=result['future'].timestamp, y=result['future'].forecast, name='recursive forecast', line={'dash': 'dash'})
        fig.update_layout(title='Observed history and unverified future estimates', xaxis_title='Time (UTC)', yaxis_title=measure)
        st.plotly_chart(fig, width='stretch')
        better = result['metrics']['mae'] < result['metrics']['baseline_mae']
        st.write('Ridge ' + ('outperformed' if better else 'did not outperform') + ' persistence on this holdout. This does not establish performance on a future incident or another capture.')
        _csv('Download forecast estimates', result['future'], 'forecast.csv')
        _csv('Download holdout evaluation', result['holdout'], 'forecast_holdout.csv')


def build_cube(df, dimensions, measure=None, aggregation='count', max_categories=12):
    """Bounded observed-only OLAP; category capping preserves rows in Other."""
    dimensions = list(dict.fromkeys(dimensions))
    if not dimensions or len(dimensions) > 3 or any(c not in df for c in dimensions):
        raise ValueError('Choose one to three existing dimensions.')
    if aggregation not in ['count', 'sum', 'mean']:
        raise ValueError('Choose count, sum or mean.')
    if aggregation != 'count' and (measure not in df or measure in dimensions):
        raise ValueError('Choose a numeric measure separate from dimensions.')
    work = df[dimensions].copy()
    cap = max(2, min(20, int(max_categories)))
    capped = []
    for col in dimensions:
        values = work[col].astype('string').fillna('(Missing)')
        top = values.value_counts().head(cap).index
        if values.nunique() > cap:
            capped.append(col)
            other = '(Other grouped categories)'
            while other in set(values):
                other += '_'
            values = values.where(values.isin(top), other)
        work[col] = values
    # Pick an internal value name that cannot collide with a dataset column.
    internal = '__workspace_measure'
    while internal in dimensions:
        internal += '_'
    work[internal] = 1. if aggregation == 'count' else pd.to_numeric(df[measure], errors='coerce').replace([np.inf, -np.inf], np.nan)
    grouped = work.groupby(dimensions, observed=True, dropna=False)[internal]
    if aggregation == 'count':
        groups = grouped.size().rename('value').reset_index()
    elif aggregation == 'sum':
        groups = grouped.sum(min_count=1).rename('value').reset_index()
    else:
        groups = grouped.mean().rename('value').reset_index()
    if len(groups) > 10000:
        raise ValueError('Cube exceeds 10,000 observed cells; reduce categories or dimensions.')
    if len(dimensions) > 1:
        # Flatten remaining dimension labels to avoid accidental re-aggregation of means.
        view = groups.copy()
        face = '__workspace_face'
        while face in view:
            face += '_'
        view[face] = view[dimensions[1:]].astype(str).agg(' | '.join, axis=1)
        pivot = view.pivot(index=dimensions[0], columns=face, values='value')
    else:
        pivot = groups.set_index(dimensions[0])[['value']]
    return dict(groups=groups, pivot=pivot, dimensions=dimensions, capped=capped, input_rows=len(df))


def _olap(ctx):
    df = ctx['df'].copy()
    dimensions = list(df.select_dtypes(include=['object', 'string', 'category', 'bool']).columns)
    if ctx.get('date_col') in df:
        dates = pd.to_datetime(df[ctx['date_col']], errors='coerce', utc=True)
        for label, values in [('Time day (UTC)', dates.dt.strftime('%Y-%m-%d')), ('Time hour (UTC)', dates.dt.hour.astype('Int64').astype('string'))]:
            if label not in df:
                df[label] = values
                dimensions.append(label)
    if not dimensions:
        for col in list(df.select_dtypes(include='number').columns)[:3]:
            label = 'Quantile bin: ' + str(col)
            df[label] = pd.qcut(df[col].replace([np.inf, -np.inf], np.nan), q=4, duplicates='drop').astype('string')
            dimensions.append(label)
    selected = st.multiselect('Cube dimensions (1–3)', dimensions, default=dimensions[:3], key='workspace_olap_dimensions')
    aggregation = st.selectbox('Aggregation', ['count', 'sum', 'mean'], key='workspace_olap_aggregation')
    numerics = list(df.select_dtypes(include='number').columns)
    measure = st.selectbox('Numeric measure (ignored for count)', numerics or ['None available'], key='workspace_olap_measure')
    cap = st.slider('Top categories per dimension (others grouped)', 2, 20, 12, key='workspace_olap_cap')
    result = _saved(ctx, 'OLAP & Export', (selected, measure, aggregation, cap), lambda: build_cube(df, selected, measure, aggregation, cap))
    if result is not None:
        groups = result['groups']
        st.dataframe(result['pivot'], width='stretch')
        st.plotly_chart(px.imshow(result['pivot'], aspect='auto', title='Observed cube face — ' + aggregation, color_continuous_scale='Blues'), width='stretch')
        plot = groups.copy()
        # Coordinate encodings are category indices, never invented measures.
        axes = []
        tick_layout = {}
        for i in range(3):
            col = selected[i] if i < len(selected) else None
            codes, labels = pd.factorize(plot[col].astype(str), sort=True) if col else (np.zeros(len(plot)), ['All'])
            name = '__coordinate_' + str(i)
            while name in plot:
                name += '_'
            plot[name] = codes
            axes.append(name)
            tick_layout['xyz'[i] + 'axis'] = dict(title=col or 'All', tickvals=list(range(len(labels))), ticktext=list(labels))
        fig = px.scatter_3d(plot, x=axes[0], y=axes[1], z=axes[2], color='value', hover_data=selected + ['value'], title='3D OLAP cube — color is actual grouped value')
        fig.update_layout(scene=tick_layout)
        st.plotly_chart(fig, width='stretch')
        top = groups.dropna(subset=['value']).nlargest(1, 'value')
        if not top.empty:
            st.write(f"Highest observed {aggregation}: {top.iloc[0]['value']:.3f} for " + ', '.join(f'{c}={top.iloc[0][c]}' for c in selected) + '. Segment differences are descriptive, not evidence of attacks or causality.')
        st.caption(f"{len(groups)} observed cells from {result['input_rows']} rows. Capped dimensions: {', '.join(result['capped']) or 'none'}. Missing cube cells remain blank; mean is not summed across segments.")
        _csv('Download OLAP groups', groups, 'olap_groups.csv')
        _csv('Download pivot table', result['pivot'].reset_index(), 'olap_pivot.csv')


def current_model(ctx, model_result=None):
    """Accept fitted workspace evidence only for this data/target/feature selection."""
    if not isinstance(model_result, dict) or not ctx.get('fingerprint'):
        return None
    if (model_result.get('fingerprint') != ctx['fingerprint'] or
            model_result.get('target') != ctx.get('target') or
            list(model_result.get('features', [])) != _features(ctx)):
        return None
    try:
        check_is_fitted(model_result.get('pipeline'))
    except (NotFittedError, TypeError, AttributeError):
        return None
    return model_result


def governance_report(ctx, model_result=None):
    df = ctx['df']
    numeric = df.select_dtypes(include='number')
    missing = int(df.isna().sum().sum())
    infinite = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if len(numeric.columns) else 0
    duplicates = int(df.duplicated().sum())
    dates = pd.to_datetime(df[ctx['date_col']], errors='coerce', utc=True) if ctx.get('date_col') in df else None
    checks = pd.DataFrame([
        {'check': 'Observed rows', 'observed': len(df), 'status': 'Pass' if len(df) else 'Fail'},
        {'check': 'Missing cells', 'observed': missing, 'status': 'Review' if missing else 'Pass'},
        {'check': 'Duplicate rows', 'observed': duplicates, 'status': 'Review' if duplicates else 'Pass'},
        {'check': 'Non-finite numeric cells', 'observed': infinite, 'status': 'Fail' if infinite else 'Pass'},
        {'check': 'Unparseable or missing timestamps', 'observed': int(dates.isna().sum()) if dates is not None else None,
         'status': ('Review' if dates.isna().any() else 'Pass') if dates is not None else 'Unverified'},
        {'check': 'Missing target labels', 'observed': int(df[ctx['target']].isna().sum()) if ctx.get('target') in df else None,
         'status': ('Review' if df[ctx['target']].isna().any() else 'Pass') if ctx.get('target') in df else 'Unverified'}])
    schema = pd.DataFrame([{'column': str(c), 'dtype': str(df[c].dtype), 'missing': int(df[c].isna().sum()),
                            'unique': int(df[c].nunique(dropna=True)), 'constant': bool(df[c].nunique(dropna=True) <= 1)} for c in df],
                          columns=['column', 'dtype', 'missing', 'unique', 'constant'])
    ordered = df.loc[dates.notna()].iloc[np.argsort(dates.dropna().to_numpy(), kind='stable')] if dates is not None else df
    order_description = 'timestamp order (UTC)' if dates is not None else 'row order (no timestamp evidence)'
    half = len(ordered) // 2
    drift = []
    for c in _features(ctx)[:50]:
        early = pd.to_numeric(ordered[c].iloc[:half], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        late = pd.to_numeric(ordered[c].iloc[half:], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(early) < 4 or len(late) < 4:
            continue
        shift = float(late.mean() - early.mean())
        pooled = float(np.sqrt((early.var() + late.var()) / 2))
        effect = shift / pooled if pooled > 0 else (0. if shift == 0 else None)
        drift.append(dict(feature=c, early_n=len(early), late_n=len(late), early_mean=float(early.mean()),
                          late_mean=float(late.mean()), mean_shift=shift, standardized_shift=effect,
                          status='Review' if effect is None or abs(effect) >= .5 else 'Small shift'))
    drift = pd.DataFrame(drift, columns=['feature', 'early_n', 'late_n', 'early_mean', 'late_mean', 'mean_shift', 'standardized_shift', 'status'])
    evidence = current_model(ctx, model_result)
    model_text = 'No matching fitted model evidence. Train and evaluate on the active dataset before claiming model readiness.'
    if evidence is not None:
        model_text = f"Matching fitted model: {evidence.get('winner', 'unnamed')}; task: {evidence.get('task', 'unspecified')}; selection policy: {evidence.get('cv_policy', 'unspecified')}.\n\nRecorded metrics (not deployment validation): {evidence.get('metrics', {})}"
    card = (f"# Model card — {ctx.get('project_name', 'Cyber workspace')}\n\n"
            f"Source: {ctx.get('source_name', 'unspecified')}; fingerprint: {ctx.get('fingerprint', 'not supplied')}.\n\n"
            f"Observed dataset: {len(df):,} rows × {len(df.columns)} columns; target: {ctx.get('target')}; numeric features: {', '.join(map(str, _features(ctx))) or 'none'}.\n\n"
            f"## Evidence\n{model_text}\n\n"
            f"## Data quality\nMissing cells: {missing}; duplicates: {duplicates}; non-finite numeric cells: {infinite}.\n\n"
            f"## Distribution comparison\nEarly/late halves by {order_description}; {len(drift)} numeric features compared. Standardized mean shifts are descriptive effect sizes, not proof of model decay or statistically corrected hypothesis tests.\n\n"
            "## Intended use and limitations\nEducational offline cyber-traffic analysis. Labels and capture coverage are not independently verified. Correlation, anomalies, and traffic volume are not attack attribution. Test on later captures and distinct hosts; inspect class support, leakage, false positives and drift.\n\n"
            "## Governance\nThis report is not production approval. Human review is required for privacy/retention, label provenance, independent temporal/host evaluation, response playbooks, access controls, rollback and monitoring. Manual checklist selections are self-attestations, not automatic checks.\n")
    return dict(checks=checks, schema=schema, drift=drift, card=card, order=order_description)


def _production(ctx):
    st.warning('Local data checks are not production approval. Service availability, security controls and incident-response effectiveness remain unverified.')
    checklist = ['Label provenance reviewed', 'Independent temporal/host holdout reviewed', 'Privacy and retention approved',
                 'Human response and rollback tested', 'Monitoring owner assigned']
    manual = {name: st.checkbox('Manual attestation: ' + name, key='workspace_governance_' + str(i)) for i, name in enumerate(checklist)}
    evidence = st.session_state.get('workspace_model_result')
    result = _saved(ctx, 'Production Readiness', (manual, repr((evidence or {}).get('created_at'))), lambda: governance_report(ctx, evidence))
    if result is not None:
        st.dataframe(result['checks'], hide_index=True)
        st.dataframe(result['schema'], hide_index=True)
        st.caption('Early/late comparison: ' + result['order'] + '. Effect = mean shift / pooled standard deviation; threshold 0.5 is a review heuristic, not a production gate.')
        st.dataframe(result['drift'], hide_index=True)
        if not result['drift'].empty:
            st.plotly_chart(px.bar(result['drift'], x='feature', y='mean_shift', color='status', title='Observed late minus early means (feature units differ)'), width='stretch')
        attestations = '\n'.join(f"- [{'x' if checked else ' '}] {name} (manual)" for name, checked in manual.items())
        card = result['card'] + '\n## Manual review record\n' + attestations
        st.markdown(card)
        st.download_button('Download model card Markdown', card, 'model_card.md', 'text/markdown', key='workspace_card_download')
        _csv('Download validation checks', result['checks'], 'validation_checks.csv')
        _csv('Download drift effects', result['drift'], 'drift_effects.csv')
        _csv('Download observed schema', result['schema'], 'observed_schema.csv')


def pipeline_report(ctx, root=None):
    """Read-only fixed artifact inventory. Existence is not integrity or publication."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    df = ctx['df']
    raw = ctx.get('raw_df', df)
    stages = pd.DataFrame([
        {'stage': 'Raw dataframe', 'rows': len(raw), 'columns': len(raw.columns), 'bytes_in_memory': int(raw.memory_usage(deep=True).sum())},
        {'stage': 'Active dataframe', 'rows': len(df), 'columns': len(df.columns), 'bytes_in_memory': int(df.memory_usage(deep=True).sum())}])
    paths = ['data/source_manifest.json', 'artifacts/manifest.json', 'artifacts/flows.ndjson',
             'artifacts/features.ndjson', 'artifacts/findings.ndjson', 'artifacts/predictions.ndjson',
             'artifacts/metrics.json', 'artifacts/model.joblib', 'report_data/metrics.json',
             'report_data/workspace_provenance.json', 'compose.yaml', 'docker-compose.yml',
             'spark_jobs/batch.py', 'spark_jobs/streaming.py']
    inventory = []
    for name in paths:
        path = root / name
        try:
            present = path.is_file()
            size = path.stat().st_size if present else None
            status = 'Present locally' if present else 'Absent locally'
        except OSError:
            present, size, status = False, None, 'Unreadable locally'
        inventory.append({'path': name, 'status': status, 'size_bytes': size,
                          'evidence_scope': 'File metadata only; no dataset linkage or service verification'})
    services = pd.DataFrame([{'service': service, 'status': 'Unverified — not probed', 'required_evidence': proof}
                              for service, proof in [
                                  ('Kafka', 'Authenticated broker health, replica/ISR state and replay offsets'),
                                  ('Spark', 'Completed job, watermark, source lag and checkpoint recovery'),
                                  ('Elasticsearch / Kibana', 'Authenticated health and indexed-document/dashboard readback'),
                                  ('MongoDB', 'Replica-set health and exact document readback'),
                                  ('Cassandra', 'Ring health, consistency level and partition-key readback')]])
    notes = ('Local artifacts are not linked to the active dataframe unless provenance is independently verified. '
             'A present file is not a valid checksum, a fitted-model result, a running service, or successful publication. '
             'This public app does not run Docker, open service sockets, deserialize model files, or write to infrastructure.')
    return dict(stages=stages, artifacts=pd.DataFrame(inventory), services=services, notes=notes,
                audit=ctx.get('cleaning_audit', []))


def _pipeline(ctx):
    st.caption('Read-only lineage and local artifact checks; no infrastructure commands or network activity.')
    result = _saved(ctx, 'Data Pipeline', (), lambda: pipeline_report(ctx))
    if result is not None:
        st.dataframe(result['stages'], hide_index=True)
        st.plotly_chart(px.bar(result['stages'], x='stage', y='rows', title='Observed dataframe stage sizes (not inferred row lineage)'), width='stretch')
        st.write('Raw minus active row count:', len(ctx.get('raw_df', ctx['df'])) - len(ctx['df']))
        st.write('Cleaning audit supplied by the workspace:')
        if isinstance(result['audit'], pd.DataFrame):
            st.dataframe(result['audit'], hide_index=True)
        else:
            st.write(result['audit'] if result['audit'] else 'No recorded cleaning actions supplied.')
        st.dataframe(result['artifacts'], hide_index=True)
        st.dataframe(result['services'], hide_index=True)
        st.warning(result['notes'])
        _csv('Download observed pipeline stages', result['stages'], 'pipeline_stages.csv')
        _csv('Download local artifact inventory', result['artifacts'], 'artifact_inventory.csv')
        _csv('Download unverified service checklist', result['services'], 'pipeline_services.csv')


def big_data_report(ctx, projected_rows=1000000, partition_mb=128):
    """Observed pandas footprint plus explicitly assumed linear capacity scenarios."""
    df = ctx['df']
    if not len(df) or not np.isfinite([projected_rows, partition_mb]).all() or projected_rows < 1 or partition_mb < 1:
        raise ValueError('Need observed rows and positive finite capacity assumptions.')
    memory = int(df.memory_usage(deep=True).sum())
    per_row = memory / len(df)
    projected_bytes = float(projected_rows) * per_row
    keys = []
    for col in df.columns[:100]:
        frequencies = df[col].value_counts(dropna=False)
        keys.append(dict(column=str(col), dtype=str(df[col].dtype), distinct=int(df[col].nunique()),
                         null_fraction=float(df[col].isna().mean()),
                         largest_key_fraction=float(frequencies.iloc[0] / len(df)) if len(frequencies) else 0.))
    pipeline = pipeline_report(ctx)
    scenario = dict(projected_rows=int(projected_rows), estimated_memory_bytes=projected_bytes,
                    estimated_partitions=max(1, int(np.ceil(projected_bytes / (partition_mb * 1024 ** 2)))),
                    partition_target_mb=float(partition_mb), basis='Linear assumption from pandas deep-memory bytes/row; excludes shuffle, replication, compression and executor overhead. Not a Spark benchmark.')
    return dict(observed=dict(rows=len(df), columns=len(df.columns), memory_bytes=memory, bytes_per_row=per_row),
                scenario=scenario, keys=pd.DataFrame(keys), services=pipeline['services'], artifacts=pipeline['artifacts'])


def _big_data(ctx):
    rows = st.number_input('Scenario row count (not ingested)', min_value=1, max_value=1000000000, value=max(1000000, len(ctx['df'])), key='workspace_big_rows')
    partition = st.slider('Assumed target partition size (MiB)', 16, 512, 128, key='workspace_big_partition')
    st.caption('Profiles only the local dataframe and known artifact metadata. Does not start Docker, Kafka, Spark, databases or network probes.')
    result = _saved(ctx, 'Big Data Readiness', (rows, partition), lambda: big_data_report(ctx, rows, partition))
    if result is not None:
        st.markdown('#### Observed local footprint')
        st.json(result['observed'])
        st.dataframe(result['keys'], hide_index=True)
        st.markdown('#### Assumed scaling scenario — not measured throughput')
        st.json(result['scenario'])
        chart = pd.DataFrame({'rows': [len(ctx['df']), rows], 'estimated_MiB': [result['observed']['memory_bytes'] / 1024**2, result['scenario']['estimated_memory_bytes'] / 1024**2], 'scope': ['observed local', 'linear assumption']})
        st.plotly_chart(px.bar(chart, x='scope', y='estimated_MiB', hover_data=['rows'], title='Local footprint vs capacity assumption'), width='stretch')
        st.write('Avoid partition keys dominated by a single value; preserve event-time and stable event IDs. Profile host/time skew before choosing partitioning. Replay tests must verify idempotency, offsets, checkpoint recovery and exact sink IDs.')
        st.dataframe(result['artifacts'], hide_index=True)
        st.dataframe(result['services'], hide_index=True)
        st.warning('Distributed readiness is unverified. Local files and a capacity estimate do not establish service health, replication, streaming correctness or production security.')
        _csv('Download candidate-key profile', result['keys'], 'partition_key_profile.csv')
        _csv('Download capacity assumption', pd.DataFrame([result['scenario']]), 'capacity_scenario.csv')


def business_scenario(ctx, model_result=None, events=1000, prevalence=.1, miss_rate=.2,
                      false_positive_rate=.05, miss_cost=100., review_cost=2., positive_label=None):
    """Expected cost scenario; optional verified local holdout error rates, never ROI proof."""
    numbers = [events, prevalence, miss_rate, false_positive_rate, miss_cost, review_cost]
    if not np.isfinite(numbers).all() or events < 1 or min(miss_cost, review_cost) < 0 or any(not 0 <= x <= 1 for x in [prevalence, miss_rate, false_positive_rate]):
        raise ValueError('Use finite nonnegative costs, positive events and rates between zero and one.')
    rate_source = 'User-estimated rates — scenario, not observed model performance'
    support = None
    model = current_model(ctx, model_result)
    if model is not None and model.get('task') == 'classification':
        try:
            positions = np.asarray(model.get('test_indices', []), dtype=int)
            truth = np.asarray(model.get('y_test', []))
            pred = np.asarray(model.get('predictions', []))
            if (len(positions) < 2 or truth.shape != pred.shape or len(truth) != len(positions)
                    or len(np.unique(positions)) != len(positions) or positions.min() < 0 or positions.max() >= len(ctx['df'])):
                raise ValueError('No aligned held-out prediction evidence.')
            observed = ctx['df'].iloc[positions][ctx['target']].to_numpy()
            if not np.array_equal(observed, truth):
                raise ValueError('Held-out truth differs from active rows.')
            repeated = model['pipeline'].predict(ctx['df'].iloc[positions][_features(ctx)])
            if not np.array_equal(repeated, pred):
                raise ValueError('Recorded predictions differ from fitted model.')
            # Use one common class mapping for truth and predictions, including an all-negative prediction vector.
            combined = _classification_labels(pd.Series(np.concatenate([truth, pred])), positive_label).to_numpy(dtype=float)
            actual, predicted = combined[:len(truth)], combined[len(truth):]
            positives, negatives = actual == 1, actual == 0
            if not positives.any() or not negatives.any() or not np.isfinite(combined).all():
                raise ValueError('Both classes must have held-out support to estimate both error rates.')
            miss_rate = float((predicted[positives] == 0).mean())
            false_positive_rate = float((predicted[negatives] == 1).mean())
            support = dict(positive=int(positives.sum()), negative=int(negatives.sum()))
            rate_source = 'Observed held-out error rates from matching fitted model; scaled costs remain a scenario'
        except (ValueError, TypeError, KeyError, IndexError):
            pass  # Keep visibly user-estimated rates rather than invent undefined rates.
    attacks = float(events) * prevalence
    missed = attacks * miss_rate
    false_alerts = float(events) * (1 - prevalence) * false_positive_rate
    true_alerts = attacks - missed
    costs = pd.DataFrame([
        dict(component='Missed attacks', expected_count=missed, cost_per_event=miss_cost, estimated_cost=missed * miss_cost),
        dict(component='False alert reviews', expected_count=false_alerts, cost_per_event=review_cost, estimated_cost=false_alerts * review_cost),
        dict(component='True alert reviews', expected_count=true_alerts, cost_per_event=review_cost, estimated_cost=true_alerts * review_cost)])
    return dict(costs=costs, total_cost=float(costs.estimated_cost.sum()), baseline_cost=attacks * miss_cost,
                rate_source=rate_source, miss_rate=miss_rate, false_positive_rate=false_positive_rate,
                support=support, events=int(events), prevalence=float(prevalence),
                assumptions='Expected costs in user-defined currency units. Assumes stable error rates and prevalence, fixed review cost, and all alerted attacks resolved. Excludes implementation, response and harm variability. Not realized savings.')


def _business(ctx):
    events = st.number_input('Scenario traffic events', min_value=1, max_value=1000000000, value=max(1, len(ctx['df'])), key='workspace_cost_events')
    prevalence = st.slider('Assumed attack prevalence', 0., 1., .1, key='workspace_cost_prevalence')
    missed = st.slider('Estimated missed-attack rate (fallback)', 0., 1., .2, key='workspace_cost_miss')
    false = st.slider('Estimated false-positive rate (fallback)', 0., 1., .05, key='workspace_cost_false')
    miss_cost = st.number_input('Cost per missed attack (assumed units)', min_value=0., value=100., key='workspace_cost_missed_unit')
    review_cost = st.number_input('Cost per reviewed alert (assumed units)', min_value=0., value=2., key='workspace_cost_review_unit')
    positive = None
    if ctx.get('target') in ctx['df']:
        values = ctx['df'][ctx['target']]
        if not pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and values.nunique() == 2:
            positive = st.selectbox('Positive (attack) class for cost rates', list(values.dropna().unique()), key='workspace_cost_positive')
    model = st.session_state.get('workspace_model_result')
    st.caption('If a matching fitted classifier has aligned held-out predictions and both classes, use its error rates. Otherwise use the explicitly estimated fallback rates. Traffic volume, prevalence and costs are always scenario inputs.')
    result = _saved(ctx, 'Business Impact', (events, prevalence, missed, false, miss_cost, review_cost, positive, repr((model or {}).get('created_at'))),
                    lambda: business_scenario(ctx, model, events, prevalence, missed, false, miss_cost, review_cost, positive))
    if result is not None:
        st.markdown('**' + result['rate_source'] + '**')
        st.json({k: v for k, v in result.items() if k not in ['costs', 'assumptions']})
        st.dataframe(result['costs'], hide_index=True)
        st.plotly_chart(px.bar(result['costs'], x='component', y='estimated_cost', title='Scenario expected cost — not realized savings'), width='stretch')
        st.warning(result['assumptions'])
        _csv('Download cost scenario', result['costs'], 'business_cost_scenario.csv')
        assumptions = '# Business impact scenario\n\n' + result['rate_source'] + '\n\n' + result['assumptions'] + '\n\n' + repr({k: v for k, v in result.items() if k not in ['costs', 'assumptions']})
        st.download_button('Download scenario assumptions', assumptions, 'business_assumptions.md', 'text/markdown', key='workspace_business_download')


def conclusion_report(ctx, model_result=None):
    df = ctx['df']
    report = governance_report(ctx, model_result)
    counts = pd.DataFrame(columns=['label', 'rows'])
    target = ctx.get('target')
    label_text = 'No target is selected; supervised detection performance is not established.'
    if target in df:
        freq = df[target].astype('string').fillna('(Missing)').value_counts()
        counts = freq.rename_axis('label').rename('rows').reset_index()
        label_text = f"Target `{target}` has {df[target].nunique()} observed non-missing values; most frequent label/value covers {freq.iloc[0] / max(1, len(df)):.1%} of rows." if len(freq) else 'Target has no observed values.'
    checks = report['checks']
    findings = checks.loc[checks.status.isin(['Review', 'Fail']), 'check'].tolist()
    actions = ['Resolve or document ' + str(item).lower() + '.' for item in findings]
    if current_model(ctx, model_result) is None:
        actions.append('Train a baseline using leakage-filtered features and an independent ordered/host-aware evaluation; report held-out class support.')
    actions += ['Confirm label provenance and capture coverage before interpreting anomalies as attacks.',
                'Review temporal/host leakage, drift and false-alert costs with an analyst before any deployment.']
    markdown = ('# Evidence-based conclusion\n\n' + f"Project: {ctx.get('project_name', 'Cyber workspace')}; source: {ctx.get('source_name', 'unspecified')}.\n\n"
                + f"Observed active data: {len(df):,} rows, {len(df.columns)} columns, {len(_features(ctx))} eligible numeric features.\n\n"
                + label_text + '\n\n'
                + 'Data checks requiring review: ' + (', '.join(findings) or 'none detected by these limited checks') + '.\n\n'
                + '## Next actions\n' + '\n'.join('- ' + a for a in actions) + '\n\n'
                + report['card'])
    return dict(markdown=markdown, checks=checks, target_counts=counts)


def _conclusion(ctx):
    model = st.session_state.get('workspace_model_result')
    st.caption('Generate a conclusion from the active dataset and matching trained evidence only. Does not infer completed experiments or service health.')
    result = _saved(ctx, 'Conclusion', repr((model or {}).get('created_at')), lambda: conclusion_report(ctx, model))
    if result is not None:
        st.markdown(result['markdown'])
        st.dataframe(result['checks'], hide_index=True)
        if not result['target_counts'].empty:
            st.plotly_chart(px.bar(result['target_counts'].head(20), x='label', y='rows', title='Observed target support (top 20 values)'), width='stretch')
        st.download_button('Download evidence-based conclusion', result['markdown'], 'conclusion.md', 'text/markdown', key='workspace_conclusion_download')
        _csv('Download target support', result['target_counts'], 'conclusion_target_support.csv')


def render(page, ctx):
    """Render one exact PAGES entry using the active pandas dataframe context."""
    if page not in PAGES:
        raise ValueError(f'Unknown advanced page: {page}')
    st.subheader(page)
    if page == 'Unsupervised Lab':
        _unsupervised(ctx)
    elif page == 'Reinforcement Lab':
        _reinforcement(ctx)
    elif page == 'Forecast':
        _forecast(ctx)
    elif page == 'OLAP & Export':
        _olap(ctx)
    elif page == 'Production Readiness':
        _production(ctx)
    elif page == 'Data Pipeline':
        _pipeline(ctx)
    elif page == 'Big Data Readiness':
        _big_data(ctx)
    elif page == 'Business Impact':
        _business(ctx)
    elif page == 'Conclusion':
        _conclusion(ctx)
