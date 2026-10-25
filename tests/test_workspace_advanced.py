"""Advanced workspace acceptance tests; use only deterministic offline fixtures."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def advanced():
    assert (ROOT / 'cyberlab/workspace_advanced.py').exists(), 'Advanced implementation is missing'
    return importlib.import_module('cyberlab.workspace_advanced')


def fixture_frame(n=120):
    rng = np.random.default_rng(42)
    return pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
                         'bytes': np.arange(n, dtype=float) + rng.normal(0, 2, n),
                         'packets': rng.uniform(1, 20, n),
                         'protocol': np.where(np.arange(n) % 2, 'TCP', 'UDP'),
                         'malicious': np.arange(n) % 3 == 0})


def context(df=None):
    df = fixture_frame() if df is None else df
    return dict(df=df, raw_df=df.copy(), target='malicious', date_col='timestamp',
                features=['bytes', 'packets'], project_name='Offline cyber lab',
                source_name='test fixture', mode='Beginner', fingerprint='fixture', cleaning_audit=[])


def app(page, df=None):
    # AppTest runs real Streamlit rendering without a live service or secrets.
    source = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from tests.test_workspace_advanced import context
from cyberlab.workspace_advanced import render
render({page!r}, context())
"""
    # Import this test module through its exact file, independent of tests package layout.
    source = source.replace('from tests.test_workspace_advanced import context',
                            "import runpy\ncontext = runpy.run_path(" + repr(str(Path(__file__))) + ")[\"context\"]")
    return AppTest.from_string(source, default_timeout=30).run()


@pytest.mark.parametrize('method', ['KMeans', 'DBSCAN', 'PCA', 'IsolationForest'])
def test_unsupervised_real_finite_bounded_results(method):
    mod = advanced()
    df = fixture_frame(5101)
    df.loc[0, 'bytes'] = np.inf
    df.loc[1, 'packets'] = np.nan
    result = mod.run_unsupervised(df, ['bytes', 'packets'], method)
    assert len(result['rows']) == 5000
    assert result['sampled'] and result['input_rows'] == 5101
    assert np.isfinite(result['rows'][['PC1', 'PC2']]).all().all()
    assert result['metrics']['rows'] == 5000
    if method == 'IsolationForest':
        assert result['metrics']['anomalies'] > 0
    if method == 'PCA':
        assert 0 <= result['metrics']['explained_variance'] <= 1.000001


def test_unsupervised_page_runs_on_explicit_button():
    at = app('Unsupervised Lab')
    assert not at.exception
    assert len(at.get('plotly_chart')) == 0
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.get('download_button')) >= 1


def test_bandit_learns_offline_policy_and_compares_ordered_holdout():
    mod = advanced()
    df = fixture_frame()
    df['bytes'] = np.where(df['malicious'], 100., 0.)
    result = mod.run_bandit(df, 'malicious', 'bytes', 'timestamp', miss_cost=20, review_cost=1, epochs=30)
    assert result['train_rows'] + len(result['evaluation']) == len(df)
    assert result['evaluation']['source_row'].astype(int).min() >= result['train_rows']
    assert result['evaluation']['policy_reward'].sum() > result['evaluation']['ignore_reward'].sum()
    assert set(result['policy']['action']) <= {'ignore', 'review'}
    pd.testing.assert_frame_equal(result['evaluation'], mod.run_bandit(df, 'malicious', 'bytes', 'timestamp', miss_cost=20, review_cost=1, epochs=30)['evaluation'])


def test_bandit_requires_classification_labels():
    with pytest.raises(ValueError, match='classification|labels'):
        advanced().run_bandit(fixture_frame(), 'packets', 'bytes', 'timestamp')


def test_reinforcement_page_has_real_reward_plot_and_download():
    at = app('Reinforcement Lab')
    assert not at.exception
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.get('download_button')) >= 1


def test_forecast_uses_ordered_holdout_persistence_and_recursive_future():
    mod = advanced()
    df = fixture_frame()
    result = mod.run_forecast(df, 'timestamp', 'bytes', frequency='h', aggregation='mean', lags=4, horizon=8)
    holdout, future = result['holdout'], result['future']
    assert len(future) == 8 and future['timestamp'].min() > df.timestamp.max()
    assert 'actual' not in future
    assert result['metrics']['mae'] >= 0 and result['metrics']['baseline_mae'] >= 0
    assert holdout.timestamp.min() > result['train_end']
    assert holdout.iloc[0]['persistence'] == pytest.approx(df.loc[df.timestamp < holdout.timestamp.min(), 'bytes'].iloc[-1])
    changed = df.copy()
    changed.loc[len(df)-1, 'bytes'] += 100000
    later = mod.run_forecast(changed, 'timestamp', 'bytes', frequency='h', lags=4, horizon=8)
    assert later['holdout'].iloc[0]['ridge'] == pytest.approx(holdout.iloc[0]['ridge'])


def test_forecast_count_aggregates_real_rows_and_guards_short_series():
    result = advanced().run_forecast(fixture_frame(), 'timestamp', None, frequency='h', lags=3)
    assert result['series']['value'].sum() == 120
    with pytest.raises(ValueError, match='period|observations'):
        advanced().run_forecast(fixture_frame(5), 'timestamp', 'bytes', frequency='h')


def test_forecast_page_executes_and_exports():
    at = app('Forecast')
    assert not at.exception
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 2
    assert len(at.get('download_button')) >= 2


def test_olap_preserves_count_and_caps_high_cardinality():
    mod = advanced()
    df = fixture_frame()
    df['host'] = ['host' + str(i) for i in range(len(df))]
    result = mod.build_cube(df, ['host', 'protocol'], aggregation='count', max_categories=5)
    assert result['groups']['value'].sum() == len(df)
    assert result['groups']['host'].nunique() <= 6
    assert result['groups']['host'].str.contains('Other').any()
    sums = mod.build_cube(df, ['protocol'], measure='packets', aggregation='sum')
    assert sums['groups']['value'].sum() == pytest.approx(df.packets.sum())
    assert len(result['pivot']) > 0


def test_olap_page_shows_actual_3d_grouped_values_and_heatmap():
    at = app('OLAP & Export')
    assert not at.exception
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    plots = at.get('plotly_chart')
    assert len(plots) >= 2
    import json
    traces = [trace for chart in plots for trace in json.loads(chart.proto.spec)['data']]
    assert any(trace['type'] == 'scatter3d' for trace in traces)
    assert any(trace['type'] in ['heatmap', 'image'] for trace in traces)
    assert len(at.get('download_button')) >= 1


def test_governance_measures_actual_quality_and_early_late_drift():
    mod = advanced()
    df = fixture_frame()
    df.loc[0, 'packets'] = np.nan
    df.loc[1, 'bytes'] = np.inf
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    result = mod.governance_report(context(df))
    checks = result['checks'].set_index('check')
    assert checks.loc['Missing cells', 'observed'] == 1
    assert checks.loc['Duplicate rows', 'observed'] == 1
    assert checks.loc['Non-finite numeric cells', 'observed'] == 1
    assert result['drift'].set_index('feature').loc['bytes', 'mean_shift'] > 20
    assert 'No matching fitted model evidence' in result['card']
    assert 'not production approval' in result['card']
    assert 'timestamp' in result['schema']['column'].tolist()


def test_readiness_does_not_trust_stale_or_unfitted_model_results():
    ctx = context()
    report = advanced().governance_report(ctx, {'fingerprint': 'old', 'metrics': {'accuracy': 1.}})
    assert 'No matching fitted model evidence' in report['card']
    report = advanced().governance_report(ctx, {'fingerprint': ctx['fingerprint'], 'metrics': {'accuracy': 1.}})
    assert 'No matching fitted model evidence' in report['card']


def test_production_readiness_manual_checklist_and_report_exports():
    at = app('Production Readiness')
    assert not at.exception
    assert len(at.checkbox) >= 4
    assert not any(box.value for box in at.checkbox)
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) >= 3
    assert len(at.get('download_button')) >= 3
    assert any('No matching fitted model evidence' in block.value for block in at.markdown)


def test_pipeline_separates_active_rows_local_artifacts_and_unverified_services(tmp_path):
    mod = advanced()
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts' / 'features.ndjson').write_text('{"fixture": true}\n')
    ctx = context()
    ctx['raw_df'] = pd.concat([ctx['df'], ctx['df'].iloc[:3]], ignore_index=True)
    result = mod.pipeline_report(ctx, tmp_path)
    stages = result['stages'].set_index('stage')
    assert stages.loc['Raw dataframe', 'rows'] == 123
    assert stages.loc['Active dataframe', 'rows'] == 120
    inventory = result['artifacts'].set_index('path')
    assert inventory.loc['artifacts/features.ndjson', 'status'] == 'Present locally'
    assert inventory.loc['artifacts/flows.ndjson', 'status'] == 'Absent locally'
    assert all(result['services']['status'] == 'Unverified — not probed')
    assert 'not linked to the active dataframe' in result['notes']


def test_data_pipeline_button_reports_real_inventory_and_downloads():
    at = app('Data Pipeline')
    assert not at.exception
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) >= 3
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.get('download_button')) >= 2


def test_big_data_profile_separates_observations_and_scaling_assumptions():
    result = advanced().big_data_report(context(), projected_rows=1000000, partition_mb=64)
    assert result['observed']['rows'] == 120
    assert result['observed']['memory_bytes'] > 0
    assert result['scenario']['projected_rows'] == 1000000
    assert result['scenario']['estimated_partitions'] >= 1
    assert 'assumption' in result['scenario']['basis'].lower()
    assert all(result['services']['status'] == 'Unverified — not probed')
    assert 'protocol' in result['keys']['column'].tolist()


def test_big_data_page_profiles_without_starting_services():
    at = app('Big Data Readiness')
    assert not at.exception
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.dataframe) >= 2
    assert len(at.get('download_button')) >= 1


def fitted_result(ctx):
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(DummyClassifier(strategy='constant', constant=False))
    model.fit(ctx['df'][ctx['features']].iloc[:90], ctx['df'][ctx['target']].iloc[:90])
    test = list(range(90, 120))
    return dict(fingerprint=ctx['fingerprint'], target=ctx['target'], features=ctx['features'],
                pipeline=model, task='classification', winner='Dummy baseline', test_indices=test,
                y_test=ctx['df'][ctx['target']].iloc[test].to_numpy(),
                predictions=model.predict(ctx['df'][ctx['features']].iloc[test]))


def test_business_costs_are_explicit_scenarios_without_fitted_evidence():
    result = advanced().business_scenario(context(), events=1000, prevalence=.1,
                                          miss_rate=.2, false_positive_rate=.05,
                                          miss_cost=100, review_cost=2)
    assert result['rate_source'] == 'User-estimated rates — scenario, not observed model performance'
    assert result['costs'].set_index('component').loc['Missed attacks', 'expected_count'] == pytest.approx(20)
    assert result['total_cost'] == pytest.approx(2250)
    assert result['baseline_cost'] == pytest.approx(10000)


def test_business_uses_matching_fitted_holdout_rates_only():
    mod = advanced()
    ctx = context()
    result = mod.business_scenario(ctx, fitted_result(ctx), events=1000, prevalence=.1)
    assert result['rate_source'].startswith('Observed held-out')
    assert result['miss_rate'] == 1 and result['false_positive_rate'] == 0
    bad = fitted_result(ctx)
    bad['fingerprint'] = 'stale'
    assert mod.business_scenario(ctx, bad)['rate_source'].startswith('User-estimated')


def test_business_impact_page_downloads_cost_assumptions():
    at = app('Business Impact')
    assert not at.exception
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.get('download_button')) >= 2
    assert any('scenario' in block.value.lower() for block in at.markdown)


def test_conclusion_is_dataset_grounded_and_does_not_invent_training():
    result = advanced().conclusion_report(context())
    assert '120' in result['markdown']
    assert 'test fixture' in result['markdown']
    assert 'No matching fitted model evidence' in result['markdown']
    assert result['target_counts']['rows'].sum() == 120
    ctx = context()
    modeled = advanced().conclusion_report(ctx, fitted_result(ctx))
    assert 'Dummy baseline' in modeled['markdown']
    assert 'production approval' in modeled['markdown']


def test_conclusion_page_generates_downloadable_evidence_summary():
    at = app('Conclusion')
    assert not at.exception
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.get('download_button')) >= 1
    assert any('120' in block.value for block in at.markdown)


def test_unsupervised_rejects_unusable_input():
    with pytest.raises(ValueError, match='finite|usable'):
        advanced().run_unsupervised(pd.DataFrame({'a': [np.inf] * 10}), ['a'], 'KMeans')
