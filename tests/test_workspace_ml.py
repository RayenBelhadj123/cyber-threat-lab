"""Workspace modeling: deterministic, isolated, leak-safe behavior."""
import importlib
import numpy as np
import pandas as pd
import pytest


def api():
    return importlib.import_module('cyberlab.workspace_ml')


def fixture(regression=False, grouped=False):
    rng = np.random.default_rng(12)
    x = rng.normal(size=120)
    df = pd.DataFrame({'x': x, 'z': rng.normal(size=120),
                       'label': x * 2 + rng.normal(0, .1, 120) if regression else np.where(x > 0, 'attack', 'normal')})
    if grouped:
        df['time_group'] = np.repeat(np.arange(12), 10)
    return dict(df=df, raw_df=df.copy(), features=['x', 'z'], target='label',
                fingerprint='fixture-v1', project_name='Fixture', source_name='synthetic',
                date_col=None, mode='explore', cleaning_audit=[])


@pytest.mark.parametrize('regression', [False, True])
def test_validation_prediction_and_frozen_sensitivity(regression):
    ml = api()
    ctx = fixture(regression=regression)
    result = ml.train_models(ctx, models=['Ridge'] if regression else ['Logistic Regression'])
    assert result['task'] == ('regression' if regression else 'classification')
    frame = ctx['df'][['x', 'z']].iloc[:4]
    before = result['pipeline'].predict(frame)
    pred = ml.predict_frame(result, frame)
    assert pred.prediction.tolist() == before.tolist()
    with pytest.raises(ValueError, match='Missing'):
        ml.predict_frame(result, frame.drop(columns='z'))
    with pytest.raises(ValueError, match='numeric'):
        ml.predict_frame(result, frame.assign(x='bad'))
    with pytest.raises(ValueError, match='changed'):
        ml.require_current(result, dict(ctx, fingerprint='changed'))
    table = ml.sensitivity(result, result['feature_defaults'], 'x', points=7)
    assert len(table) == 7 and 'prediction' in table
    assert result['pipeline'].predict(frame).tolist() == before.tolist()
    importance = ml.feature_importance(result, ctx)
    assert set(importance.feature) == {'x', 'z'}
    assert set(importance.method) == {'development permutation'}


@pytest.mark.parametrize('feature', ['score_truth', 'flagged_truth', 'reference_label', 'time_group', 'source_id', 'label'])
def test_refuses_leaky_features(feature):
    ctx = fixture()
    ctx['df'][feature] = 1
    ctx['features'] = ['x', feature]
    with pytest.raises(ValueError, match='Leakage'):
        api().train_models(ctx)


@pytest.mark.parametrize('regression,model', [(False, 'Logistic Regression'), (False, 'Random Forest'), (False, 'Histogram Gradient Boosting'), (False, 'MLP'), (True, 'Ridge'), (True, 'Random Forest'), (True, 'Histogram Gradient Boosting'), (True, 'MLP')])
def test_safe_bundle_roundtrip(regression, model):
    ml = api()
    ctx = fixture(regression=regression)
    result = ml.train_models(ctx, models=[model], folds=2)
    if model in {'Random Forest', 'Histogram Gradient Boosting'}:
        # skops 0.16 deliberately refuses internal native tree objects.
        with pytest.raises(ValueError, match='Untrusted serialization'):
            ml.export_bundle(result)
        return
    bundle = ml.export_bundle(result)
    import io, zipfile, skops.io as sio
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert sio.get_untrusted_types(data=archive.read('model.skops')) == []
    loaded = ml.load_bundle(bundle, ctx)
    assert loaded['winner'] == model
    assert loaded['fingerprint'] == ctx['fingerprint']
    assert np.all(ml.predict_frame(result, ctx['df']).prediction == ml.predict_frame(loaded, ctx['df']).prediction)


def test_bundle_rejects_paths_versions_unknown_types_and_bad_metadata():
    import io, json, zipfile
    ml = api()
    ctx = fixture()
    original = ml.export_bundle(ml.train_models(ctx, models=['Logistic Regression']))
    def edit(name, data):
        with zipfile.ZipFile(io.BytesIO(original)) as z:
            parts = {n: z.read(n) for n in z.namelist()}
        parts[name] = data
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as z:
            for n, value in parts.items():
                z.writestr(n, value)
        return out.getvalue()
    with pytest.raises(ValueError):
        ml.load_bundle(edit('../escape', b'bad'), ctx)
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        metadata = json.loads(z.read('metadata.json'))
    metadata['schema_version'] = 999
    with pytest.raises(ValueError, match='version'):
        ml.load_bundle(edit('metadata.json', json.dumps(metadata).encode()), ctx)
    metadata['schema_version'] = 1
    metadata['features'] = ['incorrect']
    with pytest.raises(ValueError):
        ml.load_bundle(edit('metadata.json', json.dumps(metadata).encode()), ctx)
    # A real skops document containing an unknown type must be denied, not trusted.
    import skops.io as sio
    class Forbidden:
        pass
    with pytest.raises(ValueError, match='Untrusted'):
        ml.load_bundle(edit('model.skops', sio.dumps(Forbidden())), ctx)
    with pytest.raises(ValueError):
        ml.load_bundle(b'not a zip', ctx)


def test_bounded_controls_tracking_registry_and_diagnostics():
    ml = api()
    ctx = fixture()
    result = ml.train_models(ctx, models=['Random Forest'], folds=2,
                             params={'n_estimators': 20, 'max_depth': 3, 'regularization': 2., 'mlp_width': 8, 'mlp_iterations': 50})
    assert result['pipeline'].named_steps['model'].n_estimators == 20
    assert result['pipeline'].named_steps['model'].max_depth == 3
    state = {}
    ml.record_experiment(state, result)
    ml.record_experiment(state, result)
    assert len(state['workspace_experiments']) == 1
    candidate = ml.promote_candidate(state, result, 'Investigate grouped attack recall')
    assert candidate['status'] == 'candidate — not deployed'
    assert all(k.startswith('workspace_') for k in state)
    assert len(ml.diagnose(result)) > 0
    assert len(ml.diagnose(dict(result, metrics={'generalization_gap': .5}))) > 0
    with pytest.raises(ValueError, match='bounds'):
        ml.train_models(ctx, params={'n_estimators': 1000})


APP_SOURCE = '''
import pandas as pd
import streamlit as st
from cyberlab.workspace_ml import PAGES, render
df = pd.read_csv('report_data/workspace_features.csv')
features = [c for c in df.select_dtypes('number') if c not in {'malicious', 'time_group'}]
ctx = dict(df=df, raw_df=df.copy(), features=features, target='malicious', date_col='timestamp',
           fingerprint=st.session_state.get('workspace_test_fingerprint', 'default-fixture'),
           project_name='Actual fixture', source_name='report_data/workspace_features.csv', mode='explore', cleaning_audit=[])
page = st.selectbox('Test page', PAGES, key='workspace_test_page')
render(page, ctx)
'''


def test_apptest_actual_fixture_explicit_training_and_all_pages():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string(APP_SOURCE, default_timeout=120).run()
    assert not app.exception
    assert 'workspace_model_result' not in app.session_state
    app.button(key='workspace_train').click().run()
    assert not app.exception
    result = app.session_state['workspace_model_result']
    assert len(result['features']) == 8
    assert len(result['train_indices']) + len(result['test_indices']) == 16174
    assert len(result['comparison']) == 2
    assert len(app.session_state['workspace_experiments']) == 1
    for page in api().PAGES[1:]:
        app.selectbox(key='workspace_test_page').select(page).run()
        assert not app.exception, page
        assert not app.error, (page, [e.value for e in app.error])
        if page == 'Prediction Page':
            app.button(key='workspace_predict_single').click().run()
            assert not app.exception
            assert len(app.get('download_button')) >= 2
        if page == 'Model Explainability':
            app.button(key='workspace_importance').click().run()
            assert not app.exception
            assert len(app.session_state['workspace_importance_output']['data']) == 8
        if page == 'Scenario Simulator':
            app.button(key='workspace_simulate').click().run()
            assert not app.exception
            assert len(app.session_state['workspace_scenario_output']['data']) == 21
        if page == 'Model Registry':
            app.button(key='workspace_promote').click().run()
            assert not app.exception
            assert len(app.session_state['workspace_registry']) == 1
    app.session_state['workspace_test_fingerprint'] = 'changed'
    app.selectbox(key='workspace_test_page').select('Prediction Page').run()
    assert not app.exception
    assert any('changed' in e.value for e in app.warning)


def test_holdout_labels_never_affect_cv_selection_or_preprocessing():
    ml = api()
    ctx = fixture(regression=True, grouped=True)
    first = ml.train_models(ctx, models=['Ridge', 'Random Forest'])
    altered = dict(ctx, df=ctx['df'].copy())
    altered['df'].iloc[first['test_indices'], altered['df'].columns.get_loc('label')] += 10000
    second = ml.train_models(altered, models=['Ridge', 'Random Forest'])
    pd.testing.assert_frame_equal(first['comparison'], second['comparison'])
    assert first['winner'] == second['winner']
    median = ctx['df'].iloc[first['train_indices']][first['features']].median().to_numpy()
    np.testing.assert_allclose(first['pipeline'].named_steps['imputer'].statistics_, median)
    assert first['metrics']['test_score'] != second['metrics']['test_score']


@pytest.mark.parametrize('mutation', ['missing_metrics', 'bad_predictions', 'bad_ranges', 'bad_comparison', 'bad_version', 'duplicate_members', 'oversized'])
def test_bundle_schema_hardening(mutation):
    import io, json, zipfile
    ml = api()
    ctx = fixture()
    data = ml.export_bundle(ml.train_models(ctx, models=['Logistic Regression']))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = {n: archive.read(n) for n in archive.namelist()}
    meta = json.loads(parts['metadata.json'])
    if mutation == 'missing_metrics':
        del meta['metrics']
    elif mutation == 'bad_predictions':
        meta['predictions'] = []
    elif mutation == 'bad_ranges':
        meta['feature_ranges']['x'] = ['bad', 'data']
    elif mutation == 'bad_comparison':
        meta['comparison'] = []
    elif mutation == 'bad_version':
        meta['sklearn_version'] = '0.0.0'
    elif mutation == 'oversized':
        with pytest.raises(ValueError):
            ml.load_bundle(b'X' * (ml.BUNDLE_LIMIT + 1), ctx)
        return
    parts['metadata.json'] = json.dumps(meta).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
        if mutation == 'duplicate_members':
            with pytest.warns(UserWarning):
                archive.writestr('model.skops', parts['model.skops'])
    with pytest.raises(ValueError):
        ml.load_bundle(out.getvalue(), ctx)


def test_training_bounds_and_missing_input_are_explicit():
    ml = api()
    ctx = fixture()
    for changes in [dict(features=[]), dict(features=['missing']), dict(features=['x'] * 41),
                    dict(df=ctx['df'].head(10)), dict(df=ctx['df'].assign(x=np.nan))]:
        with pytest.raises(ValueError):
            ml.train_models(dict(ctx, **changes))
    assert ml.infer_task(pd.Series([1, 0, 1], dtype='int64')) == 'classification'
    assert ml.infer_task(pd.Series([True, False])) == 'classification'
    assert ml.infer_task(pd.Series([1.1, 2.2])) == 'regression'


def test_training_captures_convergence_warnings_and_refuses_empty_development_feature():
    ml = api()
    ctx = fixture(regression=True, grouped=True)
    result = ml.train_models(ctx, models=['MLP'], params={'mlp_iterations': 50})
    assert any('converg' in text.lower() for text in result['warnings'])
    changed = dict(ctx, df=ctx['df'].copy())
    changed['df'].iloc[result['train_indices'], changed['df'].columns.get_loc('x')] = np.nan
    with pytest.raises(ValueError, match='development'):
        ml.train_models(changed, models=['Ridge'])


def test_apptest_regression_export_restore_and_no_implicit_refits():
    from streamlit.testing.v1 import AppTest
    source = APP_SOURCE.replace("df = pd.read_csv('report_data/workspace_features.csv')", "df = pd.DataFrame({'x': [i / 10 for i in range(120)], 'z': [i % 7 for i in range(120)], 'malicious': [i / 3 + .2 for i in range(120)]})")
    app = AppTest.from_string(source, default_timeout=120).run()
    assert not app.exception
    app.multiselect(key='workspace_models_regression').set_value(['Ridge']).run()
    assert 'workspace_model_result' not in app.session_state
    app.button(key='workspace_train').click().run()
    assert not app.exception
    created = app.session_state['workspace_model_result']['created_at']
    for page in ('Evaluation', 'Fit Diagnostics', 'Model Explainability', 'Scenario Simulator', 'Model Save / Load'):
        app.selectbox(key='workspace_test_page').select(page).run()
        assert not app.exception
        assert not app.error
    app.button(key='workspace_export').click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 1
    assert api().load_bundle(app.session_state['workspace_bundle_output']['data'],
                             dict(df=pd.DataFrame({'x': [i / 10 for i in range(120)], 'z': [i % 7 for i in range(120)], 'malicious': [i / 3 + .2 for i in range(120)]}),
                                  features=['x', 'z'], target='malicious', fingerprint='default-fixture'))['winner'] == 'Ridge'
    app.selectbox(key='workspace_test_page').select('Experiment Tracker').run()
    app.button(key='workspace_activate').click().run()
    assert app.session_state['workspace_model_result']['created_at'] == created
    assert len(app.session_state['workspace_experiments']) == 1


def test_batch_csv_contract_and_formula_safe_download():
    ml = api()
    ctx = fixture()
    result = ml.train_models(ctx, models=['Logistic Regression'])
    data = ctx['df'].iloc[:5].to_csv(index=False).encode()
    predictions = ml.predict_csv(result, data)
    assert len(predictions) == 5
    assert predictions.prediction.tolist() == ml.predict_frame(result, ctx['df'].iloc[:5]).prediction.tolist()
    with pytest.raises(ValueError, match='Missing'):
        ml.predict_csv(result, b'x\n1\n')
    with pytest.raises(ValueError, match='numeric'):
        ml.predict_csv(result, b'x,z\nBAD,1\n')
    with pytest.raises(ValueError, match='10 MiB'):
        ml.predict_csv(result, b'X' * (10 * 1024 * 1024 + 1))
    downloaded = pd.read_csv(__import__('io').BytesIO(ml.csv_bytes(pd.DataFrame({'label': ['=1+1', '@evil', 'normal']}))))
    assert downloaded.label.tolist() == ["'=1+1", "'@evil", 'normal']


def test_training_selects_by_cv_and_keeps_holdout_isolated():
    ml = api()
    result = ml.train_models(fixture(grouped=True), models=['Logistic Regression', 'Random Forest'], folds=3)
    assert result['task'] == 'classification'
    assert result['winner'] == result['comparison'].sort_values('cv_score', ascending=False).iloc[0]['model']
    assert set(result['train_indices']).isdisjoint(result['test_indices'])
    df = fixture(grouped=True)['df']
    assert set(df.iloc[result['train_indices']].time_group).isdisjoint(df.iloc[result['test_indices']].time_group)
    assert 'group' in result['cv_policy'].lower()
    for train, valid in result['cv_splits']:
        assert set(df.iloc[train].time_group).isdisjoint(df.iloc[valid].time_group)
    assert set(result['metrics']) >= {'train_score', 'test_score', 'generalization_gap'}
    assert result['pipeline'].named_steps['imputer'].statistics_.shape == (2,)
    assert result['fingerprint'] == 'fixture-v1'
