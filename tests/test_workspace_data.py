"""Small, explicitly synthetic fixtures test behavior, never production data."""
import importlib
import json
import numpy as np
import pandas as pd
import pytest


def module():
    return importlib.import_module('cyberlab.workspace_data')


def fixture():
    return pd.DataFrame({'bytes': [1., 2., 3., np.nan], 'packets': [2., 4., 6., 8.],
                         'malicious': [0, 0, 1, 1], 'timestamp': ['2026-01-01'] * 4,
                         'time_group': ['A', 'A', 'B', 'B'], 'label': ['benign', 'benign', 'attack', None]})


def test_profile_dictionary_and_setup_preserve_labels_and_exclude_leaks():
    m = module()
    data = fixture()
    data['malicious_copy'] = data.malicious
    setup = m.smart_setup(data)
    assert setup['target'] == 'malicious'
    assert setup['date_col'] == 'timestamp'
    assert setup['task'] == 'Classification'
    assert 'bytes' in setup['features']
    assert not {'malicious', 'malicious_copy', 'time_group', 'timestamp'} & set(setup['features'])
    assert m.smart_setup(pd.DataFrame({'name': ['a', 'b']}))['target'] is None
    assert m.smart_setup(pd.DataFrame({'label': ['normal', 'attack']}))['task'] == 'Classification'
    assert m.smart_setup(pd.DataFrame({'measurement': [1, 2, 3]}))['date_col'] is None
    profile = m.dataset_profile(data)
    assert profile['rows'] == 4 and profile['missing_cells'] == 2
    dictionary = m.data_dictionary(data, 'timestamp', 'malicious', ['bytes'])
    assert dictionary.set_index('column').loc['malicious', 'role'] == 'Target'
    assert dictionary.set_index('column').loc['packets', 'role'] == 'Other'
    assert m.dataset_profile(pd.DataFrame())['missing_cells'] == 0
    json.dumps(setup)


def test_cleaning_is_copy_audited_handles_all_missing_and_categorical():
    m = module()
    data = fixture()
    data['empty'] = np.nan
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    original = data.copy(deep=True)
    cleaned, audit = m.clean_dataframe(data, {'deduplicate': True, 'drop_empty': True,
        'drop_columns': ['timestamp'], 'missing_columns': ['bytes'], 'missing_strategy': 'median',
        'categorical_columns': ['label']}, target='malicious')
    pd.testing.assert_frame_equal(data, original)
    assert len(cleaned) == 4 and 'empty' not in cleaned and 'timestamp' not in cleaned
    assert cleaned.bytes.isna().sum() == 0 and cleaned.label.isna().sum() == 0
    assert audit and all('action' in row and 'changed' in row for row in audit)
    json.dumps(audit)
    out, audit = m.clean_dataframe(data, {'missing_columns': ['empty'], 'missing_strategy': 'correlation'})
    assert out['empty'].isna().all()
    assert not any(row['changed'] for row in audit)
    with pytest.raises(ValueError, match='strategy'):
        m.clean_dataframe(data, {'missing_strategy': 'magic'})


@pytest.mark.parametrize('strategy', ['cap', 'mean', 'median', 'correlation', 'remove'])
def test_outlier_strategies_and_target_protection(strategy):
    m = module()
    data = pd.DataFrame({'x': [1., 2., 3., 4., 5., 6., 7., 8., 99.],
                         'z': [2., 4., 6., 8., 10., 12., 14., 16., 18.], 'malicious': [0] * 8 + [1]})
    out, audit = m.clean_dataframe(data, {'outlier_columns': ['x'], 'outlier_strategy': strategy})
    assert len(out) == (8 if strategy == 'remove' else 9)
    if strategy != 'remove':
        assert out.x.iloc[-1] < 99
    assert sum(row['changed'] for row in audit) >= 1
    with pytest.raises(ValueError, match='target'):
        m.clean_dataframe(data, {'drop_columns': ['malicious']}, target='malicious')
    out, audit = m.clean_dataframe(data, {'drop_columns': ['malicious'], 'allow_target': True}, target='malicious')
    assert 'malicious' not in out and audit[0]['target_affected']


def test_correlated_imputation_uses_observed_rows_not_target():
    m = module()
    data = pd.DataFrame({'x': list(map(float, range(10))) + [np.nan],
                         'z': list(map(float, range(0, 22, 2))), 'malicious': [1] * 11})
    out, audit = m.clean_dataframe(data, {'missing_columns': ['x'], 'missing_strategy': 'correlation'}, target='malicious')
    assert out.x.iloc[-1] == pytest.approx(10.)
    assert any('z' in row.get('detail', '') for row in audit)


def test_exploration_filters_grouped_stats_and_numeric_correlation():
    m = module()
    data = fixture()
    view = m.filter_rows(data, 'label', ['attack'], 'packets', 3., 7.)
    assert len(view) == 1 and view.packets.iloc[0] == 6.
    pd.testing.assert_frame_equal(m.filter_rows(data), data)
    grouped = m.compare_groups(data, 'time_group', ['packets'], 'mean')
    assert grouped.set_index('group').loc['A', 'packets'] == 3.
    assert grouped.set_index('group').loc['B', 'packets'] == 7.
    corr = m.numeric_correlation(data, ['bytes', 'packets', 'label'])
    assert corr.shape == (2, 2) and corr.loc['bytes', 'packets'] == pytest.approx(1.)
    assert m.compare_groups(data, 'bad', ['bytes']).empty
    assert m.numeric_correlation(pd.DataFrame({'x': ['a']})).empty


def app(page, variant='normal', extra=''):
    from streamlit.testing.v1 import AppTest
    script = '''
import pandas as pd
import numpy as np
from cyberlab.workspace_data import render
# Synthetic test fixture, deliberately local to the harness.
df = pd.DataFrame({'bytes': [1., 2., 3., np.nan], 'packets': [2., 4., 6., 8.],
'malicious': [0, 0, 1, 1], 'timestamp': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
 'time_group': ['A', 'A', 'B', 'B'], 'label': ['benign', 'benign', 'attack', None]})
VARIANT
ctx = dict(df=df, raw_df=df.copy(), target='malicious', date_col='timestamp', features=['bytes', 'packets'],
project_name='Fixture project', source_name='TEST FIXTURE', mode='test', fingerprint='test-fixture', cleaning_audit=[])
EXTRA
render(PAGE, ctx)
'''.replace('PAGE', repr(page)).replace('EXTRA', extra).replace('VARIANT', {'normal': '', 'empty': 'df = pd.DataFrame()',
    'text': "df = pd.DataFrame({'name': ['a', 'b']})", 'duplicate': "df.columns = ['x'] * len(df.columns)"}[variant])
    return AppTest.from_string(script, default_timeout=30).run()


def test_project_manager_exports_actual_profile_and_dataset():
    at = app('Project Manager')
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert len(at.get('download_button')) >= 2
    assert any('Fixture project' in str(t.value) for t in at.markdown)


def test_smart_setup_has_evidence_and_downloadable_recommendations():
    at = app('Smart Auto-Setup')
    assert not at.exception
    assert len(at.dataframe) >= 1 and len(at.get('download_button')) >= 1
    assert any('Classification' in str(t.value) for t in at.markdown)


def test_cleaning_ui_requires_preview_and_applies_working_copy_only():
    at = app('Data Cleaning Studio')
    assert not at.exception
    apply = next(b for b in at.button if b.label == 'Apply preview')
    assert apply.disabled
    at.multiselect(key='workspace_data_data_cleaning_studio_missing').set_value(['bytes'])
    at.selectbox(key='workspace_data_data_cleaning_studio_missing_strategy').set_value('median')
    next(b for b in at.button if b.label == 'Preview changes').click().run()
    assert not at.exception
    assert not next(b for b in at.button if b.label == 'Apply preview').disabled
    next(b for b in at.button if b.label == 'Apply preview').click().run()
    assert not at.exception
    assert at.session_state['workspace_cleaned'].bytes.isna().sum() == 0
    assert isinstance(at.session_state['workspace_cleaning_audit'], list)
    at.selectbox(key='workspace_data_data_cleaning_studio_missing_strategy').set_value('mean').run()
    assert next(b for b in at.button if b.label == 'Apply preview').disabled


@pytest.mark.parametrize('page', ['Overview', 'Data Quality', 'Data Dictionary'])
def test_analytic_summary_pages_have_tables_charts_and_exports(page):
    at = app(page)
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.get('download_button')) >= 1


def test_dictionary_search_filters_real_column_rows():
    at = app('Data Dictionary')
    at.text_input(key='workspace_data_data_dictionary_search').set_value('packets').run()
    assert not at.exception
    assert at.dataframe[0].value['column'].tolist() == ['packets']


@pytest.mark.parametrize('chart', ['Histogram', 'Scatter', 'Line', 'Box', 'Violin'])
def test_explore_chart_controls_produce_actual_charts_and_filtered_export(chart):
    at = app('Explore')
    assert not at.exception
    assert len(at.selectbox) > 0
    at.selectbox(key='workspace_data_explore_chart').set_value(chart).run()
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.get('download_button')) >= 1
    assert len(at.dataframe) >= 1


def test_explore_category_filter_changes_rows():
    at = app('Explore')
    at.selectbox(key='workspace_data_explore_category').set_value('label').run()
    at.multiselect(key='workspace_data_explore_values').set_value(['attack']).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 1


def test_compare_group_metrics_are_real_and_exported():
    at = app('Compare')
    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1 and len(at.get('download_button')) >= 1
    at.selectbox(key='workspace_data_compare_group').set_value('time_group')
    at.multiselect(key='workspace_data_compare_metrics').set_value(['packets']).run()
    assert not at.exception
    result = at.dataframe[0].value.set_index('group')
    assert result.loc['A', 'packets'] == 3. and result.loc['B', 'packets'] == 7.
    at.selectbox(key='workspace_data_compare_group').set_value('timestamp').run()
    assert not at.exception


@pytest.mark.parametrize('page', ['Project Manager', 'Smart Auto-Setup', 'Data Cleaning Studio', 'Overview', 'Explore', 'Data Quality', 'Compare', 'Data Dictionary'])
@pytest.mark.parametrize('variant', ['empty', 'text', 'duplicate'])
def test_all_routes_guard_empty_text_and_corrupt_schema(page, variant):
    at = app(page, variant)
    assert not at.exception
    if variant == 'empty':
        assert at.info
    if variant == 'duplicate':
        assert at.error


def test_page_contract_and_invalid_route():
    m = module()
    assert m.PAGES == ('Project Manager', 'Smart Auto-Setup', 'Data Cleaning Studio', 'Overview',
                       'Explore', 'Data Quality', 'Compare', 'Data Dictionary')
    with pytest.raises(ValueError, match='Unknown'):
        m.render('unowned route', {})


def test_cleaning_numeric_mean_inf_and_category_mode_with_duplicate_indices():
    m = module()
    df = pd.DataFrame({'x': [1., np.inf, np.nan, 5.],
        'cat': pd.Categorical(['a', None, 'a', 'b']), 'blank': [None] * 4}, index=[1, 1, 2, 2])
    original = df.copy(deep=True)
    out, audit = m.clean_dataframe(df, {'missing_columns': ['x', 'absent', 'cat'], 'missing_strategy': 'mean',
                                      'categorical_columns': ['cat', 'blank', 'absent'],
                                      'outlier_columns': ['absent', 'cat'], 'outlier_strategy': 'cap'})
    assert out.x.tolist() == [1., 3., 3., 5.]
    assert out['cat'].isna().sum() == 0
    assert out['blank'].isna().all()
    assert any(r.get('detail') == 'Unchanged: no observed mode' for r in audit)
    pd.testing.assert_frame_equal(df, original)
    with pytest.raises(ValueError, match='positive'):
        m.clean_dataframe(df, {'iqr_factor': 0})


def test_compare_empty_selection_missing_groups_and_invalid_aggregation():
    m = module()
    df = pd.DataFrame({'cohort': ['A', None, 'A'], 'x': [1, 2, 3]})
    result = m.compare_groups(df, 'cohort', ['x'], 'count').set_index('group')
    assert result.loc['(missing)', 'x'] == 1
    assert m.compare_groups(df, 'cohort', []).empty
    with pytest.raises(ValueError, match='Unsupported'):
        m.compare_groups(df, 'cohort', ['x'], 'invalid')


def test_compare_preserves_metric_named_group_without_output_collision():
    m = module()
    data = pd.DataFrame({'cohort': ['A', 'A', 'B'], 'group': [1., 3., 10.]})
    result = m.compare_groups(data, 'cohort', ['group'])
    assert result.columns.is_unique
    assert result['group'].tolist() == [2., 10.]
    assert result.iloc[:, 0].tolist() == ['A', 'B']


def test_compare_ui_handles_metric_names_colliding_with_internal_labels():
    at = app('Compare', extra="df['group'] = [1., 3., 5., 7.]\ndf['value'] = [2., 4., 6., 8.]\nctx.update(features=['group', 'value'])")
    assert not at.exception
    assert at.dataframe[0].value['group'].tolist() == [2., 6.]


def test_cleaning_target_confirmation_and_preview_invalidation():
    at = app('Data Cleaning Studio')
    at.multiselect(key='workspace_data_data_cleaning_studio_drop').set_value(['malicious'])
    at.button(key='workspace_data_data_cleaning_studio_preview_button').click().run()
    assert not at.exception and at.error
    assert at.button(key='workspace_data_data_cleaning_studio_apply').disabled
    at.checkbox(key='workspace_data_data_cleaning_studio_allow_target').check()
    at.button(key='workspace_data_data_cleaning_studio_preview_button').click().run()
    assert not at.exception and not at.button(key='workspace_data_data_cleaning_studio_apply').disabled
    at.checkbox(key='workspace_data_data_cleaning_studio_deduplicate').check().run()
    assert at.button(key='workspace_data_data_cleaning_studio_apply').disabled
    assert any('fresh preview' in i.value for i in at.info)


def test_explore_range_color_and_no_results_controls():
    at = app('Explore')
    at.selectbox(key='workspace_data_explore_range').set_value('packets').run()
    at.number_input(key='workspace_data_explore_min').set_value(4.)
    at.number_input(key='workspace_data_explore_max').set_value(6.)
    at.selectbox(key='workspace_data_explore_color').set_value('label').run()
    assert not at.exception and len(at.dataframe[0].value) == 2
    at.number_input(key='workspace_data_explore_min').set_value(100.).run()
    assert not at.exception and at.dataframe[0].value.empty and at.info


@pytest.mark.parametrize('extra', ["ctx['date_col'] = None", "df['timestamp'] = 'not-a-date'"])
def test_explore_line_gracefully_handles_missing_or_invalid_dates(extra):
    at = app('Explore', extra=extra)
    at.selectbox(key='workspace_data_explore_chart').set_value('Line').run()
    assert not at.exception and at.info


def test_large_scatter_is_explicitly_bounded_but_export_source_is_full():
    at = app('Explore', extra="df = pd.concat([df] * 1300, ignore_index=True)\nctx['df'] = df")
    at.selectbox(key='workspace_data_explore_chart').set_value('Scatter').run()
    assert not at.exception
    assert any('5,000' in c.value for c in at.caption)
    assert any('5,200' in c.value for c in at.caption)


def test_quality_audit_export_and_dictionary_role_filter():
    at = app('Data Quality', extra="ctx['cleaning_audit'] = [{'action': 'Test fixture audit', 'changed': 1}]")
    at.checkbox(key='workspace_data_data_quality_issues').check().run()
    assert not at.exception and len(at.get('download_button')) == 3
    at = app('Data Dictionary')
    at.multiselect(key='workspace_data_data_dictionary_roles').set_value(['Target']).run()
    assert not at.exception
    assert at.dataframe[0].value['column'].tolist() == ['malicious']


def test_compare_no_metrics_and_high_cardinality_guard():
    at = app('Compare')
    at.multiselect(key='workspace_data_compare_metrics').set_value([]).run()
    assert not at.exception and at.info
    at = app('Compare', extra="df = pd.DataFrame({'x': range(110)})\nctx.update(df=df, target=None, date_col=None, features=['x'])")
    assert not at.exception and at.info


def test_overview_without_target_uses_available_numeric_columns():
    at = app('Overview', extra="ctx.update(target=None, date_col=None, features=[])")
    assert not at.exception and len(at.get('plotly_chart')) >= 1


def test_setup_excludes_target_value_clones_and_profiles_empty_columns():
    m = module()
    df = fixture()
    df['disguised'] = df['malicious']
    assert 'disguised' not in m.smart_setup(df)['features']
    assert m.data_dictionary(pd.DataFrame(columns=['x'])).iloc[0]['missing_pct'] == 0.


@pytest.mark.parametrize('page', ['Project Manager', 'Smart Auto-Setup', 'Data Cleaning Studio', 'Overview', 'Explore', 'Data Quality', 'Compare', 'Data Dictionary'])
def test_real_workspace_feature_csv_renders_without_invented_data(page):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    path = Path(__file__).resolve().parents[1] / 'report_data' / 'workspace_features.csv'
    if not path.exists():
        pytest.skip('Real feature artifact not supplied in this checkout')
    script = f'''
import pandas as pd
from cyberlab.workspace_data import render
path = {str(path)!r}
df = pd.read_csv(path)
features = [c for c in df.select_dtypes(include='number') if c not in ('malicious', 'time_group')]
ctx = dict(df=df, raw_df=df.copy(), target='malicious', date_col='timestamp', features=features,
project_name='Real feature artifact', source_name=path, mode='research', fingerprint='real-artifact-test', cleaning_audit=[])
render({page!r}, ctx)
'''
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    assert at.subheader[0].value == page
