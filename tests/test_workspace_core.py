from pathlib import Path
import importlib
import io
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'cyberlab/workspace.py').exists(), 'Analytics workspace core is missing'
    return importlib.import_module('cyberlab.workspace')


def test_reference_feature_inventory_has_every_reference_route():
    m=module()
    names=set(m.PAGE_NAMES)
    required={'Start Here','Project Manager','Smart Auto-Setup','Data Cleaning Studio','Overview','Explore','Data Quality','Compare','ML Lab','Evaluation','Unsupervised Lab','Reinforcement Lab','Forecast','Conclusion','Domain Review','Code Lab','OLAP & Export','Executive Summary','Data Dictionary','Scenario Simulator','Production Readiness','Experiment Tracker','Model Registry','Data Pipeline','Big Data Readiness','Business Impact','Fit Diagnostics','Prediction Page','Report Generator','Model Save / Load','Model Explainability','Assistant'}
    assert names==required
    assert len(names)==len(m.PAGE_NAMES)
    assert len(m.APP_MODES)==4
    assert 'Cyber Threat Detection' in m.TEMPLATES


def test_upload_limits_and_smart_roles_exclude_scores_identifiers_and_labels():
    m=module()
    df=m.read_workspace_csv(b'timestamp,malicious,flow_count,bytes_total,score_model,source_ip\n2017-07-06 12:00:00,0,2,100,0.1,192.0.2.1\n2017-07-06 12:01:00,1,9,900,0.9,192.0.2.2\n')
    roles=m.suggest_roles(df)
    assert roles['target']=='malicious'
    assert roles['date_col']=='timestamp'
    assert set(roles['features'])=={'flow_count','bytes_total'}
    assert roles['task']=='Classification'
    with pytest.raises(ValueError,match='size'):
        m.read_workspace_csv(b'x'*100,max_bytes=50)
    with pytest.raises(ValueError,match='rows'):
        m.read_workspace_csv(b'a\n1\n2\n3\n',max_rows=2)


def test_default_workspace_uses_real_feature_rows_not_prediction_scores():
    m=module()
    df=m.load_default_workspace(ROOT)
    assert len(df)==16174
    assert int(df.malicious.sum())==68
    assert {'timestamp','time_group','flow_count','bytes_total'}<=set(df.columns)
    assert not any(c.startswith(('score_','flagged_')) for c in df.columns)
    assert not {'source_ip','window_id','split'}&set(df.columns)
    f=m.dataset_fingerprint(df,'malicious','timestamp',['flow_count'])
    assert f!=m.dataset_fingerprint(df,'malicious','timestamp',['bytes_total'])


def test_main_app_can_open_workspace_and_create_report(monkeypatch):
    from streamlit.testing.v1 import AppTest
    m=module()
    monkeypatch.delenv('CYBERLAB_BENCHMARK_DIR',raising=False)
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
    toggles=[x for x in app.sidebar.toggle if x.label=='Full analytics workspace']
    assert toggles, 'Full workspace toggle is missing'
    toggles[0].set_value(True).run()
    assert not app.exception and not app.error
    assert app.title[0].value=='Start Here'
    app.sidebar.radio[0].set_value('Report Generator').run()
    assert not app.exception and not app.error
    assert len(app.get('download_button'))>=3


def test_workspace_report_handles_categorical_only_uploads_and_escapes_html():
    m=module()
    df=pd.DataFrame({'team':['blue','red'],'label':['safe','attack']})
    ctx={'df':df,'raw_df':df,'target':'label','date_col':None,'features':[],
         'project_name':'<script>bad()</script>','source_name':'uploaded.csv','mode':'Beginner',
         'fingerprint':'fixture','cleaning_audit':[]}
    md,html=m.workspace_report(ctx)
    assert '<script>' not in html
    assert 'Governance' in md or 'independent' in md
    assert 'No numeric columns' in md


def test_workspace_search_ranks_page_keywords_and_tolerates_typo():
    m=module()
    assert hasattr(m,'search_workspace'), 'Ranked workspace search is missing'
    pages,cols=m.search_workspace('predicton',['flow_count'],m.PAGE_NAMES)
    assert 'Prediction Page' in pages[:3]
    pages,cols=m.search_workspace('f1',['flow_count'],m.PAGE_NAMES)
    assert 'Evaluation' in pages
    pages,cols=m.search_workspace('flow',['flow_count','bytes_total'],m.PAGE_NAMES)
    assert cols==['flow_count']


def test_workspace_reads_expanded_preview_scope_from_provenance(tmp_path):
    import json
    target=tmp_path/'report_data'
    target.mkdir()
    record={'workspace_label':'Expanded development-only feature preview','scope':'Full model uses all verified raw rows; this is only a bounded interactive preview.','rows':50000,'model_input_rows':4610049}
    (target/'workspace_provenance.json').write_text(json.dumps(record),encoding='utf-8')
    m=module()
    assert hasattr(m,'default_workspace_provenance'), 'Default dataset scope is still hardcoded'
    assert m.default_workspace_provenance(tmp_path)==record
