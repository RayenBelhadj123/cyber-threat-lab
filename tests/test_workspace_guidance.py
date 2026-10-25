"""Additional regression checks for the new guidance and inference boundaries."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from cyberlab.workspace_guidance import code_preview, assistant_answer

ROOT=Path(__file__).resolve().parents[1]


def context():
    df=pd.DataFrame({'x':[1.,2.,3.],'group':['a','a','b'],'malicious':[False,False,True]})
    return dict(df=df,features=['x'],target='malicious',source_name='test fixture',fingerprint='fixture',project_name='Test')


@pytest.mark.parametrize('recipe',['Dataset profile','Numeric correlation','Grouped summary'])
def test_generated_code_is_parseable_and_only_a_preview(recipe):
    code=code_preview(context(),recipe,'group')
    ast.parse(code)
    assert 'input.csv' in code
    assert not any(token in code for token in ('exec(', 'eval(', 'subprocess', 'pickle', 'joblib'))


def test_code_preview_refuses_unknown_recipe_or_column():
    with pytest.raises(ValueError):code_preview(context(),'execute')
    with pytest.raises(ValueError):code_preview(context(),'Grouped summary','missing')


@pytest.mark.parametrize('question',['attacks live ip','missing values','overfit model score','future forecast','execute code','help overview'])
def test_local_assistant_uses_observed_evidence_and_no_untrained_metrics(question):
    answer=assistant_answer(context(),question)
    assert '3 rows × 3 columns' in answer
    assert 'No current matching workspace model' in answer
    assert 'Local rule-based assistance only' in answer
    assert '99%' not in answer


def test_guidance_page_actions_work_through_real_entrypoint():
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=45).run()
    app.toggle(key='full_analytics_workspace').set_value(True).run()
    for page,button,download in (
        ('Code Lab','workspace_code_generate','workspace_code_download'),
        ('Domain Review','workspace_domain_save','workspace_domain_download'),
        ('Assistant','workspace_assistant_answer','workspace_assistant_download')):
        app.radio(key='workspace_page').set_value(page).run()
        app.button(key=button).click().run()
        assert not app.exception and not app.error,page
        assert any(x.key==download for x in app.get('download_button'))


def test_detector_rejects_tampering_and_nonfinite_features(tmp_path):
    from cyberlab.live_detection import load_detector,score_windows
    model,meta=load_detector(ROOT)
    base=tmp_path/'report_data'
    base.mkdir()
    (base/'detector.skops').write_bytes(b'not a trusted model')
    (base/'detector_metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):
        load_detector(tmp_path)
    frame=pd.read_csv(ROOT/'report_data/workspace_features.csv').head(2)
    frame.loc[0,'flow_count']=np.inf
    with pytest.raises(ValueError,match='finite'):
        score_windows(frame,model,meta)


def test_live_ui_is_paused_until_explicitly_started(monkeypatch):
    import cyberlab.live_detection as detection
    def forbidden(*args,**kwargs):
        raise AssertionError('A paused live page must not query Elasticsearch.')
    monkeypatch.setattr(detection,'fetch_live_windows',forbidden)
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=45).run()
    app.get('button_group')[0].set_value('Live data').run()
    assert not app.exception
    assert app.toggle(key='detection_live_active').value is False
    assert any('No network query runs while paused' in x.value for x in app.info)
