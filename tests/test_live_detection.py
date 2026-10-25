from pathlib import Path
import importlib
import json
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'cyberlab/live_detection.py').exists(), 'Detection mode is missing'
    return importlib.import_module('cyberlab.live_detection')


def test_published_detector_scores_real_features_at_frozen_threshold():
    m=module()
    assert (ROOT/'report_data'/'detector.skops').exists(), 'Cloud-safe detector export is missing'
    detector,meta=m.load_detector(ROOT)
    frame=pd.read_csv(ROOT/'report_data'/'workspace_features.csv')
    scored=m.score_windows(frame,detector,meta)
    assert len(scored)==len(frame)
    assert {'detector_score','suspicious','source_ip'}<=set(scored)
    assert scored.source_ip.isna().all()
    assert np.isfinite(scored.detector_score).all()
    assert (scored.suspicious==(scored.detector_score>=meta['threshold'])).all()
    assert meta['model_name']=='extra_trees'
    assert meta['threshold']==pytest.approx(0.6365746297698786)
    with pytest.raises(ValueError,match='Missing model features'):
        m.score_windows(frame.drop(columns=['flow_count']),detector,meta)


def test_live_query_filters_recent_events_and_labels_history_explicitly():
    m=module()
    q=m.build_live_query(limit=50,lookback_minutes=15,include_history=False)
    assert q['query']['range']['@timestamp']['gte']=='now-15m'
    assert q['size']==50
    assert m.build_live_query(50,15,True)['query']=={'match_all':{}}
    assert m.INDEX=='cyber-features-spark-stream-v1'
    with pytest.raises(ValueError):m.build_live_query(10001,15,False)


def test_source_hosts_are_only_reported_if_observed_in_the_input():
    m=module()
    frame=pd.DataFrame({'@timestamp':['2017-07-06T12:00:00Z'], 'source':[{'ip':'192.0.2.50'}],'flow_count':[4]})
    flat=m.flatten_live_events([{'_id':'event','_source':frame.iloc[0].to_dict()}])
    assert flat.iloc[0].source_ip=='192.0.2.50'
    assert flat.iloc[0].event_id=='event'
    empty=m.flatten_live_events([])
    assert empty.empty


def test_detection_ui_has_run_and_live_mode_without_replacing_benchmark():
    from streamlit.testing.v1 import AppTest
    m=module()
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
    controls=app.get('button_group')
    assert any(x.key=='cyber_detection_mode' for x in controls)
    control=next(x for x in controls if x.key=='cyber_detection_mode')
    control.set_value('Run data').run()
    assert not app.exception
    assert app.title[0].value=='Run data — threat scoring'
    assert any('not confirmed attackers' in x.value.lower() for x in app.warning)


def test_unlabeled_flow_upload_reuses_ingestion_and_retains_observed_ip():
    m=module()
    row={'Source IP':'192.0.2.1','Destination IP':'198.51.100.2',
         'Source Port':1234,'Destination Port':443,'Protocol':6,
         'Timestamp':'6/7/2017 10:00:00','Flow Duration':100.,
         'Total Fwd Packets':3,'Total Backward Packets':2,
         'Total Length of Fwd Packets':1000.,'Total Length of Bwd Packets':500.,
         'Flow Bytes/s':15000.}
    windows,audit=m.prepare_input(pd.DataFrame([row,row]).to_csv(index=False).encode(),'CICIDS flow CSV')
    assert len(windows)==1
    assert audit['duplicate_rows']==1
    assert audit['labels_available'] is False
    assert windows.source_ip.tolist()==['192.0.2.1']
    assert windows.flow_count.tolist()==[1]
    assert windows.bytes_total.tolist()==[1500.]
    model,meta=m.load_detector(ROOT)
    assert m.score_windows(windows,model,meta).source_ip.tolist()==['192.0.2.1']


def test_packaged_detector_denies_unknown_types(monkeypatch):
    import skops.io
    monkeypatch.setattr(skops.io,'get_untrusted_types',lambda **kwargs:['os.system'])
    with pytest.raises(ValueError,match='unsupported Python types'):
        module().load_detector(ROOT)
