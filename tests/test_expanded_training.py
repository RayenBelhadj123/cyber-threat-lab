import hashlib
import importlib
import json
from pathlib import Path
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def api():
    assert (ROOT/'cyberlab/expanded_training.py').exists(), 'Expanded training pipeline is missing'
    return importlib.import_module('cyberlab.expanded_training')


def test_inventory_uses_verified_manifest_files_and_excludes_old_thursday(tmp_path):
    for dataset,folder,manifest,name in (
        ('CICIDS2017','cicids2017_full','cicids2017_full_manifest.json','Monday-WorkingHours.pcap_ISCX.csv'),
        ('CICDDoS2019','cicddos2019_extension','cicddos2019_extension/source_manifest.json','01-12_DrDoS_NTP.csv')):
        path=tmp_path/'data'/folder/name
        path.parent.mkdir(parents=True,exist_ok=True)
        content=b'fixture\n1\n'
        path.write_bytes(content)
        doc={'output':'data/'+folder,'files':[{'name':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'rows':1}]}
        destination=tmp_path/'data'/manifest
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(doc))
    old=tmp_path/'data/raw/Thursday-WorkingHours.pcap_ISCX.csv'
    old.parent.mkdir(parents=True)
    old.write_text('overlapping file must not be included')
    entries=api().input_inventory(tmp_path)
    assert len(entries)==2
    assert all(entry['path']!=old for entry in entries)
    assert {entry['dataset'] for entry in entries}=={'CICIDS2017','CICDDoS2019'}
    entries[0]['path'].write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum|size'):
        api().input_inventory(tmp_path)


def flow_row(port=80):
    return {'Source IP':'192.0.2.1','Destination IP':'198.51.100.2','Source Port':1000,
            'Destination Port':port,'Protocol':6,'Timestamp':'3/7/2017 09:00:00','Flow Duration':100.,
            'Total Fwd Packets':2,'Total Backward Packets':1,'Total Length of Fwd Packets':200.,
            'Total Length of Bwd Packets':100.,'Flow Bytes/s':300.,'Label':'BENIGN' if port==80 else 'Attack'}


def test_preparation_cleans_each_complete_file_and_deduplicates_across_files(tmp_path):
    entries=[]
    for dataset,name,rows in [
        ('CICIDS2017','Monday-A.csv',[flow_row(),flow_row(),flow_row(81)]),
        ('CICIDS2017','Monday-B.csv',[flow_row(),flow_row(82)]),
        ('CICDDoS2019','01-12_NTP.csv',[flow_row(),flow_row(81)])]:
        path=tmp_path/name
        pd.DataFrame(rows).to_csv(path,index=False)
        entries.append({'dataset':dataset,'name':name,'path':path,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'expected_rows':len(rows)})
    windows,lineage=api().prepare_windows(entries,tmp_path/'prepared')
    assert lineage['input_rows']==7
    assert lineage['valid_rows']==5
    assert lineage['duplicate_rows']==2
    assert len(windows)==3
    assert windows.window_id.is_unique
    assert set(windows.capture_group)=={'CICIDS2017:Monday','CICDDoS2019:01-12'}
    assert int(windows.flow_count.sum())==5
    cached,cached_lineage=api().prepare_windows(entries,tmp_path/'prepared')
    pd.testing.assert_frame_equal(windows,cached,check_dtype=False)
    assert cached_lineage==lineage
    changed=[dict(entry) for entry in entries]
    changed[0]['expected_rows']=0
    with pytest.raises(ValueError,match='row count|input changed'):
        api().prepare_windows(changed,tmp_path/'prepared')


def capture_windows():
    from datetime import datetime, timedelta, timezone
    from cyberlab.pipeline import FEATURES
    groups=['CICIDS2017:'+day for day in ('Monday','Tuesday','Wednesday','Thursday','Friday')]+['CICDDoS2019:01-12','CICDDoS2019:03-11']
    rows=[]
    for group in groups:
        for i in range(12):
            rows.append({'capture_group':group,'timestamp':datetime(2017,7,3,12,tzinfo=timezone.utc)+timedelta(minutes=i),
                         'window_id':group+str(i),'source_ip':'192.0.2.1','malicious':i%3==0,
                         'reference_label':'Attack' if i%3==0 else 'BENIGN',
                         **{feature:float(1+i%3) if feature!='failure_ratio' else 0. for feature in FEATURES}})
    return pd.DataFrame(rows)


def test_capture_split_holds_whole_friday_and_portmap_and_never_reuses_test_in_cv():
    import numpy as np
    windows=capture_windows()
    split=api().create_capture_splits(windows)
    groups=split['groups']
    dev,test=split['dev_indices'],split['test_indices']
    assert set(groups[test])=={'CICIDS2017:Friday','CICDDoS2019:03-11'}
    assert not set(groups[dev])&set(groups[test])
    covered=[]
    for fold in split['inner_folds']:
        train,validation=fold['train_indices'],fold['validation_indices']
        assert not set(groups[train])&set(groups[validation])
        assert not set(test)&set(np.r_[train,validation])
        assert windows.iloc[train].malicious.nunique()==2
        assert windows.iloc[validation].malicious.nunique()==2
        covered.extend(validation.tolist())
    assert sorted(covered)==sorted(dev.tolist())


def test_benchmark_accepts_prepared_capture_windows_without_thursday_metadata(tmp_path):
    from cyberlab.benchmark import run_model_benchmark
    windows=capture_windows()
    prepared={'windows':windows,'lineage':{'input_rows':5000,'valid_rows':4900,'row_labels':{'BENIGN':4000,'Attack':900}},
              'split':api().create_capture_splits(windows),
              'description':'All eleven complete input CSVs; full-week and attack-extension evidence.',
              'caveats':['Capture-disjoint is not host-disjoint or globally chronological.']}
    report=run_model_benchmark(None,tmp_path/'report',prepared=prepared,model_names=('prior_baseline','extra_trees'))
    assert report['dataset']['input_rows']==5000
    assert report['dataset']['valid_flows']==4900
    assert report['split']['temporal_policy']=='capture_disjoint_mixed_source_clocks_not_global_forward'
    assert report['selection']['test_used_for_selection'] is False
    assert report['dataset']['description']==prepared['description']
    html=(tmp_path/'report/report.html').read_text(encoding='utf-8')
    assert prepared['description'] in html
    assert 'Full available Thursday slice' not in html
    assert 'All positives still come from one source host' not in html


@pytest.mark.parametrize('corruption',['heldout_in_cv','validation_in_train','missing_oof','duplicate_oof','invalid_index','group_alignment'])
def test_prepared_benchmark_rejects_invalid_splits_before_fitting(tmp_path,monkeypatch,corruption):
    import numpy as np
    from cyberlab import benchmark
    windows=capture_windows()
    split=api().create_capture_splits(windows)
    fold=split['inner_folds'][0]
    if corruption=='heldout_in_cv':
        fold['train_indices']=np.append(fold['train_indices'],split['test_indices'][0])
    elif corruption=='validation_in_train':
        fold['train_indices']=np.append(fold['train_indices'],fold['validation_indices'][0])
    elif corruption=='missing_oof':
        split['inner_folds']=split['inner_folds'][1:]
    elif corruption=='duplicate_oof':
        split['inner_folds'].append(dict(fold))
    elif corruption=='invalid_index':
        fold['train_indices']=np.append(fold['train_indices'],-1)
    else:
        split['groups']=np.roll(split['groups'],1)
    def must_not_fit(*args,**kwargs):
        pytest.fail('Invalid split reached model fitting')
    monkeypatch.setattr(benchmark,'fit_candidate',must_not_fit)
    prepared={'windows':windows,'lineage':{'valid_rows':84,'row_labels':{}},'split':split}
    with pytest.raises(ValueError,match='split|indices|group|coverage'):
        benchmark.run_model_benchmark(None,tmp_path/'report',prepared=prepared,model_names=('prior_baseline',))
    assert not (tmp_path/'report').exists()


def test_loader_counts_structurally_blank_csv_rows_separately(tmp_path):
    from cyberlab.pipeline import load_flows
    frame=pd.DataFrame([flow_row(),{key:None for key in flow_row()},flow_row(81)])
    path=tmp_path/'blank_records.csv'
    frame.to_csv(path,index=False)
    flows,lineage=load_flows(path)
    assert lineage['input_rows']==2
    assert lineage['blank_rows']==1
    assert lineage['invalid_rows']==0
    assert len(flows)==2


@pytest.mark.parametrize('name',['extra_trees','random_forest','hist_gradient_boosting','logistic_regression','isolation_forest','prior_baseline'])
def test_supported_selected_detector_exports_without_pickle_and_matches_scores(tmp_path,name):
    import numpy as np
    from cyberlab.benchmark import fit_candidate,_score_model
    from cyberlab.pipeline import FEATURES
    from cyberlab import live_detection
    frame=capture_windows()
    x=frame[FEATURES].to_numpy(dtype=float)
    y=frame.malicious.to_numpy(dtype=bool)
    model,_,_,_=fit_candidate(name,x,y,x,y)
    assert hasattr(live_detection,'export_detector'), 'Model-independent safe detector export is missing'
    live_detection.export_detector(model,name,0.5,tmp_path,scope='Unit-test fixture only')
    restored,meta=live_detection.load_detector(tmp_path)
    actual=live_detection.score_windows(frame,restored,meta)
    np.testing.assert_allclose(actual.detector_score,_score_model(name,model,x),atol=1e-12)
    assert meta['model_name']==name


def test_neural_detector_uses_safe_numpy_weights_without_torch_at_inference(tmp_path,monkeypatch):
    import builtins
    import numpy as np
    from cyberlab.benchmark import fit_candidate,_score_model
    from cyberlab.pipeline import FEATURES
    from cyberlab import live_detection
    frame=capture_windows()
    x=frame[FEATURES].to_numpy(dtype=float)
    y=frame.malicious.to_numpy(dtype=bool)
    model,_,_,_=fit_candidate('pytorch_mlp',x,y,x,y,fixed_epochs=2)
    expected=_score_model('pytorch_mlp',model,x)
    live_detection.export_detector(model,'pytorch_mlp',0.5,tmp_path,scope='Neural fixture only')
    assert (tmp_path/'report_data/detector.npz').exists()
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='torch' or name.startswith('torch.'):
            raise AssertionError('Cloud inference must not import PyTorch.')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    restored,metadata=live_detection.load_detector(tmp_path)
    actual=live_detection.score_windows(frame,restored,metadata).detector_score.to_numpy()
    assert metadata['format']=='numpy_mlp'
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
