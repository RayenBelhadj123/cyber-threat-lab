import importlib
import json
from pathlib import Path
import runpy
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def release_source(tmp_path):
    from cyberlab.expanded_training import create_capture_splits
    from cyberlab.benchmark import run_model_benchmark
    windows=runpy.run_path(str(ROOT/'tests/test_expanded_training.py'))['capture_windows']()
    prepared={'windows':windows,'lineage':{'input_rows':5000,'valid_rows':4900,'row_labels':{'BENIGN':4000,'Attack':900}},
              'split':create_capture_splits(windows),'description':'Expanded fixture input only.',
              'caveats':['Capture-disjoint fixture; not a production evaluation.']}
    source=tmp_path/'source'
    run_model_benchmark(None,source,prepared=prepared,model_names=('prior_baseline',))
    return source,windows


def test_snapshot_refuses_to_overwrite_existing_release(release_source,tmp_path):
    from cyberlab.reporting_release import build_snapshot
    source,windows=release_source
    stage=tmp_path/'release'
    build_snapshot(source,stage,windows,preview_limit=20)
    before={p.relative_to(stage):p.read_bytes() for p in stage.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='fresh|empty|exists'):
        build_snapshot(source,stage,windows,preview_limit=20)
    assert before=={p.relative_to(stage):p.read_bytes() for p in stage.rglob('*') if p.is_file()}


@pytest.mark.parametrize('bad_score',[float('nan'),float('inf'),float('-inf')])
def test_snapshot_rejects_nonfinite_saved_scores(release_source,tmp_path,bad_score):
    from cyberlab.reporting_release import build_snapshot
    source,windows=release_source
    path=source/'held_out_test_predictions.csv'
    predictions=pd.read_csv(path)
    predictions.loc[0,'score_prior_baseline']=bad_score
    predictions.to_csv(path,index=False)
    stage=tmp_path/'release'
    with pytest.raises(ValueError,match='finite|score'):
        build_snapshot(source,stage,windows,preview_limit=20)
    assert not (stage/'release-verification.json').exists()


def test_snapshot_rejects_heldout_rows_in_development_preview(release_source,tmp_path):
    from cyberlab.reporting_release import build_snapshot
    source,windows=release_source
    path=source/'development_oof_predictions.csv'
    dev=pd.read_csv(path)
    test=pd.read_csv(source/'held_out_test_predictions.csv')
    # Substituting an identity leaves the file length and numeric scores intact.
    dev.loc[0,'window_id']=test.loc[0,'window_id']
    dev.to_csv(path,index=False)
    stage=tmp_path/'release'
    with pytest.raises(ValueError,match='overlap|disjoint|identit'):
        build_snapshot(source,stage,windows,preview_limit=20)
    assert not (stage/'release-verification.json').exists()


@pytest.mark.parametrize('corruption',['unknown','duplicate','capture_swap'])
def test_snapshot_requires_complete_capture_disjoint_identities(release_source,tmp_path,corruption):
    from cyberlab.reporting_release import build_snapshot
    source,windows=release_source
    dev_path=source/'development_oof_predictions.csv'
    test_path=source/'held_out_test_predictions.csv'
    dev,test=pd.read_csv(dev_path),pd.read_csv(test_path)
    if corruption=='unknown':
        dev.loc[0,'window_id']='not-a-prepared-window'
    elif corruption=='duplicate':
        dev.loc[0,'window_id']=dev.loc[1,'window_id']
    else:
        dev.loc[0,'window_id'],test.loc[0,'window_id']=test.loc[0,'window_id'],dev.loc[0,'window_id']
    dev.to_csv(dev_path,index=False)
    test.to_csv(test_path,index=False)
    stage=tmp_path/'release'
    with pytest.raises(ValueError,match='identit|capture|unique'):
        build_snapshot(source,stage,windows,preview_limit=20)
    assert not (stage/'release-verification.json').exists()


@pytest.mark.parametrize('rewrite_groups',[False,True])
def test_snapshot_rejects_whole_capture_partition_swap(release_source,tmp_path,rewrite_groups):
    from cyberlab.reporting_release import build_snapshot
    source,windows=release_source
    dev_path=source/'development_oof_predictions.csv'
    test_path=source/'held_out_test_predictions.csv'
    dev,test=pd.read_csv(dev_path),pd.read_csv(test_path)
    dev_mask=dev.time_group.eq('CICIDS2017:Monday')
    test_mask=test.time_group.eq('CICIDS2017:Friday')
    dev_ids=dev.loc[dev_mask,'window_id'].to_numpy().copy()
    test_ids=test.loc[test_mask,'window_id'].to_numpy().copy()
    dev.loc[dev_mask,'window_id']=test_ids
    test.loc[test_mask,'window_id']=dev_ids
    if rewrite_groups:
        dev.loc[dev_mask,'time_group']='CICIDS2017:Friday'
        test.loc[test_mask,'time_group']='CICIDS2017:Monday'
    dev.to_csv(dev_path,index=False)
    test.to_csv(test_path,index=False)
    stage=tmp_path/'release'
    with pytest.raises(ValueError,match='capture|partition|group'):
        build_snapshot(source,stage,windows,preview_limit=20)
    assert not (stage/'release-verification.json').exists()


@pytest.mark.parametrize('partition',['development_oof_predictions.csv','held_out_test_predictions.csv'])
def test_snapshot_rejects_labels_swapped_between_window_identities(release_source,tmp_path,partition):
    from cyberlab.reporting_release import build_snapshot
    source,windows=release_source
    path=source/partition
    frame=pd.read_csv(path)
    positive=frame.index[frame.malicious][0]
    negative=frame.index[~frame.malicious][0]
    frame.loc[positive,'malicious']=False
    frame.loc[negative,'malicious']=True
    frame.to_csv(path,index=False)
    stage=tmp_path/'release'
    with pytest.raises(ValueError,match='label'):
        build_snapshot(source,stage,windows,preview_limit=20)
    assert not (stage/'release-verification.json').exists()


@pytest.mark.parametrize('summary',['model','heldout','all_models'])
@pytest.mark.parametrize('field',['precision','confusion_matrix'])
def test_snapshot_rejects_metrics_that_disagree_with_predictions(release_source,tmp_path,summary,field):
    from cyberlab.reporting_release import build_snapshot
    source,windows=release_source
    path=source/'metrics.json'
    metrics=json.loads(path.read_text())
    selected=metrics['selection']['model']
    target={'model':metrics['models'][selected]['test_metrics'],
            'heldout':metrics['held_out_test']['metrics'],
            'all_models':metrics['held_out_test']['model_test_metrics'][selected]}[summary]
    if field=='precision':
        target[field]=1.0
    else:
        target[field]={'tn':16,'fp':0,'fn':0,'tp':8}
    path.write_text(json.dumps(metrics))
    stage=tmp_path/'release'
    with pytest.raises(ValueError,match='metric|summary'):
        build_snapshot(source,stage,windows,preview_limit=20)
    assert not (stage/'release-verification.json').exists()


@pytest.mark.parametrize('candidates',[('prior_baseline','extra_trees'),('pytorch_mlp',)])
def test_release_snapshot_contains_full_metrics_safe_detector_and_dev_only_preview(tmp_path,candidates):
    from cyberlab.expanded_training import create_capture_splits
    from cyberlab.benchmark import run_model_benchmark
    windows=runpy.run_path(str(ROOT/'tests/test_expanded_training.py'))['capture_windows']()
    split=create_capture_splits(windows)
    prepared={'windows':windows,'lineage':{'input_rows':5000,'valid_rows':4900,'row_labels':{'BENIGN':4000,'Attack':900}},
              'split':split,'description':'Expanded fixture input only.','caveats':['Capture-disjoint fixture; not a production evaluation.']}
    source=tmp_path/'source'
    run_model_benchmark(None,source,prepared=prepared,model_names=candidates)
    assert (ROOT/'cyberlab/reporting_release.py').exists(),'Verified snapshot builder is missing'
    module=importlib.import_module('cyberlab.reporting_release')
    stage=tmp_path/'release'
    result=module.build_snapshot(source,stage,windows,preview_limit=20)
    destination=stage/'report_data'
    metrics=json.loads((destination/'metrics.json').read_text())
    preview=pd.read_csv(destination/'workspace_features.csv')
    provenance=json.loads((destination/'workspace_provenance.json').read_text())
    assert metrics['dataset']['input_rows']==5000
    assert 'positive_source_ips' not in metrics['dataset']
    assert len(preview)==20
    assert not {'source_ip','window_id','reference_label'}.intersection(preview.columns)
    assert not set(preview.time_group)&set(windows.iloc[split['test_indices']].capture_group)
    assert provenance['rows']==20 and provenance['model_input_rows']==5000
    assert result['heldout_score_max_difference']<(1e-6 if candidates==('pytorch_mlp',) else 1e-12)
    assert result['heldout_flag_mismatches']==0
