"""Build a privacy-reduced release and verify inference parity before activation."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_heldout_metrics(metrics, test):
    """Reconcile published summaries with saved scores at frozen thresholds."""
    from .benchmark import evaluate_scores

    def agrees(saved, actual):
        if isinstance(actual, dict):
            return (isinstance(saved, dict) and saved.keys() == actual.keys()
                    and all(agrees(saved[key], value) for key, value in actual.items()))
        if actual is None or isinstance(actual, int):
            return saved == actual
        return (isinstance(saved, (int, float)) and np.isfinite(saved)
                and np.isclose(saved, actual, rtol=1e-12, atol=1e-12))

    selected=metrics['selection']['model']
    heldout=metrics['held_out_test']
    if (heldout['selected_model']!=selected
            or set(heldout['model_test_metrics'])!=set(metrics['models'])):
        raise ValueError('Held-out summary model identities disagree with the benchmark.')
    for name,record in metrics['models'].items():
        threshold=record['threshold_from_dev_oof_max_f1']
        if not np.isfinite(threshold):
            raise ValueError('Saved metric threshold must be finite.')
        actual=evaluate_scores(test.malicious,test[f'score_{name}'],threshold)
        summaries=[record['test_metrics'],heldout['model_test_metrics'][name]]
        if name==selected:
            summaries.append(heldout['metrics'])
        if not all(agrees(saved,actual) for saved in summaries):
            raise ValueError(f'Saved held-out metrics disagree with prediction evidence for {name}.')


def build_snapshot(source, stage, windows, preview_limit=50_000):
    import joblib
    from .pipeline import FEATURES
    from .live_detection import export_detector,load_detector,score_windows
    from .reporting import load_benchmark
    source,stage=Path(source),Path(stage)
    if stage.exists() and (not stage.is_dir() or any(stage.iterdir())):
        raise ValueError('Release staging requires a fresh or empty directory; existing releases are never overwritten.')
    destination=stage/'report_data'
    destination.mkdir(parents=True,exist_ok=True)
    source_bytes=(source/'metrics.json').read_bytes()
    full=json.loads(source_bytes)
    metrics=copy.deepcopy(full)
    metrics['dataset'].pop('positive_source_ips',None)
    metrics['reporting_snapshot']={'source_metrics_sha256':hashlib.sha256(source_bytes).hexdigest(),
                                   'source_artifact_directory':source.name,
                                   'privacy':'No host IPs, window IDs or row-level attack-label strings in public prediction/feature tables.'}
    predictions={}
    for name in ('development_oof_predictions.csv','held_out_test_predictions.csv'):
        frame=pd.read_csv(source/name)
        predictions[name]=frame
    dev_ids=set(predictions['development_oof_predictions.csv'].window_id)
    test_ids=set(predictions['held_out_test_predictions.csv'].window_id)
    if dev_ids & test_ids:
        raise ValueError('Development and held-out window identities overlap; release evidence must be disjoint.')
    for frame in (windows,*predictions.values()):
        if frame.window_id.isna().any() or not frame.window_id.is_unique:
            raise ValueError('Prepared and prediction window identities must be non-null and unique.')
    if dev_ids | test_ids != set(windows.window_id):
        raise ValueError('Prediction identities must cover all prepared windows exactly once.')
    development=windows.loc[windows.window_id.isin(dev_ids)].copy()
    heldout=windows.loc[windows.window_id.isin(test_ids)]
    if set(development.capture_group) & set(heldout.capture_group):
        raise ValueError('Development and held-out capture identities must be disjoint.')
    for frame in predictions.values():
        aligned=frame[['window_id','time_group','malicious']].merge(
            windows[['window_id','capture_group','malicious']],on='window_id',
            validate='one_to_one',suffixes=('_saved','_prepared'))
        if not aligned.time_group.eq(aligned.capture_group).all():
            raise ValueError('Saved prediction groups disagree with prepared capture identities.')
        if (not aligned.malicious_saved.isin([0,1]).all()
                or not aligned.malicious_prepared.isin([0,1]).all()
                or not aligned.malicious_saved.eq(aligned.malicious_prepared).all()):
            raise ValueError('Saved prediction labels disagree with prepared window identities.')
    _validate_heldout_metrics(metrics,predictions['held_out_test_predictions.csv'])
    # This builder releases the expanded experiment, whose capture policy is frozen.
    # Disjointness alone cannot detect a whole-capture partition swap.
    from .expanded_training import create_capture_splits
    frozen_split=create_capture_splits(windows)
    if (dev_ids!=set(windows.iloc[frozen_split['dev_indices']].window_id)
            or test_ids!=set(windows.iloc[frozen_split['test_indices']].window_id)):
        raise ValueError('Prediction partition membership violates the frozen expanded capture split.')
    for name,frame in predictions.items():
        public=frame.drop(columns=['source_ip','window_id','reference_label'],errors='ignore')
        assert {'source_ip','window_id','reference_label'}.isdisjoint(public.columns)
        public.to_csv(destination/name,index=False,lineterminator='\n')
    (destination/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    selected=metrics['selection']['model']
    bundle_path=source/'models.joblib'
    # This is the trusted local training output, never an uploaded pickle.
    bundle=joblib.load(bundle_path)
    if bundle['selected_model']!=selected or bundle['features']!=FEATURES:
        raise ValueError('Training bundle and saved evidence disagree.')
    threshold=metrics['models'][selected]['threshold_from_dev_oof_max_f1']
    if threshold!=bundle['thresholds'][selected]:
        raise ValueError('Training bundle threshold disagrees with frozen OOF threshold.')
    scope=(f"Expanded detector: {metrics['dataset']['input_rows']:,} complete raw data rows processed; "
           f"{metrics['dataset']['valid_flows']:,} valid unique flows; {metrics['dataset']['window_count']:,} model windows. "
           'Model fitted on development captures only; Friday and Portmap/03-11 remain held-out evidence. Suspicious sources are not confirmed attackers.')
    model_metadata=export_detector(bundle['models'][selected],selected,threshold,stage,scope=scope,
                                   source_bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest())
    model_metadata['benchmark_input_rows']=metrics['dataset']['input_rows']
    (destination/'detector_metadata.json').write_text(json.dumps(model_metadata,indent=2),encoding='utf-8')
    restored,metadata=load_detector(stage)
    test=predictions['held_out_test_predictions.csv']
    joined=test[['window_id',f'score_{selected}',f'flagged_{selected}']].merge(
        windows[['window_id',*FEATURES]],on='window_id',how='left',validate='one_to_one')
    if len(joined)!=len(test) or joined[FEATURES].isna().any().any():
        raise ValueError('Held-out feature/prediction identities do not match.')
    scores=[]
    for start in range(0,len(joined),50_000):
        scores.append(score_windows(joined.iloc[start:start+50_000],restored,metadata).detector_score.to_numpy())
    values=np.concatenate(scores)
    difference=float(np.max(np.abs(values-joined[f'score_{selected}'].to_numpy())))
    mismatches=int(np.count_nonzero((values>=threshold)!=joined[f'flagged_{selected}'].to_numpy(dtype=bool)))
    tolerance=1e-6 if metadata['format']=='numpy_mlp' else 1e-12
    if difference>tolerance or mismatches:
        raise ValueError(f'Inference release disagrees with saved held-out evidence: max score delta={difference}; flag mismatches={mismatches}.')
    if not 1<=preview_limit<=50_000:
        raise ValueError('Interactive preview limit must be between 1 and 50,000.')
    if len(development)>preview_limit:
        mandatory=development.groupby(['capture_group','reference_label'],sort=True).head(1)
        if len(mandatory)>preview_limit:
            raise ValueError('Preview limit cannot preserve one example per capture/label stratum.')
        remaining=development.drop(index=mandatory.index).sample(n=preview_limit-len(mandatory),random_state=42)
        preview=pd.concat([mandatory,remaining]).sort_index()
    else:
        preview=development
    public_features=preview[['timestamp','malicious',*FEATURES]].copy()
    public_features.insert(1,'time_group',preview.capture_group.to_numpy())
    public_features.to_csv(destination/'workspace_features.csv',index=False,lineterminator='\n')
    provenance={'source':'Actual expanded-benchmark development feature windows; bounded preview, not a training-data cap.',
                'workspace_label':'Expanded development-only feature preview',
                'snapshot_sha256':hashlib.sha256((destination/'workspace_features.csv').read_bytes()).hexdigest(),
                'model_input_rows':metrics['dataset']['input_rows'],'full_model_windows':len(windows),
                'development_windows':len(development),'rows':len(public_features),
                'positive_windows':int(public_features.malicious.sum()),'columns':list(public_features.columns),
                'sampling_policy':'All development windows if within the limit; otherwise one per development capture/label stratum and seeded random fill. Held-out captures are excluded.',
                'privacy':'IPs, window IDs, source-file labels and reference-label strings removed.',
                'scope':f"The published model benchmark processes {metrics['dataset']['input_rows']:,} raw data rows. This {len(public_features):,}-row development-only preview is solely for bounded interactive experiments and does not truncate detector training. time_group preserves whole development captures."}
    (destination/'workspace_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (destination/'README.md').write_text('# Expanded real benchmark snapshot\n\n'+metrics['dataset']['description']+'\n\n'+scope+'\n\n'+provenance['scope']+'\n\nThe default reports read saved evidence only. Run/Live inference loads a hash/version-checked packaged detector. No arbitrary pickle upload, production approval or confirmed attacker attribution is implied. Downloaded source timestamps are not silently corrected.\n',encoding='utf-8')
    load_benchmark(destination)
    result={'selected_model':selected,'model_input_rows':metrics['dataset']['input_rows'],
            'heldout_windows_verified':len(test),'heldout_score_max_difference':difference,'heldout_score_tolerance':tolerance,
            'inference_backend':metadata['format'],
            'heldout_flag_mismatches':mismatches,'interactive_preview_rows':len(public_features),
            'scope':'Staged and verified only; not activated or pushed by this function.'}
    (stage/'release-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
