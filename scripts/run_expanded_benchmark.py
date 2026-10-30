"""Train on all verified full-week + extension files with whole-capture holdout.

The old Thursday CSV is not included. Models/reports remain in a new output until
explicit promotion; this command never mutates the active snapshot or GitHub.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import pandas as pd
    from cyberlab.expanded_training import input_inventory,prepare_windows,create_capture_splits
    from cyberlab.benchmark import run_model_benchmark,evaluate_scores,render_html_report
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/model_benchmark_expanded')
    parser.add_argument('--prepared',type=Path,default=ROOT/'artifacts/expanded_preparation')
    args=parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Benchmark output requires a fresh or empty directory; choose a new --output to preserve existing evidence.')
    windows,lineage=prepare_windows(input_inventory(ROOT),args.prepared)
    split=create_capture_splits(windows)
    print(json.dumps({'raw_data_rows':lineage['input_rows'],'valid_unique_flows':lineage['valid_rows'],
                      'feature_windows':len(windows),'blank_placeholder_rows':lineage['blank_rows'],
                      'capture_counts':windows.groupby('capture_group').malicious.agg(['size','sum']).to_dict()},indent=2),flush=True)
    prepared={'windows':windows,'lineage':lineage,'split':split,
              'description':'Complete CICIDS2017 full-week distribution (8 CSVs) plus CIC-DDoS2019 targeted NTP/UDP-lag/Portmap extension (3 CSVs); no row subsampling. Model samples are source-file/host/minute feature windows.',
              'caveats':[
                  'All eleven verified CSVs are processed; the overlapping old Thursday input is excluded. Invalid rows, blank padding and canonical duplicate flows are audited separately.',
                  'Entire CICIDS2017 Friday and the CIC-DDoS Portmap/03-11 capture are excluded from development, model selection and OOF threshold tuning. Three whole-capture CV folds select the winner.',
                  'Capture-disjoint does not mean host-disjoint: source hosts/networks can recur across captures, and all sources are laboratory/mirror data.',
                  'CIC-DDoS mirror timestamp strings appear to use 2018 despite the dataset name. Raw values are not corrected; established dayfirst/America-Moncton parsing is an explicit source-clock assumption. This mixed-source experiment is not globally forward-time validation.',
                  'Window aggregation is source-file scoped; canonical flows are deduplicated across files within each dataset. File-boundary fragmentation and label/provenance errors remain possible.',
                  'Maximum-F1 thresholds are chosen only from development OOF scores, not business costs or the held-out captures. Inspect per-capture and per-attack support; aggregate scores can hide failures.',
                  'This is an expanded exploratory detector, not evidence of production readiness or automatically better generalization than the earlier, easier Thursday split.',
              ]}
    report=run_model_benchmark(None,args.output,prepared=prepared)
    dev=pd.read_csv(args.output/'development_oof_predictions.csv')
    test=pd.read_csv(args.output/'held_out_test_predictions.csv')
    selected=report['selection']['model']
    threshold=report['models'][selected]['threshold_from_dev_oof_max_f1']
    report['evaluation_by_capture']={str(group):evaluate_scores(frame.malicious,frame[f'score_{selected}'],threshold)
                                     for group,frame in test.groupby('time_group')}
    labels=sorted({label for value in test.loc[test.malicious,'reference_label'] for label in str(value).split('|') if label.upper() not in ('BENIGN','UNKNOWN')})
    attack_rows=[]
    for label in labels:
        mask=test.reference_label.map(lambda value:label in str(value).split('|')) & test.malicious
        support=int(mask.sum())
        detected=int((test.loc[mask,f'score_{selected}']>=threshold).sum())
        attack_rows.append({'label':label,'positive_windows':support,'detected_windows':detected,'recall':detected/support if support else None})
    report['held_out_attack_detection']=attack_rows
    pd.DataFrame(attack_rows).to_csv(args.output/'held_out_attack_detection.csv',index=False)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    html=render_html_report(report,dev.malicious,{name:dev[f'score_{name}'].to_numpy() for name in report['models']},test.malicious,{name:test[f'score_{name}'].to_numpy() for name in report['models']})
    (args.output/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'selected_model':selected,'mean_capture_cv_ap':report['selection']['selected_mean_cv_average_precision'],
                      'held_out_test':report['held_out_test']['metrics'],'by_capture':report['evaluation_by_capture'],
                      'attack_detection':attack_rows,'output':str(args.output)},indent=2),flush=True)


if __name__=='__main__':
    main()
