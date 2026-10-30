"""Full-dataset acquisition boundaries and capture-disjoint training preparation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def input_inventory(root):
    root=Path(root)
    sources=(('CICIDS2017','data/cicids2017_full_manifest.json'),
             ('CICDDoS2019','data/cicddos2019_extension/source_manifest.json'))
    entries=[]
    for dataset,manifest_name in sources:
        manifest=json.loads((root/manifest_name).read_text(encoding='utf-8'))
        directory=root/Path(manifest['output'].replace(chr(92),'/'))
        for item in manifest['files']:
            path=directory/item['name']
            if path.stat().st_size!=item['bytes']:
                raise ValueError(f'Input size mismatch: {path}')
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):
                    h.update(block)
            if h.hexdigest()!=item['sha256']:
                raise ValueError(f'Input checksum mismatch: {path}')
            entries.append({'dataset':dataset,'path':path,'name':item['name'],'sha256':item['sha256'],'expected_rows':item['rows']})
    return entries


def prepare_windows(entries, output):
    """Process complete files one at a time; disk-backed global canonical-flow dedup."""
    import gc
    import sqlite3
    import time
    from collections import Counter
    import pandas as pd
    from .pipeline import load_flows, aggregate_windows, digest
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    signature=digest([(e['dataset'],e['name'],e['sha256']) for e in entries])
    metadata_path=output/'preparation.json'
    windows_path=output/'windows.csv'
    if metadata_path.exists() and windows_path.exists():
        saved=json.loads(metadata_path.read_text(encoding='utf-8'))
        if saved['input_signature']!=signature:
            raise ValueError('Prepared input changed; use a fresh output directory.')
        expected_counts=[(e['dataset'],e['name'],e['expected_rows']) for e in entries]
        cached_counts=[(f['dataset'],f['file'],f['input_rows']) for f in saved['lineage']['files']]
        if cached_counts!=expected_counts:
            raise ValueError('Cached manifest row count mismatch; use verified inputs and a fresh output directory.')
        h=hashlib.sha256(windows_path.read_bytes()).hexdigest()
        if h!=saved['windows_sha256']:
            raise ValueError('Prepared windows checksum mismatch.')
        windows=pd.read_csv(windows_path)
        windows['timestamp']=pd.to_datetime(windows.timestamp,utc=True)
        return windows,saved['lineage']
    db_path=output/'canonical-flow-dedup.sqlite'
    if db_path.exists():
        db_path.unlink()
    database=sqlite3.connect(db_path)
    database.execute('PRAGMA journal_mode=OFF')
    database.execute('PRAGMA synchronous=OFF')
    database.execute('PRAGMA cache_size=-16384')
    database.execute('CREATE TABLE seen (dataset TEXT NOT NULL, event_id TEXT NOT NULL, PRIMARY KEY(dataset,event_id)) WITHOUT ROWID')
    frames=[]
    labels=Counter()
    lineage={'input_rows':0,'blank_rows':0,'valid_rows':0,'invalid_rows':0,'duplicate_rows':0,'cross_file_duplicates':0,'files':[],
             'timestamp_policy':'Established CICIDS parser: dayfirst, America/Moncton local clock converted to UTC. Mirror DDoS 2018-looking strings are preserved in raw files; source timezone/year authenticity is not established. Capture grouping uses declared file/capture identity, not cross-dataset calendar ordering.'}
    try:
        for entry in entries:
            started=time.monotonic()
            print(f"Preparing {entry['dataset']} / {entry['name']} ({entry['expected_rows']:,} raw rows)",flush=True)
            flows,info=load_flows(entry['path'])
            if info['input_rows']!=entry['expected_rows']:
                raise ValueError(f"Manifest row count mismatch: {entry['name']}")
            cursor=database.cursor()
            keep=[cursor.execute('INSERT OR IGNORE INTO seen VALUES (?,?)',(entry['dataset'],eid)).rowcount==1 for eid in flows.event_id]
            database.commit()
            cross_duplicates=len(keep)-sum(keep)
            flows=flows.loc[keep].reset_index(drop=True)
            lineage['input_rows']+=info['input_rows']
            lineage['blank_rows']+=info['blank_rows']
            lineage['invalid_rows']+=info['invalid_rows']
            lineage['duplicate_rows']+=info['duplicate_rows']+cross_duplicates
            lineage['cross_file_duplicates']+=cross_duplicates
            lineage['valid_rows']+=len(flows)
            labels.update(flows.label.value_counts().to_dict())
            windows=aggregate_windows(flows) if len(flows) else pd.DataFrame()
            if len(windows):
                windows['dataset']=entry['dataset']
                windows['source_file']=entry['name']
                capture=entry['name'].split('-')[0] if entry['dataset']=='CICIDS2017' else entry['name'].split('_')[0]
                windows['capture_group']=entry['dataset']+':'+capture
                windows['window_id']=[digest([entry['dataset'],entry['name'],value]) for value in windows.window_id]
                frames.append(windows)
            record={'dataset':entry['dataset'],'file':entry['name'],'sha256':entry['sha256'],
                    'input_rows':info['input_rows'],'blank_rows':info['blank_rows'],'valid_unique_flows':len(flows),'invalid_rows':info['invalid_rows'],
                    'duplicate_rows':info['duplicate_rows']+cross_duplicates,'windows':len(windows)}
            lineage['files'].append(record)
            (output/'progress.json').write_text(json.dumps(lineage,indent=2),encoding='utf-8')
            print(f"Prepared {entry['name']}: {len(flows):,} unique valid flows -> {len(windows):,} windows in {time.monotonic()-started:.1f}s",flush=True)
            del flows
            gc.collect()
    finally:
        database.close()
    if not frames:
        raise ValueError('No feature windows remain.')
    windows=pd.concat(frames,ignore_index=True).sort_values(['dataset','source_file','timestamp','source_ip']).reset_index(drop=True)
    if not windows.window_id.is_unique:
        raise ValueError('Window identities overlap across prepared inputs.')
    lineage['row_labels']={str(k):int(v) for k,v in labels.items()}
    assert lineage['input_rows']==lineage['valid_rows']+lineage['invalid_rows']+lineage['duplicate_rows']
    windows.to_csv(windows_path,index=False)
    metadata_path.write_text(json.dumps({'input_signature':signature,'windows_sha256':hashlib.sha256(windows_path.read_bytes()).hexdigest(),'lineage':lineage},indent=2),encoding='utf-8')
    return windows,lineage


def create_capture_splits(windows):
    import numpy as np
    groups=windows.capture_group.to_numpy(dtype=str)
    y=windows.malicious.to_numpy(dtype=bool)
    heldout={'CICIDS2017:Friday','CICDDoS2019:03-11'}
    validation_groups=[{'CICIDS2017:Wednesday'},
                       {'CICIDS2017:Tuesday','CICIDS2017:Thursday'},
                       {'CICIDS2017:Monday','CICDDoS2019:01-12'}]
    expected=heldout|set().union(*validation_groups)
    if set(groups)!=expected:
        raise ValueError(f'Unexpected or missing capture groups: {set(groups)^expected}')
    test=np.flatnonzero(np.isin(groups,list(heldout)))
    dev=np.flatnonzero(~np.isin(groups,list(heldout)))
    folds=[]
    for values in validation_groups:
        validation=dev[np.isin(groups[dev],list(values))]
        train=dev[~np.isin(groups[dev],list(values))]
        if np.unique(y[train]).size!=2 or np.unique(y[validation]).size!=2:
            raise ValueError('A whole-capture CV fold lacks a class; refusing row-shuffled fallback.')
        folds.append({'train_indices':train,'validation_indices':validation})
    if np.unique(y[test]).size!=2:
        raise ValueError('Whole-capture holdout lacks a class.')
    return {'dev_indices':dev,'test_indices':test,'inner_folds':folds,'groups':groups,
            'group_minutes':0,'random_state':42,'outer_test_fold':None,
            'method':'Capture-disjoint: entire CICIDS2017 Friday and CIC-DDoS Portmap/03-11 capture held out; three predefined whole-capture development CV folds.',
            'temporal_policy':'capture_disjoint_mixed_source_clocks_not_global_forward'}
