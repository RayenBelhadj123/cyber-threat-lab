"""Acquire an attributed, pinned CIC-DDoS2019 extension; no automatic training.

Retain raw timestamps verbatim. The mirror contains 2018-looking timestamp
strings despite its CIC-DDoS2019 name; do not silently reinterpret them.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import requests

REVISION='1b3677ee26ba69caa5a9c1ba91cc75cec7ad976e'
BASE_URL=f'https://huggingface.co/datasets/baalajimaestro/CICDDoS2019/resolve/{REVISION}/'
FILES=(
    {'remote_path':'01-12/DrDoS_NTP.csv','bytes':645008606,'sha256':'b4ae33b2a22975f2c4c8b0e2bfc501fee38dae274dca37d4b01010de059c9c2c'},
    {'remote_path':'01-12/UDPLag.csv','bytes':157968770,'sha256':'f67708a8462509932759ffed7eb92dca0f423267609d1be38a461708d4ea5b7e'},
    {'remote_path':'03-11/Portmap.csv','bytes':78611080,'sha256':'d0148da21f3c645b32b21b59386721eddd497a16e3de4e2950391e575fca2e28'},
)


def download_file(url,target,expected_size,expected_sha256):
    """Retry transient transport failures; never retry a checksum mismatch."""
    for attempt in range(3):
        try:
            return _download_once(url,target,expected_size,expected_sha256)
        except requests.RequestException:
            if attempt == 2:
                raise
            print(f'Transient download interruption; retry {attempt+2}/3',flush=True)


def _download_once(url,target,expected_size,expected_sha256):
    target=Path(target)
    if target.exists():
        with target.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if target.stat().st_size!=expected_size or digest!=expected_sha256:
            raise ValueError(f'Checksum/size mismatch for existing file: {target}')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix('.csv.part')
    digest=hashlib.sha256()
    size=partial.stat().st_size if partial.exists() else 0
    if size>expected_size:
        partial.unlink()
        raise ValueError('Partial download exceeds pinned size')
    if size:
        with partial.open('rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''):
                digest.update(block)
    while size<expected_size:
        end=min(size+16*1024*1024,expected_size)-1
        with requests.get(url,headers={'Range':f'bytes={size}-{end}'},stream=True,timeout=(30,60)) as response:
            response.raise_for_status()
            if response.status_code==206:
                content_range=response.headers.get('Content-Range','')
                if not content_range.startswith(f'bytes {size}-') or not content_range.endswith(f'/{expected_size}'):
                    raise ValueError('Invalid resumed Content-Range')
            elif response.status_code==200 and size:
                raise ValueError('Server ignored required resume range')
            before=size
            with partial.open('ab') as stream:
                for block in response.iter_content(1024*1024):
                    if not block: continue
                    size+=len(block)
                    if size>expected_size:
                        raise ValueError('Download exceeds pinned size')
                    stream.write(block)
                    digest.update(block)
            if size==before:
                raise requests.ConnectionError('Empty response before end of download')
        print(f'{target.name}: {size}/{expected_size} bytes',flush=True)
    if digest.hexdigest()!=expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError('Downloaded checksum/size differs from pinned mirror metadata')
    partial.rename(target)


def _inspector():
    path=Path(__file__).with_name('download_cicids2017_full.py')
    spec=importlib.util.spec_from_file_location('cicids_csv_inspector',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspect_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/cicddos2019_extension')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    inspect_csv=_inspector()
    with nullcontext(output.with_name(output.name+'.staging')) as staging:
        stage=Path(staging)
        stage.mkdir(exist_ok=True)
        def acquire(spec):
            name=spec['remote_path'].replace('/','_')
            url=BASE_URL+spec['remote_path']
            target=stage/name
            download_file(url,target,spec['bytes'],spec['sha256'])
            record=inspect_csv(target)
            record.update({'source_url':url,'remote_path':spec['remote_path']})
            print(f'Validated {name}: {record["rows"]} raw rows',flush=True)
            return record
        with ThreadPoolExecutor(max_workers=2) as pool:
            records=list(pool.map(acquire,FILES))
        labels=Counter()
        for record in records: labels.update(record['labels'])
        manifest={
            'dataset':'CIC-DDoS2019 targeted attack extension (unofficial mirror)',
            'original_publisher':'https://www.unb.ca/cic/datasets/ddos-2019.html',
            'mirror':'baalajimaestro/CICDDoS2019', 'mirror_revision':REVISION,
            'downloaded_at':datetime.now(timezone.utc).isoformat(),
            'output':str(output), 'file_count':len(records),
            'bytes':sum(x['bytes'] for x in records),
            'rows':sum(x['rows'] for x in records),
            'labels':dict(sorted(labels.items())), 'files':records,
            'checksum_scope':'Matches pinned mirror LFS SHA256 values, not independently verified original-publisher hashes.',
            'scope':'Three complete CSV files; not the whole CIC-DDoS2019 distribution. No row downsampling or benign-row removal.',
            'integration_status':'Downloaded and schema-validated only. Kept separate from CICIDS2017. Models and deployed reporting snapshot unchanged.',
            'timestamp_warning':'Raw mirror timestamps include 2018-12-01 / 2018-11-03 forms despite dataset name and folder dates. Preserve verbatim; capture-day/format/timezone alignment must be independently established before chronological or cross-dataset training.',
            'evaluation_policy':'Reserve an independent capture/domain holdout before any tuning. Deduplicate/group flows; preserve benign evaluation coverage. Avoid letting attack-heavy extension dictate deployment class prevalence. No performance improvement claimed before evaluation.',
        }
        (stage/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        stage.rename(output)
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
