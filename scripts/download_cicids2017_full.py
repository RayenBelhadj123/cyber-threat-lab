"""Download a pinned public mirror of original CICIDS2017 labeled flows.

The original publisher is CIC/UNB. This is an unofficial mirror, not an
independently corrected dataset. Full identity/time fields are retained locally.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import requests

REVISION = '811c007f08d693a8a8b4226c197c138391806ce6'
URL = f'https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/{REVISION}/csvs/GeneratedLabelledFlows.zip'
EXPECTED_SHA256 = '7bdbef286f8893f31c6db12105fa097fa5c2dcc6733179037a08129d150ea27a'
EXPECTED_BYTES = 283876488
EXPECTED_FILES = (
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_csv(path):
    """Validate original identity/time schema and count rows/labels without sampling."""
    path = Path(path)
    labels = Counter()
    rows = malformed = blank = 0
    # Original distribution uses Windows-1252 for web-attack label punctuation.
    with path.open(encoding='cp1252', newline='') as source:
        reader = csv.reader(source)
        header = [column.strip().lstrip('\ufeff') for column in next(reader, [])]
        required = {'Source IP', 'Destination IP', 'Timestamp', 'Label'}
        if not required.issubset(header):
            raise ValueError(f'Invalid CSV schema in {path.name}: missing {sorted(required-set(header))}')
        label_index = header.index('Label')
        for row in reader:
            if not row or not any(value.strip() for value in row):
                blank += 1
                continue
            rows += 1
            if len(row) != len(header):
                malformed += 1
            label = row[label_index].strip() if label_index < len(row) else ''
            labels[label or '<missing>'] += 1
    if rows == 0:
        raise ValueError(f'Empty CSV: {path.name}')
    return {'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path),
            'rows': rows, 'labels': dict(sorted(labels.items())), 'columns': header,
            'malformed_rows': malformed, 'blank_rows': blank, 'encoding': 'cp1252'}


def extract_validated(archive, output, expected_files=EXPECTED_FILES):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage inside data/ so failed validation never publishes a partial dataset.
    with tempfile.TemporaryDirectory(prefix='.cicids-staging-', dir=output.parent) as staging:
        stage = Path(staging)
        records = []
        with zipfile.ZipFile(archive) as zipped:
            members = {}
            total_size = 0
            for item in zipped.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in item.filename or ':' in item.filename:
                    raise ValueError(f'Unsafe archive member: {item.filename}')
                if item.is_dir() or not item.filename.lower().endswith('.csv'):
                    continue
                if path.name not in expected_files or path.name in members:
                    raise ValueError(f'Unexpected/duplicate CSV: {item.filename}')
                total_size += item.file_size
                if total_size > 5_000_000_000:
                    raise ValueError('Archive uncompressed data exceeds safety bound')
                members[path.name] = item
            if set(members) != set(expected_files):
                raise ValueError(f'Archive missing expected CSVs: {sorted(set(expected_files)-set(members))}')
            for name in expected_files:
                target = stage / name
                with zipped.open(members[name]) as source, target.open('wb') as dest:
                    shutil.copyfileobj(source, dest, length=1024*1024)
                records.append(inspect_csv(target))
        stage.rename(output)
    return records


def download_archive(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != EXPECTED_BYTES or sha256(target) != EXPECTED_SHA256:
            raise ValueError(f'Existing archive has wrong checksum: {target}')
        return
    partial = target.with_suffix('.zip.part')
    try:
        digest = hashlib.sha256()
        total = 0
        with requests.get(URL, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with partial.open('wb') as dest:
                for block in response.iter_content(1024*1024):
                    if not block:
                        continue
                    total += len(block)
                    if total > EXPECTED_BYTES:
                        raise ValueError('Download exceeds pinned size')
                    dest.write(block)
                    digest.update(block)
        if total != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError('Download checksum/size differs from pinned mirror metadata')
        partial.rename(target)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/cicids2017_full')
    parser.add_argument('--archive', default='data/downloads/GeneratedLabelledFlows.zip')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}; existing Thursday and benchmark outputs are preserved')
    download_archive(args.archive)
    records = extract_validated(args.archive, output)
    labels = Counter()
    for record in records:
        labels.update(record['labels'])
    manifest = {
        'dataset': 'CICIDS2017 full labeled-flow CSV distribution (unofficial mirror)',
        'original_publisher': 'https://www.unb.ca/cic/datasets/ids-2017.html',
        'source_url': URL, 'mirror_revision': REVISION,
        'archive_sha256': EXPECTED_SHA256, 'archive_bytes': EXPECTED_BYTES,
        'checksum_scope': 'Matches pinned mirror LFS SHA256, not an independently verified original-publisher checksum.',
        'downloaded_at': datetime.now(timezone.utc).isoformat(),
        'output': str(output), 'file_count': len(records),
        'rows': sum(record['rows'] for record in records),
        'csv_bytes': sum(record['bytes'] for record in records),
        'labels': dict(sorted(labels.items())), 'files': records,
        'scope': 'All eight original labeled-flow CSVs across Monday-Friday. Do not combine with the existing Thursday mirror: web-attack rows overlap. No retraining performed.',
    }
    path = output.parent / 'cicids2017_full_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'files'}, indent=2, ensure_ascii=False))
    print(f'Manifest: {path}')


if __name__ == '__main__':
    main()
