"""Download an explicitly attributed public CICIDS2017 Thursday mirror.
Original publisher: https://www.unb.ca/cic/datasets/ids-2017.html
Mirror is not an independent assertion of corrected labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import urllib.request

URL = 'https://raw.githubusercontent.com/WSU-AI-CyberSecurity/data/master/CICIDS2017-thursday.pcap_ISCX.csv'


def validate_csv(path):
    with Path(path).open(encoding='utf-8-sig', errors='replace', newline='') as f:
        columns = {x.strip() for x in next(csv.reader(f), [])}
    required = {'Source IP', 'Destination IP', 'Timestamp', 'Label'}
    if not required.issubset(columns):
        raise ValueError(f'Invalid CSV schema: missing {sorted(required - columns)}')
    return sorted(columns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/raw')
    p.add_argument('--url', default=URL)
    p.add_argument('--sha256', help='Optional expected hash; refuses a changed download')
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'Thursday-WorkingHours.pcap_ISCX.csv'
    temporary = target.with_suffix('.csv.part')
    sha = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(args.url, headers={'User-Agent': 'cyber-threat-lab/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as f:
            final_url = response.url
            while block := response.read(1024 * 1024):
                f.write(block)
                sha.update(block)
                size += len(block)
        digest = sha.hexdigest()
        if args.sha256 and digest.lower() != args.sha256.lower():
            raise ValueError('Downloaded SHA256 does not match expected checksum')
        columns = validate_csv(temporary)
        temporary.replace(target)
        manifest = {'dataset': 'CICIDS2017 Thursday public mirror', 'original_publisher': 'https://www.unb.ca/cic/datasets/ids-2017.html', 'requested_url': args.url, 'resolved_url': final_url, 'downloaded_at': datetime.now(timezone.utc).isoformat(), 'sha256': digest, 'bytes': size, 'path': str(target), 'columns': columns, 'scope': 'Thursday only; not the complete benchmark. Mirrored labels not independently corrected.'}
        (root.parent / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(manifest, indent=2))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
