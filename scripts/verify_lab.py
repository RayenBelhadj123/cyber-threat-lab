"""Read back this lab's exact indices and dashboards after publishing."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import requests


def count_documents(path):
    with Path(path).open(encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', default='artifacts')
    p.add_argument('--es-url', default=os.getenv('ELASTICSEARCH_URL', 'http://localhost:19200'))
    p.add_argument('--kibana-url', default=os.getenv('KIBANA_URL', 'http://localhost:15601'))
    args = p.parse_args()
    root = Path(args.artifacts)
    report = {'verified_at': datetime.now(timezone.utc).isoformat(), 'indices': {}, 'dashboards': {}}
    for kind in ('flows', 'features', 'findings'):
        expected = count_documents(root / f'{kind}.ndjson')
        index = f'cyber-{kind}-v1'
        response = requests.get(f'{args.es_url.rstrip("/")}/{index}/_count', timeout=30)
        response.raise_for_status()
        actual = response.json()['count']
        if expected != actual:
            raise ValueError(f'{index}: local {expected} != indexed {actual}; check duplicate/mixed experiments')
        sample = requests.get(f'{args.es_url.rstrip("/")}/{index}/_search', params={'size': 1}, timeout=30)
        sample.raise_for_status()
        if expected and not sample.json()['hits']['hits']:
            raise ValueError(f'{index}: no readable documents')
        report['indices'][index] = {'local_count': expected, 'indexed_count': actual}
    for name in ('overview', 'findings', 'host', 'model'):
        identity = f'cyber-{name}'
        response = requests.get(f'{args.kibana_url.rstrip("/")}/api/saved_objects/dashboard/{identity}', headers={'kbn-xsrf': 'true'}, timeout=30)
        response.raise_for_status()
        obj = response.json()
        panels = json.loads(obj['attributes'].get('panelsJSON', '[]'))
        if not panels:
            raise ValueError(f'{identity}: dashboard has no panels')
        report['dashboards'][identity] = {'title': obj['attributes']['title'], 'panels': len(panels)}
    output = root / 'verification.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
