import importlib.util
from pathlib import Path


def test_count_documents_ignores_empty_lines(tmp_path):
    spec = importlib.util.spec_from_file_location('verify_lab', Path(__file__).parents[1] / 'scripts/verify_lab.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / 'records.ndjson'
    p.write_text('{"a":1}\n\n{"a":2}\n')
    assert mod.count_documents(p) == 2
