import importlib.util
from pathlib import Path
import pytest


def test_validate_rejects_html_and_missing_identity(tmp_path):
    spec = importlib.util.spec_from_file_location('download', Path(__file__).parents[1] / 'scripts/download_data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bad = tmp_path / 'bad.csv'
    bad.write_text('<html>login</html>')
    with pytest.raises(ValueError, match='schema'):
        mod.validate_csv(bad)
    good = tmp_path / 'good.csv'
    good.write_text(' Source IP, Destination IP, Timestamp, Label\n192.0.2.1,192.0.2.2,6/7/2017 8:59,BENIGN\n')
    assert 'Source IP' in mod.validate_csv(good)
    bad.write_text('Label,Timestamp\nBENIGN,6/7/2017 8:59\n')
    with pytest.raises(ValueError, match='schema'):
        mod.validate_csv(bad)
