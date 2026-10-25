import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Tuesday-WorkingHours.pcap_ISCX.csv'


def module():
    path = ROOT/'scripts'/'download_cicids2017_full.py'
    assert path.exists(), 'Full-week downloader is missing'
    spec = importlib.util.spec_from_file_location('download_full',path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extraction_validates_and_records_real_csv_rows(tmp_path):
    mod = module()
    archive=tmp_path/'input.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('TrafficLabelling/'+NAME,' Source IP, Destination IP, Timestamp, Label\n192.0.2.1,192.0.2.2,4/7/2017 8:59,BENIGN\n192.0.2.1,192.0.2.2,4/7/2017 9:30,FTP-Patator\n')
    records=mod.extract_validated(archive,tmp_path/'ready',expected_files=(NAME,))
    assert len(records)==1
    assert records[0]['rows']==2
    assert records[0]['labels']=={'BENIGN':1,'FTP-Patator':1}
    assert len(records[0]['sha256'])==64
    assert (tmp_path/'ready'/NAME).exists()
    with pytest.raises(FileExistsError):
        mod.extract_validated(archive,tmp_path/'ready',expected_files=(NAME,))


def test_archive_traversal_or_wrong_schema_never_publishes(tmp_path):
    mod=module()
    archive=tmp_path/'input.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('../'+NAME,'bad')
    with pytest.raises(ValueError,match='Unsafe'):
        mod.extract_validated(archive,tmp_path/'ready',expected_files=(NAME,))
    assert not (tmp_path/'ready').exists()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(NAME,'Label,Destination Port\nBENIGN,80\n')
    with pytest.raises(ValueError,match='schema'):
        mod.extract_validated(archive,tmp_path/'ready',expected_files=(NAME,))
    assert not (tmp_path/'ready').exists()


def test_pinned_source_is_labeled_flows_not_ml_features_only():
    mod=module()
    assert 'GeneratedLabelledFlows.zip' in mod.URL
    assert mod.REVISION in mod.URL
    assert len(mod.EXPECTED_SHA256)==64
    assert len(mod.EXPECTED_FILES)==8
