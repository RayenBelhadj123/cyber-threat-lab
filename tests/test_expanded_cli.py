"""Expanded experiment CLI must preserve existing evidence before costly work."""
import sys

import pytest

from scripts import run_expanded_benchmark as cli
from cyberlab import expanded_training


@pytest.mark.parametrize('existing', ['directory', 'file'])
def test_existing_output_is_rejected_before_reading_inputs(tmp_path, monkeypatch, capsys, existing):
    output = tmp_path / 'experiment'
    if existing == 'directory':
        output.mkdir()
        evidence = output / 'metrics.json'
    else:
        evidence = output
    evidence.write_bytes(b'existing evidence must survive')

    def must_not_read(*args, **kwargs):
        pytest.fail('Existing output reached expensive input verification')

    monkeypatch.setattr(expanded_training, 'input_inventory', must_not_read)
    monkeypatch.setattr(sys, 'argv', ['run_expanded_benchmark.py', '--output', str(output)])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    assert 'fresh or empty directory' in capsys.readouterr().err
    assert evidence.read_bytes() == b'existing evidence must survive'


@pytest.mark.parametrize('already_empty', [False, True])
def test_fresh_or_empty_output_reaches_preparation(tmp_path, monkeypatch, already_empty):
    output = tmp_path / 'experiment'
    if already_empty:
        output.mkdir()

    class PreparationReached(Exception):
        pass

    def stop_before_real_data(*args, **kwargs):
        raise PreparationReached

    monkeypatch.setattr(expanded_training, 'input_inventory', stop_before_real_data)
    monkeypatch.setattr(sys, 'argv', ['run_expanded_benchmark.py', '--output', str(output)])
    with pytest.raises(PreparationReached):
        cli.main()
