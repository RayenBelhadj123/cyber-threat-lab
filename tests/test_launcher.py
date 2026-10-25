from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_declares_winexe_icon_and_dashboard_target():
    project = ROOT / "launcher" / "CyberLabLauncher.csproj"
    source = ROOT / "launcher" / "Program.cs"
    assert project.exists()
    assert source.exists()
    project_text = project.read_text(encoding="utf-8")
    source_text = source.read_text(encoding="utf-8")
    assert "<OutputType>WinExe</OutputType>" in project_text
    assert "<ApplicationIcon>cyberlab-shield.ico</ApplicationIcon>" in project_text
    assert "http://localhost:15601/app/dashboards#/view/cyber-overview" in source_text
    assert "UseShellExecute = true" in source_text
    assert 'start.ArgumentList.Add("elasticsearch")' in source_text
    assert 'start.ArgumentList.Add("kibana")' in source_text


def test_launcher_icon_is_a_valid_png_backed_ico():
    data = (ROOT / "launcher" / "cyberlab-shield.ico").read_bytes()
    reserved, image_type, count = struct.unpack_from("<HHH", data, 0)
    width, height, _, _, planes, bits, image_size, image_offset = struct.unpack_from("<BBBBHHII", data, 6)
    assert (reserved, image_type, count) == (0, 1, 1)
    assert (width, height) == (0, 0)  # ICO encodes 256px dimensions as zero.
    assert (planes, bits) == (1, 32)
    assert image_offset == 22
    assert data[image_offset : image_offset + 8] == b"\x89PNG\r\n\x1a\n"
    assert image_size == len(data) - image_offset


def test_launcher_automatically_starts_docker_desktop_and_waits_for_engine():
    source = (ROOT / "launcher" / "Program.cs").read_text(encoding="utf-8")
    assert "Docker Desktop.exe" in source
    assert "WaitForDockerEngine" in source
    assert 'ArgumentList.Add("info")' in source


def test_published_launcher_check_mode():
    executable = ROOT / "dist" / "CyberThreatLabLauncher.exe"
    if not executable.exists():
        pytest.skip("Publish the Windows launcher before running its executable smoke test")
    result = subprocess.run([str(executable), "--check"], capture_output=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
