from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_report_service_is_localhost_only_and_mounts_latest_artifacts():
    lines = (ROOT / "compose.yaml").read_text(encoding="utf-8").splitlines()
    start = lines.index("  model-report:")
    block = []
    for line in lines[start + 1:]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        block.append(line)
    service = "\n".join(block)
    assert "image: caddy:2.10-alpine" in service
    assert "profiles: [report]" in service
    assert '127.0.0.1:18090:80' in service
    assert "./artifacts/model_benchmark:/usr/share/caddy:ro" in service
    assert "./deploy/report/Caddyfile:/etc/caddy/Caddyfile:ro" in service
    caddyfile = (ROOT / "deploy" / "report" / "Caddyfile").read_text(encoding="utf-8")
    assert "rewrite @root /report.html" in caddyfile
    assert "file_server" in caddyfile
