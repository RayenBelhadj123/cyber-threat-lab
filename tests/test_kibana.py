"""Kibana provisioning contract tests; no running Elastic stack required."""
import importlib.util
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import subprocess
import sys
import pytest


@pytest.fixture
def kibana_api():
    """A protocol test server, not a claim of live Kibana compatibility."""
    state = {"views": {}, "objects": {}, "writes": [], "reads": [], "reject": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_POST(self):
            assert self.headers.get("kbn-xsrf")
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["writes"].append((self.path, payload))
            if state["reject"]:
                return self.reply(500, {"error": "intentional test failure"})
            if self.path == "/api/data_views/data_view":
                view = payload["data_view"]
                assert payload["override"] is True
                state["views"][view["id"]] = view
                return self.reply(200, {"data_view": view})
            if self.path == "/api/saved_objects/_export":
                assert payload["includeReferencesDeep"] is True
                self.send_response(200)
                self.end_headers()
                objects = list(state["objects"].values())
                objects += [{"type": "index-pattern", "id": k, "attributes": v}
                            for k, v in state["views"].items()]
                self.wfile.write(("\n".join(json.dumps(o) for o in objects) + "\n").encode())
                return
            prefix = "/api/saved_objects/"
            assert self.path.startswith(prefix)
            assert self.path.endswith("?overwrite=true")
            kind, object_id = self.path[len(prefix):].split("?")[0].split("/")
            obj = {"id": object_id, "type": kind, **payload}
            state["objects"][(kind, object_id)] = obj
            self.reply(200, obj)

        def do_GET(self):
            state["reads"].append(self.path)
            if self.path.startswith("/api/data_views/data_view/"):
                return self.reply(200, {"data_view": state["views"][self.path.rsplit("/", 1)[1]]})
            kind, object_id = self.path[len("/api/saved_objects/"):].split("/")
            self.reply(200, state["objects"][(kind, object_id)])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    worker.join()


def test_provision_writes_reads_back_is_idempotent_and_exports(tmp_path, kibana_api):
    module = load_provisioner()
    url, state = kibana_api
    artifact = tmp_path / "nested" / "kibana.ndjson"
    result = module.provision(url, output=artifact)
    assert result == {"data_views": 4, "visualizations": 45, "dashboards": 5}
    assert len(state["reads"]) == 54
    assert len(state["views"]) == 4
    assert len(state["objects"]) == 50
    exported = [json.loads(line) for line in artifact.read_text().splitlines()]
    assert len(exported) == 54
    assert {x["id"] for x in exported if x["type"] == "dashboard"} == {
        "cyber-overview", "cyber-findings", "cyber-host", "cyber-model", "cyber-benchmark"}
    module.provision(url)
    assert len(state["objects"]) == 50


def test_http_errors_are_not_reported_as_success(kibana_api):
    import requests
    url, state = kibana_api
    state["reject"] = True
    with pytest.raises(requests.HTTPError):
        load_provisioner().provision(url)



def test_cli_uses_environment_url_and_writes_export(tmp_path, kibana_api, monkeypatch):
    url, state = kibana_api
    monkeypatch.setenv("KIBANA_URL", url)
    artifact = tmp_path / "cli.ndjson"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "provision_kibana.py"),
                             "--output", str(artifact)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["dashboards"] == 5
    assert artifact.exists()
    assert len(state["objects"]) == 50


def test_cli_default_url(monkeypatch):
    monkeypatch.delenv("KIBANA_URL", raising=False)
    assert load_provisioner().parse_args([]).url == "http://localhost:15601"



def test_compose_and_python_image_are_isolated_local_development():
    import shutil
    path = ROOT / "compose.yaml"
    assert path.exists(), "Compose deployment is not implemented"
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is needed to validate Compose; no containers are started")
    result = subprocess.run(["docker", "compose", "-f", str(path), "--profile", "tools",
                             "config", "--format", "json"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    compose = json.loads(result.stdout)
    assert compose["name"] == "cyber-threat-lab"
    services = compose["services"]
    es, kb, pipeline = (services[name] for name in ("elasticsearch", "kibana", "pipeline"))
    assert es["image"] == "docker.elastic.co/elasticsearch/elasticsearch:8.17.0"
    assert kb["image"] == "docker.elastic.co/kibana/kibana:8.17.0"
    for service, published, target in ((es, "19200", 9200), (kb, "15601", 5601)):
        assert len(service["ports"]) == 1
        assert service["ports"][0]["host_ip"] == "127.0.0.1"
        assert service["ports"][0]["published"] == published
        assert service["ports"][0]["target"] == target
    assert es["environment"]["discovery.type"] == "single-node"
    assert str(es["environment"]["xpack.security.enabled"]).lower() == "false"
    assert es["environment"]["ES_JAVA_OPTS"] == "-Xms512m -Xmx512m"
    assert int(es["mem_limit"]) == 1536 * 1024 * 1024
    assert int(kb["mem_limit"]) == 1024 * 1024 * 1024
    assert es["healthcheck"]["test"]
    assert kb["healthcheck"]["test"][0] == "CMD-SHELL"
    assert "curl -fsS" in kb["healthcheck"]["test"][1]
    assert kb["depends_on"]["elasticsearch"]["condition"] == "service_healthy"
    assert pipeline["profiles"] == ["tools"]
    assert pipeline["command"] == ["python", "-m", "cyberlab", "run", "--input", "/app/data/raw", "--output", "/app/artifacts", "--publish"]
    assert pipeline["depends_on"]["elasticsearch"]["condition"] == "service_healthy"
    assert {v["target"] for v in pipeline["volumes"]} == {"/app/data", "/app/artifacts"}
    assert all(v["type"] == "bind" and not v.get("read_only", False) for v in pipeline["volumes"])
    assert pipeline["environment"]["PYTHONPATH"] == "/app"
    assert pipeline["environment"]["ELASTICSEARCH_URL"] == "http://elasticsearch:9200"
    assert pipeline["environment"]["KIBANA_URL"] == "http://kibana:5601"
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.12-slim" in dockerfile
    assert "pip install" in dockerfile and "requirements.txt" in dockerfile
    assert "requirements-dl.txt" in dockerfile and "download.pytorch.org/whl/cpu" in dockerfile
    assert "WORKDIR /app" in dockerfile
    dl_requirements = (ROOT / "requirements-dl.txt").read_text().splitlines()
    assert dl_requirements == ["torch==2.13.0"]
    required = {"pandas", "numpy", "scikit-learn", "requests", "pytest", "joblib", "tzdata", "kafka-python", "streamlit", "plotly"}
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    package_names = {r.split(">=", 1)[0] if ">=" in r else r.split("==", 1)[0]
                     for r in requirements if r and not r.startswith("#")}
    assert package_names == required
    assert all((">=" in r and ",<" in r) or "==" in r for r in requirements if r and not r.startswith("#"))
    assert "data/" in (ROOT / ".dockerignore").read_text()
    assert "artifacts/" in (ROOT / ".gitignore").read_text()


ROOT = Path(__file__).resolve().parents[1]


def load_provisioner():
    path = ROOT / "scripts" / "provision_kibana.py"
    assert path.exists(), "Kibana provisioner is not implemented"
    spec = importlib.util.spec_from_file_location("provision_kibana", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboards_have_richer_graphs_and_full_width_layout():
    _, objects = load_provisioner().build_saved_objects()
    lookup = {(o["type"], o["id"]): o for o in objects}
    dashboards = [o for o in objects if o["type"] == "dashboard"]
    for dashboard in dashboards:
        panels = json.loads(dashboard["attributes"]["panelsJSON"])
        assert len(panels) >= 7, f"{dashboard['id']} should have KPI cards and multiple graph panels"
        assert all(p["gridData"]["w"] == 24 and p["gridData"]["h"] >= 15 for p in panels)
        refs = {r["name"]: r for r in dashboard["references"]}
        types = set()
        for panel in panels:
            vis = lookup[("visualization", refs[panel["panelRefName"]]["id"])]
            state = json.loads(vis["attributes"]["visState"])
            if state["type"] == "metric":
                assert state["params"].get("type") == "metric", f"{vis['id']} needs the metric renderer type"
                assert state["params"].get("metric", {}).get("style", {}).get("fontSize")
            types.add(state["type"])
        assert len(types) >= 3, f"{dashboard['id']} should mix chart types, got {types}"


def test_model_flagged_kpi_counts_only_flagged_records():
    _, objects = load_provisioner().build_saved_objects()
    vis = next(o for o in objects if o["id"] == "cyber-model-panel-2")
    search = json.loads(vis["attributes"]["kibanaSavedObjectMeta"]["searchSourceJSON"])
    assert search["query"]["query"] == "flagged: true"


def test_benchmark_dashboard_references_benchmark_data_and_report_metrics():
    views, objects = load_provisioner().build_saved_objects()
    view = next(v for v in views if v["id"] == "cyber-benchmark-view")
    assert view["title"] == "cyber-benchmark*"
    dashboard = next(o for o in objects if o["id"] == "cyber-benchmark")
    assert dashboard["attributes"]["timeFrom"] == "now-30d"
    panels = json.loads(dashboard["attributes"]["panelsJSON"])
    assert len(panels) == 13
    lookup = {(o["type"], o["id"]): o for o in objects}
    refs = {r["name"]: r for r in dashboard["references"]}
    panel_titles = set()
    for panel in panels:
        ref = refs[panel["panelRefName"]]
        vis = lookup[(ref["type"], ref["id"])]
        panel_titles.add(vis["attributes"]["title"])
        assert vis["references"][0]["id"] == "cyber-benchmark-view"
        state = json.loads(vis["attributes"]["visState"])
        assert any(agg["schema"] == "metric" for agg in state["aggs"])
    assert {"Model comparison table", "Selected-model confusion matrix", "Learning-size diagnostics"} <= panel_titles
    comparison = next(o for o in objects if o["id"] == "cyber-benchmark-panel-12")
    fields = {agg.get("params", {}).get("field") for agg in json.loads(comparison["attributes"]["visState"])["aggs"]}
    assert {"model_name", "mean_cv_average_precision", "test_average_precision", "test_f1", "test_fp", "test_fn"} <= fields


def test_five_dashboards_contain_real_aggregations_and_resolved_references():
    views, objects = load_provisioner().build_saved_objects()
    assert {v["title"] for v in views} == {"cyber-flows*", "cyber-features*", "cyber-findings*", "cyber-benchmark*"}
    assert all(v["timeFieldName"] == "@timestamp" for v in views)
    lookup = {(o["type"], o["id"]): o for o in objects}
    lookup.update({("index-pattern", v["id"]): v for v in views})
    dashboards = [o for o in objects if o["type"] == "dashboard"]
    assert {d["id"] for d in dashboards} == {"cyber-overview", "cyber-findings", "cyber-host", "cyber-model", "cyber-benchmark"}
    assert all(d["attributes"]["title"].startswith("🛡️ ") for d in dashboards)
    for obj in objects:
        for ref in obj["references"]:
            assert (ref["type"], ref["id"]) in lookup
    for dashboard in dashboards:
        attrs = dashboard["attributes"]
        assert attrs["timeRestore"] is True
        if dashboard["id"] == "cyber-benchmark":
            assert attrs["timeFrom"] == "now-30d"
            assert attrs["timeTo"] == "now"
        else:
            assert attrs["timeFrom"] == "2017-07-06T00:00:00.000Z"
            assert attrs["timeTo"] == "2017-07-07T23:59:59.999Z"
        panels = json.loads(attrs["panelsJSON"])
        assert len(panels) >= 4
        references = {r["name"]: r for r in dashboard["references"]}
        for panel in panels:
            ref = references[panel["panelRefName"]]
            vis = lookup[(ref["type"], ref["id"]) ]
            state = json.loads(vis["attributes"]["visState"])
            assert state["type"] != "markdown"
            assert state["aggs"]
            assert any(a["schema"] == "metric" for a in state["aggs"])
            search = json.loads(vis["attributes"]["kibanaSavedObjectMeta"]["searchSourceJSON"])
            assert search["indexRefName"] in {r["name"] for r in vis["references"]}
