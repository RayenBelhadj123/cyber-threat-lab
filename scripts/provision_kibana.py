"""Provision real aggregation dashboards through Kibana 8.17 public APIs.

Local unauthenticated development only. Stable IDs make reruns replace the lab
objects, not unrelated dashboards. No direct writes to the .kibana index.
Elastic references:
https://www.elastic.co/docs/api/doc/kibana/v8/operation/operation-createdataviewdefaultw
https://www.elastic.co/guide/en/kibana/8.17/saved-objects-api-create.html
https://www.elastic.co/guide/en/kibana/8.17/saved-objects-api-export.html
"""
import json


def build_saved_objects():
    """Return four data views and linked, editable aggregation visualizations."""
    views = [{"id": f"cyber-{kind}-view", "title": f"cyber-{kind}*",
              "name": f"Cyber {kind}", "timeFieldName": "@timestamp", "allowNoIndex": True}
             for kind in ("flows", "features", "findings", "benchmark")]
    # (title, data view, style, metric, metric field, bucket field)
    specs = {
        "cyber-overview": ("Cyber Threat Overview", "", [
            ("Network flows", "flows", "metric", "count", None, None),
            ("Bytes observed", "flows", "metric", "sum", "network.bytes", None),
            ("Source hosts", "flows", "metric", "cardinality", "source.ip", None),
            ("Traffic timeline", "flows", "area", "count", None, "@timestamp"),
            ("Ground-truth labels", "flows", "donut", "count", None, "reference_label"),
            ("Transport protocols", "flows", "donut", "count", None, "network.transport"),
            ("Top destination ports", "flows", "bar", "count", None, "destination.port"),
            ("Top source hosts", "flows", "bar", "count", None, "source.ip")]),
        "cyber-findings": ("Threat Findings", "flagged: true", [
            ("Flagged detector findings", "findings", "metric", "count", None, None),
            ("Affected source hosts", "findings", "metric", "cardinality", "source.ip", None),
            ("Findings over time", "findings", "area", "count", None, "@timestamp"),
            ("Findings by detector", "findings", "donut", "count", None, "detector"),
            ("Top source hosts", "findings", "bar", "count", None, "source.ip"),
            ("Mean detector score", "findings", "bar", "avg", "score", "detector"),
            ("Findings by split", "findings", "bar", "count", None, "split"),
            ("Reference attack labels", "findings", "donut", "count", None, "reference_label")]),
        "cyber-host": ("Host Investigation — filter source.ip", "", [
            ("Flow volume", "flows", "metric", "count", None, None),
            ("Bytes observed", "flows", "metric", "sum", "network.bytes", None),
            ("Host traffic over time", "flows", "area", "count", None, "@timestamp"),
            ("Top destination IPs", "flows", "bar", "count", None, "destination.ip"),
            ("Top destination ports", "flows", "bar", "count", None, "destination.port"),
            ("Transport mix", "flows", "donut", "count", None, "network.transport"),
            ("Unique destinations by window", "features", "line", "avg", "unique_destinations", "@timestamp"),
            ("Unique ports by window", "features", "line", "avg", "unique_ports", "@timestamp")]),
        "cyber-model": ("Detector & Model Diagnostics", "", [
            ("Scored detector windows", "findings", "metric", "count", None, None),
            ("Flagged finding records", "findings", "metric", "count", None, None),
            ("Findings over time", "findings", "area", "count", None, "@timestamp"),
            ("Flagged vs unflagged", "findings", "donut", "count", None, "flagged"),
            ("Mean score by detector", "findings", "bar", "avg", "score", "detector"),
            ("Evaluation split coverage", "findings", "bar", "count", None, "split"),
            ("Reference label support", "findings", "bar", "count", None, "reference_label"),
            ("Finding volume by detector", "findings", "bar", "count", None, "detector")]),
        "cyber-benchmark": ("Model Benchmark Report", "", [
            ("Dataset windows", "benchmark", "metric", "max", "dataset_window_count", None),
            ("Positive windows", "benchmark", "metric", "max", "dataset_positive_windows", None),
            ("Selected model grouped-CV AP", "benchmark", "metric", "max", "selected_mean_cv_average_precision", None),
            ("Selected model held-out test F1", "benchmark", "metric", "max", "selected_test_f1", None),
            ("Grouped-CV AP by model", "benchmark", "bar", "avg", "mean_cv_average_precision", "model_name"),
            ("Held-out test AP by model", "benchmark", "bar", "avg", "test_average_precision", "model_name"),
            ("Held-out test F1 by model", "benchmark", "bar", "avg", "test_f1", "model_name"),
            ("Held-out test precision by model", "benchmark", "bar", "avg", "test_precision", "model_name"),
            ("Held-out test recall by model", "benchmark", "bar", "avg", "test_recall", "model_name"),
            ("Train–CV AP gap by model", "benchmark", "bar", "avg", "train_cv_ap_gap", "model_name"),
            ("Selected-model confusion matrix", "benchmark", "table", "avg", "test_tp", "selected_model_name"),
            ("Model comparison table", "benchmark", "table", "avg", "mean_cv_average_precision", "model_name"),
            ("Learning-size diagnostics", "benchmark", "table", "avg", "cv_ap", "training_fraction")]),
    }
    objects = []
    for dashboard_id, (title, query, charts) in specs.items():
        panels, refs = [], []
        for i, (chart_title, kind, chart_type, metric, field, bucket) in enumerate(charts):
            vis_id = f"{dashboard_id}-panel-{i + 1}"
            aggs = [{"id": "1", "enabled": True, "type": metric, "schema": "metric",
                     "params": {"field": field} if field else {}}]
            if bucket:
                params = {"field": bucket, "size": 10, "order": "desc", "orderBy": "1"}
                agg_type = "terms"
                if bucket == "@timestamp":
                    agg_type = "date_histogram"
                    params = {"field": bucket, "interval": "auto", "min_doc_count": 1,
                              "extended_bounds": {}}
                aggs.append({"id": "2", "enabled": True, "type": agg_type,
                             "schema": "segment" if chart_type not in {"table", "metric"} else "bucket",
                             "params": params})
            if dashboard_id == "cyber-benchmark" and chart_type == "table":
                if chart_title == "Selected-model confusion matrix":
                    metric_fields = ["selected_test_tn", "selected_test_fp", "selected_test_fn", "selected_test_tp"]
                elif chart_title == "Model comparison table":
                    metric_fields = ["mean_cv_average_precision", "std_cv_average_precision",
                                     "test_average_precision", "test_f1", "test_precision", "test_recall",
                                     "test_fp", "test_fn"]
                else:
                    metric_fields = ["mean_train_windows", "mean_train_positive_windows", "train_ap", "cv_ap", "cv_ap_std"]
                bucket = aggs[-1]
                if chart_title == "Learning-size diagnostics":
                    bucket["params"].update({"order": "asc", "orderBy": "_key"})
                metric_ids = ["1", *[str(index + 2) for index in range(1, len(metric_fields))]]
                aggs = [{"id": metric_id, "enabled": True, "type": "avg", "schema": "metric",
                         "params": {"field": metric_field}}
                        for metric_id, metric_field in zip(metric_ids, metric_fields)] + [bucket]
            params = {"addTooltip": True, "addLegend": chart_type in {"donut", "pie", "area", "line"}}
            if chart_type in {"donut", "pie"}:
                params.update({"type": "pie", "isDonut": chart_type == "donut",
                               "labels": {"show": True, "values": True, "percent": True,
                                          "show v labels": True, "show h labels": False}})
                vis_type = "pie"
            elif chart_type == "metric":
                params.update({"type": "metric", "metric": {
                    "percentageMode": False, "useRanges": False,
                    "colorSchema": "Green to Red", "metricColorMode": "None",
                    "colorsRange": [{"from": 0, "to": 10000}],
                    "labels": {"show": True}, "invertColors": False,
                    "style": {"bgFill": "#000", "bgColor": False, "labelColor": False,
                              "subText": "", "fontSize": 48}}})
                vis_type = "metric"
            elif chart_type == "table":
                params.update({"perPage": 10, "showPartialRows": False, "showMetricsAtAllLevels": False,
                               "sort": {"columnIndex": 1, "direction": "desc"}, "showTotal": False})
                vis_type = "table"
            else:
                series_type = {"histogram": "histogram", "bar": "histogram", "line": "line", "area": "area"}[chart_type]
                params.update({"type": "histogram", "grid": {"categoryLines": False},
                               "categoryAxes": [{"id": "CategoryAxis-1", "type": "category", "position": "bottom",
                                   "show": True, "style": {}, "scale": {"type": "linear"},
                                   "labels": {"show": True, "truncate": 80}, "title": {}}],
                               "valueAxes": [{"id": "ValueAxis-1", "name": "LeftAxis-1", "type": "value",
                                   "position": "left", "show": True, "style": {},
                                   "scale": {"type": "linear", "mode": "normal"},
                                   "labels": {"show": True}, "title": {"text": metric}}],
                               "seriesParams": [{"show": True, "type": series_type, "mode": "normal",
                                   "data": {"id": "1", "label": metric}, "valueAxis": "ValueAxis-1",
                                   "drawLinesBetweenPoints": True, "showCircles": chart_type == "line"}]})
                vis_type = "histogram"
            vis_query = ""
            if dashboard_id == "cyber-model" and i == 1:
                vis_query = "flagged: true"
            elif dashboard_id == "cyber-benchmark":
                vis_query = ("doc_type: learning_curve" if chart_title == "Learning-size diagnostics"
                             else "doc_type: benchmark_summary" if i < 4 or chart_title == "Selected-model confusion matrix"
                             else "doc_type: benchmark_model")
            objects.append({"type": "visualization", "id": vis_id,
                            "attributes": {"title": chart_title, "description": "Cyber threat lab aggregate",
                                           "version": 1, "uiStateJSON": "{}",
                                           "visState": json.dumps({"title": chart_title, "type": vis_type,
                                                                   "params": params, "aggs": aggs}),
                                           "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps({
                                               "indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index",
                                               "query": {"language": "kuery", "query": vis_query}, "filter": []})}},
                            "references": [{"type": "index-pattern", "id": f"cyber-{kind}-view",
                                            "name": "kibanaSavedObjectMeta.searchSourceJSON.index"}]})
            name = f"panel_{i + 1}"
            refs.append({"type": "visualization", "id": vis_id, "name": name})
            panels.append({"version": "8.17.0", "type": "visualization", "panelIndex": str(i + 1),
                           "panelRefName": name, "embeddableConfig": {},
                           "gridData": {"x": (i % 2) * 24, "y": (i // 2) * 15,
                                        "w": 24, "h": 15, "i": str(i + 1)}})
        time_from = "now-30d" if dashboard_id == "cyber-benchmark" else "2017-07-06T00:00:00.000Z"
        time_to = "now" if dashboard_id == "cyber-benchmark" else "2017-07-07T23:59:59.999Z"
        description = (
            "Grouped-CV model benchmark report. The available Thursday slice has only 68 positive windows from one source host and one continuous attack period; grouped buckets are shuffled, not a forward-time test. Treat results as exploratory."
            if dashboard_id == "cyber-benchmark" else
            "Historical data included. Use KQL source.ip and detector/split filters. Reference labels are ground truth, not predictions."
        )
        objects.append({"type": "dashboard", "id": dashboard_id, "references": refs,
                        "attributes": {"title": f"🛡️ {title}", "description": description,
                                       "version": 1, "panelsJSON": json.dumps(panels),
                                       "optionsJSON": json.dumps({"useMargins": True, "hidePanelTitles": False}),
                                       "timeRestore": True, "timeFrom": time_from, "timeTo": time_to,
                                       "refreshInterval": {"pause": True, "value": 0},
                                       "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps({
                                           "query": {"language": "kuery", "query": query}, "filter": []})}}})
    return views, objects


def provision(url, output=None):
    """Create/replace the lab objects, verify each via GET, optionally export.

    The saved-object CRUD API is deprecated but supported in the pinned 8.17
    release. An actual server export is preserved verbatim, including migration
    metadata; we never pretend our authored JSON is a Kibana-generated export.
    HTTP, verification, and export failures propagate to produce a nonzero CLI exit.
    """
    from pathlib import Path
    from urllib.parse import quote
    import requests

    url = url.rstrip("/")
    views, objects = build_saved_objects()
    with requests.Session() as session:
        session.headers.update({"kbn-xsrf": "cyber-threat-lab", "Content-Type": "application/json"})

        def request(method, path, payload=None):
            response = session.request(method, url + path, json=payload, timeout=60)
            response.raise_for_status()
            return response

        for view in views:
            request("POST", "/api/data_views/data_view", {"data_view": view, "override": True})
            actual = request("GET", "/api/data_views/data_view/" + quote(view["id"], safe="")).json()["data_view"]
            if any(actual.get(key) != view[key] for key in ("id", "title", "timeFieldName")):
                raise RuntimeError(f"Data view read-back mismatch: {view['id']}")
        for obj in objects:
            path = "/api/saved_objects/" + obj["type"] + "/" + quote(obj["id"], safe="")
            payload = {"attributes": obj["attributes"], "references": obj["references"]}
            request("POST", path + "?overwrite=true", payload)
            actual = request("GET", path).json()
            if (actual.get("id") != obj["id"] or actual.get("type") != obj["type"]
                    or any(actual.get("attributes", {}).get(k) != v for k, v in obj["attributes"].items())
                    or sorted(actual.get("references", []), key=lambda r: r["name"])
                    != sorted(obj["references"], key=lambda r: r["name"])):
                raise RuntimeError(f"Saved object read-back mismatch: {obj['id']}")
        if output is not None:
            response = request("POST", "/api/saved_objects/_export", {
                "objects": [{"type": "dashboard", "id": obj["id"]} for obj in objects if obj["type"] == "dashboard"],
                "includeReferencesDeep": True, "excludeExportDetails": True})
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
    return {"data_views": len(views),
            "visualizations": sum(o["type"] == "visualization" for o in objects),
            "dashboards": sum(o["type"] == "dashboard" for o in objects)}


def parse_args(argv=None):
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("KIBANA_URL", "http://localhost:15601"),
                        help="Kibana base URL (default KIBANA_URL or http://localhost:15601)")
    parser.add_argument("--output", help="Optional server-generated saved objects NDJSON artifact path")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(provision(args.url, args.output), sort_keys=True))
