"""Run Spark batch analytics over normalized flows and detector predictions."""
import argparse
import json
import os
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from spark_jobs.common import sink_partition


FEATURE_COLUMNS = {
    "window_id", "flow_count", "unique_destinations", "unique_ports",
    "bytes_total", "packets_total", "duration_mean", "rate_mean",
    "failure_ratio", "malicious", "reference_label", "@timestamp", "source",
}


def build_features(spark, flows_path):
    flows = spark.read.json(str(flows_path))
    required = {"@timestamp", "source", "destination", "network", "duration", "rate", "malicious", "reference_label"}
    missing = required - set(flows.columns)
    if missing:
        raise ValueError(f"Normalized flow artifact is missing fields: {sorted(missing)}")
    event_time = F.to_timestamp(F.col("`@timestamp`"))
    valid = flows.withColumn("event_time", event_time).filter(
        F.col("event_time").isNotNull() & F.col("source.ip").isNotNull()
    )
    grouped = valid.groupBy(
        F.window("event_time", "1 minute").alias("window"),
        F.col("source.ip").alias("source_ip"),
    ).agg(
        F.count(F.lit(1)).alias("flow_count"),
        F.approx_count_distinct("destination.ip").alias("unique_destinations"),
        F.approx_count_distinct("destination.port").alias("unique_ports"),
        F.sum(F.col("network.bytes")).alias("bytes_total"),
        F.sum(F.col("network.packets")).alias("packets_total"),
        F.avg("duration").alias("duration_mean"),
        F.avg("rate").alias("rate_mean"),
        F.max(F.col("malicious").cast("int")).alias("malicious_int"),
        F.sort_array(F.collect_set("reference_label")).alias("labels"),
    )
    with_ids = grouped.withColumn(
        "window_id",
        F.sha2(F.concat_ws("|", F.col("source_ip"), F.col("window.start").cast("string")), 256),
    )
    return with_ids.select(
        F.col("window.end").alias("@timestamp"),
        F.col("window.start").alias("window_start"),
        F.struct(F.col("source_ip").alias("ip")).alias("source"),
        "window_id", "flow_count", "unique_destinations", "unique_ports",
        "bytes_total", "packets_total", "duration_mean", "rate_mean",
        F.lit(0.0).alias("failure_ratio"),
        (F.col("malicious_int") > 0).alias("malicious"),
        F.concat_ws("|", F.col("labels")).alias("reference_label"),
        F.lit("spark_batch").alias("pipeline_mode"),
    )


def persist_dataframe(frame, collection, index, kind):
    def write_partition(rows):
        sink_partition(rows, collection, index, kind)
    frame.foreachPartition(write_partition)


def run(artifact_dir, output_path):
    root = Path(artifact_dir)
    flow_path = root / "flows.ndjson"
    predictions_path = root / "predictions.ndjson"
    if not flow_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Expected flows.ndjson and predictions.ndjson in {root}")

    started = time.monotonic()
    spark = (SparkSession.builder.appName("cyberlab-batch")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        features = build_features(spark, flow_path).persist()
        predictions = spark.read.json(str(predictions_path)).persist()
        if "window_id" not in predictions.columns or "detector" not in predictions.columns:
            raise ValueError("Predictions artifact must contain window_id and detector")
        feature_count = features.count()
        prediction_count = predictions.count()
        if feature_count == 0:
            raise ValueError("Spark batch produced no one-minute source windows")
        persist_dataframe(features, "batch_features", "cyber-features-spark-batch-v1", "batch_feature")
        persist_dataframe(predictions, "batch_predictions", "cyber-findings-spark-batch-v1", "batch_prediction")
        report = {
            "pipeline": "spark-batch", "feature_windows": feature_count,
            "prediction_records": prediction_count,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "spark_version": spark.version,
            "executor_count": max(0, len(spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()) - 1),
            "indices": ["cyber-features-spark-batch-v1", "cyber-findings-spark-batch-v1"],
            "stores": ["elasticsearch", "mongodb", "cassandra"],
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return report
    finally:
        spark.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="/app/artifacts")
    parser.add_argument("--output", default="/app/artifacts/spark_batch_metrics.json")
    args = parser.parse_args(argv)
    run(args.artifacts, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
