"""Consume Kafka flow/prediction topics and persist Spark micro-batch outputs."""
import argparse
import json
import os
import threading
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, IntegerType, MapType, StringType, StructField,
    StructType,
)

from spark_jobs.common import sink_partition


FLOW_SCHEMA = StructType([
    StructField("@timestamp", StringType()),
    StructField("event", StructType([StructField("id", StringType())])),
    StructField("source", StructType([StructField("ip", StringType()), StructField("port", IntegerType())])),
    StructField("destination", StructType([StructField("ip", StringType()), StructField("port", IntegerType())])),
    StructField("network", StructType([
        StructField("bytes", DoubleType()), StructField("packets", DoubleType()),
        StructField("transport", StringType()),
    ])),
    StructField("duration", DoubleType()),
    StructField("rate", DoubleType()),
    StructField("malicious", BooleanType()),
    StructField("reference_label", StringType()),
])

PREDICTION_SCHEMA = StructType([
    StructField("@timestamp", StringType()),
    StructField("event", StructType([StructField("id", StringType())])),
    StructField("source", StructType([StructField("ip", StringType())])),
    StructField("window_id", StringType()),
    StructField("detector", StringType()),
    StructField("score", DoubleType()),
    StructField("threshold", DoubleType()),
    StructField("flagged", BooleanType()),
    StructField("status", StringType()),
    StructField("reason", StringType()),
    StructField("split", StringType()),
    StructField("malicious", BooleanType()),
    StructField("reference_label", StringType()),
    StructField("features", MapType(StringType(), DoubleType(), True)),
])


def _decode_kafka(spark, topic, schema):
    options = {
        "kafka.bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka-1:9092,kafka-2:9092,kafka-3:9092"),
        "subscribe": topic,
        "startingOffsets": os.environ.get("KAFKA_STARTING_OFFSETS", "earliest"),
        "maxOffsetsPerTrigger": os.environ.get("KAFKA_MAX_OFFSETS_PER_TRIGGER", "10000"),
    }
    raw = spark.readStream.format("kafka").options(**options).load()
    return raw.select(F.from_json(F.col("value").cast("string"), schema).alias("doc")) \
        .filter(F.col("doc").isNotNull()).select("doc.*")


def _feature_windows(flows):
    event_time = F.to_timestamp(F.col("`@timestamp`"))
    valid = flows.withColumn("event_time", event_time).filter(
        F.col("event_time").isNotNull() & F.col("source.ip").isNotNull()
    ).withWatermark("event_time", os.environ.get("SPARK_WATERMARK", "10 minutes"))
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
        F.first("reference_label", ignorenulls=True).alias("reference_label"),
    )
    return grouped.withColumn(
        "window_id", F.sha2(F.concat_ws("|", F.col("source_ip"), F.col("window.start").cast("string")), 256)
    ).select(
        F.col("window.end").alias("@timestamp"), F.col("window.start").alias("window_start"),
        F.struct(F.col("source_ip").alias("ip")).alias("source"),
        "window_id", "flow_count", "unique_destinations", "unique_ports",
        "bytes_total", "packets_total", "duration_mean", "rate_mean",
        F.lit(0.0).alias("failure_ratio"), (F.col("malicious_int") > 0).alias("malicious"),
        "reference_label", F.lit("spark_stream").alias("pipeline_mode"),
    )


def _sink_batch(frame, collection, index, kind):
    def write_partition(rows):
        sink_partition(rows, collection, index, kind)
    frame.foreachPartition(write_partition)


def _report(path, queries, status):
    snapshots = []
    for query in queries:
        snapshots.append({
            "name": query.name,
            "active": query.isActive,
            "last_progress": query.lastProgress,
            "exception": str(query.exception()) if query.exception() else None,
        })
    payload = {"pipeline": "spark-streaming", "status": status,
               "observed_at_epoch_seconds": time.time(), "queries": snapshots}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def run(checkpoint_dir, metrics_path):
    started = time.monotonic()
    spark = (SparkSession.builder.appName("cyberlab-streaming")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    checkpoints = Path(checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    queries = []
    stop_monitor = threading.Event()
    monitor_thread = None
    try:
        flows = _decode_kafka(spark, os.environ.get("KAFKA_FLOW_TOPIC", "cyber-flows"), FLOW_SCHEMA)
        features = _feature_windows(flows)
        feature_query = (features.writeStream.queryName("cyber-flow-windows")
                         .outputMode("append")
                         .option("checkpointLocation", str(checkpoints / "flow-windows"))
                         .foreachBatch(lambda frame, epoch: _sink_batch(
                             frame, "stream_features", "cyber-features-spark-stream-v1", "stream_feature"))
                         .trigger(processingTime=os.environ.get("SPARK_TRIGGER_INTERVAL", "5 seconds"))
                         .start())
        queries.append(feature_query)

        predictions = _decode_kafka(spark, os.environ.get("KAFKA_PREDICTION_TOPIC", "cyber-predictions"), PREDICTION_SCHEMA) \
            .filter(F.col("window_id").isNotNull() & F.col("detector").isNotNull())
        prediction_query = (predictions.writeStream.queryName("cyber-predictions")
                            .outputMode("append")
                            .option("checkpointLocation", str(checkpoints / "predictions"))
                            .foreachBatch(lambda frame, epoch: _sink_batch(
                                frame, "stream_predictions", "cyber-findings-spark-stream-v1", "stream_prediction"))
                            .trigger(processingTime=os.environ.get("SPARK_TRIGGER_INTERVAL", "5 seconds"))
                            .start())
        queries.append(prediction_query)

        report_path = Path(metrics_path)
        _report(report_path, queries, "running")

        def monitor():
            while not stop_monitor.wait(15):
                _report(report_path, queries, "running")

        monitor_thread = threading.Thread(target=monitor, name="spark-progress-writer", daemon=True)
        monitor_thread.start()
        print(json.dumps({"pipeline": "spark-streaming", "status": "running",
                          "queries": [query.name for query in queries],
                          "elapsed_startup_seconds": round(time.monotonic() - started, 3)}), flush=True)
        spark.streams.awaitAnyTermination()
        failures = [query for query in queries if query.exception()]
        if failures:
            raise RuntimeError("Spark streaming query failed: " + "; ".join(str(q.exception()) for q in failures))
    finally:
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)
        for query in queries:
            if query.isActive:
                query.stop()
        if queries:
            _report(Path(metrics_path), queries, "stopped")
        spark.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="/app/checkpoints")
    parser.add_argument("--metrics", default="/app/artifacts/spark_stream_metrics.json")
    args = parser.parse_args(argv)
    run(args.checkpoints, args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
