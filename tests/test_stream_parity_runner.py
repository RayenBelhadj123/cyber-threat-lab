import pytest

from scripts.run_stream_parity import marker_timestamp, max_offsets_behind_latest, run_names


def test_run_names_are_isolated_and_docker_safe():
    names = run_names("trial-20261020")
    assert names["flow_topic"] == "cyber-flows-parity-trial-20261020"
    assert names["prediction_topic"] == "cyber-predictions-parity-trial-20261020"
    assert names["checkpoint"] == "trial-20261020"
    assert names["container"] == "cyber-threat-lab-spark-stream-parity-trial-20261020"


def test_replay_environment_exposes_project_package_to_script():
    from scripts.run_stream_parity import replay_environment
    env = replay_environment({"PYTHONPATH": "existing-path"})
    paths = env["PYTHONPATH"].split(__import__("os").pathsep)
    assert str(__import__("pathlib").Path(__file__).resolve().parents[1]) in paths
    assert "existing-path" in paths


def test_run_names_reject_path_or_topic_injection():
    with pytest.raises(ValueError):
        run_names("../unsafe")


def test_parity_runner_refuses_reused_topics_or_checkpoint(tmp_path):
    from scripts.run_stream_parity import validate_fresh_run
    names = run_names("clean-001")
    checkpoint_root = tmp_path / "checkpoints"
    validate_fresh_run(names, checkpoint_root, tmp_path, set())
    with pytest.raises(ValueError):
        validate_fresh_run(names, checkpoint_root, tmp_path, {names["flow_topic"]})
    (checkpoint_root / names["checkpoint"]).mkdir(parents=True)
    with pytest.raises(ValueError):
        validate_fresh_run(names, checkpoint_root, tmp_path, set())


def test_catchup_requires_active_error_free_queries_and_zero_lag():
    progress = {
        "status": "running",
        "queries": [
            {"active": True, "exception": None, "last_progress": {"sources": [{"metrics": {"maxOffsetsBehindLatest": "0"}}]}},
            {"active": True, "exception": None, "last_progress": {"sources": [{"metrics": {"maxOffsetsBehindLatest": 0}}]}},
        ],
    }
    assert max_offsets_behind_latest(progress) == 0
    progress["queries"][1]["last_progress"]["sources"][0]["metrics"]["maxOffsetsBehindLatest"] = "3"
    assert max_offsets_behind_latest(progress) == 3
    progress["queries"][1]["active"] = False
    assert max_offsets_behind_latest(progress) is None


def test_watermark_marker_follows_latest_flow_by_sixteen_minutes():
    assert marker_timestamp("2017-07-06T15:59:00+00:00") == "2017-07-06T16:15:00+00:00"


def test_default_stream_restart_requires_new_active_metrics():
    from scripts.run_stream_parity import default_stream_ready
    metrics = {"status": "running", "observed_at_epoch_seconds": 12,
               "queries": [{"active": True, "exception": None}, {"active": True, "exception": None}]}
    assert default_stream_ready(metrics, observed_after=10)
    assert not default_stream_ready(metrics, observed_after=12)
    metrics["queries"][1]["active"] = False
    assert not default_stream_ready(metrics, observed_after=10)
