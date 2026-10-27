"""Check shared-path identity, process isolation, and generation cost reporting."""

import json
import os
import subprocess

import pytest
import torch

from lfu_tree_self_play.shared_path_benchmark import (
    BenchmarkConfig,
    _prepare,
    verify_shared_paths,
)


@pytest.mark.parametrize(
    "settings",
    [
        {"seeds": ()},
        {"seeds": (-1,)},
        {"seeds": (True,)},
        {"seeds": (3, 3)},
        {"roots_per_seed": 0},
        {"repeats": 0},
        {"warmups": -1},
        {"width": 0},
        {"num_blocks": -1},
        {"levels": 3},
        {"strategy": "unknown"},
        {"max_active_units": 33},
        {"max_batch_size": 0},
        {"rss_limit_bytes": 0},
        {"device_limit_bytes": 3 * 1024**3},
    ],
)
def test_rejects_invalid_or_out_of_budget_configuration(settings):
    # Without validation an invalid measurement could be labeled as evidence.
    with pytest.raises((TypeError, ValueError)):
        BenchmarkConfig(**settings)


def test_cpu_model_preparation_preserves_unowned_accelerator_rng(monkeypatch, tmp_path):
    # CPU model initialization must not reseed an existing accelerator session.
    def reject_accelerator_reseed(seed):
        raise AssertionError("unowned accelerator RNG was reseeded")

    monkeypatch.setattr(torch.mps, "manual_seed", reject_accelerator_reseed)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", reject_accelerator_reseed)
    before = torch.get_rng_state().clone()
    manifest = _prepare(
        tmp_path,
        "cpu",
        BenchmarkConfig(
            seeds=(7,),
            roots_per_seed=1,
            repeats=1,
            warmups=0,
            width=4,
            num_blocks=0,
            levels=0,
        ),
    )
    assert torch.equal(before, torch.get_rng_state())
    assert len(manifest["metadata"]["model_sha256"]) == 64


def test_small_cpu_benchmark_regenerates_equal_outputs_in_fresh_workers():
    # Reusing reference outputs or process-lifetime counters must fail this check.
    before = torch.get_rng_state().clone()
    report = verify_shared_paths(
        device="cpu",
        config=BenchmarkConfig(
            seeds=(7,),
            roots_per_seed=2,
            repeats=2,
            warmups=1,
            width=4,
            num_blocks=0,
            levels=1,
        ),
    )
    assert report["passed"], report["errors"]
    assert report["errors"] == []
    assert torch.equal(before, torch.get_rng_state())
    assert len(report["metadata"]["roots"]) == 2
    assert len(report["metadata"]["model_sha256"]) == 64
    assert report["metadata"]["dtype"] == "float32"
    rows = report["measurements"]
    assert [row["mode"] for row in rows] == [
        "shared",
        "recompute",
        "recompute",
        "shared",
    ]
    assert len({row["worker_pid"] for row in rows}) == 4
    assert all(row["worker_pid"] != os.getpid() for row in rows)
    for row in rows:
        assert row["elapsed_generation_seconds"] > 0
        assert row["environment_transitions"] > 0
        assert row["evaluated_positions"] > 0
        assert row["inference_batches"] > 0
        assert row["baseline_rss_bytes"] > 0
        assert row["rss_sampled_peak_bytes"] >= row["baseline_rss_bytes"]
        assert row["device_peak_bytes"] is None
        assert row["completed_units"] == 2
        assert row["warmups_completed"] == 1
        assert row["identity"]["passed"]
        assert row["identity"]["max_value_error"] <= 1e-6
        assert row["identity"]["max_log_probability_error"] <= 1e-6
        assert (
            sum(int(size) * count for size, count in row["batch_sizes"].items())
            == row["evaluated_positions"]
        )
        assert sum(row["batch_sizes"].values()) == row["inference_batches"]
        assert row["request_wait_seconds"]["count"] == row["evaluated_positions"]
    assert len(report["pairs"]) == 2
    for pair in report["pairs"]:
        assert pair["identity"]["passed"]
        assert (
            pair["shared_environment_transitions"]
            < pair["recompute_environment_transitions"]
        )
        assert (
            pair["shared_evaluated_positions"] < pair["recompute_evaluated_positions"]
        )
    json.dumps(report, allow_nan=False)


def test_failed_worker_process_is_reported_without_stale_success(monkeypatch):
    # Losing a subprocess must fail the overall report even if other repeats run.
    def failed_process(*args, **kwargs):
        raise subprocess.CalledProcessError(9, args[0], stderr="worker terminated")

    monkeypatch.setattr(subprocess, "run", failed_process)
    report = verify_shared_paths(
        device="cpu",
        config=BenchmarkConfig(
            seeds=(7,),
            roots_per_seed=1,
            repeats=1,
            warmups=0,
            width=4,
            num_blocks=0,
            levels=0,
        ),
    )
    assert report["passed"] is False
    assert len(report["measurements"]) == 2
    assert all(row["passed"] is False for row in report["measurements"])
    assert any("worker terminated" in error for error in report["errors"])
    assert report["pairs"] == []
    json.dumps(report, allow_nan=False)


def test_actual_worker_memory_limit_failure_remains_a_failed_report():
    # A process over budget cannot turn into a successful empty measurement.
    report = verify_shared_paths(
        device="cpu",
        config=BenchmarkConfig(
            seeds=(7,),
            roots_per_seed=1,
            repeats=1,
            warmups=0,
            width=4,
            num_blocks=0,
            levels=0,
            rss_limit_bytes=1,
        ),
    )
    assert report["passed"] is False
    assert len(report["measurements"]) == 2
    assert all(row["passed"] is False for row in report["measurements"])
    assert any(
        "memory" in error.lower() or "rss" in error.lower()
        for error in report["errors"]
    )
    json.dumps(report, allow_nan=False)


def test_unavailable_mps_returns_a_failed_serializable_report(monkeypatch):
    # An unavailable accelerator must never silently run the CPU benchmark.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    report = verify_shared_paths(device="mps")
    assert report["passed"] is False
    assert report["measurements"] == []
    assert report["errors"]
    assert report["device"] == "mps"
    json.dumps(report, allow_nan=False)
