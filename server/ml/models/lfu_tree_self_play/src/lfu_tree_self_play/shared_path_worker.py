"""One disposable process measuring exactly one H1 mode and repetition."""

import gc
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from .batch_inference import FrozenBatchEvaluator
from .batch_scheduler import BatchScheduler, SchedulerLimits, SystemMemoryProbe
from .collection_session import CollectionLane
from .dataset_codec import encode_unit
from .game import ReversiPyGame
from .model import PolicyValueConfig, PolicyValueModel
from .shared_path_benchmark import BenchmarkConfig, _check_device, _load_tree, _sha256
from .tree_replay import TreeReplay, compare_trees, finalize_replay


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _wait_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": statistics.median(values) if values else 0.0,
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        if values
        else 0.0,
        "max": max(values, default=0.0),
    }


def _generate(evaluator, trees, mode, config, device):
    owners = [TreeReplay(evaluator) for _ in trees]
    lanes = [
        CollectionLane(owner, iter((owner.create_session(tree, mode=mode),)))
        for owner, tree in zip(owners, trees, strict=True)
    ]
    probe = SystemMemoryProbe(torch.device(device))
    scheduler = BatchScheduler(
        evaluator,
        limits=SchedulerLimits(
            max_active_units=config.max_active_units,
            max_batch_size=config.max_batch_size,
            rss_limit_bytes=config.rss_limit_bytes,
            device_limit_bytes=config.device_limit_bytes if device == "mps" else None,
        ),
        memory_probe=probe,
    )
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    _synchronize(device)
    if device == "mps":
        torch.accelerator.reset_peak_memory_stats(torch.device(device))
    baseline = probe.sample()
    outputs = []
    started = time.perf_counter()
    metrics = scheduler.run(lanes, on_complete=outputs.append)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if (
        metrics.stop_reason != "complete"
        or metrics.completed_units != len(trees)
        or metrics.discarded_units
        or len(outputs) != len(trees)
    ):
        raise RuntimeError(f"incomplete generation: {asdict(metrics)}")
    if metrics.rss_sampled_peak_bytes > config.rss_limit_bytes:
        raise RuntimeError("RSS limit exceeded")
    if (
        device == "mps"
        and max(
            metrics.device_allocated_sampled_peak_bytes or 0,
            metrics.device_peak_bytes or 0,
        )
        > config.device_limit_bytes
    ):
        raise RuntimeError("accelerator memory limit exceeded")
    row = asdict(metrics)
    row["request_wait_seconds"] = _wait_summary(row["request_wait_seconds"])
    row.update(
        {
            "elapsed_generation_seconds": elapsed,
            "baseline_rss_bytes": baseline.rss_bytes,
            "baseline_device_allocated_bytes": baseline.device_allocated_bytes,
            "environment_transitions": sum(
                output.record.cost.environment_transitions for output in outputs
            ),
        }
    )
    return outputs, row


def run_worker(directory: Path, device: str, mode: str, repeat: int) -> dict:
    row = {
        "passed": False,
        "errors": [],
        "mode": mode,
        "repeat": repeat,
        "worker_pid": os.getpid(),
    }
    try:
        _check_device(device)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["device"] != device:
            raise ValueError("worker device differs from fixed input")
        settings = manifest["config"]
        settings["seeds"] = tuple(settings["seeds"])
        config = BenchmarkConfig(**settings)
        if (
            _sha256(directory / "model.safetensors")
            != manifest["metadata"]["model_sha256"]
        ):
            raise ValueError("fixed model checksum mismatch")
        with torch.random.fork_rng(devices=[]):
            model = (
                PolicyValueModel(
                    PolicyValueConfig(width=config.width, num_blocks=config.num_blocks)
                )
                .float()
                .eval()
            )
        model.load_state_dict(load_file(directory / "model.safetensors"))
        model.to(device)
        evaluator = FrozenBatchEvaluator(model, model_generation=0)
        del model
        game = ReversiPyGame()
        trees = []
        for unit in manifest["units"]:
            path = directory / unit["file"]
            if _sha256(path) != unit["sha256"]:
                raise ValueError("fixed reference checksum mismatch")
            trees.append(_load_tree(path, game))
        for _ in range(config.warmups):
            warmup_outputs, warmup_metrics = _generate(
                evaluator, trees, mode, config, device
            )
            del warmup_outputs, warmup_metrics
            gc.collect()
        outputs, measured = _generate(evaluator, trees, mode, config, device)
        row.update(measured)
        row["warmups_completed"] = config.warmups
        row["runtime"] = {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "accelerator_peak_method": "torch.accelerator.max_memory_allocated after reset"
            if device == "mps"
            else None,
        }
        reference_by_id = {tree.record.unit_id: tree for tree in trees}
        actual_ids = [raw.record.unit_id for raw in outputs]
        if (
            len(set(actual_ids)) != len(actual_ids)
            or set(actual_ids) != reference_by_id.keys()
        ):
            raise ValueError("measured output identities differ from fixed references")
        identity = {
            "passed": True,
            "max_value_error": 0.0,
            "max_log_probability_error": 0.0,
        }
        row["outputs"] = []
        for raw in outputs:
            actual = finalize_replay(raw, game)
            comparison = compare_trees(reference_by_id[actual.record.unit_id], actual)
            for key in ("max_value_error", "max_log_probability_error"):
                identity[key] = max(identity[key], comparison[key])
            path = directory / f"actual-{repeat}-{mode}-{actual.record.unit_id}.json"
            path.write_bytes(encode_unit(actual.record, actual.targets, game))
            row["outputs"].append(
                {
                    "file": path.name,
                    "sha256": _sha256(path),
                    "unit_id": actual.record.unit_id,
                }
            )
        row["identity"] = identity
        row["passed"] = True
    # A disposable worker must return evidence that the coordinator can persist.
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"{type(exc).__name__}: {exc}")
    return row
