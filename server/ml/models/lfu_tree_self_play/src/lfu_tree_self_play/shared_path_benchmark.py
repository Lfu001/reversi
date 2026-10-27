"""Compare generation costs of shared paths and repeated root-to-leaf replay."""

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from .batch_inference import FrozenBatchEvaluator
from .dataset_codec import decode_unit, encode_unit
from .game import ReversiPyGame
from .model import PolicyValueConfig, PolicyValueModel
from .roots import generate_root_position
from .schedule import summarize_branch_execution
from .tree_collector import CollectedTree, TreeCollector
from .tree_replay import compare_trees


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple[int, ...] = (314159, 271828, 161803)
    roots_per_seed: int = 4
    repeats: int = 3
    warmups: int = 1
    width: int = 64
    num_blocks: int = 4
    levels: int = 2
    strategy: str = "uniform"
    max_active_units: int = 32
    max_batch_size: int = 32
    rss_limit_bytes: int = 2 * 1024**3
    device_limit_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.seeds, tuple)
            or not self.seeds
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError(
                "seeds must be a nonempty tuple of distinct nonnegative integers"
            )
        for name in (
            "roots_per_seed",
            "repeats",
            "width",
            "max_active_units",
            "max_batch_size",
            "rss_limit_bytes",
            "device_limit_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmups", "num_blocks"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.levels) is not int or not 0 <= self.levels <= 2:
            raise ValueError("levels must be an integer from zero to two")
        if self.strategy != "uniform":
            raise ValueError(
                "the shared-path benchmark requires uniform branch positions"
            )
        if self.max_active_units > 32 or self.max_batch_size > 32:
            raise ValueError(
                "the shared-path benchmark is limited to 32 active units and batch size 32"
            )
        if max(self.rss_limit_bytes, self.device_limit_bytes) > 2 * 1024**3:
            raise ValueError(
                "the shared-path benchmark memory limits cannot exceed 2 GiB"
            )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derived_seed(seed: int, purpose: str, index: int) -> int:
    label = f"lfu-h1-v1/{purpose}/{seed}/{index}".encode()
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big") % (2**63)


def _check_device(device: str) -> None:
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; CPU fallback is disabled")
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS fallback must be disabled before benchmarking")


def _load_tree(path: Path, game: ReversiPyGame) -> CollectedTree:
    unit = decode_unit(path.read_bytes(), game)
    return CollectedTree(
        unit.record, unit.targets, summarize_branch_execution(unit.record, game)
    )


def _prepare(directory: Path, device: str, config: BenchmarkConfig) -> dict:
    game = ReversiPyGame()
    model_seed = _derived_seed(config.seeds[0], "model", 0)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed)
        model = (
            PolicyValueModel(
                PolicyValueConfig(width=config.width, num_blocks=config.num_blocks)
            )
            .float()
            .eval()
        )
    save_file(model.state_dict(), directory / "model.safetensors")
    model.to(device)
    evaluator = FrozenBatchEvaluator(model, model_generation=0)
    roots, units = [], []
    for seed in config.seeds:
        for index in range(config.roots_per_seed):
            root_seed = _derived_seed(seed, "root", index)
            collection_seed = _derived_seed(seed, "collection", index)
            root = generate_root_position(root_seed)
            unit_id = f"h1-{seed}-{index}"
            collector = TreeCollector(
                evaluator,
                experiment_id="h1-shared-paths",
                round_id="fixed-input",
                model_generation=0,
                run_seed=collection_seed,
                levels=config.levels,
                strategy=config.strategy,
                game=game,
            )
            tree = collector.collect_tree(root, unit_id=unit_id)
            path = directory / f"{unit_id}.json"
            path.write_bytes(encode_unit(tree.record, tree.targets, game))
            units.append({"file": path.name, "sha256": _sha256(path)})
            roots.append(
                {
                    "seed": seed,
                    "index": index,
                    "root_seed": root_seed,
                    "collection_seed": collection_seed,
                    "placements": root.placements,
                    "actions": list(root.actions),
                    "key": root.key,
                    "unit_id": unit_id,
                    "branch_positions": list(tree.record.branch_plan.positions),
                    "terminal_leaf_count": tree.terminal_leaf_count,
                    "nodes": len(tree.record.nodes),
                    "edges": len(tree.record.edges),
                    "reference_sha256": _sha256(path),
                }
            )
    manifest = {
        "config": asdict(config),
        "device": device,
        "units": units,
        "metadata": {
            "roots": roots,
            "model_seed": model_seed,
            "model_sha256": _sha256(directory / "model.safetensors"),
            "model_config": {"width": config.width, "num_blocks": config.num_blocks},
            "dtype": "float32",
            "branch_width": 4,
            "seed_derivation": "sha256(utf8(lfu-h1-v1/{purpose}/{seed}/{index}))[:8], big endian, modulo 2**63",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "mps_available": torch.backends.mps.is_available(),
            "fallback_enabled": False,
            "memory_scope": "worker RSS and accelerator allocation; scheduler boundary sampling plus reset native accelerator peak",
            "generation_interval": "scheduler.run only; excludes preparation, validation, aggregation and persistence",
        },
    }
    _write_json(directory / "manifest.json", manifest)
    del tree, collector, evaluator, model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return manifest


def _run_worker(directory: Path, device: str, mode: str, repeat: int) -> dict:
    output = directory / f"measurement-{repeat}-{mode}.json"
    command = [
        sys.executable,
        "-m",
        "lfu_tree_self_play.shared_path_benchmark",
        "--worker",
        str(directory),
        "--device",
        device,
        "--mode",
        mode,
        "--repeat",
        str(repeat),
        "--output",
        str(output),
    ]
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    error = None
    try:
        subprocess.run(
            command,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError):
            error += f"; stderr: {exc.stderr[-2000:]}"
    try:
        row = json.loads(output.read_text())
        if not isinstance(row, dict) or not isinstance(row.get("passed"), bool):
            raise TypeError("invalid worker report shape")
    except (OSError, TypeError, ValueError) as exc:
        row = {"passed": False, "errors": [f"missing/invalid worker report: {exc}"]}
    row.update({"mode": mode, "repeat": repeat})
    if error:
        row["passed"] = False
        row.setdefault("errors", []).append(error)
    return row


def _compare_pair(directory: Path, rows: list[dict], repeat: int) -> dict:
    shared, recompute = (
        next(row for row in rows if row["mode"] == mode)
        for mode in ("shared", "recompute")
    )
    if not shared["passed"] or not recompute["passed"]:
        raise ValueError(f"repeat {repeat}: a worker failed")
    if shared["worker_pid"] == recompute["worker_pid"]:
        raise ValueError("measured modes did not run in distinct processes")
    game = ReversiPyGame()
    identity = {
        "passed": True,
        "max_value_error": 0.0,
        "max_log_probability_error": 0.0,
    }
    if len(shared["outputs"]) != len(recompute["outputs"]):
        raise ValueError("paired workers produced different unit counts")
    shared_outputs = {item["unit_id"]: item for item in shared["outputs"]}
    recompute_outputs = {item["unit_id"]: item for item in recompute["outputs"]}
    if (
        len(shared_outputs) != len(shared["outputs"])
        or len(recompute_outputs) != len(recompute["outputs"])
        or shared_outputs.keys() != recompute_outputs.keys()
    ):
        raise ValueError(
            "paired workers produced different or duplicate unit identities"
        )
    for unit_id in sorted(shared_outputs):
        left, right = shared_outputs[unit_id], recompute_outputs[unit_id]
        for item in (left, right):
            if _sha256(directory / item["file"]) != item["sha256"]:
                raise ValueError("measured output checksum mismatch")
        comparison = compare_trees(
            _load_tree(directory / left["file"], game),
            _load_tree(directory / right["file"], game),
        )
        for key in ("max_value_error", "max_log_probability_error"):
            identity[key] = max(identity[key], comparison[key])
    return {
        "repeat": repeat,
        "identity": identity,
        "shared_environment_transitions": shared["environment_transitions"],
        "recompute_environment_transitions": recompute["environment_transitions"],
        "shared_evaluated_positions": shared["evaluated_positions"],
        "recompute_evaluated_positions": recompute["evaluated_positions"],
        "generation_speedup": recompute["elapsed_generation_seconds"]
        / shared["elapsed_generation_seconds"],
    }


def verify_shared_paths(*, device: str, config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()
    report = {
        "schema_version": 1,
        "benchmark": "Shared-path replay",
        "device": device,
        "config": asdict(config),
        "passed": False,
        "measurements": [],
        "pairs": [],
        "errors": [],
        "metadata": {},
    }
    try:
        _check_device(device)
        with TemporaryDirectory(prefix="lfu-h1-") as temporary:
            directory = Path(temporary)
            manifest = _prepare(directory, device, config)
            report["metadata"] = manifest["metadata"]
            for repeat in range(config.repeats):
                modes = (
                    ("shared", "recompute")
                    if repeat % 2 == 0
                    else ("recompute", "shared")
                )
                rows = [_run_worker(directory, device, mode, repeat) for mode in modes]
                report["measurements"].extend(rows)
                for row in rows:
                    if not row["passed"]:
                        report["errors"].extend(
                            f"repeat {repeat} {row['mode']}: {error}"
                            for error in row.get("errors", ["worker failed"])
                        )
                try:
                    report["pairs"].append(_compare_pair(directory, rows, repeat))
                except (ValueError, OSError, KeyError) as exc:
                    report["errors"].append(str(exc))
            for row in report["measurements"]:
                row.pop("outputs", None)
    # The public report boundary preserves diagnostics even for worker/setup failures.
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not report["errors"] and len(report["pairs"]) == config.repeats
    if report["passed"]:
        report["summary"] = {
            "median_generation_speedup": statistics.median(
                pair["generation_speedup"] for pair in report["pairs"]
            ),
            "interpretation": "generation cost comparison only; no learning quality or playing strength claim",
        }
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--mode", choices=("shared", "recompute"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    from .shared_path_worker import run_worker

    row = run_worker(
        arguments.worker, arguments.device, arguments.mode, arguments.repeat
    )
    _write_json(arguments.output, row)
    raise SystemExit(0 if row["passed"] else 1)


if __name__ == "__main__":
    _main()
