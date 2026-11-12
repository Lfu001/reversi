"""One pinned collection generation, one gradient pass, one atomic update."""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .advantages import compute_advantages
from .checkpoints import CheckpointManifest, CheckpointStore
from .collector import weight_independent_records
from .dataset_codec import canonical_json
from .dataset_store import _sync_directory
from .losses import build_loss_batch, ppo_value_loss
from .model import masked_policy_logits


@dataclass(frozen=True)
class TrainingOptions:
    advantage: str = "standard"
    clip_epsilon: float = 0.2
    value_coefficient: float = 1.0

    def __post_init__(self):
        if self.advantage not in ("standard", "leave_one_out"):
            raise ValueError("unknown advantage mode")
        for name in ("clip_epsilon", "value_coefficient"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.clip_epsilon < 1 or self.value_coefficient < 0:
            raise ValueError("invalid loss coefficients")


@dataclass(frozen=True)
class TrainingResult:
    manifest: CheckpointManifest
    report: dict


def _batches(store, inputs, root_groups, options):
    selected = tuple(identity.unit_id for identity in inputs.selected)
    flattened = tuple(uid for group in root_groups for uid in group)
    if (
        not root_groups
        or any(not group for group in root_groups)
        or len(set(flattened)) != len(flattened)
        or set(flattened) != set(selected)
    ):
        raise ValueError(
            "root groups must partition the selected manifest exactly once"
        )
    batches = []
    for group in root_groups:
        units = tuple(
            store.dataset_store.load(uid, model_generation=inputs.model_generation)
            for uid in group
        )
        records = tuple(unit.record for unit in units)
        if len(units) > 1:
            targets = weight_independent_records(records, store.dataset_store.game)
        else:
            targets = (units[0].targets,)
        for record, target in zip(records, targets, strict=True):
            advantages = compute_advantages(record, store.dataset_store.game)
            values = (
                advantages.standard_edge_advantages
                if options.advantage == "standard"
                else advantages.leave_one_out_edge_advantages
            )
            batches.append(build_loss_batch(record, target, values))
    return batches


def _policy_logs(output, batch):
    mask = batch.observations[:, 3].flatten(1).bool()
    return masked_policy_logits(output.policy_logits, mask).log_softmax(-1), mask


def _diagnostics(model, loader, accelerator, old_logs, clip_epsilon):
    totals = {
        "kl": 0.0,
        "clip_fraction": 0.0,
        "policy_entropy": 0.0,
        "value_error": 0.0,
    }
    node_mass = edge_mass = 0.0
    with torch.no_grad():
        for batch, old in zip(loader, old_logs, strict=True):
            if not batch.node_ids:
                continue
            with accelerator.autocast():
                output = model(batch.observations)
            logs, mask = _policy_logs(output, batch)
            # Zero illegal log entries before multiplication: 0 * -inf is NaN.
            safe_logs = logs.masked_fill(~mask, 0)
            safe_old = old.masked_fill(~mask, 0)
            kl = (old.exp() * (safe_old - safe_logs)).sum(-1)
            entropy = -(logs.exp() * safe_logs).sum(-1)
            ratios = (
                logs[batch.edge_parent_indices, batch.actions]
                - batch.old_log_probabilities
            ).exp()
            clipped = (ratios - 1).abs() > clip_epsilon
            totals["kl"] += float((kl * batch.node_weights).sum())
            totals["policy_entropy"] += float((entropy * batch.node_weights).sum())
            totals["value_error"] += float(
                (
                    (output.value - batch.value_targets).square() * batch.node_weights
                ).sum()
            )
            totals["clip_fraction"] += float((clipped * batch.edge_weights).sum())
            node_mass += float(batch.node_weights.sum())
            edge_mass += float(batch.edge_weights.sum())
    return {
        key: value / (edge_mass if key == "clip_fraction" else node_mass)
        if (edge_mass if key == "clip_fraction" else node_mass)
        else 0.0
        for key, value in totals.items()
    }


def train_one_epoch(
    store: CheckpointStore,
    *,
    optimizer_factory,
    scheduler_factory=lambda optimizer: None,
    options: TrainingOptions | None = None,
    root_groups: tuple[tuple[str, ...], ...] | None = None,
    accelerator: Accelerator | None = None,
) -> TrainingResult:
    """Restore the latest checkpoint and consume its pinned inputs once.

    Each tree (or independent K=1 trajectory) is one root by default. Supply
    explicit root_groups for independent K>1 paths; no grouping is guessed from
    equal board states. Groups partition the pinned range and are persisted.
    All units contribute to one optimizer step, normalized by 60 * root count.
    BatchNorm stays in eval mode, matching collection; gradients remain enabled.

    A pending checkpoint binds settings before any update. Failures must be
    retried through this entry point, which restores committed state. A complete
    checkpoint returns its stored report without another step. Factories must
    match the saved optimizer/scheduler; schedulers must accept step() without
    a metric. Single-process FP32/bfloat16 execution is supported. Distributed
    execution and FP16 require per-rank/scaler checkpoint state and are rejected.
    Exact replay requires the same deterministic backend and execution settings.
    """
    options = options or TrainingOptions()
    accelerator = accelerator or Accelerator(mixed_precision="no")
    if (
        accelerator.num_processes != 1
        or accelerator.mixed_precision not in ("no", "bf16")
        or accelerator.gradient_accumulation_steps != 1
    ):
        raise ValueError(
            "trainer requires one process, FP32/bfloat16, and accumulation=1"
        )
    recovered = store.recover()
    if recovered is None:
        raise ValueError("training requires a committed collection checkpoint")
    manifest = recovered.manifest
    inputs = manifest.inputs
    selected = tuple(identity.unit_id for identity in inputs.selected)
    report_path = recovered.path / "training.json"
    previous_report = (
        json.loads(report_path.read_bytes()) if report_path.exists() else None
    )
    if previous_report is not None and previous_report.get("version") != 1:
        raise ValueError("unsupported training report version")
    if root_groups is None:
        root_groups = (
            tuple(tuple(group) for group in previous_report["plan"]["root_groups"])
            if previous_report
            else tuple((uid,) for uid in selected)
        )
    plan = json.loads(
        canonical_json(
            {
                "options": asdict(options),
                "root_groups": root_groups,
                "device": str(accelerator.device),
                "mixed_precision": accelerator.mixed_precision,
                "epochs": 1,
                "optimizer_steps": 1,
            }
        )
    )
    if previous_report:
        if previous_report["plan"] != plan:
            raise ValueError("training plan mismatch on resume")
        if previous_report["status"] == "complete":
            if (
                manifest.progress.model_generation != inputs.model_generation + 1
                or not set(selected).issubset(manifest.progress.consumed_unit_ids)
            ):
                raise ValueError("training report consumption mismatch")
            # A prior attempt may have stopped after rename but before fsync.
            _sync_directory(store.root / "committed")
            _sync_directory(store.root / "staging")
            return TrainingResult(manifest, previous_report)
        if previous_report["status"] != "pending":
            raise ValueError("invalid training report status")
    if (
        not selected
        or inputs.model_generation != manifest.progress.model_generation
        or set(selected).intersection(manifest.progress.consumed_unit_ids)
    ):
        raise ValueError("empty, stale, or already consumed input manifest")
    started = time.perf_counter()
    batches = _batches(store, inputs, root_groups, options)
    if not any(batch.node_ids for batch in batches):
        raise ValueError("manifest has no trainable positions")
    runtime = store.restore(
        recovered,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    model, optimizer, scheduler = runtime.model, runtime.optimizer, runtime.scheduler
    if not previous_report:
        pending_id = (
            "train-"
            + hashlib.sha256(
                canonical_json(
                    {
                        "parent": manifest.commit_id,
                        "plan": plan,
                        "inputs": asdict(inputs),
                    }
                )
            ).hexdigest()[:32]
        )
        report = {"version": 1, "status": "pending", "plan": plan}
        manifest = store.commit(
            pending_id,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=runtime.rng,
            inputs=inputs,
            expected_parent=manifest.commit_id,
            progress=replace(manifest.progress, stage="learning"),
            training_report=report,
        )
    loader = DataLoader(batches, batch_size=None, shuffle=False)
    model.eval()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    old_logs = []
    with torch.no_grad():
        for batch in loader:
            if not batch.node_ids:
                old_logs.append(None)
                continue
            with accelerator.autocast():
                output = model(batch.observations)
            logs, mask = _policy_logs(output, batch)
            old_logs.append(logs.detach())
            saved_logs = logs[batch.edge_parent_indices, batch.actions]
            # BF16 collection/update comparisons have lower numerical precision.
            tolerance = 0.03 if accelerator.mixed_precision == "bf16" else 1e-5
            if (
                not torch.allclose(
                    saved_logs,
                    batch.old_log_probabilities,
                    atol=tolerance,
                    rtol=tolerance,
                )
                or not torch.isfinite(logs[mask]).all()
            ):
                raise ValueError(
                    "checkpoint policy does not match the collection policy"
                )
    before = _diagnostics(model, loader, accelerator, old_logs, options.clip_epsilon)
    optimizer.zero_grad(set_to_none=True)
    policy_loss = value_loss = 0.0
    for batch in loader:
        if not batch.node_ids:
            continue
        with accelerator.autocast():
            losses = ppo_value_loss(
                model(batch.observations),
                batch,
                num_roots=len(root_groups),
                clip_epsilon=options.clip_epsilon,
            )
        accelerator.backward(losses.policy + options.value_coefficient * losses.value)
        policy_loss += float(losses.policy.detach())
        value_loss += float(losses.value.detach())
    gradient_norm = float(accelerator.clip_grad_norm_(model.parameters(), float("inf")))
    if not math.isfinite(gradient_norm):
        raise ValueError("nonfinite gradient norm")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    after = _diagnostics(model, loader, accelerator, old_logs, options.clip_epsilon)
    report = {
        "version": 1,
        "status": "complete",
        "plan": plan,
        "metrics": {
            "before": before,
            "after": after,
            "gradient_norm": gradient_norm,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "num_roots": len(root_groups),
            "num_units": len(selected),
            "num_nodes": sum(len(batch.node_ids) for batch in batches),
            "num_edges": sum(len(batch.edge_ids) for batch in batches),
            "learning_seconds": time.perf_counter() - started,
        },
    }
    # Reject nonfinite diagnostics before publishing any model or consumption.
    canonical_json(report)
    progress = replace(
        manifest.progress,
        model_generation=inputs.model_generation + 1,
        cumulative_cost=manifest.progress.cumulative_cost
        + report["metrics"]["learning_seconds"],
        consumed_unit_ids=manifest.progress.consumed_unit_ids + selected,
    )
    updated = store.commit(
        manifest.commit_id + "-updated",
        model=accelerator.unwrap_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        rng=runtime.rng,
        inputs=inputs,
        progress=progress,
        expected_parent=manifest.commit_id,
        training_report=report,
    )
    return TrainingResult(updated, report)
