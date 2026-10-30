"""Sampled-action PPO and node-value MSE with conserved path weights."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from .model import PolicyValueOutput, masked_policy_logits
from .records import CollectionRecord
from .returns import ReturnTargets

PLACEMENT_NORMALIZER = 60


@dataclass(frozen=True)
class LossBatch:
    """One unit: unique nonterminal nodes, and every saved edge sample in order.

    Observations and value vectors follow node_ids; edge vectors follow edge_ids.
    Weights may already include an independent trajectory's 1/K root weight.
    All tensors are on CPU when built; ppo_value_loss moves targets to the output
    device and detaches them. Only the current model output receives gradients.
    """

    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    observations: Tensor
    edge_parent_indices: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    edge_weights: Tensor
    value_targets: Tensor
    node_weights: Tensor


class PPOValueLoss(NamedTuple):
    """Separately scaled objectives and per-sample ratios for diagnostics."""

    policy: Tensor
    value: Tensor
    ratios: Tensor


def build_loss_batch(
    record: CollectionRecord,
    targets: ReturnTargets,
    advantages: Mapping[str, float],
) -> LossBatch:
    """Pack a validated record and its aggregated (possibly scaled) targets.

    The caller supplies matching outputs of aggregate_returns or the collectors'
    weighted_targets, and one mover-perspective advantage per saved edge. This
    boundary does not calculate advantages, replay games, or change path weights.
    Missing/extra IDs are rejected rather than silently dropping samples.
    """
    nodes = tuple(node for node in record.nodes if not node.state.is_terminal)
    node_ids = tuple(node.node_id for node in nodes)
    edge_ids = tuple(edge.edge_id for edge in record.edges)
    if len(set(node_ids)) != len(node_ids) or len(set(edge_ids)) != len(edge_ids):
        raise ValueError("node and edge IDs must be unique")
    for name, mapping, expected in (
        ("advantages", advantages, set(edge_ids)),
        ("edge weights", targets.edge_weights, set(edge_ids)),
        ("node weights", targets.node_weights, {n.node_id for n in record.nodes}),
        (
            "node targets",
            targets.node_player_targets,
            {n.node_id for n in record.nodes},
        ),
    ):
        if set(mapping) != expected:
            raise ValueError(f"{name} IDs must match the record")
    indices = {node_id: index for index, node_id in enumerate(node_ids)}
    if any(edge.parent_node_id not in indices for edge in record.edges):
        raise ValueError("each edge must have a nonterminal parent")
    observations = (
        np.stack([node.state.observation for node in nodes])
        if nodes
        else np.empty((0, 4, 8, 8), dtype=np.float32)
    )
    return LossBatch(
        node_ids=node_ids,
        edge_ids=edge_ids,
        observations=torch.from_numpy(observations),
        edge_parent_indices=torch.tensor(
            [indices[edge.parent_node_id] for edge in record.edges], dtype=torch.long
        ),
        actions=torch.tensor([edge.action for edge in record.edges], dtype=torch.long),
        old_log_probabilities=torch.tensor(
            [edge.old_log_probability for edge in record.edges], dtype=torch.float64
        ),
        advantages=torch.tensor(
            [advantages[key] for key in edge_ids], dtype=torch.float64
        ),
        edge_weights=torch.tensor(
            [targets.edge_weights[key] for key in edge_ids], dtype=torch.float64
        ),
        value_targets=torch.tensor(
            [targets.node_player_targets[key] for key in node_ids], dtype=torch.float64
        ),
        node_weights=torch.tensor(
            [targets.node_weights[key] for key in node_ids], dtype=torch.float64
        ),
    )


def ppo_value_loss(
    output: PolicyValueOutput,
    batch: LossBatch,
    *,
    num_roots: int,
    clip_epsilon: float = 0.2,
) -> PPOValueLoss:
    """Compute a unit's contribution divided by 60 * num_roots.

    Sum contributions using the same total root count across units/minibatches;
    K independent paths of one root count as one root with pre-scaled 1/K weights.
    No averaging by edge count, leaf count, or realized trajectory length occurs.
    The output rows must follow batch.node_ids. Evaluate the current model with
    the same inference mode as collection to reproduce ratios at theta = old.
    Log-softmax uses temperature one, matching sample_policy. Computation uses
    float64, or float32 on MPS where float64 is unavailable.
    A terminal-only unit contributes differentiable zeros for empty outputs.
    Nonfinite legal logits, predicted values, targets, or weights raise ValueError,
    as do invalid shapes or actions. Illegal-action logits are masked and ignored.
    """
    if type(num_roots) is not int or num_roots < 1:
        raise ValueError("num_roots must be a positive integer")
    if (
        type(clip_epsilon) not in (int, float)
        or not math.isfinite(clip_epsilon)
        or not 0 < clip_epsilon < 1
    ):
        raise ValueError("clip_epsilon must be finite and in (0, 1)")
    logits, values = output
    n, e = len(batch.node_ids), len(batch.edge_ids)
    if (
        logits.shape != (n, 64)
        or values.shape != (n,)
        or not logits.is_floating_point()
        or not values.is_floating_point()
        or logits.device != values.device
        or batch.observations.shape != (n, 4, 8, 8)
    ):
        raise ValueError(
            "expected matching floating-point node outputs and observations"
        )
    device = logits.device
    loss_dtype = torch.float32 if device.type == "mps" else torch.float64
    parents = batch.edge_parent_indices.detach().to(device)
    actions = batch.actions.detach().to(device)
    if (
        parents.shape != (e,)
        or actions.shape != (e,)
        or parents.dtype != torch.long
        or actions.dtype != torch.long
        or not ((parents >= 0) & (parents < n)).all()
        or not ((actions >= 0) & (actions < 64)).all()
    ):
        raise ValueError("edge parents and actions must be valid int64 vectors")
    old_logs, advantages, edge_weights, value_targets, node_weights = (
        tensor.detach().to(device=device, dtype=loss_dtype)
        for tensor in (
            batch.old_log_probabilities,
            batch.advantages,
            batch.edge_weights,
            batch.value_targets,
            batch.node_weights,
        )
    )
    for tensor, size in (
        (old_logs, e),
        (advantages, e),
        (edge_weights, e),
        (value_targets, n),
        (node_weights, n),
    ):
        if tensor.shape != (size,) or not torch.isfinite(tensor).all():
            raise ValueError("targets and weights must be finite, matching vectors")
    if (
        (old_logs > 0).any()
        or (edge_weights <= 0).any()
        or (node_weights <= 0).any()
        or (value_targets.abs() > 1).any()
    ):
        raise ValueError(
            "invalid old log probabilities, path weights, or value targets"
        )
    mask = batch.observations[:, 3].flatten(1).to(device=device).bool()
    if not mask[parents, actions].all():
        raise ValueError("saved actions must be legal at their parent nodes")
    if not torch.isfinite(logits[mask]).all() or not torch.isfinite(values).all():
        raise ValueError("legal logits and predicted values must be finite")
    logs = masked_policy_logits(logits.to(dtype=loss_dtype), mask).log_softmax(-1)
    ratios = (logs[parents, actions] - old_logs).exp()
    if not torch.isfinite(ratios).all():
        raise ValueError("policy ratios must be finite")
    surrogate = torch.minimum(
        ratios * advantages,
        ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages,
    )
    denominator = PLACEMENT_NORMALIZER * num_roots
    return PPOValueLoss(
        policy=-(edge_weights * surrogate).sum() / denominator,
        value=(
            node_weights * (values.to(dtype=loss_dtype) - value_targets).square()
        ).sum()
        / denominator,
        ratios=ratios,
    )
