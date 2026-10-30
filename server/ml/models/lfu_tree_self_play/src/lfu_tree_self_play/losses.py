"""Sampled-action PPO and node-value MSE with conserved path weights."""

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from .model import PolicyValueOutput, masked_policy_logits
from .records import CollectionRecord
from .returns import ReturnTargets

PLACEMENT_NORMALIZER = 60


class LossBatch(NamedTuple):
    """One unit: unique nonterminal nodes, and every saved edge sample in order.

    Observations and value vectors follow node_ids; edge vectors follow edge_ids.
    Weights may already include an independent trajectory's 1/K root weight.
    Each item is a complete, variable-size unit. Pass batch_size=None to
    DataLoader before Accelerate preparation to disable automatic batching.
    The training runtime owns device placement and execution precision.
    ppo_value_loss detaches targets; only current model outputs receive gradients.
    """

    node_ids: Sequence[str]
    edge_ids: Sequence[str]
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


class _LossInputs(NamedTuple):
    legal_mask: Tensor
    parents: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    edge_weights: Tensor
    value_targets: Tensor
    node_weights: Tensor


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
            [edge.old_log_probability for edge in record.edges], dtype=torch.float32
        ),
        advantages=torch.tensor(
            [advantages[key] for key in edge_ids], dtype=torch.float32
        ),
        edge_weights=torch.tensor(
            [targets.edge_weights[key] for key in edge_ids], dtype=torch.float32
        ),
        value_targets=torch.tensor(
            [targets.node_player_targets[key] for key in node_ids], dtype=torch.float32
        ),
        node_weights=torch.tensor(
            [targets.node_weights[key] for key in node_ids], dtype=torch.float32
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
    Inputs must already be placed together by the training runtime. The loss
    uses the supplied tensor dtypes and temperature one, matching sample_policy.
    A terminal-only unit contributes differentiable zeros for empty outputs.
    Nonfinite legal logits, predicted values, targets, or weights raise ValueError,
    as do invalid shapes or actions. Illegal-action logits are masked and ignored.
    """
    _validate_loss_options(num_roots, clip_epsilon)
    inputs = _prepare_loss_inputs(output, batch)
    ratios = _sampled_policy_ratios(output.policy_logits, inputs)
    return _weighted_objectives(
        ratios,
        output.value,
        inputs,
        num_roots=num_roots,
        clip_epsilon=clip_epsilon,
    )


def _validate_loss_options(num_roots: int, clip_epsilon: float) -> None:
    if type(num_roots) is not int or num_roots < 1:
        raise ValueError("num_roots must be a positive integer")
    if (
        type(clip_epsilon) not in (int, float)
        or not math.isfinite(clip_epsilon)
        or not 0 < clip_epsilon < 1
    ):
        raise ValueError("clip_epsilon must be finite and in (0, 1)")


def _prepare_loss_inputs(output: PolicyValueOutput, batch: LossBatch) -> _LossInputs:
    logits, values = output
    node_count, edge_count = len(batch.node_ids), len(batch.edge_ids)
    if (
        logits.shape != (node_count, 64)
        or values.shape != (node_count,)
        or not logits.is_floating_point()
        or not values.is_floating_point()
        or batch.observations.shape != (node_count, 4, 8, 8)
    ):
        raise ValueError(
            "expected matching floating-point node outputs and observations"
        )
    inputs = _LossInputs(
        legal_mask=batch.observations[:, 3].flatten(1).bool(),
        parents=batch.edge_parent_indices.detach(),
        actions=batch.actions.detach(),
        old_log_probabilities=batch.old_log_probabilities.detach(),
        advantages=batch.advantages.detach(),
        edge_weights=batch.edge_weights.detach(),
        value_targets=batch.value_targets.detach(),
        node_weights=batch.node_weights.detach(),
    )
    _validate_edge_samples(inputs, node_count, edge_count)
    _validate_frozen_targets(inputs, node_count, edge_count)
    if (
        not torch.isfinite(logits[inputs.legal_mask]).all()
        or not torch.isfinite(values).all()
    ):
        raise ValueError("legal logits and predicted values must be finite")
    return inputs


def _validate_edge_samples(
    inputs: _LossInputs, node_count: int, edge_count: int
) -> None:
    parents, actions = inputs.parents, inputs.actions
    if (
        parents.shape != (edge_count,)
        or actions.shape != (edge_count,)
        or parents.dtype != torch.long
        or actions.dtype != torch.long
        or not ((parents >= 0) & (parents < node_count)).all()
        or not ((actions >= 0) & (actions < 64)).all()
    ):
        raise ValueError("edge parents and actions must be valid int64 vectors")
    if not inputs.legal_mask[parents, actions].all():
        raise ValueError("saved actions must be legal at their parent nodes")


def _validate_frozen_targets(
    inputs: _LossInputs, node_count: int, edge_count: int
) -> None:
    for tensor, size in (
        (inputs.old_log_probabilities, edge_count),
        (inputs.advantages, edge_count),
        (inputs.edge_weights, edge_count),
        (inputs.value_targets, node_count),
        (inputs.node_weights, node_count),
    ):
        if tensor.shape != (size,) or not torch.isfinite(tensor).all():
            raise ValueError("targets and weights must be finite, matching vectors")
    if (
        (inputs.old_log_probabilities > 0).any()
        or (inputs.edge_weights <= 0).any()
        or (inputs.node_weights <= 0).any()
        or (inputs.value_targets.abs() > 1).any()
    ):
        raise ValueError(
            "invalid old log probabilities, path weights, or value targets"
        )


def _sampled_policy_ratios(logits: Tensor, inputs: _LossInputs) -> Tensor:
    logs = masked_policy_logits(logits, inputs.legal_mask).log_softmax(-1)
    ratios = (logs[inputs.parents, inputs.actions] - inputs.old_log_probabilities).exp()
    if not torch.isfinite(ratios).all():
        raise ValueError("policy ratios must be finite")
    return ratios


def _weighted_objectives(
    ratios: Tensor,
    values: Tensor,
    inputs: _LossInputs,
    *,
    num_roots: int,
    clip_epsilon: float,
) -> PPOValueLoss:
    surrogate = torch.minimum(
        ratios * inputs.advantages,
        ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * inputs.advantages,
    )
    denominator = PLACEMENT_NORMALIZER * num_roots
    value_errors = (values - inputs.value_targets).square()
    return PPOValueLoss(
        policy=-(inputs.edge_weights * surrogate).sum() / denominator,
        value=(inputs.node_weights * value_errors).sum() / denominator,
        ratios=ratios,
    )
