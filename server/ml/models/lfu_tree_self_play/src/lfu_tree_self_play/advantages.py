"""Fixed-scale policy advantages for the same collected edges in B and C."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .game import Game
from .records import CollectionRecord
from .returns import aggregate_returns


@dataclass(frozen=True)
class AdvantageTargets:
    """Detached, immutable edge targets and the fraction of zero LOO groups.

    The fraction is None when no sibling groups were reached. Both maps include
    every collected edge once; path weights are applied separately by the loss.
    """

    standard_edge_advantages: Mapping[str, float]
    leave_one_out_edge_advantages: Mapping[str, float]
    zero_advantage_group_fraction: float | None

    def __post_init__(self) -> None:
        for name in ("standard_edge_advantages", "leave_one_out_edge_advantages"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def leave_one_out_advantages(returns: tuple[float, ...]) -> tuple[float, ...]:
    """Subtract the other samples' mean, with returns in one parent's view."""
    if len(returns) < 2:
        raise ValueError("leave-one-out requires at least two siblings")
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("sibling returns must be finite")
    # Identical conditional averages must yield exact zeros even after rounding.
    if all(value == returns[0] for value in returns):
        return (0.0,) * len(returns)
    total = math.fsum(returns)
    return tuple(value - (total - value) / (len(returns) - 1) for value in returns)


def compute_advantages(record: CollectionRecord, game: Game) -> AdvantageTargets:
    """Validate a complete record and construct B/C targets without mutation.

    B uses R - V_old(parent) on all edges. C replaces only sibling groups with
    leave-one-out advantages, retaining duplicate-action samples. R is the
    recursively averaged terminal return in the actual parent's turn, including
    forced passes. No batch centering, standard-deviation scaling, or clipping
    is applied. Invalid records raise RecordValidationError via aggregate_returns.
    """
    targets = aggregate_returns(record, game)
    nodes = {node.node_id: node for node in record.nodes}
    standard = {}
    for edge in record.edges:
        old_value = nodes[edge.parent_node_id].old_value
        assert old_value is not None  # Guaranteed for edge parents by validation.
        standard[edge.edge_id] = targets.edge_parent_targets[edge.edge_id] - old_value

    relative = standard.copy()
    zero_groups = 0
    for group in record.sibling_groups:
        values = tuple(targets.edge_parent_targets[eid] for eid in group.edge_ids)
        advantages = leave_one_out_advantages(values)
        relative.update(zip(group.edge_ids, advantages, strict=True))
        zero_groups += not any(advantages)

    return AdvantageTargets(
        standard_edge_advantages=standard,
        leave_one_out_edge_advantages=relative,
        zero_advantage_group_fraction=(
            zero_groups / len(record.sibling_groups) if record.sibling_groups else None
        ),
    )
