"""Standard and sibling leave-one-out policy advantages for collected edges."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .game import Game
from .records import CollectionRecord
from .returns import aggregate_returns


@dataclass(frozen=True)
class AdvantageTargets:
    """Immutable, unweighted advantages keyed by collected edge ID.

    Both copied, read-only maps include each collected edge once. Callers apply
    path weights when constructing a loss. The fraction counts sibling groups
    whose leave-one-out advantages are all zero; it is None when no sibling
    groups were reached.
    """

    standard_edge_advantages: Mapping[str, float]
    leave_one_out_edge_advantages: Mapping[str, float]
    zero_advantage_group_fraction: float | None

    def __post_init__(self) -> None:
        for name in ("standard_edge_advantages", "leave_one_out_edge_advantages"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def leave_one_out_advantages(returns: tuple[float, ...]) -> tuple[float, ...]:
    """Subtract the mean of the other sibling returns from each return.

    Provide at least two finite returns from the same parent state, all in the
    perspective of the player to move at that state. Raises ValueError for fewer
    than two returns or any non-finite value.
    """
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
    """Validate a complete record and compute both advantage variants.

    Standard advantages subtract each parent's saved value estimate from its
    edge's terminal return. The estimate comes from the fixed model used during
    collection. Returns recursively average descendant outcomes and use the
    actual parent player's perspective, including after forced passes.

    For each sibling group, leave-one-out advantages subtract the mean return
    of the other samples from each sample's return. All samples in a group
    share a parent; duplicate actions remain separate samples. Edges outside
    sibling groups retain their standard advantages.

    Neither variant centers values across groups, divides by a standard
    deviation, or clips values. The input is unchanged. Invalid records raise
    RecordValidationError through aggregate_returns.
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
