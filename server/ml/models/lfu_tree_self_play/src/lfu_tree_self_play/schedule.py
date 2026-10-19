"""Pre-sampled placement positions, independent of policy and outcomes."""

import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Literal

from .game import Game, GameState
from .record_validation import validate_record
from .records import BranchPlan, CollectionRecord

PositionStrategy = Literal["uniform", "stratified", "fixed"]


def sample_branch_plan(
    empty_squares: int,
    rng: random.Random,
    *,
    levels: int = 2,
    strategy: PositionStrategy = "uniform",
) -> BranchPlan:
    """Sample one width-four plan for every path in a tree before collection.

    Positions count disk placements from the root, excluding forced passes.
    Use the branch_position RNG, never an action or continuation RNG.
    """
    if type(empty_squares) is not int or not 0 <= empty_squares <= 64:
        raise ValueError("empty_squares must be an integer between 0 and 64")
    if type(levels) is not int or levels not in (0, 1, 2):
        raise ValueError("levels must be an integer in {0, 1, 2}")
    if strategy not in ("uniform", "stratified", "fixed"):
        raise ValueError(f"unknown position strategy: {strategy}")

    count = min(levels, empty_squares)
    if not count:
        positions = ()
    elif strategy == "fixed":
        positions = tuple(d for d in (20, 40)[:levels] if d < empty_squares)
    elif strategy == "uniform":
        positions = tuple(sorted(rng.sample(range(empty_squares), count)))
    else:
        size, remainder = divmod(empty_squares, 3)
        intervals = []
        start = 0
        for index in range(3):
            stop = start + size + (index < remainder)
            if stop > start:
                intervals.append(range(start, stop))
            start = stop
        selected = rng.sample(intervals, min(count, len(intervals)))
        positions = tuple(sorted(rng.choice(interval) for interval in selected))
    return BranchPlan(width=4, positions=positions)


@dataclass(frozen=True)
class BranchPath:
    """Immutable execution history inherited separately by every child path."""

    plan: BranchPlan
    executed_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        plan = BranchPlan.model_validate(self.plan)
        if plan.positions != tuple(sorted(set(plan.positions))):
            raise ValueError("plan.positions must be sorted and unique")
        object.__setattr__(self, "plan", plan)
        executed = self.executed_positions
        if (
            type(executed) is not tuple
            or any(type(d) is not int or d not in plan.positions for d in executed)
            or executed != tuple(sorted(set(executed)))
        ):
            raise ValueError(
                "executed_positions must be sorted unique planned positions"
            )

    @property
    def planned_count(self) -> int:
        return len(self.plan.positions)

    @property
    def executed_count(self) -> int:
        return len(self.executed_positions)

    @property
    def unexecuted_positions(self) -> tuple[int, ...]:
        return tuple(d for d in self.plan.positions if d not in self.executed_positions)

    def visit(
        self, state: GameState, placement_depth: int
    ) -> tuple[bool, "BranchPath"]:
        """Consume a branch just before placement, after pass/terminal handling.

        The caller supplies the number of actual placements from the root.
        On True, execute the planned width and give the returned history to
        each child. Unreached positions remain in the plan without resampling.
        """
        if type(placement_depth) is not int or placement_depth < 0:
            raise ValueError("placement_depth must be a nonnegative integer")
        if (
            state.is_terminal
            or placement_depth not in self.plan.positions
            or placement_depth in self.executed_positions
        ):
            return False, self
        executed = tuple(sorted((*self.executed_positions, placement_depth)))
        return True, replace(self, executed_positions=executed)


@dataclass(frozen=True)
class BranchExecution:
    """Scheduled depths and actual group counts at those depths in a whole tree."""

    planned_positions: tuple[int, ...]
    executions_by_position: tuple[int, ...]
    width: int

    @property
    def planned_by_position(self) -> tuple[int, ...]:
        """Group budget at each depth if every earlier branch reaches it."""
        return tuple(self.width**index for index in range(len(self.planned_positions)))

    @property
    def planned_count(self) -> int:
        """Total group budget if all paths reach every planned position."""
        return sum(self.planned_by_position)

    @property
    def executed_count(self) -> int:
        """Total sibling groups; multiple paths can branch at the same depth."""
        return sum(self.executions_by_position)


def summarize_branch_execution(record: CollectionRecord, game: Game) -> BranchExecution:
    """Validate a completed unit and derive counts from its persisted groups."""
    validate_record(record, game)
    nodes = {node.node_id: node for node in record.nodes}
    counts = Counter(
        nodes[group.parent_node_id].placement_depth for group in record.sibling_groups
    )
    positions = record.branch_plan.positions
    return BranchExecution(
        positions, tuple(counts[d] for d in positions), record.branch_plan.width
    )
