"""Strict typed records for collected self-play data."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    InstanceOf,
    StrictInt,
    StrictStr,
)

from .game import GameState


def _numeric(value: object) -> object:
    if type(value) not in (int, float):
        raise ValueError("expected a Python int or float, excluding bool")
    return value


RecordId = Annotated[StrictStr, Field(min_length=1, pattern=r"\S")]
Count = Annotated[StrictInt, Field(ge=0)]
Real = Annotated[float, BeforeValidator(_numeric)]
Seconds = Annotated[Real, Field(ge=0)]
Value = Annotated[Real, Field(ge=-1, le=1)]
Result = Annotated[StrictInt, Field(ge=-1, le=1)]


class RecordModel(BaseModel):
    """Base configuration shared by every persisted record."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        revalidate_instances="always",
    )


class BranchPlan(RecordModel):
    width: Annotated[StrictInt, Field(ge=2)]
    positions: tuple[Count, ...]


class RNGStreamRecord(RecordModel):
    stream_id: RecordId
    purpose: Literal["branch_position", "action_sampling", "sibling_rollout"]
    seed: Count


class RNGRecord(RecordModel):
    algorithm: Literal["python.random.MT19937-v1"]
    run_seed: Count
    streams: tuple[RNGStreamRecord, ...]


class SamplingRecord(RecordModel):
    stream_id: RecordId
    draw_index: Count
    continuation_seed: Count | None


class NodeRecord(RecordModel):
    node_id: RecordId
    incoming_edge_id: RecordId | None
    state: InstanceOf[GameState]
    placement_depth: Count
    model_generation: Count
    old_value: Value | None
    black_result: Result | None


class EdgeRecord(RecordModel):
    edge_id: RecordId
    parent_node_id: RecordId
    child_node_id: RecordId
    action: Annotated[StrictInt, Field(ge=0, lt=64)]
    old_log_probability: Annotated[Real, Field(le=0)]
    model_generation: Count
    sampling: SamplingRecord
    sibling_group_id: RecordId | None
    sample_index: Count | None


class SiblingGroupRecord(RecordModel):
    group_id: RecordId
    parent_node_id: RecordId
    edge_ids: tuple[RecordId, ...]
    distinct_action_count: Count


class CostRecord(RecordModel):
    environment_transitions: Count
    evaluated_positions: Count
    inference_batches: Count
    collection_seconds: Seconds
    inference_seconds: Seconds
    transition_seconds: Seconds
    management_seconds: Seconds
    peak_memory_bytes: Count


class CollectionRecord(RecordModel):
    schema_version: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    unit_id: RecordId
    experiment_id: RecordId
    round_id: RecordId
    model_generation: Count
    mode: Literal["independent", "tree"]
    root_node_id: RecordId
    branch_plan: BranchPlan
    rng: RNGRecord
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    sibling_groups: tuple[SiblingGroupRecord, ...]
    cost: CostRecord
