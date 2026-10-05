import numpy as np
import pytest
from pydantic import ValidationError

from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.records import (
    BranchPlan,
    CollectionRecord,
    CostRecord,
    EdgeRecord,
    NodeRecord,
    RNGRecord,
    RNGStreamRecord,
    SamplingRecord,
    SiblingGroupRecord,
)


def costs(**updates):
    values = {
        "environment_transitions": 0,
        "evaluated_positions": 0,
        "inference_batches": 0,
        "collection_seconds": 0.0,
        "inference_seconds": 0.0,
        "transition_seconds": 0.0,
        "management_seconds": 0.0,
        "peak_memory_bytes": 0,
    }
    values.update(updates)
    return values


def collection(**updates):
    values = {
        "unit_id": "u0",
        "experiment_id": "x0",
        "round_id": "r0",
        "model_generation": 0,
        "mode": "tree",
        "root_node_id": "n0",
        "branch_plan": BranchPlan(width=2, positions=()),
        "rng": RNGRecord(algorithm="python.random.MT19937-v1", run_seed=0, streams=()),
        "nodes": (),
        "edges": (),
        "sibling_groups": (),
        "cost": CostRecord(**costs()),
    }
    values.update(updates)
    return values


@pytest.mark.parametrize("bad", [True, "1", 1.5, -1])
def test_transition_count_requires_nonnegative_integer(bad):
    with pytest.raises(ValidationError):
        CostRecord(**costs(environment_transitions=bad))


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf"), -0.1])
def test_collection_seconds_rejects_invalid_value(bad):
    with pytest.raises(ValidationError):
        CostRecord(**costs(collection_seconds=bad))


def test_branch_positions_require_tuple_and_integer_entries():
    with pytest.raises(ValidationError):
        BranchPlan(width=4, positions=[0])
    with pytest.raises(ValidationError):
        BranchPlan(width=4, positions=(True,))


def test_frozen_record_rejects_assignment_and_unknown_field():
    plan = BranchPlan(width=4, positions=())
    with pytest.raises(ValidationError):
        plan.width = 2
    with pytest.raises(ValidationError):
        BranchPlan(width=4, positions=(), position=(0,))


def test_state_keeps_existing_game_contract():
    state = ReversiPyGame().initial_state()
    node = NodeRecord(
        node_id="n0",
        incoming_edge_id=None,
        state=state,
        placement_depth=0,
        model_generation=0,
        old_value=0.0,
        black_result=None,
    )
    assert node.state is state
    original_observation = state.observation
    observation = node.state.observation
    observation[0] = 0
    np.testing.assert_array_equal(node.state.observation, original_observation)


@pytest.mark.parametrize("bad", [True, 0, -1, 2])
def test_schema_version_is_strictly_one(bad):
    with pytest.raises(ValidationError):
        CollectionRecord(**collection(schema_version=bad))


@pytest.mark.parametrize("bad", [True, -2, 2])
def test_black_result_is_strict_and_bounded(bad):
    with pytest.raises(ValidationError):
        NodeRecord(
            node_id="n0",
            incoming_edge_id=None,
            state=ReversiPyGame().initial_state(),
            placement_depth=0,
            model_generation=0,
            old_value=None,
            black_result=bad,
        )


@pytest.mark.parametrize("field", ["model_generation", "seed", "sample_index"])
def test_generations_seeds_and_sample_indices_reject_negative_values(field):
    if field == "model_generation":
        with pytest.raises(ValidationError):
            NodeRecord(
                node_id="n0",
                incoming_edge_id=None,
                state=ReversiPyGame().initial_state(),
                placement_depth=0,
                model_generation=-1,
                old_value=None,
                black_result=None,
            )
    elif field == "seed":
        with pytest.raises(ValidationError):
            RNGStreamRecord(stream_id="s0", purpose="action_sampling", seed=-1)
    else:
        with pytest.raises(ValidationError):
            EdgeRecord(
                edge_id="e0",
                parent_node_id="n0",
                child_node_id="n1",
                action=0,
                old_log_probability=0.0,
                model_generation=0,
                sampling=SamplingRecord(
                    stream_id="s0", draw_index=0, continuation_seed=None
                ),
                sibling_group_id=None,
                sample_index=-1,
            )


@pytest.mark.parametrize("bad", [-1.01, 1.01])
def test_old_value_must_be_in_unit_interval(bad):
    with pytest.raises(ValidationError):
        NodeRecord(
            node_id="n0",
            incoming_edge_id=None,
            state=ReversiPyGame().initial_state(),
            placement_depth=0,
            model_generation=0,
            old_value=bad,
            black_result=None,
        )


def test_log_probability_cannot_be_positive():
    with pytest.raises(ValidationError):
        EdgeRecord(
            edge_id="e0",
            parent_node_id="n0",
            child_node_id="n1",
            action=0,
            old_log_probability=0.01,
            model_generation=0,
            sampling=SamplingRecord(
                stream_id="s0", draw_index=0, continuation_seed=None
            ),
            sibling_group_id=None,
            sample_index=None,
        )


@pytest.mark.parametrize(
    "record_type,values,id_field",
    [
        (
            RNGStreamRecord,
            {"stream_id": "   ", "purpose": "action_sampling", "seed": 0},
            "stream_id",
        ),
        (
            SamplingRecord,
            {"stream_id": "   ", "draw_index": 0, "continuation_seed": None},
            "stream_id",
        ),
        (
            SiblingGroupRecord,
            {
                "group_id": "   ",
                "parent_node_id": "n0",
                "edge_ids": (),
                "distinct_action_count": 0,
            },
            "group_id",
        ),
        (
            NodeRecord,
            {
                "node_id": "   ",
                "incoming_edge_id": None,
                "state": ReversiPyGame().initial_state(),
                "placement_depth": 0,
                "model_generation": 0,
                "old_value": None,
                "black_result": None,
            },
            "node_id",
        ),
        (CollectionRecord, collection(unit_id="   "), "unit_id"),
    ],
)
def test_ids_reject_whitespace_only(record_type, values, id_field):
    with pytest.raises(ValidationError) as exc_info:
        record_type(**values)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (id_field,)
    assert errors[0]["type"] == "string_pattern_mismatch"
