"""Acceptance tests for sampled-action, path-weighted PPO and value loss."""

import math
from dataclasses import replace

import pytest
import torch
from tests.test_records import build_record, position
from tests.test_returns import uneven_record

from lfu_tree_self_play.advantages import compute_advantages
from lfu_tree_self_play.collector import scale_return_targets
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.losses import LossBatch, build_loss_batch, ppo_value_loss
from lfu_tree_self_play.model import (
    PolicyValueConfig,
    PolicyValueModel,
    PolicyValueOutput,
)
from lfu_tree_self_play.returns import aggregate_returns
from lfu_tree_self_play.tree_collector import collect_tree
from lfu_tree_self_play.verification import endgame_position


def uniform_output(batch):
    return PolicyValueOutput(
        torch.zeros((len(batch.node_ids), 64), dtype=torch.float64, requires_grad=True),
        torch.zeros(len(batch.node_ids), dtype=torch.float64, requires_grad=True),
    )


def batch_for(record, game=None, targets=None):
    return build_loss_batch(
        record,
        targets or aggregate_returns(record, game or ReversiPyGame()),
        {edge.edge_id: 1.0 for edge in record.edges},
    )


def test_old_policy_ratios_match_collected_logs_and_model_gets_gradients():
    torch.manual_seed(100)
    model = PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)).eval()
    tree = collect_tree(
        model,
        endgame_position(-1),
        unit_id="loss-tree",
        experiment_id="loss-test",
        round_id="round-0",
        model_generation=0,
        run_seed=100,
        levels=2,
    )
    batch = batch_for(tree.record, targets=tree.targets)
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    result = ppo_value_loss(model(batch.observations), batch, num_roots=1)

    torch.testing.assert_close(result.ratios, torch.ones_like(result.ratios))
    assert result.ratios.shape == (len(tree.record.edges),)
    (result.policy + result.value).backward()
    assert model.policy_head.weight.grad.abs().sum() > 0
    assert model.value_head[2].weight.grad.abs().sum() > 0
    assert model.stem[0].weight.grad.abs().sum() > 0
    assert all(
        torch.equal(tensor, before[key]) for key, tensor in model.state_dict().items()
    )


def clipping_batch():
    observations = torch.zeros((4, 4, 8, 8))
    observations[:, 3, 0, :2] = 1
    batch = LossBatch(
        node_ids=("a", "b", "c", "d"),
        edge_ids=("e0", "e1", "e2", "e3"),
        observations=observations,
        edge_parent_indices=torch.arange(4),
        actions=torch.zeros(4, dtype=torch.long),
        old_log_probabilities=torch.full((4,), math.log(0.5), dtype=torch.float64),
        advantages=torch.tensor([1.0, 1.0, -1.0, -1.0]),
        edge_weights=torch.tensor([1.0, 0.5, 0.25, 0.125]),
        value_targets=torch.tensor([1.0, -1.0, 0.5, 0.0]),
        node_weights=torch.tensor([1.0, 0.5, 0.25, 0.125]),
    )
    logits = torch.zeros((4, 64), dtype=torch.float64)
    logits[:, 0] = torch.tensor([math.log(3), -math.log(3)] * 2, dtype=torch.float64)
    return batch, PolicyValueOutput(
        logits.requires_grad_(), torch.zeros(4, requires_grad=True)
    )


def test_clipping_both_advantage_signs_matches_hand_calculation():
    batch, output = clipping_batch()
    result = ppo_value_loss(output, batch, num_roots=2)
    torch.testing.assert_close(result.ratios, torch.tensor([1.5, 0.5] * 2).double())
    assert result.policy.item() == pytest.approx(
        -(1.2 + 0.5 * 0.5 - 0.25 * 1.5 - 0.125 * 0.8) / 120
    )
    assert result.value.item() == pytest.approx((1 + 0.5 + 0.25 * 0.25) / 120)
    result.policy.backward()
    gradient = output.policy_logits.grad[:, 0]
    assert gradient[0] == 0  # positive advantage clipped above 1 + epsilon
    assert gradient[1] < 0
    assert gradient[2] > 0
    assert gradient[3] == 0  # negative advantage clipped below 1 - epsilon
    assert torch.count_nonzero(output.policy_logits.grad[:, 2:]) == 0


def test_old_probabilities_advantages_targets_and_weights_receive_no_gradients():
    batch, output = clipping_batch()
    teachers = {
        field: getattr(batch, field).clone().requires_grad_()
        for field in (
            "old_log_probabilities",
            "advantages",
            "edge_weights",
            "value_targets",
            "node_weights",
        )
    }
    batch = replace(batch, **teachers)
    result = ppo_value_loss(output, batch, num_roots=1)
    (result.policy + result.value).backward()
    assert all(tensor.grad is None for tensor in teachers.values())
    assert output.policy_logits.grad.abs().sum() > 0
    assert output.value.grad.abs().sum() > 0


def test_uneven_tree_uses_node_mean_targets_and_actual_mover_perspective():
    record, game = uneven_record()
    batch = batch_for(record, game)
    # This enumerated topology reuses a Reversi template's historical logs.
    # Set its old policy to the uniform policy over the enumerated legal moves.
    batch = replace(
        batch,
        old_log_probabilities=-batch.observations[:, 3]
        .flatten(1)
        .sum(-1)
        .double()
        .log()[batch.edge_parent_indices],
    )
    result = ppo_value_loss(uniform_output(batch), batch, num_roots=1)
    assert batch.node_ids == ("root", "right", "sample1", "sample2")
    torch.testing.assert_close(
        batch.value_targets, torch.tensor([-0.25, 0.5, -1.0, 0.0]).double()
    )
    # The early terminal leaf and two deeper leaves do not get equal mass.
    assert batch.edge_weights.tolist() == [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
    assert result.policy.item() == pytest.approx(-2 / 60)
    assert result.value.item() == pytest.approx((0.25**2 + 0.5 * 0.5**2 + 0.25) / 60)


@pytest.mark.parametrize(
    ("field", "expected_policy"),
    [
        ("standard_edge_advantages", 0.875 / 60),
        ("leave_one_out_edge_advantages", 0.375 / 60),
    ],
)
def test_computed_b_and_c_advantages_feed_the_same_weighted_loss(
    field, expected_policy
):
    record, game = uneven_record()
    old_values = {"root": 0.4, "right": 0.2, "sample1": -0.25, "sample2": 0.75}
    nodes = {node.node_id: node for node in record.nodes}
    record = record.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"old_value": old_values[node.node_id]})
                if node.node_id in old_values
                else node
                for node in record.nodes
            ),
            "edges": tuple(
                edge.model_copy(
                    update={
                        "old_log_probability": -math.log(
                            len(nodes[edge.parent_node_id].state.legal_actions)
                        )
                    }
                )
                for edge in record.edges
            ),
        }
    )
    advantages = compute_advantages(record, game)
    batch = build_loss_batch(
        record, aggregate_returns(record, game), getattr(advantages, field)
    )
    result = ppo_value_loss(uniform_output(batch), batch, num_roots=1)
    torch.testing.assert_close(result.ratios, torch.ones_like(result.ratios))
    assert result.policy.item() == pytest.approx(expected_policy)
    assert result.value.item() == pytest.approx(0.4375 / 60)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_shared_prefix_is_counted_once_when_descendants_are_duplicated(width):
    plain = batch_for(build_record(positions=()))
    branched = batch_for(build_record(positions=(1,), width=width))
    plain_output, branched_output = uniform_output(plain), uniform_output(branched)
    plain_loss = ppo_value_loss(plain_output, plain, num_roots=1)
    branched_loss = ppo_value_loss(branched_output, branched, num_roots=1)
    assert len(branched.edge_ids) > len(plain.edge_ids)
    torch.testing.assert_close(plain_loss.policy, branched_loss.policy)
    torch.testing.assert_close(plain_loss.value, branched_loss.value)
    plain_loss.policy.backward()
    branched_loss.policy.backward()
    torch.testing.assert_close(
        plain_output.policy_logits.grad[0], branched_output.policy_logits.grad[0]
    )


def test_independent_k_paths_keep_one_roots_mass():
    record = build_record(mode="independent", positions=())
    targets = aggregate_returns(record, ReversiPyGame())
    plain = batch_for(record, targets=targets)
    weighted = batch_for(record, targets=scale_return_targets(targets, 1 / 4))
    first = ppo_value_loss(uniform_output(plain), plain, num_roots=1)
    path = ppo_value_loss(uniform_output(weighted), weighted, num_roots=1)
    torch.testing.assert_close(path.policy * 4, first.policy)
    torch.testing.assert_close(path.value * 4, first.value)


def test_short_game_keeps_fixed_sixty_and_root_average():
    record = build_record(
        root=position((".WBBBBBB",) + ("BBBBBBBB",) * 7, 1, (0,)), positions=()
    )
    batch = batch_for(record)
    result = ppo_value_loss(uniform_output(batch), batch, num_roots=2)
    assert len(batch.edge_ids) == 1
    assert result.policy.item() == pytest.approx(-1 / 120)
    assert result.value.item() == pytest.approx(1 / 120)


def test_terminal_root_contributes_zero_without_creating_a_policy_sample():
    record = build_record(root=position(("BBBBBBBB",) * 8, 1, ()), positions=())
    batch = batch_for(record)
    output = uniform_output(batch)
    result = ppo_value_loss(output, batch, num_roots=1)
    assert batch.observations.shape == (0, 4, 8, 8)
    assert result.ratios.shape == (0,)
    assert result.policy.item() == result.value.item() == 0
    (result.policy + result.value).backward()
    assert output.policy_logits.grad is not None
    assert output.value.grad is not None


def test_record_order_is_used_and_missing_advantages_are_rejected():
    record = build_record()
    record = record.model_copy(
        update={"nodes": record.nodes[::-1], "edges": record.edges[::-1]}
    )
    targets = aggregate_returns(record, ReversiPyGame())
    advantages = {edge.edge_id: float(i) for i, edge in enumerate(record.edges)}
    batch = build_loss_batch(record, targets, advantages)
    assert batch.edge_ids == tuple(advantages)
    assert batch.advantages.tolist() == list(advantages.values())
    assert batch.edge_parent_indices.tolist() == [
        batch.node_ids.index(edge.parent_node_id) for edge in record.edges
    ]
    with pytest.raises(ValueError, match="advantages IDs"):
        build_loss_batch(record, targets, {})
    with pytest.raises(ValueError, match="edge weights IDs"):
        build_loss_batch(record, replace(targets, edge_weights={}), advantages)


@pytest.mark.parametrize("num_roots", [0, -1, True, 1.5])
def test_invalid_root_count_is_rejected(num_roots):
    batch, output = clipping_batch()
    with pytest.raises(ValueError, match="num_roots"):
        ppo_value_loss(output, batch, num_roots=num_roots)


@pytest.mark.parametrize("epsilon", [0, -1, 1, True, float("nan"), float("inf")])
def test_invalid_clip_epsilon_is_rejected(epsilon):
    batch, output = clipping_batch()
    with pytest.raises(ValueError, match="clip_epsilon"):
        ppo_value_loss(output, batch, num_roots=1, clip_epsilon=epsilon)


@pytest.mark.parametrize(
    "field",
    [
        "old_log_probabilities",
        "advantages",
        "value_targets",
        "edge_weights",
        "node_weights",
    ],
)
@pytest.mark.parametrize("bad", ["shape", "nan"])
def test_targets_and_weights_cannot_broadcast_or_propagate_nan(field, bad):
    batch, output = clipping_batch()
    value = torch.zeros((4, 1)) if bad == "shape" else torch.full((4,), torch.nan)
    with pytest.raises(ValueError, match="finite, matching vectors"):
        ppo_value_loss(output, replace(batch, **{field: value}), num_roots=1)


@pytest.mark.parametrize(
    "case",
    [
        "illegal",
        "parent",
        "actions",
        "terminal",
        "nan_logits",
        "nan_value",
        "value_shape",
        "overflow",
    ],
)
def test_invalid_predictions_or_samples_fail_clearly(case):
    batch, output = clipping_batch()
    if case == "illegal":
        batch = replace(batch, actions=torch.full((4,), 63))
    elif case == "parent":
        batch = replace(batch, edge_parent_indices=torch.full((4,), 4))
    elif case == "actions":
        batch = replace(batch, actions=batch.actions.float())
    elif case == "terminal":
        batch = replace(batch, observations=torch.zeros_like(batch.observations))
    elif case == "nan_logits":
        output = output._replace(policy_logits=torch.full((4, 64), torch.nan))
    elif case == "nan_value":
        output = output._replace(value=torch.full((4,), torch.nan))
    elif case == "value_shape":
        output = output._replace(value=torch.zeros((4, 1)))
    else:
        batch = replace(batch, old_log_probabilities=torch.full((4,), -10000.0))
    with pytest.raises(ValueError):
        ppo_value_loss(output, batch, num_roots=1)


def test_illegal_logits_are_masked_even_if_nan():
    batch, output = clipping_batch()
    logits = output.policy_logits.detach().clone()
    logits[:, 2:] = torch.nan
    logits.requires_grad_()
    result = ppo_value_loss(output._replace(policy_logits=logits), batch, num_roots=1)
    result.policy.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_keeps_loss_and_gradients_on_device_without_float64():
    batch, cpu_output = clipping_batch()
    output = PolicyValueOutput(
        cpu_output.policy_logits.detach()
        .to(device="mps", dtype=torch.float32)
        .requires_grad_(),
        cpu_output.value.detach().to(device="mps").requires_grad_(),
    )
    expected = ppo_value_loss(cpu_output, batch, num_roots=1)
    result = ppo_value_loss(output, batch, num_roots=1)
    assert result.policy.device.type == result.value.device.type == "mps"
    assert result.policy.item() == pytest.approx(expected.policy.item(), abs=1e-7)
    assert result.value.item() == pytest.approx(expected.value.item(), abs=1e-7)
    (result.policy + result.value).backward()
    assert torch.isfinite(output.policy_logits.grad).all()
    assert torch.isfinite(output.value.grad).all()
