"""
Tests for LossCalculator
Verifies Value Loss, Policy Loss, and GRPO Loss computations.
"""

import pytest
import torch
from mcts_grpo.loss_calculator import LossCalculator
from mcts_grpo.settings import TrainingConfig, TreeGRPOConfig


@pytest.fixture
def loss_calculator():
    """Create a LossCalculator with default configs."""
    training_config = TrainingConfig(
        batch_size=4,
        replay_buffer_size=1000,
        learning_rate=1e-3,
        weight_decay=0.01,
        total_training_steps=100,
        games_per_iteration=10,
        training_steps_per_iteration=10,
        warmup_steps=10,
        lambda_value=1.0,
    )
    grpo_config = TreeGRPOConfig(lambda_grpo=0.1, grpo_num_pairs=3)
    return LossCalculator(training_config, grpo_config)


@pytest.fixture
def sample_batch():
    """Create sample batch data for testing."""
    batch_size = 4
    # Random logits for policy
    pred_logits = torch.randn(batch_size, 8, 8)
    pred_values = torch.randn(batch_size, 1)

    # States: 4 channels (dark, light, turn, legal_moves)
    states = torch.zeros(batch_size, 4, 8, 8)
    # Set some legal moves
    states[:, 3, 2, 3] = 1.0
    states[:, 3, 3, 2] = 1.0
    states[:, 3, 4, 5] = 1.0
    states[:, 3, 5, 4] = 1.0

    # Target policy (normalized)
    pis = torch.zeros(batch_size, 8, 8)
    pis[:, 2, 3] = 0.25
    pis[:, 3, 2] = 0.25
    pis[:, 4, 5] = 0.25
    pis[:, 5, 4] = 0.25

    # Outcomes: -1, 0, 1
    outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])

    # Q-values (64 dimensions)
    q_vals = torch.zeros(batch_size, 8, 8)
    q_vals[:, 2, 3] = 0.5
    q_vals[:, 3, 2] = 0.3
    q_vals[:, 4, 5] = -0.1
    q_vals[:, 5, 4] = 0.2

    # Visit counts (64 dimensions)
    visit_counts = torch.zeros(batch_size, 8, 8)
    visit_counts[:, 2, 3] = 10.0
    visit_counts[:, 3, 2] = 8.0
    visit_counts[:, 4, 5] = 5.0
    visit_counts[:, 5, 4] = 7.0

    return {
        "pred_logits": pred_logits,
        "pred_values": pred_values,
        "states": states,
        "pis": pis,
        "outcomes": outcomes,
        "q_vals": q_vals,
        "visit_counts": visit_counts,
    }


class TestValueLoss:
    """Tests for Value Loss computation."""

    def test_value_loss_computation(self, loss_calculator, sample_batch):
        """MSE loss should be correctly computed."""
        pred_values = sample_batch["pred_values"]
        outcomes = sample_batch["outcomes"]

        value_loss = loss_calculator._compute_value_loss(pred_values, outcomes)

        assert value_loss >= 0, "Value loss should be non-negative"
        assert torch.isfinite(value_loss), "Value loss should be finite"

    def test_value_loss_perfect_prediction(self, loss_calculator):
        """MSE loss should be 0 when prediction equals target."""
        outcomes = torch.tensor([1.0, -1.0, 0.0])
        pred_values = outcomes.clone()

        value_loss = loss_calculator._compute_value_loss(pred_values, outcomes)

        assert torch.isclose(value_loss, torch.tensor(0.0)), (
            "Perfect prediction should have 0 loss"
        )


class TestGRPOLoss:
    """Tests for GRPO Loss computation."""

    def test_grpo_loss_explored_filtering(self, loss_calculator):
        """Only actions with visit_count > 0 should be considered explored."""
        batch_size = 1
        pred_logits = torch.randn(batch_size, 64)
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        non_terminal_mask = torch.tensor([True])

        # 4 legal moves at positions 0, 1, 2, 3
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0
        legal_moves[:, 0, 2] = 1.0
        legal_moves[:, 0, 3] = 1.0

        # Only 2 of them have visit_count > 0
        visit_counts[:, 0] = 10  # explored
        visit_counts[:, 1] = 5  # explored
        visit_counts[:, 2] = 0  # NOT explored
        visit_counts[:, 3] = 0  # NOT explored

        # Q-values (only matters for explored nodes)
        q_vals[:, 0] = 0.8  # best
        q_vals[:, 1] = 0.3  # suboptimal

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, non_terminal_mask
        )

        assert grpo_loss > 0, (
            "GRPO loss should be positive when we have 2+ explored nodes"
        )
        assert torch.isfinite(grpo_loss), "GRPO loss should be finite"

    def test_grpo_loss_insufficient_explored(self, loss_calculator):
        """GRPO loss should be 0 when less than 2 actions are explored."""
        batch_size = 1
        pred_logits = torch.randn(batch_size, 64)
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        non_terminal_mask = torch.tensor([True])

        # 4 legal moves
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0
        legal_moves[:, 0, 2] = 1.0
        legal_moves[:, 0, 3] = 1.0

        # Only 1 has visit_count > 0
        visit_counts[:, 0] = 10  # only one explored

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, non_terminal_mask
        )

        assert torch.isclose(grpo_loss, torch.tensor(0.0)), (
            "GRPO loss should be 0 when < 2 explored nodes"
        )

    def test_grpo_loss_terminal_state(self, loss_calculator):
        """GRPO loss should be 0 for terminal states (no legal moves)."""
        batch_size = 1
        pred_logits = torch.randn(batch_size, 64)
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)  # No legal moves
        visit_counts = torch.zeros(batch_size, 64)
        non_terminal_mask = torch.tensor([False])  # Terminal

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, non_terminal_mask
        )

        assert torch.isclose(grpo_loss, torch.tensor(0.0)), (
            "GRPO loss should be 0 for terminal states"
        )


class TestComputeAllLosses:
    """Tests for the combined loss computation."""

    def test_all_losses_computed(self, loss_calculator, sample_batch):
        """All three loss components should be computed."""
        losses = loss_calculator.compute_all_losses(
            sample_batch["pred_logits"],
            sample_batch["pred_values"],
            sample_batch["states"],
            sample_batch["pis"],
            sample_batch["outcomes"],
            sample_batch["q_vals"],
            sample_batch["visit_counts"],
        )

        assert "value_loss" in losses
        assert "policy_loss" in losses
        assert "grpo_loss" in losses
        assert all(torch.isfinite(v) for v in losses.values())

    def test_total_loss_combination(self, loss_calculator, sample_batch):
        """Total loss should combine components with correct weights."""
        losses = loss_calculator.compute_all_losses(
            sample_batch["pred_logits"],
            sample_batch["pred_values"],
            sample_batch["states"],
            sample_batch["pis"],
            sample_batch["outcomes"],
            sample_batch["q_vals"],
            sample_batch["visit_counts"],
        )

        total_loss = loss_calculator.compute_total_loss(losses)

        expected = (
            loss_calculator.training_config.lambda_value * losses["value_loss"]
            + losses["policy_loss"]
            + loss_calculator.grpo_config.lambda_grpo * losses["grpo_loss"]
        )
        assert torch.isclose(total_loss, expected), (
            "Total loss should match weighted sum"
        )
