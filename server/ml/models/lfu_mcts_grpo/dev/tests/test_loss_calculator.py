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
        train_batch_size=4,
        replay_buffer_size=1000,
        learning_rate=1e-3,
        weight_decay=0.01,
        total_training_steps=100,
        training_steps_per_iteration=10,
        warmup_steps=10,
        lambda_value=1.0,
    )
    grpo_config = TreeGRPOConfig(lambda_grpo=0.1, clip_epsilon=0.2)
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
    """Tests for GRPO Loss computation (Tree-GRPO paper approach)."""

    def test_grpo_loss_explored_filtering(self, loss_calculator):
        """Only actions with visit_count > 0 should be considered explored."""
        batch_size = 1
        pred_logits = torch.randn(batch_size, 64)
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True])

        # 4 legal moves at positions 0, 1, 2, 3
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0
        legal_moves[:, 0, 2] = 1.0
        legal_moves[:, 0, 3] = 1.0

        # Set old probabilities for importance sampling
        pis[:, 0, 0] = 0.3
        pis[:, 0, 1] = 0.3
        pis[:, 0, 2] = 0.2
        pis[:, 0, 3] = 0.2

        # Only 2 of them have visit_count > 0
        visit_counts[:, 0] = 10  # explored
        visit_counts[:, 1] = 5  # explored
        visit_counts[:, 2] = 0  # NOT explored
        visit_counts[:, 3] = 0  # NOT explored

        # Q-values (only matters for explored nodes)
        q_vals[:, 0] = 0.8  # best
        q_vals[:, 1] = 0.3  # suboptimal

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )

        assert grpo_loss != 0, (
            "GRPO loss should be non-zero when we have 2+ explored nodes with Q diff"
        )
        assert torch.isfinite(grpo_loss), "GRPO loss should be finite"

    def test_grpo_loss_insufficient_explored(self, loss_calculator):
        """GRPO loss should be 0 when less than 2 actions are explored."""
        batch_size = 1
        pred_logits = torch.randn(batch_size, 64)
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True])

        # 4 legal moves
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0
        legal_moves[:, 0, 2] = 1.0
        legal_moves[:, 0, 3] = 1.0

        pis[:, 0, 0] = 0.25
        pis[:, 0, 1] = 0.25
        pis[:, 0, 2] = 0.25
        pis[:, 0, 3] = 0.25

        # Only 1 has visit_count > 0
        visit_counts[:, 0] = 10  # only one explored

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
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
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([False])  # Terminal

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )

        assert torch.isclose(grpo_loss, torch.tensor(0.0)), (
            "GRPO loss should be 0 for terminal states"
        )

    def test_grpo_loss_advantage_normalization(self, loss_calculator):
        """Q-values should be normalized to advantages with mean 0 and std 1."""
        batch_size = 1
        pred_logits = torch.zeros(batch_size, 64)  # uniform logits
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True])

        # 4 legal moves, all explored
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0
        legal_moves[:, 0, 2] = 1.0
        legal_moves[:, 0, 3] = 1.0

        visit_counts[:, 0] = 10
        visit_counts[:, 1] = 10
        visit_counts[:, 2] = 10
        visit_counts[:, 3] = 10

        # Uniform old policy
        pis[:, 0, 0] = 0.25
        pis[:, 0, 1] = 0.25
        pis[:, 0, 2] = 0.25
        pis[:, 0, 3] = 0.25

        # Q-values with known mean and std
        # Q = [1.0, 0.5, 0.0, -0.5], mean=0.25, std≈0.56
        q_vals[:, 0] = 1.0
        q_vals[:, 1] = 0.5
        q_vals[:, 2] = 0.0
        q_vals[:, 3] = -0.5

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )

        # Loss should be non-zero with these Q-value differences
        assert grpo_loss != 0, "GRPO loss should be non-zero with Q-value variance"
        assert torch.isfinite(grpo_loss), "GRPO loss should be finite"

    def test_grpo_loss_same_q_values(self, loss_calculator):
        """GRPO loss should be 0 when all Q-values are the same (no advantage)."""
        batch_size = 1
        pred_logits = torch.randn(batch_size, 64)
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True])

        # 4 legal moves, all explored
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0
        legal_moves[:, 0, 2] = 1.0
        legal_moves[:, 0, 3] = 1.0

        visit_counts[:, 0] = 10
        visit_counts[:, 1] = 10
        visit_counts[:, 2] = 10
        visit_counts[:, 3] = 10

        pis[:, 0, 0] = 0.25
        pis[:, 0, 1] = 0.25
        pis[:, 0, 2] = 0.25
        pis[:, 0, 3] = 0.25

        # All Q-values are the same
        q_vals[:, 0] = 0.5
        q_vals[:, 1] = 0.5
        q_vals[:, 2] = 0.5
        q_vals[:, 3] = 0.5

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )

        assert torch.isclose(grpo_loss, torch.tensor(0.0)), (
            "GRPO loss should be 0 when all Q-values are the same"
        )

    def test_grpo_loss_importance_ratio_clipping(self, loss_calculator):
        """Importance ratio should be clipped when policy changes significantly."""
        batch_size = 1
        # Create logits that will produce very different probabilities from old policy
        pred_logits = torch.zeros(batch_size, 64)
        pred_logits[:, 0] = 10.0  # Will have very high probability
        pred_logits[:, 1] = -10.0  # Will have very low probability

        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True])

        # 2 legal moves
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0

        visit_counts[:, 0] = 10
        visit_counts[:, 1] = 10

        # Old policy was uniform
        pis[:, 0, 0] = 0.5
        pis[:, 0, 1] = 0.5

        # Different Q-values
        q_vals[:, 0] = 0.8
        q_vals[:, 1] = 0.2

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )

        # Loss should be finite (clipping prevents extreme values)
        assert torch.isfinite(grpo_loss), "GRPO loss should be finite with clipping"


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


class TestGradientFlowThroughLoss:
    """Tests for gradient flow through actual training loss."""

    def test_parameters_updated_after_training_step(self):
        """Verify that model parameters are updated after a complete training step."""
        from mcts_grpo.model.configuration_urm import URMConfig
        from mcts_grpo.model.modeling_urm import URMModel

        # Setup model with no truncation to ensure gradients flow
        config = URMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_layers=1,
            truncation_steps=0,
        )
        model = URMModel(config)
        model.train()

        training_config = TrainingConfig(
            train_batch_size=4,
            replay_buffer_size=1000,
            learning_rate=1e-3,
            weight_decay=0.01,
            total_training_steps=100,
            training_steps_per_iteration=10,
            warmup_steps=10,
            lambda_value=1.0,
        )
        grpo_config = TreeGRPOConfig(lambda_grpo=1.5, clip_epsilon=0.2)
        loss_calc = LossCalculator(training_config, grpo_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Save initial weights
        initial_weights = {
            name: param.data.clone() for name, param in model.named_parameters()
        }

        # Create sample data
        batch_size = 4
        states = torch.zeros(batch_size, 4, 8, 8)
        states[:, 3, 2, 3] = 1.0  # legal moves
        states[:, 3, 3, 2] = 1.0
        states[:, 3, 4, 5] = 1.0
        states[:, 3, 5, 4] = 1.0

        pis = torch.zeros(batch_size, 8, 8)
        pis[:, 2, 3] = 0.25
        pis[:, 3, 2] = 0.25
        pis[:, 4, 5] = 0.25
        pis[:, 5, 4] = 0.25

        outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])

        q_vals = torch.zeros(batch_size, 8, 8)
        q_vals[:, 2, 3] = 0.5
        q_vals[:, 3, 2] = 0.3
        q_vals[:, 4, 5] = -0.1
        q_vals[:, 5, 4] = 0.2

        visit_counts = torch.zeros(batch_size, 8, 8)
        visit_counts[:, 2, 3] = 10.0
        visit_counts[:, 3, 2] = 8.0
        visit_counts[:, 4, 5] = 5.0
        visit_counts[:, 5, 4] = 7.0

        # Forward pass
        optimizer.zero_grad()
        policy_logits, value = model(states)

        losses = loss_calc.compute_all_losses(
            policy_logits,
            value.unsqueeze(-1),
            states,
            pis,
            outcomes,
            q_vals,
            visit_counts,
        )
        total_loss = loss_calc.compute_total_loss(losses)

        # Verify loss has gradient
        # Backward + step
        total_loss.backward()

        # Check ALL trainable parameters have gradients
        params_with_grad = []
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad.append(name)
                else:
                    params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"All trainable parameters should have non-zero gradients.\n"
            f"Params WITH gradients ({len(params_with_grad)}): {params_with_grad}\n"
            f"Params WITHOUT gradients ({len(params_without_grad)}): {params_without_grad}"
        )

        # Now apply optimizer step
        optimizer.step()

        # Count changed weights
        weights_changed = []
        weights_unchanged = []
        for name, param in model.named_parameters():
            if not torch.equal(param.data, initial_weights[name]):
                weights_changed.append(name)
            else:
                weights_unchanged.append(name)

        assert len(weights_changed) > 0, (
            f"At least some weights should change after training step.\n"
            f"Changed: {weights_changed}\n"
            f"Unchanged: {weights_unchanged}"
        )

    def test_all_loss_components_contribute_gradients(self):
        """Verify each loss component contributes to gradients."""
        from mcts_grpo.model.configuration_urm import URMConfig
        from mcts_grpo.model.modeling_urm import URMModel

        config = URMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_layers=1,
            truncation_steps=0,
        )
        model = URMModel(config)
        model.train()

        training_config = TrainingConfig(
            train_batch_size=4,
            replay_buffer_size=1000,
            learning_rate=1e-3,
            weight_decay=0.01,
            total_training_steps=100,
            training_steps_per_iteration=10,
            warmup_steps=10,
            lambda_value=1.0,
        )
        grpo_config = TreeGRPOConfig(lambda_grpo=1.5, clip_epsilon=0.2)
        loss_calc = LossCalculator(training_config, grpo_config)

        # Create sample data with enough explored nodes for GRPO
        batch_size = 4
        states = torch.zeros(batch_size, 4, 8, 8)
        states[:, 3, 2, 3] = 1.0
        states[:, 3, 3, 2] = 1.0
        states[:, 3, 4, 5] = 1.0
        states[:, 3, 5, 4] = 1.0

        pis = torch.zeros(batch_size, 8, 8)
        pis[:, 2, 3] = 0.25
        pis[:, 3, 2] = 0.25
        pis[:, 4, 5] = 0.25
        pis[:, 5, 4] = 0.25

        outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])

        q_vals = torch.zeros(batch_size, 8, 8)
        q_vals[:, 2, 3] = 0.5
        q_vals[:, 3, 2] = 0.3
        q_vals[:, 4, 5] = -0.1
        q_vals[:, 5, 4] = 0.2

        visit_counts = torch.zeros(batch_size, 8, 8)
        visit_counts[:, 2, 3] = 10.0
        visit_counts[:, 3, 2] = 8.0
        visit_counts[:, 4, 5] = 5.0
        visit_counts[:, 5, 4] = 7.0

        # Forward
        policy_logits, value = model(states)

        losses = loss_calc.compute_all_losses(
            policy_logits,
            value.unsqueeze(-1),
            states,
            pis,
            outcomes,
            q_vals,
            visit_counts,
        )

        # Check each loss has gradient
        assert losses["value_loss"].requires_grad, "Value loss should require gradients"
        assert losses["policy_loss"].requires_grad, (
            "Policy loss should require gradients"
        )
        # GRPO may be 0 if no valid pairs, but should still be a tensor
        assert isinstance(losses["grpo_loss"], torch.Tensor), (
            "GRPO loss should be tensor"
        )

    def test_gradients_with_production_truncation(self):
        """Test gradient flow with production config (truncation_steps=2).

        This tests the TBPTL mechanism where early loops run without gradients.
        The model should still learn - backbone layers and heads should receive
        gradients from the non-truncated loops.
        """
        from mcts_grpo.model.configuration_urm import URMConfig
        from mcts_grpo.model.modeling_urm import URMModel

        # Production-like config with truncation
        config = URMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_layers=1,
            num_inner_loops=4,
            truncation_steps=2,  # Half of loops truncated
        )
        model = URMModel(config)
        model.train()

        training_config = TrainingConfig(
            train_batch_size=4,
            replay_buffer_size=1000,
            learning_rate=1e-3,
            weight_decay=0.01,
            total_training_steps=100,
            training_steps_per_iteration=10,
            warmup_steps=10,
            lambda_value=1.0,
        )
        grpo_config = TreeGRPOConfig(lambda_grpo=1.5, clip_epsilon=0.2)
        loss_calc = LossCalculator(training_config, grpo_config)

        # Create sample data
        batch_size = 4
        states = torch.zeros(batch_size, 4, 8, 8)
        states[:, 3, 2, 3] = 1.0
        states[:, 3, 3, 2] = 1.0
        states[:, 3, 4, 5] = 1.0
        states[:, 3, 5, 4] = 1.0

        pis = torch.zeros(batch_size, 8, 8)
        pis[:, 2, 3] = 0.25
        pis[:, 3, 2] = 0.25
        pis[:, 4, 5] = 0.25
        pis[:, 5, 4] = 0.25

        outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])

        q_vals = torch.zeros(batch_size, 8, 8)
        q_vals[:, 2, 3] = 0.5
        q_vals[:, 3, 2] = 0.3
        q_vals[:, 4, 5] = -0.1
        q_vals[:, 5, 4] = 0.2

        visit_counts = torch.zeros(batch_size, 8, 8)
        visit_counts[:, 2, 3] = 10.0
        visit_counts[:, 3, 2] = 8.0
        visit_counts[:, 4, 5] = 5.0
        visit_counts[:, 5, 4] = 7.0

        # Forward
        policy_logits, value = model(states)

        losses = loss_calc.compute_all_losses(
            policy_logits,
            value.unsqueeze(-1),
            states,
            pis,
            outcomes,
            q_vals,
            visit_counts,
        )
        total_loss = loss_calc.compute_total_loss(losses)
        total_loss.backward()

        # Check gradients for critical components
        params_with_grad = []
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad.append(name)
                else:
                    params_without_grad.append(name)

        # With TBPTL, input_embed should still receive gradients
        # because input_embeddings are added at each loop iteration
        input_embed_has_grad = any("input_embed" in p for p in params_with_grad)
        backbone_has_grad = any("backbone" in p for p in params_with_grad)
        policy_head_has_grad = any("policy_head" in p for p in params_with_grad)
        value_head_has_grad = any("value_head" in p for p in params_with_grad)

        assert backbone_has_grad, (
            f"Backbone should have gradients even with truncation.\n"
            f"Params with grad: {params_with_grad}\n"
            f"Params without grad: {params_without_grad}"
        )
        assert policy_head_has_grad, "Policy head should have gradients"
        assert value_head_has_grad, "Value head should have gradients"
        assert input_embed_has_grad, (
            f"input_embed should have gradients (added at each loop).\n"
            f"Params with grad: {params_with_grad}"
        )


class TestGRPOGradientFlow:
    """Specific tests for GRPO loss gradient flow through vectorized operations."""

    def test_grpo_loss_has_gradient(self, loss_calculator):
        """Verify that GRPO loss maintains computation graph and gradients flow."""
        batch_size = 4
        # Create logits that require grad
        pred_logits = torch.randn(batch_size, 64, requires_grad=True)
        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True, True, True, True])

        # Set up 4 legal moves per sample, all explored
        for i in range(4):
            legal_moves[:, 0, i] = 1.0
            visit_counts[:, i] = 10.0
            pis[:, 0, i] = 0.25

        # Different Q-values to ensure non-zero advantage
        q_vals[:, 0] = 0.8
        q_vals[:, 1] = 0.5
        q_vals[:, 2] = 0.2
        q_vals[:, 3] = -0.1

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )

        # Verify loss requires grad
        assert grpo_loss.requires_grad, "GRPO loss should require gradients"

        # Backward and check gradient exists
        grpo_loss.backward()
        assert pred_logits.grad is not None, "pred_logits should have gradients"
        assert pred_logits.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_grpo_vectorized_gradients_match_expected_direction(self, loss_calculator):
        """Verify gradients push probabilities in expected direction based on advantages."""
        batch_size = 1
        # Start with uniform logits
        pred_logits = torch.zeros(batch_size, 64, requires_grad=True)

        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True])

        # 2 legal moves
        legal_moves[:, 0, 0] = 1.0
        legal_moves[:, 0, 1] = 1.0
        visit_counts[:, 0] = 10.0
        visit_counts[:, 1] = 10.0
        pis[:, 0, 0] = 0.5
        pis[:, 0, 1] = 0.5

        # Action 0 is better (higher Q-value)
        q_vals[:, 0] = 0.9
        q_vals[:, 1] = 0.1

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )
        grpo_loss.backward()

        # Gradient for action 0 should be negative (decrease loss = increase probability)
        # Gradient for action 1 should be positive (decrease loss = decrease probability)
        grad = pred_logits.grad[0]
        assert grad[0] < grad[1], (
            f"Better action should have lower gradient (to increase prob). "
            f"grad[0]={grad[0].item():.4f}, grad[1]={grad[1].item():.4f}"
        )

    def test_inter_tree_advantage_contributes(self, loss_calculator):
        """Verify inter-tree advantage (batch-wide normalization) contributes to loss."""
        batch_size = 2
        pred_logits = torch.zeros(batch_size, 64, requires_grad=True)

        q_vals = torch.zeros(batch_size, 64)
        legal_moves = torch.zeros(batch_size, 8, 8)
        visit_counts = torch.zeros(batch_size, 64)
        pis = torch.zeros(batch_size, 8, 8)
        non_terminal_mask = torch.tensor([True, True])

        # Sample 0: Q-values [0.8, 0.2]
        # Sample 1: Q-values [0.6, 0.4]
        for i in range(2):
            legal_moves[:, 0, i] = 1.0
            visit_counts[:, i] = 10.0
            pis[:, 0, i] = 0.5

        q_vals[0, 0] = 0.8
        q_vals[0, 1] = 0.2
        q_vals[1, 0] = 0.6
        q_vals[1, 1] = 0.4

        grpo_loss = loss_calculator._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
        )

        # Loss should be non-zero (both intra and inter-tree contribute)
        assert grpo_loss != 0, "GRPO loss should be non-zero with inter-tree"
        assert grpo_loss.requires_grad, "GRPO loss should require gradients"

        grpo_loss.backward()
        assert pred_logits.grad is not None, "Gradients should flow"
        assert pred_logits.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_nanstd_gradient_flow(self):
        """Verify custom _nanstd function preserves gradient flow."""
        from mcts_grpo.loss_calculator import _nanstd

        x = torch.tensor([[1.0, 2.0, float("nan"), 4.0]], requires_grad=True)
        std = _nanstd(x, dim=1)

        assert std.requires_grad, "_nanstd should preserve requires_grad"

        std.sum().backward()
        assert x.grad is not None, "Gradient should flow through _nanstd"
        # Non-NaN positions should have finite gradients
        assert torch.isfinite(x.grad[0, 0]), (
            "Valid position 0 should have finite gradient"
        )
        assert torch.isfinite(x.grad[0, 1]), (
            "Valid position 1 should have finite gradient"
        )
        assert torch.isfinite(x.grad[0, 3]), (
            "Valid position 3 should have finite gradient"
        )
        # At least some non-NaN positions should have non-zero gradients
        assert (x.grad[0, 0] != 0) or (x.grad[0, 1] != 0) or (x.grad[0, 3] != 0), (
            "Valid positions should have non-zero gradients"
        )
