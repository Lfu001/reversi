"""
Tests for MetricsCalculator
Verifies Policy Entropy, Value Accuracy, Illegal Move Prob, and Gradient Norm computations.
"""

import torch
import torch.nn as nn
from mcts_grpo.metrics_calculator import MetricsCalculator


class TestPolicyEntropy:
    """Tests for Policy Entropy computation."""

    def test_entropy_uniform_distribution(self):
        """Uniform distribution should have maximum entropy."""
        batch_size = 2
        # 4 legal moves, uniform probability
        pred_logits = torch.zeros(batch_size, 8, 8)
        legal_mask = torch.zeros(batch_size, 8, 8)
        legal_mask[:, 0, 0:4] = 1.0  # 4 legal moves

        entropy = MetricsCalculator.compute_policy_entropy(pred_logits, legal_mask)

        # Maximum entropy for 4 choices = log(4) ≈ 1.386
        assert abs(entropy - 1.386) < 0.1, f"Expected ~1.386, got {entropy}"

    def test_entropy_deterministic_distribution(self):
        """Deterministic distribution should have near-zero entropy."""
        batch_size = 2
        pred_logits = torch.zeros(batch_size, 8, 8)
        pred_logits[:, 0, 0] = 100.0  # Very high logit for one action
        legal_mask = torch.zeros(batch_size, 8, 8)
        legal_mask[:, 0, 0:4] = 1.0  # 4 legal moves

        entropy = MetricsCalculator.compute_policy_entropy(pred_logits, legal_mask)

        assert entropy < 0.1, f"Expected near 0, got {entropy}"

    def test_entropy_single_legal_move(self):
        """Single legal move should have zero entropy."""
        batch_size = 1
        pred_logits = torch.randn(batch_size, 8, 8)
        legal_mask = torch.zeros(batch_size, 8, 8)
        legal_mask[:, 0, 0] = 1.0  # Only 1 legal move

        entropy = MetricsCalculator.compute_policy_entropy(pred_logits, legal_mask)

        assert abs(entropy) < 1e-5, f"Expected 0, got {entropy}"


class TestValueAccuracy:
    """Tests for Value Accuracy computation."""

    def test_accuracy_perfect_prediction(self):
        """Perfect predictions should have accuracy 1.0."""
        pred_values = torch.tensor([0.8, -0.5, 0.3])
        outcomes = torch.tensor([1.0, -1.0, 1.0])

        accuracy = MetricsCalculator.compute_value_accuracy(pred_values, outcomes)

        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_accuracy_opposite_prediction(self):
        """Opposite predictions should have accuracy 0.0."""
        pred_values = torch.tensor([-0.8, 0.5, -0.3])
        outcomes = torch.tensor([1.0, -1.0, 1.0])

        accuracy = MetricsCalculator.compute_value_accuracy(pred_values, outcomes)

        assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"

    def test_accuracy_ignores_draws(self):
        """Draws (outcome=0) should be excluded from calculation."""
        pred_values = torch.tensor([0.8, 0.0, -0.5])
        outcomes = torch.tensor([1.0, 0.0, -1.0])  # Middle is draw

        accuracy = MetricsCalculator.compute_value_accuracy(pred_values, outcomes)

        # Only first and last count, both correct
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_accuracy_all_draws(self):
        """All draws should return 1.0 (convention)."""
        pred_values = torch.tensor([0.5, 0.0, -0.5])
        outcomes = torch.tensor([0.0, 0.0, 0.0])  # All draws

        accuracy = MetricsCalculator.compute_value_accuracy(pred_values, outcomes)

        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"


class TestIllegalMoveProb:
    """Tests for Illegal Move Probability computation."""

    def test_illegal_prob_all_legal(self):
        """When softmax is over legal moves only logits, illegal prob should be non-zero
        because we compute softmax without masking.
        """
        batch_size = 1
        pred_logits = torch.zeros(batch_size, 8, 8)
        # Make legal moves have high logits
        pred_logits[:, 0, 0:4] = 10.0
        legal_mask = torch.zeros(batch_size, 8, 8)
        legal_mask[:, 0, 0:4] = 1.0  # 4 legal moves

        illegal_prob = MetricsCalculator.compute_illegal_move_prob(
            pred_logits, legal_mask
        )

        # With high logits on legal moves, illegal prob should be near 0
        assert illegal_prob < 0.01, f"Expected near 0, got {illegal_prob}"

    def test_illegal_prob_high_on_illegal(self):
        """High logits on illegal moves should result in high illegal prob."""
        batch_size = 1
        pred_logits = torch.zeros(batch_size, 8, 8)
        pred_logits[:, 0, 4] = 10.0  # High logit on illegal move
        legal_mask = torch.zeros(batch_size, 8, 8)
        legal_mask[:, 0, 0:4] = 1.0  # Only 0-3 are legal, 4 is illegal

        illegal_prob = MetricsCalculator.compute_illegal_move_prob(
            pred_logits, legal_mask
        )

        # Most probability should be on illegal move
        assert illegal_prob > 0.9, f"Expected > 0.9, got {illegal_prob}"


class TestGradientNorm:
    """Tests for Gradient Norm computation."""

    def test_gradient_norm_simple_model(self):
        """Gradient norm should be computed correctly for simple model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        x = torch.randn(2, 4)
        y = model(x)
        loss = y.sum()
        loss.backward()

        grad_norm = MetricsCalculator.compute_gradient_norm(model)

        # Should be positive
        assert grad_norm > 0, f"Expected positive gradient norm, got {grad_norm}"

    def test_gradient_norm_zero_grad(self):
        """Zero gradients should have norm 0."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        # Don't do backward, so no gradients

        grad_norm = MetricsCalculator.compute_gradient_norm(model)

        assert grad_norm == 0.0, f"Expected 0, got {grad_norm}"

    def test_gradient_norm_does_not_mutate_model(self):
        """compute_gradient_norm should not mutate model parameters or gradients."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        x = torch.randn(2, 4)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Capture state before
        params_before = {
            name: param.clone() for name, param in model.named_parameters()
        }
        grads_before = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        # Call the function
        _ = MetricsCalculator.compute_gradient_norm(model)

        # Verify state after
        for name, param in model.named_parameters():
            assert torch.equal(param, params_before[name]), (
                f"Parameter {name} was mutated"
            )
            if param.grad is not None:
                assert torch.equal(param.grad, grads_before[name]), (
                    f"Gradient of {name} was mutated"
                )


class TestClipFraction:
    """Tests for Clip Fraction metric (computed in loss_calculator)."""

    def test_clip_fraction_integrated(self):
        """Clip fraction should be returned from loss calculator."""
        from mcts_grpo.loss_calculator import LossCalculator
        from mcts_grpo.settings import TrainingConfig, TreeGRPOConfig

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
        loss_calculator = LossCalculator(training_config, grpo_config)

        batch_size = 4
        pred_logits = torch.randn(batch_size, 8, 8)
        pred_values = torch.randn(batch_size, 1)
        states = torch.zeros(batch_size, 4, 8, 8)
        states[:, 3, 0, 0:4] = 1.0  # legal moves

        pis = torch.zeros(batch_size, 8, 8)
        pis[:, 0, 0:4] = 0.25

        outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])

        q_vals = torch.zeros(batch_size, 8, 8)
        q_vals[:, 0, 0] = 0.8
        q_vals[:, 0, 1] = 0.5
        q_vals[:, 0, 2] = 0.2
        q_vals[:, 0, 3] = -0.1

        visit_counts = torch.zeros(batch_size, 8, 8)
        visit_counts[:, 0, 0:4] = 10.0

        losses = loss_calculator.compute_all_losses(
            pred_logits, pred_values, states, pis, outcomes, q_vals, visit_counts
        )

        assert "clip_fraction" in losses, "clip_fraction should be in losses dict"
        clip_frac = losses["clip_fraction"].item()
        assert 0.0 <= clip_frac <= 1.0, (
            f"Clip fraction should be in [0, 1], got {clip_frac}"
        )
