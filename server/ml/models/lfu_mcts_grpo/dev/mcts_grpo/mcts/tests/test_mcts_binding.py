import mcts
import numpy as np
import pytest


class MockModel:
    def inference(self, states):
        # states: [B, 4, 8, 8]
        batch_size = states.shape[0]
        # Return normalized policy probabilities (shape: [batch, 64]) and value (shape: [batch, 1])
        # Note: When testing RustMCTS directly (bypassing Python MCTS wrapper),
        # the callback should return already-normalized probabilities since the
        # mask+softmax is done in the Python MCTS wrapper
        policy_probs = np.random.dirichlet(np.ones(64), size=batch_size).astype(
            np.float32
        )
        value = np.random.uniform(-1, 1, size=(batch_size, 1)).astype(np.float32)
        return policy_probs, value


@pytest.fixture
def initial_states():
    """Create initial Reversi board states for testing."""
    batch_size = 2
    states = np.zeros((batch_size, 4, 8, 8), dtype=np.float32)

    # Set up initial Reversi board state
    # Standard starting position: Dark at (3,3) and (4,4), Light at (3,4) and (4,3)
    for i in range(batch_size):
        # Channel 0: Dark disks
        states[i, 0, 3, 3] = 1.0
        states[i, 0, 4, 4] = 1.0
        # Channel 1: Light disks
        states[i, 1, 3, 4] = 1.0
        states[i, 1, 4, 3] = 1.0
        # Channel 2: Turn (1.0 for Dark)
        states[i, 2, :, :] = 1.0
        # Channel 3: Legal moves (will be filled by MCTS internally, but set some dummy values)
        states[i, 3, 2, 3] = 1.0
        states[i, 3, 3, 2] = 1.0
        states[i, 3, 4, 5] = 1.0
        states[i, 3, 5, 4] = 1.0

    return states


def test_mcts_run(initial_states):
    """Test MCTS run with mock model."""
    # MCTS with explicit batch configuration
    max_inference_batch_size = 32
    states_per_inference = 8
    # Use RustMCTS to test the low-level Rust binding
    mcts_instance = mcts.RustMcts(max_inference_batch_size, states_per_inference)
    model = MockModel()

    # MCTS parameters
    num_simulations = 10
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 1.0
    c_puct = 1.0

    # Run MCTS - now returns (pi, q_values, visit_counts) where pi is already normalized
    pi, q_values, visit_counts = mcts_instance.run(
        initial_states,
        model.inference,
        num_simulations,
        dirichlet_epsilon,
        dirichlet_alpha,
        c_puct,
        None,  # seed
    )

    print(f"Pi shape: {pi.shape}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Pi:\n{pi}")
    print(f"Q-values:\n{q_values}")

    # Assertions
    batch_size = initial_states.shape[0]

    # Check shapes
    assert pi.shape == (batch_size, 64), (
        f"Expected pi shape ({batch_size}, 64), got {pi.shape}"
    )
    assert q_values.shape == (batch_size, 64), (
        f"Expected q_values shape ({batch_size}, 64), got {q_values.shape}"
    )

    # Check types (Rust returns f64, which becomes float64 in numpy)
    assert pi.dtype == np.float64, f"Expected pi dtype float64, got {pi.dtype}"
    assert q_values.dtype == np.float64, (
        f"Expected q_values dtype float64, got {q_values.dtype}"
    )

    # Check that pi values are non-negative and sum to 1.0
    assert np.all(pi >= 0), "Pi values should be non-negative"
    assert np.all(pi <= 1), "Pi values should be at most 1.0"

    # Check that each state's pi sums to approximately 1.0 (normalized distribution)
    for i in range(batch_size):
        pi_sum = pi[i].sum()
        assert np.isclose(pi_sum, 1.0, atol=1e-6), (
            f"State {i} pi sum is {pi_sum}, expected 1.0"
        )

        # Check that Q-values are finite
        assert np.all(np.isfinite(q_values[i])), (
            f"Q-values should be finite for state {i}"
        )


def test_mcts_seed_argument(initial_states):
    """Test that MCTS accepts a seed argument."""
    max_inference_batch_size = 32
    states_per_inference = 8
    mcts_instance = mcts.MCTS(max_inference_batch_size, states_per_inference)
    model = MockModel()

    # MCTS parameters
    num_simulations = 10
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 1.0
    c_puct = 1.0
    seed = 42

    # Run with seed
    pi, q, visit_counts = mcts_instance.run_simulations(
        model.inference,
        initial_states,
        None,  # device (not used by MockModel)
        num_simulations,
        dirichlet_epsilon,
        dirichlet_alpha,
        c_puct,
        seed=seed,
    )

    # Check output shapes and validity
    batch_size = initial_states.shape[0]
    assert pi.shape == (batch_size, 64)
    assert q.shape == (batch_size, 64)
    assert np.all(pi >= 0)
    assert np.all(pi <= 1)
    assert np.all(np.isfinite(q))


def test_policy_no_nan_with_legal_moves():
    """Test that policy does not contain NaN when there are legal moves."""
    import torch

    batch_size = 4
    # Create policy logits (random values)
    policy_logits = torch.randn(batch_size, 8, 8)

    # Create legal mask with some legal moves
    legal_mask = torch.zeros(batch_size, 8, 8)
    legal_mask[:, 2, 3] = 1.0  # Position (2,3) is legal
    legal_mask[:, 3, 2] = 1.0  # Position (3,2) is legal
    legal_mask[:, 4, 5] = 1.0  # Position (4,5) is legal
    legal_mask[:, 5, 4] = 1.0  # Position (5,4) is legal

    # Apply masking logic (same as in MCTS wrapper)
    legal_mask_flat = legal_mask.reshape(-1, 64)
    illegal_mask = legal_mask_flat == 0.0
    policy_logits_flat = policy_logits.reshape(-1, 64)
    policy_logits_flat.masked_fill_(illegal_mask, float("-inf"))
    policy_probs = torch.softmax(policy_logits_flat, dim=-1)

    # Check no NaN
    assert not torch.isnan(policy_probs).any(), "Policy should not contain NaN"
    # Check probabilities sum to 1
    assert torch.allclose(
        policy_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5
    ), "Policy should sum to 1"
    # Check only legal positions have non-zero probability
    assert (policy_probs * illegal_mask.float()).sum() == 0, (
        "Illegal moves should have zero probability"
    )


def test_policy_no_nan_with_no_legal_moves():
    """Test that policy does not contain NaN when all moves are illegal (edge case)."""
    import torch

    batch_size = 2
    # Create policy logits
    policy_logits = torch.randn(batch_size, 8, 8)

    # All moves are illegal (terminal state)
    legal_mask = torch.zeros(batch_size, 8, 8)

    # Apply masking logic
    legal_mask_flat = legal_mask.reshape(-1, 64)
    illegal_mask = legal_mask_flat == 0.0
    policy_logits_flat = policy_logits.reshape(-1, 64)
    policy_logits_flat.masked_fill_(illegal_mask, float("-inf"))
    policy_probs = torch.softmax(policy_logits_flat, dim=-1)

    # After softmax of all -inf, we get NaN - apply the fallback
    nan_mask = torch.isnan(policy_probs)
    if nan_mask.any():
        policy_probs.masked_fill_(nan_mask, 1.0 / 64.0)

    # Check no NaN after fallback
    assert not torch.isnan(policy_probs).any(), (
        "Policy should not contain NaN after fallback"
    )
    # Check probabilities are valid (sum to ~1)
    assert torch.allclose(
        policy_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5
    ), "Policy should sum to 1"


def test_policy_probabilities_only_on_legal_moves():
    """Test that probability mass is only on legal moves."""
    import torch

    batch_size = 3
    policy_logits = torch.randn(batch_size, 8, 8)

    # Create different legal masks for each batch item
    legal_mask = torch.zeros(batch_size, 8, 8)
    legal_mask[0, 0, 0] = 1.0  # Only one legal move for batch 0
    legal_mask[1, 3, 3] = 1.0  # Only one legal move for batch 1
    legal_mask[1, 4, 4] = 1.0  # Two legal moves for batch 1
    legal_mask[2, :, :] = 0.0  # No legal moves for batch 2 (terminal)

    # Apply masking
    legal_mask_flat = legal_mask.reshape(-1, 64)
    illegal_mask = legal_mask_flat == 0.0
    policy_logits_flat = policy_logits.reshape(-1, 64)
    policy_logits_flat.masked_fill_(illegal_mask, float("-inf"))
    policy_probs = torch.softmax(policy_logits_flat, dim=-1)

    # Handle NaN for terminal state
    nan_mask = torch.isnan(policy_probs)
    if nan_mask.any():
        policy_probs.masked_fill_(nan_mask, 1.0 / 64.0)

    # Check batch 0: only position 0 should have probability 1.0
    assert torch.isclose(policy_probs[0, 0], torch.tensor(1.0), atol=1e-5), (
        "Single legal move should have probability 1.0"
    )

    # Check batch 1: positions 27 (3*8+3) and 36 (4*8+4) should have all probability
    assert policy_probs[1, 27] > 0, "Legal move at (3,3) should have positive prob"
    assert policy_probs[1, 36] > 0, "Legal move at (4,4) should have positive prob"
    assert torch.isclose(
        policy_probs[1, 27] + policy_probs[1, 36], torch.tensor(1.0), atol=1e-5
    ), "Probability should sum to 1 for legal moves only"

    # Check no NaN in any batch
    assert not torch.isnan(policy_probs).any(), "No NaN values should exist"
