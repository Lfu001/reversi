import mcts
import numpy as np
import pytest


class MockModel:
    def inference(self, states):
        # states: [B, 4, 8, 8]
        batch_size = states.shape[0]
        # Return random policy and value
        policy = np.random.dirichlet(np.ones(64), size=batch_size).astype(np.float64)
        value = np.random.uniform(-1, 1, size=(batch_size, 1)).astype(np.float64)
        return policy, value


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
    mcts_instance = mcts.RustMCTS(max_inference_batch_size, states_per_inference)
    model = MockModel()

    # MCTS parameters
    num_simulations = 10
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 1.0
    c_puct = 1.0

    # Run MCTS - now returns (pi, q_values) where pi is already normalized
    pi, q_values = mcts_instance.run(
        initial_states,
        model.inference,
        num_simulations,
        dirichlet_epsilon,
        dirichlet_alpha,
        c_puct,
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
