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
    mcts_instance = mcts.MCTS()
    model = MockModel()

    # MCTS parameters
    num_simulations = 10
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 1.0
    c_puct = 1.0

    # Run MCTS
    actions = mcts_instance.run(
        initial_states,
        model.inference,
        num_simulations,
        dirichlet_epsilon,
        dirichlet_alpha,
        c_puct,
    )

    print(f"Best actions: {actions}")

    # Assertions
    assert len(actions) == initial_states.shape[0]
    for action in actions:
        assert action is None or (0 <= action < 64)
