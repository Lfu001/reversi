import numpy as np
import pytest

from reversi import ReversiEnvironment


@pytest.fixture
def single_env():
    """A fixture to provide a clean environment with batch_size=1 for each test."""
    return ReversiEnvironment(batch_size=1)


@pytest.fixture
def batch_env():
    """A fixture to provide a clean environment with batch_size=2 for each test."""
    return ReversiEnvironment(batch_size=2)


def test_initialization():
    """Tests that the environment is initialized with the correct batch size."""
    env = ReversiEnvironment(batch_size=4)
    assert env.batch_size == 4


def test_reset_return_spec(single_env: ReversiEnvironment):
    """Tests that reset() returns a state with the correct shape and dtype."""
    # Act
    state = single_env.reset()

    # Assert - Focus on I/O specification
    assert isinstance(state, np.ndarray)
    assert state.shape == (1, 4, 8, 8)
    assert state.dtype == np.float32


def test_initial_board_state(single_env: ReversiEnvironment):
    """Tests the board configuration of the initial state."""
    state = single_env.reset()

    my_pieces = state[0, 0]
    opponent_pieces = state[0, 1]
    turn_plane = state[0, 2]

    # Assert initial pieces
    expected_my = np.zeros((8, 8), dtype=np.float32)
    expected_opponent = np.zeros((8, 8), dtype=np.float32)
    expected_my[3, 4] = 1
    expected_my[4, 3] = 1
    expected_opponent[3, 3] = 1
    expected_opponent[4, 4] = 1
    np.testing.assert_array_equal(my_pieces, expected_my)
    np.testing.assert_array_equal(opponent_pieces, expected_opponent)

    # Assert it's Dark's turn (plane of 1s)
    np.testing.assert_array_equal(turn_plane, np.ones((8, 8), dtype=np.float32))


def test_initial_legal_moves(single_env: ReversiEnvironment):
    """Tests the legal moves in the initial state."""
    state = single_env.reset()
    legal_moves_plane = state[0, 3]

    expected_legal_moves = np.zeros((8, 8), dtype=np.float32)
    expected_legal_moves[2, 3] = 1  # D3
    expected_legal_moves[3, 2] = 1  # C4
    expected_legal_moves[4, 5] = 1  # F5
    expected_legal_moves[5, 4] = 1  # E6

    np.testing.assert_array_equal(legal_moves_plane, expected_legal_moves)


def test_step_return_spec(single_env: ReversiEnvironment):
    """Tests that step_batch() returns values with the correct shape and dtype."""
    single_env.reset()
    action = np.zeros((8, 8), dtype=np.float32)
    action[2, 3] = 1.0  # Play a valid move D3

    # Act: Use deterministic=True for predictable outcomes
    next_state, done = single_env.step_batch(action, deterministic=True)

    # Assert
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (1, 4, 8, 8)
    assert next_state.dtype == np.float32

    assert isinstance(done, np.ndarray)
    assert done.shape == (1,)
    assert done.dtype == np.bool_  # Check for boolean dtype


def test_step_deterministic_board_mutation(single_env: ReversiEnvironment):
    """Tests that the board state is correctly mutated after a deterministic step."""
    single_env.reset()
    action = np.zeros((8, 8), dtype=np.float32)
    action[2, 3] = 1.0  # Dark plays at D3

    # Act
    next_state, done = single_env.step_batch(action, deterministic=True)

    # Assert
    # It's now Light's turn. The state is from Light's perspective.
    light_pieces = next_state[0, 0]
    dark_pieces = next_state[0, 1]
    turn_plane = next_state[0, 2]

    # Assert turn changed to Light (plane of 0s)
    np.testing.assert_array_equal(turn_plane, np.zeros((8, 8), dtype=np.float32))
    assert not done[0]

    # Check dark pieces: original 2, 1 new, 1 flipped
    expected_dark = np.zeros((8, 8), dtype=np.float32)
    expected_dark[2, 3] = 1  # New piece at D3
    expected_dark[3, 3] = 1  # Flipped piece at D4
    expected_dark[3, 4] = 1  # Original piece at E4
    expected_dark[4, 3] = 1  # Original piece at D5
    np.testing.assert_array_equal(dark_pieces, expected_dark)

    # Check light pieces: one was flipped
    expected_light = np.zeros((8, 8), dtype=np.float32)
    expected_light[4, 4] = 1  # Original piece at E5
    np.testing.assert_array_equal(light_pieces, expected_light)


def test_step_stochastic_selects_legal_move(single_env: ReversiEnvironment):
    """
    Tests that a stochastic step selects one of the provided legal moves.
    """
    # Create an action where two legal moves have non-zero probability
    action = np.zeros((8, 8), dtype=np.float32)
    action[2, 3] = 0.5  # D3
    action[3, 2] = 0.5  # C4

    # Act: Use deterministic=False for stochastic sampling
    next_state, _ = single_env.step_batch(action, deterministic=False)

    # Assert
    # The new piece must be at one of the two locations we gave probability to.
    # The state is from Light's perspective, so we check the opponent (Dark) plane.
    dark_pieces = next_state[0, 1]

    # Check if a piece was placed at D3 (resulting board)
    board_if_d3 = np.zeros((8, 8), dtype=np.float32)
    board_if_d3[2, 3] = 1
    board_if_d3[3, 3] = 1
    board_if_d3[3, 4] = 1
    board_if_d3[4, 3] = 1

    # Check if a piece was placed at C4 (resulting board)
    board_if_c4 = np.zeros((8, 8), dtype=np.float32)
    board_if_c4[3, 2] = 1
    board_if_c4[3, 3] = 1
    board_if_c4[4, 3] = 1
    board_if_c4[3, 4] = 1

    d3_was_played = np.array_equal(dark_pieces, board_if_d3)
    c4_was_played = np.array_equal(dark_pieces, board_if_c4)

    assert d3_was_played or c4_was_played, (
        "The move played was not one of the provided options"
    )


def test_batch_step_independent(batch_env: ReversiEnvironment):
    """Tests that games in a batch are processed independently."""
    batch_env.reset()

    # Action for the first game: play at D3 (2,3)
    action1 = np.zeros((8, 8), dtype=np.float32)
    action1[2, 3] = 1.0

    # Action for the second game: play at C4 (3,2)
    action2 = np.zeros((8, 8), dtype=np.float32)
    action2[3, 2] = 1.0

    actions = np.stack([action1, action2])  # Shape: (2, 8, 8)

    # Act
    next_states, _ = batch_env.step_batch(actions, deterministic=True)

    # Assert state of the first game (played at D3)
    dark_pieces_game1 = next_states[0, 1]
    expected_dark_game1 = np.zeros((8, 8), dtype=np.float32)
    expected_dark_game1[2, 3] = 1
    expected_dark_game1[3, 3] = 1
    expected_dark_game1[3, 4] = 1
    expected_dark_game1[4, 3] = 1
    np.testing.assert_array_equal(dark_pieces_game1, expected_dark_game1)

    # Assert state of the second game (played at C4)
    dark_pieces_game2 = next_states[1, 1]
    expected_dark_game2 = np.zeros((8, 8), dtype=np.float32)
    expected_dark_game2[3, 2] = 1
    expected_dark_game2[4, 3] = 1
    expected_dark_game2[3, 3] = 1
    expected_dark_game2[3, 4] = 1
    np.testing.assert_array_equal(dark_pieces_game2, expected_dark_game2)


# --- Error Condition Tests ---


def test_init_invalid_batch_size():
    """Tests that ValueError is raised for invalid batch_size."""
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        ReversiEnvironment(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        ReversiEnvironment(batch_size=-1)


def test_step_invalid_action_shape(single_env: ReversiEnvironment):
    """Tests that ValueError is raised for actions with an incorrect shape."""
    single_env.reset()
    # Test with one incorrect dimension
    with pytest.raises(ValueError, match="Expected actions with shape"):
        invalid_action = np.zeros((1, 7, 8), dtype=np.float32)
        single_env.step_batch(invalid_action, deterministic=True)

    # Test with shape that would be correct for a different batch size
    with pytest.raises(ValueError, match="Expected actions with shape"):
        invalid_action_batch = np.zeros((2, 8, 8), dtype=np.float32)
        single_env.step_batch(invalid_action_batch, deterministic=True)

    # Test with incorrect 2D shape that gets expanded
    with pytest.raises(ValueError, match="Expected actions with shape"):
        invalid_action_2d = np.zeros((8, 7), dtype=np.float32)
        single_env.step_batch(invalid_action_2d, deterministic=True)
