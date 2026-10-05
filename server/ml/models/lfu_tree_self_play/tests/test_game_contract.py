"""Contract tests for the research-facing Reversi game interface."""

import numpy as np
import pytest

from lfu_tree_self_play.game import GameState, ReversiPyGame


def _position(rows: tuple[str, ...], turn: int, legal: tuple[int, ...]) -> GameState:
    """Create a fixed, hand-inspectable position for game-rule contracts."""
    planes = np.zeros((4, 8, 8), dtype=np.float32)
    for row, cells in enumerate(rows):
        for column, cell in enumerate(cells):
            if cell == "B":
                planes[0, row, column] = 1
            elif cell == "W":
                planes[1, row, column] = 1
    planes[2] = turn
    for action in legal:
        planes[3, action // 8, action % 8] = 1
    return GameState(planes)


def _rows(state: GameState) -> tuple[str, ...]:
    planes = state.observation
    return tuple(
        "".join(
            "B" if planes[0, row, column] else "W" if planes[1, row, column] else "."
            for column in range(8)
        )
        for row in range(8)
    )


def test_initial_state_has_black_to_move_and_four_known_actions() -> None:
    state = ReversiPyGame().initial_state()

    assert state.to_play == 1
    assert state.legal_actions == (19, 26, 37, 44)
    assert state.is_terminal is False
    assert state.black_result is None

    board = state.observation
    assert board.shape == (4, 8, 8)
    assert board[0, 3, 4] == board[0, 4, 3] == 1
    assert board[1, 3, 3] == board[1, 4, 4] == 1


def test_d3_placement_flips_one_disk_and_gives_white_three_actions() -> None:
    game = ReversiPyGame()
    start = game.initial_state()

    after = game.step(start, 19)

    assert after.to_play == -1
    assert after.legal_actions == (18, 20, 34)
    board = after.observation
    assert board[0].sum() == 4
    assert board[1].sum() == 1
    assert board[0, 2, 3] == board[0, 3, 3] == 1
    assert board[1, 3, 3] == 0
    assert start.legal_actions == (19, 26, 37, 44)
    assert start.observation[1, 3, 3] == 1


@pytest.mark.parametrize("action", [-1, 27, 64])
def test_illegal_action_is_rejected_without_changing_state(action: int) -> None:
    game = ReversiPyGame()
    state = game.initial_state()

    with pytest.raises(ValueError, match="legal action"):
        game.step(state, action)

    assert state.legal_actions == (19, 26, 37, 44)
    assert state.observation[1, 3, 3] == 1


def test_observation_is_a_defensive_copy() -> None:
    state = ReversiPyGame().initial_state()
    observation = state.observation

    observation[0, 3, 4] = 0

    assert state.observation[0, 3, 4] == 1


def test_state_copies_constructor_input() -> None:
    planes = ReversiPyGame().initial_state().observation
    state = GameState(planes)

    planes[0, 3, 4] = 0

    assert state.observation[0, 3, 4] == 1


def test_equivalent_positions_compare_and_hash_by_observation() -> None:
    game = ReversiPyGame()
    first = game.initial_state()
    equivalent = game.initial_state()
    moved = game.step(first, 19)

    assert first == equivalent
    assert hash(first) == hash(equivalent)
    assert first != moved
    assert len({first, equivalent, moved}) == 2


def test_malformed_observation_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        GameState(np.zeros((3, 8, 8), dtype=np.float32))


def test_non_integer_action_is_rejected_consistently() -> None:
    game = ReversiPyGame()

    with pytest.raises(ValueError, match="legal action"):
        game.step(game.initial_state(), 19.0)  # type: ignore[arg-type]


def test_forced_pass_keeps_white_to_move_after_a_white_placement() -> None:
    before = _position(
        (
            "..BBBBWB",
            "BBBBB.WB",
            "BWBWBBWB",
            "BWBBWBWB",
            "BWWWBWWB",
            "BWBWWBWB",
            "BWWWWWBB",
            "BWWBBBBB",
        ),
        turn=-1,
        legal=(1, 13),
    )

    after = ReversiPyGame().step(before, 13)

    assert after.to_play == -1
    assert after.legal_actions == (0, 1)
    assert after.is_terminal is False
    assert _rows(after) == (
        "..BBBBWB",
        "BBBBBWWB",
        "BWBWWWWB",
        "BWBWWWWB",
        "BWWWBWWB",
        "BWBWWBWB",
        "BWWWWWBB",
        "BWWBBBBB",
    )


@pytest.mark.parametrize(
    ("rows", "turn", "expected"),
    [
        ((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",), 1, 1),
        ((".BWWWWWW",) + ("WWWWWWWW",) * 6 + ("WWWWWWW.",), -1, -1),
        ((".WBBBBBB",) + ("BBBBBBBB",) * 3 + ("WWWWWWWW",) * 4, 1, 0),
    ],
)
def test_terminal_result_handles_black_win_white_win_and_draw(
    rows: tuple[str, ...], turn: int, expected: int
) -> None:
    before = _position(rows, turn=turn, legal=(0,))

    after = ReversiPyGame().step(before, 0)

    assert after.is_terminal is True
    assert after.legal_actions == ()
    assert after.black_result == expected


def test_consecutive_no_move_termination_preserves_empty_square() -> None:
    before = _position(
        (".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",),
        turn=1,
        legal=(0,),
    )

    after = ReversiPyGame().step(before, 0)

    assert _rows(after)[7] == "BBBBBBB."
    assert after.is_terminal is True
    assert after.black_result == 1


def test_swapping_colors_and_player_preserves_geometry_and_reverses_result() -> None:
    game = ReversiPyGame()
    first = game.initial_state()
    swapped_planes = first.observation
    swapped_planes[[0, 1]] = swapped_planes[[1, 0]]
    swapped_planes[2] *= -1
    swapped = GameState(swapped_planes)

    black_after = game.step(first, 19)
    white_after = game.step(swapped, 19)

    np.testing.assert_array_equal(
        black_after.observation[0], white_after.observation[1]
    )
    np.testing.assert_array_equal(
        black_after.observation[1], white_after.observation[0]
    )
    assert black_after.legal_actions == white_after.legal_actions
    assert black_after.to_play == -white_after.to_play

    black_finish = _position(
        (".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",),
        turn=1,
        legal=(0,),
    )
    white_finish = _position(
        (".BWWWWWW",) + ("WWWWWWWW",) * 6 + ("WWWWWWW.",),
        turn=-1,
        legal=(0,),
    )
    assert game.step(black_finish, 0).black_result == 1
    assert game.step(white_finish, 0).black_result == -1
