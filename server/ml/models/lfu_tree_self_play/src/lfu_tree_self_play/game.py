"""The research-facing game boundary for Reversi."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from reversi import ReversiEnvironment


@dataclass(frozen=True, eq=False)
class GameState:
    """One position in the four-plane Reversi observation format."""

    _planes: NDArray[np.float32]

    def __post_init__(self) -> None:
        planes = np.array(self._planes, dtype=np.float32, copy=True)
        if planes.shape != (4, 8, 8):
            raise ValueError("observation must have shape (4, 8, 8)")
        planes.setflags(write=False)
        object.__setattr__(self, "_planes", planes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GameState) and bool(
            np.array_equal(self._planes, other._planes)
        )

    def __hash__(self) -> int:
        return hash(self._planes.tobytes())

    @property
    def observation(self) -> NDArray[np.float32]:
        """Return a copy of the black, white, turn, and legal-move planes."""
        return self._planes.copy()

    @property
    def to_play(self) -> int:
        """Return +1 for black and -1 for white."""
        return int(self._planes[2, 0, 0])

    @property
    def legal_actions(self) -> tuple[int, ...]:
        """Return legal placement squares in row-major order."""
        return tuple(int(index) for index in np.flatnonzero(self._planes[3]))

    @property
    def is_terminal(self) -> bool:
        """Whether neither player can move after forced-pass normalization."""
        return not self.legal_actions

    @property
    def black_result(self) -> int | None:
        """Return the terminal outcome from black's perspective."""
        if not self.is_terminal:
            return None
        black = int(np.count_nonzero(self._planes[0]))
        white = int(np.count_nonzero(self._planes[1]))
        return (black > white) - (black < white)


class Game(Protocol):
    """The game operations used by research code, independent of a backend."""

    def initial_state(self) -> GameState: ...

    def step(self, state: GameState, action: int) -> GameState: ...


class ReversiPyGame:
    """Use the shared Rust rules through the reversi-py binding."""

    def initial_state(self) -> GameState:
        """Return the standard initial position."""
        return GameState(ReversiEnvironment(batch_size=1).reset()[0])

    def step(self, state: GameState, action: int) -> GameState:
        """Place a disk at one legal square and resolve any forced pass."""
        if (
            isinstance(action, bool)
            or not isinstance(action, (int, np.integer))
            or action not in state.legal_actions
        ):
            raise ValueError(f"{action} is not a legal action")
        next_observation = ReversiEnvironment.get_next_state(
            state.observation, int(action)
        )
        return GameState(next_observation)
