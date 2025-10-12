"""
Reversi environment for reinforcement learning.

This package provides a Python interface to the Reversi game,
optimized for reinforcement learning with batch processing support.
"""

import numpy as np
from ._core import ReversiEnvironment as _ReversiEnvironment
from .exceptions import InvalidMoveError, ReversiError
from .types import BoardState, NDArray

__version__ = "0.1.0"
__all__ = ["ReversiEnvironment", "BoardState", "ReversiError", "InvalidMoveError"]


class ReversiEnvironment:
    """Reversi environment for reinforcement learning.

    This class provides a Python interface to the Reversi game,
    optimized for reinforcement learning with batch processing support.

    Example:
        >>> env = ReversiEnvironment(batch_size=32)
        >>> state = env.reset()
        >>> # state.shape == (32, 4, 8, 8)
        >>> actions = np.random.random((32, 8, 8))
        >>> next_state, dones = env.step_batch(actions)
    """

    def __init__(self, batch_size: int):
        """Initialize the environment.

        Args:
            batch_size: Number of parallel games to manage.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._env = _ReversiEnvironment(batch_size)

    def reset(self) -> BoardState:
        """Reset all games to their initial state.

        Returns:
            Initial game states with shape (batch_size, 4, 8, 8).
        """
        return self._env.reset()

    def get_state(self) -> BoardState:
        """Get the current state of all games.

        Returns:
            Current game states with shape (batch_size, 4, 8, 8).
        """
        return self._env.get_state()

    def step_batch(
        self,
        actions: NDArray,  # Shape: (batch, 8, 8) or (8, 8) if batch_size=1
    ) -> tuple[BoardState, NDArray]:
        """Apply actions to all games in the batch.

        Args:
            actions: Array of action probabilities with shape (batch, 8, 8).
                     For batch_size=1, an (8, 8) array is also accepted.
                     Only the relative probabilities of valid moves matter.

        Returns:
            Tuple of (next_states, dones):
            - next_states: Array with shape (batch, 4, 8, 8)
            - dones: Boolean array with shape (batch,)

        Raises:
            ValueError: If actions have incorrect shape.
            ReversiError: For other game-related errors.
        """
        # For batch_size=1, allow (8, 8) actions by adding a batch dimension.
        if self.batch_size == 1 and actions.shape == (8, 8):
            actions = np.expand_dims(actions, axis=0)

        if actions.shape != (self.batch_size, 8, 8):
            raise ValueError(
                f"Expected actions with shape {(self.batch_size, 8, 8)}, "
                f"got {actions.shape}"
            )

        try:
            return self._env.step_batch(actions)
        except Exception as e:
            raise ReversiError(str(e)) from e

    @property
    def batch_size(self) -> int:
        """Get the number of parallel games."""
        return self._env.batch_size
