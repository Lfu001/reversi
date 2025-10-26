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

    def __init__(self, batch_size: int, seed: int = 0):
        """Initialize the environment.

        Args:
            batch_size: Number of parallel games to manage.
            seed: Seed value used to determine actions stochastically.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._env = _ReversiEnvironment(batch_size, seed)

    def reset(self) -> BoardState:
        """Reset all games to their initial state.

        Returns:
            Initial game states with shape (batch_size, 4, 8, 8).
        """
        return self._env.reset()

    def reset_indices(self, indices: list[int]) -> BoardState:
        """Reset games specified by `indices` to their initial state.

        Args:
            indices: A list of indices of the games to reset.

        Returns:
            Initial game states with shape (batch_size, 4, 8, 8).
        """
        return self._env.reset_indices(indices)

    def get_state(self) -> BoardState:
        """Get the current state of all games.

        Returns:
            Current game states with shape (batch_size, 4, 8, 8).
        """
        return self._env.get_state()

    @staticmethod
    def get_next_state(state: BoardState, action: int) -> BoardState:
        """Compute the next state from the given state and action.

        This is a static method that works with a single game state.

        Args:
            state: Current game state with shape (4, 8, 8).
            action: Action index (0-63) representing the position to place the disk.

        Returns:
            Next game state with shape (4, 8, 8) after applying the action.
        """
        return _ReversiEnvironment.get_next_state(state, action)

    def step_batch(
        self,
        actions: NDArray,
        deterministic: bool,
    ) -> tuple[BoardState, NDArray]:
        """
        Apply actions to all games in the batch and advance them by one step.

        Args:
            actions: An array of action probabilities.
                     Shape should be (batch_size, 8, 8). For a batch_size of 1,
                     an (8, 8) array is also accepted.
            deterministic: If True, the action with the highest probability is
                           always chosen (greedy policy, for evaluation/inference).
                           If False, an action is sampled from the probability
                           distribution (stochastic policy, for training/exploration).

        Returns:
            A tuple of (next_states, dones):
            - next_states: An array of shape (batch_size, 4, 8, 8) representing the
                           new board states.
            - dones: A boolean array of shape (batch_size,) indicating which games
                     have finished.

        Raises:
            ValueError: If the actions array has an incorrect shape.
            ReversiError: For internal game logic errors.
        """
        # Ensure actions are of the correct float32 dtype for the Rust extension.
        if actions.dtype != np.float32:
            actions = actions.astype(np.float32)

        # For batch_size=1, allow (8, 8) actions by adding a batch dimension.
        if self.batch_size == 1 and actions.shape == (8, 8):
            actions = np.expand_dims(actions, axis=0)

        if actions.shape != (self.batch_size, 8, 8):
            raise ValueError(
                f"Expected actions with shape {(self.batch_size, 8, 8)}, "
                f"got {actions.shape}"
            )

        try:
            # Choose the appropriate Rust method based on the deterministic flag.
            if deterministic:
                return self._env.step_batch_deterministic(actions)
            else:
                return self._env.step_batch_stochastic(actions)
        except Exception as e:
            # Wrap Rust errors in a custom Python exception.
            raise ReversiError(f"An error occurred in the Rust core: {e}") from e

    @property
    def batch_size(self) -> int:
        """Get the number of parallel games."""
        return self._env.batch_size
