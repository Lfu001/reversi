"""
Tests for GameResultProcessor
Verifies winner determination and outcome calculation.
"""

import numpy as np
import pytest
from mcts_grpo.training.game_result import GameResultProcessor, Winner


class TestWinnerDetermination:
    """Tests for winner determination from terminal state."""

    def test_dark_wins(self):
        """Dark player wins when they have more disks."""
        terminal_state = np.zeros((4, 8, 8), dtype=np.float32)
        # Dark has 10 disks
        terminal_state[0, 0, :] = 1.0
        terminal_state[0, 1, :2] = 1.0
        # Light has 5 disks
        terminal_state[1, 2, :5] = 1.0

        winner = GameResultProcessor.determine_winner(terminal_state)
        assert winner == Winner.DARK

    def test_light_wins(self):
        """Light player wins when they have more disks."""
        terminal_state = np.zeros((4, 8, 8), dtype=np.float32)
        # Dark has 3 disks
        terminal_state[0, 0, :3] = 1.0
        # Light has 10 disks
        terminal_state[1, 1, :] = 1.0
        terminal_state[1, 2, :2] = 1.0

        winner = GameResultProcessor.determine_winner(terminal_state)
        assert winner == Winner.LIGHT

    def test_draw(self):
        """Draw when both players have equal disks."""
        terminal_state = np.zeros((4, 8, 8), dtype=np.float32)
        # Both have 5 disks
        terminal_state[0, 0, :5] = 1.0
        terminal_state[1, 1, :5] = 1.0

        winner = GameResultProcessor.determine_winner(terminal_state)
        assert winner == Winner.DRAW


class TestOutcomeComputation:
    """Tests for outcome calculation from current player perspective."""

    @pytest.fixture
    def processor(self):
        """Create a processor with mock settings and buffer."""
        from unittest.mock import MagicMock

        settings = MagicMock()
        replay_buffer = MagicMock()
        return GameResultProcessor(settings, replay_buffer)

    def test_dark_turn_dark_wins(self, processor):
        """Outcome is 1.0 when current player (dark) wins."""
        state = np.zeros((4, 8, 8), dtype=np.float32)
        state[2, :, :] = 1.0  # Dark's turn

        outcome = processor._compute_outcome(state, Winner.DARK)
        assert outcome == 1.0

    def test_dark_turn_light_wins(self, processor):
        """Outcome is -1.0 when current player (dark) loses."""
        state = np.zeros((4, 8, 8), dtype=np.float32)
        state[2, :, :] = 1.0  # Dark's turn

        outcome = processor._compute_outcome(state, Winner.LIGHT)
        assert outcome == -1.0

    def test_light_turn_light_wins(self, processor):
        """Outcome is 1.0 when current player (light) wins."""
        state = np.zeros((4, 8, 8), dtype=np.float32)
        state[2, :, :] = 0.0  # Light's turn

        outcome = processor._compute_outcome(state, Winner.LIGHT)
        assert outcome == 1.0

    def test_light_turn_dark_wins(self, processor):
        """Outcome is -1.0 when current player (light) loses."""
        state = np.zeros((4, 8, 8), dtype=np.float32)
        state[2, :, :] = 0.0  # Light's turn

        outcome = processor._compute_outcome(state, Winner.DARK)
        assert outcome == -1.0

    def test_draw_outcome(self, processor):
        """Outcome is 0.0 for a draw regardless of current player."""
        dark_turn_state = np.zeros((4, 8, 8), dtype=np.float32)
        dark_turn_state[2, :, :] = 1.0

        light_turn_state = np.zeros((4, 8, 8), dtype=np.float32)
        light_turn_state[2, :, :] = 0.0

        assert processor._compute_outcome(dark_turn_state, Winner.DRAW) == 0.0
        assert processor._compute_outcome(light_turn_state, Winner.DRAW) == 0.0
