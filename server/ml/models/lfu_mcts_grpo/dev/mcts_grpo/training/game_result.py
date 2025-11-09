"""
ゲーム結果の処理
勝者判定とoutcome計算
"""

from enum import Enum

import numpy as np

from ..replay_buffer import Experience, ReplayBuffer
from ..settings import Settings


class Winner(Enum):
    """ゲーム結果の判定"""

    DRAW = 0
    DARK = 1
    LIGHT = -1


class GameResultProcessor:
    """ゲーム結果の処理とリプレイバッファへの追加"""

    def __init__(self, settings: Settings, replay_buffer: ReplayBuffer):
        self.settings = settings
        self.replay_buffer = replay_buffer

    def process_game_history(
        self, game_history: list[dict[str, np.ndarray]], winner: int
    ):
        """ゲーム履歴を逆順で処理し、各局面のoutcomeを計算"""
        for experience_data in reversed(game_history):
            outcome = self._compute_outcome(experience_data["state"], winner)
            self.replay_buffer.push(
                Experience(
                    state=experience_data["state"],
                    pi=experience_data["pi"],
                    outcome=outcome,
                    q_values=experience_data["q_values"],
                )
            )

    def _compute_outcome(self, state: np.ndarray, winner: int) -> float:
        """局面の手番プレイヤーから見たoutcomeを計算"""
        is_dark_turn = np.all(state[2] > 0)

        if winner == Winner.DRAW:
            return 0.0

        if (winner == Winner.DARK and is_dark_turn) or (
            winner == Winner.LIGHT and not is_dark_turn
        ):
            return 1.0
        else:
            return -1.0

    @staticmethod
    def determine_winner(terminal_state: np.ndarray) -> int:
        """終局盤面から勝者を判定"""
        num_dark = np.sum(terminal_state[0])
        num_light = np.sum(terminal_state[1])

        if num_dark > num_light:
            return Winner.DARK
        elif num_light > num_dark:
            return Winner.LIGHT
        else:
            return Winner.DRAW
