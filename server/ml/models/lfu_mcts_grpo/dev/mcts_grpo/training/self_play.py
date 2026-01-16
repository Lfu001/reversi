"""
Self-Play実行
ゲーム生成とリプレイバッファへのデータ追加
"""

from collections.abc import MutableSequence

import numpy as np
import torch
from mcts import MCTS
from reversi import ReversiEnvironment

from ..replay_buffer import ReplayBuffer
from ..settings import Settings
from .game_result import GameResultProcessor

GameHistory = list[dict[str, np.typing.NDArray[np.float32]]]
OngoingGamesData = MutableSequence[GameHistory]


class SelfPlayExecutor:
    """Self-Play（ゲーム生成）の実行"""

    def __init__(
        self,
        settings: Settings,
        env: ReversiEnvironment,
        replay_buffer: ReplayBuffer,
    ):
        self.settings = settings
        self.env = env
        self.replay_buffer = replay_buffer

    def run_self_play(
        self, iteration: int, model: torch.nn.Module, device: torch.device
    ):
        """Self-Playを実行してリプレイバッファにデータを追加"""
        mcts = MCTS(
            max_inference_batch_size=self.settings.mcts.max_inference_batch_size,
            states_per_inference=self.settings.mcts.states_per_inference,
        )
        model.eval()

        games_completed = 0
        ongoing_games_data: OngoingGamesData = [
            [] for _ in range(self.settings.mcts.parallel_games)
        ]

        current_states = self.env.reset()
        while games_completed < self.settings.mcts.games_per_iteration:
            pi, q_values, visit_counts = mcts.run_simulations(
                model=model,
                states=current_states,
                device=device,
                num_simulations=self.settings.mcts.num_simulations,
                dirichlet_epsilon=self.settings.mcts.dirichlet_epsilon,
                dirichlet_alpha=self.settings.mcts.dirichlet_alpha,
                c_puct=self.settings.mcts.c_puct,
            )
            pi_reshaped = pi.reshape(-1, 8, 8)
            next_states, dones = self.env.step_batch(pi_reshaped, deterministic=False)

            for i in range(self.settings.mcts.parallel_games):
                self._record_step(
                    ongoing_games_data,
                    i,
                    current_states[i],
                    pi[i],
                    q_values[i],
                    visit_counts[i],
                )

                if dones[i]:
                    games_completed += 1
                    self._process_game_end(ongoing_games_data, i, next_states[i])

            current_states = self.env.reset_indices(np.where(dones)[0])

    def _record_step(
        self,
        ongoing_games_data: OngoingGamesData,
        game_idx: int,
        state: np.ndarray,
        pi: np.ndarray,
        q_values: np.ndarray,
        visit_counts: np.ndarray,
    ) -> None:
        """ゲームの1ステップを記録（ongoing_games_dataを変更）"""
        ongoing_games_data[game_idx].append(
            {
                "state": state,
                "pi": pi,
                "q_values": q_values,
                "visit_counts": visit_counts,
            }
        )

    def _process_game_end(
        self,
        ongoing_games_data: OngoingGamesData,
        game_idx: int,
        terminal_state: np.ndarray,
    ) -> None:
        """ゲーム終了時の処理（ongoing_games_dataを変更）"""
        winner = GameResultProcessor.determine_winner(terminal_state)
        processor = GameResultProcessor(self.settings, self.replay_buffer)
        processor.process_game_history(ongoing_games_data[game_idx], winner)
        ongoing_games_data[game_idx] = []
