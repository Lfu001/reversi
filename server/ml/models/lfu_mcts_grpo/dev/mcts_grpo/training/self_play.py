"""
Self-Play実行
ゲーム生成とリプレイバッファへのデータ追加
"""

import numpy as np
import torch
from mcts import MCTS
from reversi import ReversiEnvironment
from tqdm.rich import tqdm

from ..replay_buffer import ReplayBuffer
from ..settings import Settings
from .game_result import GameResultProcessor


class SelfPlayExecutor:
    """Self-Play（ゲーム生成）の実行"""

    def __init__(
        self,
        settings: Settings,
        env: ReversiEnvironment,
        replay_buffer: ReplayBuffer,
        ongoing_games_data: list[list[dict[str, np.ndarray]]],
    ):
        self.settings = settings
        self.env = env
        self.replay_buffer = replay_buffer
        self.ongoing_games_data = ongoing_games_data

    def run_self_play(
        self, iteration: int, model: torch.nn.Module, device: torch.device
    ):
        """Self-Playを実行してリプレイバッファにデータを追加"""
        # Initialize MCTS with config parameters
        mcts = MCTS(
            max_inference_batch_size=self.settings.mcts.max_inference_batch_size,
            states_per_inference=self.settings.mcts.states_per_inference,
        )
        model.eval()

        games_completed = 0
        pbar = tqdm(
            total=self.settings.training.games_per_iteration,
            desc=f"[Iter {iteration + 1}] Self-Play",
        )

        current_states = self.env.reset()
        while games_completed < self.settings.training.games_per_iteration:
            pi, q_values = mcts.run_simulations(
                model=model,
                states=current_states,
                device=device,
                num_simulations=self.settings.mcts.num_simulations,
                dirichlet_epsilon=self.settings.mcts.dirichlet_epsilon,
                dirichlet_alpha=self.settings.mcts.dirichlet_alpha,
                c_puct=self.settings.mcts.c_puct,
            )
            # Reshape pi from (batch, 64) to (batch, 8, 8) for env.step_batch
            pi_reshaped = pi.reshape(-1, 8, 8)
            next_states, dones = self.env.step_batch(pi_reshaped, deterministic=False)

            current_done_count = 0
            for i in range(self.settings.training.batch_size):
                self._record_step(i, current_states[i], pi[i], q_values[i])

                if dones[i]:
                    games_completed += 1
                    self._process_game_end(i, next_states[i])
                    current_done_count += 1
            pbar.update(current_done_count)
            current_states = self.env.reset_indices(np.where(dones)[0])

        pbar.close()

    def _record_step(
        self, game_idx: int, state: np.ndarray, pi: np.ndarray, q_values: np.ndarray
    ):
        """ゲームの1ステップを記録"""
        self.ongoing_games_data[game_idx].append(
            {
                "state": state,
                "pi": pi,
                "q_values": q_values,
            }
        )

    def _process_game_end(self, game_idx: int, terminal_state: np.ndarray):
        """ゲーム終了時の処理：勝者判定とリプレイバッファへの追加"""
        winner = GameResultProcessor.determine_winner(terminal_state)
        processor = GameResultProcessor(self.settings, self.replay_buffer)
        processor.process_game_history(self.ongoing_games_data[game_idx], winner)
        self.ongoing_games_data[game_idx] = []
