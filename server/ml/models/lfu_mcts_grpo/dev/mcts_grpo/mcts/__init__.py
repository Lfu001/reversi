"""
MCTS (Monte Carlo Tree Search) 実装
"""

import numpy as np
import torch

from ..settings import MCTSConfig
from .evaluator import NetworkEvaluator
from .simulator import MCTSSimulator
from .tree import MCTSTree


class MCTS:
    """MCTSの統合インターフェース"""

    def __init__(self, mcts_config: MCTSConfig):
        self.mcts_config = mcts_config
        self.tree = MCTSTree(mcts_config)

    def run_simulations(
        self, model: torch.nn.Module, root_states: np.ndarray, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        """MCTSシミュレーションを実行

        Args:
            model: ニューラルネットワークモデル
            root_states: ルート状態のバッチ (batch_size, channels, height, width)
            device: 使用するデバイス

        Returns:
            tuple[np.ndarray, np.ndarray]:
            - pi: 改善された方策 (batch_size, 8, 8)
            - q_values: Q値 (batch_size, 8, 8)
        """
        evaluator = NetworkEvaluator(model, device)
        simulator = MCTSSimulator(self.tree)

        return simulator.run_simulation(
            root_states, evaluator, self.mcts_config.num_simulations
        )
