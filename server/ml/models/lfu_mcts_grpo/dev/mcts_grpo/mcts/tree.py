"""
MCTS木の状態管理
Q値、訪問回数、方策を管理
"""

import numpy as np

from ..settings import MCTSConfig


class MCTSTree:
    """MCTS木の状態を管理"""

    def __init__(self, mcts_config: MCTSConfig):
        self.mcts_config = mcts_config
        # Q(s,a): state s で action a を取ったときの平均行動価値
        self.Q: dict[bytes, np.ndarray] = {}
        # N(s,a): state s で action a を選択した回数
        self.N: dict[bytes, np.ndarray] = {}
        # P(s): state s における方策 (NNからの出力)
        self.P: dict[bytes, np.ndarray] = {}
        # V(s): state s の訪問回数 (デバッグ用)
        self.visit_counts: dict[bytes, int] = {}

    def is_node_expanded(self, state_key: bytes) -> bool:
        """ノードが展開済みか判定"""
        return state_key in self.P

    def expand_node(self, state_key: bytes, policy_probs: np.ndarray):
        """ノードを展開"""
        self.P[state_key] = policy_probs
        self.Q[state_key] = np.zeros(64, dtype=np.float32)
        self.N[state_key] = np.zeros(64, dtype=np.float32)
        self.visit_counts[state_key] = 0

    def add_dirichlet_noise(self, state_key: bytes):
        """ルートノードにディリクレノイズを追加"""
        noise = np.random.dirichlet([self.mcts_config.dirichlet_alpha] * 64)
        self.P[state_key] = (
            self.P[state_key] * (1 - self.mcts_config.dirichlet_epsilon)
            + noise * self.mcts_config.dirichlet_epsilon
        )

    def compute_puct_scores(self, state_key: bytes) -> np.ndarray:
        """PUCT スコアを計算"""
        visit_sum = np.sqrt(np.sum(self.N[state_key]))
        return self.Q[state_key] + self.mcts_config.c_puct * self.P[
            state_key
        ] * visit_sum / (1 + self.N[state_key])

    def backup(self, state_key: bytes, action: int, value: float):
        """Q値と訪問回数を更新"""
        self.Q[state_key][action] = (
            self.N[state_key][action] * self.Q[state_key][action] + value
        ) / (self.N[state_key][action] + 1)
        self.N[state_key][action] += 1
        self.visit_counts[state_key] += 1

    def get_policy_and_q_values(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ルートノードの方策とQ値を取得"""
        state_key = state.tobytes()
        visit_counts = self.N[state_key]
        policy = (
            visit_counts / np.sum(visit_counts)
            if np.sum(visit_counts) > 0
            else visit_counts
        )

        q_value = self.Q[state_key]
        legal_moves = state[3].flatten()
        unexplored_legal_mask = (visit_counts == 0) & (legal_moves == 1)
        policy[unexplored_legal_mask] = -100.0

        return policy, q_value
