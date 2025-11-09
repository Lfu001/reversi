"""
MCTSシミュレーション実行ロジック
Selection、Expansion、Evaluation、Backupの各フェーズを実装
"""

import numpy as np
from reversi import ReversiEnvironment

from .evaluator import NetworkEvaluator
from .tree import MCTSTree


class MCTSSimulator:
    """MCTSシミュレーションを実行"""

    BOARD_SIZE = 8
    ACTION_SIZE = BOARD_SIZE * BOARD_SIZE

    def __init__(self, tree: MCTSTree):
        self.tree = tree

    def run_simulation(
        self,
        root_states: np.ndarray,
        evaluator: NetworkEvaluator,
        num_simulations: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """MCTSシミュレーションを実行

        Args:
            root_states: ルート状態のバッチ (batch_size, channels, height, width)
            evaluator: ニューラルネットワーク評価器
            num_simulations: シミュレーション回数

        Returns:
            tuple[np.ndarray, np.ndarray]:
            - pis: 改善された方策 (batch_size, 8, 8)
            - q_values: Q値 (batch_size, 8, 8)
        """
        batch_size = root_states.shape[0]

        # ルートノードの初期展開
        self._expand_root_nodes(root_states, evaluator)

        # ディリクレノイズを追加
        for state in root_states:
            state_key = state.tobytes()
            self.tree.add_dirichlet_noise(state_key)

        # シミュレーションループ
        for _ in range(num_simulations):
            self._run_single_simulation(root_states, evaluator, batch_size)

        # 最終的な方策とQ値を計算
        pis, q_values = self._compute_final_outputs(root_states, batch_size)

        return pis, q_values

    def _expand_root_nodes(self, root_states: np.ndarray, evaluator: NetworkEvaluator):
        """ルートノードを展開"""
        new_states_to_evaluate_mask = [
            state.tobytes() not in self.tree.P for state in root_states
        ]

        if any(new_states_to_evaluate_mask):
            states_to_eval = np.stack(
                [
                    s
                    for s, needs_eval in zip(root_states, new_states_to_evaluate_mask)
                    if needs_eval
                ]
            )

            policy_probs, _ = evaluator.evaluate_states(states_to_eval)

            eval_idx = 0
            for i, needs_eval in enumerate(new_states_to_evaluate_mask):
                if needs_eval:
                    state_key = root_states[i].tobytes()
                    self.tree.expand_node(state_key, policy_probs[eval_idx])
                    eval_idx += 1

    def _run_single_simulation(
        self, root_states: np.ndarray, evaluator: NetworkEvaluator, batch_size: int
    ):
        """1回のシミュレーションを実行"""
        current_states = root_states.copy()
        paths: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(batch_size)]
        is_terminal = np.zeros(batch_size, dtype=bool)

        # Selection
        for i in range(batch_size):
            state = current_states[i]
            while state.tobytes() in self.tree.P:
                if np.sum(state[3]) == 0:  # 合法手がなければ終端状態
                    is_terminal[i] = True
                    break

                puct_scores = self.tree.compute_puct_scores(state.tobytes())
                legal_moves = state[3].flatten()
                puct_scores[legal_moves == 0] = -np.inf
                action = np.argmax(puct_scores)

                paths[i].append((state, action))
                state = ReversiEnvironment.get_next_state(state, action)

            current_states[i] = state

        # Expansion & Evaluation
        leaf_states = current_states
        values_np = self._evaluate_leaf_states(leaf_states, is_terminal, evaluator)

        # Backup
        self._backup_values(paths, leaf_states, values_np, batch_size)

    def _evaluate_leaf_states(
        self,
        leaf_states: np.ndarray,
        is_terminal: np.ndarray,
        evaluator: NetworkEvaluator,
    ) -> np.ndarray:
        """リーフノードを評価"""
        batch_size = leaf_states.shape[0]
        values_np = np.zeros(batch_size, dtype=np.float32)

        needs_eval_mask = ~is_terminal
        if np.any(needs_eval_mask):
            states_to_eval = leaf_states[needs_eval_mask]
            policy_probs, network_values = evaluator.evaluate_states(states_to_eval)

            values_np[needs_eval_mask] = network_values

            eval_idx = 0
            for i in range(batch_size):
                if needs_eval_mask[i]:
                    state_key = leaf_states[i].tobytes()
                    self.tree.expand_node(state_key, policy_probs[eval_idx])
                    eval_idx += 1

        # 終端状態の価値を計算
        terminal_mask = is_terminal
        if np.any(terminal_mask):
            terminal_states = leaf_states[terminal_mask]
            for i, state in enumerate(terminal_states):
                outcome = self._compute_terminal_outcome(state)
                values_np[np.where(terminal_mask)[0][i]] = outcome

        return values_np

    def _compute_terminal_outcome(self, state: np.ndarray) -> float:
        """終端状態の価値を計算"""
        num_black = np.sum(state[0])
        num_white = np.sum(state[1])
        is_black_perspective = np.all(state[2] == 1)

        if num_black > num_white:
            return 1.0 if is_black_perspective else -1.0
        elif num_white > num_black:
            return -1.0 if is_black_perspective else 1.0
        else:
            return 0.0

    def _backup_values(
        self,
        paths: list[list[tuple[np.ndarray, int]]],
        leaf_states: np.ndarray,
        values_np: np.ndarray,
        batch_size: int,
    ):
        """価値をバックアップ"""
        for i in range(batch_size):
            value = values_np[i]
            child_state = leaf_states[i]
            child_player = self._get_player_from_state(child_state)

            for parent_state, action in reversed(paths[i]):
                parent_player = self._get_player_from_state(parent_state)

                if parent_player != child_player:
                    value *= -1

                state_key = parent_state.tobytes()
                self.tree.backup(state_key, action, value)

                child_state = parent_state
                child_player = parent_player

    def _compute_final_outputs(
        self, root_states: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """最終的な方策とQ値を計算"""
        pis = np.zeros((batch_size, self.ACTION_SIZE), dtype=np.float32)
        q_values = np.zeros((batch_size, self.ACTION_SIZE), dtype=np.float32)

        for i in range(batch_size):
            state = root_states[i]
            state_key = state.tobytes()
            policy, q_value = self.tree.get_policy_and_q_values(state_key)
            pis[i] = policy
            q_values[i] = q_value

        return (
            pis.reshape(batch_size, self.BOARD_SIZE, self.BOARD_SIZE),
            q_values.reshape(batch_size, self.BOARD_SIZE, self.BOARD_SIZE),
        )

    @staticmethod
    def _get_player_from_state(state: np.ndarray) -> int:
        """状態から現在のプレイヤーを取得"""
        return 1 if np.all(state[2] == 1) else 0
