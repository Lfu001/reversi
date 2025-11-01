import numpy as np
import torch
import torch.nn.functional as F
from reversi import ReversiEnvironment

from reversizero_model.configuration_reversizero import ReversiZeroConfig
from reversizero_model.modeling_reversizero import ReversiZeroModel


class MCTS:
    def __init__(self, config: ReversiZeroConfig):
        self.config = config
        # Q(s,a): state s で action a を取ったときの平均行動価値
        self.Q: dict[bytes, np.ndarray] = {}
        # N(s,a): state s で action a を選択した回数
        self.N: dict[bytes, np.ndarray] = {}
        # P(s): state s における方策 (NNからの出力)
        self.P: dict[bytes, np.ndarray] = {}
        # V(s): state s の訪問回数 (デバッグ用)
        self.visit_counts: dict[bytes, int] = {}
        # W(s): state s からシミュレートしたゲームの合計報酬 (value)
        # 終局状態の価値を直接格納するために使用する
        self.W: dict[bytes, float] = {}

    def _get_player_from_state(self, state: np.ndarray) -> int:
        """
        stateのc2チャネルから現在のプレイヤーを返す (1: 黒, 0: 白)。
        これはヘルパー関数であり、可読性を向上させるために使用します。
        """
        # c2チャネルがすべて1なら黒番
        return 1 if np.all(state[2] == 1) else 0

    def run_simulations(
        self, model: ReversiZeroModel, root_states: np.ndarray, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        与えられたバッチのルート状態に対して、MCTSシミュレーションを実行する。

        Returns:
            tuple[np.ndarray, np.ndarray]:
            - pi (np.ndarray): 改善された方策 (探索回数の分布) Shape: (batch, 8, 8)
            - Q (np.ndarray): ルートノードにおける各アクションのQ値 Shape: (batch, 8, 8)
        """
        # --- ルートノードの初期展開 ---
        # まだ木に存在しないルートノードがあれば、NNで評価して木に追加する
        new_states_to_evaluate_mask = [
            state.tobytes() not in self.P for state in root_states
        ]
        if any(new_states_to_evaluate_mask):
            states_to_eval = np.stack(
                [
                    s
                    for s, needs_eval in zip(root_states, new_states_to_evaluate_mask)
                    if needs_eval
                ]
            )

            with torch.no_grad():
                policy_logits, values = model(
                    torch.from_numpy(states_to_eval).to(device)
                )

            legal_moves_mask = states_to_eval[:, 3, :, :].reshape(
                states_to_eval.shape[0], -1
            )
            policy_logits = policy_logits.view(states_to_eval.shape[0], -1)
            policy_logits[
                torch.from_numpy(legal_moves_mask == 0).to(device)
            ] = -torch.inf
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()

            eval_idx = 0
            for i, needs_eval in enumerate(new_states_to_evaluate_mask):
                if needs_eval:
                    state_key = root_states[i].tobytes()
                    self.P[state_key] = policy_probs[eval_idx]
                    self.Q[state_key] = np.zeros(64, dtype=np.float32)
                    self.N[state_key] = np.zeros(64, dtype=np.float32)
                    self.visit_counts[state_key] = 0
                    eval_idx += 1

        # --- ディリクレノイズをルートノードの方策に追加 ---
        for state in root_states:
            state_key = state.tobytes()
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * 64)
            self.P[state_key] = (
                self.P[state_key] * (1 - self.config.dirichlet_epsilon)
                + noise * self.config.dirichlet_epsilon
            )

        # --- シミュレーションループ ---
        for _ in range(self.config.num_simulations):
            current_states = root_states.copy()
            paths: list[list[tuple[np.ndarray, int]]] = [
                [] for _ in range(self.config.batch_size)
            ]
            is_terminal = np.zeros(self.config.batch_size, dtype=bool)

            # --- 1. Selection (選択) ---
            for i in range(self.config.batch_size):
                state = current_states[i]

                while state.tobytes() in self.P:
                    if np.sum(state[3]) == 0:
                        is_terminal[i] = True
                        break

                    visit_sum = np.sqrt(np.sum(self.N[state.tobytes()]))
                    puct_scores = self.Q[state.tobytes()] + self.config.c_puct * self.P[
                        state.tobytes()
                    ] * visit_sum / (1 + self.N[state.tobytes()])

                    legal_moves = state[3].flatten()
                    puct_scores[legal_moves == 0] = -np.inf
                    action = np.argmax(puct_scores)  # .item()は不要

                    paths[i].append((state, action))
                    state = ReversiEnvironment.get_next_state(state, action)

                current_states[i] = state

            # --- 2. Expansion (展開) & Evaluation (評価) ---
            leaf_states = current_states
            values_np = np.zeros(self.config.batch_size, dtype=np.float32)

            needs_eval_mask = ~is_terminal
            if np.any(needs_eval_mask):
                states_to_eval = leaf_states[needs_eval_mask]
                with torch.no_grad():
                    policy_logits, network_values = model(
                        torch.from_numpy(states_to_eval).to(device)
                    )

                legal_moves_mask = states_to_eval[:, 3, :, :].reshape(
                    states_to_eval.shape[0], -1
                )
                policy_logits = policy_logits.view(states_to_eval.shape[0], -1)
                policy_logits[
                    torch.from_numpy(legal_moves_mask == 0).to(device)
                ] = -torch.inf
                policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
                values_np[needs_eval_mask] = network_values.cpu().numpy()

                eval_idx = 0
                for i in range(self.config.batch_size):
                    if needs_eval_mask[i]:
                        state_key = leaf_states[i].tobytes()
                        self.P[state_key] = policy_probs[eval_idx]
                        self.Q[state_key] = np.zeros(64, dtype=np.float32)
                        self.N[state_key] = np.zeros(64, dtype=np.float32)
                        self.visit_counts[state_key] = 0
                        eval_idx += 1

            terminal_mask = is_terminal
            if np.any(terminal_mask):
                terminal_states = leaf_states[terminal_mask]
                outcomes = np.zeros(terminal_states.shape[0])
                for i, state in enumerate(terminal_states):
                    num_black = np.sum(state[0])
                    num_white = np.sum(state[1])
                    is_black_perspective = np.all(state[2] == 1)
                    if num_black > num_white:
                        outcomes[i] = 1.0 if is_black_perspective else -1.0
                    elif num_white > num_black:
                        outcomes[i] = -1.0 if is_black_perspective else 1.0
                    else:
                        outcomes[i] = 0.0
                values_np[terminal_mask] = outcomes

            # --- 3. Backup (バックアップ) ---
            # ★★★ 修正の核心部分 ★★★
            for i in range(self.config.batch_size):
                value = values_np[i]
                child_state = leaf_states[i]  # バックアップの開始点(子)はリーフノード

                # パスを逆順にたどる
                for parent_state, action in reversed(paths[i]):
                    state_key = parent_state.tobytes()

                    self.Q[state_key][action] = (
                        self.N[state_key][action] * self.Q[state_key][action] + value
                    ) / (self.N[state_key][action] + 1)
                    self.N[state_key][action] += 1
                    self.visit_counts[state_key] += 1

                    parent_player = self._get_player_from_state(parent_state)
                    child_player = self._get_player_from_state(child_state)

                    if parent_player != child_player:
                        value *= -1

                    # 次のループのために、現在の親を未来の子として設定する
                    child_state = parent_state

        # --- 最終的な方策πとQ値を計算 ---
        pis = np.zeros((self.config.batch_size, 64), dtype=np.float32)
        q_values = np.zeros((self.config.batch_size, 64), dtype=np.float32)
        for i, state in enumerate(root_states):
            state_key = state.tobytes()
            visit_counts = self.N[state_key]
            if np.sum(visit_counts) > 0:
                pis[i] = visit_counts / np.sum(visit_counts)
            q_values[i] = self.Q[state_key]

        return pis.reshape(-1, 8, 8), q_values.reshape(-1, 8, 8)
