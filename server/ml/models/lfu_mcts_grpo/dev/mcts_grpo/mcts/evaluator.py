"""
ニューラルネットワークによる状態評価
"""

import numpy as np
import torch
import torch.nn.functional as F


class NetworkEvaluator:
    """ニューラルネットワークで状態を評価"""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate_states(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """複数の状態を評価

        Args:
            states: 評価する状態のバッチ (batch_size, channels, height, width)

        Returns:
            tuple[np.ndarray, np.ndarray]:
            - policy_probs: 方策確率 (batch_size, 64)
            - values: 状態価値 (batch_size,)
        """
        with torch.no_grad():
            policy_logits, values = self.model(torch.from_numpy(states).to(self.device))

        legal_moves_mask = states[:, 3, :, :].reshape(states.shape[0], -1)
        policy_logits = policy_logits.view(states.shape[0], -1)

        # 非合法手に-infを設定
        policy_logits[
            torch.from_numpy(legal_moves_mask == 0).to(self.device)
        ] = -torch.inf

        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
        values_np = values.cpu().numpy()

        return policy_probs, values_np

    def evaluate_single_state(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """単一の状態を評価

        Args:
            state: 評価する状態 (channels, height, width)

        Returns:
            tuple[np.ndarray, float]:
            - policy_probs: 方策確率 (64,)
            - value: 状態価値 (スカラー)
        """
        states_batch = np.expand_dims(state, axis=0)
        policy_probs, values = self.evaluate_states(states_batch)
        return policy_probs[0], values[0]
