"""
損失計算の一元化
Value Loss、Policy Loss、GRPO Lossを統一的に計算
"""

import torch
import torch.nn.functional as F

from .settings import TrainingConfig, TreeGRPOConfig


class LossCalculator:
    """全ての損失を計算するクラス"""

    def __init__(self, training_config: TrainingConfig, grpo_config: TreeGRPOConfig):
        self.training_config = training_config
        self.grpo_config = grpo_config

    def compute_all_losses(
        self,
        pred_logits: torch.Tensor,
        pred_values: torch.Tensor,
        states: torch.Tensor,
        pis: torch.Tensor,
        outcomes: torch.Tensor,
        q_vals: torch.Tensor,
        visit_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """全ての損失を計算"""
        value_loss = self._compute_value_loss(pred_values, outcomes)

        legal_moves = states[:, 3, :, :]
        non_terminal_mask = self._compute_non_terminal_mask(legal_moves)

        policy_loss, grpo_loss = self._compute_policy_and_grpo_loss(
            pred_logits, pis, legal_moves, q_vals, visit_counts, non_terminal_mask
        )

        return {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "grpo_loss": grpo_loss,
        }

    def compute_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """総損失を計算"""
        return (
            self.training_config.lambda_value * losses["value_loss"]
            + losses["policy_loss"]
            + self.grpo_config.lambda_grpo * losses["grpo_loss"]
        )

    def _compute_value_loss(
        self, pred_values: torch.Tensor, outcomes: torch.Tensor
    ) -> torch.Tensor:
        """Value Lossを計算"""
        # pred_values: (batch, 1), outcomes: (batch,) -> squeeze to match
        return F.mse_loss(pred_values.squeeze(-1), outcomes)

    def _compute_non_terminal_mask(self, legal_moves: torch.Tensor) -> torch.Tensor:
        """非終局盤面のマスクを計算"""
        batch_size = legal_moves.shape[0]
        return torch.sum(legal_moves.view(batch_size, -1), dim=1) > 0

    def _compute_policy_and_grpo_loss(
        self,
        pred_logits: torch.Tensor,
        pis: torch.Tensor,
        legal_moves: torch.Tensor,
        q_vals: torch.Tensor,
        visit_counts: torch.Tensor,
        non_terminal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy LossとGRPO Lossを計算"""
        if not non_terminal_mask.any():
            device = pred_logits.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        policy_loss = self._compute_policy_loss(pred_logits, pis, non_terminal_mask)
        grpo_loss = self._compute_grpo_loss(
            pred_logits, q_vals, legal_moves, visit_counts, non_terminal_mask
        )

        return policy_loss, grpo_loss

    def _compute_policy_loss(
        self,
        pred_logits: torch.Tensor,
        pis: torch.Tensor,
        non_terminal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Policy Lossを計算"""
        pred_logits_masked = pred_logits[non_terminal_mask]
        pis_masked = pis[non_terminal_mask]
        pred_logits_masked = torch.clamp(pred_logits_masked, -100.0, 100.0)

        policy_logits_flat = pred_logits_masked.view(-1, 64)
        pis_flat = pis_masked.view(-1, 64)
        return F.cross_entropy(policy_logits_flat, pis_flat)

    def _compute_grpo_loss(
        self,
        pred_logits: torch.Tensor,
        q_vals: torch.Tensor,
        legal_moves: torch.Tensor,
        visit_counts: torch.Tensor,
        non_terminal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """GRPO Lossを計算"""
        # マスクを適用
        pred_logits = pred_logits[non_terminal_mask]
        q_values = q_vals[non_terminal_mask]
        legal_moves = legal_moves[non_terminal_mask]
        visit_counts = visit_counts[non_terminal_mask]

        if pred_logits.shape[0] == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        current_batch_size = pred_logits.shape[0]
        total_loss = torch.zeros((), device=pred_logits.device)

        log_probs = F.log_softmax(pred_logits.view(current_batch_size, -1), dim=1)

        for i in range(current_batch_size):
            item_log_probs = log_probs[i]
            item_q_values = q_values[i].view(-1)
            item_legal_moves = legal_moves[i].view(-1)
            item_visit_counts = visit_counts[i].view(-1)

            legal_indices = torch.where(item_legal_moves > 0)[0]

            # 1. 合法手のvisit_countsを取得
            legal_visits = item_visit_counts[legal_indices]

            # 2. visit_count > 0 のノードが探索済み
            explored_mask = legal_visits > 0
            explored_legal_indices = legal_indices[explored_mask]

            # 3. 探索済みの合法手が2手未満（比較対象がいない）場合は、
            #    このサンプルのGRPO損失は 0 とする
            if len(explored_legal_indices) < 2:
                continue

            # 4. 探索済みの手の中だけで Q_best と Q_suboptimal を決める
            explored_legal_q = item_q_values[explored_legal_indices]
            best_action_idx = explored_legal_indices[torch.argmax(explored_legal_q)]

            suboptimal_indices = explored_legal_indices[
                explored_legal_indices != best_action_idx
            ]

            if len(suboptimal_indices) == 0:
                continue

            num_samples = min(self.grpo_config.grpo_num_pairs, len(suboptimal_indices))
            perm_indices = torch.randperm(
                len(suboptimal_indices), device=pred_logits.device
            )[:num_samples]
            sampled_suboptimal_indices = suboptimal_indices[perm_indices]

            logp_best = item_log_probs[best_action_idx]
            logp_suboptimal = item_log_probs[sampled_suboptimal_indices]

            loss = F.softplus(-(logp_best - logp_suboptimal)).mean()
            total_loss += loss
        return (
            total_loss / current_batch_size
            if current_batch_size > 0
            else torch.tensor(0.0, device=pred_logits.device)
        )
