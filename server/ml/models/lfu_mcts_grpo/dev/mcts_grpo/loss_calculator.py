"""
損失計算の一元化
Value Loss、Policy Loss、GRPO Lossを統一的に計算

Tree-GRPO論文 (arXiv 2509.21240) に基づくMCTS-GRPO損失関数を実装。
Intra-tree + Inter-tree アドバンテージをベクトル化して計算。
"""

import torch
import torch.nn.functional as F

from .settings import TrainingConfig, TreeGRPOConfig

# Numerical stability constant for advantage normalization
_ADVANTAGE_EPS = 1e-8


def _nanstd(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """NaNを無視した標準偏差を計算（計算グラフを保持、NaN勾配伝播を防止）

    NaN位置を0で置き換えてから計算することで、NaN勾配の伝播を防ぐ。
    """
    valid_mask = ~torch.isnan(x)
    # NaN位置を0で置き換え（勾配が流れないように）
    x_filled = torch.where(valid_mask, x, torch.zeros_like(x))

    if dim is None:
        count = valid_mask.sum().float()
        if count < 2:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # 合計を有効要素数で割って平均
        mean = x_filled.sum() / count
        # 差の二乗（NaN位置は0なのでそのまま0になる）
        diff_sq = (x_filled - mean * valid_mask.float()) ** 2
        diff_sq = torch.where(valid_mask, diff_sq, torch.zeros_like(diff_sq))
        variance = diff_sq.sum() / count
        return variance.sqrt()
    else:
        count = valid_mask.sum(dim=dim, keepdim=True).float()
        # 合計を有効要素数で割って平均
        sum_x = x_filled.sum(dim=dim, keepdim=True)
        mean = sum_x / torch.clamp(count, min=1)
        # 差の二乗（NaN位置は0で、meanもbroadcast）
        diff = x_filled - mean * valid_mask.float()
        diff_sq = diff**2
        diff_sq = torch.where(valid_mask, diff_sq, torch.zeros_like(diff_sq))
        variance = diff_sq.sum(dim=dim, keepdim=True) / torch.clamp(count, min=1)
        return variance.sqrt()


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
            pred_logits, q_vals, legal_moves, visit_counts, pis, non_terminal_mask
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
        pis: torch.Tensor,
        non_terminal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """GRPO Lossを計算（Tree-GRPO論文に基づくPPO-style clipped objective）

        ベクトル化された実装:
        1. Intra-tree: 各サンプル内でQ値を正規化してアドバンテージ計算
        2. Inter-tree: バッチ全体でQ値を正規化してアドバンテージ計算
        3. 両者を合成してPPO-style clipped objectiveで損失計算

        Args:
            pred_logits: 現在のポリシーネットワークの出力 (batch, 8, 8)
            q_vals: MCTSが算出したQ値 (batch, 8, 8)
            legal_moves: 合法手マスク (batch, 8, 8)
            visit_counts: 各アクションの訪問回数 (batch, 8, 8)
            pis: 保存時のポリシー確率（importance sampling用）(batch, 8, 8)
            non_terminal_mask: 非終局盤面のマスク (batch,)

        Returns:
            GRPO損失値
        """
        # マスク適用してflatten
        pred_logits = pred_logits[non_terminal_mask].view(-1, 64)  # (B, 64)
        q_values = q_vals[non_terminal_mask].view(-1, 64)  # (B, 64)
        legal_mask = legal_moves[non_terminal_mask].view(-1, 64) > 0  # (B, 64)
        visit_mask = visit_counts[non_terminal_mask].view(-1, 64) > 0  # (B, 64)
        old_pis = pis[non_terminal_mask].view(-1, 64)  # (B, 64)

        batch_size = pred_logits.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        # 探索済みマスク: 合法手かつvisit_count > 0
        explored_mask = legal_mask & visit_mask  # (B, 64)

        # 各サンプルで2つ以上の探索済みアクションがあるかチェック
        explored_count = explored_mask.sum(dim=1)  # (B,)
        valid_samples = explored_count >= 2  # (B,)

        if not valid_samples.any():
            return torch.tensor(0.0, device=pred_logits.device)

        # 有効サンプルのみ抽出
        pred_logits = pred_logits[valid_samples]  # (B', 64)
        q_values = q_values[valid_samples]
        explored_mask = explored_mask[valid_samples]
        old_pis = old_pis[valid_samples]
        batch_size = pred_logits.shape[0]

        # 現在のポリシー確率
        current_probs = F.softmax(pred_logits, dim=1)  # (B', 64)

        # マスク付きQ値: 非探索部分はNaN
        masked_q = q_values.masked_fill(~explored_mask, float("nan"))  # (B', 64)

        # === Intra-tree アドバンテージ（各サンプル内で正規化）===
        intra_mean = torch.nanmean(masked_q, dim=1, keepdim=True)  # (B', 1)
        intra_std = _nanstd(masked_q, dim=1)  # (B', 1)

        # 標準偏差がほぼ0のサンプルはスキップ（マスクで処理）
        valid_intra = intra_std.squeeze(-1) >= _ADVANTAGE_EPS  # (B',)

        # Intra-tree advantage（NaN部分は0になる）
        advantages_intra = torch.where(
            explored_mask & valid_intra.unsqueeze(1),
            (masked_q - intra_mean) / (intra_std + _ADVANTAGE_EPS),
            torch.zeros_like(masked_q),
        )

        # === Inter-tree アドバンテージ（バッチ全体で正規化）===
        # バッチ全体の探索済みQ値で mean/std を計算
        all_explored_q = masked_q[explored_mask]  # flatten
        if all_explored_q.numel() < 2:
            inter_mean = torch.tensor(0.0, device=pred_logits.device)
            inter_std = torch.tensor(1.0, device=pred_logits.device)
        else:
            inter_mean = all_explored_q.mean()
            inter_std = all_explored_q.std()

        # Inter-tree advantage
        if inter_std >= _ADVANTAGE_EPS:
            advantages_inter = torch.where(
                explored_mask,
                (q_values - inter_mean) / (inter_std + _ADVANTAGE_EPS),
                torch.zeros_like(q_values),
            )
        else:
            advantages_inter = torch.zeros_like(q_values)

        # === 合成アドバンテージ ===
        advantages = advantages_intra + advantages_inter  # (B', 64)

        # === Importance sampling ratio ===
        old_pis_clamped = torch.clamp(old_pis, min=1e-8)
        ratio = current_probs / old_pis_clamped  # (B', 64)

        # === PPO-style clipped objective ===
        epsilon = self.grpo_config.clip_epsilon
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        # min(r*A, clip(r)*A)
        surrogate_min = torch.min(surrogate1, surrogate2)

        # 探索済みアクションのみで損失を計算
        # マスク適用: 非探索アクションは0
        masked_loss = torch.where(
            explored_mask, surrogate_min, torch.zeros_like(surrogate_min)
        )

        # 最大化 -> 最小化のため負号、各サンプルで合計してからバッチ平均
        loss_per_sample = -masked_loss.sum(dim=1)  # (B',)
        return loss_per_sample.mean()
