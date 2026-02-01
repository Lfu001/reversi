"""
Training時の診断メトリクスを計算するモジュール

学習の進行状況とモデルの品質を監視するための各種指標を計算。
"""

import torch
import torch.nn.functional as F


class MetricsCalculator:
    """Training時の診断メトリクスを計算するクラス"""

    @staticmethod
    def compute_policy_entropy(
        pred_logits: torch.Tensor, legal_mask: torch.Tensor
    ) -> float:
        """ポリシーのエントロピーを計算

        探索多様性の維持確認に使用。
        高いエントロピー = 多様な探索、低いエントロピー = 決定的なポリシー

        Args:
            pred_logits: ポリシーネットワークの出力 (batch, 8, 8) or (batch, 64)
            legal_mask: 合法手マスク (batch, 8, 8) or (batch, 64)

        Returns:
            バッチ平均のエントロピー値
        """
        batch_size = pred_logits.shape[0]
        logits = pred_logits.view(batch_size, -1)  # (B, 64)
        mask = legal_mask.view(batch_size, -1) > 0  # (B, 64)

        # マスクされたsoftmax（非合法手は-inf）
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(masked_logits, dim=-1)

        # エントロピー: -sum(p * log(p))、0の確率は無視
        # log(0)を避けるためclamp
        log_probs = torch.log(probs.clamp(min=1e-6))
        entropy = -(probs * log_probs).sum(dim=-1)

        # NaN/Infをフィルタ（全て非合法のサンプルがあり得る）
        valid_entropy = entropy[torch.isfinite(entropy)]
        if valid_entropy.numel() == 0:
            return 0.0

        return valid_entropy.mean().item()

    @staticmethod
    def compute_value_accuracy(
        pred_values: torch.Tensor, outcomes: torch.Tensor
    ) -> float:
        """Value予測の精度を計算

        予測値と実際の勝敗の符号一致率。
        Value headが勝敗を正しく予測できているかの指標。

        Args:
            pred_values: 予測値 (batch,) or (batch, 1)
            outcomes: 実際の勝敗 (batch,) - 値は-1, 0, 1

        Returns:
            符号一致率 [0, 1]
        """
        pred = pred_values.view(-1)
        target = outcomes.view(-1)

        # 引き分け（outcome=0）は予測困難なので除外
        non_draw_mask = target != 0
        if not non_draw_mask.any():
            return 1.0  # 引き分けのみなら完璧とみなす

        pred_sign = torch.sign(pred[non_draw_mask])
        target_sign = torch.sign(target[non_draw_mask])

        accuracy = (pred_sign == target_sign).float().mean()
        return accuracy.item()

    @staticmethod
    def compute_illegal_move_prob(
        pred_logits: torch.Tensor, legal_mask: torch.Tensor
    ) -> float:
        """非合法手に割り当てられた確率の合計を計算

        ポリシーネットワークが非合法手に確率を割り当てていないかを確認。
        理想的には0に近い値。

        Args:
            pred_logits: ポリシーネットワークの出力 (batch, 8, 8) or (batch, 64)
            legal_mask: 合法手マスク (batch, 8, 8) or (batch, 64)

        Returns:
            バッチ平均の非合法手確率合計
        """
        batch_size = pred_logits.shape[0]
        logits = pred_logits.view(batch_size, -1)  # (B, 64)
        mask = legal_mask.view(batch_size, -1) > 0  # (B, 64)

        # softmaxで確率を計算（マスクなし）
        probs = F.softmax(logits, dim=-1)

        # 非合法手の確率合計
        illegal_prob = probs.masked_fill(mask, 0.0).sum(dim=-1)

        return illegal_prob.mean().item()

    @staticmethod
    def compute_gradient_norm(model: torch.nn.Module) -> float:
        """モデルの勾配L2ノルムを計算

        勾配爆発/消失の検出に使用。
        backward()の後、optimizer.step()の前に呼び出す。

        Args:
            model: 勾配が計算されたモデル

        Returns:
            全パラメータの勾配L2ノルム
        """
        # Cast to float32 to avoid overflow with fp16 gradients
        return (
            sum(
                param.grad.data.float().norm(2).item() ** 2
                for param in model.parameters()
                if param.grad is not None
            )
            ** 0.5
        )
