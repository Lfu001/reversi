import torch
import torch.nn.functional as F


def compute_grpo_loss(
    pred_logits: torch.Tensor,
    q_values: torch.Tensor,
    legal_moves: torch.Tensor,
    num_pairs: int,
    non_terminal_mask: torch.Tensor,
) -> torch.Tensor:
    # マスクを適用して、終局盤面のlogitとq_valueをスライス
    pred_logits = pred_logits[non_terminal_mask]
    q_values = q_values[non_terminal_mask]
    legal_moves = legal_moves[non_terminal_mask]

    # もしバッチ内に非終局盤面がなければ、損失は0
    if pred_logits.shape[0] == 0:
        return torch.tensor(0.0, device=pred_logits.device)

    current_batch_size = pred_logits.shape[0]
    total_loss = 0.0

    log_probs = F.log_softmax(pred_logits.view(current_batch_size, -1), dim=1)

    for i in range(current_batch_size):
        item_log_probs = log_probs[i]  # [64]
        item_q_values = q_values[i].view(-1)  # [64]
        item_legal_moves = legal_moves[i].view(-1)  # [64]

        legal_indices = torch.where(item_legal_moves > 0)[0]
        # このチェックは理論上不要になるが、安全のために残す
        if len(legal_indices) < 2:
            continue

        legal_q = item_q_values[legal_indices]

        # 安定性のため、Q値が同じ場合のtie-breakingを追加
        best_action_idx = legal_indices[torch.argmax(legal_q)]

        suboptimal_indices = legal_indices[legal_indices != best_action_idx]
        if len(suboptimal_indices) == 0:
            continue

        num_samples = min(num_pairs, len(suboptimal_indices))
        # torch.randpermの引数が0にならないように保護
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
