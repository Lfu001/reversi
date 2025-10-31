from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from reversi import ReversiEnvironment
from tqdm.rich import tqdm

from loss import compute_grpo_loss
from mcts import MCTS
from replay_buffer import Experience, ReplayBuffer
from reversizero_model.configuration_reversizero import ReversiZeroConfig
from reversizero_model.modeling_reversizero import ReversiZeroModel


class Winner(Enum):
    DRAW = 0
    DARK = 1
    LIGHT = -1


def train():
    set_seed(0)
    config = ReversiZeroConfig()
    accelerator = Accelerator(
        log_with="wandb",
        project_dir="logs",
        project_config=ProjectConfiguration(
            total_limit=3, automatic_checkpoint_naming=True
        ),
    )
    accelerator.init_trackers("reversi_zero", config=config.to_dict())

    device = accelerator.device

    env = ReversiEnvironment(batch_size=config.batch_size)
    model = ReversiZeroModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_training_steps
    )
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    mcts = MCTS(config)

    ongoing_games_data = [[] for _ in range(config.batch_size)]

    # --- メインの学習ループ ---
    num_iterations = config.total_training_steps // config.training_steps_per_iteration
    for iteration in tqdm(range(num_iterations), desc="Iteration"):
        # --- Self-Play (データ生成) ---
        model.eval()
        games_completed_in_iteration = 0
        pbar_selfplay = tqdm(
            total=config.games_per_iteration, desc=f"[Iter {iteration + 1}] Self-Play"
        )
        step_count = 0
        current_states = env.reset()
        while games_completed_in_iteration < config.games_per_iteration:
            pi, q_values = mcts.run_simulations(model, current_states, device)
            next_states, dones = env.step_batch(pi, deterministic=False)
            step_count += 1
            for i in range(config.batch_size):
                ongoing_games_data[i].append(
                    {"state": current_states[i], "pi": pi[i], "q_values": q_values[i]}
                )
                if dones[i]:
                    games_completed_in_iteration += 1
                    pbar_selfplay.update(1)

                    # 1. 終局盤面を取得
                    terminal_state = next_states[i]

                    # 2. 石の数を数える
                    num_dark = np.sum(terminal_state[0])
                    num_light = np.sum(terminal_state[1])

                    # 3. 勝者を判定
                    winner = Winner.DRAW
                    if num_dark > num_light:
                        winner = Winner.DARK
                    elif num_light > num_dark:
                        winner = Winner.LIGHT

                    # 4. ゲーム履歴をたどりながら、各プレイヤー視点のoutcomeを計算
                    #    ongoing_games_data[i] には、そのゲームの [state_0, state_1, ...] が格納されている

                    for experience_data in reversed(ongoing_games_data[i]):
                        # その局面(state)の手番プレイヤーを取得
                        # c2チャネルが全面1なら黒番、全面-1なら白番
                        is_dark_turn = np.all(experience_data["state"][2] > 0)

                        if winner == Winner.DRAW:  # 引き分けの場合
                            current_outcome = 0.0
                        elif (winner == Winner.DARK and is_dark_turn) or (
                            winner == Winner.LIGHT and not is_dark_turn
                        ):
                            # 勝ちプレイヤー(黒)の手番、あるいは、負けプレイヤー(白)の相手の手番
                            current_outcome = 1.0
                        else:
                            # 負けプレイヤー(黒)の手番、あるいは、勝ちプレイヤー(白)の相手の手番
                            current_outcome = -1.0

                        replay_buffer.push(
                            Experience(
                                state=experience_data["state"],
                                pi=experience_data["pi"],
                                outcome=current_outcome,  # ★ 正しいoutcomeを設定
                                q_values=experience_data["q_values"],
                            )
                        )

                    ongoing_games_data[i] = []  # ゲーム履歴をクリア
            current_states = env.reset_indices(np.where(dones)[0])
            if games_completed_in_iteration >= config.games_per_iteration:
                break
        pbar_selfplay.close()

        # --- Training (ネットワーク更新) ---
        model.train()
        optimizer.zero_grad()

        pbar_train = tqdm(
            total=config.training_steps_per_iteration,
            desc=f"[Iter {iteration + 1}] Training",
        )
        for step in range(config.training_steps_per_iteration):
            experiences = replay_buffer.sample(config.batch_size)

            states = torch.from_numpy(np.stack([e.state for e in experiences])).to(
                device
            )
            pis = torch.from_numpy(np.stack([e.pi for e in experiences])).to(device)
            outcomes = torch.tensor(
                [e.outcome for e in experiences], dtype=torch.float32
            ).to(device)
            q_vals = torch.from_numpy(np.stack([e.q_values for e in experiences])).to(
                dtype=torch.float32, device=device
            )

            legal_moves = states[:, 3, :, :]
            non_terminal_mask = (
                torch.sum(legal_moves.view(config.batch_size, -1), dim=1) > 0
            )

            pred_logits, pred_values = model(states)

            # Value Loss
            value_loss = F.mse_loss(pred_values, outcomes)

            if non_terminal_mask.any():
                # マスクを適用して、終局盤面のサンプルを損失計算から除外
                pred_logits_masked = pred_logits[non_terminal_mask]
                pis_masked = pis[non_terminal_mask]

                # --- 数値安定性のための追加対策：logitのクランプ ---
                # logitが極端な値になるのを防ぐ
                pred_logits_masked = torch.clamp(pred_logits_masked, -100.0, 100.0)

                # Policy Loss (AlphaZero standard)
                policy_logits_flat = pred_logits_masked.view(-1, 64)
                pis_flat = pis_masked.view(-1, 64)
                policy_loss = F.cross_entropy(policy_logits_flat, pis_flat)

                # Tree-GRPO Loss
                grpo_loss = compute_grpo_loss(
                    pred_logits,  # 元のサイズのものを渡す
                    q_vals,  # 元のサイズのものを渡す
                    legal_moves,  # 元のサイズのものを渡す
                    config.grpo_num_pairs,
                    non_terminal_mask,  # マスクを渡す
                )
            else:
                policy_loss = torch.tensor(0.0, device=device)
                grpo_loss = torch.tensor(0.0, device=device)

            # Total Loss
            total_loss = value_loss + policy_loss + config.lambda_grpo * grpo_loss

            accelerator.backward(total_loss)
            optimizer.step()
            lr_scheduler.step()

            pbar_train.update(1)

            # Logging
            accelerator.log(
                {
                    "loss/total": total_loss.item(),
                    "loss/value": value_loss.item(),
                    "loss/policy": policy_loss.item(),
                    "loss/grpo": grpo_loss.item(),
                },
                step=iteration * config.training_steps_per_iteration + step,
            )
            pbar_train.set_postfix({"loss": f"{total_loss.item():.4f}"})

        pbar_train.close()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_state()

    accelerator.end_training()
    print("Training finished.")

    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./reversi_zero_model_final")


if __name__ == "__main__":
    print("Starting Reversi Zero training with Tree-GRPO.")
    train()
