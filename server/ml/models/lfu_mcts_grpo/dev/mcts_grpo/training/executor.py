"""
ネットワークの学習実行
"""

import numpy as np
import torch
from accelerate import Accelerator
from tqdm.rich import tqdm

from ..loss_calculator import LossCalculator
from ..replay_buffer import Experience, ReplayBuffer
from ..settings import Settings


class TrainingExecutor:
    """ネットワークの学習実行"""

    def __init__(
        self,
        settings: Settings,
        accelerator: Accelerator,
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        self.settings = settings
        self.accelerator = accelerator
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_calculator = LossCalculator(settings.training, settings.grpo)

    def run_training(self, iteration: int, replay_buffer: ReplayBuffer):
        """トレーニングループを実行"""
        self.model.train()
        self.optimizer.zero_grad()

        pbar = tqdm(
            total=self.settings.training.training_steps_per_iteration,
            desc=f"[Iter {iteration + 1}] Training",
        )

        for step in range(self.settings.training.training_steps_per_iteration):
            self._training_step(step, iteration, replay_buffer, pbar)

        pbar.close()

    def _training_step(
        self, step: int, iteration: int, replay_buffer: ReplayBuffer, pbar: tqdm
    ):
        """1ステップのトレーニング"""
        experiences = replay_buffer.sample(self.settings.training.train_batch_size)
        batch = self._prepare_batch(experiences)
        losses = self._compute_losses(batch)
        total_loss = self._compute_total_loss(losses)

        self.accelerator.backward(total_loss)
        self.optimizer.step()
        self.lr_scheduler.step()

        pbar.update(1)
        self._log_metrics(losses, total_loss, step, iteration)
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

    def _prepare_batch(self, experiences: list[Experience]) -> dict[str, torch.Tensor]:
        """経験データをテンソルに変換"""
        states = torch.from_numpy(np.stack([e.state for e in experiences])).to(
            device=self.device, dtype=torch.float32
        )
        pis = torch.from_numpy(np.stack([e.pi for e in experiences])).to(self.device)
        outcomes = torch.tensor(
            [e.outcome for e in experiences], dtype=torch.float32
        ).to(self.device)
        q_vals = torch.from_numpy(np.stack([e.q_values for e in experiences])).to(
            dtype=torch.float32, device=self.device
        )
        visit_counts = torch.from_numpy(
            np.stack([e.visit_counts for e in experiences])
        ).to(dtype=torch.float32, device=self.device)
        return {
            "states": states,
            "pis": pis,
            "outcomes": outcomes,
            "q_vals": q_vals,
            "visit_counts": visit_counts,
        }

    def _compute_losses(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """損失を計算"""
        states = batch["states"]
        pis = batch["pis"]
        outcomes = batch["outcomes"]
        q_vals = batch["q_vals"]
        visit_counts = batch["visit_counts"]

        pred_logits, pred_values = self.model(states)

        return self.loss_calculator.compute_all_losses(
            pred_logits, pred_values, states, pis, outcomes, q_vals, visit_counts
        )

    def _compute_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """総損失を計算"""
        return self.loss_calculator.compute_total_loss(losses)

    def _log_metrics(
        self,
        losses: dict[str, torch.Tensor],
        total_loss: torch.Tensor,
        step: int,
        iteration: int,
    ):
        """メトリクスをログ出力"""
        self.accelerator.log(
            {
                "loss/total": total_loss.item(),
                "loss/value": losses["value_loss"].item(),
                "loss/policy": losses["policy_loss"].item(),
                "loss/grpo": losses["grpo_loss"].item(),
            },
            step=iteration * self.settings.training.training_steps_per_iteration + step,
        )
