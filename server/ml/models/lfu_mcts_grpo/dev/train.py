import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import release_memory, set_seed
from mcts_grpo.replay_buffer import ReplayBuffer
from mcts_grpo.settings import Settings
from mcts_grpo.training import (
    GameSetup,
    SelfPlayExecutor,
    TrainingExecutor,
)
from reversi import ReversiEnvironment
from tqdm import tqdm


def train():
    set_seed(0)
    settings = Settings()
    setup = GameSetup(settings)
    accelerator, device, model, optimizer, lr_scheduler, env, replay_buffer = (
        setup.initialize()
    )
    accelerator.init_trackers("reversi-mcts-grpo", config=settings.model_dump())

    ongoing_games_data = [[] for _ in range(settings.training.batch_size)]
    num_iterations = (
        settings.training.total_training_steps
        // settings.training.training_steps_per_iteration
    )

    for iteration in tqdm(
        range(num_iterations), desc="Iteration", disable=not accelerator.is_main_process
    ):
        _run_iteration(
            iteration,
            settings,
            accelerator,
            device,
            model,
            optimizer,
            lr_scheduler,
            env,
            replay_buffer,
            ongoing_games_data,
        )

    accelerator.end_training()
    print("Training finished.")
    _save_model(accelerator, model)


def _run_iteration(
    iteration: int,
    settings: Settings,
    accelerator: Accelerator,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    env: ReversiEnvironment,
    replay_buffer: ReplayBuffer,
    ongoing_games_data: list[list[dict[str, np.ndarray]]],
):
    """1つのイテレーション（Self-Play + Training）を実行"""
    executor = SelfPlayExecutor(settings, env, replay_buffer, ongoing_games_data)
    executor.run_self_play(iteration, model, device)

    release_memory()

    trainer = TrainingExecutor(
        settings, accelerator, device, model, optimizer, lr_scheduler
    )
    trainer.run_training(iteration, replay_buffer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state()


def _save_model(accelerator: Accelerator, model: torch.nn.Module):
    """モデルを保存"""
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./reversi_zero_model_final")


if __name__ == "__main__":
    print("Starting Reversi Zero training with Tree-GRPO.")
    train()
