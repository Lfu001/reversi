import torch
from accelerate import Accelerator
from accelerate.utils import release_memory, set_seed
from evaluate import evaluate
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
    accelerator, device, model, optimizers, lr_schedulers, env, replay_buffer = (
        setup.initialize()
    )
    accelerator.init_trackers("reversi-mcts-grpo", config=settings.model_dump())
    best_win_rate = 0.0

    num_iterations = (
        settings.training.total_training_steps
        // settings.training.training_steps_per_iteration
    )

    for iteration in tqdm(
        range(num_iterations), desc="Iteration", disable=not accelerator.is_main_process
    ):
        best_win_rate = _run_iteration(
            iteration,
            settings,
            accelerator,
            device,
            model,
            optimizers,
            lr_schedulers,
            env,
            replay_buffer,
            best_win_rate,
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
    optimizers: list[torch.optim.Optimizer],
    lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    env: ReversiEnvironment,
    replay_buffer: ReplayBuffer,
    best_win_rate: float,
) -> float:
    """1つのイテレーション（Self-Play + Training）を実行"""
    executor = SelfPlayExecutor(settings, env, replay_buffer)
    executor.run_self_play(iteration, model, device)

    release_memory()

    trainer = TrainingExecutor(
        settings, accelerator, device, model, optimizers, lr_schedulers
    )
    trainer.run_training(iteration, replay_buffer)

    # Win rate evaluation at end of each iteration
    if accelerator.is_main_process:
        best_win_rate = _evaluate_and_log(
            iteration, settings, accelerator, model, device, best_win_rate
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state()

    return best_win_rate


def _evaluate_and_log(
    iteration: int,
    settings: Settings,
    accelerator: Accelerator,
    model: torch.nn.Module,
    device: torch.device,
    best_win_rate: float,
) -> float:
    """モデルの勝率を評価してログ"""
    # 評価中はモデルをeval modeに
    model.eval()

    # 計算されたglobal stepを使用
    global_step = (iteration + 1) * settings.training.training_steps_per_iteration

    # 3種類の相手に対して10ゲーム（5ペア）ずつ評価
    opponent_types = ["random", "greedy", "mobility"]
    win_rates: list[float] = []
    for opponent in opponent_types:
        results = evaluate(
            num_games=10,
            mcts_sims=settings.mcts.num_simulations,
            opponent_type=opponent,
            model=model,
            device=device,
            max_random_moves=6,
            quiet=True,
        )
        win_rates.append(results["win_rate"])

        accelerator.log(
            {
                f"eval/{opponent}/win_rate": results["win_rate"],
                f"eval/{opponent}/wins": results["wins"],
                f"eval/{opponent}/losses": results["losses"],
                f"eval/{opponent}/draws": results["draws"],
            },
            step=global_step,
        )

    # Calculate average win rate and save best checkpoint if improved
    avg_win_rate = sum(win_rates) / len(win_rates)
    accelerator.log({"eval/avg_win_rate": avg_win_rate}, step=global_step)

    if avg_win_rate > best_win_rate:
        print(
            f"New best avg win rate: {avg_win_rate:.2%} (previous: {best_win_rate:.2%})"
        )
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("logs/best_model")
        best_win_rate = avg_win_rate

    return best_win_rate


def _save_model(accelerator: Accelerator, model: torch.nn.Module):
    """モデルを保存"""
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./reversi_zero_model_final")


if __name__ == "__main__":
    print("Starting Reversi Zero training with Tree-GRPO.")
    train()
