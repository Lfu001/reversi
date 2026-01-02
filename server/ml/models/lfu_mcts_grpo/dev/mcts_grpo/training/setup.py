"""
ゲーム環境とモデルの初期化
"""

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from reversi import ReversiEnvironment

from ..model.configuration_urm import URMConfig
from ..model.modeling_urm import URMModel
from ..replay_buffer import ReplayBuffer
from ..settings import Settings


class GameSetup:
    """ゲーム環境とモデルの初期化"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def initialize(
        self,
    ) -> tuple[
        Accelerator,
        torch.device,
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
        ReversiEnvironment,
        ReplayBuffer,
    ]:
        """全ての必要なコンポーネントを初期化して返す"""
        accelerator = self._create_accelerator()
        device = accelerator.device
        model, optimizer, lr_scheduler = self._setup_model_and_optimizer(accelerator)
        env = ReversiEnvironment(batch_size=self.settings.training.batch_size)
        replay_buffer = ReplayBuffer(self.settings.training.replay_buffer_size)
        return accelerator, device, model, optimizer, lr_scheduler, env, replay_buffer

    def _create_accelerator(self) -> Accelerator:
        """Acceleratorを作成"""
        return Accelerator(
            log_with="wandb",
            project_dir="logs",
            project_config=ProjectConfiguration(
                total_limit=3, automatic_checkpoint_naming=True
            ),
        )

    def _setup_model_and_optimizer(
        self, accelerator: Accelerator
    ) -> tuple[
        torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
    ]:
        """モデル、オプティマイザ、スケジューラを初期化"""
        model = URMModel(URMConfig(**self.settings.model.model_dump()))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.settings.training.learning_rate,
            weight_decay=self.settings.training.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.settings.training.total_training_steps
        )
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
        return model, optimizer, lr_scheduler
