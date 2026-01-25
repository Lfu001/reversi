"""
ゲーム環境とモデルの初期化
"""

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from reversi import ReversiEnvironment
from transformers import get_constant_schedule_with_warmup

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
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler],
        ReversiEnvironment,
        ReplayBuffer,
    ]:
        """全ての必要なコンポーネントを初期化して返す"""
        accelerator = self._create_accelerator()
        device = accelerator.device
        model, optimizers, lr_schedulers = self._setup_model_and_optimizers(accelerator)
        env = ReversiEnvironment(batch_size=self.settings.mcts.parallel_games)
        replay_buffer = ReplayBuffer(self.settings.training.replay_buffer_size)
        return accelerator, device, model, optimizers, lr_schedulers, env, replay_buffer

    def _create_accelerator(self) -> Accelerator:
        """Acceleratorを作成"""
        return Accelerator(
            log_with="wandb",
            project_dir="logs",
            project_config=ProjectConfiguration(
                total_limit=3, automatic_checkpoint_naming=True
            ),
        )

    def _setup_model_and_optimizers(
        self, accelerator: Accelerator
    ) -> tuple[
        torch.nn.Module,
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler],
    ]:
        """モデル、オプティマイザ、スケジューラを初期化"""
        model = URMModel(URMConfig(**self.settings.model.model_dump()))

        # Muonは2Dパラメータのみ対応（行列のみ、畳み込み層の3D以上は除外）
        # それ以外（1D: バイアス、LayerNorm、3D+: 畳み込み）はAdamWを使用
        muon_params = []
        adamw_params = []
        for param in model.parameters():
            if param.requires_grad:
                if param.ndim == 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

        # 各オプティマイザを個別に作成
        muon_optimizer = torch.optim.Muon(
            muon_params,
            lr=self.settings.training.learning_rate,
            weight_decay=self.settings.training.weight_decay,
        )
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=self.settings.training.learning_rate,
            weight_decay=self.settings.training.weight_decay,
        )

        # 各オプティマイザに対応するスケジューラを作成
        muon_scheduler = get_constant_schedule_with_warmup(
            muon_optimizer, num_warmup_steps=self.settings.training.warmup_steps
        )
        adamw_scheduler = get_constant_schedule_with_warmup(
            adamw_optimizer, num_warmup_steps=self.settings.training.warmup_steps
        )

        model, muon_optimizer, adamw_optimizer, muon_scheduler, adamw_scheduler = (
            accelerator.prepare(
                model, muon_optimizer, adamw_optimizer, muon_scheduler, adamw_scheduler
            )
        )

        return (
            model,
            [muon_optimizer, adamw_optimizer],
            [muon_scheduler, adamw_scheduler],
        )
