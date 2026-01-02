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

        # Muonは2Dパラメータ（隠れ層の重み）に使用
        # 1Dパラメータ（バイアス、埋め込み、LayerNorm）はAdamWを使用
        muon_params = []
        adamw_params = []
        for param in model.parameters():
            if param.requires_grad:
                if param.ndim >= 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

        # MuonとAdamWを組み合わせたChainedOptimizer
        optimizer = ChainedOptimizer(
            torch.optim.Muon(
                muon_params,
                lr=self.settings.training.learning_rate,
                weight_decay=self.settings.training.weight_decay,
            ),
            torch.optim.AdamW(
                adamw_params,
                lr=self.settings.training.learning_rate,
                weight_decay=self.settings.training.weight_decay,
            ),
        )
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.settings.training.warmup_steps
        )
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
        return model, optimizer, lr_scheduler


class ChainedOptimizer(torch.optim.Optimizer):
    """複数のオプティマイザを1つとして扱うラッパー"""

    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = list(optimizers)
        # 全param_groupsを統合（LRScheduler互換性のため）
        param_groups = []
        for opt in self.optimizers:
            param_groups.extend(opt.param_groups)
        # Optimizer基底クラスの初期化をスキップし、必要な属性のみ設定
        self.param_groups = param_groups
        self.defaults = {}
        self.state: dict = {}

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for opt in self.optimizers:
            loss = opt.step(closure)
        return loss

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)
