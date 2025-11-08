from transformers import PretrainedConfig


class ReversiZeroConfig(PretrainedConfig):
    model_type = "reversi_zero"

    def __init__(
        self,
        num_channels: int = 128,
        num_residual_blocks: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
