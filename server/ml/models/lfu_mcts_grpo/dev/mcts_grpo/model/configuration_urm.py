"""
URMConfig: Configuration class for the Universal Reasoning Model.

Based on: https://arxiv.org/abs/2512.14693v3
"""

from transformers import PretrainedConfig

__all__ = ["URMConfig"]


class URMConfig(PretrainedConfig):
    """
    Configuration for URM (Universal Reasoning Model).

    This is a Universal Transformer architecture with:
    - Shared-weight blocks (looped)
    - ConvSwiGLU FFN (SwiGLU + depthwise convolution)
    - Truncated Backpropagation Through Loops (TBPTL)

    Args:
        hidden_size: Hidden dimension (default: 512 per paper)
        num_attention_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (shared) (default: 4)
        num_inner_loops: Number of inner loop iterations (default: 8)
        truncation_steps: Forward-only steps for TBPTL (default: 2)
        expansion_ratio: MLP expansion ratio (default: 4, uses 2/3 scaling per official code)
        conv_kernel_size: Depthwise convolution kernel size (default: 2)
        dropout: Dropout rate (default: 0.0)
        layer_norm_eps: Layer normalization epsilon (default: 1e-5)
        input_channels: Number of input channels for board state (default: 4)
    """

    model_type = "urm"

    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 4,
        num_inner_loops: int = 8,
        truncation_steps: int = 2,
        expansion_ratio: int = 4,
        conv_kernel_size: int = 2,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        input_channels: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if truncation_steps >= num_inner_loops:
            raise ValueError(
                f"truncation_steps ({truncation_steps}) must be less than "
                f"num_inner_loops ({num_inner_loops})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.num_inner_loops = num_inner_loops
        self.truncation_steps = truncation_steps
        self.expansion_ratio = expansion_ratio
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.input_channels = input_channels

    @property
    def intermediate_size(self) -> int:
        """FFN intermediate dimension (official code: expansion * hidden * 2/3, rounded to 256)."""
        # Official URM uses 2/3 scaling for SwiGLU parity, rounded to multiple of 256
        raw = int(self.hidden_size * self.expansion_ratio * 2 / 3)
        return ((raw + 255) // 256) * 256
