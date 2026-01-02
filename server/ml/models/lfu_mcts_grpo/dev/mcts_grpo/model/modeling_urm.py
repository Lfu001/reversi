"""
URMModel: Universal Reasoning Model for Reversi.

Based on: https://arxiv.org/html/2512.14693v3

A Universal Transformer with:
- ConvSwiGLU FFN (SwiGLU + depthwise convolution)
- Shared-weight looped blocks
- Truncated Backpropagation Through Loops (TBPTL)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_urm import URMConfig

__all__ = ["URMModel"]


class ConvSwiGLU(nn.Module):
    """
    SwiGLU FFN with depthwise 1D convolution.

    Following the URM paper, the short convolution is applied after
    the SwiGLU gating to strengthen nonlinearity.

    Architecture:
        x -> gate_up_proj -> split -> [silu(gate) * up] -> dwconv -> down_proj
    """

    def __init__(self, config: URMConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        conv_kernel_size = config.conv_kernel_size

        # Fused gate and up projection (2x intermediate for split)
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)

        # Depthwise 1D convolution (groups = intermediate_size for depthwise)
        self.dwconv = nn.Conv1d(
            in_channels=intermediate_size,
            out_channels=intermediate_size,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,  # Causal-ish padding
            groups=intermediate_size,
            bias=True,
        )

        # Down projection back to hidden size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        # Project to 2x intermediate size
        gate_up = self.gate_up_proj(x)  # [batch, seq, inter*2]

        # Split and apply SwiGLU activation
        gate, up = gate_up.chunk(2, dim=-1)  # [batch, seq, inter] each
        # SwiGLU: silu(gate) * up
        gated = F.silu(gate) * up  # [batch, seq, inter]

        # Apply depthwise convolution + activation
        # Conv1d expects [batch, channels, seq]
        conv_in = gated.transpose(1, 2)  # [batch, inter, seq]
        conv_out = self.dwconv(conv_in)
        # Trim to original sequence length (due to padding)
        conv_out = conv_out[:, :, : x.size(1)]
        conv_out = F.silu(conv_out)
        conv_out = conv_out.transpose(1, 2)  # [batch, seq, inter]

        # Project back to hidden size
        return self.down_proj(conv_out)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Pre-computes and caches cos/sin values for efficient RoPE application.
    """

    def __init__(self, dim: int, max_seq_len: int = 64, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin values for the given sequence length.

        Returns:
            cos: [seq_len, dim]
            sin: [seq_len, dim]
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    Returns:
        Rotated q and k tensors with same shape
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # Expand cos/sin for batch and heads: [seq, dim] -> [1, 1, seq, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class URMAttention(nn.Module):
    """
    Multi-head self-attention with RoPE and PyTorch SDPA.

    Uses torch.nn.functional.scaled_dot_product_attention for efficient
    Flash Attention without external dependencies.
    """

    def __init__(self, config: URMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        # QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=64)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask (not commonly needed for full attention)
        Returns:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, head_dim]

        # Apply RoPE
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with PyTorch SDPA (Flash Attention when available)
        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,  # Full attention for board game, not causal
        )
        # Shape: [batch, num_heads, seq_len, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)


class URMBlock(nn.Module):
    """
    Single URM block: Attention + ConvSwiGLU with residual connections.

    Pre-norm architecture (RMSNorm/LayerNorm before each sub-layer).
    """

    def __init__(self, config: URMConfig):
        super().__init__()
        self.self_attn = URMAttention(config)
        self.mlp = ConvSwiGLU(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
        Returns:
            [batch, seq_len, hidden_size]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class URMBackbone(nn.Module):
    """
    Universal Transformer backbone with shared-weight looped blocks.

    Implements the inner loop recurrence with Truncated Backpropagation
    Through Loops (TBPTL): gradients are only computed for the later loops.
    """

    def __init__(self, config: URMConfig):
        super().__init__()
        self.config = config

        # Shared transformer blocks (weight sharing across loops)
        self.layers = nn.ModuleList(
            [URMBlock(config) for _ in range(config.num_layers)]
        )

        # Final layer norm
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
        Returns:
            [batch, seq_len, hidden_size]
        """
        # Note: Position encoding is handled by RoPE in the attention layer

        # Inner loop recurrence with TBPTL
        for loop_idx in range(self.config.num_inner_loops):
            # Apply all transformer layers
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)

            # TBPTL: Detach gradients AFTER truncated loops complete
            # This allows gradients to flow to input, but stops gradients from
            # later loops (which we optimize) from flowing to earlier loops
            if self.training and loop_idx < self.config.truncation_steps:
                hidden_states = hidden_states.detach()

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class URMModel(PreTrainedModel):
    """
    Universal Reasoning Model for Reversi.

    Takes board state input [batch, 4, 8, 8] and outputs:
    - policy_logits: [batch, 8, 8] - move probabilities (before softmax)
    - value: [batch] - position evaluation in [-1, 1]
    """

    config_class = URMConfig

    def __init__(self, config: URMConfig):
        super().__init__(config)

        # Input embedding: project [4] channels to hidden_size
        self.input_embed = nn.Linear(config.input_channels, config.hidden_size)

        # Transformer backbone
        self.backbone = URMBackbone(config)

        # Policy head: predict move probabilities
        self.policy_head = nn.Linear(config.hidden_size, 1)

        # Value head: predict position evaluation
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

        # Initialize weights
        self.post_init()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Board state [batch, 4, 8, 8]
               Channel 0: Current player's disks
               Channel 1: Opponent's disks
               Channel 2: Turn indicator
               Channel 3: Legal moves

        Returns:
            policy_logits: [batch, 8, 8]
            value: [batch]
        """
        batch_size = x.shape[0]

        # Flatten spatial dimensions: [batch, 4, 8, 8] -> [batch, 64, 4]
        x = x.view(batch_size, self.config.input_channels, -1)  # [batch, 4, 64]
        x = x.transpose(1, 2)  # [batch, 64, 4]

        # Project to hidden dimension
        hidden_states = self.input_embed(x)  # [batch, 64, hidden_size]

        # Apply transformer backbone
        hidden_states = self.backbone(hidden_states)  # [batch, 64, hidden_size]

        # Policy head: per-position logits
        policy_logits = self.policy_head(hidden_states)  # [batch, 64, 1]
        policy_logits = policy_logits.squeeze(-1)  # [batch, 64]
        policy_logits = policy_logits.view(batch_size, 8, 8)  # [batch, 8, 8]

        # Value head: aggregate and predict
        # Use mean pooling over sequence
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
        value = self.value_head(pooled)  # [batch, 1]
        value = torch.tanh(value.squeeze(-1))  # [batch]

        return policy_logits, value
