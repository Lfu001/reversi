"""
Unit tests for URMModel.

Tests:
- Configuration initialization and validation
- Forward pass shapes
- Gradient flow (including TBPTL)
- PreTrainedModel save/load
"""

import tempfile

import numpy as np
import pytest
import torch

from .configuration_urm import URMConfig
from .modeling_urm import (
    ConvSwiGLU,
    RotaryEmbedding,
    URMAttention,
    URMBlock,
    URMModel,
    apply_rotary_pos_emb,
)


class TestURMConfig:
    """Tests for URMConfig."""

    def test_default_config(self):
        """Test default configuration values match paper."""
        config = URMConfig()
        assert config.hidden_size == 512
        assert config.num_attention_heads == 8
        assert config.num_layers == 4
        assert config.num_inner_loops == 8
        assert config.truncation_steps == 2
        assert config.expansion_ratio == 4
        assert config.conv_kernel_size == 2
        assert config.intermediate_size == 1536  # 512 * 4 * 2/3 → round to 256

    def test_custom_config(self):
        """Test custom configuration."""
        config = URMConfig(hidden_size=256, num_layers=2, num_inner_loops=4)
        assert config.hidden_size == 256
        assert config.num_layers == 2
        assert config.num_inner_loops == 4

    def test_truncation_validation(self):
        """Test truncation_steps < num_inner_loops validation."""
        with pytest.raises(ValueError, match="truncation_steps"):
            URMConfig(num_inner_loops=4, truncation_steps=4)

        with pytest.raises(ValueError, match="truncation_steps"):
            URMConfig(num_inner_loops=4, truncation_steps=5)


class TestRotaryEmbedding:
    """Tests for RotaryEmbedding (RoPE) module."""

    def test_cos_sin_shapes(self):
        """Test cos/sin cache shapes are correct."""
        dim, max_seq = 64, 128
        rope = RotaryEmbedding(dim, max_seq_len=max_seq)

        seq_len = 64
        cos, sin = rope(seq_len)

        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

    def test_cache_expansion(self):
        """Test cache expands when seq_len exceeds max."""
        dim, max_seq = 64, 32
        rope = RotaryEmbedding(dim, max_seq_len=max_seq)

        # Request longer sequence
        cos, sin = rope(64)

        assert cos.shape == (64, dim)
        assert sin.shape == (64, dim)
        assert rope.max_seq_len == 64

    def test_values_bounded(self):
        """Test cos/sin values are in [-1, 1]."""
        rope = RotaryEmbedding(64, max_seq_len=64)
        cos, sin = rope(64)

        assert (cos >= -1).all() and (cos <= 1).all()
        assert (sin >= -1).all() and (sin <= 1).all()

    def test_different_positions_different_embeddings(self):
        """Test that different positions get different embeddings."""
        rope = RotaryEmbedding(64, max_seq_len=64)
        cos, sin = rope(4)

        # Position 0 and position 1 should have different embeddings
        assert not torch.allclose(cos[0], cos[1])
        assert not torch.allclose(sin[0], sin[1])


class TestApplyRotaryPosEmb:
    """Tests for apply_rotary_pos_emb function."""

    def test_output_shapes(self):
        """Test output shapes match input shapes."""
        batch, heads, seq, dim = 2, 8, 64, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)

        rope = RotaryEmbedding(dim, max_seq_len=seq)
        cos, sin = rope(seq)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_values(self):
        """Test that RoPE actually modifies the input tensors."""
        batch, heads, seq, dim = 2, 8, 64, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)

        rope = RotaryEmbedding(dim, max_seq_len=seq)
        cos, sin = rope(seq)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Rotated tensors should be different from original
        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)

    def test_gradient_flow(self):
        """Test gradients flow through RoPE."""
        batch, heads, seq, dim = 2, 4, 16, 32
        q = torch.randn(batch, heads, seq, dim, requires_grad=True)
        k = torch.randn(batch, heads, seq, dim, requires_grad=True)

        rope = RotaryEmbedding(dim, max_seq_len=seq)
        cos, sin = rope(seq)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()


class TestConvSwiGLU:
    """Tests for ConvSwiGLU module."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        config = URMConfig(hidden_size=128)
        module = ConvSwiGLU(config)

        batch, seq = 2, 64
        x = torch.randn(batch, seq, config.hidden_size)
        out = module(x)

        assert out.shape == (batch, seq, config.hidden_size)

    def test_gradient_flow(self):
        """Test gradients flow through the module."""
        config = URMConfig(hidden_size=128)
        module = ConvSwiGLU(config)

        x = torch.randn(2, 64, config.hidden_size, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestURMAttention:
    """Tests for URMAttention module."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        config = URMConfig(hidden_size=128, num_attention_heads=4)
        module = URMAttention(config)

        batch, seq = 2, 64
        x = torch.randn(batch, seq, config.hidden_size)
        out = module(x)

        assert out.shape == (batch, seq, config.hidden_size)

    def test_head_dim_validation(self):
        """Test hidden_size must be divisible by num_attention_heads."""
        with pytest.raises(ValueError, match="divisible"):
            URMAttention(URMConfig(hidden_size=100, num_attention_heads=8))


class TestURMBlock:
    """Tests for URMBlock module."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        config = URMConfig(hidden_size=128, num_attention_heads=4)
        module = URMBlock(config)

        batch, seq = 2, 64
        x = torch.randn(batch, seq, config.hidden_size)
        out = module(x)

        assert out.shape == (batch, seq, config.hidden_size)


class TestURMModel:
    """Tests for URMModel."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return URMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_layers=1,
            num_inner_loops=4,
            truncation_steps=1,
        )

    def test_forward_shape(self, small_config):
        """Test output shapes for policy and value."""
        model = URMModel(small_config)
        batch = 4
        x = torch.randn(batch, 4, 8, 8)

        policy, value = model(x)

        assert policy.shape == (batch, 8, 8)
        assert value.shape == (batch,)

    def test_value_range(self, small_config):
        """Test value output is in [-1, 1] (tanh)."""
        model = URMModel(small_config)
        x = torch.randn(4, 4, 8, 8)

        _, value = model(x)

        assert (value >= -1).all()
        assert (value <= 1).all()

    def test_gradient_flow(self, small_config):
        """Test gradients flow through model (in eval mode, no TBPTL)."""
        model = URMModel(small_config)
        model.eval()  # TBPTL only active in training mode
        x = torch.randn(2, 4, 8, 8, requires_grad=True)

        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_tbptl_skips_gradient_for_early_loops(self, small_config):
        """Test TBPTL: early loops run without gradient tracking."""
        model = URMModel(small_config)
        model.train()

        x = torch.randn(2, 4, 8, 8)
        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()

        # Verify backbone layers have gradients (from non-truncated loops)
        # Note: input_embed won't have gradients because truncated loops break the chain
        for layer in model.backbone.layers:
            for param in layer.parameters():
                if param.requires_grad:
                    assert param.grad is not None, (
                        "Backbone layer params should have gradients"
                    )
                    assert not torch.isnan(param.grad).any(), (
                        "Gradients should not be NaN"
                    )

        # Verify policy/value heads have gradients
        for param in model.policy_head.parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    "Policy head params should have gradients"
                )

        for param in model.value_head.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Value head params should have gradients"

    def test_tbptl_input_embed_receives_gradients(self, small_config):
        """Test that input_embed receives gradients during training.

        With the fixed TBPTL implementation, input_embeddings are added at each
        loop iteration, including the gradient-enabled loops. This preserves
        the gradient flow to input_embed.
        """
        model = URMModel(small_config)
        model.train()

        x = torch.randn(2, 4, 8, 8)
        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()

        # input_embed should receive gradients for the model to learn properly
        for param in model.input_embed.parameters():
            if param.requires_grad:
                assert param.grad is not None, "input_embed should receive gradients"

    def test_save_load_pretrained(self, small_config):
        """Test PreTrainedModel save and load functionality."""
        model = URMModel(small_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_pretrained(tmpdir)

            # Load
            loaded_model = URMModel.from_pretrained(tmpdir)

            # Compare outputs
            x = torch.randn(2, 4, 8, 8)
            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                policy1, value1 = model(x)
                policy2, value2 = loaded_model(x)

            assert torch.allclose(policy1, policy2)
            assert torch.allclose(value1, value2)

    def test_config_saved_correctly(self, small_config):
        """Test config is saved and loaded correctly."""
        model = URMModel(small_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded_model = URMModel.from_pretrained(tmpdir)

            assert loaded_model.config.hidden_size == small_config.hidden_size
            assert loaded_model.config.num_layers == small_config.num_layers
            assert loaded_model.config.num_inner_loops == small_config.num_inner_loops

    def test_compatible_with_reversi_input(self):
        """Test model works with typical Reversi board input format."""
        config = URMConfig(hidden_size=64, num_attention_heads=4, num_layers=1)
        model = URMModel(config)

        # Standard Reversi input: [black_disks, white_disks, turn, legal_moves]
        batch = 4
        states = np.zeros((batch, 4, 8, 8), dtype=np.float32)

        # Initial position
        states[:, 0, 3, 3] = 1.0  # Black
        states[:, 0, 4, 4] = 1.0
        states[:, 1, 3, 4] = 1.0  # White
        states[:, 1, 4, 3] = 1.0
        states[:, 2, :, :] = 1.0  # Black's turn
        states[:, 3, 2, 3] = 1.0  # Legal moves
        states[:, 3, 3, 2] = 1.0
        states[:, 3, 4, 5] = 1.0
        states[:, 3, 5, 4] = 1.0

        x = torch.from_numpy(states)
        policy, value = model(x)

        assert policy.shape == (batch, 8, 8)
        assert value.shape == (batch,)
        assert not torch.isnan(policy).any()
        assert not torch.isnan(value).any()
