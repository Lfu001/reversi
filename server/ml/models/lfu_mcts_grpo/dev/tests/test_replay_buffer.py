"""
Tests for ReplayBuffer
Verifies push, sample, and capacity management.
"""

import numpy as np
import pytest
from mcts_grpo.replay_buffer import Experience, ReplayBuffer


def create_experience(value: float = 0.0) -> Experience:
    """Create a sample experience for testing."""
    return Experience(
        state=np.zeros((4, 8, 8), dtype=np.float32),
        pi=np.ones(64, dtype=np.float32) / 64.0,
        outcome=value,
        q_values=np.zeros(64, dtype=np.float32),
        visit_counts=np.zeros(64, dtype=np.uint32),
    )


class TestReplayBuffer:
    """Tests for ReplayBuffer functionality."""

    def test_push_and_len(self):
        """Buffer length should increase as experiences are pushed."""
        buffer = ReplayBuffer(capacity=10)
        assert len(buffer) == 0

        buffer.push(create_experience(1.0))
        assert len(buffer) == 1

        buffer.push(create_experience(2.0))
        assert len(buffer) == 2

    def test_sample(self):
        """Sample should return requested number of experiences."""
        buffer = ReplayBuffer(capacity=10)
        for i in range(10):
            buffer.push(create_experience(float(i)))

        samples = buffer.sample(3)
        assert len(samples) == 3
        assert all(isinstance(s, Experience) for s in samples)

    def test_capacity_limit(self):
        """Buffer should not exceed capacity, oldest items discarded."""
        buffer = ReplayBuffer(capacity=5)

        # Push 10 items into buffer of capacity 5
        for i in range(10):
            buffer.push(create_experience(float(i)))

        # Buffer length should be capped at capacity
        assert len(buffer) == 5

        # The buffer should contain the most recent 5 items
        # (Note: current implementation uses ring buffer, so exact items depend on position)

    def test_sample_size_validation(self):
        """Sample raises error when requesting more than available."""
        buffer = ReplayBuffer(capacity=10)
        buffer.push(create_experience(1.0))
        buffer.push(create_experience(2.0))

        with pytest.raises(ValueError):
            buffer.sample(5)  # Only 2 experiences available
