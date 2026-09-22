import random
from dataclasses import dataclass

from numpy import ndarray


@dataclass
class Experience:
    state: ndarray
    pi: ndarray
    outcome: float
    q_values: ndarray  # Q-values for all actions at the root
    visit_counts: ndarray  # Visit counts for all actions at the root


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[Experience] = []
        self.position = 0

    def push(self, experience: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
