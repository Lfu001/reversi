from typing import Any

import numpy as np
from typing_extensions import TypeAlias

NDArray: TypeAlias = np.ndarray[Any, Any]
BoardState: TypeAlias = NDArray  # Shape: (batch, 4, 8, 8)
Dones: TypeAlias = NDArray  # Shape: (batch,)

class ReversiEnvironment:
    def __init__(self, batch_size: int) -> None: ...
    def reset(self) -> BoardState: ...
    def get_state(self) -> BoardState: ...
    def step_batch(
        self,
        actions: NDArray,  # Shape: (batch, 8, 8)
    ) -> tuple[BoardState, Dones]: ...
    @property
    def batch_size(self) -> int: ...
