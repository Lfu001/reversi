from typing import Any, TypeAlias, TypeVar

import numpy as np

# Type variables
T = TypeVar("T", bound=np.generic)

# Type aliases
NDArray: TypeAlias = np.ndarray[Any, Any]
BoardState: TypeAlias = NDArray  # Shape: (batch, 4, 8, 8)
Dones: TypeAlias = NDArray  # Shape: (batch,)
ActionProbs: TypeAlias = NDArray  # Shape: (batch, 8, 8)

# Game result
GameResult: TypeAlias = tuple[BoardState, Dones]
