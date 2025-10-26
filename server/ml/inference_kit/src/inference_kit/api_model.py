from enum import Enum
from typing import Union

import numpy as np
from pydantic import BaseModel, Field, field_validator


class RowLabel(Enum):
    ONE = "One"
    TWO = "Two"
    THREE = "Three"
    FOUR = "Four"
    FIVE = "Five"
    SIX = "Six"
    SEVEN = "Seven"
    EIGHT = "Eight"


class ColumnLabel(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"


# Mapping from string labels to numeric indices
ROW_LABEL_TO_IDX = {row.value: i for i, row in enumerate(RowLabel)}
COL_LABEL_TO_IDX = {col.value: i for i, col in enumerate(ColumnLabel)}

# Reverse mappings for conversion back to strings
ROW_IDX_TO_LABEL = {i: row.value for i, row in enumerate(RowLabel)}
COL_IDX_TO_LABEL = {i: col.value for i, col in enumerate(ColumnLabel)}


class Position(BaseModel):
    """
    Position on the game board using string labels (e.g., "One", "A")
    """

    row: Union[str, int] = Field(
        ..., description="Row label (e.g., 'One', 'Two') or index (0-7)"
    )
    column: Union[str, int] = Field(
        ..., description="Column label (e.g., 'A', 'B') or index (0-7)"
    )

    # Internal numeric indices (not part of the model)
    _row_idx: int = 0
    _col_idx: int = 0

    @field_validator("row", mode="before")
    @classmethod
    def validate_row(cls, v: Union[str, int]) -> Union[str, int]:
        """
        Validates the row label or index.

        If the row is an integer, it must be between 0 and 7.
        If the row is a string, it must be one of the valid row labels (e.g., "One", "Two").

        Args:
            v: The row label or index to validate.

        Returns:
            The validated row label or index.
        """
        if isinstance(v, int):
            if 0 <= v <= 7:
                return v
            raise ValueError("Row index must be between 0 and 7")
        if isinstance(v, str) and v in ROW_LABEL_TO_IDX:
            return v
        raise ValueError(f"Invalid row label: {v}")

    @field_validator("column", mode="before")
    @classmethod
    def validate_column(cls, v: Union[str, int]) -> Union[str, int]:
        """
        Validates the column label or index.

        If the column is an integer, it must be between 0 and 7.
        If the column is a string, it must be one of the valid column labels (e.g., "A", "B").

        Args:
            v: The column label or index to validate.

        Returns:
            The validated column label or index.
        """
        if isinstance(v, int):
            if 0 <= v <= 7:
                return v
            raise ValueError("Column index must be between 0 and 7")
        if isinstance(v, str) and v.upper() in COL_LABEL_TO_IDX:
            return v.upper()
        raise ValueError(f"Invalid column label: {v}")

    def __init__(self, **data):
        super().__init__(**data)
        # Convert string labels to numeric indices for internal use
        self._row_idx = (
            self.row if isinstance(self.row, int) else ROW_LABEL_TO_IDX.get(self.row, 0)
        )
        self._col_idx = (
            self.column
            if isinstance(self.column, int)
            else COL_LABEL_TO_IDX.get(str(self.column).upper(), 0)
        )

    @property
    def row_idx(self) -> int:
        """Get the numeric row index (0-7)"""
        return self._row_idx

    @property
    def col_idx(self) -> int:
        """Get the numeric column index (0-7)"""
        return self._col_idx

    @property
    def to_tuple(self) -> tuple[int, int]:
        """Return the position as a (row, column) tuple of indices"""
        return (self._row_idx, self._col_idx)


class DiskColor(Enum):
    LIGHT = "Light"
    DARK = "Dark"


class Bitboard(BaseModel):
    """
    A bitboard representation of the Reversi game board.

    Attributes:
        dark_plane: Bitmask for dark disks (1 for dark disk, 0 otherwise)
        light_plane: Bitmask for light disks (1 for light disk, 0 otherwise)
    """

    dark_plane: int
    light_plane: int

    def to_numpy(self) -> np.ndarray:
        """Convert the bitboard to a numpy array with shape (2, 8, 8).

        The array contains two channels:
        - Channel 0: Dark disks (1.0 if disk exists, 0.0 otherwise)
        - Channel 1: Light disks (1.0 if disk exists, 0.0 otherwise)

        Returns:
            np.ndarray: A 3D numpy array with shape (2, 8, 8)
        """
        board = np.zeros((2, 8, 8), dtype=np.float32)

        for i in range(64):
            mask = 1 << i
            row = i // 8
            col = i % 8

            if self.dark_plane & mask:
                board[0, row, col] = 1.0
            elif self.light_plane & mask:
                board[1, row, col] = 1.0

        return board


class TableState(BaseModel):
    """
    Current state of the game table

    Parameters:
        board: Bitboard representing the game board (8x8)
        turn: Current player's turn
        puttable_positions: List of positions where the current player can place a disk
    """

    board: Bitboard
    turn: DiskColor
    puttable_positions: list[Position]

    def to_numpy(self) -> np.ndarray:
        """
        Convert the table state to a numpy array with shape (1, 4, 8, 8).

        The 4 channels are:
        - Channel 0: Positions of the dark disks (1.0 if disk exists, 0.0 otherwise)
        - Channel 1: Positions of the light disks (1.0 if disk exists, 0.0 otherwise)
        - Channel 2: Current turn (1.0 for Dark, 0.0 for Light)
        - Channel 3: Legal moves (1.0 if move is legal, 0.0 otherwise)

        Returns:
            np.ndarray: A 4D numpy array with shape (1, 4, 8, 8)
        """
        state = np.zeros((1, 4, 8, 8), dtype=np.float32)

        # Get board as a numpy array (2, 8, 8) and copy to the output
        state[0, :2] = self.board.to_numpy()  # Dark and light disks

        # Set turn plane (1.0 for Dark, 0.0 for Light)
        if self.turn == DiskColor.DARK:
            state[0, 2, :, :] = 1.0

        # Set legal moves plane
        for pos in self.puttable_positions:
            state[0, 3, pos.row, pos.col] = 1.0

        return state


class SuggestedPosition(BaseModel):
    """
    A single suggested position with its confidence score.
    """

    position: Position
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the suggestion (0.0 to 1.0)",
    )


class ModelResponse(BaseModel):
    """
    Response model for the suggestion API.
    This matches the Rust ModelResponse struct.
    """

    positions: list[SuggestedPosition] = Field(
        ..., description="List of suggested positions with confidence scores"
    )


class PredictionOutput(BaseModel):
    """
    Prediction result with row and column
    """

    row: int = Field(..., ge=0, le=7, description="Row of the predicted position (0-7)")
    column: int = Field(
        ..., ge=0, le=7, description="Column of the predicted position (0-7)"
    )
