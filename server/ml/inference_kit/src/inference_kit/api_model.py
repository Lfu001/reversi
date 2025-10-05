from enum import Enum
from typing import Optional, Union

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


class TableState(BaseModel):
    """
    Current state of the game table

    Parameters:
        board: List of 64 elements representing the game board (8x8)
        turn: Current player's turn
        puttable_positions: List of positions where the current player can place a disk
    """

    board: list[Optional[DiskColor]]
    turn: DiskColor
    puttable_positions: list[Position]

    @field_validator("board")
    @classmethod
    def validate_board_size(cls, v: list[str]) -> list[str]:
        """
        Validates the board size.

        Args:
            v: The board to validate.

        Returns:
            The validated board.
        """
        if len(v) != 64:
            raise ValueError("Board must have exactly 64 elements (8x8)")
        return v


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
