from typing import List

from pydantic import BaseModel, Field


class TableState(BaseModel):
    """
    Current state of the game table

    Parameters:
        board (List[int]): The current state of the board
        turn (int): The current player
        puttable_positions (List[int]): The positions that the current player can put
    """

    board: List[int]
    turn: int
    puttable_positions: List[int]


class PredictionOutput(BaseModel):
    """
    Prediction result
    """

    row: int = Field(ge=0, le=7, description="Row of the predicted position")
    column: int = Field(ge=0, le=7, description="Column of the predicted position")
