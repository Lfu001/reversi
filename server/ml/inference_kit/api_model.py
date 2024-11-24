from pydantic import BaseModel, Field


class TableState(BaseModel):
    """
    Current state of the game table

    Parameters:
        board (list[int]): The current state of the board
        turn (int): The current player
        puttable_positions (list[int]): The positions that the current player can put
    """

    board: list[int]
    turn: int
    puttable_positions: list[int]


class PredictionOutput(BaseModel):
    """
    Prediction result
    """

    row: int = Field(ge=0, le=7, description="Row of the predicted position")
    column: int = Field(ge=0, le=7, description="Column of the predicted position")
