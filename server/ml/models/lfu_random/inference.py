from random import randint
from typing import Any, Callable, override

from inference_kit.api_model import Position, SuggestedPosition, TableState
from inference_kit.base_inference_handler import BaseInferenceHandler


class InferenceHandler(BaseInferenceHandler):
    @override
    def load_model(self) -> Callable[..., Any]:
        return lambda x: x

    @override
    def prepare_input(self, table_state: TableState) -> TableState:
        return table_state

    @override
    def predict(self, data: TableState, model: Callable[..., Any]) -> TableState:
        return data

    @override
    def output(self, prediction: TableState) -> list[SuggestedPosition]:
        """
        Returns all possible moves with confidence scores.
        The randomly selected move gets confidence 1.0, others get 0.0.

        Args:
            prediction: The table state containing possible moves

        Returns:
            List of suggested positions with confidence scores
        """
        if not prediction.puttable_positions:
            return []

        selected_idx = randint(0, len(prediction.puttable_positions) - 1)

        return [
            SuggestedPosition(
                position=Position(row=pos.row, column=pos.column),
                confidence=1.0 if i == selected_idx else 0.0,
            )
            for i, pos in enumerate(prediction.puttable_positions)
        ]
