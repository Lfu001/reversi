from random import choice
from typing import Any, Callable, override

from inference_kit.api_model import PredictionOutput, TableState
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
    def output(self, prediction: TableState) -> PredictionOutput:
        pos_flat = choice(prediction.puttable_positions)
        return PredictionOutput(row=pos_flat // 8, column=pos_flat % 8)
