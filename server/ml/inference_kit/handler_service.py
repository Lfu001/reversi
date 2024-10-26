import importlib

from api_model import PredictionOutput, TableState

from .base_inference_handler import BaseInferenceHandler

USER_MODULE = "inference"
USER_HANDLER = "InferenceHandler"


class HandlerService:
    """
    Service class responsible for handling inference operations.
    """

    def __init__(self):
        inference_handler = importlib.import_module(USER_MODULE)
        handler = getattr(inference_handler, USER_HANDLER)
        if not issubclass(handler, BaseInferenceHandler):
            raise TypeError(
                f"{USER_HANDLER} must be a subclass of BaseInferenceHandler."
            )
        self.handler: BaseInferenceHandler = handler
        self.model = self.handler.load_model()

    def transform(self, table_state: TableState) -> PredictionOutput:
        """
        Transforms the table state into a prediction output using the inference handler.

        Args:
            table_state (TableState): The current state of the game table.

        Returns:
            PredictionOutput: The prediction result formatted as an output object.
        """
        data = self.handler.prepare_input(table_state)
        prediction = self.handler.predict(data, self.model)
        result = self.handler.output(prediction)

        return result
