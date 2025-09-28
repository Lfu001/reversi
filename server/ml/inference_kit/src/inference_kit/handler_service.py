import importlib
import os
import sys

from inference_kit.api_model import PredictionOutput, TableState
from inference_kit.base_inference_handler import BaseInferenceHandler

USER_MODULE_NAME = "inference"
USER_HANDLER = "InferenceHandler"


class HandlerService:
    """
    Service class responsible for handling inference operations.
    """

    def __init__(self):
        model_dir = os.environ.get("MODEL_DIR")
        if not model_dir:
            model_dir = os.getcwd()

        if not os.path.isdir(model_dir):
            raise ImportError(
                f"Could not find a valid model directory at '{model_dir}'. "
                "Please set the MODEL_DIR environment variable or run from the model's root directory."
            )

        # Add the parent directory of the model to the Python search path
        models_root_dir = os.path.dirname(model_dir)
        sys.path.insert(0, models_root_dir)

        model_name = os.path.basename(model_dir)
        module_to_import = f"{model_name}.{USER_MODULE_NAME}"

        try:
            # Import the module as part of a package (e.g., "lfu_random.inference")
            inference_module = importlib.import_module(module_to_import)
            handler_cls = getattr(inference_module, USER_HANDLER)
            if not issubclass(handler_cls, BaseInferenceHandler):
                raise TypeError(
                    f"{USER_HANDLER} must be a subclass of BaseInferenceHandler."
                )
            self.handler = handler_cls()
            self.model = self.handler.load_model()
        finally:
            # Clean up the path to avoid affecting other modules
            if sys.path[0] == models_root_dir:
                sys.path.pop(0)

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
