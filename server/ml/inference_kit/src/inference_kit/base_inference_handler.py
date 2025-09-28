from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

from .api_model import PredictionOutput, TableState


class BaseInferenceHandler(metaclass=ABCMeta):
    """
    Base class for handling inference-related operations. This class defines
    the necessary methods that any subclass should implement to perform
    inference tasks.
    """

    @abstractmethod
    def load_model(self) -> Callable[..., Any]:
        """
        Loads the machine learning model required for inference.
        Subclasses should implement the model loading logic.

        Returns:
            The loaded model object.
        """
        ...

    @abstractmethod
    def prepare_input(self, table_state: TableState) -> Any:
        """
        Prepares and processes the input data from the given board state
        for the inference model. Subclasses should implement the input preparation logic.

        Args:
            table_state: The current state of the game table.

        Returns:
            The processed input data.
        """
        ...

    @abstractmethod
    def predict(self, data: Any, model: Callable[..., Any]) -> Any:
        """
        Runs prediction on the processed input data using the loaded model.
        Subclasses should implement the prediction logic.

        Args:
            data: The processed input data.
            model: The loaded model object.

        Returns:
            The prediction result.
        """
        ...

    @abstractmethod
    def output(self, prediction: Any) -> PredictionOutput:
        """
        Processes and formats the prediction result to a desired output format.
        Subclasses should define the output transformation logic.

        Args:
            prediction: The prediction result.

        Returns:
            The formatted output.
        """
        ...
