from .api_model import PredictionOutput, TableState


class BaseInferenceHandler:
    """
    Base class for handling inference-related operations. This class defines
    the necessary methods that any subclass should implement to perform
    inference tasks.
    """

    def load_model(self):
        """
        Loads the machine learning model required for inference.
        Subclasses should implement the model loading logic.

        Returns:
            The loaded model object.
        """
        raise NotImplementedError("Please implement load_model()")

    def prepare_input(self, table_state: TableState):
        """
        Prepares and processes the input data from the given board state
        for the inference model. Subclasses should implement the input preparation logic.

        Args:
            table_state: The current state of the game table.

        Returns:
            The processed input data.
        """
        raise NotImplementedError("Please implement prepare_input()")

    def predict(self, data, model):
        """
        Runs prediction on the processed input data using the loaded model.
        Subclasses should implement the prediction logic.

        Args:
            data: The processed input data.
            model: The loaded model object.

        Returns:
            The prediction result.
        """
        raise NotImplementedError("Please implement predict()")

    def output(self, prediction) -> PredictionOutput:
        """
        Processes and formats the prediction result to a desired output format.
        Subclasses should define the output transformation logic.

        Args:
            prediction: The prediction result.

        Returns:
            The formatted output.
        """
        raise NotImplementedError("Please implement output()")
