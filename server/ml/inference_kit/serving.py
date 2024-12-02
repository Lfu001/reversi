from fastapi import FastAPI

from .api_model import PredictionOutput, TableState
from .handler_service import HandlerService

app = FastAPI()
handler_service = HandlerService()


@app.post("/invocations", response_model=PredictionOutput)
def invocations(data: TableState):
    return handler_service.transform(data)
