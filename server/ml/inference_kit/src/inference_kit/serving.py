from fastapi import FastAPI, Response, status

from .api_model import PredictionOutput, TableState
from .handler_service import HandlerService

app = FastAPI()
handler_service = HandlerService()


@app.get("/health")
def health():
    return Response(status_code=status.HTTP_200_OK)


@app.post("/invocations", response_model=PredictionOutput)
def invocations(data: TableState):
    return handler_service.transform(data)
