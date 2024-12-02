from fastapi.testclient import TestClient

from .serving import app

client = TestClient(app)


def test_invocations():
    # fmt: off
    board = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 2, 0, 0, 0,
            0, 0, 0, 2, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
    ]
    # fmt: on

    response = client.post(
        "/invocations",
        json={
            "board": board,
            "turn": 1,
            "puttable_positions": [20, 29, 34, 43],
        },
    )

    assert response.status_code == 200

    response_json = response.json()
    assert "row" in response_json and "column" in response_json

    row = response_json["row"]
    column = response_json["column"]
    assert (row, column) in [(2, 4), (3, 5), (4, 2), (5, 3)]
