from fastapi.testclient import TestClient
from inference_kit.serving import app

client = TestClient(app)


def test_invocations():
    data = {
        "board": {
            "dark_plane": 34628173824,
            "light_plane": 68853694464,
        },
        "turn": "Dark",
        "puttable_positions": [
            {"row": 2, "column": 4},
            {"row": 3, "column": 5},
            {"row": 4, "column": 2},
            {"row": 5, "column": 3},
        ],
    }

    response = client.post(
        "/invocations",
        json=data,
    )

    assert response.status_code == 200

    response_json = response.json()
    assert "positions" in response_json
    assert isinstance(response_json["positions"], list)
    assert len(response_json["positions"]) > 0

    # Check each suggested position
    valid_positions = [(2, 4), (3, 5), (4, 2), (5, 3)]
    for position in response_json["positions"]:
        assert "position" in position
        assert "row" in position["position"]
        assert "column" in position["position"]
        assert "confidence" in position
        assert 0 <= position["confidence"] <= 1.0

        row = position["position"]["row"]
        column = position["position"]["column"]
        assert (row, column) in valid_positions
