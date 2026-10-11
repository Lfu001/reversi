"""Value-preserving, pickle-free runtime artifacts and strict decoding."""

import importlib
import json
import math

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file


def codec():
    assert importlib.util.find_spec("lfu_tree_self_play.runtime_codec") is not None
    return importlib.import_module("lfu_tree_self_play.runtime_codec")


def test_preserves_nested_types_and_independent_tensor_values(tmp_path):
    shared = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state = {
        0: (None, True, -2, "text", -0.0, float("inf"), float("-inf"), float("nan")),
        "0": [shared.T, shared[:, 1:], shared, shared, torch.tensor(2), torch.empty(0)],
        "array": np.arange(4, dtype=np.uint32),
        "device": torch.device("cpu"),
    }
    before = shared.clone()
    codec().save_runtime(state, tmp_path)
    restored = codec().load_runtime(tmp_path)
    assert list(restored) == [0, "0", "array", "device"]
    assert type(restored[0]) is tuple and type(restored["0"]) is list
    assert restored[0][:4] == (None, True, -2, "text")
    assert math.copysign(1, restored[0][4]) == -1
    assert restored[0][5:7] == (float("inf"), float("-inf"))
    assert math.isnan(restored[0][7])
    for expected, actual in zip(state["0"], restored["0"], strict=True):
        assert actual.device.type == "cpu"
        assert expected.dtype == actual.dtype and torch.equal(expected, actual)
    assert restored["array"].dtype == np.uint32
    assert np.array_equal(restored["array"], [0, 1, 2, 3])
    assert restored["device"] == torch.device("cpu")
    assert torch.equal(shared, before)
    first = {
        name: (tmp_path / name).read_bytes()
        for name in ("runtime.json", "runtime.safetensors")
    }
    codec().save_runtime(state, tmp_path)
    assert all((tmp_path / name).read_bytes() == data for name, data in first.items())
    assert len(load_file(tmp_path / "runtime.safetensors")) == 7


@pytest.mark.parametrize(
    "value", [object(), {True: 1}, np.array([object()]), torch.ones(2).to_sparse()]
)
def test_rejects_unsupported_values(tmp_path, value):
    with pytest.raises(ValueError):
        codec().save_runtime(value, tmp_path)


def test_rejects_cycles(tmp_path):
    state = []
    state.append(state)
    with pytest.raises(ValueError):
        codec().save_runtime(state, tmp_path)


@pytest.mark.parametrize(
    "state",
    [
        {"type": "int", "value": True},
        {"type": "bool", "value": 1},
        {"type": "none", "extra": 1},
        {"type": "float", "value": "Infinity"},
        {"type": "device", "value": "bogus"},
        {"type": "unknown"},
        {
            "type": "dict",
            "items": [[{"type": "int", "value": 0}, {"type": "none"}]] * 2,
        },
        {"type": "tensor", "name": "missing"},
    ],
)
def test_rejects_malformed_tagged_metadata(tmp_path, state):
    save_file({}, tmp_path / "runtime.safetensors")
    (tmp_path / "runtime.json").write_text(json.dumps({"version": 1, "state": state}))
    with pytest.raises(ValueError):
        codec().load_runtime(tmp_path)


@pytest.mark.parametrize(
    "metadata",
    [
        '{"version":1,"version":1,"state":{"type":"none"}}',
        '{"version":1,"state":{"type":"float","value":NaN}}',
        '{"version":true,"state":{"type":"none"}}',
        '{"version":2,"state":{"type":"none"}}',
        '{"version":1,"state":{"type":"none"},"extra":0}',
    ],
)
def test_rejects_invalid_json_envelope(tmp_path, metadata):
    save_file({}, tmp_path / "runtime.safetensors")
    (tmp_path / "runtime.json").write_text(metadata)
    with pytest.raises(ValueError):
        codec().load_runtime(tmp_path)


def test_rejects_unreferenced_and_duplicate_tensor_payloads(tmp_path):
    codec().save_runtime([torch.tensor(1), torch.tensor(2)], tmp_path)
    metadata = json.loads((tmp_path / "runtime.json").read_text())
    metadata["state"]["items"][1] = metadata["state"]["items"][0]
    (tmp_path / "runtime.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        codec().load_runtime(tmp_path)
    codec().save_runtime(None, tmp_path)
    save_file({"unexpected": torch.tensor(1)}, tmp_path / "runtime.safetensors")
    with pytest.raises(ValueError):
        codec().load_runtime(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("dtype", "object"), ("dtype", "int32"), ("shape", [True]), ("shape", [3])],
)
def test_rejects_numpy_dtype_and_shape_mismatches(tmp_path, field, value):
    codec().save_runtime(np.arange(2, dtype=np.uint32), tmp_path)
    metadata = json.loads((tmp_path / "runtime.json").read_text())
    metadata["state"][field] = value
    (tmp_path / "runtime.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        codec().load_runtime(tmp_path)


def test_normalizes_corrupt_safetensors_to_value_error(tmp_path):
    codec().save_runtime(torch.tensor(1), tmp_path)
    (tmp_path / "runtime.safetensors").write_bytes(b"broken")
    with pytest.raises(ValueError):
        codec().load_runtime(tmp_path)


def test_rejects_excessively_deep_trees_before_writing(tmp_path):
    state = None
    for _ in range(150):
        state = [state]
    with pytest.raises(ValueError):
        codec().save_runtime(state, tmp_path)
    assert not list(tmp_path.iterdir())
    node = {"type": "none"}
    for _ in range(150):
        node = {"type": "list", "items": [node]}
    save_file({}, tmp_path / "runtime.safetensors")
    (tmp_path / "runtime.json").write_text(json.dumps({"version": 1, "state": node}))
    with pytest.raises(ValueError):
        codec().load_runtime(tmp_path)


def test_dictionary_order_does_not_change_runtime_artifact_identity(tmp_path):
    first = {"z": {1: torch.tensor(1), "1": torch.tensor(2)}, -1: (3, 4)}
    second = {-1: (3, 4), "z": {"1": torch.tensor(2), 1: torch.tensor(1)}}
    codec().save_runtime(first, tmp_path)
    expected = {
        name: (tmp_path / name).read_bytes()
        for name in ("runtime.json", "runtime.safetensors")
    }
    codec().save_runtime(second, tmp_path)
    assert all(
        (tmp_path / name).read_bytes() == data for name, data in expected.items()
    )
