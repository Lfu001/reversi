"""Explicit value trees in JSON; independently snapshotted tensors in safetensors."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from .dataset_codec import canonical_json

_TENSOR_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_ARRAY_DTYPES = {
    name: np.dtype(name)
    for name in (
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    )
}
_FLOAT_TOKENS = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}
_MAX_DEPTH = 128
_MAX_NODES = 100_000


def _snapshot(value):
    if (
        value.layout != torch.strided
        or value.is_meta
        or value.is_quantized
        or value.dtype not in _TENSOR_DTYPES
    ):
        raise ValueError("unsupported runtime tensor")
    # detach/to/contiguous alone may retain storage shared with the caller.
    return value.detach().cpu().resolve_conj().resolve_neg().contiguous().clone()


def save_runtime(state, directory):
    """Save a supported tree without coercing values or executing object hooks."""
    tensors = {}
    active = set()
    nodes = 0

    def encode(value, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > _MAX_DEPTH or nodes > _MAX_NODES:
            raise ValueError("runtime tree limit exceeded")
        kind = type(value)
        if value is None:
            return {"type": "none"}
        if kind in (bool, int, str, float):
            if kind is float and not math.isfinite(value):
                value = "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
            return {"type": kind.__name__, "value": value}
        if isinstance(value, torch.Tensor) or kind is np.ndarray:
            name = f"tensor_{len(tensors):08d}"
            if kind is np.ndarray:
                if value.dtype.name not in _ARRAY_DTYPES or not value.dtype.isnative:
                    raise ValueError("unsupported runtime array dtype")
                tensors[name] = _snapshot(torch.from_numpy(value.copy()))
                return {
                    "type": "ndarray",
                    "name": name,
                    "dtype": value.dtype.name,
                    "shape": list(value.shape),
                }
            tensors[name] = _snapshot(value)
            return {"type": "tensor", "name": name}
        if kind is torch.device:
            return {"type": "device", "value": str(value)}
        if kind not in (dict, list, tuple):
            raise ValueError("unsupported runtime value")
        if id(value) in active:
            raise ValueError("cyclic runtime tree")
        active.add(id(value))
        try:
            if kind is dict:
                if any(type(key) not in (int, str) for key in value):
                    raise ValueError("unsupported runtime dictionary key")
                items = [
                    [encode(key, depth + 1), encode(item, depth + 1)]
                    for key, item in sorted(
                        value.items(),
                        key=lambda entry: (type(entry[0]) is str, entry[0]),
                    )
                ]
            else:
                items = [encode(item, depth + 1) for item in value]
            return {"type": kind.__name__, "items": items}
        finally:
            active.remove(id(value))

    try:
        metadata = canonical_json({"version": 1, "state": encode(state)})
        directory = Path(directory)
        save_file(tensors, directory / "runtime.safetensors")
        (directory / "runtime.json").write_bytes(metadata)
    except (RecursionError, SafetensorError) as error:
        raise ValueError("invalid runtime state") from error


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate runtime JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite runtime JSON literal")


def load_runtime(directory):
    """Load on CPU, validating the entire metadata tree and payload reference set."""
    directory = Path(directory)
    try:
        envelope = json.loads(
            (directory / "runtime.json").read_bytes(),
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
        )
        if (
            type(envelope) is not dict
            or set(envelope) != {"version", "state"}
            or type(envelope["version"]) is not int
            or envelope["version"] != 1
        ):
            raise ValueError("runtime codec version or fields mismatch")
        tensors = load_file(directory / "runtime.safetensors", device="cpu")
        used = set()
        nodes = 0

        def decode(node, depth=0):
            nonlocal nodes
            nodes += 1
            if depth > _MAX_DEPTH or nodes > _MAX_NODES:
                raise ValueError("runtime tree limit exceeded")
            if type(node) is not dict or type(node.get("type")) is not str:
                raise ValueError("invalid runtime tag")
            tag = node["type"]
            fields = {
                "none": {"type"},
                "bool": {"type", "value"},
                "int": {"type", "value"},
                "str": {"type", "value"},
                "float": {"type", "value"},
                "device": {"type", "value"},
                "dict": {"type", "items"},
                "list": {"type", "items"},
                "tuple": {"type", "items"},
                "tensor": {"type", "name"},
                "ndarray": {"type", "name", "dtype", "shape"},
            }
            if tag not in fields or set(node) != fields[tag]:
                raise ValueError("runtime tag fields mismatch")
            if tag == "none":
                return None
            if tag in ("bool", "int", "str"):
                if (
                    type(node["value"])
                    is not {"bool": bool, "int": int, "str": str}[tag]
                ):
                    raise ValueError("runtime primitive type mismatch")
                return node["value"]
            if tag == "float":
                value = node["value"]
                if type(value) is float and math.isfinite(value):
                    return value
                if type(value) is str and value in _FLOAT_TOKENS:
                    return _FLOAT_TOKENS[value]
                raise ValueError("invalid runtime float")
            if tag == "device":
                if type(node["value"]) is not str:
                    raise ValueError("invalid runtime device")
                return torch.device(node["value"])
            if tag in ("tensor", "ndarray"):
                name = node["name"]
                if type(name) is not str or name not in tensors or name in used:
                    raise ValueError("invalid runtime tensor reference")
                used.add(name)
                tensor = tensors[name]
                if tensor.dtype not in _TENSOR_DTYPES:
                    raise ValueError("unsupported runtime tensor dtype")
                if tag == "tensor":
                    return tensor
                dtype, shape = node["dtype"], node["shape"]
                if (
                    type(dtype) is not str
                    or dtype not in _ARRAY_DTYPES
                    or type(shape) is not list
                    or any(type(n) is not int or n < 0 for n in shape)
                    or tuple(shape) != tuple(tensor.shape)
                ):
                    raise ValueError("invalid runtime array metadata")
                array = tensor.numpy()
                if array.dtype != _ARRAY_DTYPES[dtype]:
                    raise ValueError("runtime array dtype mismatch")
                return array.copy()
            items = node["items"]
            if type(items) is not list:
                raise ValueError("invalid runtime container")
            if tag == "dict":
                result = {}
                for pair in items:
                    if type(pair) is not list or len(pair) != 2:
                        raise ValueError("invalid runtime dictionary entry")
                    key = decode(pair[0], depth + 1)
                    if type(key) not in (int, str) or key in result:
                        raise ValueError("invalid runtime dictionary key")
                    result[key] = decode(pair[1], depth + 1)
                return result
            result = [decode(item, depth + 1) for item in items]
            return tuple(result) if tag == "tuple" else result

        state = decode(envelope["state"])
        if used != tensors.keys():
            raise ValueError("unreferenced runtime tensor payload")
        return state
    except (RecursionError, SafetensorError, RuntimeError, TypeError) as error:
        raise ValueError("invalid runtime artifact") from error
