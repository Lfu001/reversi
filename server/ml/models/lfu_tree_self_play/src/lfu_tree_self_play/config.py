"""Read experiment settings without imposing an experiment schema."""

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load TOML values, propagating file, decoding and syntax errors."""
    with Path(path).open("rb") as file:
        return tomllib.load(file)
