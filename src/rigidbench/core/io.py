from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write a JSON file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Atomically write a compressed NumPy archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
