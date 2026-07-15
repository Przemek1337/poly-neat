from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_as_json(data: dict[str, Any], file_path: Path) -> None:
    """Write ``data`` as indented UTF-8 JSON to ``file_path``."""
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_from_json(file_path: Path) -> dict[str, Any]:
    """Load a JSON file written by :func:`save_as_json`."""
    return json.loads(file_path.read_text(encoding="utf-8"))


def save_as_pickle(obj: Any, file_path: Path) -> None:
    """Pickle ``obj`` to ``file_path`` (convenience format; JSON is canonical)."""
    file_path.write_bytes(pickle.dumps(obj))


def load_from_pickle(file_path: Path) -> Any:
    """Unpickle a file written by :func:`save_as_pickle`. Trust the source."""
    return pickle.loads(file_path.read_bytes())
