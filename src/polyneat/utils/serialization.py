from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_as_json(data: dict[str, Any], file_path: Path) -> None:
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_from_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def save_as_pickle(obj: Any, file_path: Path) -> None:
    file_path.write_bytes(pickle.dumps(obj))


def load_from_pickle(file_path: Path) -> Any:
    return pickle.loads(file_path.read_bytes())
