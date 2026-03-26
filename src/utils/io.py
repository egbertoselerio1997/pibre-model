"""Reusable JSON input-output helpers for repository configuration and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_file(path: str | Path, data: dict[str, Any], *, indent: int = 2) -> Path:
    """Persist a dictionary as JSON and create parent directories when needed."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)
        handle.write("\n")
    return file_path


__all__ = ["load_json_file", "save_json_file"]