"""Reusable JSON input-output helpers for repository configuration and artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


_TIMESTAMPED_STEM_PATTERN = re.compile(r"^(?P<artifact_name>.+)_(?P<timestamp>\d{8}_\d{6})$")


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


def load_pickle_file(path: str | Path) -> Any:
    """Load a pickled Python object from disk."""

    file_path = Path(path)
    with file_path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle_file(path: str | Path, data: Any) -> Path:
    """Persist a Python object using pickle and create parent directories when needed."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as handle:
        pickle.dump(data, handle)
    return file_path


def save_dataframe_csv(path: str | Path, data: Any, *, index: bool = True) -> Path:
    """Persist a tabular artifact as CSV and create parent directories when needed."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        frame = data.to_frame()
    else:
        frame = pd.DataFrame(data)
    frame.to_csv(file_path, index=index)
    return file_path


def load_dataframe_csv(path: str | Path, *, index_col: int | str | None = None) -> pd.DataFrame:
    """Load one CSV artifact into a dataframe."""

    return pd.read_csv(Path(path), index_col=index_col)


def save_matplotlib_figure(
    path: str | Path,
    figure: Any,
    *,
    dpi: int = 140,
    bbox_inches: str = "tight",
) -> Path:
    """Persist one matplotlib figure and create parent directories when needed."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
    return file_path


def compute_file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 digest for one file path."""

    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def split_timestamped_stem(stem: str) -> tuple[str, str]:
    """Split an artifact stem of the form name_YYYYMMDD_HHMMSS."""

    match = _TIMESTAMPED_STEM_PATTERN.match(str(stem))
    if match is None:
        raise ValueError(f"Stem '{stem}' does not end with a supported timestamp tag.")
    return str(match.group("artifact_name")), str(match.group("timestamp"))


def build_timestamped_file_index(
    directory: str | Path,
    *,
    suffixes: Sequence[str] | None = None,
    recursive: bool = True,
) -> dict[str, dict[str, Path]]:
    """Index timestamp-tagged files by timestamp and artifact key."""

    root_directory = Path(directory)
    if not root_directory.exists():
        return {}

    normalized_suffixes = None
    if suffixes is not None:
        normalized_suffixes = {str(suffix).lower() for suffix in suffixes}

    indexed_files: dict[str, dict[str, Path]] = defaultdict(dict)
    candidate_paths = root_directory.rglob("*") if recursive else root_directory.glob("*")
    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            continue
        if normalized_suffixes is not None and candidate_path.suffix.lower() not in normalized_suffixes:
            continue

        try:
            artifact_name, timestamp = split_timestamped_stem(candidate_path.stem)
        except ValueError:
            continue

        relative_path = candidate_path.relative_to(root_directory)
        relative_parent = relative_path.parent
        if str(relative_parent) == ".":
            artifact_key = artifact_name
        else:
            artifact_key = (relative_parent / artifact_name).as_posix()
        indexed_files[timestamp][artifact_key] = candidate_path

    return {
        timestamp: dict(sorted(artifact_paths.items()))
        for timestamp, artifact_paths in sorted(indexed_files.items())
    }


def select_latest_timestamped_file_bundle(
    directory: str | Path,
    *,
    required_artifact_keys: Sequence[str] | None = None,
    suffixes: Sequence[str] | None = None,
    recursive: bool = True,
) -> tuple[str, dict[str, Path]]:
    """Return the newest timestamp-tagged file bundle matching the requested artifact keys."""

    indexed_files = build_timestamped_file_index(
        directory,
        suffixes=suffixes,
        recursive=recursive,
    )
    if not indexed_files:
        raise FileNotFoundError(f"No timestamped artifacts were found in {directory}.")

    resolved_required_keys = {str(artifact_key) for artifact_key in (required_artifact_keys or [])}
    for timestamp in sorted(indexed_files.keys(), reverse=True):
        artifact_bundle = indexed_files[timestamp]
        if resolved_required_keys.issubset(artifact_bundle.keys()):
            return timestamp, artifact_bundle

    missing_display = ", ".join(sorted(resolved_required_keys))
    raise FileNotFoundError(
        f"No timestamped artifact bundle in {directory} contains the required keys: {missing_display}."
    )


__all__ = [
    "build_timestamped_file_index",
    "compute_file_sha256",
    "load_dataframe_csv",
    "load_json_file",
    "load_pickle_file",
    "save_dataframe_csv",
    "save_json_file",
    "save_matplotlib_figure",
    "save_pickle_file",
    "select_latest_timestamped_file_bundle",
    "split_timestamped_stem",
]