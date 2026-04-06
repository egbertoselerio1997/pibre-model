"""Reusable simulation helpers for configuration and artifact persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import load_json_file, save_json_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_repo_root(repo_root: str | Path | None = None) -> Path:
	"""Return the repository root or a caller-provided override."""

	if repo_root is None:
		return REPO_ROOT

	return Path(repo_root).resolve()


def load_paths_config(repo_root: str | Path | None = None) -> dict[str, Any]:
	"""Load the repository path configuration."""

	root = get_repo_root(repo_root)
	return load_json_file(root / "config" / "paths.json")


def load_params_config(repo_root: str | Path | None = None) -> dict[str, Any]:
	"""Load the repository parameter configuration."""

	root = get_repo_root(repo_root)
	return load_json_file(root / "config" / "params.json")


def load_ml_orchestration_params(repo_root: str | Path | None = None) -> dict[str, Any]:
	"""Load the shared notebook orchestration parameters."""

	params = load_params_config(repo_root)
	if "ml_orchestration" not in params:
		raise KeyError("Machine-learning orchestration parameters not found.")

	return params["ml_orchestration"]


def load_model_params(model_name: str, repo_root: str | Path | None = None) -> dict[str, Any]:
	"""Load the parameter namespace for a specific model."""

	params = load_params_config(repo_root)
	if model_name not in params:
		raise KeyError(f"Model parameters not found for '{model_name}'.")

	return params[model_name]


def make_simulation_timestamp(timestamp: str | None = None) -> str:
	"""Create a timestamp matching the configured simulation artifact patterns."""

	if timestamp is not None:
		return timestamp

	return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_artifact_component(value: str) -> str:
	return Path(str(value).strip().replace("\\", "/")).as_posix().strip("/")


def resolve_notebook_tabular_group_dir(
	artifact_group: str,
	*,
	repo_root: str | Path | None = None,
	paths_config: Mapping[str, Any] | None = None,
) -> Path:
	"""Resolve the configured directory for one notebook tabular artifact group."""

	root = get_repo_root(repo_root)
	config = dict(paths_config) if paths_config is not None else load_paths_config(root)
	return root / Path(config["notebook_tabular_results_dir"]) / Path(_normalize_artifact_component(artifact_group))


def resolve_notebook_plot_group_dir(
	artifact_group: str,
	*,
	repo_root: str | Path | None = None,
	paths_config: Mapping[str, Any] | None = None,
) -> Path:
	"""Resolve the configured directory for one notebook plot artifact group."""

	root = get_repo_root(repo_root)
	config = dict(paths_config) if paths_config is not None else load_paths_config(root)
	return root / Path(config["notebook_plot_results_dir"]) / Path(_normalize_artifact_component(artifact_group))


def render_notebook_tabular_artifact_path(
	artifact_group: str,
	artifact_name: str,
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
	paths_config: Mapping[str, Any] | None = None,
) -> Path:
	"""Resolve one configured notebook tabular artifact path."""

	root = get_repo_root(repo_root)
	config = dict(paths_config) if paths_config is not None else load_paths_config(root)
	date_time = make_simulation_timestamp(timestamp)
	resolved_pattern = config["notebook_tabular_artifact_pattern"].format(
		artifact_group=_normalize_artifact_component(artifact_group),
		artifact_name=_normalize_artifact_component(artifact_name),
		date_time=date_time,
	)
	return root / Path(resolved_pattern)


def render_notebook_plot_artifact_path(
	artifact_group: str,
	artifact_name: str,
	*,
	extension: str,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
	paths_config: Mapping[str, Any] | None = None,
) -> Path:
	"""Resolve one configured notebook plot artifact path."""

	root = get_repo_root(repo_root)
	config = dict(paths_config) if paths_config is not None else load_paths_config(root)
	date_time = make_simulation_timestamp(timestamp)
	resolved_pattern = config["notebook_plot_artifact_pattern"].format(
		artifact_group=_normalize_artifact_component(artifact_group),
		artifact_name=_normalize_artifact_component(artifact_name),
		date_time=date_time,
		extension=str(extension).lstrip("."),
	)
	return root / Path(resolved_pattern)


def render_simulation_artifact_paths(
	simulation_name: str,
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
	paths_config: Mapping[str, Any] | None = None,
	data_pattern_key: str = "simulation_data_pattern",
	metadata_pattern_key: str = "simulation_metadata_pattern",
) -> tuple[Path, Path, str]:
	"""Resolve the configured dataset and metadata artifact paths."""

	root = get_repo_root(repo_root)
	config = dict(paths_config) if paths_config is not None else load_paths_config(root)
	date_time = make_simulation_timestamp(timestamp)

	dataset_relative = Path(
		config[data_pattern_key].format(
			simulation_name=simulation_name,
			date_time=date_time,
		)
	)
	metadata_relative = Path(
		config[metadata_pattern_key].format(
			simulation_name=simulation_name,
			date_time=date_time,
		)
	)

	return root / dataset_relative, root / metadata_relative, dataset_relative.as_posix()


def save_simulation_artifacts(
	dataset: pd.DataFrame,
	metadata: Mapping[str, Any],
	simulation_name: str,
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
	paths_config: Mapping[str, Any] | None = None,
	data_pattern_key: str = "simulation_data_pattern",
	metadata_pattern_key: str = "simulation_metadata_pattern",
) -> tuple[Path, Path, dict[str, Any]]:
	"""Persist a simulation dataset and matching metadata contract."""

	dataset_path, metadata_path, dataset_relative = render_simulation_artifact_paths(
		simulation_name,
		repo_root=repo_root,
		timestamp=timestamp,
		paths_config=paths_config,
		data_pattern_key=data_pattern_key,
		metadata_pattern_key=metadata_pattern_key,
	)

	dataset_path.parent.mkdir(parents=True, exist_ok=True)
	dataset.to_csv(dataset_path, index=False)

	persisted_metadata = dict(metadata)
	persisted_metadata["dataset_file"] = dataset_relative
	save_json_file(metadata_path, persisted_metadata)

	return dataset_path, metadata_path, persisted_metadata


__all__ = [
	"get_repo_root",
	"load_ml_orchestration_params",
	"load_model_params",
	"load_params_config",
	"load_paths_config",
	"make_simulation_timestamp",
	"render_notebook_plot_artifact_path",
	"render_notebook_tabular_artifact_path",
	"render_simulation_artifact_paths",
	"resolve_notebook_plot_group_dir",
	"resolve_notebook_tabular_group_dir",
	"save_simulation_artifacts",
]