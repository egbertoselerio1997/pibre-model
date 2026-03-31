"""Reusable analysis helpers for model-comparison sweeps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .process import DatasetSplit, SupervisedDatasetFrames, make_train_test_split
from .simulation import load_ml_orchestration_params, load_params_config


ModelRunner = Callable[..., dict[str, Any]]


def describe_and_display_table(
	title: str,
	description: str,
	table: Any,
) -> Any:
	"""Print a short description before displaying one notebook table."""

	print(title)
	print(description)
	from IPython.display import display as ipython_display

	ipython_display(table)
	return table

_DEFAULT_ANALYSIS_SETTINGS = {
	"min_total_samples": 100,
	"max_total_samples": 10000,
	"total_sample_step": 330,
	"n_repeats": 30,
	"test_fraction": 0.2,
	"random_seed": 42,
}

_DEFAULT_COBRE_RESPONSE_SURFACE_SETTINGS = {
	"grid_points_per_axis": 49,
	"contour_levels": 18,
	"operational_extension_fraction": 0.5,
	"fixed_influent_profile": "midpoint",
}


def load_analysis_defaults(repo_root: str | Path | None = None) -> dict[str, int | float]:
	"""Load configurable sweep defaults for notebook analysis runs."""

	orchestration_params = load_ml_orchestration_params(repo_root)
	analysis_params = dict(orchestration_params.get("analysis", {}))
	hyperparameters = dict(orchestration_params.get("hyperparameters", {}))

	return {
		"min_total_samples": int(analysis_params.get("min_total_samples", _DEFAULT_ANALYSIS_SETTINGS["min_total_samples"])),
		"max_total_samples": int(analysis_params.get("max_total_samples", _DEFAULT_ANALYSIS_SETTINGS["max_total_samples"])),
		"total_sample_step": int(analysis_params.get("total_sample_step", _DEFAULT_ANALYSIS_SETTINGS["total_sample_step"])),
		"n_repeats": int(analysis_params.get("n_repeats", _DEFAULT_ANALYSIS_SETTINGS["n_repeats"])),
		"test_fraction": float(analysis_params.get("test_fraction", hyperparameters.get("test_fraction", _DEFAULT_ANALYSIS_SETTINGS["test_fraction"]))),
		"random_seed": int(hyperparameters.get("random_seed", _DEFAULT_ANALYSIS_SETTINGS["random_seed"])),
	}


def load_cobre_response_surface_defaults(repo_root: str | Path | None = None) -> dict[str, int | float | str]:
	"""Load configurable defaults for the COBRE operational response-surface study."""

	orchestration_params = load_ml_orchestration_params(repo_root)
	analysis_params = dict(orchestration_params.get("analysis", {}))
	response_surface_params = dict(analysis_params.get("cobre_response_surface", {}))

	return {
		"grid_points_per_axis": int(
			response_surface_params.get(
				"grid_points_per_axis",
				_DEFAULT_COBRE_RESPONSE_SURFACE_SETTINGS["grid_points_per_axis"],
			)
		),
		"contour_levels": int(
			response_surface_params.get(
				"contour_levels",
				_DEFAULT_COBRE_RESPONSE_SURFACE_SETTINGS["contour_levels"],
			)
		),
		"operational_extension_fraction": float(
			response_surface_params.get(
				"operational_extension_fraction",
				_DEFAULT_COBRE_RESPONSE_SURFACE_SETTINGS["operational_extension_fraction"],
			)
		),
		"fixed_influent_profile": str(
			response_surface_params.get(
				"fixed_influent_profile",
				_DEFAULT_COBRE_RESPONSE_SURFACE_SETTINGS["fixed_influent_profile"],
			)
		),
	}


def _resolve_response_surface_metadata(
	metadata: Mapping[str, Any] | None,
	*,
	repo_root: str | Path | None = None,
) -> dict[str, list[str]]:
	params = load_params_config(repo_root)
	simulation_params = dict(params["asm2d_tcn_simulation"])
	workbook_params = dict(simulation_params.get("workbook", {}))
	resolved_metadata = dict(metadata or {})

	operational_columns = list(
		resolved_metadata.get("operational_columns", simulation_params.get("operational_columns", []))
	)
	state_columns = list(
		resolved_metadata.get(
			"state_columns",
			workbook_params.get("state_columns", list(dict(simulation_params.get("influent_state_ranges", {})).keys())),
		)
	)
	measured_output_columns = list(
		resolved_metadata.get("measured_output_columns", simulation_params.get("measured_output_columns", []))
	)

	if "HRT" not in operational_columns or "Aeration" not in operational_columns:
		raise ValueError("The COBRE response surface requires HRT and Aeration operational columns.")
	if not state_columns:
		raise ValueError("At least one influent state column is required to build a COBRE response surface.")
	if not measured_output_columns:
		raise ValueError("At least one measured output column is required to build a COBRE response surface.")

	return {
		"operational_columns": operational_columns,
		"state_columns": state_columns,
		"measured_output_columns": measured_output_columns,
	}


def _build_midpoint_influent_profile(
	state_columns: list[str],
	*,
	repo_root: str | Path | None = None,
) -> dict[str, float]:
	params = load_params_config(repo_root)
	influent_ranges = dict(params["asm2d_tcn_simulation"]["influent_state_ranges"])
	profile: dict[str, float] = {}

	for state_column in state_columns:
		if state_column not in influent_ranges:
			raise KeyError(f"Influent state range not found for '{state_column}'.")
		lower_bound, upper_bound = influent_ranges[state_column]
		profile[state_column] = 0.5 * (float(lower_bound) + float(upper_bound))

	return profile


def _resolve_fixed_influent_profile(
	state_columns: list[str],
	fixed_influent_profile: str | Mapping[str, Any] | None,
	*,
	repo_root: str | Path | None = None,
) -> dict[str, float]:
	if fixed_influent_profile is None or fixed_influent_profile == "midpoint":
		return _build_midpoint_influent_profile(state_columns, repo_root=repo_root)

	if isinstance(fixed_influent_profile, Mapping):
		resolved_profile: dict[str, float] = {}
		for state_column in state_columns:
			if state_column not in fixed_influent_profile:
				raise KeyError(f"Fixed influent profile is missing '{state_column}'.")
			resolved_profile[state_column] = float(fixed_influent_profile[state_column])
		return resolved_profile

	raise ValueError("fixed_influent_profile must be 'midpoint', None, or a mapping of state values.")


def _resolve_operational_domain(
	operational_columns: list[str],
	*,
	operational_extension_fraction: float,
	repo_root: str | Path | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
	"""Resolve the trained and extrapolated operating domains while preserving non-negative controls."""

	if operational_extension_fraction < 0.0:
		raise ValueError("operational_extension_fraction must be greater than or equal to 0.")

	params = load_params_config(repo_root)
	operational_ranges = dict(params["asm2d_tcn_simulation"]["operational_ranges"])
	training_domain: dict[str, dict[str, float]] = {}
	extended_domain: dict[str, dict[str, float]] = {}

	for column_name in operational_columns:
		if column_name not in operational_ranges:
			raise KeyError(f"Operational range not found for '{column_name}'.")
		lower_bound, upper_bound = operational_ranges[column_name]
		lower_value = float(lower_bound)
		upper_value = float(upper_bound)
		width = upper_value - lower_value
		extension = float(operational_extension_fraction) * width
		training_domain[column_name] = {"min": lower_value, "max": upper_value}
		extended_domain[column_name] = {
			"min": max(0.0, lower_value - extension),
			"max": upper_value + extension,
		}

	return training_domain, extended_domain


def build_cobre_response_surface_prediction_data(
	model_path: str | Path,
	*,
	metadata: Mapping[str, Any] | None = None,
	repo_root: str | Path | None = None,
	grid_points_per_axis: int | None = None,
	operational_extension_fraction: float | None = None,
	fixed_influent_profile: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	"""Build a COBRE prediction grid over HRT and Aeration with a fixed influent profile."""

	from src.models.ml import predict_cobre_model

	defaults = load_cobre_response_surface_defaults(repo_root)
	selected_grid_points = int(
		grid_points_per_axis
		if grid_points_per_axis is not None
		else defaults["grid_points_per_axis"]
	)
	selected_extension_fraction = float(
		operational_extension_fraction
		if operational_extension_fraction is not None
		else defaults["operational_extension_fraction"]
	)
	selected_profile = (
		fixed_influent_profile
		if fixed_influent_profile is not None
		else defaults["fixed_influent_profile"]
	)

	if selected_grid_points < 2:
		raise ValueError("grid_points_per_axis must be at least 2.")

	resolved_metadata = _resolve_response_surface_metadata(metadata, repo_root=repo_root)
	operational_columns = list(resolved_metadata["operational_columns"])
	state_columns = list(resolved_metadata["state_columns"])
	training_domain, extended_domain = _resolve_operational_domain(
		operational_columns,
		operational_extension_fraction=selected_extension_fraction,
		repo_root=repo_root,
	)
	resolved_influent_profile = _resolve_fixed_influent_profile(
		state_columns,
		selected_profile,
		repo_root=repo_root,
	)

	hrt_axis = np.linspace(
		extended_domain["HRT"]["min"],
		extended_domain["HRT"]["max"],
		selected_grid_points,
		dtype=float,
	)
	aeration_axis = np.linspace(
		extended_domain["Aeration"]["min"],
		extended_domain["Aeration"]["max"],
		selected_grid_points,
		dtype=float,
	)
	hrt_mesh, aeration_mesh = np.meshgrid(hrt_axis, aeration_axis)
	row_count = int(hrt_mesh.size)

	feature_columns = {
		"HRT": hrt_mesh.reshape(-1),
		"Aeration": aeration_mesh.reshape(-1),
	}
	for state_column in state_columns:
		feature_columns[f"In_{state_column}"] = np.full(
			row_count,
			resolved_influent_profile[state_column],
			dtype=float,
		)
	feature_frame = pd.DataFrame(feature_columns)
	constraint_reference = pd.DataFrame(
		{
			state_column: np.full(row_count, resolved_influent_profile[state_column], dtype=float)
			for state_column in state_columns
		}
	)

	prediction_result = predict_cobre_model(
		{
			"features": feature_frame,
			"constraint_reference": constraint_reference,
		},
		model_path,
	)
	projected_predictions = prediction_result["projected_predictions"].copy()
	per_target_surfaces = {
		target_name: projected_predictions[target_name].to_numpy(dtype=float).reshape(hrt_mesh.shape)
		for target_name in projected_predictions.columns
	}
	prediction_table = pd.concat(
		[
			feature_frame,
			constraint_reference.add_prefix("ConstraintReference_"),
			projected_predictions.add_prefix("Projected_"),
		],
		axis=1,
	)

	return {
		"response_surface_config": {
			"grid_points_per_axis": selected_grid_points,
			"operational_extension_fraction": selected_extension_fraction,
			"fixed_influent_profile": "explicit" if isinstance(selected_profile, Mapping) else str(selected_profile),
		},
		"fixed_influent_profile": pd.Series(resolved_influent_profile, name="value"),
		"operational_axes": {
			"HRT": hrt_axis,
			"Aeration": aeration_axis,
		},
		"operational_meshes": {
			"HRT": hrt_mesh,
			"Aeration": aeration_mesh,
		},
		"training_domain": training_domain,
		"extended_domain": extended_domain,
		"feature_grid": feature_frame,
		"constraint_reference": constraint_reference,
		"projected_predictions": projected_predictions,
		"prediction_table": prediction_table,
		"per_target_surfaces": per_target_surfaces,
	}


def build_dataset_size_schedule(
	total_available_samples: int,
	*,
	min_total_samples: int,
	max_total_samples: int,
	total_sample_step: int,
) -> list[int]:
	"""Build an inclusive dataset-size schedule capped by the available rows."""

	if total_available_samples < 2:
		raise ValueError("At least two samples are required to build a train-test analysis schedule.")
	if min_total_samples < 2:
		raise ValueError("min_total_samples must be at least 2.")
	if max_total_samples < min_total_samples:
		raise ValueError("max_total_samples must be greater than or equal to min_total_samples.")
	if total_sample_step <= 0:
		raise ValueError("total_sample_step must be greater than 0.")

	usable_max = min(int(max_total_samples), int(total_available_samples))
	if usable_max < min_total_samples:
		raise ValueError("The available dataset is smaller than the requested minimum analysis size.")

	schedule = list(range(int(min_total_samples), usable_max + 1, int(total_sample_step)))
	if schedule[-1] != usable_max:
		schedule.append(usable_max)

	return schedule


def _validate_split_fraction(test_fraction: float) -> None:
	if not 0.0 < test_fraction < 1.0:
		raise ValueError("test_fraction must be between 0 and 1.")


def _validate_repetition_count(n_repeats: int) -> None:
	if n_repeats <= 0:
		raise ValueError("n_repeats must be greater than 0.")


def _ensure_split_feasibility(sample_size: int, test_fraction: float) -> None:
	test_size = int(np.ceil(sample_size * test_fraction))
	train_size = sample_size - test_size
	if train_size <= 0 or test_size <= 0:
		raise ValueError(
			"The selected dataset size and test_fraction do not leave at least one sample in both train and test splits."
		)


def _sample_supervised_dataset(
	dataset: SupervisedDatasetFrames | DatasetSplit,
	*,
	sample_size: int,
	random_seed: int,
) -> DatasetSplit:
	if sample_size > len(dataset.features):
		raise ValueError("sample_size cannot be greater than the available dataset length.")

	random_generator = np.random.default_rng(int(random_seed))
	sampled_indices = random_generator.choice(dataset.features.index.to_numpy(), size=int(sample_size), replace=False)
	sampled_index = pd.Index(sampled_indices)

	return DatasetSplit(
		features=dataset.features.loc[sampled_index].copy(),
		targets=dataset.targets.loc[sampled_index].copy(),
		constraint_reference=dataset.constraint_reference.loc[sampled_index].copy(),
	)


def _insert_metadata_columns(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> pd.DataFrame:
	result = frame.copy()
	for column_name, value in reversed(list(metadata.items())):
		result.insert(0, column_name, value)
	return result.reset_index(drop=True)


def _normalize_aggregate_metrics(
	report: Mapping[str, pd.DataFrame],
	*,
	split_name: str,
	metadata: Mapping[str, Any],
) -> pd.DataFrame:
	frame = report["aggregate_metrics"].copy()
	frame.insert(0, "split_name", split_name)
	return _insert_metadata_columns(frame, metadata)


def _normalize_per_target_metrics(
	report: Mapping[str, pd.DataFrame],
	*,
	split_name: str,
	metadata: Mapping[str, Any],
) -> pd.DataFrame:
	frame = report["per_target_metrics"].copy()
	frame.insert(0, "split_name", split_name)
	return _insert_metadata_columns(frame, metadata)


def _build_prediction_table(
	dataset_split: DatasetSplit,
	report: Mapping[str, pd.DataFrame],
	*,
	split_name: str,
	metadata: Mapping[str, Any],
) -> pd.DataFrame:
	frame_parts = [
		dataset_split.targets.add_prefix("Actual_"),
		report["raw_predictions"],
		report["projected_predictions"],
		dataset_split.constraint_reference.add_prefix("ConstraintReference_"),
		report["constraint_residuals"],
	]
	projection_diagnostics = report.get("projection_diagnostics")
	if projection_diagnostics is not None:
		frame_parts.append(projection_diagnostics)

	prediction_frame = pd.concat(frame_parts, axis=1)
	prediction_frame.insert(0, "sample_index", prediction_frame.index)
	prediction_frame.insert(1, "split_name", split_name)
	return _insert_metadata_columns(prediction_frame, metadata)


def _resolve_run_metadata(
	*,
	model_name: str,
	dataset_size_total: int,
	repeat_index: int,
	train_size: int,
	test_size: int,
	run_seed: int,
) -> dict[str, Any]:
	return {
		"model_name": model_name,
		"dataset_size_total": int(dataset_size_total),
		"repeat_index": int(repeat_index),
		"train_size": int(train_size),
		"test_size": int(test_size),
		"run_seed": int(run_seed),
	}


def run_model_dataset_size_analysis(
	model_name: str,
	supervised_dataset: SupervisedDatasetFrames | DatasetSplit,
	A_matrix: np.ndarray,
	model_runner: ModelRunner,
	*,
	model_params: Mapping[str, Any] | None = None,
	model_hyperparameters: Mapping[str, Any] | None = None,
	repo_root: str | Path | None = None,
	min_total_samples: int | None = None,
	max_total_samples: int | None = None,
	total_sample_step: int | None = None,
	n_repeats: int | None = None,
	test_fraction: float | None = None,
	random_seed: int | None = None,
	show_progress: bool = True,
	show_runner_progress: bool | None = None,
	persist_artifacts: bool = False,
	include_prediction_tables: bool = True,
	extra_runner_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	"""Run a configurable train-test sweep across dataset sizes for one model."""

	defaults = load_analysis_defaults(repo_root)
	selected_min_total_samples = int(min_total_samples if min_total_samples is not None else defaults["min_total_samples"])
	selected_max_total_samples = int(max_total_samples if max_total_samples is not None else defaults["max_total_samples"])
	selected_total_sample_step = int(total_sample_step if total_sample_step is not None else defaults["total_sample_step"])
	selected_n_repeats = int(n_repeats if n_repeats is not None else defaults["n_repeats"])
	selected_test_fraction = float(test_fraction if test_fraction is not None else defaults["test_fraction"])
	selected_random_seed = int(random_seed if random_seed is not None else defaults["random_seed"])
	selected_show_runner_progress = show_progress if show_runner_progress is None else bool(show_runner_progress)

	_validate_split_fraction(selected_test_fraction)
	_validate_repetition_count(selected_n_repeats)
	dataset_sizes = build_dataset_size_schedule(
		len(supervised_dataset.features),
		min_total_samples=selected_min_total_samples,
		max_total_samples=selected_max_total_samples,
		total_sample_step=selected_total_sample_step,
	)
	for dataset_size_total in dataset_sizes:
		_ensure_split_feasibility(int(dataset_size_total), selected_test_fraction)

	analysis_config = {
		"min_total_samples": selected_min_total_samples,
		"max_total_samples": selected_max_total_samples,
		"total_sample_step": selected_total_sample_step,
		"n_repeats": selected_n_repeats,
		"test_fraction": selected_test_fraction,
		"random_seed": selected_random_seed,
	}

	aggregate_frames: list[pd.DataFrame] = []
	per_target_frames: list[pd.DataFrame] = []
	prediction_tables: list[pd.DataFrame] = []
	run_rows: list[dict[str, Any]] = []
	progress_bar = tqdm(
		total=len(dataset_sizes) * selected_n_repeats,
		desc=f"Analyze {model_name}",
		unit="run",
		disable=not show_progress,
	)

	try:
		for dataset_size_index, dataset_size_total in enumerate(dataset_sizes):
			for repeat_index in range(selected_n_repeats):
				run_seed = selected_random_seed + dataset_size_index * selected_n_repeats + repeat_index
				sampled_dataset = _sample_supervised_dataset(
					supervised_dataset,
					sample_size=int(dataset_size_total),
					random_seed=run_seed,
				)
				dataset_splits = make_train_test_split(
					sampled_dataset,
					test_fraction=selected_test_fraction,
					random_seed=run_seed,
				)
				runner_kwargs = {
					"repo_root": repo_root,
					"model_params": dict(model_params) if model_params is not None else None,
					"model_hyperparameters": dict(model_hyperparameters) if model_hyperparameters is not None else None,
					"persist_artifacts": persist_artifacts,
					"show_progress": selected_show_runner_progress,
				}
				if extra_runner_kwargs is not None:
					runner_kwargs.update(dict(extra_runner_kwargs))
				clean_runner_kwargs = {key: value for key, value in runner_kwargs.items() if value is not None}
				result = model_runner(
					dataset_splits.train,
					dataset_splits.test,
					A_matrix,
					**clean_runner_kwargs,
				)

				run_metadata = _resolve_run_metadata(
					model_name=model_name,
					dataset_size_total=int(dataset_size_total),
					repeat_index=repeat_index,
					train_size=len(dataset_splits.train.features),
					test_size=len(dataset_splits.test.features),
					run_seed=run_seed,
				)
				aggregate_frames.append(
					_normalize_aggregate_metrics(result["train_report"], split_name="train", metadata=run_metadata)
				)
				aggregate_frames.append(
					_normalize_aggregate_metrics(result["test_report"], split_name="test", metadata=run_metadata)
				)
				per_target_frames.append(
					_normalize_per_target_metrics(result["train_report"], split_name="train", metadata=run_metadata)
				)
				per_target_frames.append(
					_normalize_per_target_metrics(result["test_report"], split_name="test", metadata=run_metadata)
				)
				if include_prediction_tables:
					prediction_tables.append(
						_build_prediction_table(
							dataset_splits.train,
							result["train_report"],
							split_name="train",
							metadata=run_metadata,
						)
					)
					prediction_tables.append(
						_build_prediction_table(
							dataset_splits.test,
							result["test_report"],
							split_name="test",
							metadata=run_metadata,
						)
					)

				artifact_paths = result.get("artifact_paths", {})
				run_rows.append(
					{
						**run_metadata,
						"artifact_model_bundle_path": None if artifact_paths.get("model_bundle") is None else str(artifact_paths["model_bundle"]),
						"artifact_metrics_path": None if artifact_paths.get("metrics") is None else str(artifact_paths["metrics"]),
						"artifact_optuna_path": None if artifact_paths.get("optuna") is None else str(artifact_paths["optuna"]),
					}
				)
				progress_bar.update(1)
				progress_bar.set_postfix(size=int(dataset_size_total), repeat=repeat_index + 1)
	finally:
		progress_bar.close()

	return {
		"analysis_config": analysis_config,
		"dataset_sizes": dataset_sizes,
		"run_metadata": pd.DataFrame(run_rows),
		"aggregate_metrics": pd.concat(aggregate_frames, ignore_index=True),
		"per_target_metrics": pd.concat(per_target_frames, ignore_index=True),
		"prediction_tables": prediction_tables,
	}


__all__ = [
	"ModelRunner",
	"build_cobre_response_surface_prediction_data",
	"build_dataset_size_schedule",
	"describe_and_display_table",
	"load_cobre_response_surface_defaults",
	"load_analysis_defaults",
	"run_model_dataset_size_analysis",
]