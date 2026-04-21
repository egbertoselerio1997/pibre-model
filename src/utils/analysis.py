"""Reusable analysis helpers for model-comparison sweeps."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.models.simulation.asm2d_tsn_simulation import (
	get_asm2d_tsn_matrices,
	load_asm2d_tsn_simulation_params,
)

from .io import load_dataframe_csv, save_dataframe_csv, select_latest_timestamped_file_bundle
from .process import (
	DatasetSplit,
	SupervisedDatasetFrames,
	collapse_fractional_states_to_measured_outputs,
	make_train_test_split,
)
from .simulation import (
	load_ml_orchestration_params,
	load_params_config,
	make_simulation_timestamp,
	render_notebook_tabular_artifact_path,
	resolve_notebook_tabular_group_dir,
)


ModelRunner = Callable[..., dict[str, Any]]
PredictionFrameSpec = tuple[str, str, str]

DEFAULT_NEGATIVE_PREDICTION_FRAME_SPECS: tuple[PredictionFrameSpec, ...] = (
	("raw", "raw_predictions", "Raw_"),
	("affine", "affine_predictions", "Affine_"),
	("projected", "projected_predictions", "Projected_"),
)

icsor_NEGATIVE_PREDICTION_FAMILY_SPECS: dict[str, tuple[PredictionFrameSpec, ...]] = {
	"composite": DEFAULT_NEGATIVE_PREDICTION_FRAME_SPECS,
	"fractional": (
		("raw", "raw_fractional_predictions", "RawFractional_"),
		("affine", "affine_fractional_predictions", "AffineFractional_"),
		("projected", "projected_fractional_predictions", "ProjectedFractional_"),
	),
}

_NOTEBOOK_ARTIFACT_SLUG_PATTERN = re.compile(r"[^0-9A-Za-z]+")
_ANALYSIS_RESULT_REQUIRED_ARTIFACTS: tuple[str, ...] = (
	"analysis_config",
	"dataset_sizes",
	"run_metadata",
	"aggregate_metrics",
	"per_target_metrics",
)


def _slugify_notebook_artifact_name(value: str) -> str:
	resolved_value = _NOTEBOOK_ARTIFACT_SLUG_PATTERN.sub("_", str(value).strip().lower()).strip("_")
	return resolved_value or "artifact"


def _coerce_tabular_frame(table: Any) -> pd.DataFrame:
	if isinstance(table, pd.DataFrame):
		return table.copy()
	if isinstance(table, pd.Series):
		return table.to_frame()
	if isinstance(table, Mapping):
		return pd.DataFrame([dict(table)])
	return pd.DataFrame(table)


def _artifact_paths_to_frame(artifact_paths: Mapping[str, Any]) -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"artifact_key": str(artifact_key),
				"artifact_path": None if artifact_path is None else str(artifact_path),
			}
			for artifact_key, artifact_path in artifact_paths.items()
		]
	)


def _artifact_paths_from_frame(frame: pd.DataFrame) -> dict[str, Path | None]:
	if frame.empty:
		return {}

	resolved_paths: dict[str, Path | None] = {}
	for _, row in frame.iterrows():
		artifact_key = str(row["artifact_key"])
		artifact_value = row["artifact_path"]
		resolved_paths[artifact_key] = None if pd.isna(artifact_value) else Path(str(artifact_value))
	return resolved_paths


def _load_tensor_slices_from_tables(
	tables: Mapping[str, pd.DataFrame],
	*,
	prefix: str,
) -> np.ndarray | None:
	matching_keys = sorted(
		artifact_key
		for artifact_key in tables
		if artifact_key.startswith(f"{prefix}/")
	)
	if not matching_keys:
		return None

	return np.stack(
		[tables[artifact_key].to_numpy(dtype=float) for artifact_key in matching_keys],
		axis=0,
	)


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


def build_notebook_table_recorder(
	artifact_group: str,
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
	index: bool = True,
) -> Callable[[str, str, Any], Any]:
	"""Build a describe-and-display wrapper that also persists timestamped CSV tables."""

	resolved_timestamp = make_simulation_timestamp(timestamp)

	def _record_table(title: str, description: str, table: Any) -> Any:
		artifact_path = render_notebook_tabular_artifact_path(
			artifact_group,
			_slugify_notebook_artifact_name(title),
			repo_root=repo_root,
			timestamp=resolved_timestamp,
		)
		save_dataframe_csv(artifact_path, _coerce_tabular_frame(table), index=index)
		return describe_and_display_table(title, description, table)

	return _record_table


def persist_named_table_artifacts(
	artifact_group: str,
	named_tables: Mapping[str, Any],
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
	default_index: bool = True,
	index_by_artifact: Mapping[str, bool] | None = None,
) -> dict[str, Path]:
	"""Persist one named set of tabular notebook artifacts under a shared timestamp."""

	resolved_timestamp = make_simulation_timestamp(timestamp)
	persisted_paths: dict[str, Path] = {}
	resolved_index_by_artifact = dict(index_by_artifact or {})

	for artifact_name, artifact_table in named_tables.items():
		artifact_path = render_notebook_tabular_artifact_path(
			artifact_group,
			str(artifact_name),
			repo_root=repo_root,
			timestamp=resolved_timestamp,
		)
		save_dataframe_csv(
			artifact_path,
			_coerce_tabular_frame(artifact_table),
			index=bool(resolved_index_by_artifact.get(str(artifact_name), default_index)),
		)
		persisted_paths[str(artifact_name)] = artifact_path

	return persisted_paths


def load_latest_named_table_artifacts(
	artifact_group: str,
	*,
	repo_root: str | Path | None = None,
	required_artifact_names: Sequence[str] | None = None,
	index_col_by_artifact: Mapping[str, int | str | None] | None = None,
) -> dict[str, Any]:
	"""Load the newest timestamp-compatible set of named CSV artifacts for one group."""

	artifact_directory = resolve_notebook_tabular_group_dir(
		artifact_group,
		repo_root=repo_root,
	)
	timestamp, artifact_paths = select_latest_timestamped_file_bundle(
		artifact_directory,
		required_artifact_keys=[str(artifact_name) for artifact_name in (required_artifact_names or [])],
		suffixes=(".csv",),
		recursive=True,
	)
	resolved_index_cols = dict(index_col_by_artifact or {})
	loaded_tables = {
		artifact_name: load_dataframe_csv(
			artifact_path,
			index_col=resolved_index_cols.get(artifact_name),
		)
		for artifact_name, artifact_path in artifact_paths.items()
	}
	return {
		"artifact_timestamp": timestamp,
		"artifact_paths": artifact_paths,
		"tables": loaded_tables,
	}


def persist_analysis_result_artifacts(
	model_name: str,
	analysis_result: Mapping[str, Any],
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
) -> dict[str, Any]:
	"""Persist one model's dataset-size sweep outputs as timestamped CSV artifacts."""

	artifact_group = f"analysis/{str(model_name)}"
	resolved_timestamp = make_simulation_timestamp(timestamp)
	named_tables: dict[str, Any] = {
		"analysis_config": pd.DataFrame([dict(analysis_result["analysis_config"])]),
		"dataset_sizes": pd.DataFrame({"dataset_size_total": list(analysis_result["dataset_sizes"])}),
		"run_metadata": analysis_result["run_metadata"],
		"aggregate_metrics": analysis_result["aggregate_metrics"],
		"per_target_metrics": analysis_result["per_target_metrics"],
	}
	for prediction_index, prediction_table in enumerate(analysis_result.get("prediction_tables", [])):
		named_tables[f"prediction_tables/prediction_table_{prediction_index:04d}"] = prediction_table

	persisted_paths = persist_named_table_artifacts(
		artifact_group,
		named_tables,
		repo_root=repo_root,
		timestamp=resolved_timestamp,
		default_index=False,
	)
	return {
		"artifact_group": artifact_group,
		"artifact_timestamp": resolved_timestamp,
		"artifact_paths": persisted_paths,
	}


def load_latest_analysis_result(
	model_name: str,
	*,
	repo_root: str | Path | None = None,
) -> dict[str, Any]:
	"""Load the newest persisted dataset-size sweep outputs for one model."""

	artifact_group = f"analysis/{str(model_name)}"
	loaded_bundle = load_latest_named_table_artifacts(
		artifact_group,
		repo_root=repo_root,
		required_artifact_names=_ANALYSIS_RESULT_REQUIRED_ARTIFACTS,
	)
	loaded_tables = dict(loaded_bundle["tables"])
	prediction_table_keys = sorted(
		artifact_name
		for artifact_name in loaded_tables
		if artifact_name.startswith("prediction_tables/")
	)

	return {
		"artifact_group": artifact_group,
		"artifact_timestamp": loaded_bundle["artifact_timestamp"],
		"artifact_paths": loaded_bundle["artifact_paths"],
		"analysis_config": loaded_tables["analysis_config"].iloc[0].to_dict(),
		"dataset_sizes": loaded_tables["dataset_sizes"]["dataset_size_total"].astype(int).tolist(),
		"run_metadata": loaded_tables["run_metadata"],
		"aggregate_metrics": loaded_tables["aggregate_metrics"],
		"per_target_metrics": loaded_tables["per_target_metrics"],
		"prediction_tables": [loaded_tables[artifact_name] for artifact_name in prediction_table_keys],
	}


def persist_classical_training_context(
	model_name: str,
	result: Mapping[str, Any],
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
) -> dict[str, Any]:
	"""Persist the latest classical training context needed by downstream notebook cells."""

	artifact_group = f"training/{str(model_name)}/context"
	resolved_timestamp = make_simulation_timestamp(timestamp)
	named_tables: dict[str, Any] = {
		"best_hyperparameters": pd.DataFrame([dict(result["best_hyperparameters"])]),
		"artifact_paths": _artifact_paths_to_frame(dict(result.get("artifact_paths", {}))),
	}
	if "test_report" in result and "report_metadata" in result["test_report"]:
		named_tables["report_metadata"] = result["test_report"]["report_metadata"]

	persisted_paths = persist_named_table_artifacts(
		artifact_group,
		named_tables,
		repo_root=repo_root,
		timestamp=resolved_timestamp,
		default_index=False,
	)
	return {
		"artifact_group": artifact_group,
		"artifact_timestamp": resolved_timestamp,
		"artifact_paths": persisted_paths,
	}


def load_latest_classical_training_context(
	model_name: str,
	*,
	repo_root: str | Path | None = None,
) -> dict[str, Any]:
	"""Load the newest persisted classical training context for one model."""

	artifact_group = f"training/{str(model_name)}/context"
	loaded_bundle = load_latest_named_table_artifacts(
		artifact_group,
		repo_root=repo_root,
		required_artifact_names=("best_hyperparameters", "artifact_paths"),
	)
	loaded_tables = dict(loaded_bundle["tables"])
	return {
		"artifact_group": artifact_group,
		"artifact_timestamp": loaded_bundle["artifact_timestamp"],
		"artifact_paths": loaded_bundle["artifact_paths"],
		"best_hyperparameters": loaded_tables["best_hyperparameters"].iloc[0].dropna().to_dict(),
		"training_artifact_paths": _artifact_paths_from_frame(loaded_tables["artifact_paths"]),
		"report_metadata": loaded_tables.get("report_metadata"),
	}


def persist_icsor_training_context(
	result: Mapping[str, Any],
	dataset_splits: Mapping[str, DatasetSplit] | Any,
	*,
	repo_root: str | Path | None = None,
	timestamp: str | None = None,
) -> dict[str, Any]:
	"""Persist the latest icsor training context needed by downstream notebook cells."""

	artifact_group = "training/icsor/context"
	resolved_timestamp = make_simulation_timestamp(timestamp)
	train_split = dataset_splits["train"] if isinstance(dataset_splits, Mapping) else dataset_splits.train
	test_split = dataset_splits["test"] if isinstance(dataset_splits, Mapping) else dataset_splits.test
	named_tables: dict[str, Any] = {
		"best_hyperparameters": pd.DataFrame([dict(result["best_hyperparameters"])]),
		"artifact_paths": _artifact_paths_to_frame(dict(result.get("artifact_paths", {}))),
		"train_split/targets": train_split.targets,
		"test_split/targets": test_split.targets,
		"train_split/constraint_reference": train_split.constraint_reference,
		"test_split/constraint_reference": test_split.constraint_reference,
	}
	for split_name, report in (("train_report", result["train_report"]), ("test_report", result["test_report"])):
		for report_key in [
			"raw_predictions",
			"affine_predictions",
			"projected_predictions",
			"raw_fractional_predictions",
			"affine_fractional_predictions",
			"projected_fractional_predictions",
		]:
			if report_key in report:
				named_tables[f"{split_name}/{report_key}"] = report[report_key]

	effective_coefficients = dict(result.get("model_bundle", {}).get("effective_coefficients", {}))
	if "W_u" in effective_coefficients:
		named_tables["effective_coefficients/w_u"] = pd.DataFrame(effective_coefficients["W_u"])
	if "W_in" in effective_coefficients:
		named_tables["effective_coefficients/w_in"] = pd.DataFrame(effective_coefficients["W_in"])
	if "b" in effective_coefficients:
		named_tables["effective_coefficients/b"] = pd.DataFrame({"value": np.asarray(effective_coefficients["b"], dtype=float)})
	for tensor_name in ["Theta_uu", "Theta_cc", "Theta_uc"]:
		if tensor_name not in effective_coefficients:
			continue
		for target_index, tensor_slice in enumerate(np.asarray(effective_coefficients[tensor_name], dtype=float)):
			named_tables[f"effective_coefficients/{tensor_name.lower()}/target_{target_index:03d}"] = pd.DataFrame(tensor_slice)

	persisted_paths = persist_named_table_artifacts(
		artifact_group,
		named_tables,
		repo_root=repo_root,
		timestamp=resolved_timestamp,
		default_index=False,
	)
	return {
		"artifact_group": artifact_group,
		"artifact_timestamp": resolved_timestamp,
		"artifact_paths": persisted_paths,
	}


def load_latest_icsor_training_context(
	*,
	repo_root: str | Path | None = None,
) -> dict[str, Any]:
	"""Load the newest persisted icsor training context for downstream notebook cells."""

	artifact_group = "training/icsor/context"
	loaded_bundle = load_latest_named_table_artifacts(
		artifact_group,
		repo_root=repo_root,
		required_artifact_names=(
			"best_hyperparameters",
			"artifact_paths",
			"train_split/targets",
			"test_split/targets",
			"train_split/constraint_reference",
			"test_split/constraint_reference",
			"train_report/projected_predictions",
			"test_report/projected_predictions",
		),
	)
	loaded_tables = dict(loaded_bundle["tables"])
	train_report = {
		artifact_name.removeprefix("train_report/"): table
		for artifact_name, table in loaded_tables.items()
		if artifact_name.startswith("train_report/")
	}
	test_report = {
		artifact_name.removeprefix("test_report/"): table
		for artifact_name, table in loaded_tables.items()
		if artifact_name.startswith("test_report/")
	}
	effective_coefficients: dict[str, Any] = {}
	if "effective_coefficients/w_u" in loaded_tables:
		effective_coefficients["W_u"] = loaded_tables["effective_coefficients/w_u"].to_numpy(dtype=float)
	if "effective_coefficients/w_in" in loaded_tables:
		effective_coefficients["W_in"] = loaded_tables["effective_coefficients/w_in"].to_numpy(dtype=float)
	if "effective_coefficients/b" in loaded_tables:
		effective_coefficients["b"] = loaded_tables["effective_coefficients/b"]["value"].to_numpy(dtype=float)
	loaded_theta_uu = _load_tensor_slices_from_tables(loaded_tables, prefix="effective_coefficients/theta_uu")
	if loaded_theta_uu is not None:
		effective_coefficients["Theta_uu"] = loaded_theta_uu
	loaded_theta_cc = _load_tensor_slices_from_tables(loaded_tables, prefix="effective_coefficients/theta_cc")
	if loaded_theta_cc is not None:
		effective_coefficients["Theta_cc"] = loaded_theta_cc
	loaded_theta_uc = _load_tensor_slices_from_tables(loaded_tables, prefix="effective_coefficients/theta_uc")
	if loaded_theta_uc is not None:
		effective_coefficients["Theta_uc"] = loaded_theta_uc

	return {
		"artifact_group": artifact_group,
		"artifact_timestamp": loaded_bundle["artifact_timestamp"],
		"artifact_paths": loaded_bundle["artifact_paths"],
		"best_hyperparameters": loaded_tables["best_hyperparameters"].iloc[0].dropna().to_dict(),
		"training_artifact_paths": _artifact_paths_from_frame(loaded_tables["artifact_paths"]),
		"train_targets": loaded_tables["train_split/targets"],
		"test_targets": loaded_tables["test_split/targets"],
		"train_constraint_reference": loaded_tables["train_split/constraint_reference"],
		"test_constraint_reference": loaded_tables["test_split/constraint_reference"],
		"train_report": train_report,
		"test_report": test_report,
		"effective_coefficients": effective_coefficients,
	}


def _iter_prediction_frames(
	report: Mapping[str, pd.DataFrame],
	*,
	frame_specs: Sequence[PredictionFrameSpec] = DEFAULT_NEGATIVE_PREDICTION_FRAME_SPECS,
) -> list[tuple[str, pd.DataFrame, str]]:
	resolved_frame_specs = tuple(frame_specs)
	resolved_frames: list[tuple[str, pd.DataFrame, str]] = []

	for prediction_type, frame_key, column_prefix in resolved_frame_specs:
		if frame_key in report:
			resolved_frames.append((prediction_type, report[frame_key].copy(), column_prefix))
	if not resolved_frames:
		requested_keys = ", ".join(frame_key for _, frame_key, _ in resolved_frame_specs)
		raise KeyError(
			"Report must include at least one prediction frame from: "
			f"{requested_keys}."
		)

	return resolved_frames


def _report_has_prediction_frames(
	report: Mapping[str, pd.DataFrame],
	*,
	frame_specs: Sequence[PredictionFrameSpec],
) -> bool:
	return any(frame_key in report for _, frame_key, _ in frame_specs)


def _build_negative_prediction_frames(
	report: Mapping[str, pd.DataFrame],
	*,
	split_name: str,
	frame_specs: Sequence[PredictionFrameSpec] = DEFAULT_NEGATIVE_PREDICTION_FRAME_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	incidence_rows: list[dict[str, Any]] = []
	per_target_frames: list[pd.DataFrame] = []

	for prediction_type, prediction_frame, column_prefix in _iter_prediction_frames(
		report,
		frame_specs=frame_specs,
	):
		resolved_prediction_frame = prediction_frame.rename(
			columns=lambda column_name: str(column_name).removeprefix(column_prefix)
		)
		negative_mask = resolved_prediction_frame.lt(0.0)
		negative_prediction_count = int(negative_mask.to_numpy(dtype=bool).sum())
		total_prediction_count = int(resolved_prediction_frame.size)
		total_sample_count = int(len(resolved_prediction_frame))
		affected_sample_counts = negative_mask.sum(axis=1)
		affected_sample_count = int((affected_sample_counts > 0).sum())
		affected_samples = affected_sample_counts[affected_sample_counts > 0]
		negative_values = resolved_prediction_frame.where(negative_mask).stack()

		incidence_rows.append(
			{
				"split": split_name,
				"prediction_type": prediction_type,
				"total_predictions": total_prediction_count,
				"negative_predictions": negative_prediction_count,
				"negative_prediction_rate_pct": (
					100.0 * negative_prediction_count / total_prediction_count
					if total_prediction_count > 0
					else 0.0
				),
				"total_samples": total_sample_count,
				"samples_with_any_negative": affected_sample_count,
				"sample_incidence_rate_pct": (
					100.0 * affected_sample_count / total_sample_count
					if total_sample_count > 0
					else 0.0
				),
				"avg_negative_targets_per_affected_sample": (
					float(affected_samples.mean()) if not affected_samples.empty else 0.0
				),
				"minimum_prediction": float(resolved_prediction_frame.min().min()),
				"mean_negative_prediction": (
					float(negative_values.mean()) if not negative_values.empty else np.nan
				),
				"median_negative_prediction": (
					float(negative_values.median()) if not negative_values.empty else np.nan
				),
			}
		)

		per_target_frames.append(
			pd.DataFrame(
				{
					"split": split_name,
					"prediction_type": prediction_type,
					"target": resolved_prediction_frame.columns,
					"negative_predictions": negative_mask.sum(axis=0).to_numpy(dtype=int),
					"negative_prediction_rate_pct": negative_mask.mean(axis=0).mul(100.0).to_numpy(dtype=float),
					"minimum_prediction": resolved_prediction_frame.min(axis=0).to_numpy(dtype=float),
					"mean_negative_prediction": resolved_prediction_frame.where(negative_mask).mean(axis=0).to_numpy(dtype=float),
					"median_negative_prediction": resolved_prediction_frame.where(negative_mask).median(axis=0).to_numpy(dtype=float),
				}
			)
		)

	return pd.DataFrame(incidence_rows), pd.concat(per_target_frames, ignore_index=True)


def build_negative_prediction_tables(
	reports_by_split: Mapping[str, Mapping[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
	"""Summarize negative measured-output predictions across train/test style reports."""

	if not reports_by_split:
		raise ValueError("reports_by_split must include at least one split report.")

	summary_frames: list[pd.DataFrame] = []
	per_target_frames: list[pd.DataFrame] = []

	for split_name, report in reports_by_split.items():
		split_summary, split_per_target = _build_negative_prediction_frames(
			report,
			split_name=str(split_name),
			frame_specs=DEFAULT_NEGATIVE_PREDICTION_FRAME_SPECS,
		)
		summary_frames.append(split_summary)
		per_target_frames.append(split_per_target)

	return {
		"summary": pd.concat(summary_frames, ignore_index=True),
		"per_target": pd.concat(per_target_frames, ignore_index=True).sort_values(
			["split", "prediction_type", "negative_predictions", "minimum_prediction"],
			ascending=[True, True, False, True],
		).reset_index(drop=True),
	}


def build_separated_negative_prediction_tables(
	reports_by_split: Mapping[str, Mapping[str, pd.DataFrame]],
	*,
	family_specs: Mapping[str, Sequence[PredictionFrameSpec]] | None = None,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
	"""Summarize negative predictions separately by family and prediction type."""

	if not reports_by_split:
		raise ValueError("reports_by_split must include at least one split report.")

	resolved_family_specs = dict(family_specs or icsor_NEGATIVE_PREDICTION_FAMILY_SPECS)
	separated_tables: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}

	for family_name, frame_specs in resolved_family_specs.items():
		resolved_frame_specs = tuple(frame_specs)
		available_split_names = [
			str(split_name)
			for split_name, report in reports_by_split.items()
			if _report_has_prediction_frames(report, frame_specs=resolved_frame_specs)
		]
		if not available_split_names:
			continue

		missing_split_names = [
			str(split_name)
			for split_name, report in reports_by_split.items()
			if not _report_has_prediction_frames(report, frame_specs=resolved_frame_specs)
		]
		if missing_split_names:
			missing_display = ", ".join(missing_split_names)
			raise KeyError(
				f"Negative-prediction family '{family_name}' is missing prediction frames for splits: "
				f"{missing_display}."
			)

		summary_frames: list[pd.DataFrame] = []
		per_target_frames: list[pd.DataFrame] = []
		for split_name, report in reports_by_split.items():
			split_summary, split_per_target = _build_negative_prediction_frames(
				report,
				split_name=str(split_name),
				frame_specs=resolved_frame_specs,
			)
			summary_frames.append(split_summary)
			per_target_frames.append(split_per_target)

		family_summary = pd.concat(summary_frames, ignore_index=True)
		family_per_target = pd.concat(per_target_frames, ignore_index=True)
		family_tables: dict[str, dict[str, pd.DataFrame]] = {}
		for prediction_type, _, _ in resolved_frame_specs:
			prediction_summary = family_summary.loc[
				family_summary["prediction_type"] == prediction_type
			].reset_index(drop=True)
			if prediction_summary.empty:
				continue

			prediction_per_target = family_per_target.loc[
				family_per_target["prediction_type"] == prediction_type
			].sort_values(
				["split", "negative_predictions", "minimum_prediction"],
				ascending=[True, False, True],
			).reset_index(drop=True)
			family_tables[prediction_type] = {
				"summary": prediction_summary,
				"per_target": prediction_per_target,
			}

		if family_tables:
			separated_tables[str(family_name)] = family_tables

	if not separated_tables:
		raise KeyError("No matching negative-prediction families were found in the provided reports.")

	return separated_tables

_DEFAULT_ANALYSIS_SETTINGS = {
	"min_total_samples": 100,
	"max_total_samples": 10000,
	"total_sample_step": 330,
	"n_repeats": 30,
	"test_fraction": 0.2,
	"random_seed": 42,
}

_DEFAULT_icsor_RESPONSE_SURFACE_SETTINGS = {
	"grid_points_per_axis": 49,
	"contour_levels": 18,
	"operational_extension_fraction": 0.5,
	"fixed_influent_profile": "midpoint",
}

COMPARISON_METRIC_DIRECTIONS: dict[str, str] = {
	"R2": "higher",
	"MSE": "lower",
	"RMSE": "lower",
	"MAE": "lower",
	"MAPE": "lower",
}

COMPARISON_METRIC_BASENAMES: tuple[str, ...] = tuple(COMPARISON_METRIC_DIRECTIONS)

_PREDICTION_DIAGNOSTIC_METADATA_COLUMNS: tuple[str, ...] = (
	"model_name",
	"model_key",
	"model_label",
	"model_family",
	"model_order",
	"dataset_size_total",
	"repeat_index",
	"train_size",
	"test_size",
	"run_seed",
	"split_name",
)


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


def load_icsor_response_surface_defaults(repo_root: str | Path | None = None) -> dict[str, int | float | str]:
	"""Load configurable defaults for the icsor operational response-surface study."""

	orchestration_params = load_ml_orchestration_params(repo_root)
	analysis_params = dict(orchestration_params.get("analysis", {}))
	response_surface_params = dict(analysis_params.get("icsor_response_surface", {}))

	return {
		"grid_points_per_axis": int(
			response_surface_params.get(
				"grid_points_per_axis",
				_DEFAULT_icsor_RESPONSE_SURFACE_SETTINGS["grid_points_per_axis"],
			)
		),
		"contour_levels": int(
			response_surface_params.get(
				"contour_levels",
				_DEFAULT_icsor_RESPONSE_SURFACE_SETTINGS["contour_levels"],
			)
		),
		"operational_extension_fraction": float(
			response_surface_params.get(
				"operational_extension_fraction",
				_DEFAULT_icsor_RESPONSE_SURFACE_SETTINGS["operational_extension_fraction"],
			)
		),
		"fixed_influent_profile": str(
			response_surface_params.get(
				"fixed_influent_profile",
				_DEFAULT_icsor_RESPONSE_SURFACE_SETTINGS["fixed_influent_profile"],
			)
		),
	}


def _build_component_response_surface_prediction_data(
	model_path: str | Path,
	*,
	predictor: Callable[..., dict[str, pd.DataFrame]],
	metadata: Mapping[str, Any] | None = None,
	repo_root: str | Path | None = None,
	grid_points_per_axis: int | None = None,
	operational_extension_fraction: float | None = None,
	fixed_influent_profile: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	"""Build a component-model response surface over HRT and Aeration with a fixed influent profile."""

	defaults = load_icsor_response_surface_defaults(repo_root)
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
	measured_output_columns = list(resolved_metadata["measured_output_columns"])
	composition_matrix = np.asarray(resolved_metadata["composition_matrix"], dtype=float)
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

	prediction_result = predictor(
		{
			"features": feature_frame,
			"constraint_reference": constraint_reference,
		},
		model_path,
	)
	affine_fractional_predictions = prediction_result.get("affine_fractional_predictions")
	projected_fractional_predictions = prediction_result.get(
		"projected_fractional_predictions",
		prediction_result["projected_predictions"],
	).copy()
	affine_predictions = None
	if affine_fractional_predictions is not None:
		affine_predictions = collapse_fractional_states_to_measured_outputs(
			affine_fractional_predictions,
			state_columns,
			composition_matrix,
			measured_output_columns,
			output_prefix="Out_",
		)
	projected_predictions = collapse_fractional_states_to_measured_outputs(
		projected_fractional_predictions,
		state_columns,
		composition_matrix,
		measured_output_columns,
		output_prefix="Out_",
	)
	affine_core_prediction_standard_errors = prediction_result.get("affine_core_prediction_standard_errors")
	affine_core_prediction_interval_lower = prediction_result.get("affine_core_prediction_interval_lower")
	affine_core_prediction_interval_upper = prediction_result.get("affine_core_prediction_interval_upper")
	per_target_surfaces = {
		target_name: projected_predictions[target_name].to_numpy(dtype=float).reshape(hrt_mesh.shape)
		for target_name in projected_predictions.columns
	}
	frame_parts = [feature_frame, constraint_reference.add_prefix("ConstraintReference_")]
	if affine_predictions is not None:
		frame_parts.append(affine_predictions.add_prefix("Affine_"))
	frame_parts.append(projected_predictions.add_prefix("Projected_"))
	projection_stage_diagnostics = prediction_result.get("projection_stage_diagnostics")
	if projection_stage_diagnostics is not None:
		frame_parts.append(projection_stage_diagnostics)
	prediction_table = pd.concat(frame_parts, axis=1)
	if affine_core_prediction_standard_errors is not None:
		prediction_table = pd.concat(
			[
				prediction_table,
				affine_core_prediction_standard_errors.add_prefix("AffineCoreSE_"),
			],
			axis=1,
		)
	if affine_core_prediction_interval_lower is not None and affine_core_prediction_interval_upper is not None:
		prediction_table = pd.concat(
			[
				prediction_table,
				affine_core_prediction_interval_lower.add_prefix("AffineCorePI95Lower_"),
				affine_core_prediction_interval_upper.add_prefix("AffineCorePI95Upper_"),
			],
			axis=1,
		)

	result = {
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
		"projected_fractional_predictions": projected_fractional_predictions,
		"projected_predictions": projected_predictions,
		"prediction_table": prediction_table,
		"per_target_surfaces": per_target_surfaces,
	}
	if affine_predictions is not None:
		result["affine_predictions"] = affine_predictions.copy()
		result["affine_fractional_predictions"] = affine_fractional_predictions.copy()
	if projection_stage_diagnostics is not None:
		result["projection_stage_diagnostics"] = projection_stage_diagnostics.copy()
	if "projection_stage_summary" in prediction_result:
		result["projection_stage_summary"] = prediction_result["projection_stage_summary"].copy()
	if affine_core_prediction_standard_errors is not None:
		result["affine_core_prediction_standard_errors"] = affine_core_prediction_standard_errors.copy()
		result["per_target_standard_error_surfaces"] = {
			target_name: affine_core_prediction_standard_errors[target_name].to_numpy(dtype=float).reshape(hrt_mesh.shape)
			for target_name in affine_core_prediction_standard_errors.columns
		}
	if affine_core_prediction_interval_lower is not None:
		result["affine_core_prediction_interval_lower"] = affine_core_prediction_interval_lower.copy()
	if affine_core_prediction_interval_upper is not None:
		result["affine_core_prediction_interval_upper"] = affine_core_prediction_interval_upper.copy()
	if "prediction_uncertainty_metadata" in prediction_result:
		result["prediction_uncertainty_metadata"] = dict(prediction_result["prediction_uncertainty_metadata"])
	if "prediction_uncertainty_summary" in prediction_result:
		result["prediction_uncertainty_summary"] = prediction_result["prediction_uncertainty_summary"].copy()

	return result


def _resolve_response_surface_metadata(
	metadata: Mapping[str, Any] | None,
	*,
	repo_root: str | Path | None = None,
) -> dict[str, list[str]]:
	params = load_params_config(repo_root)
	simulation_params = dict(params["asm2d_tsn_simulation"])
	workbook_params = dict(simulation_params.get("workbook", {}))
	resolved_metadata = dict(metadata or {})
	matrix_bundle = get_asm2d_tsn_matrices(
		load_asm2d_tsn_simulation_params(repo_root),
		repo_root=repo_root,
	)

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
		resolved_metadata.get(
			"measured_output_columns",
			[] if matrix_bundle is None else list(matrix_bundle.get("measured_output_columns", [])),
		)
	)

	if "HRT" not in operational_columns or "Aeration" not in operational_columns:
		raise ValueError("The icsor response surface requires HRT and Aeration operational columns.")
	if not state_columns:
		raise ValueError("At least one influent state column is required to build a icsor response surface.")
	if not measured_output_columns:
		raise ValueError("At least one measured output column is required to build a icsor response surface.")

	return {
		"operational_columns": operational_columns,
		"state_columns": state_columns,
		"measured_output_columns": measured_output_columns,
		"composition_matrix": np.asarray(matrix_bundle["composition_matrix"], dtype=float),
	}


def _build_midpoint_influent_profile(
	state_columns: list[str],
	*,
	repo_root: str | Path | None = None,
) -> dict[str, float]:
	params = load_params_config(repo_root)
	influent_ranges = dict(params["asm2d_tsn_simulation"]["influent_state_ranges"])
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
	operational_ranges = dict(params["asm2d_tsn_simulation"]["operational_ranges"])
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


def build_icsor_response_surface_prediction_data(
	model_path: str | Path,
	*,
	metadata: Mapping[str, Any] | None = None,
	repo_root: str | Path | None = None,
	grid_points_per_axis: int | None = None,
	operational_extension_fraction: float | None = None,
	fixed_influent_profile: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	"""Build a icsor prediction grid over HRT and Aeration with a fixed influent profile."""

	from src.models.ml import predict_icsor_model

	return _build_component_response_surface_prediction_data(
		model_path,
		predictor=predict_icsor_model,
		metadata=metadata,
		repo_root=repo_root,
		grid_points_per_axis=grid_points_per_axis,
		operational_extension_fraction=operational_extension_fraction,
		fixed_influent_profile=fixed_influent_profile,
	)


def build_icsor_coupled_qp_response_surface_prediction_data(
	model_path: str | Path,
	*,
	metadata: Mapping[str, Any] | None = None,
	repo_root: str | Path | None = None,
	grid_points_per_axis: int | None = None,
	operational_extension_fraction: float | None = None,
	fixed_influent_profile: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	"""Build a coupled-QP response surface over HRT and Aeration with a fixed influent profile."""

	from src.models.ml import predict_icsor_coupled_qp_model

	return _build_component_response_surface_prediction_data(
		model_path,
		predictor=predict_icsor_coupled_qp_model,
		metadata=metadata,
		repo_root=repo_root,
		grid_points_per_axis=grid_points_per_axis,
		operational_extension_fraction=operational_extension_fraction,
		fixed_influent_profile=fixed_influent_profile,
	)


def build_icsor_coupled_qp_coefficient_frames(
	model_bundle: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
	"""Build labeled coefficient matrices for the coupled-QP bundle."""

	resolved_bundle = dict(model_bundle)
	design_schema = dict(resolved_bundle.get("design_schema", {}))
	design_columns = [str(column_name) for column_name in design_schema.get("design_columns", [])]
	output_labels = [str(column_name).removeprefix("Out_") for column_name in resolved_bundle.get("target_columns", [])]
	constraint_labels = [str(column_name) for column_name in resolved_bundle.get("constraint_columns", [])]

	b_matrix = np.asarray(resolved_bundle["B_matrix"], dtype=float)
	gamma_matrix = np.asarray(resolved_bundle["Gamma_matrix"], dtype=float)
	r_matrix = np.asarray(resolved_bundle["R_matrix"], dtype=float)

	if not design_columns:
		design_columns = [f"design_{column_index}" for column_index in range(b_matrix.shape[1])]
	if not output_labels:
		output_labels = [f"output_{row_index}" for row_index in range(b_matrix.shape[0])]
	if not constraint_labels:
		constraint_labels = [f"state_{row_index}" for row_index in range(gamma_matrix.shape[0])]

	return {
		"B_matrix": pd.DataFrame(b_matrix, index=output_labels, columns=design_columns),
		"Gamma_matrix": pd.DataFrame(gamma_matrix, index=constraint_labels, columns=constraint_labels),
		"R_matrix": pd.DataFrame(r_matrix, index=constraint_labels, columns=constraint_labels),
	}


def build_icsor_coupled_qp_b_matrix_block_frames(
	model_bundle: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
	"""Return B-matrix column blocks split by the persisted design-schema partition."""

	coefficient_frames = build_icsor_coupled_qp_coefficient_frames(model_bundle)
	b_frame = coefficient_frames["B_matrix"]
	resolved_bundle = dict(model_bundle)
	design_schema = dict(resolved_bundle.get("design_schema", {}))
	block_ranges = dict(design_schema.get("block_ranges", {}))
	if not block_ranges:
		return {"full": b_frame.copy()}

	b_columns = list(b_frame.columns)
	b_block_frames: dict[str, pd.DataFrame] = {}
	for block_name, block_range in block_ranges.items():
		start_index = int(block_range.get("start", 0))
		stop_index = int(block_range.get("stop", start_index))
		if stop_index <= start_index:
			continue
		if start_index < 0 or stop_index > len(b_columns):
			raise ValueError(
				f"Invalid design-schema block range for '{block_name}': [{start_index}, {stop_index}) "
				f"outside B-matrix width {len(b_columns)}."
			)
		selected_columns = b_columns[start_index:stop_index]
		b_block_frames[str(block_name)] = b_frame.loc[:, selected_columns].copy()

	if not b_block_frames:
		return {"full": b_frame.copy()}

	return b_block_frames


def build_icsor_coupled_qp_b_matrix_block_metadata(
	model_bundle: Mapping[str, Any],
) -> pd.DataFrame:
	"""Summarize B-matrix design-schema blocks for notebook reporting and validation."""

	resolved_bundle = dict(model_bundle)
	design_schema = dict(resolved_bundle.get("design_schema", {}))
	block_ranges = dict(design_schema.get("block_ranges", {}))
	b_width = int(np.asarray(resolved_bundle["B_matrix"], dtype=float).shape[1])
	if not block_ranges:
		return pd.DataFrame(
			[
				{
					"block_name": "full",
					"start": 0,
					"stop": b_width,
					"width": b_width,
				}
			]
		)

	block_rows: list[dict[str, Any]] = []
	for block_name, block_range in block_ranges.items():
		start_index = int(block_range.get("start", 0))
		stop_index = int(block_range.get("stop", start_index))
		width = max(0, stop_index - start_index)
		block_rows.append(
			{
				"block_name": str(block_name),
				"start": start_index,
				"stop": stop_index,
				"width": width,
			}
		)

	metadata_frame = pd.DataFrame(block_rows).sort_values("start").reset_index(drop=True)
	return metadata_frame


def build_icsor_coupled_qp_b_matrix_block_interpretation_table(
	model_bundle: Mapping[str, Any],
) -> pd.DataFrame:
	"""Map persisted design-schema B-matrix blocks to driver-level interpretations."""

	block_metadata = build_icsor_coupled_qp_b_matrix_block_metadata(model_bundle)
	label_map: dict[str, dict[str, str]] = {
		"bias": {
			"conceptual_block": "baseline_driver",
			"driver_term": "b",
			"interpretation": "Constant baseline driver term before operational and influent effects.",
		},
		"linear_operational": {
			"conceptual_block": "linear_operational",
			"driver_term": "W_u u",
			"interpretation": "First-order operating-condition contribution to the feature-driven driver.",
		},
		"linear_influent": {
			"conceptual_block": "linear_influent",
			"driver_term": "W_in c_in",
			"interpretation": "First-order influent ASM-state contribution to the feature-driven driver.",
		},
		"quadratic_operational": {
			"conceptual_block": "quadratic_operational",
			"driver_term": "Theta_uu(u kron u)",
			"interpretation": "Second-order curvature and interactions within operational variables.",
		},
		"quadratic_influent": {
			"conceptual_block": "quadratic_influent",
			"driver_term": "Theta_cc(c_in kron c_in)",
			"interpretation": "Second-order curvature and interactions within influent ASM states.",
		},
		"interaction_operational_influent": {
			"conceptual_block": "interaction_operational_influent",
			"driver_term": "Theta_uc(u kron c_in)",
			"interpretation": "Cross interactions between operating conditions and influent ASM states.",
		},
	}

	block_rows: list[dict[str, Any]] = []
	for metadata_row in block_metadata.itertuples(index=False):
		block_name = str(metadata_row.block_name)
		resolved_label = label_map.get(
			block_name,
			{
				"conceptual_block": "custom_design_block",
				"driver_term": "custom",
				"interpretation": "User-defined design-schema partition retained from persisted training metadata.",
			},
		)
		block_rows.append(
			{
				"block_name": block_name,
				"start": int(metadata_row.start),
				"stop": int(metadata_row.stop),
				"width": int(metadata_row.width),
				"conceptual_block": str(resolved_label["conceptual_block"]),
				"driver_term": str(resolved_label["driver_term"]),
				"interpretation": str(resolved_label["interpretation"]),
			}
		)

	return pd.DataFrame(block_rows)


def build_icsor_coupled_qp_coefficient_metadata(
	model_bundle: Mapping[str, Any],
) -> pd.DataFrame:
	"""Summarize coupled-QP coefficient-estimation metadata for notebook reporting."""

	resolved_bundle = dict(model_bundle)
	design_schema = dict(resolved_bundle.get("design_schema", {}))
	best_restart_summary = dict(resolved_bundle.get("best_restart_summary", {}))
	training_diagnostics = dict(resolved_bundle.get("training_diagnostics", {}))
	coupled_qp_settings = dict(resolved_bundle.get("coupled_qp_settings", {}))
	training_options = dict(resolved_bundle.get("training_options", {}))

	metadata_row = {
		"training_method": resolved_bundle.get("training_method"),
		"objective": training_options.get("objective_name"),
		"include_bias_term": design_schema.get("include_bias_term"),
		"design_dimension": design_schema.get("dimensions", {}).get("design"),
		"operational_dimension": design_schema.get("dimensions", {}).get("operational"),
		"influent_dimension": design_schema.get("dimensions", {}).get("influent"),
		"output_dimension": len(resolved_bundle.get("target_columns", [])),
		"constraint_dimension": len(resolved_bundle.get("constraint_columns", [])),
		"gamma_abs_bound": coupled_qp_settings.get("gamma_abs_bound"),
		"conditioning": best_restart_summary.get("conditioning"),
		"shrink_factor": best_restart_summary.get("shrink_factor"),
		"best_iteration": best_restart_summary.get("best_iteration"),
		"n_iterations": best_restart_summary.get("n_iterations"),
		"n_restarts_requested": coupled_qp_settings.get("n_restarts"),
		"n_restarts_completed": len(training_diagnostics.get("restart_summaries", [])),
		"chat_update_steps": len(training_diagnostics.get("chat_update_history", [])),
		"gamma_update_steps": len(training_diagnostics.get("gamma_update_history", [])),
		"convergence_steps": len(training_diagnostics.get("convergence_history", [])),
	}
	return pd.DataFrame([metadata_row])


def build_icsor_coupled_qp_coefficient_contract_table(
	model_bundle: Mapping[str, Any],
) -> pd.DataFrame:
	"""Summarize coefficient roles and where hard constraints are enforced."""

	resolved_bundle = dict(model_bundle)
	training_options = dict(resolved_bundle.get("training_options", {}))

	return pd.DataFrame(
		[
			{
				"training_method": resolved_bundle.get("training_method"),
				"objective": training_options.get("objective_name"),
				"driver_matrix": "B_matrix",
				"driver_sign_constraint": "Unrestricted (mixed-sign coefficients allowed).",
				"coupling_matrix": "Gamma_matrix",
				"system_matrix": "R_matrix",
				"system_matrix_definition": "R = I - Gamma",
				"training_nonnegativity_enforcement": "Hard constraints on fitted predictions in optimization.",
				"deployment_nonnegativity_enforcement": "Hard constraints in deployment inference LP.",
				"training_invariant_handling": "Penalty term on invariant residuals.",
				"deployment_invariant_handling": "Hard equality constraints A c = A c_in.",
			}
		]
	)


def build_icsor_coupled_qp_coefficient_density_tables(
	model_bundle: Mapping[str, Any],
	*,
	retention_fraction: float = 1e-4,
) -> dict[str, pd.DataFrame]:
	"""Build block-level and per-output coefficient-density summaries for coupled-QP bundles."""

	if retention_fraction <= 0.0:
		raise ValueError("retention_fraction must be positive.")

	coefficient_frames = build_icsor_coupled_qp_coefficient_frames(model_bundle)
	block_specs = [
		("B_matrix", "Driver matrix", False),
		("Gamma_matrix", "Coupling matrix", True),
		("R_matrix", "System matrix", False),
	]
	block_rows: list[dict[str, Any]] = []
	per_output_rows: list[dict[str, Any]] = []

	for block_name, block_label, exclude_diagonal in block_specs:
		block_frame = coefficient_frames[block_name]
		block_array = np.asarray(block_frame, dtype=float)
		selectable_mask = np.ones(block_array.shape, dtype=bool)
		if exclude_diagonal and block_array.shape[0] == block_array.shape[1]:
			np.fill_diagonal(selectable_mask, False)

		selectable_count = int(selectable_mask.sum())
		retained_count = 0
		absolute_values = np.abs(block_array)

		for row_index, row_label in enumerate(block_frame.index):
			row_mask = selectable_mask[row_index]
			row_selectable_count = int(row_mask.sum())
			if row_selectable_count == 0:
				threshold_value = 0.0
				row_retained_count = 0
			else:
				row_max_abs = float(np.max(absolute_values[row_index, row_mask]))
				threshold_value = retention_fraction * row_max_abs
				row_retained_count = int(np.sum(absolute_values[row_index, row_mask] >= threshold_value))

			retained_count += row_retained_count
			per_output_rows.append(
				{
					"block_name": block_name,
					"block_label": block_label,
					"row_label": str(row_label),
					"retention_fraction": float(retention_fraction),
					"absolute_threshold": float(threshold_value),
					"selectable_coefficients": row_selectable_count,
					"retained_coefficients": row_retained_count,
					"retained_fraction_pct": (
						0.0 if row_selectable_count == 0 else 100.0 * row_retained_count / row_selectable_count
					),
				}
			)

		block_rows.append(
			{
				"block_name": block_name,
				"block_label": block_label,
				"retention_fraction": float(retention_fraction),
				"selectable_coefficients": selectable_count,
				"retained_coefficients": retained_count,
				"retained_fraction_pct": 0.0 if selectable_count == 0 else 100.0 * retained_count / selectable_count,
			}
		)

	return {
		"summary": pd.DataFrame(block_rows),
		"per_output_thresholds": pd.DataFrame(per_output_rows),
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
		dataset_split.constraint_reference.add_prefix("ConstraintReference_"),
	]
	affine_predictions = report.get("affine_predictions")
	if affine_predictions is not None:
		frame_parts.insert(2, affine_predictions)
	projected_predictions = report.get("projected_predictions")
	if projected_predictions is not None:
		insert_position = 3 if affine_predictions is not None else 2
		frame_parts.insert(insert_position, projected_predictions)

	constraint_residuals = report.get("constraint_residuals")
	if constraint_residuals is not None:
		frame_parts.append(constraint_residuals)

	projection_diagnostics = report.get("projection_diagnostics")
	if projection_diagnostics is not None:
		frame_parts.append(projection_diagnostics)

	for optional_key in [
		"affine_core_prediction_standard_errors",
		"affine_core_prediction_confidence_interval_lower",
		"affine_core_prediction_confidence_interval_upper",
		"affine_core_prediction_interval_lower",
		"affine_core_prediction_interval_upper",
		"affine_core_prediction_interval_standard_errors",
	]:
		optional_frame = report.get(optional_key)
		if optional_frame is not None:
			frame_parts.append(optional_frame)

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
				run_started_at = time.perf_counter()
				result = model_runner(
					dataset_splits.train,
					dataset_splits.test,
					A_matrix,
					**clean_runner_kwargs,
				)
				run_elapsed_seconds = float(time.perf_counter() - run_started_at)

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
						"elapsed_seconds": run_elapsed_seconds,
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


def _resolve_metric_basename(metric_name: str) -> str:
	resolved_metric_name = str(metric_name)
	if resolved_metric_name in COMPARISON_METRIC_DIRECTIONS:
		return resolved_metric_name

	metric_basename = resolved_metric_name.rsplit("_", 1)[-1]
	if metric_basename not in COMPARISON_METRIC_DIRECTIONS:
		raise ValueError(
			f"Unsupported metric '{resolved_metric_name}'. Expected one of: "
			f"{', '.join(COMPARISON_METRIC_BASENAMES)}."
		)

	return metric_basename


def get_metric_direction(metric_name: str) -> str:
	"""Return whether a comparison metric should be maximized or minimized."""

	return COMPARISON_METRIC_DIRECTIONS[_resolve_metric_basename(metric_name)]


def is_higher_better_metric(metric_name: str) -> bool:
	"""Return True when a higher value indicates better model performance."""

	return get_metric_direction(metric_name) == "higher"


def add_effective_metric_columns(
	metric_frame: pd.DataFrame,
	*,
	metric_basenames: Sequence[str] | None = None,
	prefix_preference: Sequence[str] = ("projected", "raw"),
) -> pd.DataFrame:
	"""Add row-wise effective metrics that prefer projected values and fall back to raw values."""

	if not isinstance(metric_frame, pd.DataFrame):
		raise ValueError("metric_frame must be a pandas DataFrame.")

	resolved_metric_basenames = [
		_resolve_metric_basename(metric_basename)
		for metric_basename in (metric_basenames or COMPARISON_METRIC_BASENAMES)
	]
	result = metric_frame.copy()

	for metric_basename in resolved_metric_basenames:
		effective_metric_name = f"effective_{metric_basename}"
		source_metric_name = f"{effective_metric_name}_source"
		result[effective_metric_name] = np.nan
		result[source_metric_name] = pd.Series(pd.NA, index=result.index, dtype="object")

		for prefix in prefix_preference:
			candidate_metric_name = f"{str(prefix)}_{metric_basename}"
			if candidate_metric_name not in result.columns:
				continue

			candidate_values = pd.to_numeric(result[candidate_metric_name], errors="coerce")
			missing_mask = result[effective_metric_name].isna() & candidate_values.notna()
			if not bool(missing_mask.any()):
				continue

			result.loc[missing_mask, effective_metric_name] = candidate_values.loc[missing_mask]
			result.loc[missing_mask, source_metric_name] = str(prefix)

	return result


def build_effective_aggregate_metrics(
	aggregate_metric_frame: pd.DataFrame,
) -> pd.DataFrame:
	"""Select the deployed aggregate metric row for each run and split.

	Aggregate metric frames store one row per prediction type. This helper keeps the projected
	row when it exists and otherwise falls back to the raw row, while exposing the selected
	metrics under effective_* column names.
	"""

	if not isinstance(aggregate_metric_frame, pd.DataFrame):
		raise ValueError("aggregate_metric_frame must be a pandas DataFrame.")

	required_columns = {"prediction_type", *COMPARISON_METRIC_BASENAMES}
	missing_columns = sorted(required_columns.difference(aggregate_metric_frame.columns))
	if missing_columns:
		missing_display = ", ".join(missing_columns)
		raise KeyError(f"aggregate_metric_frame is missing required columns: {missing_display}")

	group_columns = [
		str(column_name)
		for column_name in aggregate_metric_frame.columns
		if column_name not in {"prediction_type", *COMPARISON_METRIC_BASENAMES}
	]
	if not group_columns:
		raise ValueError("aggregate_metric_frame must include metadata columns that identify each run.")

	effective_rows: list[pd.Series] = []
	for _, group_frame in aggregate_metric_frame.groupby(group_columns, dropna=False, sort=False):
		resolved_prediction_types = group_frame["prediction_type"].astype(str).tolist()
		effective_prediction_type = "projected" if "projected" in resolved_prediction_types else "raw"
		selected_row = group_frame.loc[
			group_frame["prediction_type"].astype(str) == effective_prediction_type
		].iloc[0].copy()
		selected_row["effective_prediction_type"] = effective_prediction_type
		for metric_basename in COMPARISON_METRIC_BASENAMES:
			selected_row[f"effective_{metric_basename}"] = float(selected_row[metric_basename])
		effective_rows.append(selected_row)

	return pd.DataFrame(effective_rows).reset_index(drop=True)


def summarize_metric_distribution(
	metric_frame: pd.DataFrame,
	*,
	metric_name: str,
	group_columns: Sequence[str],
) -> pd.DataFrame:
	"""Summarize one metric distribution across arbitrary grouping columns."""

	if not isinstance(metric_frame, pd.DataFrame):
		raise ValueError("metric_frame must be a pandas DataFrame.")

	resolved_group_columns = [str(column_name) for column_name in group_columns]
	if not resolved_group_columns:
		raise ValueError("group_columns must include at least one column name.")

	resolved_metric_name = str(metric_name)
	required_columns = set(resolved_group_columns)
	required_columns.add(resolved_metric_name)
	missing_columns = sorted(required_columns.difference(metric_frame.columns))
	if missing_columns:
		missing_display = ", ".join(missing_columns)
		raise KeyError(f"metric_frame is missing required columns: {missing_display}")

	filtered_frame = metric_frame.loc[
		metric_frame[resolved_metric_name].notna(),
		[*resolved_group_columns, resolved_metric_name],
	].copy()
	if filtered_frame.empty:
		raise ValueError(f"metric_frame does not contain any non-null values for '{resolved_metric_name}'.")

	grouped_metric = filtered_frame.groupby(resolved_group_columns, dropna=False)[resolved_metric_name]
	summary = grouped_metric.agg(
		observation_count="count",
		metric_mean="mean",
		metric_std="std",
		metric_median="median",
		metric_min="min",
		metric_max="max",
	).reset_index()
	metric_q25 = grouped_metric.quantile(0.25).reset_index(name="metric_q25")
	metric_q75 = grouped_metric.quantile(0.75).reset_index(name="metric_q75")
	summary = summary.merge(metric_q25, on=resolved_group_columns, how="left").merge(
		metric_q75,
		on=resolved_group_columns,
		how="left",
	)
	summary["metric_std"] = summary["metric_std"].fillna(0.0)
	summary.insert(len(resolved_group_columns), "metric_name", resolved_metric_name)

	return summary.sort_values(resolved_group_columns).reset_index(drop=True)


def rank_metric_summary(
	summary_frame: pd.DataFrame,
	*,
	group_columns: Sequence[str],
	ranking_column: str = "metric_mean",
	higher_is_better: bool | None = None,
	metric_name: str | None = None,
	rank_column_name: str = "metric_rank",
) -> pd.DataFrame:
	"""Rank grouped metric summaries within each comparison slice."""

	if not isinstance(summary_frame, pd.DataFrame):
		raise ValueError("summary_frame must be a pandas DataFrame.")

	resolved_group_columns = [str(column_name) for column_name in group_columns]
	required_columns = set(resolved_group_columns)
	required_columns.add(str(ranking_column))
	missing_columns = sorted(required_columns.difference(summary_frame.columns))
	if missing_columns:
		missing_display = ", ".join(missing_columns)
		raise KeyError(f"summary_frame is missing required columns: {missing_display}")

	resolved_metric_name = str(metric_name) if metric_name is not None else None
	if resolved_metric_name is None and "metric_name" in summary_frame.columns:
		non_null_metric_names = summary_frame["metric_name"].dropna().astype(str).unique().tolist()
		if len(non_null_metric_names) == 1:
			resolved_metric_name = non_null_metric_names[0]

	resolved_higher_is_better = (
		bool(higher_is_better)
		if higher_is_better is not None
		else is_higher_better_metric(resolved_metric_name or str(ranking_column))
	)
	result = summary_frame.copy()
	result[rank_column_name] = result.groupby(resolved_group_columns, dropna=False)[str(ranking_column)].rank(
		method="dense",
		ascending=not resolved_higher_is_better,
	)
	result[rank_column_name] = result[rank_column_name].astype(int)

	return result.sort_values(
		[*resolved_group_columns, rank_column_name, str(ranking_column)],
		ascending=[*[True for _ in resolved_group_columns], True, not resolved_higher_is_better],
	).reset_index(drop=True)


def build_train_test_gap_summary(
	summary_frame: pd.DataFrame,
	*,
	group_columns: Sequence[str],
	split_column: str = "split_name",
	higher_is_better: bool | None = None,
	metric_name: str | None = None,
) -> pd.DataFrame:
	"""Compute train-test gaps from grouped metric summaries.

	Positive generalization_gap values always indicate that the train split outperforms the
	test split, regardless of whether the metric is maximized or minimized.
	"""

	if not isinstance(summary_frame, pd.DataFrame):
		raise ValueError("summary_frame must be a pandas DataFrame.")

	resolved_group_columns = [str(column_name) for column_name in group_columns]
	resolved_split_column = str(split_column)
	required_columns = set(resolved_group_columns)
	required_columns.add(resolved_split_column)
	required_columns.add("metric_mean")
	missing_columns = sorted(required_columns.difference(summary_frame.columns))
	if missing_columns:
		missing_display = ", ".join(missing_columns)
		raise KeyError(f"summary_frame is missing required columns: {missing_display}")

	available_splits = set(summary_frame[resolved_split_column].astype(str).unique())
	if not {"train", "test"}.issubset(available_splits):
		raise ValueError("summary_frame must contain both 'train' and 'test' rows.")

	resolved_metric_name = str(metric_name) if metric_name is not None else None
	if resolved_metric_name is None and "metric_name" in summary_frame.columns:
		non_null_metric_names = summary_frame["metric_name"].dropna().astype(str).unique().tolist()
		if len(non_null_metric_names) == 1:
			resolved_metric_name = non_null_metric_names[0]

	resolved_higher_is_better = (
		bool(higher_is_better)
		if higher_is_better is not None
		else is_higher_better_metric(resolved_metric_name or "metric_mean")
	)

	value_columns = [
		column_name
		for column_name in [
			"observation_count",
			"metric_mean",
			"metric_std",
			"metric_median",
			"metric_q25",
			"metric_q75",
			"metric_min",
			"metric_max",
		]
		if column_name in summary_frame.columns
	]
	result: pd.DataFrame | None = None

	for value_column in value_columns:
		pivot = summary_frame.pivot_table(
			index=resolved_group_columns,
			columns=resolved_split_column,
			values=value_column,
			aggfunc="first",
			dropna=False,
		)
		if not {"train", "test"}.issubset(set(pivot.columns.astype(str))):
			raise ValueError(
				f"summary_frame does not include both train and test values for '{value_column}'."
			)
		pivot = pivot.rename(
			columns={
				"train": f"train_{value_column}",
				"test": f"test_{value_column}",
			}
		).reset_index()
		result = pivot if result is None else result.merge(pivot, on=resolved_group_columns, how="inner")

	if result is None:
		raise ValueError("No summary value columns were available to compute train-test gaps.")

	result["generalization_gap"] = (
		result["train_metric_mean"] - result["test_metric_mean"]
		if resolved_higher_is_better
		else result["test_metric_mean"] - result["train_metric_mean"]
	)
	result["generalization_gap_pct"] = np.where(
		np.abs(result["train_metric_mean"]) > 1e-12,
		100.0 * result["generalization_gap"] / np.abs(result["train_metric_mean"]),
		np.nan,
	)
	if resolved_metric_name is not None:
		result.insert(len(resolved_group_columns), "metric_name", resolved_metric_name)

	return result.sort_values(resolved_group_columns).reset_index(drop=True)


def _extract_prediction_diagnostic_metadata(
	prediction_table: pd.DataFrame,
	*,
	model_labels: Mapping[str, str] | None,
	model_families: Mapping[str, str] | None,
	model_order: Mapping[str, int] | None,
) -> dict[str, Any]:
	metadata = {
		column_name: prediction_table[column_name].iloc[0]
		for column_name in _PREDICTION_DIAGNOSTIC_METADATA_COLUMNS
		if column_name in prediction_table.columns
	}
	resolved_model_name = str(metadata.get("model_name", ""))
	if resolved_model_name and "model_label" not in metadata:
		metadata["model_label"] = (
			str(model_labels[resolved_model_name])
			if model_labels is not None and resolved_model_name in model_labels
			else resolved_model_name
		)
	if resolved_model_name and "model_family" not in metadata:
		metadata["model_family"] = (
			str(model_families[resolved_model_name])
			if model_families is not None and resolved_model_name in model_families
			else "unspecified"
		)
	if resolved_model_name and "model_order" not in metadata:
		metadata["model_order"] = (
			int(model_order[resolved_model_name])
			if model_order is not None and resolved_model_name in model_order
			else np.nan
		)

	return metadata


def _resolve_prediction_column_sets(
	prediction_table: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
	actual_columns = [
		str(column_name)
		for column_name in prediction_table.columns
		if str(column_name).startswith("Actual_")
	]
	if not actual_columns:
		raise KeyError("Prediction tables must include at least one 'Actual_' target column.")

	target_names = [column_name.removeprefix("Actual_") for column_name in actual_columns]
	raw_columns = [f"Raw_{target_name}" for target_name in target_names]
	missing_raw_columns = [column_name for column_name in raw_columns if column_name not in prediction_table.columns]
	if missing_raw_columns:
		missing_display = ", ".join(missing_raw_columns)
		raise KeyError(f"Prediction tables are missing required raw prediction columns: {missing_display}")

	affine_columns = [
		f"Affine_{target_name}"
		for target_name in target_names
		if f"Affine_{target_name}" in prediction_table.columns
	]
	projected_columns = [
		f"Projected_{target_name}"
		for target_name in target_names
		if f"Projected_{target_name}" in prediction_table.columns
	]

	return target_names, actual_columns, raw_columns, affine_columns, projected_columns


def summarize_prediction_diagnostics(
	prediction_tables: Sequence[pd.DataFrame],
	*,
	model_labels: Mapping[str, str] | None = None,
	model_families: Mapping[str, str] | None = None,
	model_order: Mapping[str, int] | None = None,
) -> pd.DataFrame:
	"""Summarize negative predictions and constraint diagnostics for each run-level prediction table."""

	if not prediction_tables:
		raise ValueError("prediction_tables must include at least one prediction table.")

	diagnostic_rows: list[dict[str, Any]] = []
	for prediction_table in prediction_tables:
		if not isinstance(prediction_table, pd.DataFrame):
			raise ValueError("prediction_tables must contain only pandas DataFrames.")

		metadata = _extract_prediction_diagnostic_metadata(
			prediction_table,
			model_labels=model_labels,
			model_families=model_families,
			model_order=model_order,
		)
		target_names, _, raw_columns, _, projected_columns = _resolve_prediction_column_sets(prediction_table)
		effective_prediction_type = "projected" if len(projected_columns) == len(target_names) else "raw"
		effective_columns = projected_columns if effective_prediction_type == "projected" else raw_columns
		effective_values = prediction_table.loc[:, effective_columns].to_numpy(dtype=float)
		raw_values = prediction_table.loc[:, raw_columns].to_numpy(dtype=float)
		negative_mask = effective_values < 0.0
		negative_values = effective_values[negative_mask]

		diagnostic_row: dict[str, Any] = {
			**metadata,
			"target_count": len(target_names),
			"sample_count": int(len(prediction_table)),
			"total_predictions": int(effective_values.size),
			"effective_prediction_type": effective_prediction_type,
			"negative_predictions": int(negative_mask.sum()),
			"negative_prediction_rate_pct": (
				100.0 * float(negative_mask.mean()) if effective_values.size > 0 else 0.0
			),
			"samples_with_any_negative": int(negative_mask.any(axis=1).sum()),
			"sample_incidence_rate_pct": (
				100.0 * float(negative_mask.any(axis=1).mean()) if len(prediction_table) > 0 else 0.0
			),
			"minimum_effective_prediction": float(np.min(effective_values)),
			"mean_negative_effective_prediction": (
				float(np.mean(negative_values)) if negative_values.size > 0 else np.nan
			),
			"median_negative_effective_prediction": (
				float(np.median(negative_values)) if negative_values.size > 0 else np.nan
			),
		}

		if effective_prediction_type == "projected":
			raw_to_effective_adjustment = effective_values - raw_values
			adjustment_l2 = np.linalg.norm(raw_to_effective_adjustment, axis=1)
			diagnostic_row.update(
				{
					"raw_to_effective_adjustment_mean_l2": float(np.mean(adjustment_l2)),
					"raw_to_effective_adjustment_max_l2": float(np.max(adjustment_l2)),
					"raw_to_effective_adjustment_mean_abs": float(np.mean(np.abs(raw_to_effective_adjustment))),
					"raw_to_effective_adjustment_max_abs": float(np.max(np.abs(raw_to_effective_adjustment))),
				}
			)
		else:
			diagnostic_row.update(
				{
					"raw_to_effective_adjustment_mean_l2": 0.0,
					"raw_to_effective_adjustment_max_l2": 0.0,
					"raw_to_effective_adjustment_mean_abs": 0.0,
					"raw_to_effective_adjustment_max_abs": 0.0,
				}
			)

		for prediction_type in ["raw", "affine", "projected"]:
			constraint_column = f"{prediction_type}_constraint_l2"
			if constraint_column in prediction_table.columns:
				constraint_values = prediction_table[constraint_column].to_numpy(dtype=float)
				diagnostic_row[f"{constraint_column}_mean"] = float(np.mean(constraint_values))
				diagnostic_row[f"{constraint_column}_max"] = float(np.max(constraint_values))

		effective_constraint_column = (
			"projected_constraint_l2"
			if effective_prediction_type == "projected" and "projected_constraint_l2" in prediction_table.columns
			else "raw_constraint_l2"
			if "raw_constraint_l2" in prediction_table.columns
			else None
		)
		if effective_constraint_column is not None:
			effective_constraint_values = prediction_table[effective_constraint_column].to_numpy(dtype=float)
			diagnostic_row["effective_constraint_l2_mean"] = float(np.mean(effective_constraint_values))
			diagnostic_row["effective_constraint_l2_max"] = float(np.max(effective_constraint_values))

			if (
				effective_constraint_column != "raw_constraint_l2"
				and "raw_constraint_l2" in prediction_table.columns
			):
				raw_constraint_values = prediction_table["raw_constraint_l2"].to_numpy(dtype=float)
				constraint_improvement = raw_constraint_values - effective_constraint_values
				diagnostic_row["constraint_l2_improvement_mean"] = float(np.mean(constraint_improvement))
				diagnostic_row["constraint_l2_improvement_max"] = float(np.max(constraint_improvement))

		diagnostic_rows.append(diagnostic_row)

	return pd.DataFrame(diagnostic_rows)


def summarize_prediction_diagnostics_by_target(
	prediction_tables: Sequence[pd.DataFrame],
	*,
	model_labels: Mapping[str, str] | None = None,
	model_families: Mapping[str, str] | None = None,
	model_order: Mapping[str, int] | None = None,
) -> pd.DataFrame:
	"""Summarize effective negative-prediction behavior for each target in each run."""

	if not prediction_tables:
		raise ValueError("prediction_tables must include at least one prediction table.")

	diagnostic_rows: list[dict[str, Any]] = []
	for prediction_table in prediction_tables:
		if not isinstance(prediction_table, pd.DataFrame):
			raise ValueError("prediction_tables must contain only pandas DataFrames.")

		metadata = _extract_prediction_diagnostic_metadata(
			prediction_table,
			model_labels=model_labels,
			model_families=model_families,
			model_order=model_order,
		)
		target_names, _, raw_columns, _, projected_columns = _resolve_prediction_column_sets(prediction_table)
		effective_prediction_type = "projected" if len(projected_columns) == len(target_names) else "raw"

		for target_index, target_name in enumerate(target_names):
			raw_column = raw_columns[target_index]
			effective_column = (
				projected_columns[target_index]
				if effective_prediction_type == "projected"
				else raw_column
			)
			effective_values = prediction_table[effective_column].to_numpy(dtype=float)
			negative_mask = effective_values < 0.0
			negative_values = effective_values[negative_mask]
			diagnostic_row: dict[str, Any] = {
				**metadata,
				"target": target_name,
				"effective_prediction_type": effective_prediction_type,
				"sample_count": int(len(prediction_table)),
				"negative_predictions": int(negative_mask.sum()),
				"negative_prediction_rate_pct": (
					100.0 * float(np.mean(negative_mask)) if len(prediction_table) > 0 else 0.0
				),
				"minimum_effective_prediction": float(np.min(effective_values)),
				"mean_negative_effective_prediction": (
					float(np.mean(negative_values)) if negative_values.size > 0 else np.nan
				),
				"median_negative_effective_prediction": (
					float(np.median(negative_values)) if negative_values.size > 0 else np.nan
				),
			}

			if effective_prediction_type == "projected":
				raw_values = prediction_table[raw_column].to_numpy(dtype=float)
				adjustment_values = effective_values - raw_values
				diagnostic_row["raw_to_effective_adjustment_mean_abs"] = float(np.mean(np.abs(adjustment_values)))
				diagnostic_row["raw_to_effective_adjustment_max_abs"] = float(np.max(np.abs(adjustment_values)))
			else:
				diagnostic_row["raw_to_effective_adjustment_mean_abs"] = 0.0
				diagnostic_row["raw_to_effective_adjustment_max_abs"] = 0.0

			diagnostic_rows.append(diagnostic_row)

	return pd.DataFrame(diagnostic_rows)


def collate_model_analysis_results(
	analysis_results_by_model: Mapping[str, Mapping[str, Any]],
	*,
	model_labels: Mapping[str, str] | None = None,
	model_families: Mapping[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
	"""Collate model analysis sweeps into shared comparison-ready summary frames."""

	if not analysis_results_by_model:
		raise ValueError("analysis_results_by_model must include at least one model result.")

	resolved_model_labels = {
		str(model_name): (
			str(model_labels[model_name]) if model_labels is not None and model_name in model_labels else str(model_name)
		)
		for model_name in analysis_results_by_model
	}
	resolved_model_families = {
		str(model_name): (
			str(model_families[model_name]) if model_families is not None and model_name in model_families else "unspecified"
		)
		for model_name in analysis_results_by_model
	}
	resolved_model_order = {
		str(model_name): model_index
		for model_index, model_name in enumerate(analysis_results_by_model)
	}

	analysis_config_rows: list[dict[str, Any]] = []
	coverage_rows: list[dict[str, Any]] = []
	run_metadata_frames: list[pd.DataFrame] = []
	aggregate_metric_frames: list[pd.DataFrame] = []
	effective_aggregate_metric_frames: list[pd.DataFrame] = []
	per_target_metric_frames: list[pd.DataFrame] = []
	all_prediction_tables: list[pd.DataFrame] = []

	for model_name, analysis_result in analysis_results_by_model.items():
		resolved_model_name = str(model_name)
		model_label = resolved_model_labels[resolved_model_name]
		model_family = resolved_model_families[resolved_model_name]
		model_index = resolved_model_order[resolved_model_name]

		run_metadata = analysis_result["run_metadata"].copy()
		aggregate_metrics = add_effective_metric_columns(analysis_result["aggregate_metrics"])
		per_target_metrics = add_effective_metric_columns(analysis_result["per_target_metrics"])
		prediction_tables = list(analysis_result.get("prediction_tables", []))
		for frame in [run_metadata, aggregate_metrics, per_target_metrics]:
			frame["model_name"] = resolved_model_name
			frame["model_label"] = model_label
			frame["model_family"] = model_family
			frame["model_order"] = model_index

		run_metadata_frames.append(run_metadata)
		aggregate_metric_frames.append(aggregate_metrics)
		effective_aggregate_metric_frames.append(build_effective_aggregate_metrics(aggregate_metrics))
		per_target_metric_frames.append(per_target_metrics)
		all_prediction_tables.extend(prediction_tables)
		analysis_config_rows.append(
			{
				"model_name": resolved_model_name,
				"model_label": model_label,
				"model_family": model_family,
				"model_order": model_index,
				**dict(analysis_result["analysis_config"]),
			}
		)

		projected_metric_available = any(
			f"projected_{metric_basename}" in per_target_metrics.columns
			and per_target_metrics[f"projected_{metric_basename}"].notna().any()
			for metric_basename in COMPARISON_METRIC_BASENAMES
		)
		affine_metric_available = any(
			f"affine_{metric_basename}" in per_target_metrics.columns
			and per_target_metrics[f"affine_{metric_basename}"].notna().any()
			for metric_basename in COMPARISON_METRIC_BASENAMES
		)
		raw_metric_available = any(
			f"raw_{metric_basename}" in per_target_metrics.columns
			and per_target_metrics[f"raw_{metric_basename}"].notna().any()
			for metric_basename in COMPARISON_METRIC_BASENAMES
		)
		coverage_rows.append(
			{
				"model_name": resolved_model_name,
				"model_label": model_label,
				"model_family": model_family,
				"model_order": model_index,
				"n_analysis_runs": int(len(run_metadata)),
				"n_prediction_tables": int(len(prediction_tables)),
				"n_dataset_sizes": int(run_metadata["dataset_size_total"].nunique()),
				"min_dataset_size_total": int(run_metadata["dataset_size_total"].min()),
				"max_dataset_size_total": int(run_metadata["dataset_size_total"].max()),
				"min_train_size": int(run_metadata["train_size"].min()),
				"max_train_size": int(run_metadata["train_size"].max()),
				"min_test_size": int(run_metadata["test_size"].min()),
				"max_test_size": int(run_metadata["test_size"].max()),
				"n_targets": int(per_target_metrics["target"].nunique()),
				"raw_metric_available": raw_metric_available,
				"projected_metric_available": projected_metric_available,
				"affine_metric_available": affine_metric_available,
				"effective_metric_source": "projected" if projected_metric_available else "raw",
			}
		)

	if all_prediction_tables:
		prediction_diagnostics = summarize_prediction_diagnostics(
			all_prediction_tables,
			model_labels=resolved_model_labels,
			model_families=resolved_model_families,
			model_order=resolved_model_order,
		)
		prediction_target_diagnostics = summarize_prediction_diagnostics_by_target(
			all_prediction_tables,
			model_labels=resolved_model_labels,
			model_families=resolved_model_families,
			model_order=resolved_model_order,
		)
	else:
		prediction_diagnostics = pd.DataFrame()
		prediction_target_diagnostics = pd.DataFrame()

	prediction_diagnostic_sort_columns = [
		"model_order",
		"dataset_size_total",
		"repeat_index",
		"split_name",
	]
	prediction_target_diagnostic_sort_columns = [
		"model_order",
		"dataset_size_total",
		"repeat_index",
		"split_name",
		"target",
	]
	if prediction_diagnostics.empty:
		prediction_diagnostics = pd.DataFrame(columns=prediction_diagnostic_sort_columns)
	if prediction_target_diagnostics.empty:
		prediction_target_diagnostics = pd.DataFrame(columns=prediction_target_diagnostic_sort_columns)

	return {
		"analysis_configs": pd.DataFrame(analysis_config_rows).sort_values("model_order").reset_index(drop=True),
		"coverage": pd.DataFrame(coverage_rows).sort_values("model_order").reset_index(drop=True),
		"run_metadata": pd.concat(run_metadata_frames, ignore_index=True),
		"aggregate_metrics": pd.concat(aggregate_metric_frames, ignore_index=True),
		"effective_aggregate_metrics": pd.concat(effective_aggregate_metric_frames, ignore_index=True),
		"per_target_metrics": pd.concat(per_target_metric_frames, ignore_index=True),
		"prediction_diagnostics": prediction_diagnostics.sort_values(
			prediction_diagnostic_sort_columns
		).reset_index(drop=True),
		"prediction_target_diagnostics": prediction_target_diagnostics.sort_values(
			prediction_target_diagnostic_sort_columns
		).reset_index(drop=True),
	}


__all__ = [
	"COMPARISON_METRIC_BASENAMES",
	"COMPARISON_METRIC_DIRECTIONS",
	"add_effective_metric_columns",
	"build_icsor_coupled_qp_b_matrix_block_frames",
	"build_icsor_coupled_qp_b_matrix_block_interpretation_table",
	"build_icsor_coupled_qp_b_matrix_block_metadata",
	"build_icsor_coupled_qp_coefficient_contract_table",
	"build_icsor_coupled_qp_coefficient_density_tables",
	"build_icsor_coupled_qp_coefficient_frames",
	"build_icsor_coupled_qp_coefficient_metadata",
	"build_icsor_coupled_qp_response_surface_prediction_data",
	"build_notebook_table_recorder",
	"build_effective_aggregate_metrics",
	"build_train_test_gap_summary",
	"ModelRunner",
	"build_icsor_response_surface_prediction_data",
	"build_dataset_size_schedule",
	"collate_model_analysis_results",
	"describe_and_display_table",
	"get_metric_direction",
	"is_higher_better_metric",
	"load_latest_analysis_result",
	"load_latest_classical_training_context",
	"load_latest_icsor_training_context",
	"load_latest_named_table_artifacts",
	"load_icsor_response_surface_defaults",
	"load_analysis_defaults",
	"persist_analysis_result_artifacts",
	"persist_classical_training_context",
	"persist_icsor_training_context",
	"persist_named_table_artifacts",
	"rank_metric_summary",
	"run_model_dataset_size_analysis",
	"summarize_metric_distribution",
	"summarize_prediction_diagnostics",
	"summarize_prediction_diagnostics_by_target",
]

