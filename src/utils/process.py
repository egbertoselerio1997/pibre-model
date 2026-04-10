"""Reusable preprocessing helpers for supervised machine-learning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SupervisedDatasetFrames:
	"""Container for aligned supervised-learning dataframes."""

	features: pd.DataFrame
	targets: pd.DataFrame
	constraint_reference: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplit:
	"""Container for one supervised-learning split."""

	features: pd.DataFrame
	targets: pd.DataFrame
	constraint_reference: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplits:
	"""Container for train/validation/test splits."""

	train: DatasetSplit
	validation: DatasetSplit
	test: DatasetSplit


@dataclass(frozen=True)
class TrainTestDatasetSplits:
	"""Container for train/test splits."""

	train: DatasetSplit
	test: DatasetSplit


@dataclass(frozen=True)
class TrainTestSplitIndices:
	"""Container for authoritative train/test row indices."""

	train: pd.Index
	test: pd.Index


@dataclass(frozen=True)
class ScalingBundle:
	"""Fitted scalers and the column order they expect."""

	feature_scaler: StandardScaler | None
	target_scaler: StandardScaler | None
	feature_columns: list[str]
	target_columns: list[str]


def _ensure_columns_exist(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
	missing_columns = [column_name for column_name in required_columns if column_name not in frame.columns]
	if missing_columns:
		missing_display = ", ".join(missing_columns)
		raise KeyError(f"Dataset is missing required columns: {missing_display}")


def compute_measured_composites(
	dataset: pd.DataFrame,
	state_columns: list[str],
	composition_matrix: np.ndarray,
	measured_output_columns: list[str],
	*,
	state_prefix: str,
	output_prefix: str,
) -> pd.DataFrame:
	"""Map state columns into measured composite space with the provided composition matrix."""

	prefixed_state_columns = [f"{state_prefix}{state_name}" for state_name in state_columns]
	_ensure_columns_exist(dataset, prefixed_state_columns)
	composition_array = np.asarray(composition_matrix, dtype=float)
	expected_shape = (len(measured_output_columns), len(state_columns))
	if composition_array.shape != expected_shape:
		raise ValueError(
			"composition_matrix shape must match measured_output_columns x state_columns for measured composite mapping."
		)

	state_values = dataset.loc[:, prefixed_state_columns].to_numpy(dtype=float)
	composite_values = state_values @ composition_array.T
	output_columns = [f"{output_prefix}{column_name}" for column_name in measured_output_columns]

	return pd.DataFrame(composite_values, index=dataset.index, columns=output_columns)


def has_active_projection(A_matrix: np.ndarray) -> bool:
	"""Return whether an invariant matrix defines a non-trivial projection."""

	constraint_matrix = np.asarray(A_matrix, dtype=float)
	if constraint_matrix.ndim != 2:
		raise ValueError("A_matrix must be two-dimensional.")

	return bool(constraint_matrix.shape[0] > 0)


def build_projection_operator(A_matrix: np.ndarray) -> np.ndarray:
	"""Construct the orthogonal projector used to enforce measured-space constraints."""

	constraint_matrix = np.asarray(A_matrix, dtype=float)
	if constraint_matrix.ndim != 2:
		raise ValueError("A_matrix must be two-dimensional.")
	if not has_active_projection(constraint_matrix):
		return np.zeros((constraint_matrix.shape[1], constraint_matrix.shape[1]), dtype=float)

	gram_matrix = constraint_matrix @ constraint_matrix.T
	return constraint_matrix.T @ np.linalg.pinv(gram_matrix) @ constraint_matrix


def project_to_mass_balance(
	raw_predictions: np.ndarray,
	constraint_reference: np.ndarray,
	A_matrix: np.ndarray,
) -> np.ndarray:
	"""Project raw predictions onto the measured-space invariant subspace.

	When the invariant matrix is trivial, projection is inactive and the raw
	predictions are returned unchanged.
	"""

	raw_array = np.asarray(raw_predictions, dtype=float)
	if not has_active_projection(A_matrix):
		return raw_array.copy()

	projection_operator = build_projection_operator(A_matrix)
	reference_array = np.asarray(constraint_reference, dtype=float)
	return raw_array - (raw_array - reference_array) @ projection_operator.T


def _as_two_dimensional_array(values: np.ndarray, *, name: str) -> np.ndarray:
	array_values = np.asarray(values, dtype=float)
	if array_values.ndim == 1:
		return array_values.reshape(1, -1)
	if array_values.ndim != 2:
		raise ValueError(f"{name} must be one- or two-dimensional.")
	return array_values


def build_null_space_basis(A_matrix: np.ndarray) -> np.ndarray:
	"""Build an orthonormal basis for the admissible null-space directions."""

	constraint_matrix = np.asarray(A_matrix, dtype=float)
	if constraint_matrix.ndim != 2:
		raise ValueError("A_matrix must be two-dimensional.")
	if not has_active_projection(constraint_matrix):
		return np.eye(constraint_matrix.shape[1], dtype=float)

	basis = null_space(constraint_matrix)
	if basis.size == 0:
		return np.zeros((constraint_matrix.shape[1], 0), dtype=float)
	return np.asarray(basis, dtype=float)


def _compute_constraint_residuals(
	predictions: np.ndarray,
	constraint_reference: np.ndarray,
	constraint_matrix: np.ndarray,
) -> np.ndarray:
	if not has_active_projection(constraint_matrix):
		return np.zeros((predictions.shape[0], 0), dtype=float)
	return (constraint_matrix @ predictions.T - constraint_matrix @ constraint_reference.T).T


def _compute_feasibility_diagnostics(
	predictions: np.ndarray,
	constraint_reference: np.ndarray,
	constraint_matrix: np.ndarray,
	*,
	constraint_tolerance: float,
	nonnegativity_tolerance: float,
) -> dict[str, np.ndarray]:
	residuals = _compute_constraint_residuals(predictions, constraint_reference, constraint_matrix)
	if residuals.shape[1] == 0:
		constraint_max_abs = np.zeros(predictions.shape[0], dtype=float)
		constraint_feasible = np.ones(predictions.shape[0], dtype=bool)
	else:
		constraint_max_abs = np.max(np.abs(residuals), axis=1)
		constraint_feasible = constraint_max_abs <= float(constraint_tolerance)

	minimum_component = np.min(predictions, axis=1)
	nonnegative_feasible = minimum_component >= -float(nonnegativity_tolerance)
	return {
		"constraint_feasible": constraint_feasible,
		"constraint_max_abs": constraint_max_abs,
		"minimum_component": minimum_component,
		"nonnegative_feasible": nonnegative_feasible,
	}


def _build_reduced_lp_projection_template(
	null_space_basis: np.ndarray,
	composition_matrix: np.ndarray,
	*,
	measured_deviation_weight: float,
	component_deviation_weight: float,
	tradeoff_parameter: float,
) -> dict[str, Any]:
	null_basis = np.asarray(null_space_basis, dtype=float)
	composition_array = np.asarray(composition_matrix, dtype=float)
	state_dimension = int(null_basis.shape[0])
	null_space_dimension = int(null_basis.shape[1])
	output_dimension = int(composition_array.shape[0])
	if composition_array.ndim != 2:
		raise ValueError("composition_matrix must be two-dimensional.")
	if composition_array.shape[1] != state_dimension:
		raise ValueError("composition_matrix column count must match the prediction dimension.")

	variable_count = null_space_dimension + output_dimension + state_dimension
	objective = np.zeros(variable_count, dtype=float)
	objective[null_space_dimension : null_space_dimension + output_dimension] = float(measured_deviation_weight)
	objective[null_space_dimension + output_dimension :] = (
		float(tradeoff_parameter) * float(component_deviation_weight)
	)

	zero_state_output = np.zeros((state_dimension, output_dimension), dtype=float)
	zero_output_state = np.zeros((output_dimension, state_dimension), dtype=float)
	composed_basis = composition_array @ null_basis
	upper_bound_matrix = np.vstack(
		[
			np.hstack([-null_basis, zero_state_output, np.zeros((state_dimension, state_dimension), dtype=float)]),
			np.hstack([composed_basis, -np.eye(output_dimension, dtype=float), zero_output_state]),
			np.hstack([-composed_basis, -np.eye(output_dimension, dtype=float), zero_output_state]),
			np.hstack([null_basis, zero_state_output, -np.eye(state_dimension, dtype=float)]),
			np.hstack([-null_basis, zero_state_output, -np.eye(state_dimension, dtype=float)]),
		]
	)
	bounds = [(None, None)] * null_space_dimension + [(0.0, None)] * (output_dimension + state_dimension)
	return {
		"objective": objective,
		"A_ub": upper_bound_matrix,
		"bounds": bounds,
		"null_space_dimension": null_space_dimension,
		"output_dimension": output_dimension,
		"state_dimension": state_dimension,
	}


def _run_highs_lp_projection(
	affine_point: np.ndarray,
	lp_template: dict[str, Any],
	*,
	highs_presolve: bool,
	highs_max_iter: int,
	highs_verbose: bool,
):
	state_dimension = int(lp_template["state_dimension"])
	output_dimension = int(lp_template["output_dimension"])
	b_ub = np.concatenate(
		[
			np.asarray(affine_point, dtype=float),
			np.zeros(2 * output_dimension + 2 * state_dimension, dtype=float),
		]
	)
	options = {
		"presolve": bool(highs_presolve),
		"disp": bool(highs_verbose),
		"maxiter": int(highs_max_iter),
	}
	return linprog(
		c=np.asarray(lp_template["objective"], dtype=float),
		A_ub=np.asarray(lp_template["A_ub"], dtype=float),
		b_ub=b_ub,
		bounds=list(lp_template["bounds"]),
		method="highs",
		options=options,
	)


def _solve_reduced_nonnegative_lp_projection(
	affine_point: np.ndarray,
	constraint_reference: np.ndarray,
	constraint_matrix: np.ndarray,
	null_space_basis: np.ndarray,
	composition_matrix: np.ndarray,
	*,
	lp_template: dict[str, Any],
	highs_presolve: bool,
	highs_max_iter: int,
	highs_verbose: bool,
	highs_retry_without_presolve: bool,
	constraint_tolerance: float,
	nonnegativity_tolerance: float,
) -> tuple[np.ndarray, str, int]:
	null_space_dimension = int(lp_template["null_space_dimension"])
	affine_array = np.asarray(affine_point, dtype=float)
	if null_space_dimension == 0:
		if np.min(affine_array) < -float(nonnegativity_tolerance):
			raise RuntimeError(
				"HiGHS could not repair the icsor nonnegative LP because the affine reference has no admissible null-space directions."
			)
		return affine_array.copy(), "not_needed", 0

	presolve_attempts = [bool(highs_presolve)]
	if bool(highs_presolve) and bool(highs_retry_without_presolve):
		presolve_attempts.append(False)

	last_error: str | None = None
	for attempt_index, attempt_presolve in enumerate(presolve_attempts):
		result = _run_highs_lp_projection(
			affine_array,
			lp_template,
			highs_presolve=attempt_presolve,
			highs_max_iter=highs_max_iter,
			highs_verbose=highs_verbose,
		)
		message = str(getattr(result, "message", "")).strip()
		if not bool(getattr(result, "success", False)) or result.x is None:
			last_error = (
				"HiGHS failed to solve the icsor nonnegative LP projection"
				f" (status={getattr(result, 'status', 'unknown')}, presolve={attempt_presolve}): {message or 'no message'}."
			)
			if attempt_index < len(presolve_attempts) - 1:
				continue
			raise RuntimeError(last_error)

		projected_point = affine_array + np.asarray(null_space_basis, dtype=float) @ np.asarray(
			result.x[:null_space_dimension],
			dtype=float,
		)
		if np.min(projected_point) < -float(nonnegativity_tolerance):
			last_error = "HiGHS returned a icsor LP projection that still violates the nonnegativity tolerance."
			if attempt_index < len(presolve_attempts) - 1:
				continue
			raise RuntimeError(last_error)
		projected_point = projected_point.copy()
		projected_point[(projected_point < 0.0) & (projected_point >= -float(nonnegativity_tolerance))] = 0.0

		residuals = _compute_constraint_residuals(
			projected_point.reshape(1, -1),
			np.asarray(constraint_reference, dtype=float).reshape(1, -1),
			constraint_matrix,
		)
		if residuals.shape[1] > 0 and float(np.max(np.abs(residuals))) > float(constraint_tolerance):
			last_error = "HiGHS returned a icsor LP projection that violates the invariant-equality tolerance."
			if attempt_index < len(presolve_attempts) - 1:
				continue
			raise RuntimeError(last_error)

		if attempt_index > 0 and not attempt_presolve:
			status = "optimal_retry_without_presolve"
		elif attempt_presolve:
			status = "optimal"
		else:
			status = "optimal_no_presolve"
		return projected_point, status, int(getattr(result, "nit", 0) or 0)

	raise RuntimeError(last_error or "HiGHS failed to solve the icsor nonnegative LP projection.")


def project_to_nonnegative_feasible_set(
	raw_predictions: np.ndarray,
	constraint_reference: np.ndarray,
	A_matrix: np.ndarray,
	composition_matrix: np.ndarray,
	*,
	projection_operator: np.ndarray | None = None,
	projection_complement: np.ndarray | None = None,
	constraint_tolerance: float,
	nonnegativity_tolerance: float,
	projection_solver: str,
	measured_deviation_weight: float,
	component_deviation_weight: float,
	tradeoff_parameter: float,
	highs_presolve: bool,
	highs_max_iter: int,
	highs_verbose: bool,
	highs_retry_without_presolve: bool,
) -> dict[str, np.ndarray]:
	"""Project component predictions onto the invariant-consistent nonnegative set."""

	raw_array = _as_two_dimensional_array(raw_predictions, name="raw_predictions")
	reference_array = _as_two_dimensional_array(constraint_reference, name="constraint_reference")
	if raw_array.shape != reference_array.shape:
		raise ValueError("raw_predictions and constraint_reference must share the same shape.")
	composition_array = np.asarray(composition_matrix, dtype=float)
	if composition_array.ndim != 2:
		raise ValueError("composition_matrix must be two-dimensional.")
	if composition_array.shape[1] != raw_array.shape[1]:
		raise ValueError("composition_matrix column count must match the prediction dimension.")

	constraint_matrix = np.asarray(A_matrix, dtype=float)
	if constraint_matrix.ndim != 2:
		raise ValueError("A_matrix must be two-dimensional.")
	if constraint_matrix.shape[1] != raw_array.shape[1]:
		raise ValueError("A_matrix column count must match the prediction dimension.")

	projection_active = has_active_projection(constraint_matrix)
	if projection_operator is None:
		projection_operator = build_projection_operator(constraint_matrix)
	projection_matrix = np.asarray(projection_operator, dtype=float)
	if projection_matrix.shape != (raw_array.shape[1], raw_array.shape[1]):
		raise ValueError("projection_operator must be square with the prediction dimension.")

	if projection_complement is None:
		projection_complement = np.eye(projection_matrix.shape[0], dtype=float) - projection_matrix
	projection_complement_array = np.asarray(projection_complement, dtype=float)
	if projection_complement_array.shape != projection_matrix.shape:
		raise ValueError("projection_complement must match projection_operator shape.")

	raw_diagnostics = _compute_feasibility_diagnostics(
		raw_array,
		reference_array,
		constraint_matrix,
		constraint_tolerance=constraint_tolerance,
		nonnegativity_tolerance=nonnegativity_tolerance,
	)
	raw_feasible_mask = raw_diagnostics["constraint_feasible"] & raw_diagnostics["nonnegative_feasible"]

	if projection_active:
		affine_predictions = raw_array @ projection_complement_array.T + reference_array @ projection_matrix.T
	else:
		affine_predictions = raw_array.copy()

	affine_diagnostics = _compute_feasibility_diagnostics(
		affine_predictions,
		reference_array,
		constraint_matrix,
		constraint_tolerance=constraint_tolerance,
		nonnegativity_tolerance=nonnegativity_tolerance,
	)
	affine_feasible_mask = affine_diagnostics["constraint_feasible"] & affine_diagnostics["nonnegative_feasible"]

	projected_predictions = affine_predictions.copy()
	projected_predictions[raw_feasible_mask, :] = raw_array[raw_feasible_mask, :]
	projection_stage = np.full(raw_array.shape[0], "affine_feasible", dtype=object)
	projection_stage[raw_feasible_mask] = "raw_feasible"
	lp_active_mask = np.zeros(raw_array.shape[0], dtype=bool)
	solver_status = np.full(raw_array.shape[0], "not_needed", dtype=object)
	solver_iterations = np.zeros(raw_array.shape[0], dtype=int)
	lp_active_mask = ~(raw_feasible_mask | affine_feasible_mask)
	projection_stage[lp_active_mask] = "lp_corrected"
	if np.any(lp_active_mask):
		if str(projection_solver).strip().lower() != "highs":
			raise ValueError("icsor nonnegative projection requires projection_solver='highs'.")
		null_space_basis = build_null_space_basis(constraint_matrix)
		lp_template = _build_reduced_lp_projection_template(
			null_space_basis,
			composition_array,
			measured_deviation_weight=measured_deviation_weight,
			component_deviation_weight=component_deviation_weight,
			tradeoff_parameter=tradeoff_parameter,
		)
		for row_index in np.flatnonzero(lp_active_mask):
			projected_point, row_status, row_iterations = _solve_reduced_nonnegative_lp_projection(
				affine_predictions[row_index, :],
				reference_array[row_index, :],
				constraint_matrix,
				null_space_basis,
				composition_array,
				lp_template=lp_template,
				highs_presolve=highs_presolve,
				highs_max_iter=highs_max_iter,
				highs_verbose=highs_verbose,
				highs_retry_without_presolve=highs_retry_without_presolve,
				constraint_tolerance=constraint_tolerance,
				nonnegativity_tolerance=nonnegativity_tolerance,
			)
			projected_predictions[row_index, :] = projected_point
			solver_status[row_index] = row_status
			solver_iterations[row_index] = row_iterations

	projected_diagnostics = _compute_feasibility_diagnostics(
		projected_predictions,
		reference_array,
		constraint_matrix,
		constraint_tolerance=constraint_tolerance,
		nonnegativity_tolerance=nonnegativity_tolerance,
	)
	projected_feasible_mask = projected_diagnostics["constraint_feasible"] & projected_diagnostics["nonnegative_feasible"]
	if not bool(np.all(projected_feasible_mask)):
		raise RuntimeError("icsor nonnegative projection produced an infeasible deployed component state.")

	return {
		"affine_predictions": affine_predictions,
		"projected_predictions": projected_predictions,
		"raw_feasible_mask": raw_feasible_mask,
		"affine_feasible_mask": affine_feasible_mask,
		"lp_active_mask": lp_active_mask,
		"projection_stage": projection_stage,
		"solver_status": solver_status,
		"solver_iterations": solver_iterations,
		"raw_constraint_max_abs": raw_diagnostics["constraint_max_abs"],
		"affine_constraint_max_abs": affine_diagnostics["constraint_max_abs"],
		"projected_constraint_max_abs": projected_diagnostics["constraint_max_abs"],
		"raw_min_component": raw_diagnostics["minimum_component"],
		"affine_min_component": affine_diagnostics["minimum_component"],
		"projected_min_component": projected_diagnostics["minimum_component"],
		"projection_active": np.full(raw_array.shape[0], projection_active, dtype=bool),
	}


def build_measured_supervised_dataset(
	dataset: pd.DataFrame,
	metadata: dict[str, Any],
	composition_matrix: np.ndarray,
) -> SupervisedDatasetFrames:
	"""Build measured-space features, targets, and projection references from the simulation metadata contract."""

	state_columns = list(metadata["state_columns"])
	measured_output_columns = list(metadata["measured_output_columns"])
	operational_columns = list(metadata["operational_columns"])
	target_columns = [f"Out_{column_name}" for column_name in measured_output_columns]

	_ensure_columns_exist(dataset, operational_columns)
	_ensure_columns_exist(dataset, target_columns)

	influent_measured = compute_measured_composites(
		dataset,
		state_columns,
		composition_matrix,
		measured_output_columns,
		state_prefix="In_",
		output_prefix="In_",
	)
	constraint_reference = influent_measured.rename(columns=lambda column_name: column_name.replace("In_", "", 1))
	targets = dataset.loc[:, target_columns].copy()
	features = pd.concat([dataset.loc[:, operational_columns].copy(), influent_measured], axis=1)

	return SupervisedDatasetFrames(
		features=features,
		targets=targets,
		constraint_reference=constraint_reference,
	)

def build_fractional_input_measured_output_dataset(
	dataset: pd.DataFrame,
	metadata: dict[str, Any],
	composition_matrix: np.ndarray,
) -> SupervisedDatasetFrames:
	"""Build the icsor-aligned classical benchmark dataset.

	Features stay in the operational-plus-fractional influent basis so the classical
	regressors consume the same inputs as icsor, while the constraint reference
	remains in measured composite space for post-projection diagnostics.
	"""

	state_columns = list(metadata["state_columns"])
	measured_output_columns = list(metadata["measured_output_columns"])
	operational_columns = list(metadata["operational_columns"])
	influent_state_columns = [f"In_{column_name}" for column_name in state_columns]
	target_columns = [f"Out_{column_name}" for column_name in measured_output_columns]

	_ensure_columns_exist(dataset, operational_columns)
	_ensure_columns_exist(dataset, influent_state_columns)
	_ensure_columns_exist(dataset, target_columns)

	influent_measured = compute_measured_composites(
		dataset,
		state_columns,
		composition_matrix,
		measured_output_columns,
		state_prefix="In_",
		output_prefix="In_",
	)
	features = pd.concat(
		[
			dataset.loc[:, operational_columns].copy(),
			dataset.loc[:, influent_state_columns].copy(),
		],
		axis=1,
	)
	targets = dataset.loc[:, target_columns].copy()
	constraint_reference = influent_measured.rename(columns=lambda column_name: column_name.replace("In_", "", 1))

	return SupervisedDatasetFrames(
		features=features,
		targets=targets,
		constraint_reference=constraint_reference,
	)


def build_icsor_supervised_dataset(
	dataset: pd.DataFrame,
	metadata: dict[str, Any],
	composition_matrix: np.ndarray,
) -> SupervisedDatasetFrames:
	"""Build the strict icsor dataset with fractional influent states and measured effluent targets."""

	state_columns = list(metadata["state_columns"])
	measured_output_columns = list(metadata["measured_output_columns"])
	operational_columns = list(metadata["operational_columns"])
	influent_state_columns = [f"In_{column_name}" for column_name in state_columns]
	target_columns = [f"Out_{column_name}" for column_name in measured_output_columns]

	_ensure_columns_exist(dataset, operational_columns)
	_ensure_columns_exist(dataset, influent_state_columns)
	_ensure_columns_exist(dataset, target_columns)

	features = pd.concat(
		[
			dataset.loc[:, operational_columns].copy(),
			dataset.loc[:, influent_state_columns].copy(),
		],
		axis=1,
	)
	constraint_reference = dataset.loc[:, influent_state_columns].copy().rename(
		columns=lambda column_name: column_name.replace("In_", "", 1)
	)
	targets = dataset.loc[:, target_columns].copy()

	if np.asarray(composition_matrix, dtype=float).shape != (len(measured_output_columns), len(state_columns)):
		raise ValueError(
			"composition_matrix shape must match measured_output_columns x state_columns for the icsor contract."
		)

	return SupervisedDatasetFrames(
		features=features,
		targets=targets,
		constraint_reference=constraint_reference,
	)


def _split_frame(frame: pd.DataFrame, indices: pd.Index) -> pd.DataFrame:
	return frame.loc[indices].copy()


def _select_dataset_split(
	dataset: SupervisedDatasetFrames | DatasetSplit,
	indices: pd.Index,
) -> DatasetSplit:
	return DatasetSplit(
		features=_split_frame(dataset.features, indices),
		targets=_split_frame(dataset.targets, indices),
		constraint_reference=_split_frame(dataset.constraint_reference, indices),
	)


def select_dataset_rows(
	dataset: SupervisedDatasetFrames | DatasetSplit,
	indices: pd.Index,
) -> DatasetSplit:
	"""Select aligned rows from one supervised dataset using explicit indices."""

	return _select_dataset_split(dataset, pd.Index(indices))


def make_train_test_split_indices(
	indices: pd.Index | np.ndarray | list[Any],
	*,
	test_fraction: float,
	random_seed: int,
) -> TrainTestSplitIndices:
	"""Create a reproducible authoritative train/test index split."""

	if not 0.0 < test_fraction < 1.0:
		raise ValueError("test_fraction must be between 0 and 1.")

	all_indices = pd.Index(indices)
	train_indices, test_indices = train_test_split(
		all_indices.to_numpy(),
		test_size=test_fraction,
		random_state=random_seed,
		shuffle=True,
	)

	return TrainTestSplitIndices(
		train=pd.Index(train_indices),
		test=pd.Index(test_indices),
	)


def apply_train_test_split_indices(
	supervised_dataset: SupervisedDatasetFrames | DatasetSplit,
	split_indices: TrainTestSplitIndices,
) -> TrainTestDatasetSplits:
	"""Apply authoritative train/test indices to one supervised dataset."""

	return TrainTestDatasetSplits(
		train=_select_dataset_split(supervised_dataset, split_indices.train),
		test=_select_dataset_split(supervised_dataset, split_indices.test),
	)


def sample_dataset_split_indices(
	indices: pd.Index | np.ndarray | list[Any],
	*,
	fraction: float,
	random_seed: int,
) -> pd.Index:
	"""Sample a reproducible subset of row indices from one prepared split."""

	if not 0.0 < fraction <= 1.0:
		raise ValueError("fraction must be between 0 and 1.")

	all_indices = pd.Index(indices)
	if fraction == 1.0:
		return all_indices.copy()

	sampled_indices, _ = train_test_split(
		all_indices.to_numpy(),
		train_size=fraction,
		random_state=random_seed,
		shuffle=True,
	)
	return pd.Index(sampled_indices)


def make_train_test_split(
	supervised_dataset: SupervisedDatasetFrames | DatasetSplit,
	*,
	test_fraction: float,
	random_seed: int,
) -> TrainTestDatasetSplits:
	"""Create a reproducible train/test split with aligned indices."""

	split_indices = make_train_test_split_indices(
		supervised_dataset.features.index,
		test_fraction=test_fraction,
		random_seed=random_seed,
	)
	return apply_train_test_split_indices(supervised_dataset, split_indices)


def sample_dataset_fraction(
	dataset_split: DatasetSplit,
	*,
	fraction: float,
	random_seed: int,
) -> DatasetSplit:
	"""Sample a reproducible subset from one prepared dataset split."""

	if fraction == 1.0:
		return DatasetSplit(
			features=dataset_split.features.copy(),
			targets=dataset_split.targets.copy(),
			constraint_reference=dataset_split.constraint_reference.copy(),
		)

	sampled_indices = sample_dataset_split_indices(
		dataset_split.features.index,
		fraction=fraction,
		random_seed=random_seed,
	)
	return _select_dataset_split(dataset_split, sampled_indices)


def make_train_validation_test_splits(
	supervised_dataset: SupervisedDatasetFrames,
	*,
	test_fraction: float,
	validation_fraction: float,
	random_seed: int,
) -> DatasetSplits:
	"""Create reproducible train, validation, and test splits with aligned indices."""

	if not 0.0 < test_fraction < 1.0:
		raise ValueError("test_fraction must be between 0 and 1.")
	if not 0.0 <= validation_fraction < 1.0:
		raise ValueError("validation_fraction must be between 0 and 1.")
	if test_fraction + validation_fraction >= 1.0:
		raise ValueError("test_fraction plus validation_fraction must be less than 1.")

	all_indices = supervised_dataset.features.index.to_numpy()
	train_validation_indices, test_indices = train_test_split(
		all_indices,
		test_size=test_fraction,
		random_state=random_seed,
		shuffle=True,
	)

	if validation_fraction == 0.0:
		train_indices = train_validation_indices
		validation_indices = np.array([], dtype=train_validation_indices.dtype)
	else:
		adjusted_validation_fraction = validation_fraction / (1.0 - test_fraction)
		train_indices, validation_indices = train_test_split(
			train_validation_indices,
			test_size=adjusted_validation_fraction,
			random_state=random_seed,
			shuffle=True,
		)

	train_index = pd.Index(train_indices)
	validation_index = pd.Index(validation_indices)
	test_index = pd.Index(test_indices)

	return DatasetSplits(
		train=DatasetSplit(
			features=_split_frame(supervised_dataset.features, train_index),
			targets=_split_frame(supervised_dataset.targets, train_index),
			constraint_reference=_split_frame(supervised_dataset.constraint_reference, train_index),
		),
		validation=DatasetSplit(
			features=_split_frame(supervised_dataset.features, validation_index),
			targets=_split_frame(supervised_dataset.targets, validation_index),
			constraint_reference=_split_frame(supervised_dataset.constraint_reference, validation_index),
		),
		test=DatasetSplit(
			features=_split_frame(supervised_dataset.features, test_index),
			targets=_split_frame(supervised_dataset.targets, test_index),
			constraint_reference=_split_frame(supervised_dataset.constraint_reference, test_index),
		),
	)


def fit_scalers(
	train_split: DatasetSplit,
	*,
	scale_features: bool,
	scale_targets: bool,
) -> ScalingBundle:
	"""Fit feature and optional target scalers on the training split."""

	feature_scaler = None
	if scale_features:
		feature_scaler = StandardScaler()
		feature_scaler.fit(train_split.features)

	target_scaler = None
	if scale_targets:
		target_scaler = StandardScaler()
		target_scaler.fit(train_split.targets)

	return ScalingBundle(
		feature_scaler=feature_scaler,
		target_scaler=target_scaler,
		feature_columns=list(train_split.features.columns),
		target_columns=list(train_split.targets.columns),
	)


def _transform_frame(frame: pd.DataFrame, scaler: StandardScaler | None) -> pd.DataFrame:
	if scaler is None:
		return frame.copy()

	transformed_values = scaler.transform(frame)
	return pd.DataFrame(transformed_values, index=frame.index, columns=frame.columns)


def transform_dataset_split(split: DatasetSplit, scaling_bundle: ScalingBundle) -> DatasetSplit:
	"""Apply fitted scalers to one split while preserving dataframe structure."""

	return DatasetSplit(
		features=_transform_frame(split.features.loc[:, scaling_bundle.feature_columns], scaling_bundle.feature_scaler),
		targets=_transform_frame(split.targets.loc[:, scaling_bundle.target_columns], scaling_bundle.target_scaler),
		constraint_reference=split.constraint_reference.copy(),
	)


def transform_dataset_splits(dataset_splits: DatasetSplits, scaling_bundle: ScalingBundle) -> DatasetSplits:
	"""Apply fitted scalers to all splits."""

	return DatasetSplits(
		train=transform_dataset_split(dataset_splits.train, scaling_bundle),
		validation=transform_dataset_split(dataset_splits.validation, scaling_bundle),
		test=transform_dataset_split(dataset_splits.test, scaling_bundle),
	)


def combine_dataset_splits(*splits: DatasetSplit) -> DatasetSplit:
	"""Concatenate multiple dataset splits into one aligned split."""

	if not splits:
		raise ValueError("At least one dataset split is required.")

	return DatasetSplit(
		features=pd.concat([split.features for split in splits], axis=0),
		targets=pd.concat([split.targets for split in splits], axis=0),
		constraint_reference=pd.concat([split.constraint_reference for split in splits], axis=0),
	)


def inverse_transform_targets(values: np.ndarray | pd.DataFrame, scaling_bundle: ScalingBundle) -> np.ndarray:
	"""Inverse-transform target values when target scaling is enabled."""

	array_values = np.asarray(values, dtype=float)
	if scaling_bundle.target_scaler is None:
		return array_values

	return scaling_bundle.target_scaler.inverse_transform(array_values)


__all__ = [
	"DatasetSplit",
	"DatasetSplits",
	"ScalingBundle",
	"SupervisedDatasetFrames",
	"TrainTestDatasetSplits",
	"build_null_space_basis",
	"build_projection_operator",
	"build_icsor_supervised_dataset",
	"build_measured_supervised_dataset",
	"combine_dataset_splits",
	"compute_measured_composites",
	"fit_scalers",
	"has_active_projection",
	"inverse_transform_targets",
	"make_train_validation_test_splits",
	"make_train_test_split",
	"make_train_test_split_indices",
	"project_to_mass_balance",
	"project_to_nonnegative_feasible_set",
	"select_dataset_rows",
	"apply_train_test_split_indices",
	"sample_dataset_fraction",
	"sample_dataset_split_indices",
	"build_fractional_input_measured_output_dataset",
	"transform_dataset_split",
	"transform_dataset_splits",
	"TrainTestSplitIndices",
]

