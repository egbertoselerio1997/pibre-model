"""Reusable preprocessing helpers for supervised machine-learning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
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

	state_values = dataset.loc[:, prefixed_state_columns].to_numpy(dtype=float)
	composite_values = state_values @ np.asarray(composition_matrix, dtype=float).T
	output_columns = [f"{output_prefix}{column_name}" for column_name in measured_output_columns]

	return pd.DataFrame(composite_values, index=dataset.index, columns=output_columns)


def build_projection_operator(A_matrix: np.ndarray) -> np.ndarray:
	"""Construct the orthogonal projector used to enforce measured-space constraints."""

	constraint_matrix = np.asarray(A_matrix, dtype=float)
	gram_matrix = constraint_matrix @ constraint_matrix.T
	return constraint_matrix.T @ np.linalg.pinv(gram_matrix) @ constraint_matrix


def project_to_mass_balance(
	raw_predictions: np.ndarray,
	constraint_reference: np.ndarray,
	A_matrix: np.ndarray,
) -> np.ndarray:
	"""Project raw predictions onto the measured-space invariant subspace."""

	projection_operator = build_projection_operator(A_matrix)
	raw_array = np.asarray(raw_predictions, dtype=float)
	reference_array = np.asarray(constraint_reference, dtype=float)
	return raw_array - (raw_array - reference_array) @ projection_operator.T


def build_measured_supervised_dataset(
	dataset: pd.DataFrame,
	metadata: dict[str, Any],
	composition_matrix: np.ndarray,
) -> SupervisedDatasetFrames:
	"""Build measured-space features, targets, and projection references from the ASM1 contract."""

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


def make_train_test_split(
	supervised_dataset: SupervisedDatasetFrames | DatasetSplit,
	*,
	test_fraction: float,
	random_seed: int,
) -> TrainTestDatasetSplits:
	"""Create a reproducible train/test split with aligned indices."""

	if not 0.0 < test_fraction < 1.0:
		raise ValueError("test_fraction must be between 0 and 1.")

	all_indices = supervised_dataset.features.index.to_numpy()
	train_indices, test_indices = train_test_split(
		all_indices,
		test_size=test_fraction,
		random_state=random_seed,
		shuffle=True,
	)

	train_index = pd.Index(train_indices)
	test_index = pd.Index(test_indices)

	return TrainTestDatasetSplits(
		train=_select_dataset_split(supervised_dataset, train_index),
		test=_select_dataset_split(supervised_dataset, test_index),
	)


def sample_dataset_fraction(
	dataset_split: DatasetSplit,
	*,
	fraction: float,
	random_seed: int,
) -> DatasetSplit:
	"""Sample a reproducible subset from one prepared dataset split."""

	if not 0.0 < fraction <= 1.0:
		raise ValueError("fraction must be between 0 and 1.")

	if fraction == 1.0:
		return DatasetSplit(
			features=dataset_split.features.copy(),
			targets=dataset_split.targets.copy(),
			constraint_reference=dataset_split.constraint_reference.copy(),
		)

	all_indices = dataset_split.features.index.to_numpy()
	sampled_indices, _ = train_test_split(
		all_indices,
		train_size=fraction,
		random_state=random_seed,
		shuffle=True,
	)

	return _select_dataset_split(dataset_split, pd.Index(sampled_indices))


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
	"build_projection_operator",
	"build_measured_supervised_dataset",
	"combine_dataset_splits",
	"compute_measured_composites",
	"fit_scalers",
	"inverse_transform_targets",
	"make_train_validation_test_splits",
	"make_train_test_split",
	"project_to_mass_balance",
	"sample_dataset_fraction",
	"transform_dataset_split",
	"transform_dataset_splits",
]