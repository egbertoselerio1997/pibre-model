"""Reusable regression and constraint-metric helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_mean_absolute_percentage_error(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	*,
	epsilon: float = 1e-8,
) -> float:
	"""Compute MAPE while guarding against division by zero."""

	y_true_array = np.asarray(y_true, dtype=float)
	y_pred_array = np.asarray(y_pred, dtype=float)
	denominator = np.maximum(np.abs(y_true_array), epsilon)

	return float(np.mean(np.abs((y_true_array - y_pred_array) / denominator)))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
	"""Compute the repository-standard aggregate regression metrics."""

	y_true_array = np.asarray(y_true, dtype=float)
	y_pred_array = np.asarray(y_pred, dtype=float)
	mse = float(mean_squared_error(y_true_array, y_pred_array))

	return {
		"R2": float(r2_score(y_true_array, y_pred_array, multioutput="uniform_average")),
		"MSE": mse,
		"RMSE": float(np.sqrt(mse)),
		"MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
		"MAPE": safe_mean_absolute_percentage_error(y_true_array, y_pred_array),
	}


def compute_per_target_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	target_columns: Iterable[str],
) -> pd.DataFrame:
	"""Compute repository-standard metrics independently for each target column."""

	y_true_array = np.asarray(y_true, dtype=float)
	y_pred_array = np.asarray(y_pred, dtype=float)
	metric_rows: list[dict[str, float | str]] = []

	for target_index, target_name in enumerate(target_columns):
		metric_row = {"target": str(target_name)}
		metric_row.update(
			compute_regression_metrics(
				y_true_array[:, target_index],
				y_pred_array[:, target_index],
			)
		)
		metric_rows.append(metric_row)

	return pd.DataFrame(metric_rows)


def compute_mass_balance_residuals(
	predictions: np.ndarray,
	constraint_reference: np.ndarray,
	A_matrix: np.ndarray,
) -> np.ndarray:
	"""Compute the residuals of the measured-space invariance constraints."""

	prediction_array = np.asarray(predictions, dtype=float)
	reference_array = np.asarray(constraint_reference, dtype=float)
	constraint_matrix = np.asarray(A_matrix, dtype=float)

	return (constraint_matrix @ prediction_array.T - constraint_matrix @ reference_array.T).T


def summarize_mass_balance_residuals(
	predictions: np.ndarray,
	constraint_reference: np.ndarray,
	A_matrix: np.ndarray,
) -> dict[str, float]:
	"""Summarize absolute and L2 mass-balance residuals across all samples."""

	residuals = compute_mass_balance_residuals(predictions, constraint_reference, A_matrix)
	l2_values = np.linalg.norm(residuals, axis=1)

	return {
		"constraint_mean_l2": float(np.mean(l2_values)),
		"constraint_max_l2": float(np.max(l2_values)),
		"constraint_mean_abs": float(np.mean(np.abs(residuals))),
		"constraint_max_abs": float(np.max(np.abs(residuals))),
	}


__all__ = [
	"compute_mass_balance_residuals",
	"compute_per_target_metrics",
	"compute_regression_metrics",
	"safe_mean_absolute_percentage_error",
	"summarize_mass_balance_residuals",
]