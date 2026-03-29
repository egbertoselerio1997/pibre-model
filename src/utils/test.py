"""Reusable evaluation helpers for trained machine-learning models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import (
    compute_mass_balance_residuals,
    compute_per_target_metrics,
    compute_regression_metrics,
    summarize_mass_balance_residuals,
)


def build_prediction_frame(
    values: np.ndarray,
    target_columns: Iterable[str],
    *,
    index: pd.Index | None = None,
    prefix: str,
) -> pd.DataFrame:
    """Convert prediction arrays into a labeled dataframe."""

    columns = [f"{prefix}{column_name}" for column_name in target_columns]
    return pd.DataFrame(np.asarray(values, dtype=float), index=index, columns=columns)


def evaluate_prediction_bundle(
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    projected_predictions: np.ndarray,
    constraint_reference: np.ndarray,
    A_matrix: np.ndarray,
    target_columns: Iterable[str],
    *,
    index: pd.Index | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble aggregate, per-target, and residual reports for raw and projected predictions."""

    target_column_list = list(target_columns)
    raw_metrics = compute_regression_metrics(y_true, raw_predictions)
    raw_metrics.update(summarize_mass_balance_residuals(raw_predictions, constraint_reference, A_matrix))
    projected_metrics = compute_regression_metrics(y_true, projected_predictions)
    projected_metrics.update(summarize_mass_balance_residuals(projected_predictions, constraint_reference, A_matrix))

    aggregate_report = pd.DataFrame(
        [
            {"prediction_type": "raw", **raw_metrics},
            {"prediction_type": "projected", **projected_metrics},
        ]
    )

    raw_per_target = compute_per_target_metrics(y_true, raw_predictions, target_column_list).rename(
        columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    projected_per_target = compute_per_target_metrics(y_true, projected_predictions, target_column_list).rename(
        columns={metric_name: f"projected_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    per_target_report = raw_per_target.merge(projected_per_target, on="target", how="inner")

    raw_residuals = compute_mass_balance_residuals(raw_predictions, constraint_reference, A_matrix)
    projected_residuals = compute_mass_balance_residuals(projected_predictions, constraint_reference, A_matrix)
    residual_report = pd.DataFrame(
        {
            "raw_constraint_l2": np.linalg.norm(raw_residuals, axis=1),
            "projected_constraint_l2": np.linalg.norm(projected_residuals, axis=1),
        },
        index=index,
    )

    return {
        "aggregate_metrics": aggregate_report,
        "per_target_metrics": per_target_report,
        "raw_predictions": build_prediction_frame(raw_predictions, target_column_list, index=index, prefix="Raw_"),
        "projected_predictions": build_prediction_frame(
            projected_predictions,
            target_column_list,
            index=index,
            prefix="Projected_",
        ),
        "constraint_residuals": residual_report,
    }


def evaluate_cobre_prediction_bundle(
    y_true_measured: np.ndarray,
    raw_fractional_predictions: np.ndarray,
    projected_fractional_predictions: np.ndarray,
    constraint_reference: np.ndarray,
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray,
    target_columns: Iterable[str],
    state_columns: Iterable[str],
    *,
    index: pd.Index | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble COBRE reports with measured-space metrics and fractional-space constraints."""

    target_column_list = list(target_columns)
    state_column_list = list(state_columns)
    composition_array = np.asarray(composition_matrix, dtype=float)
    raw_fractional_array = np.asarray(raw_fractional_predictions, dtype=float)
    projected_fractional_array = np.asarray(projected_fractional_predictions, dtype=float)
    raw_measured_predictions = raw_fractional_array @ composition_array.T
    projected_measured_predictions = projected_fractional_array @ composition_array.T

    raw_metrics = compute_regression_metrics(y_true_measured, raw_measured_predictions)
    raw_metrics.update(summarize_mass_balance_residuals(raw_fractional_array, constraint_reference, A_matrix))
    projected_metrics = compute_regression_metrics(y_true_measured, projected_measured_predictions)
    projected_metrics.update(
        summarize_mass_balance_residuals(projected_fractional_array, constraint_reference, A_matrix)
    )

    aggregate_report = pd.DataFrame(
        [
            {"prediction_type": "raw", **raw_metrics},
            {"prediction_type": "projected", **projected_metrics},
        ]
    )

    raw_per_target = compute_per_target_metrics(
        y_true_measured,
        raw_measured_predictions,
        target_column_list,
    ).rename(columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]})
    projected_per_target = compute_per_target_metrics(
        y_true_measured,
        projected_measured_predictions,
        target_column_list,
    ).rename(
        columns={metric_name: f"projected_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    per_target_report = raw_per_target.merge(projected_per_target, on="target", how="inner")

    raw_residuals = compute_mass_balance_residuals(raw_fractional_array, constraint_reference, A_matrix)
    projected_residuals = compute_mass_balance_residuals(
        projected_fractional_array,
        constraint_reference,
        A_matrix,
    )
    residual_report = pd.DataFrame(
        {
            "raw_constraint_l2": np.linalg.norm(raw_residuals, axis=1),
            "projected_constraint_l2": np.linalg.norm(projected_residuals, axis=1),
        },
        index=index,
    )

    return {
        "aggregate_metrics": aggregate_report,
        "per_target_metrics": per_target_report,
        "raw_predictions": build_prediction_frame(
            raw_measured_predictions,
            target_column_list,
            index=index,
            prefix="Raw_",
        ),
        "projected_predictions": build_prediction_frame(
            projected_measured_predictions,
            target_column_list,
            index=index,
            prefix="Projected_",
        ),
        "raw_fractional_predictions": build_prediction_frame(
            raw_fractional_array,
            state_column_list,
            index=index,
            prefix="RawFractional_",
        ),
        "projected_fractional_predictions": build_prediction_frame(
            projected_fractional_array,
            state_column_list,
            index=index,
            prefix="ProjectedFractional_",
        ),
        "constraint_residuals": residual_report,
    }


__all__ = ["build_prediction_frame", "evaluate_cobre_prediction_bundle", "evaluate_prediction_bundle"]