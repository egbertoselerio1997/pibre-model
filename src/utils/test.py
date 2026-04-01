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
from .process import has_active_projection


def _build_report_metadata_frame(
    *,
    native_prediction_space: str,
    comparison_target_space: str,
    constraint_space: str,
    direct_comparison_scope: str,
    diagnostic_scope: str,
    projection_active: bool,
    constraint_status: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "native_prediction_space": native_prediction_space,
                "comparison_target_space": comparison_target_space,
                "constraint_space": constraint_space,
                "direct_comparison_scope": direct_comparison_scope,
                "diagnostic_scope": diagnostic_scope,
                "projection_active": bool(projection_active),
                "constraint_status": constraint_status,
            }
        ]
    )


def _compute_projection_adjustment_frame(
    raw_values: np.ndarray,
    projected_values: np.ndarray,
    *,
    index: pd.Index | None,
    prefix: str,
) -> pd.DataFrame:
    adjustment = np.asarray(projected_values, dtype=float) - np.asarray(raw_values, dtype=float)
    return pd.DataFrame(
        {
            f"{prefix}_adjustment_l2": np.linalg.norm(adjustment, axis=1),
            f"{prefix}_adjustment_mean_abs": np.mean(np.abs(adjustment), axis=1),
            f"{prefix}_adjustment_max_abs": np.max(np.abs(adjustment), axis=1),
        },
        index=index,
    )


def _summarize_projection_adjustments(
    raw_values: np.ndarray,
    projected_values: np.ndarray,
    *,
    diagnostic_name: str,
) -> pd.DataFrame:
    adjustment = np.asarray(projected_values, dtype=float) - np.asarray(raw_values, dtype=float)
    adjustment_l2 = np.linalg.norm(adjustment, axis=1)

    return pd.DataFrame(
        [
            {
                "diagnostic_name": diagnostic_name,
                "prediction_type": "raw_to_projected",
                "mean_l2": float(np.mean(adjustment_l2)),
                "max_l2": float(np.max(adjustment_l2)),
                "mean_abs": float(np.mean(np.abs(adjustment))),
                "max_abs": float(np.max(np.abs(adjustment))),
            }
        ]
    )


def _build_constraint_diagnostic_summary(
    raw_predictions: np.ndarray,
    projected_predictions: np.ndarray,
    constraint_reference: np.ndarray,
    A_matrix: np.ndarray,
    *,
    diagnostic_name: str,
) -> pd.DataFrame:
    raw_summary = summarize_mass_balance_residuals(raw_predictions, constraint_reference, A_matrix)
    projected_summary = summarize_mass_balance_residuals(projected_predictions, constraint_reference, A_matrix)

    return pd.DataFrame(
        [
            {"diagnostic_name": diagnostic_name, "prediction_type": "raw", **raw_summary},
            {"diagnostic_name": diagnostic_name, "prediction_type": "projected", **projected_summary},
        ]
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
    projection_active = has_active_projection(A_matrix)
    raw_metrics = compute_regression_metrics(y_true, raw_predictions)
    aggregate_rows = [{"prediction_type": "raw", **raw_metrics}]
    if projection_active:
        projected_metrics = compute_regression_metrics(y_true, projected_predictions)
        aggregate_rows.append({"prediction_type": "projected", **projected_metrics})
    aggregate_report = pd.DataFrame(aggregate_rows)

    raw_per_target = compute_per_target_metrics(y_true, raw_predictions, target_column_list).rename(
        columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    per_target_report = raw_per_target
    report: dict[str, pd.DataFrame] = {
        "report_metadata": _build_report_metadata_frame(
            native_prediction_space="measured",
            comparison_target_space="measured",
            constraint_space="measured",
            direct_comparison_scope="measured_output_metrics_only",
            diagnostic_scope=(
                "model_native_measured_space_diagnostics"
                if projection_active
                else "projection_inactive_trivial_measured_null_space"
            ),
            projection_active=projection_active,
            constraint_status=("active" if projection_active else "inactive_trivial_null_space"),
        ),
        "aggregate_metrics": aggregate_report,
        "per_target_metrics": per_target_report,
        "raw_predictions": build_prediction_frame(raw_predictions, target_column_list, index=index, prefix="Raw_"),
    }

    if not projection_active:
        return report

    projected_per_target = compute_per_target_metrics(y_true, projected_predictions, target_column_list).rename(
        columns={metric_name: f"projected_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    report["per_target_metrics"] = raw_per_target.merge(projected_per_target, on="target", how="inner")

    raw_residuals = compute_mass_balance_residuals(raw_predictions, constraint_reference, A_matrix)
    projected_residuals = compute_mass_balance_residuals(projected_predictions, constraint_reference, A_matrix)
    residual_report = pd.DataFrame(
        {
            "raw_constraint_l2": np.linalg.norm(raw_residuals, axis=1),
            "projected_constraint_l2": np.linalg.norm(projected_residuals, axis=1),
        },
        index=index,
    )
    projection_adjustments = _compute_projection_adjustment_frame(
        raw_predictions,
        projected_predictions,
        index=index,
        prefix="measured",
    )
    diagnostic_summary = pd.concat(
        [
            _build_constraint_diagnostic_summary(
                raw_predictions,
                projected_predictions,
                constraint_reference,
                A_matrix,
                diagnostic_name="measured_constraint_residual",
            ),
            _summarize_projection_adjustments(
                raw_predictions,
                projected_predictions,
                diagnostic_name="measured_projection_adjustment",
            ),
        ],
        ignore_index=True,
    )

    report["projected_predictions"] = build_prediction_frame(
        projected_predictions,
        target_column_list,
        index=index,
        prefix="Projected_",
    )
    report["diagnostic_summary"] = diagnostic_summary
    report["projection_diagnostics"] = projection_adjustments
    report["constraint_residuals"] = residual_report
    return report


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
    prediction_uncertainty: dict[str, Any] | None = None,
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
    projected_metrics = compute_regression_metrics(y_true_measured, projected_measured_predictions)

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
    measured_projection_adjustments = _compute_projection_adjustment_frame(
        raw_measured_predictions,
        projected_measured_predictions,
        index=index,
        prefix="measured",
    )
    fractional_projection_adjustments = _compute_projection_adjustment_frame(
        raw_fractional_array,
        projected_fractional_array,
        index=index,
        prefix="fractional",
    )
    projection_adjustments = pd.concat(
        [measured_projection_adjustments, fractional_projection_adjustments],
        axis=1,
    )
    diagnostic_summary = pd.concat(
        [
            _build_constraint_diagnostic_summary(
                raw_fractional_array,
                projected_fractional_array,
                constraint_reference,
                A_matrix,
                diagnostic_name="fractional_constraint_residual",
            ),
            _summarize_projection_adjustments(
                raw_measured_predictions,
                projected_measured_predictions,
                diagnostic_name="measured_projection_adjustment",
            ),
            _summarize_projection_adjustments(
                raw_fractional_array,
                projected_fractional_array,
                diagnostic_name="fractional_projection_adjustment",
            ),
        ],
        ignore_index=True,
    )

    report = {
        "report_metadata": _build_report_metadata_frame(
            native_prediction_space="fractional",
            comparison_target_space="measured",
            constraint_space="fractional",
            direct_comparison_scope="measured_output_metrics_only",
            diagnostic_scope="model_native_fractional_space_diagnostics",
            projection_active=True,
            constraint_status="active",
        ),
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
        "diagnostic_summary": diagnostic_summary,
        "projection_diagnostics": projection_adjustments,
        "constraint_residuals": residual_report,
    }

    if prediction_uncertainty is not None:
        report["uncertainty_metadata"] = pd.DataFrame([dict(prediction_uncertainty["metadata"])])
        report["projected_prediction_standard_errors"] = build_prediction_frame(
            prediction_uncertainty["projected_prediction_standard_errors"],
            target_column_list,
            index=index,
            prefix="ProjectedSE_",
        )
        report["projected_prediction_confidence_interval_lower"] = build_prediction_frame(
            prediction_uncertainty["projected_prediction_confidence_interval_lower"],
            target_column_list,
            index=index,
            prefix="ProjectedCI95Lower_",
        )
        report["projected_prediction_confidence_interval_upper"] = build_prediction_frame(
            prediction_uncertainty["projected_prediction_confidence_interval_upper"],
            target_column_list,
            index=index,
            prefix="ProjectedCI95Upper_",
        )
        report["projected_prediction_interval_lower"] = build_prediction_frame(
            prediction_uncertainty["projected_prediction_interval_lower"],
            target_column_list,
            index=index,
            prefix="ProjectedPI95Lower_",
        )
        report["projected_prediction_interval_upper"] = build_prediction_frame(
            prediction_uncertainty["projected_prediction_interval_upper"],
            target_column_list,
            index=index,
            prefix="ProjectedPI95Upper_",
        )
        report["projected_prediction_interval_standard_errors"] = build_prediction_frame(
            prediction_uncertainty["projected_prediction_interval_standard_errors"],
            target_column_list,
            index=index,
            prefix="ProjectedPISE_",
        )
        report["prediction_uncertainty_summary"] = prediction_uncertainty["prediction_uncertainty_summary"].copy()

    return report


__all__ = ["build_prediction_frame", "evaluate_cobre_prediction_bundle", "evaluate_prediction_bundle"]