"""Reusable evaluation helpers for trained machine-learning models."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .metrics import (
    compute_mass_balance_residuals,
    compute_per_target_metrics,
    compute_regression_metrics,
    summarize_mass_balance_residuals,
)
from .process import collapse_fractional_states_to_measured_outputs, has_active_projection


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


def _build_constraint_diagnostic_summary_for_prediction_types(
    predictions_by_type: Mapping[str, np.ndarray],
    constraint_reference: np.ndarray,
    A_matrix: np.ndarray,
    *,
    diagnostic_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "diagnostic_name": diagnostic_name,
                "prediction_type": str(prediction_type),
                **summarize_mass_balance_residuals(prediction_values, constraint_reference, A_matrix),
            }
            for prediction_type, prediction_values in predictions_by_type.items()
        ]
    )


def build_icsor_projection_stage_frame(
    projection_details: Mapping[str, Any],
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Convert staged icsor projection details into a row-aligned dataframe."""

    return pd.DataFrame(
        {
            "projection_stage": np.asarray(projection_details["projection_stage"], dtype=object),
            "raw_feasible": np.asarray(projection_details["raw_feasible_mask"], dtype=bool),
            "affine_feasible": np.asarray(projection_details["affine_feasible_mask"], dtype=bool),
            "lp_active": np.asarray(projection_details["lp_active_mask"], dtype=bool),
            "solver_status": np.asarray(projection_details["solver_status"], dtype=object),
            "solver_iterations": np.asarray(projection_details["solver_iterations"], dtype=int),
            "raw_constraint_max_abs": np.asarray(projection_details["raw_constraint_max_abs"], dtype=float),
            "affine_constraint_max_abs": np.asarray(projection_details["affine_constraint_max_abs"], dtype=float),
            "projected_constraint_max_abs": np.asarray(
                projection_details["projected_constraint_max_abs"],
                dtype=float,
            ),
            "raw_min_component": np.asarray(projection_details["raw_min_component"], dtype=float),
            "affine_min_component": np.asarray(projection_details["affine_min_component"], dtype=float),
            "projected_min_component": np.asarray(projection_details["projected_min_component"], dtype=float),
        },
        index=index,
    )


def build_icsor_projection_stage_summary(
    projection_details: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarize how often each staged icsor correction path is used."""

    stage_frame = build_icsor_projection_stage_frame(projection_details)
    total_samples = int(len(stage_frame))
    summary = (
        stage_frame.groupby("projection_stage", dropna=False)
        .agg(
            sample_count=("projection_stage", "size"),
            lp_active_count=("lp_active", "sum"),
            mean_solver_iterations=("solver_iterations", "mean"),
            max_solver_iterations=("solver_iterations", "max"),
            minimum_projected_component=("projected_min_component", "min"),
        )
        .reset_index()
    )
    summary["sample_share_pct"] = np.where(
        total_samples > 0,
        100.0 * summary["sample_count"] / float(total_samples),
        0.0,
    )
    stage_order = {"raw_feasible": 0, "affine_feasible": 1, "lp_corrected": 2}
    return summary.sort_values(
        by="projection_stage",
        key=lambda series: series.map(stage_order).fillna(len(stage_order)),
    ).reset_index(drop=True)


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


def _resolve_external_measured_output_columns(
    target_columns: Iterable[str],
    composition_matrix: np.ndarray,
    *,
    measured_output_columns: Iterable[str] | None = None,
) -> list[str]:
    """Resolve measured-output labels for external collapse and reporting.

    When no explicit labels are provided, this falls back to target-derived labels
    only when their width matches the composition matrix row count.
    """

    composition_array = np.asarray(composition_matrix, dtype=float)
    if composition_array.ndim != 2:
        raise ValueError("composition_matrix must be two-dimensional for external measured-output evaluation.")

    measured_count = int(composition_array.shape[0])
    if measured_output_columns is not None:
        resolved_columns = [str(column_name) for column_name in measured_output_columns]
        if len(resolved_columns) != measured_count:
            raise ValueError(
                "measured_output_columns length must match composition_matrix row count for external evaluation."
            )
        return resolved_columns

    target_derived_columns = [
        str(column_name).replace("Out_", "", 1) if str(column_name).startswith("Out_") else str(column_name)
        for column_name in target_columns
    ]
    if len(target_derived_columns) == measured_count:
        return target_derived_columns

    return [f"Measured_{column_index + 1}" for column_index in range(measured_count)]


def evaluate_prediction_bundle(
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    projected_predictions: np.ndarray,
    constraint_reference: np.ndarray,
    A_matrix: np.ndarray,
    target_columns: Iterable[str],
    *,
    index: pd.Index | None = None,
    composition_matrix: np.ndarray | None = None,
    state_columns: Iterable[str] | None = None,
    measured_output_columns: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble native prediction diagnostics and optional external measured comparisons."""

    target_column_list = list(target_columns)
    projection_active = has_active_projection(A_matrix)
    y_true_array = np.asarray(y_true, dtype=float)
    raw_prediction_array = np.asarray(raw_predictions, dtype=float)
    projected_prediction_array = np.asarray(projected_predictions, dtype=float)

    comparison_y_true = y_true_array
    comparison_raw_predictions = raw_prediction_array
    comparison_projected_predictions = projected_prediction_array
    comparison_target_columns = target_column_list
    report_metadata = _build_report_metadata_frame(
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
    )
    report: dict[str, pd.DataFrame]
    if composition_matrix is not None and state_columns is not None:
        state_column_list = list(state_columns)
        resolved_measured_output_columns = _resolve_external_measured_output_columns(
            target_column_list,
            np.asarray(composition_matrix, dtype=float),
            measured_output_columns=measured_output_columns,
        )
        comparison_target_columns = [f"Out_{column_name}" for column_name in resolved_measured_output_columns]
        comparison_y_true = collapse_fractional_states_to_measured_outputs(
            y_true_array,
            state_column_list,
            np.asarray(composition_matrix, dtype=float),
            resolved_measured_output_columns,
            output_prefix="Out_",
            index=index,
        ).to_numpy(dtype=float)
        comparison_raw_predictions = collapse_fractional_states_to_measured_outputs(
            raw_prediction_array,
            state_column_list,
            np.asarray(composition_matrix, dtype=float),
            resolved_measured_output_columns,
            output_prefix="Out_",
            index=index,
        ).to_numpy(dtype=float)
        comparison_projected_predictions = collapse_fractional_states_to_measured_outputs(
            projected_prediction_array,
            state_column_list,
            np.asarray(composition_matrix, dtype=float),
            resolved_measured_output_columns,
            output_prefix="Out_",
            index=index,
        ).to_numpy(dtype=float)
        report_metadata = _build_report_metadata_frame(
            native_prediction_space="fractional",
            comparison_target_space="external_measured_output",
            constraint_space="fractional",
            direct_comparison_scope="externally_collapsed_measured_output_metrics",
            diagnostic_scope=(
                "model_native_fractional_space_diagnostics"
                if projection_active
                else "projection_inactive_trivial_fractional_null_space"
            ),
            projection_active=projection_active,
            constraint_status=("active" if projection_active else "inactive_trivial_null_space"),
        )

    raw_metrics = compute_regression_metrics(comparison_y_true, comparison_raw_predictions)
    aggregate_rows = [{"prediction_type": "raw", **raw_metrics}]
    if projection_active:
        projected_metrics = compute_regression_metrics(comparison_y_true, comparison_projected_predictions)
        aggregate_rows.append({"prediction_type": "projected", **projected_metrics})
    aggregate_report = pd.DataFrame(aggregate_rows)

    raw_per_target = compute_per_target_metrics(
        comparison_y_true,
        comparison_raw_predictions,
        comparison_target_columns,
    ).rename(
        columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    per_target_report = raw_per_target
    report = {
        "report_metadata": report_metadata,
        "aggregate_metrics": aggregate_report,
        "per_target_metrics": per_target_report,
        "raw_predictions": build_prediction_frame(
            comparison_raw_predictions,
            comparison_target_columns,
            index=index,
            prefix="Raw_",
        ),
    }

    if composition_matrix is not None and state_columns is not None:
        report["fractional_aggregate_metrics"] = pd.DataFrame(
            [{"prediction_type": "raw", **compute_regression_metrics(y_true_array, raw_prediction_array)}]
        )
        report["fractional_per_target_metrics"] = compute_per_target_metrics(
            y_true_array,
            raw_prediction_array,
            target_column_list,
        ).rename(columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]})
        report["raw_fractional_predictions"] = build_prediction_frame(
            raw_prediction_array,
            list(state_columns),
            index=index,
            prefix="RawFractional_",
        )

    if not projection_active:
        return report

    projected_per_target = compute_per_target_metrics(
        comparison_y_true,
        comparison_projected_predictions,
        comparison_target_columns,
    ).rename(
        columns={metric_name: f"projected_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    report["per_target_metrics"] = raw_per_target.merge(projected_per_target, on="target", how="inner")

    raw_residuals = compute_mass_balance_residuals(raw_prediction_array, constraint_reference, A_matrix)
    projected_residuals = compute_mass_balance_residuals(projected_prediction_array, constraint_reference, A_matrix)
    residual_report = pd.DataFrame(
        {
            "raw_constraint_l2": np.linalg.norm(raw_residuals, axis=1),
            "projected_constraint_l2": np.linalg.norm(projected_residuals, axis=1),
        },
        index=index,
    )
    projection_adjustments = _compute_projection_adjustment_frame(
        comparison_raw_predictions,
        comparison_projected_predictions,
        index=index,
        prefix=("measured" if composition_matrix is not None and state_columns is not None else "native"),
    )
    diagnostic_summary = pd.concat(
        [
            _build_constraint_diagnostic_summary(
                raw_prediction_array,
                projected_prediction_array,
                constraint_reference,
                A_matrix,
                diagnostic_name=(
                    "fractional_constraint_residual"
                    if composition_matrix is not None and state_columns is not None
                    else "measured_constraint_residual"
                ),
            ),
            _summarize_projection_adjustments(
                comparison_raw_predictions,
                comparison_projected_predictions,
                diagnostic_name=(
                    "measured_projection_adjustment"
                    if composition_matrix is not None and state_columns is not None
                    else "native_projection_adjustment"
                ),
            ),
        ],
        ignore_index=True,
    )

    report["projected_predictions"] = build_prediction_frame(
        comparison_projected_predictions,
        comparison_target_columns,
        index=index,
        prefix="Projected_",
    )
    if composition_matrix is not None and state_columns is not None:
        report["fractional_aggregate_metrics"] = pd.DataFrame(
            [
                {"prediction_type": "raw", **compute_regression_metrics(y_true_array, raw_prediction_array)},
                {"prediction_type": "projected", **compute_regression_metrics(y_true_array, projected_prediction_array)},
            ]
        )
        report["fractional_per_target_metrics"] = compute_per_target_metrics(
            y_true_array,
            raw_prediction_array,
            target_column_list,
        ).rename(columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}).merge(
            compute_per_target_metrics(
                y_true_array,
                projected_prediction_array,
                target_column_list,
            ).rename(columns={metric_name: f"projected_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}),
            on="target",
            how="inner",
        )
        report["projected_fractional_predictions"] = build_prediction_frame(
            projected_prediction_array,
            list(state_columns),
            index=index,
            prefix="ProjectedFractional_",
        )
    report["diagnostic_summary"] = diagnostic_summary
    report["projection_diagnostics"] = projection_adjustments
    report["constraint_residuals"] = residual_report
    return report


def evaluate_icsor_prediction_bundle(
    y_true_fractional: np.ndarray,
    raw_fractional_predictions: np.ndarray,
    affine_fractional_predictions: np.ndarray,
    projected_fractional_predictions: np.ndarray,
    constraint_reference: np.ndarray,
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray,
    target_columns: Iterable[str],
    state_columns: Iterable[str],
    measured_output_columns: Iterable[str] | None = None,
    *,
    index: pd.Index | None = None,
    prediction_uncertainty: dict[str, Any] | None = None,
    projection_details: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble ICSOR reports with native fractional diagnostics and external measured comparisons."""

    target_column_list = list(target_columns)
    state_column_list = list(state_columns)
    resolved_measured_output_columns = _resolve_external_measured_output_columns(
        target_column_list,
        np.asarray(composition_matrix, dtype=float),
        measured_output_columns=measured_output_columns,
    )
    measured_target_columns = [f"Out_{column_name}" for column_name in resolved_measured_output_columns]
    y_true_fractional_array = np.asarray(y_true_fractional, dtype=float)
    raw_fractional_array = np.asarray(raw_fractional_predictions, dtype=float)
    affine_fractional_array = np.asarray(affine_fractional_predictions, dtype=float)
    projected_fractional_array = np.asarray(projected_fractional_predictions, dtype=float)
    y_true_measured = collapse_fractional_states_to_measured_outputs(
        y_true_fractional_array,
        state_column_list,
        np.asarray(composition_matrix, dtype=float),
        resolved_measured_output_columns,
        output_prefix="Out_",
        index=index,
    ).to_numpy(dtype=float)
    raw_measured_predictions = collapse_fractional_states_to_measured_outputs(
        raw_fractional_array,
        state_column_list,
        np.asarray(composition_matrix, dtype=float),
        resolved_measured_output_columns,
        output_prefix="Out_",
        index=index,
    ).to_numpy(dtype=float)
    affine_measured_predictions = collapse_fractional_states_to_measured_outputs(
        affine_fractional_array,
        state_column_list,
        np.asarray(composition_matrix, dtype=float),
        resolved_measured_output_columns,
        output_prefix="Out_",
        index=index,
    ).to_numpy(dtype=float)
    projected_measured_predictions = collapse_fractional_states_to_measured_outputs(
        projected_fractional_array,
        state_column_list,
        np.asarray(composition_matrix, dtype=float),
        resolved_measured_output_columns,
        output_prefix="Out_",
        index=index,
    ).to_numpy(dtype=float)

    raw_metrics = compute_regression_metrics(y_true_measured, raw_measured_predictions)
    affine_metrics = compute_regression_metrics(y_true_measured, affine_measured_predictions)
    projected_metrics = compute_regression_metrics(y_true_measured, projected_measured_predictions)

    aggregate_report = pd.DataFrame(
        [
            {"prediction_type": "raw", **raw_metrics},
            {"prediction_type": "affine", **affine_metrics},
            {"prediction_type": "projected", **projected_metrics},
        ]
    )

    raw_per_target = compute_per_target_metrics(
        y_true_measured,
        raw_measured_predictions,
        measured_target_columns,
    ).rename(columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]})
    affine_per_target = compute_per_target_metrics(
        y_true_measured,
        affine_measured_predictions,
        measured_target_columns,
    ).rename(columns={metric_name: f"affine_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]})
    projected_per_target = compute_per_target_metrics(
        y_true_measured,
        projected_measured_predictions,
        measured_target_columns,
    ).rename(
        columns={metric_name: f"projected_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    per_target_report = raw_per_target.merge(affine_per_target, on="target", how="inner").merge(
        projected_per_target,
        on="target",
        how="inner",
    )

    fractional_aggregate_report = pd.DataFrame(
        [
            {"prediction_type": "raw", **compute_regression_metrics(y_true_fractional_array, raw_fractional_array)},
            {"prediction_type": "affine", **compute_regression_metrics(y_true_fractional_array, affine_fractional_array)},
            {
                "prediction_type": "projected",
                **compute_regression_metrics(y_true_fractional_array, projected_fractional_array),
            },
        ]
    )
    fractional_raw_per_target = compute_per_target_metrics(
        y_true_fractional_array,
        raw_fractional_array,
        target_column_list,
    ).rename(columns={metric_name: f"raw_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]})
    fractional_affine_per_target = compute_per_target_metrics(
        y_true_fractional_array,
        affine_fractional_array,
        target_column_list,
    ).rename(columns={metric_name: f"affine_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]})
    fractional_projected_per_target = compute_per_target_metrics(
        y_true_fractional_array,
        projected_fractional_array,
        target_column_list,
    ).rename(
        columns={metric_name: f"projected_{metric_name}" for metric_name in ["R2", "MSE", "RMSE", "MAE", "MAPE"]}
    )
    fractional_per_target_report = fractional_raw_per_target.merge(
        fractional_affine_per_target,
        on="target",
        how="inner",
    ).merge(
        fractional_projected_per_target,
        on="target",
        how="inner",
    )

    raw_residuals = compute_mass_balance_residuals(raw_fractional_array, constraint_reference, A_matrix)
    affine_residuals = compute_mass_balance_residuals(affine_fractional_array, constraint_reference, A_matrix)
    projected_residuals = compute_mass_balance_residuals(
        projected_fractional_array,
        constraint_reference,
        A_matrix,
    )
    residual_report = pd.DataFrame(
        {
            "raw_constraint_l2": np.linalg.norm(raw_residuals, axis=1),
            "affine_constraint_l2": np.linalg.norm(affine_residuals, axis=1),
            "projected_constraint_l2": np.linalg.norm(projected_residuals, axis=1),
        },
        index=index,
    )
    measured_raw_to_affine_adjustments = _compute_projection_adjustment_frame(
        raw_measured_predictions,
        affine_measured_predictions,
        index=index,
        prefix="measured_raw_to_affine",
    )
    measured_affine_to_projected_adjustments = _compute_projection_adjustment_frame(
        affine_measured_predictions,
        projected_measured_predictions,
        index=index,
        prefix="measured_affine_to_projected",
    )
    measured_projection_adjustments = _compute_projection_adjustment_frame(
        raw_measured_predictions,
        projected_measured_predictions,
        index=index,
        prefix="measured_raw_to_projected",
    )
    fractional_raw_to_affine_adjustments = _compute_projection_adjustment_frame(
        raw_fractional_array,
        affine_fractional_array,
        index=index,
        prefix="fractional_raw_to_affine",
    )
    fractional_affine_to_projected_adjustments = _compute_projection_adjustment_frame(
        affine_fractional_array,
        projected_fractional_array,
        index=index,
        prefix="fractional_affine_to_projected",
    )
    fractional_projection_adjustments = _compute_projection_adjustment_frame(
        raw_fractional_array,
        projected_fractional_array,
        index=index,
        prefix="fractional_raw_to_projected",
    )
    projection_adjustments = pd.concat(
        [
            measured_raw_to_affine_adjustments,
            measured_affine_to_projected_adjustments,
            measured_projection_adjustments,
            fractional_raw_to_affine_adjustments,
            fractional_affine_to_projected_adjustments,
            fractional_projection_adjustments,
        ],
        axis=1,
    )
    projection_stage_summary = None
    if projection_details is not None:
        projection_stage_frame = build_icsor_projection_stage_frame(projection_details, index=index)
        projection_adjustments = pd.concat([projection_adjustments, projection_stage_frame], axis=1)
        projection_stage_summary = build_icsor_projection_stage_summary(projection_details)

    diagnostic_summary = pd.concat(
        [
            _build_constraint_diagnostic_summary_for_prediction_types(
                {
                    "raw": raw_fractional_array,
                    "affine": affine_fractional_array,
                    "projected": projected_fractional_array,
                },
                constraint_reference,
                A_matrix,
                diagnostic_name="fractional_constraint_residual",
            ),
            _summarize_projection_adjustments(
                raw_measured_predictions,
                affine_measured_predictions,
                diagnostic_name="measured_raw_to_affine_adjustment",
            ),
            _summarize_projection_adjustments(
                affine_measured_predictions,
                projected_measured_predictions,
                diagnostic_name="measured_affine_to_projected_adjustment",
            ),
            _summarize_projection_adjustments(
                raw_measured_predictions,
                projected_measured_predictions,
                diagnostic_name="measured_raw_to_projected_adjustment",
            ),
            _summarize_projection_adjustments(
                raw_fractional_array,
                affine_fractional_array,
                diagnostic_name="fractional_raw_to_affine_adjustment",
            ),
            _summarize_projection_adjustments(
                affine_fractional_array,
                projected_fractional_array,
                diagnostic_name="fractional_affine_to_projected_adjustment",
            ),
            _summarize_projection_adjustments(
                raw_fractional_array,
                projected_fractional_array,
                diagnostic_name="fractional_raw_to_projected_adjustment",
            ),
        ],
        ignore_index=True,
    )

    report = {
        "report_metadata": _build_report_metadata_frame(
            native_prediction_space="fractional",
            comparison_target_space="external_measured_output",
            constraint_space="fractional",
            direct_comparison_scope="externally_collapsed_measured_output_metrics",
            diagnostic_scope="model_native_fractional_space_nonnegative_lp_diagnostics",
            projection_active=True,
            constraint_status="active_nonnegative_lp",
        ),
        "aggregate_metrics": aggregate_report,
        "per_target_metrics": per_target_report,
        "fractional_aggregate_metrics": fractional_aggregate_report,
        "fractional_per_target_metrics": fractional_per_target_report,
        "raw_predictions": build_prediction_frame(
            raw_measured_predictions,
            measured_target_columns,
            index=index,
            prefix="Raw_",
        ),
        "affine_predictions": build_prediction_frame(
            affine_measured_predictions,
            measured_target_columns,
            index=index,
            prefix="Affine_",
        ),
        "projected_predictions": build_prediction_frame(
            projected_measured_predictions,
            measured_target_columns,
            index=index,
            prefix="Projected_",
        ),
        "raw_fractional_predictions": build_prediction_frame(
            raw_fractional_array,
            state_column_list,
            index=index,
            prefix="RawFractional_",
        ),
        "affine_fractional_predictions": build_prediction_frame(
            affine_fractional_array,
            state_column_list,
            index=index,
            prefix="AffineFractional_",
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
    if projection_stage_summary is not None:
        report["projection_stage_summary"] = projection_stage_summary

    if prediction_uncertainty is not None:
        report["uncertainty_metadata"] = pd.DataFrame([dict(prediction_uncertainty["metadata"])])
        report["affine_core_prediction_standard_errors"] = build_prediction_frame(
            prediction_uncertainty["affine_core_prediction_standard_errors"],
            target_column_list,
            index=index,
            prefix="AffineCoreSE_",
        )
        report["affine_core_prediction_confidence_interval_lower"] = build_prediction_frame(
            prediction_uncertainty["affine_core_prediction_confidence_interval_lower"],
            target_column_list,
            index=index,
            prefix="AffineCoreCI95Lower_",
        )
        report["affine_core_prediction_confidence_interval_upper"] = build_prediction_frame(
            prediction_uncertainty["affine_core_prediction_confidence_interval_upper"],
            target_column_list,
            index=index,
            prefix="AffineCoreCI95Upper_",
        )
        report["affine_core_prediction_interval_lower"] = build_prediction_frame(
            prediction_uncertainty["affine_core_prediction_interval_lower"],
            target_column_list,
            index=index,
            prefix="AffineCorePI95Lower_",
        )
        report["affine_core_prediction_interval_upper"] = build_prediction_frame(
            prediction_uncertainty["affine_core_prediction_interval_upper"],
            target_column_list,
            index=index,
            prefix="AffineCorePI95Upper_",
        )
        report["affine_core_prediction_interval_standard_errors"] = build_prediction_frame(
            prediction_uncertainty["affine_core_prediction_interval_standard_errors"],
            target_column_list,
            index=index,
            prefix="AffineCorePISE_",
        )
        report["prediction_uncertainty_summary"] = prediction_uncertainty["prediction_uncertainty_summary"].copy()

    return report


__all__ = [
    "build_icsor_projection_stage_frame",
    "build_icsor_projection_stage_summary",
    "build_prediction_frame",
    "evaluate_icsor_prediction_bundle",
    "evaluate_prediction_bundle",
]

