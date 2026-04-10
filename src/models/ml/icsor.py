"""icsor solved in fractional space with a collapsed measured-output affine core and LP projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm as normal_distribution, t as student_t_distribution

from src.utils.optuna import create_progress_bar
from src.utils.process import (
    DatasetSplit,
    ScalingBundle,
    TrainTestDatasetSplits,
    build_icsor_supervised_dataset,
    build_projection_operator,
    fit_scalers,
    project_to_nonnegative_feasible_set,
    transform_dataset_split,
)
from src.utils.simulation import load_model_params
from src.utils.test import (
    build_icsor_projection_stage_frame,
    build_icsor_projection_stage_summary,
    evaluate_icsor_prediction_bundle,
)
from src.utils.train import (
    load_model_bundle,
    persist_training_artifacts,
    resolve_model_hyperparameters,
    resolve_training_objective_label,
    serialize_report_frames,
    transform_feature_frame,
)


MODEL_NAME = "icsor"
VALID_AFFINE_ESTIMATORS = {"ols", "ridge"}
VALID_OLS_BACKENDS = {"numpy_lstsq"}
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_UNCERTAINTY_METHOD = "auto"
DEFAULT_PROJECTION_SOLVER = "highs"
DEFAULT_CONSTRAINT_TOLERANCE = 1e-8
DEFAULT_NONNEGATIVITY_TOLERANCE = 1e-10
DEFAULT_MEASURED_DEVIATION_WEIGHT = 1.0
DEFAULT_COMPONENT_DEVIATION_WEIGHT = 1.0
DEFAULT_TRADEOFF_PARAMETER = 1.0
DEFAULT_HIGHS_PRESOLVE = True
DEFAULT_HIGHS_MAX_ITER = 10000
DEFAULT_HIGHS_VERBOSE = False
DEFAULT_HIGHS_RETRY_WITHOUT_PRESOLVE = True
VALID_PROJECTION_SOLVERS = {"highs"}
RANK_DEFICIENT_ANALYTIC_NOTE = (
    "Analytic coefficient intervals were computed with a non-full-column-rank design matrix; "
    "the original design-basis coefficients are not uniquely identifiable coefficientwise, "
    "so interpret these intervals with caution."
)
RIDGE_ANALYTIC_INFERENCE_NOTE = (
    "Ridge uncertainty uses a conditional-on-fixed-penalty Gaussian approximation for the affine "
    "core. It uses the analytic ridge covariance formulas with a plug-in output covariance estimate "
    "and does not account for shrinkage bias, penalty-selection uncertainty, or the final "
    "nonnegative deployment correction."
)
AFFINE_CORE_PREDICTION_UNCERTAINTY_NOTE = (
    "Prediction uncertainty describes the affine measured-space core only and is not an exact "
    "interval for the final nonnegative deployed predictor when the LP correction is active."
)


def load_icsor_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the icsor model."""

    return load_model_params(MODEL_NAME, repo_root)


def _resolve_training_options(
    training_options: Mapping[str, Any] | None,
    *,
    objective_name: str,
) -> dict[str, Any]:
    options = dict(training_options or {})
    options.setdefault("show_progress", True)
    options.setdefault("progress_description", "Training icsor")
    options.setdefault("objective_name", objective_name)
    return options


def _resolve_affine_estimator(model_hyperparameters: Mapping[str, Any]) -> str:
    estimator_name = model_hyperparameters.get("affine_estimator")
    if estimator_name is None:
        legacy_backend = str(model_hyperparameters.get("ols_backend", "numpy_lstsq")).strip().lower()
        estimator_name = "ridge" if legacy_backend == "ridge" else "ols"

    resolved_name = str(estimator_name).strip().lower()
    if resolved_name not in VALID_AFFINE_ESTIMATORS:
        valid_values = ", ".join(sorted(VALID_AFFINE_ESTIMATORS))
        raise ValueError(f"icsor requires affine_estimator to be one of: {valid_values}.")
    return resolved_name


def _resolve_ols_backend(model_hyperparameters: Mapping[str, Any]) -> str:
    backend_name = str(model_hyperparameters.get("ols_backend", "numpy_lstsq")).strip().lower()
    if backend_name not in VALID_OLS_BACKENDS:
        valid_values = ", ".join(sorted(VALID_OLS_BACKENDS))
        raise ValueError(f"icsor requires ols_backend to be one of: {valid_values}.")
    return backend_name


def _resolve_ridge_alpha(model_hyperparameters: Mapping[str, Any]) -> float:
    ridge_alpha = float(model_hyperparameters.get("ridge_alpha", 0.0))
    if ridge_alpha < 0.0:
        raise ValueError("icsor ridge_alpha must be nonnegative.")
    return ridge_alpha


def _resolve_icsor_objective_label(model_hyperparameters: Mapping[str, Any]) -> str:
    estimator_name = _resolve_affine_estimator(model_hyperparameters)
    default_label = f"projected_{estimator_name}"
    explicit_label = model_hyperparameters.get("objective")
    if explicit_label is None:
        return default_label

    resolved_label = str(explicit_label)
    if resolved_label in {"projected_ols", "projected_ridge"}:
        return default_label
    return resolved_label


def _validate_scaling_configuration(hyperparameters: Mapping[str, Any]) -> None:
    if bool(hyperparameters.get("scale_features", False)):
        raise ValueError("icsor requires scale_features=False so fractional influent states remain physical.")
    if bool(hyperparameters.get("scale_targets", False)):
        raise ValueError(
            "icsor requires scale_targets=False because the collapsed affine-core target lives in measured-output space."
        )


def _validate_composition_shape(
    composition_matrix: np.ndarray,
    *,
    target_columns: list[str],
    constraint_columns: list[str],
) -> np.ndarray:
    composition_array = np.asarray(composition_matrix, dtype=float)
    expected_shape = (len(target_columns), len(constraint_columns))
    if composition_array.shape != expected_shape:
        raise ValueError(
            "icsor requires composition_matrix shape to match target_columns x constraint_columns."
        )
    return composition_array


def _resolve_confidence_level(model_hyperparameters: Mapping[str, Any]) -> float:
    confidence_level = float(model_hyperparameters.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL))
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("icsor confidence_level must be between 0 and 1.")
    return confidence_level


def _resolve_uncertainty_method(model_hyperparameters: Mapping[str, Any]) -> str:
    method_name = str(model_hyperparameters.get("uncertainty_method", DEFAULT_UNCERTAINTY_METHOD)).strip().lower()
    if method_name not in {"auto", "analytic", "none"}:
        raise ValueError("icsor uncertainty_method must be one of: analytic, auto, none.")
    return method_name


def _resolve_projection_settings(model_hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    solver_name = str(model_hyperparameters.get("projection_solver", DEFAULT_PROJECTION_SOLVER)).strip().lower()
    if solver_name not in VALID_PROJECTION_SOLVERS:
        valid_values = ", ".join(sorted(VALID_PROJECTION_SOLVERS))
        raise ValueError(f"icsor projection_solver must be one of: {valid_values}.")

    settings = {
        "projection_solver": solver_name,
        "constraint_tolerance": float(
            model_hyperparameters.get("constraint_tolerance", DEFAULT_CONSTRAINT_TOLERANCE)
        ),
        "nonnegativity_tolerance": float(
            model_hyperparameters.get("nonnegativity_tolerance", DEFAULT_NONNEGATIVITY_TOLERANCE)
        ),
        "measured_deviation_weight": float(
            model_hyperparameters.get("measured_deviation_weight", DEFAULT_MEASURED_DEVIATION_WEIGHT)
        ),
        "component_deviation_weight": float(
            model_hyperparameters.get("component_deviation_weight", DEFAULT_COMPONENT_DEVIATION_WEIGHT)
        ),
        "tradeoff_parameter": float(model_hyperparameters.get("tradeoff_parameter", DEFAULT_TRADEOFF_PARAMETER)),
        "highs_presolve": bool(model_hyperparameters.get("highs_presolve", DEFAULT_HIGHS_PRESOLVE)),
        "highs_max_iter": int(model_hyperparameters.get("highs_max_iter", DEFAULT_HIGHS_MAX_ITER)),
        "highs_verbose": bool(model_hyperparameters.get("highs_verbose", DEFAULT_HIGHS_VERBOSE)),
        "highs_retry_without_presolve": bool(
            model_hyperparameters.get(
                "highs_retry_without_presolve",
                DEFAULT_HIGHS_RETRY_WITHOUT_PRESOLVE,
            )
        ),
    }
    if settings["constraint_tolerance"] <= 0.0:
        raise ValueError("icsor constraint_tolerance must be positive.")
    if settings["nonnegativity_tolerance"] <= 0.0:
        raise ValueError("icsor nonnegativity_tolerance must be positive.")
    if settings["measured_deviation_weight"] <= 0.0:
        raise ValueError("icsor measured_deviation_weight must be positive.")
    if settings["component_deviation_weight"] <= 0.0:
        raise ValueError("icsor component_deviation_weight must be positive.")
    if settings["tradeoff_parameter"] <= 0.0:
        raise ValueError("icsor tradeoff_parameter must be positive.")
    if settings["highs_max_iter"] < 1:
        raise ValueError("icsor highs_max_iter must be at least 1.")
    return settings


def _compute_projection_matrices(A_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projection_matrix = build_projection_operator(np.asarray(A_matrix, dtype=float))
    projection_complement = np.eye(projection_matrix.shape[0], dtype=float) - projection_matrix
    return projection_matrix, projection_complement


def _make_block_range(start: int, width: int) -> dict[str, int]:
    return {"start": int(start), "stop": int(start + width)}


def _get_block_slice(design_schema: Mapping[str, Any], block_name: str) -> slice:
    block_range = dict(design_schema["block_ranges"][block_name])
    return slice(int(block_range["start"]), int(block_range["stop"]))


def _resolve_feature_partition(
    feature_columns: list[str],
    constraint_columns: list[str],
) -> tuple[list[str], list[str]]:
    operational_columns = [column_name for column_name in feature_columns if not str(column_name).startswith("In_")]
    influent_columns = [column_name for column_name in feature_columns if str(column_name).startswith("In_")]
    expected_influent_columns = [f"In_{column_name}" for column_name in constraint_columns]

    if not operational_columns:
        raise ValueError("icsor requires at least one operational feature column.")
    if influent_columns != expected_influent_columns:
        raise ValueError(
            "icsor requires influent fractional feature columns to match constraint_reference columns in order."
        )

    return operational_columns, influent_columns


def _build_outer_product_block(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    left_values = left_frame.to_numpy(dtype=float)
    right_values = right_frame.to_numpy(dtype=float)
    block_values = (left_values[:, :, None] * right_values[:, None, :]).reshape(len(left_frame), -1)
    block_columns = [
        f"{left_column}*{right_column}"
        for left_column in left_frame.columns
        for right_column in right_frame.columns
    ]
    return block_values, block_columns


def build_icsor_design_frame(
    feature_frame: pd.DataFrame,
    constraint_columns: list[str],
    *,
    include_bias_term: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the partitioned second-order icsor design matrix over operational and fractional influent inputs."""

    feature_column_list = list(feature_frame.columns)
    operational_columns, influent_columns = _resolve_feature_partition(feature_column_list, constraint_columns)
    operational_frame = feature_frame.loc[:, operational_columns]
    influent_frame = feature_frame.loc[:, influent_columns]

    design_blocks: list[np.ndarray] = []
    design_columns: list[str] = []
    block_ranges: dict[str, dict[str, int]] = {}
    cursor = 0

    operational_values = operational_frame.to_numpy(dtype=float)
    design_blocks.append(operational_values)
    design_columns.extend(operational_columns)
    block_ranges["linear_operational"] = _make_block_range(cursor, operational_values.shape[1])
    cursor += operational_values.shape[1]

    influent_values = influent_frame.to_numpy(dtype=float)
    design_blocks.append(influent_values)
    design_columns.extend(influent_columns)
    block_ranges["linear_influent"] = _make_block_range(cursor, influent_values.shape[1])
    cursor += influent_values.shape[1]

    if include_bias_term:
        bias_values = np.ones((len(feature_frame), 1), dtype=float)
        design_blocks.append(bias_values)
        design_columns.append("Bias")
        block_ranges["bias"] = _make_block_range(cursor, 1)
        cursor += 1
    else:
        block_ranges["bias"] = _make_block_range(cursor, 0)

    operational_quadratic_values, operational_quadratic_columns = _build_outer_product_block(
        operational_frame,
        operational_frame,
    )
    design_blocks.append(operational_quadratic_values)
    design_columns.extend(operational_quadratic_columns)
    block_ranges["quadratic_operational"] = _make_block_range(cursor, operational_quadratic_values.shape[1])
    cursor += operational_quadratic_values.shape[1]

    influent_quadratic_values, influent_quadratic_columns = _build_outer_product_block(
        influent_frame,
        influent_frame,
    )
    design_blocks.append(influent_quadratic_values)
    design_columns.extend(influent_quadratic_columns)
    block_ranges["quadratic_influent"] = _make_block_range(cursor, influent_quadratic_values.shape[1])
    cursor += influent_quadratic_values.shape[1]

    interaction_values, interaction_columns = _build_outer_product_block(
        operational_frame,
        influent_frame,
    )
    design_blocks.append(interaction_values)
    design_columns.extend(interaction_columns)
    block_ranges["interaction_operational_influent"] = _make_block_range(cursor, interaction_values.shape[1])
    cursor += interaction_values.shape[1]

    design_matrix = np.concatenate(design_blocks, axis=1)
    design_frame = pd.DataFrame(design_matrix, index=feature_frame.index, columns=design_columns)

    design_schema = {
        "feature_columns": feature_column_list,
        "constraint_columns": list(constraint_columns),
        "operational_columns": operational_columns,
        "influent_columns": influent_columns,
        "include_bias_term": bool(include_bias_term),
        "design_columns": design_columns,
        "block_ranges": block_ranges,
        "dimensions": {
            "design": int(design_matrix.shape[1]),
            "operational": int(len(operational_columns)),
            "influent": int(len(influent_columns)),
        },
    }
    return design_frame, design_schema


def _unpack_parameter_blocks(
    parameter_matrix: np.ndarray,
    design_schema: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    coefficient_matrix = np.asarray(parameter_matrix, dtype=float)
    operational_dim = int(design_schema["dimensions"]["operational"])
    influent_dim = int(design_schema["dimensions"]["influent"])

    linear_operational = coefficient_matrix[_get_block_slice(design_schema, "linear_operational"), :].T
    linear_influent = coefficient_matrix[_get_block_slice(design_schema, "linear_influent"), :].T

    bias_slice = _get_block_slice(design_schema, "bias")
    if bias_slice.stop > bias_slice.start:
        bias_vector = coefficient_matrix[bias_slice, :].reshape(-1)
    else:
        bias_vector = np.zeros(coefficient_matrix.shape[1], dtype=float)

    theta_uu = coefficient_matrix[_get_block_slice(design_schema, "quadratic_operational"), :].T.reshape(
        coefficient_matrix.shape[1],
        operational_dim,
        operational_dim,
    )
    theta_cc = coefficient_matrix[_get_block_slice(design_schema, "quadratic_influent"), :].T.reshape(
        coefficient_matrix.shape[1],
        influent_dim,
        influent_dim,
    )
    theta_uc = coefficient_matrix[_get_block_slice(design_schema, "interaction_operational_influent"), :].T.reshape(
        coefficient_matrix.shape[1],
        operational_dim,
        influent_dim,
    )

    return {
        "W_u": linear_operational,
        "W_in": linear_influent,
        "b": bias_vector,
        "Theta_uu": theta_uu,
        "Theta_cc": theta_cc,
        "Theta_uc": theta_uc,
    }


def _build_block_uncertainty_payload(
    design_schema: Mapping[str, Any],
    *,
    standard_error_matrix: np.ndarray,
    confidence_interval_lower: np.ndarray,
    confidence_interval_upper: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "W_u": {
            "standard_error": _unpack_parameter_blocks(standard_error_matrix, design_schema)["W_u"],
            "confidence_interval_lower": _unpack_parameter_blocks(confidence_interval_lower, design_schema)["W_u"],
            "confidence_interval_upper": _unpack_parameter_blocks(confidence_interval_upper, design_schema)["W_u"],
        },
        "W_in": {
            "standard_error": _unpack_parameter_blocks(standard_error_matrix, design_schema)["W_in"],
            "confidence_interval_lower": _unpack_parameter_blocks(confidence_interval_lower, design_schema)["W_in"],
            "confidence_interval_upper": _unpack_parameter_blocks(confidence_interval_upper, design_schema)["W_in"],
        },
        "b": {
            "standard_error": _unpack_parameter_blocks(standard_error_matrix, design_schema)["b"],
            "confidence_interval_lower": _unpack_parameter_blocks(confidence_interval_lower, design_schema)["b"],
            "confidence_interval_upper": _unpack_parameter_blocks(confidence_interval_upper, design_schema)["b"],
        },
        "Theta_uu": {
            "standard_error": _unpack_parameter_blocks(standard_error_matrix, design_schema)["Theta_uu"],
            "confidence_interval_lower": _unpack_parameter_blocks(confidence_interval_lower, design_schema)["Theta_uu"],
            "confidence_interval_upper": _unpack_parameter_blocks(confidence_interval_upper, design_schema)["Theta_uu"],
        },
        "Theta_cc": {
            "standard_error": _unpack_parameter_blocks(standard_error_matrix, design_schema)["Theta_cc"],
            "confidence_interval_lower": _unpack_parameter_blocks(confidence_interval_lower, design_schema)["Theta_cc"],
            "confidence_interval_upper": _unpack_parameter_blocks(confidence_interval_upper, design_schema)["Theta_cc"],
        },
        "Theta_uc": {
            "standard_error": _unpack_parameter_blocks(standard_error_matrix, design_schema)["Theta_uc"],
            "confidence_interval_lower": _unpack_parameter_blocks(confidence_interval_lower, design_schema)["Theta_uc"],
            "confidence_interval_upper": _unpack_parameter_blocks(confidence_interval_upper, design_schema)["Theta_uc"],
        },
    }


def _build_effective_parameter_matrix(
    raw_parameter_matrix: np.ndarray,
    design_schema: Mapping[str, Any],
    collapse_operator: np.ndarray,
    pass_through_operator: np.ndarray,
) -> np.ndarray:
    effective_parameter_matrix = np.asarray(raw_parameter_matrix, dtype=float) @ np.asarray(
        collapse_operator,
        dtype=float,
    ).T
    return _add_pass_through_to_parameter_matrix(
        effective_parameter_matrix,
        design_schema,
        np.asarray(pass_through_operator, dtype=float),
    )


def _add_pass_through_to_parameter_matrix(
    parameter_matrix: np.ndarray,
    design_schema: Mapping[str, Any],
    pass_through_operator: np.ndarray,
) -> np.ndarray:
    adjusted_parameter_matrix = np.asarray(parameter_matrix, dtype=float).copy()
    adjusted_parameter_matrix[_get_block_slice(design_schema, "linear_influent"), :] += np.asarray(
        pass_through_operator,
        dtype=float,
    ).T
    return adjusted_parameter_matrix


def _build_effective_parameter_samples(
    identifiable_parameter_samples: np.ndarray,
    design_schema: Mapping[str, Any],
    pass_through_operator: np.ndarray,
) -> np.ndarray:
    effective_parameter_samples = np.asarray(identifiable_parameter_samples, dtype=float).copy()
    linear_influent_slice = _get_block_slice(design_schema, "linear_influent")
    effective_parameter_samples[:, linear_influent_slice, :] += np.asarray(pass_through_operator, dtype=float).T
    return effective_parameter_samples


def _predict_with_parameter_matrix(
    design_frame: pd.DataFrame,
    parameter_matrix: np.ndarray,
) -> np.ndarray:
    return design_frame.to_numpy(dtype=float) @ np.asarray(parameter_matrix, dtype=float)


def _estimate_output_covariance(
    fit_residual_matrix: np.ndarray,
    *,
    degrees_of_freedom: float,
) -> np.ndarray:
    residual_array = np.asarray(fit_residual_matrix, dtype=float)
    if degrees_of_freedom <= 0:
        raise ValueError("icsor residual degrees of freedom must be positive to estimate analytic covariance.")
    return residual_array.T @ residual_array / float(degrees_of_freedom)


def _compute_interval_bounds(
    estimate_matrix: np.ndarray,
    standard_error_matrix: np.ndarray,
    *,
    critical_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    estimate_array = np.asarray(estimate_matrix, dtype=float)
    standard_error_array = np.asarray(standard_error_matrix, dtype=float)
    margin = float(critical_value) * standard_error_array
    return estimate_array - margin, estimate_array + margin


def _compute_prediction_summary_frame(
    target_columns: list[str],
    *,
    mean_standard_errors: np.ndarray,
    prediction_standard_errors: np.ndarray,
    confidence_interval_lower: np.ndarray,
    confidence_interval_upper: np.ndarray,
    prediction_interval_lower: np.ndarray,
    prediction_interval_upper: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": target_columns,
            "mean_standard_error_mean": np.mean(mean_standard_errors, axis=0),
            "mean_standard_error_max": np.max(mean_standard_errors, axis=0),
            "prediction_standard_error_mean": np.mean(prediction_standard_errors, axis=0),
            "prediction_standard_error_max": np.max(prediction_standard_errors, axis=0),
            "mean_confidence_interval_width_mean": np.mean(
                confidence_interval_upper - confidence_interval_lower,
                axis=0,
            ),
            "prediction_interval_width_mean": np.mean(
                prediction_interval_upper - prediction_interval_lower,
                axis=0,
            ),
        }
    )


def _compute_analytic_parameter_inference(
    identifiable_parameter_matrix: np.ndarray,
    *,
    design_matrix: np.ndarray,
    design_schema: Mapping[str, Any],
    fit_residual_matrix: np.ndarray,
    design_rank: int,
    confidence_level: float,
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    design_array = np.asarray(design_matrix, dtype=float)
    affine_estimator = _resolve_affine_estimator(model_hyperparameters)
    sample_count = int(design_array.shape[0])

    if affine_estimator == "ridge":
        ridge_operators = _build_ridge_linear_operators(
            design_array,
            design_schema=design_schema,
            ridge_alpha=_resolve_ridge_alpha(model_hyperparameters),
        )
        effective_model_degrees_of_freedom = float(ridge_operators["effective_degrees_of_freedom"])
        degrees_of_freedom = float(sample_count - effective_model_degrees_of_freedom)
        working_degrees_of_freedom = max(degrees_of_freedom, 1.0)
        coefficient_covariance_factor = np.asarray(ridge_operators["coefficient_covariance_factor"], dtype=float)
        design_gram_inverse = None
        critical_value = float(normal_distribution.ppf(0.5 * (1.0 + confidence_level)))
        analytic_distribution = "gaussian"
    else:
        degrees_of_freedom = int(sample_count - design_rank)
        working_degrees_of_freedom = max(degrees_of_freedom, 1)
        effective_model_degrees_of_freedom = int(design_rank)
        coefficient_covariance_factor = np.linalg.pinv(design_array.T @ design_array)
        design_gram_inverse = coefficient_covariance_factor
        critical_value = float(
            student_t_distribution.ppf(
                0.5 * (1.0 + confidence_level),
                df=working_degrees_of_freedom,
            )
        )
        analytic_distribution = "student_t"

    output_covariance = _estimate_output_covariance(
        fit_residual_matrix,
        degrees_of_freedom=working_degrees_of_freedom,
    )
    design_direction_variances = np.clip(np.diag(coefficient_covariance_factor), a_min=0.0, a_max=None)
    output_variances = np.clip(np.diag(output_covariance), a_min=0.0, a_max=None)
    standard_error_matrix = np.sqrt(np.outer(design_direction_variances, output_variances))
    ci_lower, ci_upper = _compute_interval_bounds(
        identifiable_parameter_matrix,
        standard_error_matrix,
        critical_value=critical_value,
    )
    return {
        "method": "analytic",
        "degrees_of_freedom": degrees_of_freedom,
        "effective_model_degrees_of_freedom": effective_model_degrees_of_freedom,
        "critical_value": critical_value,
        "analytic_distribution": analytic_distribution,
        "design_gram_inverse": design_gram_inverse,
        "coefficient_covariance_factor": coefficient_covariance_factor,
        "output_covariance": output_covariance,
        "standard_error_matrix": standard_error_matrix,
        "confidence_interval_lower": ci_lower,
        "confidence_interval_upper": ci_upper,
    }


def _compute_parameter_inference(
    identifiable_parameter_matrix: np.ndarray,
    *,
    design_matrix: np.ndarray,
    design_schema: Mapping[str, Any],
    fit_residual_matrix: np.ndarray,
    design_rank: int,
    confidence_level: float,
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any] | None:
    uncertainty_method = _resolve_uncertainty_method(model_hyperparameters)
    if uncertainty_method == "none":
        return None

    affine_estimator = _resolve_affine_estimator(model_hyperparameters)
    design_dimension = int(np.asarray(design_matrix, dtype=float).shape[1])
    rank_deficient = bool(int(design_rank) < design_dimension)
    inference_result = _compute_analytic_parameter_inference(
        identifiable_parameter_matrix,
        design_matrix=design_matrix,
        design_schema=design_schema,
        fit_residual_matrix=fit_residual_matrix,
        design_rank=int(design_rank),
        confidence_level=confidence_level,
        model_hyperparameters=model_hyperparameters,
    )

    inference_result["confidence_level"] = float(confidence_level)
    inference_result["design_dimension"] = design_dimension
    inference_result["design_rank"] = int(design_rank)
    inference_result["coefficient_target"] = "identifiable_measured_space_operator"
    inference_result["rank_deficient"] = rank_deficient
    inference_result["requested_method"] = uncertainty_method
    inference_result["affine_estimator"] = affine_estimator
    inference_result["ridge_alpha"] = _resolve_ridge_alpha(model_hyperparameters) if affine_estimator == "ridge" else None

    notes: list[str] = []
    if affine_estimator == "ridge":
        notes.append(RIDGE_ANALYTIC_INFERENCE_NOTE)
    if affine_estimator == "ols" and rank_deficient:
        notes.append(RANK_DEFICIENT_ANALYTIC_NOTE)
    inference_result["note"] = " ".join(notes) if notes else None
    inference_result["bootstrap_random_seed"] = None
    inference_result["bootstrap_samples"] = None
    return inference_result


def _compute_prediction_uncertainty_from_bundle(
    design_frame: pd.DataFrame,
    model_bundle: Mapping[str, Any],
    *,
    target_columns: list[str],
) -> dict[str, Any] | None:
    coefficient_inference = model_bundle.get("coefficient_inference")
    if coefficient_inference is None:
        return None

    design_matrix = design_frame.to_numpy(dtype=float)
    effective_parameter_matrix = np.asarray(model_bundle["effective_parameter_matrix"], dtype=float)
    affine_core_predictions = design_matrix @ effective_parameter_matrix
    confidence_level = float(coefficient_inference["confidence_level"])
    lower_quantile = 0.5 * (1.0 - confidence_level)
    upper_quantile = 1.0 - lower_quantile
    method_name = str(coefficient_inference["method"])

    if method_name == "analytic":
        covariance_factor = coefficient_inference.get("coefficient_covariance_factor")
        if covariance_factor is None:
            covariance_factor = coefficient_inference["design_gram_inverse"]
        coefficient_covariance_factor = np.asarray(covariance_factor, dtype=float)
        output_covariance = np.asarray(coefficient_inference["output_covariance"], dtype=float)
        critical_value = float(coefficient_inference["critical_value"])
        leverage = np.einsum("nd,df,nf->n", design_matrix, coefficient_covariance_factor, design_matrix)
        leverage = np.clip(leverage, a_min=0.0, a_max=None)
        output_variances = np.clip(np.diag(output_covariance), a_min=0.0, a_max=None)
        mean_standard_errors = np.sqrt(np.outer(leverage, output_variances))
        prediction_standard_errors = np.sqrt(np.outer(1.0 + leverage, output_variances))
        mean_ci_lower, mean_ci_upper = _compute_interval_bounds(
            affine_core_predictions,
            mean_standard_errors,
            critical_value=critical_value,
        )
        prediction_interval_lower, prediction_interval_upper = _compute_interval_bounds(
            affine_core_predictions,
            prediction_standard_errors,
            critical_value=critical_value,
        )
    else:
        identifiable_parameter_samples = np.asarray(coefficient_inference["bootstrap_parameter_samples"], dtype=float)
        effective_parameter_samples = _build_effective_parameter_samples(
            identifiable_parameter_samples,
            model_bundle["design_schema"],
            np.asarray(model_bundle["pass_through_operator"], dtype=float),
        )
        predicted_mean_samples = np.einsum("nd,bdk->nbk", design_matrix, effective_parameter_samples)
        mean_standard_errors = np.std(predicted_mean_samples, axis=1, ddof=1)
        mean_ci_lower = np.quantile(predicted_mean_samples, lower_quantile, axis=1)
        mean_ci_upper = np.quantile(predicted_mean_samples, upper_quantile, axis=1)

        fit_residuals = np.asarray(model_bundle["fit_residuals"], dtype=float)
        rng = np.random.default_rng(int(coefficient_inference["bootstrap_random_seed"]) + int(len(design_frame)))
        sampled_residual_indices = rng.integers(0, fit_residuals.shape[0], size=(predicted_mean_samples.shape[1], len(design_frame)))
        sampled_residuals = fit_residuals[sampled_residual_indices, :]
        predictive_samples = np.transpose(predicted_mean_samples, (1, 0, 2)) + sampled_residuals
        prediction_standard_errors = np.std(predictive_samples, axis=0, ddof=1)
        prediction_interval_lower = np.quantile(predictive_samples, lower_quantile, axis=0)
        prediction_interval_upper = np.quantile(predictive_samples, upper_quantile, axis=0)

    return {
        "metadata": {
            "method": method_name,
            "requested_method": coefficient_inference.get("requested_method", method_name),
            "confidence_level": confidence_level,
            "coefficient_target": str(coefficient_inference["coefficient_target"]),
            "prediction_target": "affine_core_measured_prediction",
            "affine_estimator": coefficient_inference.get("affine_estimator", "ols"),
            "ridge_alpha": coefficient_inference.get("ridge_alpha"),
            "rank_deficient": bool(coefficient_inference["rank_deficient"]),
            "design_rank": int(coefficient_inference["design_rank"]),
            "design_dimension": int(coefficient_inference["design_dimension"]),
            "degrees_of_freedom": coefficient_inference["degrees_of_freedom"],
            "note": coefficient_inference.get("note"),
            "deployment_note": AFFINE_CORE_PREDICTION_UNCERTAINTY_NOTE,
        },
        "affine_core_prediction_standard_errors": mean_standard_errors,
        "affine_core_prediction_confidence_interval_lower": mean_ci_lower,
        "affine_core_prediction_confidence_interval_upper": mean_ci_upper,
        "affine_core_prediction_interval_lower": prediction_interval_lower,
        "affine_core_prediction_interval_upper": prediction_interval_upper,
        "affine_core_prediction_interval_standard_errors": prediction_standard_errors,
        "prediction_uncertainty_summary": _compute_prediction_summary_frame(
            target_columns,
            mean_standard_errors=mean_standard_errors,
            prediction_standard_errors=prediction_standard_errors,
            confidence_interval_lower=mean_ci_lower,
            confidence_interval_upper=mean_ci_upper,
            prediction_interval_lower=prediction_interval_lower,
            prediction_interval_upper=prediction_interval_upper,
        ),
    }


def _solve_with_numpy_lstsq(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    *,
    rcond_value: Any,
) -> dict[str, Any]:
    solution_matrix, residuals, rank, singular_values = np.linalg.lstsq(
        left_matrix,
        right_matrix,
        rcond=None if rcond_value is None else float(rcond_value),
    )
    return {
        "solution_matrix": np.asarray(solution_matrix, dtype=float),
        "residuals": np.asarray(residuals, dtype=float),
        "rank": int(rank),
        "singular_values": np.asarray(singular_values, dtype=float),
        "backend_used": "numpy_lstsq",
        "device_label": "cpu",
    }


def _build_ridge_penalty_matrix(design_schema: Mapping[str, Any], design_dimension: int) -> np.ndarray:
    penalty_matrix = np.eye(int(design_dimension), dtype=float)
    bias_slice = _get_block_slice(design_schema, "bias")
    if bias_slice.stop > bias_slice.start:
        penalty_matrix[bias_slice, bias_slice] = 0.0
    return penalty_matrix


def _build_ridge_linear_operators(
    design_matrix: np.ndarray,
    *,
    design_schema: Mapping[str, Any],
    ridge_alpha: float,
) -> dict[str, np.ndarray | float]:
    design_array = np.asarray(design_matrix, dtype=float)
    design_dimension = int(design_array.shape[1])
    penalty_matrix = _build_ridge_penalty_matrix(design_schema, design_dimension)
    normal_matrix = design_array.T @ design_array + float(ridge_alpha) * penalty_matrix
    normal_inverse = np.linalg.pinv(normal_matrix)
    operator_matrix = normal_inverse @ design_array.T
    hat_matrix = design_array @ operator_matrix
    return {
        "penalty_matrix": penalty_matrix,
        "normal_matrix": normal_matrix,
        "normal_inverse": normal_inverse,
        "operator_matrix": operator_matrix,
        "hat_matrix": hat_matrix,
        "coefficient_covariance_factor": operator_matrix @ operator_matrix.T,
        "effective_degrees_of_freedom": float(np.trace(hat_matrix)),
    }


def _solve_with_numpy_ridge(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    *,
    design_schema: Mapping[str, Any],
    ridge_alpha: float,
) -> dict[str, Any]:
    left_array = np.asarray(left_matrix, dtype=float)
    right_array = np.asarray(right_matrix, dtype=float)
    ridge_operators = _build_ridge_linear_operators(
        left_array,
        design_schema=design_schema,
        ridge_alpha=ridge_alpha,
    )
    solution_matrix = np.asarray(ridge_operators["operator_matrix"], dtype=float) @ right_array
    fitted_values = left_array @ solution_matrix
    singular_values = np.linalg.svd(left_array, compute_uv=False)
    return {
        "solution_matrix": np.asarray(solution_matrix, dtype=float),
        "residuals": np.asarray(right_array - fitted_values, dtype=float),
        "rank": int(np.linalg.matrix_rank(left_array)),
        "singular_values": np.asarray(singular_values, dtype=float),
        "backend_used": "numpy_ridge_closed_form",
        "device_label": "cpu",
        "ridge_alpha": float(ridge_alpha),
        "ridge_penalty_matrix": np.asarray(ridge_operators["penalty_matrix"], dtype=float),
        "effective_degrees_of_freedom": float(ridge_operators["effective_degrees_of_freedom"]),
    }


def _solve_identifiable_affine_parameters(
    design_matrix: np.ndarray,
    transformed_target_matrix: np.ndarray,
    *,
    design_schema: Mapping[str, Any],
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    affine_estimator = _resolve_affine_estimator(model_hyperparameters)
    if affine_estimator == "ridge":
        return _solve_with_numpy_ridge(
            design_matrix,
            transformed_target_matrix,
            design_schema=design_schema,
            ridge_alpha=_resolve_ridge_alpha(model_hyperparameters),
        )

    return _solve_with_numpy_lstsq(
        design_matrix,
        transformed_target_matrix,
        rcond_value=model_hyperparameters.get("lstsq_rcond"),
    )


def _solve_projected_affine_core(
    design_matrix: np.ndarray,
    transformed_target_matrix: np.ndarray,
    design_schema: Mapping[str, Any],
    collapse_operator: np.ndarray,
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    affine_estimator = _resolve_affine_estimator(model_hyperparameters)
    requested_backend = _resolve_ols_backend(model_hyperparameters) if affine_estimator == "ols" else "numpy_ridge_closed_form"
    transformed_parameter_result = _solve_identifiable_affine_parameters(
        design_matrix,
        transformed_target_matrix,
        design_schema=design_schema,
        model_hyperparameters=model_hyperparameters,
    )
    raw_parameter_result = _solve_with_numpy_lstsq(
        collapse_operator,
        transformed_parameter_result["solution_matrix"].T,
        rcond_value=model_hyperparameters.get("lstsq_rcond"),
    )
    raw_parameter_matrix = np.asarray(raw_parameter_result["solution_matrix"], dtype=float).T
    fit_residual_matrix = transformed_target_matrix - design_matrix @ raw_parameter_matrix @ collapse_operator.T

    return {
        "parameter_matrix": raw_parameter_matrix,
        "transformed_parameter_matrix": np.asarray(transformed_parameter_result["solution_matrix"], dtype=float),
        "fit_residual_matrix": fit_residual_matrix,
        "design_residuals": np.asarray(transformed_parameter_result["residuals"], dtype=float),
        "design_rank": int(transformed_parameter_result["rank"]),
        "design_singular_values": np.asarray(transformed_parameter_result["singular_values"], dtype=float),
        "collapse_residuals": np.asarray(raw_parameter_result["residuals"], dtype=float),
        "collapse_rank": int(raw_parameter_result["rank"]),
        "collapse_singular_values": np.asarray(raw_parameter_result["singular_values"], dtype=float),
        "estimation_metadata": {
            "affine_estimator": affine_estimator,
            "requested_backend": requested_backend,
            "backend_used": str(transformed_parameter_result["backend_used"]),
            "collapse_backend_used": str(raw_parameter_result["backend_used"]),
            "device_label": "cpu",
            "ridge_alpha": transformed_parameter_result.get("ridge_alpha"),
            "ridge_effective_degrees_of_freedom": transformed_parameter_result.get("effective_degrees_of_freedom"),
            "fallback_reason": None,
        },
    }


def _build_model_bundle(
    training_result: Mapping[str, Any],
    scaling_bundle: ScalingBundle,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    constraint_columns: list[str],
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray,
    composition_source: Mapping[str, Any] | None,
    model_hyperparameters: Mapping[str, Any],
    training_options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_bundle_schema_version": 4,
        "model_name": MODEL_NAME,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "constraint_columns": constraint_columns,
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "composition_matrix": np.asarray(composition_matrix, dtype=float),
        "composition_source": dict(composition_source or {}),
        "projection_matrix": np.asarray(training_result["projection_matrix"], dtype=float),
        "projection_complement": np.asarray(training_result["projection_complement"], dtype=float),
        "collapse_operator": np.asarray(training_result["collapse_operator"], dtype=float),
        "pass_through_operator": np.asarray(training_result["pass_through_operator"], dtype=float),
        "projection_settings": dict(training_result["projection_settings"]),
        "design_schema": dict(training_result["design_schema"]),
        "raw_parameter_matrix": np.asarray(training_result["raw_parameter_matrix"], dtype=float),
        "identifiable_parameter_matrix": np.asarray(training_result["identifiable_parameter_matrix"], dtype=float),
        "raw_measured_parameter_matrix": np.asarray(training_result["raw_measured_parameter_matrix"], dtype=float),
        "effective_parameter_matrix": np.asarray(training_result["effective_parameter_matrix"], dtype=float),
        "affine_core_parameter_matrix": np.asarray(training_result["effective_parameter_matrix"], dtype=float),
        "raw_coefficients": dict(training_result["raw_coefficients"]),
        "identifiable_coefficients": dict(training_result["identifiable_coefficients"]),
        "effective_coefficients": dict(training_result["effective_coefficients"]),
        "affine_core_coefficients": dict(training_result["effective_coefficients"]),
        "coefficient_inference": None
        if training_result["coefficient_inference"] is None
        else dict(training_result["coefficient_inference"]),
        "identifiable_coefficient_uncertainty": None
        if training_result["identifiable_coefficient_uncertainty"] is None
        else dict(training_result["identifiable_coefficient_uncertainty"]),
        "effective_coefficient_uncertainty": None
        if training_result["effective_coefficient_uncertainty"] is None
        else dict(training_result["effective_coefficient_uncertainty"]),
        "affine_core_coefficient_uncertainty": None
        if training_result["effective_coefficient_uncertainty"] is None
        else dict(training_result["effective_coefficient_uncertainty"]),
        "fit_residuals": np.asarray(training_result["fit_residuals"], dtype=float),
        "estimation_metadata": dict(training_result["estimation_metadata"]),
        "ols_metadata": dict(training_result["estimation_metadata"]),
        "scaling_bundle": scaling_bundle,
        "model_hyperparameters": dict(model_hyperparameters),
        "training_options": dict(training_options),
    }


def _predict_from_bundle(
    feature_frame: pd.DataFrame,
    constraint_reference: pd.DataFrame,
    model_bundle: Mapping[str, Any],
    *,
    scaling_bundle: ScalingBundle,
) -> dict[str, Any]:
    transformed_features = transform_feature_frame(feature_frame, scaling_bundle)
    constraint_columns = list(model_bundle["constraint_columns"])
    selected_constraint_reference = constraint_reference.loc[:, constraint_columns].copy()
    design_frame, _ = build_icsor_design_frame(
        transformed_features,
        constraint_columns,
        include_bias_term=bool(model_bundle["design_schema"]["include_bias_term"]),
    )

    raw_fractional_predictions = _predict_with_parameter_matrix(design_frame, model_bundle["raw_parameter_matrix"])
    projection_settings = dict(
        model_bundle.get(
            "projection_settings",
            _resolve_projection_settings(model_bundle.get("model_hyperparameters", {})),
        )
    )
    projection_details = project_to_nonnegative_feasible_set(
        raw_fractional_predictions,
        selected_constraint_reference.to_numpy(dtype=float),
        np.asarray(model_bundle["A_matrix"], dtype=float),
        np.asarray(model_bundle["composition_matrix"], dtype=float),
        projection_operator=np.asarray(model_bundle["projection_matrix"], dtype=float),
        projection_complement=np.asarray(model_bundle["projection_complement"], dtype=float),
        **projection_settings,
    )
    affine_fractional_predictions = np.asarray(projection_details["affine_predictions"], dtype=float)
    projected_fractional_predictions = np.asarray(projection_details["projected_predictions"], dtype=float)
    raw_measured_predictions = raw_fractional_predictions @ np.asarray(model_bundle["composition_matrix"], dtype=float).T
    affine_measured_predictions = affine_fractional_predictions @ np.asarray(model_bundle["composition_matrix"], dtype=float).T
    projected_measured_predictions = projected_fractional_predictions @ np.asarray(
        model_bundle["composition_matrix"],
        dtype=float,
    ).T
    prediction_result = {
        "design_frame": design_frame,
        "raw_measured_predictions": raw_measured_predictions,
        "affine_measured_predictions": affine_measured_predictions,
        "projected_measured_predictions": projected_measured_predictions,
        "raw_fractional_predictions": raw_fractional_predictions,
        "affine_fractional_predictions": affine_fractional_predictions,
        "projected_fractional_predictions": projected_fractional_predictions,
        "constraint_reference": selected_constraint_reference,
        "projection_details": projection_details,
    }
    prediction_uncertainty = _compute_prediction_uncertainty_from_bundle(
        design_frame,
        model_bundle,
        target_columns=list(model_bundle["target_columns"]),
    )
    if prediction_uncertainty is not None:
        prediction_result["prediction_uncertainty"] = prediction_uncertainty
    return prediction_result


def train_icsor_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, Any],
    *,
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray,
    training_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the strict icsor estimator on notebook-prepared fractional features and measured targets."""

    _validate_scaling_configuration(model_hyperparameters)
    confidence_level = _resolve_confidence_level(model_hyperparameters)
    projection_settings = _resolve_projection_settings(model_hyperparameters)
    objective_label = _resolve_icsor_objective_label(model_hyperparameters)

    feature_frame = pd.DataFrame(training_dataset["features"])
    target_frame = pd.DataFrame(training_dataset["targets"])
    constraint_frame = pd.DataFrame(training_dataset["constraint_reference"])
    composition_array = _validate_composition_shape(
        composition_matrix,
        target_columns=list(target_frame.columns),
        constraint_columns=list(constraint_frame.columns),
    )

    options = _resolve_training_options(training_options, objective_name=objective_label)
    projection_matrix, projection_complement = _compute_projection_matrices(np.asarray(A_matrix, dtype=float))
    collapse_operator = composition_array @ projection_complement
    pass_through_operator = composition_array @ projection_matrix

    design_frame, design_schema = build_icsor_design_frame(
        feature_frame,
        list(constraint_frame.columns),
        include_bias_term=bool(model_hyperparameters.get("include_bias_term", True)),
    )

    progress_bar = create_progress_bar(
        total=4,
        desc=str(options["progress_description"]),
        enabled=bool(options["show_progress"]),
        unit="stage",
    )
    try:
        progress_bar.set_postfix(stage="project_target", objective=str(options["objective_name"]))
        transformed_target_values = target_frame.to_numpy(dtype=float) - constraint_frame.to_numpy(dtype=float) @ pass_through_operator.T
        design_matrix = design_frame.to_numpy(dtype=float)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="solve", objective=str(options["objective_name"]))
        solve_result = _solve_projected_affine_core(
            design_matrix,
            transformed_target_values,
            design_schema,
            collapse_operator,
            model_hyperparameters,
        )
        raw_parameter_matrix = np.asarray(solve_result["parameter_matrix"], dtype=float)
        identifiable_parameter_matrix = np.asarray(solve_result["transformed_parameter_matrix"], dtype=float)
        fit_residual_matrix = np.asarray(solve_result["fit_residual_matrix"], dtype=float)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="assemble", objective=str(options["objective_name"]))
        raw_measured_parameter_matrix = raw_parameter_matrix @ composition_array.T
        effective_parameter_matrix = _build_effective_parameter_matrix(
            raw_parameter_matrix,
            design_schema,
            collapse_operator,
            pass_through_operator,
        )
        raw_coefficients = _unpack_parameter_blocks(raw_parameter_matrix, design_schema)
        identifiable_coefficients = _unpack_parameter_blocks(identifiable_parameter_matrix, design_schema)
        effective_coefficients = _unpack_parameter_blocks(effective_parameter_matrix, design_schema)
        coefficient_inference = _compute_parameter_inference(
            identifiable_parameter_matrix,
            design_matrix=design_matrix,
            design_schema=design_schema,
            fit_residual_matrix=fit_residual_matrix,
            design_rank=int(solve_result["design_rank"]),
            confidence_level=confidence_level,
            model_hyperparameters=model_hyperparameters,
        )
        if coefficient_inference is None:
            identifiable_coefficient_uncertainty = None
            effective_coefficient_uncertainty = None
        else:
            identifiable_coefficient_uncertainty = _build_block_uncertainty_payload(
                design_schema,
                standard_error_matrix=np.asarray(coefficient_inference["standard_error_matrix"], dtype=float),
                confidence_interval_lower=np.asarray(coefficient_inference["confidence_interval_lower"], dtype=float),
                confidence_interval_upper=np.asarray(coefficient_inference["confidence_interval_upper"], dtype=float),
            )
            effective_coefficient_uncertainty = _build_block_uncertainty_payload(
                design_schema,
                standard_error_matrix=np.asarray(coefficient_inference["standard_error_matrix"], dtype=float),
                confidence_interval_lower=np.asarray(
                    _add_pass_through_to_parameter_matrix(
                        np.asarray(coefficient_inference["confidence_interval_lower"], dtype=float),
                        design_schema,
                        pass_through_operator,
                    ),
                    dtype=float,
                ),
                confidence_interval_upper=np.asarray(
                    _add_pass_through_to_parameter_matrix(
                        np.asarray(coefficient_inference["confidence_interval_upper"], dtype=float),
                        design_schema,
                        pass_through_operator,
                    ),
                    dtype=float,
                ),
            )
        progress_bar.update(1)

        progress_bar.set_postfix(stage="predict", objective=str(options["objective_name"]))
        training_raw_fractional_predictions = _predict_with_parameter_matrix(design_frame, raw_parameter_matrix)
        training_projection_details = project_to_nonnegative_feasible_set(
            training_raw_fractional_predictions,
            constraint_frame.to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
            composition_array,
            projection_operator=projection_matrix,
            projection_complement=projection_complement,
            **projection_settings,
        )
        training_affine_fractional_predictions = np.asarray(
            training_projection_details["affine_predictions"],
            dtype=float,
        )
        training_projected_fractional_predictions = np.asarray(
            training_projection_details["projected_predictions"],
            dtype=float,
        )
        training_raw_predictions = training_raw_fractional_predictions @ composition_array.T
        training_affine_predictions = training_affine_fractional_predictions @ composition_array.T
        training_projected_predictions = training_projected_fractional_predictions @ composition_array.T
        train_affine_mse = float(np.mean((target_frame.to_numpy(dtype=float) - training_affine_predictions) ** 2))
        train_mse = float(np.mean((target_frame.to_numpy(dtype=float) - training_projected_predictions) ** 2))
        progress_bar.set_postfix(
            stage="predict",
            objective=str(options["objective_name"]),
            objective_value=f"{train_mse:.6g}",
        )
        progress_bar.update(1)
    finally:
        progress_bar.close()

    return {
        "design_schema": design_schema,
        "projection_matrix": projection_matrix,
        "projection_complement": projection_complement,
        "collapse_operator": collapse_operator,
        "pass_through_operator": pass_through_operator,
        "projection_settings": projection_settings,
        "raw_parameter_matrix": raw_parameter_matrix,
        "identifiable_parameter_matrix": identifiable_parameter_matrix,
        "raw_measured_parameter_matrix": raw_measured_parameter_matrix,
        "effective_parameter_matrix": effective_parameter_matrix,
        "raw_coefficients": raw_coefficients,
        "identifiable_coefficients": identifiable_coefficients,
        "effective_coefficients": effective_coefficients,
        "coefficient_inference": coefficient_inference,
        "identifiable_coefficient_uncertainty": identifiable_coefficient_uncertainty,
        "effective_coefficient_uncertainty": effective_coefficient_uncertainty,
        "training_raw_predictions": pd.DataFrame(
            training_raw_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_affine_predictions": pd.DataFrame(
            training_affine_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_projected_predictions": pd.DataFrame(
            training_projected_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_raw_fractional_predictions": pd.DataFrame(
            training_raw_fractional_predictions,
            index=constraint_frame.index,
            columns=constraint_frame.columns,
        ),
        "training_affine_fractional_predictions": pd.DataFrame(
            training_affine_fractional_predictions,
            index=constraint_frame.index,
            columns=constraint_frame.columns,
        ),
        "training_projected_fractional_predictions": pd.DataFrame(
            training_projected_fractional_predictions,
            index=constraint_frame.index,
            columns=constraint_frame.columns,
        ),
        "training_affine_mse": train_affine_mse,
        "training_mse": train_mse,
        "training_projection_details": training_projection_details,
        "fit_residuals": fit_residual_matrix,
        "design_residuals": np.asarray(solve_result["design_residuals"], dtype=float),
        "design_rank": int(solve_result["design_rank"]),
        "design_singular_values": np.asarray(solve_result["design_singular_values"], dtype=float),
        "collapse_residuals": np.asarray(solve_result["collapse_residuals"], dtype=float),
        "collapse_rank": int(solve_result["collapse_rank"]),
        "collapse_singular_values": np.asarray(solve_result["collapse_singular_values"], dtype=float),
        "estimation_metadata": dict(solve_result["estimation_metadata"]),
        "ols_metadata": dict(solve_result["estimation_metadata"]),
    }


def predict_icsor_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted icsor bundle and generate aligned measured and fractional predictions."""

    model_bundle = load_model_bundle(model_path)
    scaling_bundle: ScalingBundle = model_bundle["scaling_bundle"]
    bundle_composition_source = dict(model_bundle.get("composition_source") or {})

    if isinstance(test_dataset, pd.DataFrame):
        if metadata is None or composition_matrix is None:
            raise ValueError("metadata and composition_matrix are required when predicting from a raw dataset.")

        metadata_composition_source = dict(metadata.get("composition_source") or {})
        bundle_sha = bundle_composition_source.get("workbook_sha256")
        metadata_sha = metadata_composition_source.get("workbook_sha256")
        if bundle_sha is not None and metadata_sha is not None and str(bundle_sha) != str(metadata_sha):
            raise ValueError(
                "Raw-dataset prediction requires metadata composition_source workbook_sha256 to match the "
                "trained model bundle composition source."
            )

        icsor_dataset = build_icsor_supervised_dataset(
            test_dataset,
            dict(metadata),
            np.asarray(composition_matrix, dtype=float),
        )
        feature_frame = icsor_dataset.features
        constraint_reference = icsor_dataset.constraint_reference
    else:
        feature_frame = pd.DataFrame(test_dataset["features"], columns=scaling_bundle.feature_columns)
        constraint_reference = pd.DataFrame(
            test_dataset["constraint_reference"],
            columns=model_bundle["constraint_columns"],
        )

    prediction_payload = _predict_from_bundle(
        feature_frame,
        constraint_reference,
        model_bundle,
        scaling_bundle=scaling_bundle,
    )

    result = {
        "raw_predictions": pd.DataFrame(
            prediction_payload["raw_measured_predictions"],
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "affine_predictions": pd.DataFrame(
            prediction_payload["affine_measured_predictions"],
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "projected_predictions": pd.DataFrame(
            prediction_payload["projected_measured_predictions"],
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "raw_fractional_predictions": pd.DataFrame(
            prediction_payload["raw_fractional_predictions"],
            index=feature_frame.index,
            columns=model_bundle["constraint_columns"],
        ),
        "affine_fractional_predictions": pd.DataFrame(
            prediction_payload["affine_fractional_predictions"],
            index=feature_frame.index,
            columns=model_bundle["constraint_columns"],
        ),
        "projected_fractional_predictions": pd.DataFrame(
            prediction_payload["projected_fractional_predictions"],
            index=feature_frame.index,
            columns=model_bundle["constraint_columns"],
        ),
        "constraint_reference": prediction_payload["constraint_reference"],
        "projection_stage_diagnostics": build_icsor_projection_stage_frame(
            prediction_payload["projection_details"],
            index=feature_frame.index,
        ),
        "projection_stage_summary": build_icsor_projection_stage_summary(prediction_payload["projection_details"]),
    }

    prediction_uncertainty = prediction_payload.get("prediction_uncertainty")
    if prediction_uncertainty is not None:
        result.update(
            {
                "prediction_uncertainty_metadata": dict(prediction_uncertainty["metadata"]),
                "affine_core_prediction_standard_errors": pd.DataFrame(
                    prediction_uncertainty["affine_core_prediction_standard_errors"],
                    index=feature_frame.index,
                    columns=model_bundle["target_columns"],
                ),
                "affine_core_prediction_confidence_interval_lower": pd.DataFrame(
                    prediction_uncertainty["affine_core_prediction_confidence_interval_lower"],
                    index=feature_frame.index,
                    columns=model_bundle["target_columns"],
                ),
                "affine_core_prediction_confidence_interval_upper": pd.DataFrame(
                    prediction_uncertainty["affine_core_prediction_confidence_interval_upper"],
                    index=feature_frame.index,
                    columns=model_bundle["target_columns"],
                ),
                "affine_core_prediction_interval_lower": pd.DataFrame(
                    prediction_uncertainty["affine_core_prediction_interval_lower"],
                    index=feature_frame.index,
                    columns=model_bundle["target_columns"],
                ),
                "affine_core_prediction_interval_upper": pd.DataFrame(
                    prediction_uncertainty["affine_core_prediction_interval_upper"],
                    index=feature_frame.index,
                    columns=model_bundle["target_columns"],
                ),
                "affine_core_prediction_interval_standard_errors": pd.DataFrame(
                    prediction_uncertainty["affine_core_prediction_interval_standard_errors"],
                    index=feature_frame.index,
                    columns=model_bundle["target_columns"],
                ),
                "prediction_uncertainty_summary": prediction_uncertainty["prediction_uncertainty_summary"],
            }
        )

    return result


def run_icsor_pipeline(
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    A_matrix: np.ndarray,
    *,
    composition_matrix: np.ndarray,
    composition_source: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    model_params: Mapping[str, Any] | None = None,
    model_hyperparameters: Mapping[str, Any] | None = None,
    optuna_summary: Mapping[str, Any] | None = None,
    show_progress: bool = True,
    persist_artifacts: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and optionally persist one strict-theory icsor bundle."""

    params = dict(model_params) if model_params is not None else load_icsor_params(repo_root)
    split_params = dict(params["hyperparameters"])
    selected_hyperparameters = resolve_model_hyperparameters(params, model_hyperparameters)
    selected_hyperparameters["objective"] = _resolve_icsor_objective_label(selected_hyperparameters)
    _validate_scaling_configuration({**split_params, **selected_hyperparameters})
    objective_label = str(selected_hyperparameters["objective"])

    progress_bar = create_progress_bar(
        total=5,
        desc="Training icsor",
        enabled=show_progress,
        unit="stage",
    )

    try:
        progress_bar.set_postfix(stage="scaling", objective=objective_label)
        scaling_bundle = fit_scalers(
            training_split,
            scale_features=bool(split_params["scale_features"]),
            scale_targets=bool(split_params["scale_targets"]),
        )
        scaled_training_split = transform_dataset_split(training_split, scaling_bundle)
        scaled_test_split = transform_dataset_split(test_split, scaling_bundle)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="fit", objective=objective_label)
        training_result = train_icsor_model(
            {
                "features": scaled_training_split.features,
                "targets": scaled_training_split.targets,
                "constraint_reference": scaled_training_split.constraint_reference,
            },
            selected_hyperparameters,
            A_matrix=np.asarray(A_matrix, dtype=float),
            composition_matrix=np.asarray(composition_matrix, dtype=float),
            training_options={
                "show_progress": False,
                "progress_description": "Training icsor",
                "objective_name": objective_label,
            },
        )
        progress_bar.update(1)

        model_bundle = _build_model_bundle(
            training_result,
            scaling_bundle,
            feature_columns=list(training_split.features.columns),
            target_columns=list(training_split.targets.columns),
            constraint_columns=list(training_split.constraint_reference.columns),
            A_matrix=np.asarray(A_matrix, dtype=float),
            composition_matrix=np.asarray(composition_matrix, dtype=float),
            composition_source=composition_source,
            model_hyperparameters=selected_hyperparameters,
            training_options={
                "objective_name": objective_label,
                "show_progress": show_progress,
            },
        )

        progress_bar.set_postfix(stage="evaluate_train", objective=objective_label)
        train_prediction_payload = _predict_from_bundle(
            scaled_training_split.features,
            scaled_training_split.constraint_reference,
            model_bundle,
            scaling_bundle=scaling_bundle,
        )
        progress_bar.update(1)

        progress_bar.set_postfix(stage="evaluate_test", objective=objective_label)
        test_prediction_payload = _predict_from_bundle(
            scaled_test_split.features,
            scaled_test_split.constraint_reference,
            model_bundle,
            scaling_bundle=scaling_bundle,
        )
        train_report = evaluate_icsor_prediction_bundle(
            training_split.targets.to_numpy(dtype=float),
            train_prediction_payload["raw_fractional_predictions"],
            train_prediction_payload["affine_fractional_predictions"],
            train_prediction_payload["projected_fractional_predictions"],
            train_prediction_payload["constraint_reference"].to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
            np.asarray(composition_matrix, dtype=float),
            training_split.targets.columns,
            training_split.constraint_reference.columns,
            index=training_split.targets.index,
            prediction_uncertainty=train_prediction_payload.get("prediction_uncertainty"),
            projection_details=train_prediction_payload.get("projection_details"),
        )
        test_report = evaluate_icsor_prediction_bundle(
            test_split.targets.to_numpy(dtype=float),
            test_prediction_payload["raw_fractional_predictions"],
            test_prediction_payload["affine_fractional_predictions"],
            test_prediction_payload["projected_fractional_predictions"],
            test_prediction_payload["constraint_reference"].to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
            np.asarray(composition_matrix, dtype=float),
            test_split.targets.columns,
            test_split.constraint_reference.columns,
            index=test_split.targets.index,
            prediction_uncertainty=test_prediction_payload.get("prediction_uncertainty"),
            projection_details=test_prediction_payload.get("projection_details"),
        )
        progress_bar.update(1)

        dataset_splits = TrainTestDatasetSplits(train=training_split, test=test_split)
        artifact_paths: dict[str, Path | None] = {
            "model_bundle": None,
            "metrics": None,
            "optuna": None,
        }
        progress_bar.set_postfix(stage="persist", objective=objective_label)
        artifact_options = dict(params.get("artifact_options", {}))
        if persist_artifacts and bool(artifact_options.get("persist_model", True)):
            metrics_payload = {
                "train": serialize_report_frames(train_report),
                "test": serialize_report_frames(test_report),
                "split_sizes": {
                    "train": int(len(dataset_splits.train.features)),
                    "test": int(len(dataset_splits.test.features)),
                },
            }
            optuna_payload = (
                dict(optuna_summary)
                if optuna_summary is not None and bool(artifact_options.get("persist_optuna", True))
                else None
            )
            metrics_summary = metrics_payload if bool(artifact_options.get("persist_metrics", True)) else None
            artifact_paths = persist_training_artifacts(
                MODEL_NAME,
                model_bundle,
                metrics_summary=metrics_summary,
                optuna_summary=optuna_payload,
                repo_root=repo_root,
                timestamp=timestamp,
            )
        progress_bar.update(1)
    finally:
        progress_bar.close()

    return {
        "best_hyperparameters": selected_hyperparameters,
        "optuna_summary": optuna_summary,
        "artifact_paths": artifact_paths,
        "train_report": train_report,
        "test_report": test_report,
        "model_bundle": model_bundle,
        "coefficient_inference": model_bundle["coefficient_inference"],
        "identifiable_coefficient_uncertainty": model_bundle["identifiable_coefficient_uncertainty"],
        "effective_coefficient_uncertainty": model_bundle["effective_coefficient_uncertainty"],
        "dataset_splits": dataset_splits,
    }


__all__ = [
    "MODEL_NAME",
    "build_icsor_design_frame",
    "load_icsor_params",
    "predict_icsor_model",
    "run_icsor_pipeline",
    "train_icsor_model",
]

