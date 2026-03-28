"""COBRE (Constrained Bilinear Regression) solved by projected ordinary least squares in measured-output space."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.utils.optuna import create_progress_bar
from src.utils.process import (
    DatasetSplit,
    ScalingBundle,
    TrainTestDatasetSplits,
    build_measured_supervised_dataset,
    build_projection_operator,
    fit_scalers,
    project_to_mass_balance,
    transform_dataset_split,
)
from src.utils.simulation import load_model_params
from src.utils.test import evaluate_prediction_bundle
from src.utils.train import (
    get_training_device,
    load_model_bundle,
    persist_training_artifacts,
    resolve_model_hyperparameters,
    resolve_training_objective_label,
    resolve_torch_runtime_options,
    serialize_report_frames,
    transform_feature_frame,
)


MODEL_NAME = "cobre"
VALID_OLS_BACKENDS = {"auto", "numpy_lstsq", "directml_normal_equations"}


def load_cobre_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the COBRE model."""

    return load_model_params(MODEL_NAME, repo_root)


def _resolve_training_options(
    training_options: Mapping[str, Any] | None,
    *,
    objective_name: str,
) -> dict[str, Any]:
    options = dict(training_options or {})
    options.setdefault("show_progress", True)
    options.setdefault("progress_description", "Training COBRE")
    options.setdefault("objective_name", objective_name)
    return options


def _resolve_runtime_options(runtime_options: Mapping[str, Any] | None) -> dict[str, Any]:
    options = dict(runtime_options or {})
    options.setdefault("prefer_directml", True)
    return options


def _resolve_ols_backend(model_hyperparameters: Mapping[str, Any]) -> str:
    backend_name = str(model_hyperparameters.get("ols_backend", "auto")).strip().lower()
    if backend_name not in VALID_OLS_BACKENDS:
        valid_values = ", ".join(sorted(VALID_OLS_BACKENDS))
        raise ValueError(f"COBRE requires ols_backend to be one of: {valid_values}.")
    return backend_name


def _validate_scaling_configuration(hyperparameters: Mapping[str, Any]) -> None:
    if bool(hyperparameters.get("scale_features", False)):
        raise ValueError("COBRE requires scale_features=False so C_in remains in physical coordinates.")
    if bool(hyperparameters.get("scale_targets", False)):
        raise ValueError("COBRE requires scale_targets=False because the projected OLS target lives in measured-output space.")


def _predict_from_bundle(
    feature_frame: pd.DataFrame,
    constraint_reference: pd.DataFrame,
    model_bundle: Mapping[str, Any],
    *,
    scaling_bundle: ScalingBundle,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    transformed_features = transform_feature_frame(feature_frame, scaling_bundle)
    constraint_columns = list(model_bundle["constraint_columns"])
    selected_constraint_reference = constraint_reference.loc[:, constraint_columns].copy()
    design_frame, _ = build_cobre_design_frame(
        transformed_features,
        constraint_columns,
        include_bias_term=bool(model_bundle["design_schema"]["include_bias_term"]),
    )
    raw_predictions = _predict_with_parameter_matrix(design_frame, model_bundle["raw_parameter_matrix"])
    projected_predictions = _predict_with_parameter_matrix(design_frame, model_bundle["effective_parameter_matrix"])
    return raw_predictions, projected_predictions, selected_constraint_reference


def _compute_projection_matrices(A_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projection_matrix = build_projection_operator(np.asarray(A_matrix, dtype=float))
    projection_matrix = 0.5 * (projection_matrix + projection_matrix.T)
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
        raise ValueError("COBRE requires at least one operational feature column.")
    if influent_columns != expected_influent_columns:
        raise ValueError(
            "COBRE requires influent measured columns to match constraint_reference columns in order."
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


def build_cobre_design_frame(
    feature_frame: pd.DataFrame,
    constraint_columns: list[str],
    *,
    include_bias_term: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the partitioned second-order design matrix used by COBRE."""

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


def _build_effective_parameter_matrix(
    raw_parameter_matrix: np.ndarray,
    design_schema: Mapping[str, Any],
    projection_matrix: np.ndarray,
) -> np.ndarray:
    effective_parameter_matrix = np.asarray(raw_parameter_matrix, dtype=float).copy()
    effective_parameter_matrix[_get_block_slice(design_schema, "linear_influent"), :] += np.asarray(
        projection_matrix,
        dtype=float,
    ).T
    return effective_parameter_matrix


def _predict_with_parameter_matrix(
    design_frame: pd.DataFrame,
    parameter_matrix: np.ndarray,
) -> np.ndarray:
    return design_frame.to_numpy(dtype=float) @ np.asarray(parameter_matrix, dtype=float)


def _summarize_linear_solve(
    design_matrix: np.ndarray,
    target_matrix: np.ndarray,
    parameter_matrix: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    singular_values = np.linalg.svd(design_matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design_matrix))
    residual_matrix = target_matrix - (design_matrix @ parameter_matrix)
    if design_matrix.shape[0] > design_matrix.shape[1] and rank == design_matrix.shape[1]:
        residuals = np.sum(residual_matrix**2, axis=0)
    else:
        residuals = np.array([], dtype=float)
    return np.asarray(residuals, dtype=float), rank, np.asarray(singular_values, dtype=float)


def _solve_with_numpy_lstsq(
    design_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    rcond_value: Any,
) -> dict[str, Any]:
    parameter_matrix, residuals, rank, singular_values = np.linalg.lstsq(
        design_matrix,
        target_matrix,
        rcond=None if rcond_value is None else float(rcond_value),
    )
    return {
        "parameter_matrix": np.asarray(parameter_matrix, dtype=float),
        "residuals": np.asarray(residuals, dtype=float),
        "rank": int(rank),
        "singular_values": np.asarray(singular_values, dtype=float),
        "backend_used": "numpy_lstsq",
        "device_label": "cpu",
        "matrix_multiplication_dtype": None,
    }


def _compute_ols_cross_products_with_torch(
    design_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    device: Any,
    device_label: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    matrix_dtype = torch.float32 if device_label == "directml" else torch.float64
    design_tensor = torch.as_tensor(design_matrix, dtype=matrix_dtype, device=device)
    target_tensor = torch.as_tensor(target_matrix, dtype=matrix_dtype, device=device)
    gram_matrix = (design_tensor.T @ design_tensor).detach().cpu().numpy().astype(float, copy=False)
    rhs_matrix = (design_tensor.T @ target_tensor).detach().cpu().numpy().astype(float, copy=False)
    return gram_matrix, rhs_matrix, str(matrix_dtype).replace("torch.", "")


def _solve_with_directml_normal_equations(
    design_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    device: Any,
    device_label: str,
    rcond_value: Any,
) -> dict[str, Any]:
    gram_matrix, rhs_matrix, multiplication_dtype = _compute_ols_cross_products_with_torch(
        design_matrix,
        target_matrix,
        device=device,
        device_label=device_label,
    )
    parameter_matrix, _, _, _ = np.linalg.lstsq(
        gram_matrix,
        rhs_matrix,
        rcond=None if rcond_value is None else float(rcond_value),
    )
    residuals, rank, singular_values = _summarize_linear_solve(
        design_matrix,
        target_matrix,
        np.asarray(parameter_matrix, dtype=float),
    )
    return {
        "parameter_matrix": np.asarray(parameter_matrix, dtype=float),
        "residuals": residuals,
        "rank": rank,
        "singular_values": singular_values,
        "backend_used": "directml_normal_equations",
        "device_label": device_label,
        "matrix_multiplication_dtype": multiplication_dtype,
    }


def _solve_projected_ols(
    design_matrix: np.ndarray,
    target_matrix: np.ndarray,
    model_hyperparameters: Mapping[str, Any],
    runtime_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    requested_backend = _resolve_ols_backend(model_hyperparameters)
    resolved_runtime_options = _resolve_runtime_options(runtime_options)
    rcond_value = model_hyperparameters.get("lstsq_rcond")
    fallback_reason: str | None = None
    resolved_device_label = "cpu"

    if requested_backend == "numpy_lstsq":
        solve_result = _solve_with_numpy_lstsq(design_matrix, target_matrix, rcond_value=rcond_value)
    else:
        if not bool(resolved_runtime_options["prefer_directml"]):
            fallback_reason = "DirectML preference is disabled in runtime options."
            solve_result = _solve_with_numpy_lstsq(design_matrix, target_matrix, rcond_value=rcond_value)
        else:
            device, resolved_device_label = get_training_device(prefer_directml=True)
            if resolved_device_label != "directml":
                fallback_reason = (
                    f"DirectML device unavailable; resolved device '{resolved_device_label}' instead."
                )
                solve_result = _solve_with_numpy_lstsq(design_matrix, target_matrix, rcond_value=rcond_value)
            else:
                try:
                    solve_result = _solve_with_directml_normal_equations(
                        design_matrix,
                        target_matrix,
                        device=device,
                        device_label=resolved_device_label,
                        rcond_value=rcond_value,
                    )
                except Exception as error:
                    fallback_reason = f"DirectML backend failed during matrix multiplication: {error}"
                    solve_result = _solve_with_numpy_lstsq(design_matrix, target_matrix, rcond_value=rcond_value)

    solve_result["ols_metadata"] = {
        "requested_backend": requested_backend,
        "backend_used": str(solve_result["backend_used"]),
        "device_label": resolved_device_label if requested_backend != "numpy_lstsq" else "cpu",
        "fallback_reason": fallback_reason,
        "prefer_directml": bool(resolved_runtime_options["prefer_directml"]),
        "matrix_multiplication_dtype": solve_result["matrix_multiplication_dtype"],
    }
    return solve_result


def _build_model_bundle(
    training_result: Mapping[str, Any],
    scaling_bundle: ScalingBundle,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    constraint_columns: list[str],
    A_matrix: np.ndarray,
    model_hyperparameters: Mapping[str, Any],
    training_options: Mapping[str, Any],
    runtime_options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "constraint_columns": constraint_columns,
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "projection_matrix": np.asarray(training_result["projection_matrix"], dtype=float),
        "projection_complement": np.asarray(training_result["projection_complement"], dtype=float),
        "design_schema": dict(training_result["design_schema"]),
        "raw_parameter_matrix": np.asarray(training_result["raw_parameter_matrix"], dtype=float),
        "effective_parameter_matrix": np.asarray(training_result["effective_parameter_matrix"], dtype=float),
        "raw_coefficients": dict(training_result["raw_coefficients"]),
        "effective_coefficients": dict(training_result["effective_coefficients"]),
        "ols_metadata": dict(training_result["ols_metadata"]),
        "scaling_bundle": scaling_bundle,
        "model_hyperparameters": dict(model_hyperparameters),
        "training_options": dict(training_options),
        "runtime_options": dict(runtime_options),
    }


def train_cobre_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, Any],
    *,
    A_matrix: np.ndarray,
    training_options: Mapping[str, Any] | None = None,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the projected OLS COBRE estimator on a prepared dataset."""

    _validate_scaling_configuration(model_hyperparameters)

    feature_frame = pd.DataFrame(training_dataset["features"])
    target_frame = pd.DataFrame(training_dataset["targets"])
    constraint_frame = pd.DataFrame(training_dataset["constraint_reference"])

    objective_label = resolve_training_objective_label(model_hyperparameters, default="projected_ols")
    options = _resolve_training_options(training_options, objective_name=objective_label)
    projection_matrix, projection_complement = _compute_projection_matrices(np.asarray(A_matrix, dtype=float))

    design_frame, design_schema = build_cobre_design_frame(
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
        projected_target_values = target_frame.to_numpy(dtype=float) @ projection_complement.T
        design_matrix = design_frame.to_numpy(dtype=float)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="solve", objective=str(options["objective_name"]))
        solve_result = _solve_projected_ols(
            design_matrix,
            projected_target_values,
            model_hyperparameters,
            runtime_options,
        )
        raw_parameter_matrix = np.asarray(solve_result["parameter_matrix"], dtype=float)
        residuals = np.asarray(solve_result["residuals"], dtype=float)
        rank = int(solve_result["rank"])
        singular_values = np.asarray(solve_result["singular_values"], dtype=float)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="assemble", objective=str(options["objective_name"]))
        effective_parameter_matrix = _build_effective_parameter_matrix(
            raw_parameter_matrix,
            design_schema,
            projection_matrix,
        )
        raw_coefficients = _unpack_parameter_blocks(raw_parameter_matrix, design_schema)
        effective_coefficients = _unpack_parameter_blocks(effective_parameter_matrix, design_schema)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="predict", objective=str(options["objective_name"]))
        training_raw_predictions = _predict_with_parameter_matrix(design_frame, raw_parameter_matrix)
        training_projected_predictions = _predict_with_parameter_matrix(design_frame, effective_parameter_matrix)
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
        "raw_parameter_matrix": raw_parameter_matrix,
        "effective_parameter_matrix": effective_parameter_matrix,
        "raw_coefficients": raw_coefficients,
        "effective_coefficients": effective_coefficients,
        "training_raw_predictions": pd.DataFrame(
            training_raw_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_projected_predictions": pd.DataFrame(
            training_projected_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_mse": train_mse,
        "rank": int(rank),
        "singular_values": np.asarray(singular_values, dtype=float),
        "residuals": np.asarray(residuals, dtype=float),
        "ols_metadata": dict(solve_result["ols_metadata"]),
    }


def predict_cobre_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted COBRE bundle and generate aligned predictions."""

    model_bundle = load_model_bundle(model_path)
    scaling_bundle: ScalingBundle = model_bundle["scaling_bundle"]

    if isinstance(test_dataset, pd.DataFrame):
        if metadata is None or composition_matrix is None:
            raise ValueError("metadata and composition_matrix are required when predicting from a raw dataset.")
        measured_dataset = build_measured_supervised_dataset(
            test_dataset,
            dict(metadata),
            np.asarray(composition_matrix, dtype=float),
        )
        feature_frame = measured_dataset.features
        constraint_reference = measured_dataset.constraint_reference
    else:
        feature_frame = pd.DataFrame(test_dataset["features"], columns=scaling_bundle.feature_columns)
        constraint_reference = pd.DataFrame(
            test_dataset["constraint_reference"],
            columns=model_bundle["constraint_columns"],
        )

    raw_predictions, projected_predictions, selected_constraint_reference = _predict_from_bundle(
        feature_frame,
        constraint_reference,
        model_bundle,
        scaling_bundle=scaling_bundle,
    )

    return {
        "raw_predictions": pd.DataFrame(raw_predictions, index=feature_frame.index, columns=model_bundle["target_columns"]),
        "projected_predictions": pd.DataFrame(
            projected_predictions,
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "constraint_reference": selected_constraint_reference,
    }


def run_cobre_pipeline(
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    A_matrix: np.ndarray,
    *,
    repo_root: str | Path | None = None,
    model_params: Mapping[str, Any] | None = None,
    model_hyperparameters: Mapping[str, Any] | None = None,
    optuna_summary: Mapping[str, Any] | None = None,
    show_progress: bool = True,
    persist_artifacts: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and optionally persist one COBRE bundle."""

    params = dict(model_params) if model_params is not None else load_cobre_params(repo_root)
    split_params = dict(params["hyperparameters"])
    selected_hyperparameters = resolve_model_hyperparameters(params, model_hyperparameters)
    runtime_options = resolve_torch_runtime_options(params)
    _validate_scaling_configuration({**split_params, **selected_hyperparameters})
    objective_label = resolve_training_objective_label(selected_hyperparameters, default="projected_ols")

    progress_bar = create_progress_bar(
        total=5,
        desc="Training COBRE",
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
        training_result = train_cobre_model(
            {
                "features": scaled_training_split.features,
                "targets": scaled_training_split.targets,
                "constraint_reference": scaled_training_split.constraint_reference,
            },
            selected_hyperparameters,
            A_matrix=np.asarray(A_matrix, dtype=float),
            training_options={
                "show_progress": False,
                "progress_description": "Training COBRE",
                "objective_name": objective_label,
            },
            runtime_options=runtime_options,
        )
        progress_bar.update(1)

        model_bundle = _build_model_bundle(
            training_result,
            scaling_bundle,
            feature_columns=list(training_split.features.columns),
            target_columns=list(training_split.targets.columns),
            constraint_columns=list(training_split.constraint_reference.columns),
            A_matrix=np.asarray(A_matrix, dtype=float),
            model_hyperparameters=selected_hyperparameters,
            training_options={
                "objective_name": objective_label,
                "show_progress": show_progress,
            },
            runtime_options=runtime_options,
        )

        progress_bar.set_postfix(stage="evaluate_train", objective=objective_label)
        train_raw, train_projected, train_constraint_reference = _predict_from_bundle(
            scaled_training_split.features,
            scaled_training_split.constraint_reference,
            model_bundle,
            scaling_bundle=scaling_bundle,
        )
        progress_bar.update(1)

        progress_bar.set_postfix(stage="evaluate_test", objective=objective_label)
        test_raw, test_projected, test_constraint_reference = _predict_from_bundle(
            scaled_test_split.features,
            scaled_test_split.constraint_reference,
            model_bundle,
            scaling_bundle=scaling_bundle,
        )
        train_report = evaluate_prediction_bundle(
            training_split.targets.to_numpy(dtype=float),
            train_raw,
            train_projected,
            train_constraint_reference.to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
            training_split.targets.columns,
            index=training_split.targets.index,
        )
        test_report = evaluate_prediction_bundle(
            test_split.targets.to_numpy(dtype=float),
            test_raw,
            test_projected,
            test_constraint_reference.to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
            test_split.targets.columns,
            index=test_split.targets.index,
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
            metrics_summary = metrics_payload if bool(artifact_options.get("persist_metrics", True)) else None
            artifact_paths = persist_training_artifacts(
                MODEL_NAME,
                model_bundle,
                metrics_summary=metrics_summary,
                optuna_summary=None,
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
        "dataset_splits": dataset_splits,
    }


__all__ = [
    "MODEL_NAME",
    "build_cobre_design_frame",
    "load_cobre_params",
    "predict_cobre_model",
    "project_to_mass_balance",
    "run_cobre_pipeline",
    "train_cobre_model",
]
