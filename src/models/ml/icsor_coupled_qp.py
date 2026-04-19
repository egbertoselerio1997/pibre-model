"""ICSOR coupled-QP model with OSQP-based block-coordinate training and deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import osqp
import pandas as pd
from scipy import sparse as sp

from src.models.ml.icsor import build_icsor_design_frame
from src.utils.optuna import create_progress_bar
from src.utils.process import (
    DatasetSplit,
    ScalingBundle,
    TrainTestDatasetSplits,
    build_icsor_supervised_dataset,
    fit_scalers,
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
    serialize_report_frames,
    transform_feature_frame,
)


MODEL_NAME = "icsor_coupled_qp"

DEFAULT_INCLUDE_BIAS_TERM = True
DEFAULT_LAMBDA_INV = 1.0
DEFAULT_LAMBDA_SYS = 1.0
DEFAULT_LAMBDA_B = 1e-4
DEFAULT_LAMBDA_GAMMA = 1e-4
DEFAULT_LAMBDA_PRED = 1.0
DEFAULT_GAMMA_ABS_BOUND = 0.5
DEFAULT_MAX_OUTER_ITERATIONS = 50
DEFAULT_N_RESTARTS = 3
DEFAULT_OBJECTIVE_TOLERANCE = 1e-7
DEFAULT_PARAMETER_TOLERANCE = 1e-6
DEFAULT_CONDITIONING_MAX = 1e8
DEFAULT_OSQP_EPS_ABS = 1e-6
DEFAULT_OSQP_EPS_REL = 1e-6
DEFAULT_OSQP_MAX_ITER = 20000
DEFAULT_OSQP_POLISH = True
DEFAULT_OSQP_VERBOSE = False
DEFAULT_WARM_START = True
DEFAULT_NONNEGATIVITY_TOLERANCE = 1e-10
DEFAULT_CONSTRAINT_TOLERANCE = 1e-8


def load_icsor_coupled_qp_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load configured parameters for the coupled-QP ICSOR model."""

    return load_model_params(MODEL_NAME, repo_root)


def _validate_scaling_configuration(hyperparameters: Mapping[str, Any]) -> None:
    if bool(hyperparameters.get("scale_features", False)):
        raise ValueError("icsor_coupled_qp requires scale_features=False so fractional states remain physical.")
    if bool(hyperparameters.get("scale_targets", False)):
        raise ValueError("icsor_coupled_qp requires scale_targets=False for physical fractional targets.")


def _resolve_training_options(
    training_options: Mapping[str, Any] | None,
    *,
    objective_name: str,
) -> dict[str, Any]:
    options = dict(training_options or {})
    options.setdefault("show_progress", True)
    options.setdefault("progress_description", "Training icsor_coupled_qp")
    options.setdefault("objective_name", objective_name)
    return options


def _resolve_coupled_qp_settings(model_hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    settings = {
        "include_bias_term": bool(model_hyperparameters.get("include_bias_term", DEFAULT_INCLUDE_BIAS_TERM)),
        "lambda_inv": float(model_hyperparameters.get("lambda_inv", DEFAULT_LAMBDA_INV)),
        "lambda_sys": float(model_hyperparameters.get("lambda_sys", DEFAULT_LAMBDA_SYS)),
        "lambda_B": float(model_hyperparameters.get("lambda_B", DEFAULT_LAMBDA_B)),
        "lambda_gamma": float(model_hyperparameters.get("lambda_gamma", DEFAULT_LAMBDA_GAMMA)),
        "lambda_pred": float(model_hyperparameters.get("lambda_pred", DEFAULT_LAMBDA_PRED)),
        "gamma_abs_bound": float(model_hyperparameters.get("gamma_abs_bound", DEFAULT_GAMMA_ABS_BOUND)),
        "max_outer_iterations": int(model_hyperparameters.get("max_outer_iterations", DEFAULT_MAX_OUTER_ITERATIONS)),
        "n_restarts": int(model_hyperparameters.get("n_restarts", DEFAULT_N_RESTARTS)),
        "objective_tolerance": float(model_hyperparameters.get("objective_tolerance", DEFAULT_OBJECTIVE_TOLERANCE)),
        "parameter_tolerance": float(model_hyperparameters.get("parameter_tolerance", DEFAULT_PARAMETER_TOLERANCE)),
        "conditioning_max": float(model_hyperparameters.get("conditioning_max", DEFAULT_CONDITIONING_MAX)),
        "osqp_eps_abs": float(model_hyperparameters.get("osqp_eps_abs", DEFAULT_OSQP_EPS_ABS)),
        "osqp_eps_rel": float(model_hyperparameters.get("osqp_eps_rel", DEFAULT_OSQP_EPS_REL)),
        "osqp_max_iter": int(model_hyperparameters.get("osqp_max_iter", DEFAULT_OSQP_MAX_ITER)),
        "osqp_polish": bool(model_hyperparameters.get("osqp_polish", DEFAULT_OSQP_POLISH)),
        "osqp_verbose": bool(model_hyperparameters.get("osqp_verbose", DEFAULT_OSQP_VERBOSE)),
        "warm_start": bool(model_hyperparameters.get("warm_start", DEFAULT_WARM_START)),
        "nonnegativity_tolerance": float(
            model_hyperparameters.get("nonnegativity_tolerance", DEFAULT_NONNEGATIVITY_TOLERANCE)
        ),
        "constraint_tolerance": float(model_hyperparameters.get("constraint_tolerance", DEFAULT_CONSTRAINT_TOLERANCE)),
    }

    if settings["lambda_inv"] < 0.0:
        raise ValueError("icsor_coupled_qp lambda_inv must be nonnegative.")
    if settings["lambda_sys"] <= 0.0:
        raise ValueError("icsor_coupled_qp lambda_sys must be positive.")
    if settings["lambda_B"] < 0.0:
        raise ValueError("icsor_coupled_qp lambda_B must be nonnegative.")
    if settings["lambda_gamma"] < 0.0:
        raise ValueError("icsor_coupled_qp lambda_gamma must be nonnegative.")
    if settings["lambda_pred"] < 0.0:
        raise ValueError("icsor_coupled_qp lambda_pred must be nonnegative.")
    if settings["gamma_abs_bound"] <= 0.0:
        raise ValueError("icsor_coupled_qp gamma_abs_bound must be positive.")
    if settings["max_outer_iterations"] < 1:
        raise ValueError("icsor_coupled_qp max_outer_iterations must be at least 1.")
    if settings["n_restarts"] < 1:
        raise ValueError("icsor_coupled_qp n_restarts must be at least 1.")
    if settings["objective_tolerance"] <= 0.0:
        raise ValueError("icsor_coupled_qp objective_tolerance must be positive.")
    if settings["parameter_tolerance"] <= 0.0:
        raise ValueError("icsor_coupled_qp parameter_tolerance must be positive.")
    if settings["conditioning_max"] <= 1.0:
        raise ValueError("icsor_coupled_qp conditioning_max must exceed 1.")
    if settings["osqp_eps_abs"] <= 0.0:
        raise ValueError("icsor_coupled_qp osqp_eps_abs must be positive.")
    if settings["osqp_eps_rel"] <= 0.0:
        raise ValueError("icsor_coupled_qp osqp_eps_rel must be positive.")
    if settings["osqp_max_iter"] < 1:
        raise ValueError("icsor_coupled_qp osqp_max_iter must be at least 1.")
    if settings["nonnegativity_tolerance"] <= 0.0:
        raise ValueError("icsor_coupled_qp nonnegativity_tolerance must be positive.")
    if settings["constraint_tolerance"] <= 0.0:
        raise ValueError("icsor_coupled_qp constraint_tolerance must be positive.")

    return settings


def _validate_composition_shape(
    composition_matrix: np.ndarray,
    *,
    constraint_columns: list[str],
) -> np.ndarray:
    composition_array = np.asarray(composition_matrix, dtype=float)
    if composition_array.ndim != 2:
        raise ValueError("icsor_coupled_qp requires composition_matrix to be two-dimensional.")
    if composition_array.shape[1] != len(constraint_columns):
        raise ValueError(
            "icsor_coupled_qp requires composition_matrix column count to match the ASM constraint dimension."
        )
    return composition_array


def _make_osqp_solver(
    P_matrix: np.ndarray,
    q_vector: np.ndarray,
    A_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    settings: Mapping[str, Any],
) -> osqp.OSQP:
    solver = osqp.OSQP()
    solver.setup(
        P=sp.csc_matrix(0.5 * (P_matrix + P_matrix.T)),
        q=np.asarray(q_vector, dtype=float),
        A=sp.csc_matrix(A_matrix),
        l=np.asarray(lower_bounds, dtype=float),
        u=np.asarray(upper_bounds, dtype=float),
        eps_abs=float(settings["osqp_eps_abs"]),
        eps_rel=float(settings["osqp_eps_rel"]),
        max_iter=int(settings["osqp_max_iter"]),
        polishing=bool(settings["osqp_polish"]),
        verbose=bool(settings["osqp_verbose"]),
        warm_starting=bool(settings["warm_start"]),
    )
    return solver


def _is_osqp_solved(status: str | None) -> bool:
    resolved_status = str(status or "").strip().lower()
    return resolved_status in {"solved", "solved inaccurate"}


def _solve_b_update(
    design_matrix: np.ndarray,
    fitted_predictions: np.ndarray,
    gamma_matrix: np.ndarray,
    settings: Mapping[str, Any],
) -> np.ndarray:
    lambda_sys = float(settings["lambda_sys"])
    lambda_B = float(settings["lambda_B"])

    response_matrix = fitted_predictions @ (np.eye(gamma_matrix.shape[0], dtype=float) - gamma_matrix).T
    n_samples, n_features = design_matrix.shape
    if n_features > n_samples:
        # Use the dual system when features outnumber samples to avoid large dense solves.
        lhs_matrix = lambda_sys * (design_matrix @ design_matrix.T)
        if lambda_B > 0.0:
            lhs_matrix = lhs_matrix + lambda_B * np.eye(lhs_matrix.shape[0], dtype=float)
        try:
            dual_solution = np.linalg.solve(lhs_matrix, response_matrix)
        except np.linalg.LinAlgError:
            dual_solution = np.linalg.pinv(lhs_matrix, rcond=1e-10) @ response_matrix
        solved_matrix = design_matrix.T @ dual_solution
    else:
        lhs_matrix = lambda_sys * (design_matrix.T @ design_matrix)
        if lambda_B > 0.0:
            lhs_matrix = lhs_matrix + lambda_B * np.eye(lhs_matrix.shape[0], dtype=float)
        rhs_matrix = lambda_sys * (design_matrix.T @ response_matrix)
        try:
            solved_matrix = np.linalg.solve(lhs_matrix, rhs_matrix)
        except np.linalg.LinAlgError:
            solved_matrix = np.linalg.pinv(lhs_matrix, rcond=1e-10) @ rhs_matrix

    return solved_matrix.T


def _enforce_gamma_conditioning(
    gamma_matrix: np.ndarray,
    *,
    conditioning_max: float,
) -> tuple[np.ndarray, float, float]:
    identity = np.eye(gamma_matrix.shape[0], dtype=float)
    candidate = np.asarray(gamma_matrix, dtype=float)
    try:
        condition_value = float(np.linalg.cond(identity - candidate))
    except np.linalg.LinAlgError:
        condition_value = float("inf")

    if np.isfinite(condition_value) and condition_value <= conditioning_max:
        return candidate, condition_value, 1.0

    for attempt_index in range(1, 25):
        shrink_factor = 0.5**attempt_index
        shrunk_candidate = shrink_factor * candidate
        try:
            shrunk_condition = float(np.linalg.cond(identity - shrunk_candidate))
        except np.linalg.LinAlgError:
            shrunk_condition = float("inf")

        if np.isfinite(shrunk_condition) and shrunk_condition <= conditioning_max:
            return shrunk_candidate, shrunk_condition, shrink_factor

    fallback = np.zeros_like(candidate)
    return fallback, 1.0, 0.0


def _solve_gamma_update(
    fitted_predictions: np.ndarray,
    driver_matrix: np.ndarray,
    settings: Mapping[str, Any],
    *,
    initial_gamma: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_outputs = fitted_predictions.shape[1]
    gamma_abs_bound = float(settings["gamma_abs_bound"])
    lambda_sys = float(settings["lambda_sys"])
    lambda_gamma = float(settings["lambda_gamma"])

    ctc_matrix = fitted_predictions.T @ fitted_predictions
    quadratic_matrix = 2.0 * (lambda_sys * ctc_matrix + lambda_gamma * np.eye(n_outputs, dtype=float))
    identity_constraints = np.eye(n_outputs, dtype=float)

    updated_gamma = np.zeros((n_outputs, n_outputs), dtype=float)
    status_values: list[str] = []
    iteration_values: list[int] = []

    residual_target = fitted_predictions - driver_matrix
    warm_start_gamma = np.asarray(initial_gamma, dtype=float) if initial_gamma is not None else None

    for target_index in range(n_outputs):
        linear_vector = -2.0 * lambda_sys * (fitted_predictions.T @ residual_target[:, target_index])
        lower_bounds = np.full(n_outputs, -gamma_abs_bound, dtype=float)
        upper_bounds = np.full(n_outputs, gamma_abs_bound, dtype=float)
        lower_bounds[target_index] = 0.0
        upper_bounds[target_index] = 0.0

        solver = _make_osqp_solver(
            quadratic_matrix,
            linear_vector,
            identity_constraints,
            lower_bounds,
            upper_bounds,
            settings,
        )
        if warm_start_gamma is not None:
            solver.warm_start(x=np.asarray(warm_start_gamma[target_index], dtype=float))

        result = solver.solve()
        status_text = str(result.info.status)
        status_values.append(status_text)
        iteration_values.append(int(result.info.iter))

        if _is_osqp_solved(status_text) and result.x is not None:
            updated_column = np.asarray(result.x, dtype=float)
        else:
            updated_column = np.zeros(n_outputs, dtype=float)

        updated_column[target_index] = 0.0
        updated_gamma[target_index, :] = np.clip(updated_column, -gamma_abs_bound, gamma_abs_bound)

    np.fill_diagonal(updated_gamma, 0.0)
    return updated_gamma, {
        "status_counts": pd.Series(status_values).value_counts(dropna=False).to_dict(),
        "mean_iterations": float(np.mean(iteration_values) if iteration_values else 0.0),
        "max_iterations": int(max(iteration_values) if iteration_values else 0),
    }


def _solve_chat_update(
    target_matrix: np.ndarray,
    influent_matrix: np.ndarray,
    invariant_matrix: np.ndarray,
    coupled_matrix: np.ndarray,
    driver_matrix: np.ndarray,
    settings: Mapping[str, Any],
    *,
    warm_start_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    lambda_inv = float(settings["lambda_inv"])
    lambda_sys = float(settings["lambda_sys"])

    n_samples, n_outputs = target_matrix.shape
    at_a_matrix = invariant_matrix.T @ invariant_matrix
    rt_r_matrix = coupled_matrix.T @ coupled_matrix
    quadratic_matrix = 2.0 * (
        np.eye(n_outputs, dtype=float)
        + lambda_inv * at_a_matrix
        + lambda_sys * rt_r_matrix
    )

    constraint_matrix = np.eye(n_outputs, dtype=float)
    lower_bounds = np.zeros(n_outputs, dtype=float)
    upper_bounds = np.full(n_outputs, np.inf, dtype=float)

    solver = _make_osqp_solver(
        quadratic_matrix,
        np.zeros(n_outputs, dtype=float),
        constraint_matrix,
        lower_bounds,
        upper_bounds,
        settings,
    )

    fitted_predictions = np.zeros((n_samples, n_outputs), dtype=float)
    status_values: list[str] = []
    iteration_values: list[int] = []

    at_a_influents = influent_matrix @ at_a_matrix.T
    rt_driver_matrix = driver_matrix @ coupled_matrix

    for sample_index in range(n_samples):
        linear_vector = -2.0 * (
            target_matrix[sample_index]
            + lambda_inv * at_a_influents[sample_index]
            + lambda_sys * rt_driver_matrix[sample_index]
        )
        solver.update(q=np.asarray(linear_vector, dtype=float))

        if bool(settings["warm_start"]):
            if warm_start_matrix is not None:
                initial_guess = np.asarray(warm_start_matrix[sample_index], dtype=float)
            else:
                initial_guess = np.maximum(target_matrix[sample_index], 0.0)
            solver.warm_start(x=initial_guess)

        result = solver.solve()
        status_text = str(result.info.status)
        status_values.append(status_text)
        iteration_values.append(int(result.info.iter))

        if _is_osqp_solved(status_text) and result.x is not None:
            fitted_predictions[sample_index] = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        else:
            fitted_predictions[sample_index] = np.maximum(target_matrix[sample_index], 0.0)

    return fitted_predictions, {
        "status_counts": pd.Series(status_values).value_counts(dropna=False).to_dict(),
        "mean_iterations": float(np.mean(iteration_values) if iteration_values else 0.0),
        "max_iterations": int(max(iteration_values) if iteration_values else 0),
    }


def _compute_training_objective(
    target_matrix: np.ndarray,
    influent_matrix: np.ndarray,
    invariant_matrix: np.ndarray,
    design_matrix: np.ndarray,
    b_matrix: np.ndarray,
    gamma_matrix: np.ndarray,
    fitted_predictions: np.ndarray,
    settings: Mapping[str, Any],
) -> float:
    lambda_inv = float(settings["lambda_inv"])
    lambda_sys = float(settings["lambda_sys"])
    lambda_B = float(settings["lambda_B"])
    lambda_gamma = float(settings["lambda_gamma"])

    coupled_matrix = np.eye(gamma_matrix.shape[0], dtype=float) - gamma_matrix
    driver_matrix = design_matrix @ b_matrix.T

    fit_term = float(np.sum((target_matrix - fitted_predictions) ** 2))
    invariant_term = float(
        lambda_inv * np.sum(((fitted_predictions - influent_matrix) @ invariant_matrix.T) ** 2)
    )
    system_term = float(
        lambda_sys * np.sum((fitted_predictions @ coupled_matrix.T - driver_matrix) ** 2)
    )
    b_regularization = float(lambda_B * np.sum(b_matrix**2))
    gamma_regularization = float(lambda_gamma * np.sum(gamma_matrix**2))
    return fit_term + invariant_term + system_term + b_regularization + gamma_regularization


def _run_coupled_qp_restart(
    design_matrix: np.ndarray,
    target_matrix: np.ndarray,
    influent_matrix: np.ndarray,
    invariant_matrix: np.ndarray,
    settings: Mapping[str, Any],
    *,
    initial_gamma: np.ndarray,
) -> dict[str, Any]:
    gamma_matrix, conditioning_value, shrink_factor = _enforce_gamma_conditioning(
        initial_gamma,
        conditioning_max=float(settings["conditioning_max"]),
    )
    fitted_predictions = np.maximum(target_matrix, 0.0)
    b_matrix = _solve_b_update(design_matrix, fitted_predictions, gamma_matrix, settings)

    objective_history: list[float] = []
    gamma_update_history: list[dict[str, Any]] = []
    chat_update_history: list[dict[str, Any]] = []

    previous_objective = _compute_training_objective(
        target_matrix,
        influent_matrix,
        invariant_matrix,
        design_matrix,
        b_matrix,
        gamma_matrix,
        fitted_predictions,
        settings,
    )
    objective_history.append(previous_objective)

    for _ in range(int(settings["max_outer_iterations"])):
        b_updated = _solve_b_update(design_matrix, fitted_predictions, gamma_matrix, settings)
        driver_matrix = design_matrix @ b_updated.T

        gamma_candidate, gamma_update_metadata = _solve_gamma_update(
            fitted_predictions,
            driver_matrix,
            settings,
            initial_gamma=gamma_matrix,
        )
        gamma_updated, conditioning_value, shrink_factor = _enforce_gamma_conditioning(
            gamma_candidate,
            conditioning_max=float(settings["conditioning_max"]),
        )
        coupled_matrix = np.eye(gamma_updated.shape[0], dtype=float) - gamma_updated

        fitted_updated, chat_update_metadata = _solve_chat_update(
            target_matrix,
            influent_matrix,
            invariant_matrix,
            coupled_matrix,
            driver_matrix,
            settings,
            warm_start_matrix=fitted_predictions,
        )

        updated_objective = _compute_training_objective(
            target_matrix,
            influent_matrix,
            invariant_matrix,
            design_matrix,
            b_updated,
            gamma_updated,
            fitted_updated,
            settings,
        )
        objective_history.append(updated_objective)
        gamma_update_history.append(dict(gamma_update_metadata))
        chat_update_history.append(dict(chat_update_metadata))

        b_delta = float(np.linalg.norm(b_updated - b_matrix) / (1.0 + np.linalg.norm(b_matrix)))
        gamma_delta = float(np.linalg.norm(gamma_updated - gamma_matrix) / (1.0 + np.linalg.norm(gamma_matrix)))
        c_delta = float(
            np.linalg.norm(fitted_updated - fitted_predictions) / (1.0 + np.linalg.norm(fitted_predictions))
        )
        parameter_delta = max(b_delta, gamma_delta, c_delta)
        objective_delta = float(abs(updated_objective - previous_objective) / (1.0 + abs(previous_objective)))

        b_matrix = b_updated
        gamma_matrix = gamma_updated
        fitted_predictions = fitted_updated
        previous_objective = updated_objective

        if objective_delta <= float(settings["objective_tolerance"]) and parameter_delta <= float(
            settings["parameter_tolerance"]
        ):
            break

    return {
        "B_matrix": b_matrix,
        "Gamma_matrix": gamma_matrix,
        "fitted_predictions": fitted_predictions,
        "objective_history": objective_history,
        "final_objective": float(objective_history[-1]),
        "conditioning": float(conditioning_value),
        "conditioning_shrink_factor": float(shrink_factor),
        "gamma_update_history": gamma_update_history,
        "chat_update_history": chat_update_history,
        "n_iterations": int(max(0, len(objective_history) - 1)),
    }


def _solve_linear_coupled_response(
    coupled_matrix: np.ndarray,
    driver_matrix: np.ndarray,
) -> np.ndarray:
    try:
        return np.linalg.solve(coupled_matrix, driver_matrix.T).T
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(coupled_matrix, driver_matrix.T, rcond=None)[0].T


def _compute_constraint_max_abs(
    predicted_state: np.ndarray,
    reference_state: np.ndarray,
    invariant_matrix: np.ndarray,
) -> float:
    if invariant_matrix.shape[0] == 0:
        return 0.0
    residual = invariant_matrix @ (predicted_state - reference_state)
    return float(np.max(np.abs(residual)))


def _solve_deployment_qp_batch(
    driver_matrix: np.ndarray,
    influent_matrix: np.ndarray,
    coupled_matrix: np.ndarray,
    invariant_matrix: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_samples, n_outputs = driver_matrix.shape
    lambda_pred = float(settings["lambda_pred"])
    nonnegativity_tolerance = float(settings["nonnegativity_tolerance"])
    constraint_tolerance = float(settings["constraint_tolerance"])

    at_a_matrix = invariant_matrix.T @ invariant_matrix
    rt_r_matrix = coupled_matrix.T @ coupled_matrix
    quadratic_matrix = 2.0 * (rt_r_matrix + lambda_pred * at_a_matrix)

    constraint_matrix = np.eye(n_outputs, dtype=float)
    lower_bounds = np.zeros(n_outputs, dtype=float)
    upper_bounds = np.full(n_outputs, np.inf, dtype=float)
    solver = _make_osqp_solver(
        quadratic_matrix,
        np.zeros(n_outputs, dtype=float),
        constraint_matrix,
        lower_bounds,
        upper_bounds,
        settings,
    )

    raw_predictions = _solve_linear_coupled_response(coupled_matrix, driver_matrix)
    projected_predictions = np.zeros_like(raw_predictions)

    projection_details = {
        "projection_stage": [],
        "raw_feasible_mask": [],
        "affine_feasible_mask": [],
        "lp_active_mask": [],
        "solver_status": [],
        "solver_iterations": [],
        "raw_constraint_max_abs": [],
        "affine_constraint_max_abs": [],
        "projected_constraint_max_abs": [],
        "raw_min_component": [],
        "affine_min_component": [],
        "projected_min_component": [],
    }

    rt_driver_matrix = driver_matrix @ coupled_matrix
    at_a_influents = influent_matrix @ at_a_matrix.T

    for sample_index in range(n_samples):
        raw_state = raw_predictions[sample_index]
        influent_state = influent_matrix[sample_index]

        raw_constraint_max_abs = _compute_constraint_max_abs(raw_state, influent_state, invariant_matrix)
        raw_min_component = float(np.min(raw_state))
        raw_feasible = bool(
            raw_min_component >= -nonnegativity_tolerance and raw_constraint_max_abs <= constraint_tolerance
        )

        if raw_feasible:
            projected_state = raw_state.copy()
            solver_status = "skipped_raw_feasible"
            solver_iterations = 0
            stage = "raw_feasible"
            lp_active = False
        else:
            linear_vector = -2.0 * (
                rt_driver_matrix[sample_index] + lambda_pred * at_a_influents[sample_index]
            )
            solver.update(q=np.asarray(linear_vector, dtype=float))
            if bool(settings["warm_start"]):
                solver.warm_start(x=np.maximum(raw_state, 0.0))
            result = solver.solve()
            solver_status = str(result.info.status)
            solver_iterations = int(result.info.iter)
            if _is_osqp_solved(solver_status) and result.x is not None:
                projected_state = np.maximum(np.asarray(result.x, dtype=float), 0.0)
            else:
                projected_state = np.maximum(raw_state, 0.0)
            stage = "lp_corrected"
            lp_active = True

        projected_predictions[sample_index] = projected_state

        projected_constraint_max_abs = _compute_constraint_max_abs(
            projected_state,
            influent_state,
            invariant_matrix,
        )
        projection_details["projection_stage"].append(stage)
        projection_details["raw_feasible_mask"].append(raw_feasible)
        projection_details["affine_feasible_mask"].append(raw_feasible)
        projection_details["lp_active_mask"].append(lp_active)
        projection_details["solver_status"].append(solver_status)
        projection_details["solver_iterations"].append(solver_iterations)
        projection_details["raw_constraint_max_abs"].append(raw_constraint_max_abs)
        projection_details["affine_constraint_max_abs"].append(raw_constraint_max_abs)
        projection_details["projected_constraint_max_abs"].append(projected_constraint_max_abs)
        projection_details["raw_min_component"].append(raw_min_component)
        projection_details["affine_min_component"].append(raw_min_component)
        projection_details["projected_min_component"].append(float(np.min(projected_state)))

    return raw_predictions, projected_predictions, {
        key: np.asarray(values)
        for key, values in projection_details.items()
    }


def _build_model_bundle(
    *,
    scaling_bundle: ScalingBundle,
    design_schema: Mapping[str, Any],
    feature_columns: list[str],
    target_columns: list[str],
    constraint_columns: list[str],
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray,
    measured_output_columns: list[str] | None,
    B_matrix: np.ndarray,
    Gamma_matrix: np.ndarray,
    best_restart_summary: Mapping[str, Any],
    training_diagnostics: Mapping[str, Any],
    coupled_qp_settings: Mapping[str, Any],
    model_hyperparameters: Mapping[str, Any],
    training_options: Mapping[str, Any],
    composition_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    coupled_matrix = np.eye(Gamma_matrix.shape[0], dtype=float) - np.asarray(Gamma_matrix, dtype=float)

    return {
        "model_name": MODEL_NAME,
        "feature_columns": list(feature_columns),
        "target_columns": list(target_columns),
        "constraint_columns": list(constraint_columns),
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "composition_matrix": np.asarray(composition_matrix, dtype=float),
        "measured_output_columns": None if measured_output_columns is None else list(measured_output_columns),
        "composition_source": None if composition_source is None else dict(composition_source),
        "scaling_bundle": scaling_bundle,
        "design_schema": dict(design_schema),
        "B_matrix": np.asarray(B_matrix, dtype=float),
        "Gamma_matrix": np.asarray(Gamma_matrix, dtype=float),
        "R_matrix": coupled_matrix,
        "best_restart_summary": dict(best_restart_summary),
        "training_diagnostics": dict(training_diagnostics),
        "coupled_qp_settings": dict(coupled_qp_settings),
        "model_hyperparameters": dict(model_hyperparameters),
        "training_options": dict(training_options),
        "feature_space": "operational_plus_fractional_influent",
        "target_space": "fractional_component",
        "constraint_space": "fractional_component",
        "native_prediction_space": "fractional_component",
        "comparison_target_space": "external_measured_output",
        "direct_comparison_scope": "externally_collapsed_measured_output_metrics",
        "projection_active": True,
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

    coupled_qp_settings = dict(
        model_bundle.get("coupled_qp_settings", _resolve_coupled_qp_settings(model_bundle["model_hyperparameters"]))
    )
    driver_matrix = design_frame.to_numpy(dtype=float) @ np.asarray(model_bundle["B_matrix"], dtype=float).T
    raw_fractional_predictions, projected_fractional_predictions, projection_details = _solve_deployment_qp_batch(
        driver_matrix,
        selected_constraint_reference.to_numpy(dtype=float),
        np.asarray(model_bundle["R_matrix"], dtype=float),
        np.asarray(model_bundle["A_matrix"], dtype=float),
        coupled_qp_settings,
    )

    return {
        "design_frame": design_frame,
        "raw_fractional_predictions": raw_fractional_predictions,
        "affine_fractional_predictions": raw_fractional_predictions.copy(),
        "projected_fractional_predictions": projected_fractional_predictions,
        "constraint_reference": selected_constraint_reference,
        "projection_details": projection_details,
    }


def train_icsor_coupled_qp_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, Any],
    *,
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray,
    training_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the coupled-QP icsor model on notebook-prepared fractional features and targets."""

    _validate_scaling_configuration(model_hyperparameters)
    settings = _resolve_coupled_qp_settings(model_hyperparameters)
    objective_label = str(model_hyperparameters.get("objective", "coupled_qp"))
    options = _resolve_training_options(training_options, objective_name=objective_label)

    feature_frame = pd.DataFrame(training_dataset["features"])
    target_frame = pd.DataFrame(training_dataset["targets"])
    constraint_frame = pd.DataFrame(training_dataset["constraint_reference"])

    _validate_composition_shape(
        composition_matrix,
        constraint_columns=list(constraint_frame.columns),
    )

    if target_frame.shape[1] != constraint_frame.shape[1]:
        raise ValueError(
            "icsor_coupled_qp requires effluent ASM targets to match the ASM constraint dimension in width."
        )

    design_frame, design_schema = build_icsor_design_frame(
        feature_frame,
        list(constraint_frame.columns),
        include_bias_term=bool(settings["include_bias_term"]),
    )

    design_matrix = design_frame.to_numpy(dtype=float)
    target_matrix = target_frame.to_numpy(dtype=float)
    influent_matrix = constraint_frame.to_numpy(dtype=float)
    invariant_matrix = np.asarray(A_matrix, dtype=float)

    rng = np.random.default_rng(int(model_hyperparameters.get("random_seed", 42)))
    n_outputs = target_matrix.shape[1]

    progress_bar = create_progress_bar(
        total=int(settings["n_restarts"]),
        desc=str(options["progress_description"]),
        enabled=bool(options["show_progress"]),
        unit="restart",
    )

    restart_summaries: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    try:
        for restart_index in range(int(settings["n_restarts"])):
            if restart_index == 0:
                gamma_init = np.zeros((n_outputs, n_outputs), dtype=float)
            else:
                gamma_init = rng.uniform(
                    low=-float(settings["gamma_abs_bound"]),
                    high=float(settings["gamma_abs_bound"]),
                    size=(n_outputs, n_outputs),
                )
                np.fill_diagonal(gamma_init, 0.0)

            restart_result = _run_coupled_qp_restart(
                design_matrix,
                target_matrix,
                influent_matrix,
                invariant_matrix,
                settings,
                initial_gamma=gamma_init,
            )
            restart_summary = {
                "restart_index": restart_index,
                "final_objective": float(restart_result["final_objective"]),
                "conditioning": float(restart_result["conditioning"]),
                "conditioning_shrink_factor": float(restart_result["conditioning_shrink_factor"]),
                "n_iterations": int(restart_result["n_iterations"]),
            }
            restart_summaries.append(restart_summary)

            if best_result is None or float(restart_result["final_objective"]) < float(best_result["final_objective"]):
                best_result = restart_result

            progress_bar.update(1)
            progress_bar.set_postfix(
                objective=str(options["objective_name"]),
                best=f"{float(min(summary['final_objective'] for summary in restart_summaries)):.6g}",
            )
    finally:
        progress_bar.close()

    if best_result is None:
        raise RuntimeError("icsor_coupled_qp failed to produce any training restart result.")

    best_b_matrix = np.asarray(best_result["B_matrix"], dtype=float)
    best_gamma_matrix = np.asarray(best_result["Gamma_matrix"], dtype=float)
    best_c_matrix = np.asarray(best_result["fitted_predictions"], dtype=float)

    coupled_matrix = np.eye(best_gamma_matrix.shape[0], dtype=float) - best_gamma_matrix
    training_driver_matrix = design_matrix @ best_b_matrix.T
    training_raw_predictions, training_projected_predictions, training_projection_details = _solve_deployment_qp_batch(
        training_driver_matrix,
        influent_matrix,
        coupled_matrix,
        invariant_matrix,
        settings,
    )

    return {
        "design_schema": design_schema,
        "B_matrix": best_b_matrix,
        "Gamma_matrix": best_gamma_matrix,
        "R_matrix": coupled_matrix,
        "fitted_predictions": pd.DataFrame(best_c_matrix, index=target_frame.index, columns=target_frame.columns),
        "training_raw_predictions": pd.DataFrame(
            training_raw_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_affine_predictions": pd.DataFrame(
            training_raw_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_projected_predictions": pd.DataFrame(
            training_projected_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_raw_fractional_predictions": pd.DataFrame(
            training_raw_predictions,
            index=constraint_frame.index,
            columns=constraint_frame.columns,
        ),
        "training_affine_fractional_predictions": pd.DataFrame(
            training_raw_predictions,
            index=constraint_frame.index,
            columns=constraint_frame.columns,
        ),
        "training_projected_fractional_predictions": pd.DataFrame(
            training_projected_predictions,
            index=constraint_frame.index,
            columns=constraint_frame.columns,
        ),
        "training_projection_details": training_projection_details,
        "final_objective": float(best_result["final_objective"]),
        "objective_history": list(best_result["objective_history"]),
        "best_restart_summary": {
            "restart_index": int(
                min(restart_summaries, key=lambda summary: float(summary["final_objective"]))["restart_index"]
            ),
            "final_objective": float(best_result["final_objective"]),
            "conditioning": float(best_result["conditioning"]),
            "conditioning_shrink_factor": float(best_result["conditioning_shrink_factor"]),
            "n_iterations": int(best_result["n_iterations"]),
        },
        "training_diagnostics": {
            "restart_summaries": restart_summaries,
            "gamma_update_history": list(best_result.get("gamma_update_history", [])),
            "chat_update_history": list(best_result.get("chat_update_history", [])),
        },
    }


def predict_icsor_coupled_qp_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted coupled-QP bundle and generate aligned fractional predictions."""

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
                "Raw-dataset prediction requires metadata composition_source workbook_sha256 to match "
                "the trained model bundle composition source."
            )

        prepared_dataset = build_icsor_supervised_dataset(
            test_dataset,
            dict(metadata),
            np.asarray(composition_matrix, dtype=float),
        )
        feature_frame = prepared_dataset.features
        constraint_reference = prepared_dataset.constraint_reference
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

    return {
        "raw_predictions": pd.DataFrame(
            prediction_payload["raw_fractional_predictions"],
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "affine_predictions": pd.DataFrame(
            prediction_payload["affine_fractional_predictions"],
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "projected_predictions": pd.DataFrame(
            prediction_payload["projected_fractional_predictions"],
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
        "projection_stage_summary": build_icsor_projection_stage_summary(
            prediction_payload["projection_details"],
        ),
    }


def run_icsor_coupled_qp_pipeline(
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    A_matrix: np.ndarray,
    *,
    composition_matrix: np.ndarray,
    measured_output_columns: list[str] | None = None,
    composition_source: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    model_params: Mapping[str, Any] | None = None,
    model_hyperparameters: Mapping[str, Any] | None = None,
    optuna_summary: Mapping[str, Any] | None = None,
    show_progress: bool = True,
    persist_artifacts: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and optionally persist one coupled-QP icsor bundle."""

    del optuna_summary

    params = dict(model_params) if model_params is not None else load_icsor_coupled_qp_params(repo_root)
    split_params = dict(params["hyperparameters"])
    selected_hyperparameters = resolve_model_hyperparameters(params, model_hyperparameters)
    selected_hyperparameters.setdefault("objective", "coupled_qp")

    _validate_scaling_configuration({**split_params, **selected_hyperparameters})
    coupled_qp_settings = _resolve_coupled_qp_settings(selected_hyperparameters)

    objective_label = str(selected_hyperparameters["objective"])
    resolved_measured_output_columns = (
        None if measured_output_columns is None else [str(column_name) for column_name in measured_output_columns]
    )

    progress_bar = create_progress_bar(
        total=5,
        desc="Training icsor_coupled_qp",
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
        training_result = train_icsor_coupled_qp_model(
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
                "progress_description": "Training icsor_coupled_qp",
                "objective_name": objective_label,
            },
        )
        progress_bar.update(1)

        model_bundle = _build_model_bundle(
            scaling_bundle=scaling_bundle,
            design_schema=training_result["design_schema"],
            feature_columns=list(training_split.features.columns),
            target_columns=list(training_split.targets.columns),
            constraint_columns=list(training_split.constraint_reference.columns),
            A_matrix=np.asarray(A_matrix, dtype=float),
            composition_matrix=np.asarray(composition_matrix, dtype=float),
            measured_output_columns=resolved_measured_output_columns,
            B_matrix=np.asarray(training_result["B_matrix"], dtype=float),
            Gamma_matrix=np.asarray(training_result["Gamma_matrix"], dtype=float),
            best_restart_summary=training_result["best_restart_summary"],
            training_diagnostics=training_result["training_diagnostics"],
            coupled_qp_settings=coupled_qp_settings,
            model_hyperparameters=selected_hyperparameters,
            training_options={
                "objective_name": objective_label,
                "show_progress": show_progress,
            },
            composition_source=composition_source,
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
            measured_output_columns=resolved_measured_output_columns,
            index=training_split.targets.index,
            projection_details=train_prediction_payload["projection_details"],
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
            measured_output_columns=resolved_measured_output_columns,
            index=test_split.targets.index,
            projection_details=test_prediction_payload["projection_details"],
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
        "optuna_summary": None,
        "artifact_paths": artifact_paths,
        "train_report": train_report,
        "test_report": test_report,
        "model_bundle": model_bundle,
        "dataset_splits": dataset_splits,
    }


__all__ = [
    "MODEL_NAME",
    "load_icsor_coupled_qp_params",
    "predict_icsor_coupled_qp_model",
    "run_icsor_coupled_qp_pipeline",
    "train_icsor_coupled_qp_model",
]
