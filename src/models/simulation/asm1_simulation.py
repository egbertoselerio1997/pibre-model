"""Mechanistic steady-state activated-sludge CSTR simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.utils.simulation import load_model_params, save_simulation_artifacts


MODEL_NAME = "asm1_simulation"
N_PROCESSES = 6


REQUIRED_STATES = [
    "S_S",
    "S_I",
    "S_NH4_N",
    "S_NO3_N",
    "S_PO4_P",
    "S_O2",
    "S_Alkalinity",
    "X_I",
    "X_S",
    "X_H",
    "X_AUT",
]

REQUIRED_MEASURED_OUTPUTS = [
    "COD",
    "TSS",
    "VSS",
    "TN",
    "TP",
    "NH4_N",
    "NO3_N",
    "PO4_P",
    "DO",
    "Alkalinity",
]


def load_asm1_simulation_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the steady-state CSTR simulation."""

    return load_model_params(MODEL_NAME, repo_root)


def get_asm1_matrices(model_params: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Return the internal-state identity map used for downstream diagnostics."""

    state_columns = list(model_params["state_columns"])
    missing = [name for name in REQUIRED_STATES if name not in state_columns]
    if missing:
        missing_display = ", ".join(missing)
        raise KeyError(f"asm1_simulation missing required state columns: {missing_display}")

    measured_output_columns = list(model_params["measured_output_columns"])
    missing_outputs = [name for name in REQUIRED_MEASURED_OUTPUTS if name not in measured_output_columns]
    if missing_outputs:
        missing_output_display = ", ".join(missing_outputs)
        raise KeyError(f"asm1_simulation missing required measured outputs: {missing_output_display}")

    return np.eye(len(state_columns), dtype=float), state_columns


def _sample_named_ranges(
    rng: np.random.Generator,
    sample_count: int,
    ordered_names: list[str],
    ranges: Mapping[str, Any],
) -> np.ndarray:
    sampled = np.zeros((sample_count, len(ordered_names)), dtype=float)
    for column_index, column_name in enumerate(ordered_names):
        lower_bound, upper_bound = ranges[column_name]
        sampled[:, column_index] = rng.uniform(float(lower_bound), float(upper_bound), sample_count)

    return sampled


def _build_state_index(state_columns: list[str]) -> dict[str, int]:
    return {name: position for position, name in enumerate(state_columns)}


def _monod(numerator: float, half_saturation: float) -> float:
    return float(numerator) / max(float(numerator) + float(half_saturation), 1e-9)


def _steady_state_residuals(
    state: np.ndarray,
    influent_state: np.ndarray,
    hrt_hours: float,
    aeration: float,
    state_columns: list[str],
    model_params: Mapping[str, Any],
) -> np.ndarray:
    index = _build_state_index(state_columns)
    kinetics = model_params["kinetics"]
    factors = model_params["stoichiometric_factors"]
    aeration_model = model_params["aeration_model"]

    s_s = state[index["S_S"]]
    s_nh4 = state[index["S_NH4_N"]]
    s_no3 = state[index["S_NO3_N"]]
    s_po4 = state[index["S_PO4_P"]]
    s_o2 = state[index["S_O2"]]
    x_s = state[index["X_S"]]
    x_h = state[index["X_H"]]
    x_aut = state[index["X_AUT"]]

    dilution_rate = 24.0 / max(float(hrt_hours), 1e-6)
    kla = float(aeration_model["kla_base"]) + float(aeration_model["kla_per_aeration"]) * max(float(aeration), 0.0)
    do_saturation = float(aeration_model["do_saturation"])

    oxygen_term_h = _monod(s_o2, float(kinetics["heterotroph_oxygen_half_saturation"]))
    nitrate_term_h = _monod(s_no3, float(kinetics["heterotroph_nitrate_half_saturation"]))
    substrate_term = _monod(s_s, float(kinetics["readily_biodegradable_half_saturation"]))
    ammonium_term = _monod(s_nh4, float(kinetics["autotroph_ammonium_half_saturation"]))
    oxygen_term_a = _monod(s_o2, float(kinetics["autotroph_oxygen_half_saturation"]))

    hydrolysis_acceptor = oxygen_term_h + (
        float(kinetics["anoxic_hydrolysis_factor"]) * (1.0 - oxygen_term_h) * nitrate_term_h
    )

    hydrolysis_rate = (
        float(kinetics["hydrolysis_rate"])
        * x_h
        * (x_s / max(float(kinetics["hydrolysis_half_saturation"]) + x_s, 1e-9))
        * hydrolysis_acceptor
    )
    heterotroph_aerobic_rate = float(kinetics["heterotroph_max_growth_rate"]) * substrate_term * oxygen_term_h * x_h
    heterotroph_anoxic_rate = (
        float(kinetics["heterotroph_max_growth_rate"])
        * float(kinetics["anoxic_growth_factor"])
        * substrate_term
        * (1.0 - oxygen_term_h)
        * nitrate_term_h
        * x_h
    )
    autotroph_growth_rate = float(kinetics["autotroph_max_growth_rate"]) * ammonium_term * oxygen_term_a * x_aut
    heterotroph_decay_rate = float(kinetics["heterotroph_decay_rate"]) * x_h
    autotroph_decay_rate = float(kinetics["autotroph_decay_rate"]) * x_aut

    y_h = float(factors["heterotroph_yield"])
    y_a = float(factors["autotroph_yield"])
    anoxic_nitrate_factor = float(factors["anoxic_nitrate_factor"])

    substrate_consumed_aerobic = heterotroph_aerobic_rate / max(y_h, 1e-9)
    substrate_consumed_anoxic = heterotroph_anoxic_rate / max(y_h, 1e-9)
    oxygen_consumed_aerobic = ((1.0 - y_h) / max(y_h, 1e-9)) * heterotroph_aerobic_rate
    nitrate_consumed_anoxic = ((1.0 - y_h) / max(anoxic_nitrate_factor * y_h, 1e-9)) * heterotroph_anoxic_rate

    ammonium_consumed_autotroph = (1.0 / max(y_a, 1e-9) + float(factors["nitrogen_content_x_aut"])) * autotroph_growth_rate
    nitrate_generated_autotroph = autotroph_growth_rate / max(y_a, 1e-9)
    oxygen_consumed_autotroph = float(factors["oxygen_per_nh4_nitrified"]) * nitrate_generated_autotroph
    alkalinity_consumed_autotroph = float(factors["alkalinity_per_nh4_nitrified"]) * nitrate_generated_autotroph
    alkalinity_recovered_anoxic = float(factors["alkalinity_recovery_per_nox_denitrified"]) * nitrate_consumed_anoxic

    decay_to_inert_fraction_h = float(factors["heterotroph_decay_to_inert_fraction"])
    decay_to_inert_fraction_a = float(factors["autotroph_decay_to_inert_fraction"])
    decay_to_inert_h = decay_to_inert_fraction_h * heterotroph_decay_rate
    decay_to_slow_h = heterotroph_decay_rate - decay_to_inert_h
    decay_to_inert_a = decay_to_inert_fraction_a * autotroph_decay_rate
    decay_to_slow_a = autotroph_decay_rate - decay_to_inert_a

    n_release_h = (
        float(factors["nitrogen_content_x_h"])
        - (
            float(factors["nitrogen_content_x_i"]) * decay_to_inert_fraction_h
            + float(factors["nitrogen_content_x_s"]) * (1.0 - decay_to_inert_fraction_h)
        )
    ) * heterotroph_decay_rate
    n_release_a = (
        float(factors["nitrogen_content_x_aut"])
        - (
            float(factors["nitrogen_content_x_i"]) * decay_to_inert_fraction_a
            + float(factors["nitrogen_content_x_s"]) * (1.0 - decay_to_inert_fraction_a)
        )
    ) * autotroph_decay_rate
    p_release_h = (
        float(factors["phosphorus_content_x_h"])
        - (
            float(factors["phosphorus_content_x_i"]) * decay_to_inert_fraction_h
            + float(factors["phosphorus_content_x_s"]) * (1.0 - decay_to_inert_fraction_h)
        )
    ) * heterotroph_decay_rate
    p_release_a = (
        float(factors["phosphorus_content_x_aut"])
        - (
            float(factors["phosphorus_content_x_i"]) * decay_to_inert_fraction_a
            + float(factors["phosphorus_content_x_s"]) * (1.0 - decay_to_inert_fraction_a)
        )
    ) * autotroph_decay_rate

    residual = dilution_rate * (influent_state - state)

    residual[index["S_S"]] += hydrolysis_rate - substrate_consumed_aerobic - substrate_consumed_anoxic
    residual[index["S_I"]] += 0.0
    residual[index["S_NH4_N"]] += (
        -float(factors["nitrogen_content_x_h"]) * (heterotroph_aerobic_rate + heterotroph_anoxic_rate)
        - ammonium_consumed_autotroph
        + n_release_h
        + n_release_a
    )
    residual[index["S_NO3_N"]] += nitrate_generated_autotroph - nitrate_consumed_anoxic
    residual[index["S_PO4_P"]] += (
        -float(factors["phosphorus_content_x_h"]) * (heterotroph_aerobic_rate + heterotroph_anoxic_rate)
        - float(factors["phosphorus_content_x_aut"]) * autotroph_growth_rate
        + p_release_h
        + p_release_a
    )
    residual[index["S_O2"]] += kla * (do_saturation - s_o2) - oxygen_consumed_aerobic - oxygen_consumed_autotroph
    residual[index["S_Alkalinity"]] += alkalinity_recovered_anoxic - alkalinity_consumed_autotroph
    residual[index["X_I"]] += decay_to_inert_h + decay_to_inert_a
    residual[index["X_S"]] += -hydrolysis_rate + decay_to_slow_h + decay_to_slow_a
    residual[index["X_H"]] += heterotroph_aerobic_rate + heterotroph_anoxic_rate - heterotroph_decay_rate
    residual[index["X_AUT"]] += autotroph_growth_rate - autotroph_decay_rate

    return residual


def _build_initial_guess(
    influent_state: np.ndarray,
    hrt_hours: float,
    aeration: float,
    state_columns: list[str],
    model_params: Mapping[str, Any],
    previous_solution: np.ndarray | None = None,
) -> np.ndarray:
    solver = model_params["solver"]
    aeration_model = model_params["aeration_model"]
    lower_floor = float(solver["initial_guess_floor"])
    guess = np.maximum(influent_state.copy(), lower_floor)

    if previous_solution is not None:
        guess = np.maximum(0.65 * previous_solution + 0.35 * guess, lower_floor)

    index = _build_state_index(state_columns)
    if previous_solution is None:
        guess[index["X_H"]] = max(
            guess[index["X_H"]],
            guess[index["X_S"]] * float(solver["initial_heterotroph_to_xs_ratio"]),
        )
        guess[index["X_AUT"]] = max(
            guess[index["X_AUT"]],
            guess[index["S_NH4_N"]] * float(solver["initial_autotroph_to_nh4_ratio"]),
        )

    dilution_rate = 24.0 / max(float(hrt_hours), 1e-6)
    kla = float(aeration_model["kla_base"]) + float(aeration_model["kla_per_aeration"]) * max(float(aeration), 0.0)
    do_saturation = float(aeration_model["do_saturation"])
    guess[index["S_O2"]] = np.clip(
        (dilution_rate * guess[index["S_O2"]] + kla * do_saturation) / max(dilution_rate + kla, 1e-9),
        lower_floor,
        do_saturation,
    )
    guess[index["S_S"]] *= float(solver["initial_soluble_substrate_fraction"])
    guess[index["X_S"]] *= float(solver["initial_slow_substrate_fraction"])

    return guess


def _compute_measured_outputs(
    state: np.ndarray,
    state_columns: list[str],
    model_params: Mapping[str, Any],
) -> dict[str, float]:
    index = _build_state_index(state_columns)
    observation_model = model_params["observation_model"]
    particulate_states = ["X_I", "X_S", "X_H", "X_AUT"]

    cod = sum(float(state[index[name]]) for name in ["S_S", "S_I", *particulate_states])
    tss = sum(
        float(state[index[name]]) * float(observation_model["state_tss_factors"][name])
        for name in particulate_states
    )
    vss = sum(
        float(state[index[name]]) * float(observation_model["state_vss_factors"][name])
        for name in particulate_states
    )
    tn = float(state[index["S_NH4_N"]]) + float(state[index["S_NO3_N"]]) + sum(
        float(state[index[name]]) * float(observation_model["particulate_nitrogen_factors"][name])
        for name in particulate_states
    )
    tp = float(state[index["S_PO4_P"]]) + sum(
        float(state[index[name]]) * float(observation_model["particulate_phosphorus_factors"][name])
        for name in particulate_states
    )

    return {
        "COD": cod,
        "TSS": tss,
        "VSS": vss,
        "TN": tn,
        "TP": tp,
        "NH4_N": float(state[index["S_NH4_N"]]),
        "NO3_N": float(state[index["S_NO3_N"]]),
        "PO4_P": float(state[index["S_PO4_P"]]),
        "DO": float(state[index["S_O2"]]),
        "Alkalinity": float(state[index["S_Alkalinity"]]),
    }


def solve_asm1_cstr_steady_state(
    *,
    influent_state: np.ndarray,
    hrt_hours: float,
    aeration: float,
    model_params: Mapping[str, Any],
    previous_solution: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    """Solve a single mechanistic steady-state CSTR operating point."""

    state_columns = list(model_params["state_columns"])
    solver = model_params["solver"]
    lower_bounds = np.full(len(state_columns), float(solver["lower_bound"]), dtype=float)
    upper_bounds = np.full(len(state_columns), np.inf, dtype=float)
    initial_guess = _build_initial_guess(
        influent_state,
        hrt_hours,
        aeration,
        state_columns,
        model_params,
        previous_solution=previous_solution,
    )

    candidate_guesses = [initial_guess]
    biomass_rich_guess = initial_guess.copy()
    index = _build_state_index(state_columns)
    biomass_rich_guess[index["X_H"]] = max(
        biomass_rich_guess[index["X_H"]],
        influent_state[index["X_S"]] * float(solver["multistart_heterotroph_to_xs_ratio"]),
    )
    biomass_rich_guess[index["X_AUT"]] = max(
        biomass_rich_guess[index["X_AUT"]],
        influent_state[index["S_NH4_N"]] * float(solver["multistart_autotroph_to_nh4_ratio"]),
    )
    biomass_rich_guess[index["S_S"]] *= float(solver["multistart_soluble_substrate_fraction"])
    biomass_rich_guess[index["X_S"]] *= float(solver["multistart_slow_substrate_fraction"])
    candidate_guesses.append(biomass_rich_guess)

    best_result = None
    best_residual_max = np.inf
    for candidate_guess in candidate_guesses:
        result = least_squares(
            _steady_state_residuals,
            candidate_guess,
            bounds=(lower_bounds, upper_bounds),
            xtol=float(solver["variable_tolerance"]),
            ftol=float(solver["residual_tolerance"]),
            gtol=float(solver["gradient_tolerance"]),
            max_nfev=int(solver["max_nfev"]),
            args=(influent_state, hrt_hours, aeration, state_columns, model_params),
        )
        residual_max = float(np.max(np.abs(result.fun)))
        if residual_max < best_residual_max:
            best_result = result
            best_residual_max = residual_max

    if best_residual_max > float(solver["acceptance_residual_max"]):
        dynamic_result = solve_ivp(
            lambda _time, values: _steady_state_residuals(
                values,
                influent_state,
                hrt_hours,
                aeration,
                state_columns,
                model_params,
            ),
            (0.0, float(solver["dynamic_relaxation_days"])),
            np.maximum(candidate_guesses[-1], lower_bounds),
            method="BDF",
            atol=float(solver["dynamic_absolute_tolerance"]),
            rtol=float(solver["dynamic_relative_tolerance"]),
            max_step=float(solver["dynamic_max_step"]),
        )
        if dynamic_result.success:
            relaxed_guess = np.maximum(dynamic_result.y[:, -1], lower_bounds)
            result = least_squares(
                _steady_state_residuals,
                relaxed_guess,
                bounds=(lower_bounds, upper_bounds),
                xtol=float(solver["variable_tolerance"]),
                ftol=float(solver["residual_tolerance"]),
                gtol=float(solver["gradient_tolerance"]),
                max_nfev=int(solver["max_nfev"]),
                args=(influent_state, hrt_hours, aeration, state_columns, model_params),
            )
            residual_max = float(np.max(np.abs(result.fun)))
            if residual_max < best_residual_max:
                best_result = result
                best_residual_max = residual_max

    assert best_result is not None
    result = best_result

    diagnostics: dict[str, float | bool | int] = {
        "success": bool(result.success),
        "status": int(result.status),
        "nfev": int(result.nfev),
        "residual_l2": float(np.linalg.norm(result.fun)),
        "residual_max": best_residual_max,
    }

    if (not result.success) or diagnostics["residual_max"] > float(solver["acceptance_residual_max"]):
        raise RuntimeError(
            "asm1_simulation steady-state solve failed: "
            f"success={result.success}, status={result.status}, residual_max={diagnostics['residual_max']:.3e}"
        )

    return result.x, diagnostics


def build_asm1_metadata(
    model_params: Mapping[str, Any],
    *,
    sample_count: int,
    random_seed: int,
    dataset_file: str | None = None,
) -> dict[str, Any]:
    """Create the metadata contract required for simulation outputs."""

    state_columns = list(model_params["state_columns"])
    operational_columns = list(model_params["operational_columns"])
    measured_output_columns = list(model_params["measured_output_columns"])
    schema_version = str(model_params["schema_version"])

    return {
        "simulation_name": MODEL_NAME,
        "n_samples": sample_count,
        "random_seed": random_seed,
        "dependent_columns": [f"Out_{name}" for name in state_columns] + [f"Out_{name}" for name in measured_output_columns],
        "independent_columns": operational_columns + [f"In_{name}" for name in state_columns],
        "identifier_columns": [],
        "ignored_columns": [],
        "dataset_file": dataset_file,
        "state_columns": state_columns,
        "measured_output_columns": measured_output_columns,
        "operational_columns": operational_columns,
        "processes": list(model_params["processes"]),
        "schema_version": schema_version,
    }


def generate_asm1_dataset(
    *,
    model_params: Mapping[str, Any] | None = None,
    n_samples: int | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """Generate a mechanistic steady-state CSTR dataset from configured state ranges."""

    params = dict(model_params) if model_params is not None else load_asm1_simulation_params()
    configured_hyperparameters = params["hyperparameters"]
    sample_count = int(n_samples if n_samples is not None else configured_hyperparameters["n_samples"])
    seed = int(random_seed if random_seed is not None else configured_hyperparameters["seed"])
    max_sample_attempts = int(configured_hyperparameters["max_sample_attempts"])
    rng = np.random.default_rng(seed)

    state_matrix, state_columns = get_asm1_matrices(params)
    operational_columns = list(params["operational_columns"])
    measured_output_columns = list(params["measured_output_columns"])

    if len(params["processes"]) != N_PROCESSES:
        raise ValueError(f"asm1_simulation requires exactly {N_PROCESSES} configured process descriptors.")

    influent_states = np.zeros((sample_count, len(state_columns)), dtype=float)
    operational = np.zeros((sample_count, len(operational_columns)), dtype=float)
    effluent_states = np.zeros((sample_count, len(state_columns)), dtype=float)
    measured_outputs = np.zeros((sample_count, len(measured_output_columns)), dtype=float)
    previous_solution: np.ndarray | None = None
    for sample_index in range(sample_count):
        solved = False
        last_error: RuntimeError | None = None
        for _attempt_index in range(max_sample_attempts):
            candidate_influent = _sample_named_ranges(
                rng,
                1,
                state_columns,
                params["influent_state_ranges"],
            )[0]
            candidate_operational = _sample_named_ranges(
                rng,
                1,
                operational_columns,
                params["operational_ranges"],
            )[0]
            try:
                effluent_state, _ = solve_asm1_cstr_steady_state(
                    influent_state=candidate_influent,
                    hrt_hours=float(candidate_operational[0]),
                    aeration=float(candidate_operational[1]),
                    model_params=params,
                    previous_solution=previous_solution,
                )
            except RuntimeError as error:
                last_error = error
                continue

            influent_states[sample_index] = candidate_influent
            operational[sample_index] = candidate_operational
            solved = True
            break

        if not solved:
            raise RuntimeError(
                "asm1_simulation failed to generate a valid steady-state sample "
                f"after {max_sample_attempts} attempts at sample index {sample_index}."
            ) from last_error

        previous_solution = effluent_state
        effluent_states[sample_index] = effluent_state
        observations = _compute_measured_outputs(effluent_state, state_columns, params)
        measured_outputs[sample_index] = [observations[name] for name in measured_output_columns]

    influent_df = pd.DataFrame(influent_states, columns=[f"In_{name}" for name in state_columns])
    operational_df = pd.DataFrame(operational, columns=operational_columns)
    effluent_df = pd.DataFrame(effluent_states, columns=[f"Out_{name}" for name in state_columns])
    measured_df = pd.DataFrame(measured_outputs, columns=[f"Out_{name}" for name in measured_output_columns])
    dataset = pd.concat([operational_df, influent_df, effluent_df, measured_df], axis=1)

    metadata = build_asm1_metadata(
        params,
        sample_count=sample_count,
        random_seed=seed,
    )

    return dataset, metadata, state_matrix


def run_asm1_simulation(
    *,
    save_artifacts: bool = True,
    repo_root: str | Path | None = None,
    n_samples: int | None = None,
    random_seed: int | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Run the configured steady-state CSTR simulation and optionally persist artifacts."""

    params = load_asm1_simulation_params(repo_root)
    dataset, metadata, state_matrix = generate_asm1_dataset(
        model_params=params,
        n_samples=n_samples,
        random_seed=random_seed,
    )

    artifact_paths: dict[str, Path | None] = {
        "dataset_csv": None,
        "metadata_json": None,
    }

    if save_artifacts:
        dataset_path, metadata_path, persisted_metadata = save_simulation_artifacts(
            dataset,
            metadata,
            MODEL_NAME,
            repo_root=repo_root,
            timestamp=timestamp,
        )
        metadata = persisted_metadata
        artifact_paths = {
            "dataset_csv": dataset_path,
            "metadata_json": metadata_path,
        }

    return {
        "dataset": dataset,
        "metadata": metadata,
        "composite_matrix": state_matrix,
        "artifact_paths": artifact_paths,
    }


__all__ = [
    "build_asm1_metadata",
    "generate_asm1_dataset",
    "get_asm1_matrices",
    "load_asm1_simulation_params",
    "run_asm1_simulation",
    "solve_asm1_cstr_steady_state",
]