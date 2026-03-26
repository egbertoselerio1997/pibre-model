"""ASM1-inspired composite-space simulation used by the system simulation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.simulation import load_model_params, save_simulation_artifacts


MODEL_NAME = "asm1_simulation"
N_PROCESSES = 8
N_SPECIES = 13


def load_asm1_simulation_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the ASM1 simulation."""

    return load_model_params(MODEL_NAME, repo_root)


def get_asm1_matrices(model_params: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Build the composite-space stoichiometric matrix used by the simulation."""

    composites = list(model_params["composites"])
    if len(composites) != 5:
        raise ValueError("asm1_simulation expects exactly five composite variables.")

    yields = model_params["stoichiometry"]["yields"]
    constants = model_params["stoichiometry"]["constants"]

    y_h = float(yields["Y_H"])
    y_a = float(yields["Y_A"])
    f_p = float(yields["f_P"])
    i_xb = float(yields["i_XB"])
    i_xp = float(yields["i_XP"])

    anoxic_nitrate_factor = float(constants["anoxic_nitrate_factor"])
    oxygen_nitrification_factor = float(constants["oxygen_nitrification_factor"])
    nitrogen_atomic_mass_factor = float(constants["nitrogen_atomic_mass_factor"])
    alkalinity_ammonification_factor = float(constants["alkalinity_ammonification_factor"])

    nu = np.zeros((N_PROCESSES, N_SPECIES), dtype=float)
    nu[0, 1], nu[0, 4], nu[0, 7], nu[0, 9], nu[0, 12] = (
        -1.0 / y_h,
        1.0,
        -(1.0 - y_h) / y_h,
        -i_xb,
        -i_xb / nitrogen_atomic_mass_factor,
    )
    nu[1, 1], nu[1, 4], nu[1, 8], nu[1, 9], nu[1, 12] = (
        -1.0 / y_h,
        1.0,
        -(1.0 - y_h) / (anoxic_nitrate_factor * y_h),
        -i_xb,
        ((1.0 - y_h) / (nitrogen_atomic_mass_factor * anoxic_nitrate_factor * y_h))
        - (i_xb / nitrogen_atomic_mass_factor),
    )
    nu[2, 9], nu[2, 5], nu[2, 7], nu[2, 8], nu[2, 12] = (
        -1.0 / y_a - i_xb,
        1.0,
        -(oxygen_nitrification_factor - y_a) / y_a,
        1.0 / y_a,
        -i_xb / nitrogen_atomic_mass_factor - 1.0 / (alkalinity_ammonification_factor * y_a),
    )
    nu[3, 4], nu[3, 3], nu[3, 6], nu[3, 11] = -1.0, 1.0 - f_p, f_p, i_xb - f_p * i_xp
    nu[4, 5], nu[4, 3], nu[4, 6], nu[4, 11] = -1.0, 1.0 - f_p, f_p, i_xb - f_p * i_xp
    nu[5, 10], nu[5, 9], nu[5, 12] = -1.0, 1.0, 1.0 / nitrogen_atomic_mass_factor
    nu[6, 3], nu[6, 1] = -1.0, 1.0
    nu[7, 11], nu[7, 10] = -1.0, 1.0

    composition = np.zeros((N_SPECIES, len(composites)), dtype=float)
    for species_index in [0, 1, 2, 3, 4, 5, 6]:
        composition[species_index, 0] = 1.0
    for species_index in [8, 9, 10, 11]:
        composition[species_index, 1] = 1.0

    composition[4, 1], composition[5, 1], composition[6, 1] = i_xb, i_xb, i_xp
    composition[7, 2], composition[8, 3], composition[12, 4] = 1.0, 1.0, 1.0

    return np.dot(nu, composition), composites


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


def build_asm1_metadata(
    model_params: Mapping[str, Any],
    *,
    sample_count: int,
    random_seed: int,
    dataset_file: str | None = None,
) -> dict[str, Any]:
    """Create the metadata contract required for simulation outputs."""

    composites = list(model_params["composites"])
    operational_columns = list(model_params["operational_columns"])

    return {
        "simulation_name": MODEL_NAME,
        "n_samples": sample_count,
        "random_seed": random_seed,
        "dependent_columns": [f"Out_{name}" for name in composites],
        "independent_columns": operational_columns + [f"In_{name}" for name in composites],
        "identifier_columns": [],
        "ignored_columns": [],
        "dataset_file": dataset_file,
        "composites": composites,
        "operational_columns": operational_columns,
        "extent_rules": [rule["name"] for rule in model_params["extent_rules"]],
    }


def generate_asm1_dataset(
    *,
    model_params: Mapping[str, Any] | None = None,
    n_samples: int | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """Generate a synthetic influent-effluent dataset from the configured ASM1 simulation."""

    params = dict(model_params) if model_params is not None else load_asm1_simulation_params()
    configured_hyperparameters = params["hyperparameters"]
    sample_count = int(n_samples if n_samples is not None else configured_hyperparameters["n_samples"])
    seed = int(random_seed if random_seed is not None else configured_hyperparameters["seed"])
    rng = np.random.default_rng(seed)

    composite_matrix, composites = get_asm1_matrices(params)
    operational_columns = list(params["operational_columns"])

    influent = _sample_named_ranges(
        rng,
        sample_count,
        composites,
        params["influent_ranges"],
    )
    operational = _sample_named_ranges(
        rng,
        sample_count,
        operational_columns,
        params["operational_ranges"],
    )

    extent_rules = list(params["extent_rules"])
    if len(extent_rules) > N_PROCESSES:
        raise ValueError("asm1_simulation extent rules exceed the supported process count.")

    composite_index = {name: index for index, name in enumerate(composites)}
    extents = np.zeros((sample_count, N_PROCESSES), dtype=float)
    for process_index, rule in enumerate(extent_rules):
        component_name = rule["influent_component"]
        if component_name not in composite_index:
            raise KeyError(f"Unknown influent component in extent rule: {component_name}")

        lower_bound = float(rule["min_fraction"])
        upper_bound = float(rule["max_fraction"])
        source_values = influent[:, composite_index[component_name]]
        extents[:, process_index] = source_values * rng.uniform(lower_bound, upper_bound, sample_count)

    effluent = influent + np.dot(extents, composite_matrix)

    influent_df = pd.DataFrame(influent, columns=[f"In_{name}" for name in composites])
    operational_df = pd.DataFrame(operational, columns=operational_columns)
    effluent_df = pd.DataFrame(effluent, columns=[f"Out_{name}" for name in composites])
    dataset = pd.concat([operational_df, influent_df, effluent_df], axis=1)

    metadata = build_asm1_metadata(
        params,
        sample_count=sample_count,
        random_seed=seed,
    )

    return dataset, metadata, composite_matrix


def run_asm1_simulation(
    *,
    save_artifacts: bool = True,
    repo_root: str | Path | None = None,
    n_samples: int | None = None,
    random_seed: int | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Run the configured ASM1 simulation and optionally persist the dataset contract."""

    params = load_asm1_simulation_params(repo_root)
    dataset, metadata, composite_matrix = generate_asm1_dataset(
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
        "composite_matrix": composite_matrix,
        "artifact_paths": artifact_paths,
    }


__all__ = [
    "build_asm1_metadata",
    "generate_asm1_dataset",
    "get_asm1_matrices",
    "load_asm1_simulation_params",
    "run_asm1_simulation",
]