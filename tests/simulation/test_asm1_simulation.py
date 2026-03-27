"""Minimal tests for the mechanistic asm1_simulation model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.simulation.asm1_simulation import (
    _compute_measured_outputs,
    generate_asm1_dataset,
    get_asm1_matrices,
    load_asm1_simulation_params,
    run_asm1_simulation,
    solve_asm1_cstr_steady_state,
)
from src.utils.simulation import save_simulation_artifacts


class Asm1SimulationTests(unittest.TestCase):
    def test_generate_asm1_dataset_matches_metadata_contract(self) -> None:
        dataset, metadata, matrix_bundle = generate_asm1_dataset(n_samples=12, random_seed=7)

        self.assertEqual(dataset.shape, (12, 32))
        self.assertEqual(matrix_bundle["petersen_matrix"].shape, (7, 11))
        self.assertEqual(matrix_bundle["composition_matrix"].shape, (8, 11))
        self.assertEqual(metadata["simulation_name"], "asm1_simulation")
        self.assertEqual(metadata["n_samples"], 12)
        self.assertEqual(metadata["schema_version"], "3.2")
        self.assertEqual(metadata["petersen_matrix_shape"], [7, 11])
        self.assertEqual(metadata["composition_matrix_shape"], [8, 11])
        self.assertEqual(metadata["process_types"][-1], "mass_transfer")
        self.assertIsNone(metadata["dataset_file"])

        expected_independent = [
            "HRT",
            "Aeration",
            "In_S_S",
            "In_S_I",
            "In_S_NH4_N",
            "In_S_NO3_N",
            "In_S_PO4_P",
            "In_S_O2",
            "In_S_Alkalinity",
            "In_X_I",
            "In_X_S",
            "In_X_H",
            "In_X_AUT",
        ]
        expected_effluent_states = [
            "Out_S_S",
            "Out_S_I",
            "Out_S_NH4_N",
            "Out_S_NO3_N",
            "Out_S_PO4_P",
            "Out_S_O2",
            "Out_S_Alkalinity",
            "Out_X_I",
            "Out_X_S",
            "Out_X_H",
            "Out_X_AUT",
        ]
        expected_measured_outputs = [
            "Out_COD",
            "Out_TSS",
            "Out_TN",
            "Out_TP",
            "Out_NH4_N",
            "Out_NO3_N",
            "Out_PO4_P",
            "Out_Alkalinity",
        ]
        expected_dependent = expected_effluent_states + expected_measured_outputs

        self.assertEqual(metadata["independent_columns"], expected_independent)
        self.assertEqual(metadata["dependent_columns"], expected_dependent)
        self.assertEqual(list(dataset.columns), expected_independent + expected_dependent)

        self.assertTrue((dataset["Out_Alkalinity"] >= 0.0).all())
        self.assertTrue((dataset["Out_NH4_N"] >= 0.0).all())
        self.assertTrue((dataset["Out_NO3_N"] >= 0.0).all())
        self.assertTrue((dataset["Out_PO4_P"] >= 0.0).all())

    def test_explicit_matrices_capture_aeration_and_output_mapping(self) -> None:
        params = load_asm1_simulation_params()
        matrix_bundle = get_asm1_matrices(params)
        state_columns = list(matrix_bundle["state_columns"])
        measured_output_columns = list(matrix_bundle["measured_output_columns"])
        state_index = {name: position for position, name in enumerate(state_columns)}

        aeration_row = matrix_bundle["petersen_matrix"][-1]
        self.assertEqual(matrix_bundle["process_names"][-1], "Aeration")
        self.assertEqual(matrix_bundle["process_types"][-1], "mass_transfer")
        self.assertEqual(np.count_nonzero(aeration_row), 1)
        self.assertEqual(aeration_row[state_index["S_O2"]], 1.0)

        sample_state = np.array([15.0, 30.0, 4.0, 6.0, 2.0, 3.5, 120.0, 25.0, 40.0, 80.0, 10.0], dtype=float)
        observed = matrix_bundle["composition_matrix"] @ sample_state
        observation_model = params["observation_model"]

        expected = {
            "COD": 15.0 + 30.0 + 25.0 + 40.0 + 80.0 + 10.0,
            "TSS": 25.0 * observation_model["state_tss_factors"]["X_I"]
            + 40.0 * observation_model["state_tss_factors"]["X_S"]
            + 80.0 * observation_model["state_tss_factors"]["X_H"]
            + 10.0 * observation_model["state_tss_factors"]["X_AUT"],
            "TN": 4.0
            + 6.0
            + 25.0 * observation_model["particulate_nitrogen_factors"]["X_I"]
            + 40.0 * observation_model["particulate_nitrogen_factors"]["X_S"]
            + 80.0 * observation_model["particulate_nitrogen_factors"]["X_H"]
            + 10.0 * observation_model["particulate_nitrogen_factors"]["X_AUT"],
            "TP": 2.0
            + 25.0 * observation_model["particulate_phosphorus_factors"]["X_I"]
            + 40.0 * observation_model["particulate_phosphorus_factors"]["X_S"]
            + 80.0 * observation_model["particulate_phosphorus_factors"]["X_H"]
            + 10.0 * observation_model["particulate_phosphorus_factors"]["X_AUT"],
            "NH4_N": 4.0,
            "NO3_N": 6.0,
            "PO4_P": 2.0,
            "Alkalinity": 120.0,
        }

        expected_vector = np.array([expected[name] for name in measured_output_columns], dtype=float)
        np.testing.assert_allclose(observed, expected_vector)

    def test_single_operating_point_solves_to_small_residual(self) -> None:
        params = load_asm1_simulation_params()
        state_columns = list(params["state_columns"])
        midpoint_state = np.array(
            [
                np.mean(params["influent_state_ranges"][column_name])
                for column_name in state_columns
            ],
            dtype=float,
        )

        solution, diagnostics = solve_asm1_cstr_steady_state(
            influent_state=midpoint_state,
            hrt_hours=24.0,
            aeration=1.5,
            model_params=params,
        )

        self.assertTrue(diagnostics["success"])
        self.assertLess(diagnostics["residual_max"], 1e-5)
        self.assertTrue((solution >= 0.0).all())

    def test_generate_asm1_dataset_responds_to_aeration_and_hrt(self) -> None:
        baseline_params = load_asm1_simulation_params()
        matrix_bundle = get_asm1_matrices(baseline_params)
        state_columns = list(matrix_bundle["state_columns"])
        state_index = {name: position for position, name in enumerate(state_columns)}
        midpoint_state = np.array(
            [
                np.mean(baseline_params["influent_state_ranges"][column_name])
                for column_name in state_columns
            ],
            dtype=float,
        )

        low_aeration_state, _ = solve_asm1_cstr_steady_state(
            influent_state=midpoint_state,
            hrt_hours=24.0,
            aeration=1.0,
            model_params=baseline_params,
        )
        high_aeration_state, _ = solve_asm1_cstr_steady_state(
            influent_state=midpoint_state,
            hrt_hours=24.0,
            aeration=2.25,
            model_params=baseline_params,
        )

        low_measured = _compute_measured_outputs(
            low_aeration_state,
            matrix_bundle,
        )
        high_measured = _compute_measured_outputs(
            high_aeration_state,
            matrix_bundle,
        )

        self.assertGreater(high_aeration_state[state_index["S_O2"]], low_aeration_state[state_index["S_O2"]])
        self.assertLess(high_measured["NH4_N"], low_measured["NH4_N"])

        low_hrt_state, _ = solve_asm1_cstr_steady_state(
            influent_state=midpoint_state,
            hrt_hours=12.0,
            aeration=1.5,
            model_params=baseline_params,
        )
        high_hrt_state, _ = solve_asm1_cstr_steady_state(
            influent_state=midpoint_state,
            hrt_hours=48.0,
            aeration=1.5,
            model_params=baseline_params,
        )
        low_hrt_measured = _compute_measured_outputs(
            low_hrt_state,
            matrix_bundle,
        )
        high_hrt_measured = _compute_measured_outputs(
            high_hrt_state,
            matrix_bundle,
        )

        self.assertLess(high_hrt_measured["COD"], low_hrt_measured["COD"])
        self.assertLess(high_hrt_measured["NH4_N"], low_hrt_measured["NH4_N"])

    def test_save_simulation_artifacts_uses_asm1_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            paths_config = {
                "simulation_data_pattern": "data/{simulation_name}/data_{date_time}.csv",
                "simulation_metadata_pattern": "data/{simulation_name}/metadata_{date_time}.json",
            }
            dataset = pd.DataFrame({"example": [1.0, 2.0]})
            metadata = {
                "simulation_name": "asm1_simulation",
                "dependent_columns": ["y"],
                "independent_columns": ["x"],
                "identifier_columns": [],
                "ignored_columns": [],
                "dataset_file": None,
            }

            dataset_path, metadata_path, persisted_metadata = save_simulation_artifacts(
                dataset,
                metadata,
                "asm1_simulation",
                repo_root=temp_dir,
                timestamp="20260326_120000",
                paths_config=paths_config,
            )

            self.assertTrue(dataset_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(
                persisted_metadata["dataset_file"],
                "data/asm1_simulation/data_20260326_120000.csv",
            )

            with metadata_path.open("r", encoding="utf-8") as handle:
                stored_metadata = json.load(handle)

            self.assertEqual(stored_metadata["dataset_file"], persisted_metadata["dataset_file"])

    def test_run_asm1_simulation_without_artifacts_keeps_paths_empty(self) -> None:
        result = run_asm1_simulation(save_artifacts=False, n_samples=8, random_seed=11)

        self.assertEqual(result["dataset"].shape[0], 8)
        self.assertEqual(result["dataset"].shape[1], 32)
        self.assertEqual(result["petersen_matrix"].shape, (7, 11))
        self.assertEqual(result["composition_matrix"].shape, (8, 11))
        np.testing.assert_allclose(result["composite_matrix"], result["composition_matrix"])
        self.assertIsNone(result["artifact_paths"]["dataset_csv"])
        self.assertIsNone(result["artifact_paths"]["metadata_json"])


if __name__ == "__main__":
    unittest.main()