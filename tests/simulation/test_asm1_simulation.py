"""Minimal tests for the asm1_simulation model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.simulation.asm1_simulation import generate_asm1_dataset, run_asm1_simulation
from src.utils.simulation import save_simulation_artifacts


class Asm1SimulationTests(unittest.TestCase):
    def test_generate_asm1_dataset_matches_metadata_contract(self) -> None:
        dataset, metadata, composite_matrix = generate_asm1_dataset(n_samples=12, random_seed=7)

        self.assertEqual(dataset.shape, (12, 12))
        self.assertEqual(composite_matrix.shape, (8, 5))
        self.assertEqual(metadata["simulation_name"], "asm1_simulation")
        self.assertEqual(metadata["n_samples"], 12)
        self.assertIsNone(metadata["dataset_file"])

        expected_independent = [
            "HRT",
            "Aeration",
            "In_Total_COD",
            "In_Total_N",
            "In_O2",
            "In_NO3",
            "In_Alkalinity",
        ]
        expected_dependent = [
            "Out_Total_COD",
            "Out_Total_N",
            "Out_O2",
            "Out_NO3",
            "Out_Alkalinity",
        ]

        self.assertEqual(metadata["independent_columns"], expected_independent)
        self.assertEqual(metadata["dependent_columns"], expected_dependent)
        self.assertEqual(list(dataset.columns), expected_independent + expected_dependent)

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
        self.assertIsNone(result["artifact_paths"]["dataset_csv"])
        self.assertIsNone(result["artifact_paths"]["metadata_json"])


if __name__ == "__main__":
    unittest.main()