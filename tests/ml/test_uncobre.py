"""Minimal end-to-end tests for the UNCOBRE measured-space model."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

from src.models.ml.uncobre import (
    predict_uncobre_model,
    run_uncobre_pipeline,
    train_uncobre_model,
)
from src.models.simulation.asm1_simulation import generate_asm1_dataset
from src.utils.io import save_pickle_file
from src.utils.metrics import summarize_mass_balance_residuals
from src.utils.process import build_measured_supervised_dataset, make_train_test_split


def _compute_a_matrix(petersen_matrix: np.ndarray, composition_matrix: np.ndarray) -> np.ndarray:
    macroscopic_stoichiometric_matrix = petersen_matrix @ composition_matrix.T
    constraint_basis = null_space(macroscopic_stoichiometric_matrix)
    a_matrix = constraint_basis.T
    a_matrix = np.round(a_matrix, 5)
    a_matrix[np.abs(a_matrix) < 1e-10] = 0.0

    for row_index in range(a_matrix.shape[0]):
        non_zero_entries = a_matrix[row_index, a_matrix[row_index, :] != 0]
        if len(non_zero_entries) > 0:
            a_matrix[row_index, :] = a_matrix[row_index, :] / non_zero_entries[0]

    return a_matrix


class UncobreModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm1_dataset(n_samples=36, random_seed=31)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)

    def test_run_pipeline_returns_raw_and_projected_metrics(self) -> None:
        params = self._tiny_params()
        measured_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            measured_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_uncobre_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        aggregate_metrics = result["test_report"]["aggregate_metrics"]
        self.assertEqual(list(aggregate_metrics["prediction_type"]), ["raw", "projected"])
        raw_row = aggregate_metrics.loc[aggregate_metrics["prediction_type"] == "raw"].iloc[0]
        projected_row = aggregate_metrics.loc[aggregate_metrics["prediction_type"] == "projected"].iloc[0]
        self.assertGreater(float(raw_row["constraint_mean_l2"]), 1e-9)
        self.assertLess(float(projected_row["constraint_mean_l2"]), float(raw_row["constraint_mean_l2"]))

    def test_predict_roundtrip_from_saved_bundle(self) -> None:
        params = self._tiny_params()
        measured_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            measured_dataset,
            test_fraction=0.2,
            random_seed=11,
        )
        result = run_uncobre_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            model_path = Path(temp_dir_name) / "uncobre_model.pkl"
            save_pickle_file(model_path, result["model_bundle"])
            prediction_result = predict_uncobre_model(
                self.dataset.iloc[:8].copy(),
                model_path,
                metadata=self.metadata,
                composition_matrix=self.composition_matrix,
            )

        expected_output_dim = len(self.metadata["measured_output_columns"])
        self.assertEqual(prediction_result["projected_predictions"].shape, (8, expected_output_dim))
        summary = summarize_mass_balance_residuals(
            prediction_result["projected_predictions"].to_numpy(dtype=float),
            prediction_result["constraint_reference"].to_numpy(dtype=float),
            self.a_matrix,
        )
        self.assertLess(summary["constraint_max_abs"], 5e-7)

    def test_ols_configuration_branch_runs(self) -> None:
        params = self._tiny_params()
        params["training_defaults"]["regression_mode"] = "ols"
        measured_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            measured_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_uncobre_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        self.assertEqual(result["best_hyperparameters"]["regression_mode"], "ols")

    @patch("src.models.ml.uncobre.create_progress_bar")
    def test_train_enables_progress_by_default(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        measured_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            measured_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        train_uncobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
            },
            self._tiny_params()["training_defaults"],
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(progress_factory.call_args.kwargs["enabled"])

    @patch("src.models.ml.uncobre.create_progress_bar")
    def test_train_supports_progress_opt_out(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        measured_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            measured_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        train_uncobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
            },
            self._tiny_params()["training_defaults"],
            training_options={"show_progress": False},
        )

        self.assertTrue(progress_factory.called)
        self.assertFalse(progress_factory.call_args.kwargs["enabled"])

    def _tiny_params(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "hyperparameters": {
                    "random_seed": 11,
                    "scale_features": True,
                    "scale_targets": False,
                },
                "training_defaults": {
                    "regression_mode": "ridge",
                    "ridge_alpha": 0.1,
                    "fit_intercept": True,
                    "include_bias": False,
                    "interaction_only": False,
                    "random_state": 11,
                },
                "search_space": {
                    "regression_mode": {"type": "categorical", "choices": ["ols", "ridge"]},
                    "ridge_alpha": {"type": "float", "low": 0.0001, "high": 10.0, "log": True},
                    "fit_intercept": {"type": "categorical", "choices": [True, False]},
                    "include_bias": {"type": "categorical", "choices": [False, True]},
                },
                "artifact_options": {
                    "persist_model": True,
                    "persist_metrics": True,
                    "persist_optuna": True,
                },
            }
        )


if __name__ == "__main__":
    unittest.main()
