"""Minimal end-to-end tests for the COBRE measured-space model."""

from __future__ import annotations

import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

from src.models.ml.cobre import predict_cobre_model, project_to_mass_balance, run_cobre_pipeline, train_cobre_model
from src.models.simulation.asm1_simulation import generate_asm1_dataset
from src.utils.metrics import summarize_mass_balance_residuals
from src.utils.process import build_measured_supervised_dataset, make_train_test_split
from src.utils.io import save_pickle_file
from src.utils.train import get_training_device, resolve_torch_adam_options


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


class CobreModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm1_dataset(n_samples=36, random_seed=17)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)

    def test_build_measured_supervised_dataset_uses_measured_space(self) -> None:
        supervised_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )

        expected_feature_columns = list(self.metadata["operational_columns"]) + [
            f"In_{column_name}" for column_name in self.metadata["measured_output_columns"]
        ]
        expected_target_columns = [f"Out_{column_name}" for column_name in self.metadata["measured_output_columns"]]

        self.assertEqual(list(supervised_dataset.features.columns), expected_feature_columns)
        self.assertEqual(list(supervised_dataset.targets.columns), expected_target_columns)
        self.assertEqual(list(supervised_dataset.constraint_reference.columns), self.metadata["measured_output_columns"])

    def test_project_to_mass_balance_enforces_constraints(self) -> None:
        supervised_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        target_values = supervised_dataset.targets.to_numpy(dtype=float)
        reference_values = supervised_dataset.constraint_reference.to_numpy(dtype=float)
        raw_predictions = target_values + 0.25

        projected_predictions = project_to_mass_balance(raw_predictions, reference_values, self.a_matrix)
        summary = summarize_mass_balance_residuals(projected_predictions, reference_values, self.a_matrix)

        self.assertLess(summary["constraint_max_abs"], 1e-8)
        self.assertLess(summary["constraint_mean_l2"], 1e-8)

    def test_resolve_torch_adam_options_disables_foreach_on_directml_by_default(self) -> None:
        self.assertEqual(resolve_torch_adam_options(device_label="directml"), {"foreach": False})
        self.assertEqual(resolve_torch_adam_options(device_label="cuda"), {})
        self.assertEqual(resolve_torch_adam_options(device_label="directml", foreach=True), {"foreach": True})

    def test_run_cobre_pipeline_returns_metrics_and_zero_projected_residuals(self) -> None:
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
        runtime_params = cast(dict[str, Any], params["runtime"])
        self.assertIsInstance(runtime_params, dict)
        _, device_label = get_training_device(prefer_directml=bool(runtime_params["prefer_directml"]))
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            result = run_cobre_pipeline(
                dataset_splits.train,
                dataset_splits.test,
                self.a_matrix,
                model_params=params,
                show_progress=False,
                persist_artifacts=False,
            )

        if device_label == "directml":
            self.assertFalse(
                any("lerp.Scalar_out" in str(warning.message) for warning in caught_warnings),
                "COBRE DirectML training should avoid Adam CPU fallback warnings.",
            )

        aggregate_metrics = result["test_report"]["aggregate_metrics"]
        self.assertEqual(list(aggregate_metrics["prediction_type"]), ["raw", "projected"])
        projected_row = aggregate_metrics.loc[aggregate_metrics["prediction_type"] == "projected"].iloc[0]
        self.assertLess(float(projected_row["constraint_max_abs"]), 5e-4)
        self.assertLess(float(projected_row["constraint_mean_l2"]), 5e-4)

    def test_predict_cobre_model_roundtrip_from_saved_bundle(self) -> None:
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
        result = run_cobre_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            model_path = Path(temp_dir_name) / "cobre_model.pkl"
            save_pickle_file(model_path, result["model_bundle"])
            prediction_result = predict_cobre_model(
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
        self.assertLess(summary["constraint_max_abs"], 5e-4)

    @patch("src.models.ml.cobre.create_progress_bar")
    def test_train_cobre_model_enables_progress_by_default(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
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

        train_cobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            cast(dict[str, float], params["training_defaults"]),
            A_matrix=self.a_matrix,
            training_options={
                "epochs": 2,
                "batch_size": 8,
                "random_seed": 11,
                "log_interval": 1,
                "prefer_directml": True,
                "adam_foreach": None,
            },
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(progress_factory.call_args.kwargs["enabled"])

    @patch("src.models.ml.cobre.create_progress_bar")
    def test_train_cobre_model_supports_progress_opt_out(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
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

        train_cobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            cast(dict[str, float], params["training_defaults"]),
            A_matrix=self.a_matrix,
            training_options={
                "epochs": 2,
                "batch_size": 8,
                "random_seed": 11,
                "log_interval": 1,
                "prefer_directml": True,
                "adam_foreach": None,
                "show_progress": False,
            },
        )

        self.assertTrue(progress_factory.called)
        self.assertFalse(progress_factory.call_args.kwargs["enabled"])

    def _tiny_params(self) -> dict[str, object]:
        params: dict[str, object] = copy.deepcopy(
            {
                "hyperparameters": {
                    "random_seed": 11,
                    "batch_size": 16,
                    "scale_features": True,
                    "scale_targets": False,
                    "log_interval": 5,
                    "training_epochs": 12,
                },
                "runtime": {
                    "prefer_directml": True,
                    "adam_foreach": None,
                },
                "training_defaults": {
                    "learning_rate": 0.01,
                    "lambda_l1": 0.0001,
                    "weight_decay": 0.000001,
                    "clip_max_norm": 1.0,
                    "bilinear_init_scale": 0.01,
                },
                "search_space": {
                    "learning_rate": {"type": "float", "low": 0.005, "high": 0.02, "log": True},
                    "lambda_l1": {"type": "float", "low": 0.00001, "high": 0.001, "log": True},
                    "weight_decay": {"type": "float", "low": 0.00000001, "high": 0.0001, "log": True},
                    "clip_max_norm": {"type": "float", "low": 0.5, "high": 2.0, "log": False},
                    "bilinear_init_scale": {"type": "float", "low": 0.001, "high": 0.02, "log": True},
                    "batch_size": {"type": "int", "low": 8, "high": 24, "log": False},
                },
                "pruner": {
                    "type": "median",
                    "n_startup_trials": 1,
                    "n_warmup_steps": 2,
                    "interval_steps": 1,
                },
                "artifact_options": {
                    "persist_model": True,
                    "persist_metrics": True,
                    "persist_optuna": True,
                },
            }
        )
        return params


if __name__ == "__main__":
    unittest.main()