"""Minimal end-to-end tests for the COBRE projected OLS model."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

from src.models.ml.cobre import (
    build_cobre_design_frame,
    predict_cobre_model,
    run_cobre_pipeline,
    train_cobre_model,
)
from src.models.simulation.asm1_simulation import generate_asm1_dataset
from src.utils.io import save_pickle_file
from src.utils.metrics import summarize_mass_balance_residuals
from src.utils.process import build_measured_supervised_dataset, build_projection_operator, make_train_test_split


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
        dataset, metadata, matrix_bundle = generate_asm1_dataset(n_samples=36, random_seed=29)
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

        result = run_cobre_pipeline(
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

        projection_matrix = np.asarray(result["model_bundle"]["projection_matrix"], dtype=float)
        raw_w_in = np.asarray(result["model_bundle"]["raw_coefficients"]["W_in"], dtype=float)
        effective_w_in = np.asarray(result["model_bundle"]["effective_coefficients"]["W_in"], dtype=float)
        np.testing.assert_allclose(effective_w_in - raw_w_in, projection_matrix, atol=1e-10, rtol=1e-10)

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
        self.assertLess(summary["constraint_max_abs"], 5e-7)

    def test_projected_ols_matches_explicit_kronecker_solution(self) -> None:
        params = self._tiny_params(ols_backend="numpy_lstsq")
        measured_dataset = build_measured_supervised_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            measured_dataset,
            test_fraction=0.2,
            random_seed=7,
        )
        train_split = dataset_splits.train

        training_result = train_cobre_model(
            {
                "features": train_split.features,
                "targets": train_split.targets,
                "constraint_reference": train_split.constraint_reference,
            },
            params["training_defaults"],
            A_matrix=self.a_matrix,
            training_options={"show_progress": False},
        )

        design_frame, _ = build_cobre_design_frame(
            train_split.features,
            list(train_split.constraint_reference.columns),
            include_bias_term=True,
        )
        projection_matrix = build_projection_operator(self.a_matrix)
        projection_matrix = 0.5 * (projection_matrix + projection_matrix.T)
        projection_complement = np.eye(projection_matrix.shape[0], dtype=float) - projection_matrix

        y_tilde = train_split.targets.to_numpy(dtype=float) - train_split.constraint_reference.to_numpy(dtype=float) @ projection_matrix.T
        z_matrix = np.kron(design_frame.to_numpy(dtype=float), projection_complement)
        beta, *_ = np.linalg.lstsq(
            z_matrix,
            y_tilde.T.reshape(-1, order="F"),
            rcond=None,
        )
        explicit_raw = beta.reshape(
            projection_complement.shape[0],
            design_frame.shape[1],
            order="F",
        )
        explicit_projected = projection_complement @ explicit_raw

        np.testing.assert_allclose(
            np.asarray(training_result["raw_parameter_matrix"], dtype=float).T,
            explicit_projected,
            atol=1e-8,
            rtol=1e-8,
        )

    @patch("src.models.ml.cobre.get_training_device")
    def test_auto_backend_falls_back_to_numpy_when_directml_unavailable(self, get_training_device_mock: MagicMock) -> None:
        get_training_device_mock.return_value = (object(), "cpu")
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

        training_result = train_cobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            self._tiny_params(ols_backend="auto")["training_defaults"],
            A_matrix=self.a_matrix,
            training_options={"show_progress": False},
            runtime_options={"prefer_directml": True},
        )

        self.assertEqual(training_result["ols_metadata"]["requested_backend"], "auto")
        self.assertEqual(training_result["ols_metadata"]["backend_used"], "numpy_lstsq")
        self.assertEqual(training_result["ols_metadata"]["device_label"], "cpu")
        self.assertIn("DirectML device unavailable", str(training_result["ols_metadata"]["fallback_reason"]))

    @patch("src.models.ml.cobre._compute_ols_cross_products_with_torch")
    @patch("src.models.ml.cobre.get_training_device")
    def test_auto_backend_prefers_directml_when_available(
        self,
        get_training_device_mock: MagicMock,
        compute_cross_products_mock: MagicMock,
    ) -> None:
        get_training_device_mock.return_value = (object(), "directml")
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
        design_frame, _ = build_cobre_design_frame(
            dataset_splits.train.features,
            list(dataset_splits.train.constraint_reference.columns),
            include_bias_term=True,
        )
        projection_matrix = build_projection_operator(self.a_matrix)
        projection_matrix = 0.5 * (projection_matrix + projection_matrix.T)
        projection_complement = np.eye(projection_matrix.shape[0], dtype=float) - projection_matrix
        projected_targets = dataset_splits.train.targets.to_numpy(dtype=float) @ projection_complement.T
        design_values = design_frame.to_numpy(dtype=float)
        compute_cross_products_mock.return_value = (
            design_values.T @ design_values,
            design_values.T @ projected_targets,
            "float32",
        )

        training_result = train_cobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            self._tiny_params(ols_backend="auto")["training_defaults"],
            A_matrix=self.a_matrix,
            training_options={"show_progress": False},
            runtime_options={"prefer_directml": True},
        )

        self.assertEqual(training_result["ols_metadata"]["backend_used"], "directml_normal_equations")
        self.assertEqual(training_result["ols_metadata"]["device_label"], "directml")
        self.assertIsNone(training_result["ols_metadata"]["fallback_reason"])
        self.assertEqual(training_result["ols_metadata"]["matrix_multiplication_dtype"], "float32")

    @patch("src.models.ml.cobre.create_progress_bar")
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

        train_cobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            self._tiny_params()["training_defaults"],
            A_matrix=self.a_matrix,
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(progress_factory.call_args.kwargs["enabled"])

    @patch("src.models.ml.cobre.create_progress_bar")
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

        train_cobre_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            self._tiny_params()["training_defaults"],
            A_matrix=self.a_matrix,
            training_options={"show_progress": False},
        )

        self.assertTrue(progress_factory.called)
        self.assertFalse(progress_factory.call_args.kwargs["enabled"])

    def _tiny_params(self, *, ols_backend: str = "numpy_lstsq") -> dict[str, Any]:
        return copy.deepcopy(
            {
                "hyperparameters": {
                    "random_seed": 11,
                    "scale_features": False,
                    "scale_targets": False,
                },
                "runtime": {
                    "prefer_directml": True,
                    "adam_foreach": None,
                },
                "training_defaults": {
                    "objective": "projected_ols",
                    "solver": "multivariate_lstsq",
                    "ols_backend": ols_backend,
                    "include_bias_term": True,
                    "lstsq_rcond": None,
                },
                "artifact_options": {
                    "persist_model": True,
                    "persist_metrics": True,
                    "persist_optuna": False,
                },
            }
        )


if __name__ == "__main__":
    unittest.main()
