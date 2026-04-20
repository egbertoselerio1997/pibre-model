"""Minimal end-to-end tests for the icsor_coupled_qp model."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

from src.models.ml.icsor import build_icsor_design_frame
from src.models.ml.icsor_coupled_qp import (
    _resolve_coupled_qp_settings,
    _solve_chat_update,
    _solve_gamma_update,
    load_icsor_coupled_qp_params,
    predict_icsor_coupled_qp_model,
    run_icsor_coupled_qp_pipeline,
    train_icsor_coupled_qp_model,
)
from src.models.simulation.asm2d_tsn_simulation import generate_asm2d_tsn_dataset
from src.utils.io import save_pickle_file
from src.utils.process import build_icsor_supervised_dataset, make_train_test_split


def _compute_a_matrix(petersen_matrix: np.ndarray) -> np.ndarray:
    constraint_basis = null_space(petersen_matrix)
    a_matrix = constraint_basis.T
    a_matrix = np.round(a_matrix, 5)
    a_matrix[np.abs(a_matrix) < 1e-10] = 0.0

    for row_index in range(a_matrix.shape[0]):
        non_zero_entries = a_matrix[row_index, a_matrix[row_index, :] != 0]
        if len(non_zero_entries) > 0:
            a_matrix[row_index, :] = a_matrix[row_index, :] / non_zero_entries[0]

    return a_matrix


def _compute_invariant_max_abs(
    predictions: np.ndarray,
    constraint_reference: np.ndarray,
    a_matrix: np.ndarray,
) -> float:
    residuals = (np.asarray(predictions, dtype=float) - np.asarray(constraint_reference, dtype=float)) @ a_matrix.T
    if residuals.size == 0:
        return 0.0
    return float(np.max(np.abs(residuals)))


def _tiny_params() -> dict[str, object]:
    params = copy.deepcopy(load_icsor_coupled_qp_params())
    params["hyperparameters"]["random_seed"] = 11
    params["training_defaults"].update(
        {
            "training_method": "recursive_qp",
            "objective": "coupled_qp",
            "max_outer_iterations": 2,
            "n_restarts": 1,
            "objective_tolerance": 1e-6,
            "parameter_tolerance": 1e-5,
            "conditioning_max": 1e6,
            "osqp_max_iter": 2000,
            "osqp_polish": False,
            "highs_max_iter": 2000,
            "parallel_workers": 2,
            "gamma_abs_bound": 0.3,
        }
    )
    return params


class IcsorCoupledQpModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm2d_tsn_dataset(n_samples=8, random_seed=31)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix)
        cls.icsor_dataset = build_icsor_supervised_dataset(cls.dataset, cls.metadata, cls.composition_matrix)

    def test_run_pipeline_returns_raw_affine_projected_reports(self) -> None:
        params = _tiny_params()
        dataset_splits = make_train_test_split(self.icsor_dataset, test_fraction=0.2, random_seed=11)

        result = run_icsor_coupled_qp_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            measured_output_columns=list(self.metadata["measured_output_columns"]),
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        aggregate_metrics = result["test_report"]["aggregate_metrics"]
        self.assertEqual(list(aggregate_metrics["prediction_type"]), ["raw", "affine", "projected"])
        self.assertIn("report_metadata", result["test_report"])
        self.assertIn("projection_stage_summary", result["test_report"])
        self.assertIn("projection_diagnostics", result["test_report"])

        projected_fractional = result["test_report"]["projected_fractional_predictions"].to_numpy(dtype=float)
        self.assertGreaterEqual(float(projected_fractional.min()), -1e-8)
        projected_constraint_max_abs = _compute_invariant_max_abs(
            projected_fractional,
            dataset_splits.test.constraint_reference.to_numpy(dtype=float),
            self.a_matrix,
        )
        self.assertLessEqual(
            projected_constraint_max_abs,
            float(params["training_defaults"]["constraint_tolerance"]) + 1e-8,
        )

        gamma_matrix = np.asarray(result["model_bundle"]["Gamma_matrix"], dtype=float)
        gamma_abs_bound = float(params["training_defaults"]["gamma_abs_bound"])
        np.testing.assert_allclose(np.diag(gamma_matrix), np.zeros(gamma_matrix.shape[0]), atol=1e-10)
        self.assertLessEqual(float(np.max(np.abs(gamma_matrix))), gamma_abs_bound + 1e-7)

        self.assertIsNone(result["artifact_paths"]["model_bundle"])
        self.assertIsNone(result["artifact_paths"]["metrics"])
        self.assertIsNone(result["artifact_paths"]["optuna"])

    def test_predict_roundtrip_from_saved_bundle(self) -> None:
        params = _tiny_params()
        dataset_splits = make_train_test_split(self.icsor_dataset, test_fraction=0.2, random_seed=11)
        result = run_icsor_coupled_qp_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            measured_output_columns=list(self.metadata["measured_output_columns"]),
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            model_path = Path(temp_dir_name) / "icsor_coupled_qp_model.pkl"
            save_pickle_file(model_path, result["model_bundle"])
            prediction_result = predict_icsor_coupled_qp_model(
                self.dataset.iloc[:6].copy(),
                model_path,
                metadata=self.metadata,
                composition_matrix=self.composition_matrix,
            )

        expected_output_dim = len(self.metadata["state_columns"])
        self.assertEqual(prediction_result["raw_predictions"].shape, (6, expected_output_dim))
        self.assertEqual(prediction_result["projected_predictions"].shape, (6, expected_output_dim))
        self.assertIn("projection_stage_summary", prediction_result)
        self.assertGreaterEqual(
            float(prediction_result["projected_fractional_predictions"].to_numpy(dtype=float).min()),
            -1e-8,
        )
        projected_constraint_max_abs = _compute_invariant_max_abs(
            prediction_result["projected_fractional_predictions"].to_numpy(dtype=float),
            self.icsor_dataset.constraint_reference.iloc[:6].to_numpy(dtype=float),
            self.a_matrix,
        )
        self.assertLessEqual(
            projected_constraint_max_abs,
            float(params["training_defaults"]["constraint_tolerance"]) + 1e-8,
        )

    def test_training_enforces_gamma_admissibility_and_conditioning(self) -> None:
        params = _tiny_params()
        training_result = train_icsor_coupled_qp_model(
            {
                "features": self.icsor_dataset.features,
                "targets": self.icsor_dataset.targets,
                "constraint_reference": self.icsor_dataset.constraint_reference,
            },
            params["training_defaults"],
            A_matrix=self.a_matrix,
            composition_matrix=self.composition_matrix,
            training_options={"show_progress": False},
        )

        gamma_matrix = np.asarray(training_result["Gamma_matrix"], dtype=float)
        conditioning = float(training_result["best_restart_summary"]["conditioning"])
        conditioning_max = float(params["training_defaults"]["conditioning_max"])
        fitted_predictions = training_result["fitted_predictions"].to_numpy(dtype=float)

        np.testing.assert_allclose(np.diag(gamma_matrix), np.zeros(gamma_matrix.shape[0]), atol=1e-10)
        self.assertLessEqual(float(np.max(np.abs(gamma_matrix))), float(params["training_defaults"]["gamma_abs_bound"]) + 1e-7)
        self.assertLessEqual(conditioning, conditioning_max + 1e-6)
        self.assertGreaterEqual(
            float(fitted_predictions.min()),
            -float(params["training_defaults"]["nonnegativity_tolerance"]) - 1e-8,
        )

        chat_update_history = list(training_result["training_diagnostics"]["chat_update_history"])
        gamma_update_history = list(training_result["training_diagnostics"]["gamma_update_history"])
        self.assertTrue(chat_update_history)
        self.assertTrue(gamma_update_history)

        chat_statuses = {
            status_name
            for entry in chat_update_history
            for status_name in dict(entry.get("status_counts", {})).keys()
        }
        gamma_statuses = {
            status_name
            for entry in gamma_update_history
            for status_name in dict(entry.get("status_counts", {})).keys()
        }
        chat_has_solved_status = any("solved" in str(status_name).lower() for status_name in chat_statuses)
        chat_all_rows_screened = all(int(entry.get("active_qp_count", 0)) == 0 for entry in chat_update_history)
        self.assertTrue(chat_has_solved_status or chat_all_rows_screened)
        self.assertTrue(any("solved" in str(status_name).lower() for status_name in gamma_statuses))

    def test_chat_update_screening_all_interior_skips_osqp_rows(self) -> None:
        target_matrix = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.5, 0.3],
            ],
            dtype=float,
        )
        influent_matrix = np.zeros_like(target_matrix)
        invariant_matrix = np.zeros((0, target_matrix.shape[1]), dtype=float)
        coupled_matrix = np.eye(target_matrix.shape[1], dtype=float)
        driver_matrix = np.zeros_like(target_matrix)
        settings = _resolve_coupled_qp_settings(
            {
                "enable_c_hat_unconstrained_screening": True,
                "osqp_polish": False,
            }
        )

        fitted_predictions, chat_metadata = _solve_chat_update(
            target_matrix,
            influent_matrix,
            invariant_matrix,
            coupled_matrix,
            driver_matrix,
            settings,
        )

        np.testing.assert_allclose(fitted_predictions, 0.5 * target_matrix, atol=1e-7, rtol=1e-7)
        self.assertEqual(int(chat_metadata["interior_count"]), target_matrix.shape[0])
        self.assertEqual(int(chat_metadata["active_qp_count"]), 0)
        self.assertAlmostEqual(float(chat_metadata["interior_fraction"]), 1.0, places=9)
        self.assertEqual(dict(chat_metadata["status_counts"]), {})
        self.assertEqual(float(chat_metadata["mean_iterations"]), 0.0)
        self.assertEqual(int(chat_metadata["max_iterations"]), 0)

    def test_chat_update_screening_mixed_rows_uses_osqp_fallback(self) -> None:
        target_matrix = np.array(
            [
                [0.8, 0.4],
                [-1.2, 0.2],
                [0.1, -0.5],
            ],
            dtype=float,
        )
        influent_matrix = np.zeros_like(target_matrix)
        invariant_matrix = np.zeros((0, target_matrix.shape[1]), dtype=float)
        coupled_matrix = np.eye(target_matrix.shape[1], dtype=float)
        driver_matrix = np.zeros_like(target_matrix)
        settings = _resolve_coupled_qp_settings(
            {
                "enable_c_hat_unconstrained_screening": True,
                "osqp_polish": False,
            }
        )

        fitted_predictions, chat_metadata = _solve_chat_update(
            target_matrix,
            influent_matrix,
            invariant_matrix,
            coupled_matrix,
            driver_matrix,
            settings,
        )

        interior_count = int(chat_metadata["interior_count"])
        active_qp_count = int(chat_metadata["active_qp_count"])
        self.assertGreater(interior_count, 0)
        self.assertGreater(active_qp_count, 0)
        self.assertEqual(interior_count + active_qp_count, target_matrix.shape[0])
        status_counts = dict(chat_metadata["status_counts"])
        self.assertTrue(status_counts)
        self.assertTrue(any("solved" in str(status_name).lower() for status_name in status_counts.keys()))
        self.assertGreaterEqual(
            float(np.min(fitted_predictions)),
            -float(settings["nonnegativity_tolerance"]) - 1e-8,
        )

    def test_gamma_update_warm_start_tracks_usage_counts(self) -> None:
        fitted_predictions = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.3, 0.6, 0.1],
                [0.4, 0.4, 0.2],
                [0.2, 0.2, 0.6],
            ],
            dtype=float,
        )
        driver_matrix = np.zeros_like(fitted_predictions)
        initial_gamma = np.zeros((fitted_predictions.shape[1], fitted_predictions.shape[1]), dtype=float)
        settings = _resolve_coupled_qp_settings(
            {
                "osqp_polish": False,
                "enable_training_warm_start": True,
                "enable_gamma_warm_start": True,
            }
        )

        gamma_matrix, gamma_metadata = _solve_gamma_update(
            fitted_predictions,
            driver_matrix,
            settings,
            initial_gamma=initial_gamma,
        )

        self.assertEqual(int(gamma_metadata["warm_start_used_count"]), fitted_predictions.shape[1])
        self.assertEqual(int(gamma_metadata["warm_start_skipped_invalid_count"]), 0)
        self.assertTrue(any("solved" in str(status_name).lower() for status_name in gamma_metadata["status_counts"].keys()))
        np.testing.assert_allclose(np.diag(gamma_matrix), np.zeros(gamma_matrix.shape[0]), atol=1e-10)

    def test_chat_update_warm_start_tracks_active_row_usage(self) -> None:
        target_matrix = np.array(
            [
                [0.8, 0.4],
                [-1.2, 0.2],
                [0.1, -0.5],
            ],
            dtype=float,
        )
        influent_matrix = np.zeros_like(target_matrix)
        invariant_matrix = np.zeros((0, target_matrix.shape[1]), dtype=float)
        coupled_matrix = np.eye(target_matrix.shape[1], dtype=float)
        driver_matrix = np.zeros_like(target_matrix)
        warm_start_matrix = np.zeros_like(target_matrix)
        settings = _resolve_coupled_qp_settings(
            {
                "enable_c_hat_unconstrained_screening": True,
                "enable_training_warm_start": True,
                "enable_c_hat_warm_start": True,
                "osqp_polish": False,
            }
        )

        _, chat_metadata = _solve_chat_update(
            target_matrix,
            influent_matrix,
            invariant_matrix,
            coupled_matrix,
            driver_matrix,
            settings,
            warm_start_matrix=warm_start_matrix,
        )

        active_qp_count = int(chat_metadata["active_qp_count"])
        self.assertGreater(active_qp_count, 0)
        self.assertEqual(int(chat_metadata["warm_start_used_count"]), active_qp_count)
        self.assertEqual(int(chat_metadata["warm_start_skipped_invalid_count"]), 0)

    def test_default_params_enable_adam_lasso_training(self) -> None:
        params = load_icsor_coupled_qp_params()
        training_defaults = dict(params["training_defaults"])
        self.assertEqual(str(training_defaults["training_method"]), "adam_lasso")
        self.assertEqual(str(training_defaults["objective"]), "adam_lasso")

    def test_training_with_adam_lasso_returns_mode_aware_diagnostics(self) -> None:
        params = _tiny_params()
        params["training_defaults"].update(
            {
                "training_method": "adam_lasso",
                "objective": "adam_lasso",
                "adam_epochs": 20,
                "adam_learning_rate": 0.01,
                "adam_log_interval": 5,
                "lasso_lambda_B": 1e-4,
                "lasso_lambda_gamma": 1e-4,
            }
        )

        training_result = train_icsor_coupled_qp_model(
            {
                "features": self.icsor_dataset.features,
                "targets": self.icsor_dataset.targets,
                "constraint_reference": self.icsor_dataset.constraint_reference,
            },
            params["training_defaults"],
            A_matrix=self.a_matrix,
            composition_matrix=self.composition_matrix,
            training_options={"show_progress": False},
        )

        self.assertEqual(str(training_result["training_diagnostics"]["training_method"]), "adam_lasso")
        self.assertFalse(list(training_result["training_diagnostics"]["gamma_update_history"]))
        self.assertFalse(list(training_result["training_diagnostics"]["chat_update_history"]))
        adam_history = dict(training_result["training_diagnostics"]["adam_training_history"])
        self.assertEqual(len(list(adam_history["objective"])), 20)
        self.assertEqual(str(adam_history["returned_fitted_prediction_source"]), "exact_c_hat_qp")
        self.assertIn("final_chat_update", adam_history)
        self.assertEqual(training_result["B_matrix"].shape[0], self.icsor_dataset.targets.shape[1])
        self.assertEqual(training_result["Gamma_matrix"].shape[0], self.icsor_dataset.targets.shape[1])
        np.testing.assert_allclose(
            np.diag(np.asarray(training_result["Gamma_matrix"], dtype=float)),
            np.zeros(self.icsor_dataset.targets.shape[1]),
            atol=1e-10,
        )

    def test_training_with_adam_lasso_returns_exact_final_chat_solution(self) -> None:
        params = _tiny_params()
        params["training_defaults"].update(
            {
                "training_method": "adam_lasso",
                "objective": "adam_lasso",
                "adam_epochs": 3,
                "adam_learning_rate": 0.01,
                "adam_log_interval": 1,
                "lasso_lambda_B": 1e-4,
                "lasso_lambda_gamma": 1e-4,
            }
        )

        training_dataset = {
            "features": self.icsor_dataset.features,
            "targets": self.icsor_dataset.targets,
            "constraint_reference": self.icsor_dataset.constraint_reference,
        }
        training_result = train_icsor_coupled_qp_model(
            training_dataset,
            params["training_defaults"],
            A_matrix=self.a_matrix,
            composition_matrix=self.composition_matrix,
            training_options={"show_progress": False},
        )

        settings = _resolve_coupled_qp_settings(params["training_defaults"])
        design_frame, _ = build_icsor_design_frame(
            training_dataset["features"],
            constraint_columns=list(training_dataset["constraint_reference"].columns),
            include_bias_term=bool(settings["include_bias_term"]),
        )
        design_matrix = design_frame.to_numpy(dtype=float)
        target_matrix = training_dataset["targets"].to_numpy(dtype=float)
        influent_matrix = training_dataset["constraint_reference"].to_numpy(dtype=float)
        fitted_predictions = training_result["fitted_predictions"].to_numpy(dtype=float)
        gamma_matrix = np.asarray(training_result["Gamma_matrix"], dtype=float)
        driver_matrix = design_matrix @ np.asarray(training_result["B_matrix"], dtype=float).T

        exact_fitted_predictions, _ = _solve_chat_update(
            target_matrix,
            influent_matrix,
            self.a_matrix,
            np.eye(gamma_matrix.shape[0], dtype=float) - gamma_matrix,
            driver_matrix,
            settings,
            warm_start_matrix=fitted_predictions,
        )

        max_abs_gap = float(np.max(np.abs(fitted_predictions - exact_fitted_predictions)))
        self.assertLessEqual(max_abs_gap, 3e-2)


if __name__ == "__main__":
    unittest.main()
