"""Minimal end-to-end tests for the icsor projected OLS model."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from sklearn.metrics import r2_score

from src.models.ml.icsor import (
    build_icsor_design_frame,
    predict_icsor_model,
    run_icsor_pipeline,
    train_icsor_model,
)
from src.models.simulation.asm2d_tsn_simulation import generate_asm2d_tsn_dataset
from src.utils.io import save_pickle_file
from src.utils.metrics import summarize_mass_balance_residuals
from src.utils.process import (
    DatasetSplit,
    build_icsor_supervised_dataset,
    build_projection_operator,
    make_train_test_split,
    project_to_nonnegative_feasible_set,
)


def _compute_a_matrix(petersen_matrix: np.ndarray, composition_matrix: np.ndarray) -> np.ndarray:
    del composition_matrix
    constraint_basis = null_space(petersen_matrix)
    a_matrix = constraint_basis.T
    a_matrix = np.round(a_matrix, 5)
    a_matrix[np.abs(a_matrix) < 1e-10] = 0.0

    for row_index in range(a_matrix.shape[0]):
        non_zero_entries = a_matrix[row_index, a_matrix[row_index, :] != 0]
        if len(non_zero_entries) > 0:
            a_matrix[row_index, :] = a_matrix[row_index, :] / non_zero_entries[0]

    return a_matrix


class icsorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm2d_tsn_dataset(n_samples=12, random_seed=29)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)
        cls.icsor_dataset = build_icsor_supervised_dataset(cls.dataset, cls.metadata, cls.composition_matrix)

    def test_run_pipeline_returns_raw_and_projected_metrics(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_icsor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        aggregate_metrics = result["test_report"]["aggregate_metrics"]
        self.assertEqual(list(aggregate_metrics["prediction_type"]), ["raw", "affine", "projected"])
        self.assertIn("report_metadata", result["test_report"])
        self.assertIn("diagnostic_summary", result["test_report"])
        self.assertIn("projection_diagnostics", result["test_report"])
        self.assertIn("projection_stage_summary", result["test_report"])
        raw_metric_row = aggregate_metrics.loc[aggregate_metrics["prediction_type"] == "raw"].iloc[0]
        affine_metric_row = aggregate_metrics.loc[aggregate_metrics["prediction_type"] == "affine"].iloc[0]
        projected_metric_row = aggregate_metrics.loc[aggregate_metrics["prediction_type"] == "projected"].iloc[0]
        self.assertGreater(float(raw_metric_row["RMSE"]), 0.0)
        self.assertTrue(np.isfinite(float(affine_metric_row["RMSE"])))
        self.assertTrue(np.isfinite(float(projected_metric_row["RMSE"])))

        diagnostic_summary = result["test_report"]["diagnostic_summary"]
        raw_constraint_row = diagnostic_summary.loc[
            (diagnostic_summary["diagnostic_name"] == "fractional_constraint_residual")
            & (diagnostic_summary["prediction_type"] == "raw")
        ].iloc[0]
        affine_constraint_row = diagnostic_summary.loc[
            (diagnostic_summary["diagnostic_name"] == "fractional_constraint_residual")
            & (diagnostic_summary["prediction_type"] == "affine")
        ].iloc[0]
        projected_constraint_row = diagnostic_summary.loc[
            (diagnostic_summary["diagnostic_name"] == "fractional_constraint_residual")
            & (diagnostic_summary["prediction_type"] == "projected")
        ].iloc[0]
        self.assertGreater(float(raw_constraint_row["constraint_mean_l2"]), 1e-9)
        self.assertLess(float(affine_constraint_row["constraint_mean_l2"]), float(raw_constraint_row["constraint_mean_l2"]))
        self.assertLess(float(projected_constraint_row["constraint_mean_l2"]), float(raw_constraint_row["constraint_mean_l2"]))
        self.assertLess(float(affine_constraint_row["constraint_max_abs"]), 1e-6)
        self.assertLess(float(projected_constraint_row["constraint_max_abs"]), 1e-6)
        self.assertGreaterEqual(
            float(result["test_report"]["projection_diagnostics"]["projected_min_component"].min()),
            -1e-10,
        )

        raw_w_u = np.asarray(result["model_bundle"]["raw_coefficients"]["W_u"], dtype=float)
        raw_w_in = np.asarray(result["model_bundle"]["raw_coefficients"]["W_in"], dtype=float)
        raw_b = np.asarray(result["model_bundle"]["raw_coefficients"]["b"], dtype=float)
        effective_w_u = np.asarray(result["model_bundle"]["effective_coefficients"]["W_u"], dtype=float)
        effective_w_in = np.asarray(result["model_bundle"]["effective_coefficients"]["W_in"], dtype=float)
        effective_b = np.asarray(result["model_bundle"]["effective_coefficients"]["b"], dtype=float)
        np.testing.assert_allclose(effective_w_u, raw_w_u, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(effective_w_in, raw_w_in, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(effective_b, raw_b, atol=1e-10, rtol=1e-10)
        self.assertEqual(result["model_bundle"]["native_prediction_space"], "fractional_component")
        self.assertEqual(result["model_bundle"]["comparison_target_space"], "external_measured_output")

    def test_fractional_projected_per_target_r2_matches_recomputed_values(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_icsor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        true_fractional = dataset_splits.test.targets.astype(float)
        predicted_fractional = (
            result["test_report"]["projected_fractional_predictions"]
            .rename(columns=lambda name: str(name).removeprefix("ProjectedFractional_"))
            .rename(columns=lambda name: f"Out_{name}")
            .loc[:, true_fractional.columns]
            .astype(float)
        )
        reported_per_target = result["test_report"]["fractional_per_target_metrics"].set_index("target")

        self.assertIn("Out_X_S", reported_per_target.index)
        self.assertIn("Out_X_H", reported_per_target.index)

        for target_name in true_fractional.columns:
            recomputed_r2 = r2_score(
                true_fractional[target_name].to_numpy(dtype=float),
                predicted_fractional[target_name].to_numpy(dtype=float),
            )
            reported_r2 = float(reported_per_target.loc[target_name, "projected_R2"])
            self.assertAlmostEqual(recomputed_r2, reported_r2, places=12)

    def test_predict_roundtrip_from_saved_bundle(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )
        result = run_icsor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            model_path = Path(temp_dir_name) / "icsor_model.pkl"
            save_pickle_file(model_path, result["model_bundle"])
            prediction_result = predict_icsor_model(
                self.dataset.iloc[:8].copy(),
                model_path,
                metadata=self.metadata,
                composition_matrix=self.composition_matrix,
            )

        expected_output_dim = len(self.metadata["state_columns"])
        self.assertEqual(prediction_result["projected_predictions"].shape, (8, expected_output_dim))
        self.assertIn("affine_predictions", prediction_result)
        self.assertIn("projection_stage_diagnostics", prediction_result)
        self.assertIn("projection_stage_summary", prediction_result)
        self.assertIn("affine_core_prediction_standard_errors", prediction_result)
        self.assertEqual(prediction_result["affine_core_prediction_standard_errors"].shape, (8, expected_output_dim))
        summary = summarize_mass_balance_residuals(
            prediction_result["projected_fractional_predictions"].to_numpy(dtype=float),
            prediction_result["constraint_reference"].to_numpy(dtype=float),
            self.a_matrix,
        )
        self.assertLess(summary["constraint_max_abs"], 5e-7)
        self.assertGreaterEqual(
            float(prediction_result["projected_fractional_predictions"].to_numpy(dtype=float).min()),
            -1e-10,
        )

    def test_predict_rejects_mismatched_composition_source(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )
        result = run_icsor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            composition_source=self.metadata.get("composition_source"),
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        mismatched_metadata = dict(self.metadata)
        mismatched_metadata["composition_source"] = {"workbook_sha256": "mismatched_sha256"}

        with tempfile.TemporaryDirectory() as temp_dir_name:
            model_path = Path(temp_dir_name) / "icsor_model.pkl"
            save_pickle_file(model_path, result["model_bundle"])
            with self.assertRaisesRegex(ValueError, r"workbook_sha256"):
                predict_icsor_model(
                    self.dataset.iloc[:4].copy(),
                    model_path,
                    metadata=mismatched_metadata,
                    composition_matrix=self.composition_matrix,
                )

    def test_rank_deficient_training_uses_analytic_uncertainty_in_auto_mode(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_icsor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        coefficient_inference = result["model_bundle"]["coefficient_inference"]
        self.assertEqual(coefficient_inference["method"], "analytic")
        self.assertEqual(coefficient_inference["requested_method"], "auto")
        self.assertTrue(bool(coefficient_inference["rank_deficient"]))
        self.assertRegex(
            str(coefficient_inference["note"]).lower(),
            r"(rank|non-full-column-rank|not uniquely identifiable)",
        )
        self.assertIn("affine_core_prediction_standard_errors", result["test_report"])
        self.assertIn("prediction_uncertainty_summary", result["test_report"])
        prediction_metadata = result["test_report"]["uncertainty_metadata"].iloc[0].to_dict()
        self.assertEqual(prediction_metadata["method"], "analytic")
        self.assertEqual(prediction_metadata["note"], coefficient_inference["note"])
        self.assertEqual(prediction_metadata["prediction_target"], "raw_component_prediction")

        effective_uncertainty = result["model_bundle"]["effective_coefficient_uncertainty"]
        self.assertIn("W_u", effective_uncertainty)
        self.assertTrue(np.all(np.asarray(effective_uncertainty["W_u"]["standard_error"], dtype=float) >= 0.0))

    def test_rank_deficient_training_allows_forced_analytic_uncertainty(self) -> None:
        params = self._tiny_params(uncertainty_method="analytic")
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_icsor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        coefficient_inference = result["model_bundle"]["coefficient_inference"]
        self.assertEqual(coefficient_inference["method"], "analytic")
        self.assertTrue(bool(coefficient_inference["rank_deficient"]))
        self.assertIsInstance(coefficient_inference["note"], str)
        self.assertTrue(bool(coefficient_inference["note"]))
        self.assertRegex(
            str(coefficient_inference["note"]).lower(),
            r"(rank|non-full-column-rank|not uniquely identifiable)",
        )

        prediction_metadata = result["test_report"]["uncertainty_metadata"].iloc[0].to_dict()
        self.assertEqual(prediction_metadata["method"], "analytic")
        self.assertEqual(prediction_metadata["note"], coefficient_inference["note"])
        self.assertEqual(prediction_metadata["prediction_target"], "raw_component_prediction")
        self.assertIn("affine_core_prediction_standard_errors", result["test_report"])
        self.assertIn("prediction_uncertainty_summary", result["test_report"])

    def test_full_rank_training_uses_analytic_uncertainty(self) -> None:
        synthetic_train, synthetic_test, a_matrix, composition_matrix = self._make_full_rank_synthetic_splits()
        params = self._tiny_params()

        result = run_icsor_pipeline(
            synthetic_train,
            synthetic_test,
            a_matrix,
            composition_matrix=composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        coefficient_inference = result["model_bundle"]["coefficient_inference"]
        self.assertEqual(coefficient_inference["method"], "analytic")
        self.assertFalse(bool(coefficient_inference["rank_deficient"]))
        self.assertGreater(int(coefficient_inference["degrees_of_freedom"]), 0)

        uncertainty_frame = result["test_report"]["affine_core_prediction_standard_errors"]
        self.assertEqual(list(uncertainty_frame.columns), ["AffineCoreSE_Out_S1"])
        self.assertTrue((uncertainty_frame.to_numpy(dtype=float) >= 0.0).all())

        affine_predictions = result["test_report"]["affine_predictions"]
        ci_lower = result["test_report"]["affine_core_prediction_confidence_interval_lower"]
        ci_upper = result["test_report"]["affine_core_prediction_confidence_interval_upper"]
        projected_values = affine_predictions.rename(columns=lambda value: str(value).removeprefix("Affine_"))
        lower_values = ci_lower.rename(columns=lambda value: str(value).removeprefix("AffineCoreCI95Lower_"))
        upper_values = ci_upper.rename(columns=lambda value: str(value).removeprefix("AffineCoreCI95Upper_"))
        self.assertTrue((lower_values.to_numpy(dtype=float) <= projected_values.to_numpy(dtype=float)).all())
        self.assertTrue((projected_values.to_numpy(dtype=float) <= upper_values.to_numpy(dtype=float)).all())

    def test_bootstrap_uncertainty_method_is_rejected(self) -> None:
        params = self._tiny_params(uncertainty_method="bootstrap")
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        with self.assertRaisesRegex(ValueError, r"analytic, auto, none"):
            run_icsor_pipeline(
                dataset_splits.train,
                dataset_splits.test,
                self.a_matrix,
                composition_matrix=self.composition_matrix,
                model_params=params,
                show_progress=False,
                persist_artifacts=False,
            )

    def test_nonnegative_projection_uses_highs_lp_when_affine_prediction_is_negative(self) -> None:
        a_matrix = np.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        composition_matrix = np.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        raw_predictions = np.asarray([[-2.0, 22.0, 5.0]], dtype=float)
        constraint_reference = np.asarray([[10.0, 10.0, 5.0]], dtype=float)
        projection_matrix = build_projection_operator(a_matrix)
        projection_complement = np.eye(projection_matrix.shape[0], dtype=float) - projection_matrix

        projection_result = project_to_nonnegative_feasible_set(
            raw_predictions,
            constraint_reference,
            a_matrix,
            composition_matrix,
            projection_operator=projection_matrix,
            projection_complement=projection_complement,
            projection_solver="highs",
            constraint_tolerance=1e-8,
            nonnegativity_tolerance=1e-10,
            measured_deviation_weight=1.0,
            component_deviation_weight=1.0,
            tradeoff_parameter=1.0,
            highs_presolve=True,
            highs_max_iter=10000,
            highs_verbose=False,
            highs_retry_without_presolve=True,
        )

        np.testing.assert_allclose(projection_result["affine_predictions"], raw_predictions, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(projection_result["projected_predictions"], [[0.0, 20.0, 5.0]], atol=1e-8, rtol=1e-8)
        self.assertFalse(bool(projection_result["raw_feasible_mask"][0]))
        self.assertFalse(bool(projection_result["affine_feasible_mask"][0]))
        self.assertTrue(bool(projection_result["lp_active_mask"][0]))
        self.assertEqual(str(projection_result["projection_stage"][0]), "lp_corrected")

    def test_nonnegative_projection_uses_lp_when_invariant_matrix_is_trivial(self) -> None:
        raw_predictions = np.asarray([[-2.0, 22.0, 5.0]], dtype=float)
        constraint_reference = np.asarray([[0.4, 0.5, 0.6]], dtype=float)
        composition_matrix = np.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        a_matrix = np.zeros((0, 3), dtype=float)

        projection_result = project_to_nonnegative_feasible_set(
            raw_predictions,
            constraint_reference,
            a_matrix,
            composition_matrix,
            projection_operator=np.zeros((3, 3), dtype=float),
            projection_complement=np.eye(3, dtype=float),
            projection_solver="highs",
            constraint_tolerance=1e-8,
            nonnegativity_tolerance=1e-10,
            measured_deviation_weight=1.0,
            component_deviation_weight=1.0,
            tradeoff_parameter=1.0,
            highs_presolve=True,
            highs_max_iter=10000,
            highs_verbose=False,
            highs_retry_without_presolve=True,
        )

        np.testing.assert_allclose(projection_result["affine_predictions"], raw_predictions, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(projection_result["projected_predictions"], [[0.0, 22.0, 5.0]], atol=1e-10, rtol=1e-10)
        self.assertTrue(bool(projection_result["lp_active_mask"][0]))
        self.assertEqual(str(projection_result["projection_stage"][0]), "lp_corrected")

    def test_projected_ols_matches_explicit_kronecker_solution(self) -> None:
        params = self._tiny_params(ols_backend="numpy_lstsq")
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=7,
        )
        train_split = dataset_splits.train

        training_result = train_icsor_model(
            {
                "features": train_split.features,
                "targets": train_split.targets,
                "constraint_reference": train_split.constraint_reference,
            },
            params["training_defaults"],
            A_matrix=self.a_matrix,
            composition_matrix=self.composition_matrix,
            training_options={"show_progress": False},
        )

        design_frame, _ = build_icsor_design_frame(
            train_split.features,
            list(train_split.constraint_reference.columns),
            include_bias_term=True,
        )
        explicit_raw, *_ = np.linalg.lstsq(
            design_frame.to_numpy(dtype=float),
            train_split.targets.to_numpy(dtype=float),
            rcond=None,
        )

        np.testing.assert_allclose(
            np.asarray(training_result["raw_parameter_matrix"], dtype=float),
            explicit_raw,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_ridge_training_uses_analytic_uncertainty_even_when_full_rank(self) -> None:
        synthetic_train, synthetic_test, a_matrix, composition_matrix = self._make_full_rank_synthetic_splits()
        params = self._tiny_params(
            affine_estimator="ridge",
            ridge_alpha=0.5,
            uncertainty_method="auto",
        )

        result = run_icsor_pipeline(
            synthetic_train,
            synthetic_test,
            a_matrix,
            composition_matrix=composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        coefficient_inference = result["model_bundle"]["coefficient_inference"]
        self.assertEqual(coefficient_inference["method"], "analytic")
        self.assertEqual(coefficient_inference["analytic_distribution"], "gaussian")
        self.assertEqual(coefficient_inference["requested_method"], "auto")
        self.assertEqual(coefficient_inference["affine_estimator"], "ridge")
        self.assertAlmostEqual(float(coefficient_inference["ridge_alpha"]), 0.5)
        self.assertGreater(float(coefficient_inference["degrees_of_freedom"]), 0.0)
        self.assertRegex(
            str(coefficient_inference["note"]).lower(),
            r"(gaussian|fixed-penalty|shrinkage)",
        )

        prediction_metadata = result["test_report"]["uncertainty_metadata"].iloc[0].to_dict()
        self.assertEqual(prediction_metadata["method"], "analytic")
        self.assertEqual(prediction_metadata["note"], coefficient_inference["note"])
        self.assertEqual(prediction_metadata["affine_estimator"], "ridge")
        self.assertAlmostEqual(float(prediction_metadata["ridge_alpha"]), 0.5)

    def test_ridge_shrinks_identifiable_parameter_norm_relative_to_ols(self) -> None:
        synthetic_train, synthetic_test, a_matrix, composition_matrix = self._make_full_rank_synthetic_splits()
        ols_params = self._tiny_params(affine_estimator="ols", uncertainty_method="analytic")
        ridge_params = self._tiny_params(
            affine_estimator="ridge",
            ridge_alpha=5.0,
            uncertainty_method="auto",
        )

        ols_result = run_icsor_pipeline(
            synthetic_train,
            synthetic_test,
            a_matrix,
            composition_matrix=composition_matrix,
            model_params=ols_params,
            show_progress=False,
            persist_artifacts=False,
        )
        ridge_result = run_icsor_pipeline(
            synthetic_train,
            synthetic_test,
            a_matrix,
            composition_matrix=composition_matrix,
            model_params=ridge_params,
            show_progress=False,
            persist_artifacts=False,
        )

        ols_norm = float(np.linalg.norm(np.asarray(ols_result["model_bundle"]["identifiable_parameter_matrix"], dtype=float)))
        ridge_norm = float(np.linalg.norm(np.asarray(ridge_result["model_bundle"]["identifiable_parameter_matrix"], dtype=float)))
        self.assertLess(ridge_norm, ols_norm)
        self.assertEqual(ridge_result["best_hyperparameters"]["affine_estimator"], "ridge")

    @patch("src.models.ml.icsor.persist_training_artifacts")
    def test_pipeline_forwards_optuna_summary_when_present(self, persist_artifacts_mock: MagicMock) -> None:
        persist_artifacts_mock.return_value = {
            "model_bundle": Path("results/icsor/model.pkl"),
            "metrics": Path("results/icsor/metrics.json"),
            "optuna": Path("results/icsor/optuna.json"),
        }
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )
        optuna_summary = {
            "best_value": 0.123,
            "best_trial_number": 0,
            "n_trials": 1,
            "best_params": {"ridge_alpha": 0.05},
            "trial_state_counts": {"complete": 1},
        }

        result = run_icsor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            optuna_summary=optuna_summary,
            show_progress=False,
            persist_artifacts=True,
        )

        self.assertTrue(persist_artifacts_mock.called)
        self.assertEqual(persist_artifacts_mock.call_args.kwargs["optuna_summary"], optuna_summary)
        self.assertEqual(result["artifact_paths"]["optuna"], Path("results/icsor/optuna.json"))

    @patch("src.models.ml.icsor.create_progress_bar")
    def test_train_enables_progress_by_default(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        train_icsor_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            self._tiny_params()["training_defaults"],
            A_matrix=self.a_matrix,
            composition_matrix=self.composition_matrix,
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(progress_factory.call_args.kwargs["enabled"])

    @patch("src.models.ml.icsor.create_progress_bar")
    def test_train_supports_progress_opt_out(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        dataset_splits = make_train_test_split(
            self.icsor_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        train_icsor_model(
            {
                "features": dataset_splits.train.features,
                "targets": dataset_splits.train.targets,
                "constraint_reference": dataset_splits.train.constraint_reference,
            },
            self._tiny_params()["training_defaults"],
            A_matrix=self.a_matrix,
            composition_matrix=self.composition_matrix,
            training_options={"show_progress": False},
        )

        self.assertTrue(progress_factory.called)
        self.assertFalse(progress_factory.call_args.kwargs["enabled"])

    def _tiny_params(
        self,
        *,
        affine_estimator: str = "ols",
        ols_backend: str = "numpy_lstsq",
        ridge_alpha: float = 0.001,
        uncertainty_method: str = "auto",
    ) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "hyperparameters": {
                    "random_seed": 11,
                    "scale_features": False,
                    "scale_targets": False,
                },
                "training_defaults": {
                    "objective": "projected_ridge" if affine_estimator == "ridge" else "projected_ols",
                    "solver": "multivariate_lstsq",
                    "affine_estimator": affine_estimator,
                    "ols_backend": ols_backend,
                    "ridge_alpha": ridge_alpha,
                    "include_bias_term": True,
                    "lstsq_rcond": None,
                        "projection_solver": "highs",
                        "constraint_tolerance": 1e-8,
                        "nonnegativity_tolerance": 1e-10,
                        "measured_deviation_weight": 1.0,
                        "component_deviation_weight": 1.0,
                        "tradeoff_parameter": 1.0,
                        "highs_presolve": True,
                        "highs_max_iter": 10000,
                        "highs_verbose": False,
                        "highs_retry_without_presolve": True,
                    "uncertainty_method": uncertainty_method,
                    "confidence_level": 0.95,
                },
                "artifact_options": {
                    "persist_model": True,
                    "persist_metrics": True,
                },
            }
        )

    def _make_full_rank_synthetic_splits(self) -> tuple[DatasetSplit, DatasetSplit, np.ndarray, np.ndarray]:
        random_generator = np.random.default_rng(17)
        row_count = 40
        features = pd.DataFrame(
            {
                "HRT": random_generator.uniform(2.0, 12.0, size=row_count),
                "In_S1": random_generator.uniform(0.5, 4.0, size=row_count),
            }
        )
        constraint_reference = features.loc[:, ["In_S1"]].rename(columns={"In_S1": "S1"})
        composition_matrix = np.asarray([[1.0]], dtype=float)
        a_matrix = np.zeros((0, 1), dtype=float)
        design_frame, _ = build_icsor_design_frame(
            features,
            list(constraint_reference.columns),
            include_bias_term=True,
        )
        coefficient_matrix = random_generator.normal(loc=0.0, scale=0.2, size=(design_frame.shape[1], 1))
        targets = pd.DataFrame(
            design_frame.to_numpy(dtype=float) @ coefficient_matrix + random_generator.normal(0.0, 0.01, size=(row_count, 1)),
            columns=["Out_S1"],
        )

        train_indices = features.index[:32]
        test_indices = features.index[32:]
        return (
            DatasetSplit(
                features=features.loc[train_indices].copy(),
                targets=targets.loc[train_indices].copy(),
                constraint_reference=constraint_reference.loc[train_indices].copy(),
            ),
            DatasetSplit(
                features=features.loc[test_indices].copy(),
                targets=targets.loc[test_indices].copy(),
                constraint_reference=constraint_reference.loc[test_indices].copy(),
            ),
            a_matrix,
            composition_matrix,
        )


if __name__ == "__main__":
    unittest.main()


