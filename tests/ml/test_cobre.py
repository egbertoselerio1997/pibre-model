"""Minimal end-to-end tests for the COBRE projected OLS model."""

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

from src.models.ml.cobre import (
    build_cobre_design_frame,
    predict_cobre_model,
    run_cobre_pipeline,
    train_cobre_model,
)
from src.models.simulation.asm2d_tcn_simulation import generate_asm2d_tcn_dataset
from src.utils.io import save_pickle_file
from src.utils.metrics import summarize_mass_balance_residuals
from src.utils.process import (
    DatasetSplit,
    build_cobre_supervised_dataset,
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


class CobreModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm2d_tcn_dataset(n_samples=12, random_seed=29)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)
        cls.cobre_dataset = build_cobre_supervised_dataset(cls.dataset, cls.metadata, cls.composition_matrix)

    def test_run_pipeline_returns_raw_and_projected_metrics(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.cobre_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_cobre_pipeline(
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

        projection_matrix = np.asarray(result["model_bundle"]["projection_matrix"], dtype=float)
        projection_complement = np.asarray(result["model_bundle"]["projection_complement"], dtype=float)
        collapse_operator = self.composition_matrix @ projection_complement
        pass_through_operator = self.composition_matrix @ projection_matrix
        raw_w_u = np.asarray(result["model_bundle"]["raw_coefficients"]["W_u"], dtype=float)
        raw_w_in = np.asarray(result["model_bundle"]["raw_coefficients"]["W_in"], dtype=float)
        raw_b = np.asarray(result["model_bundle"]["raw_coefficients"]["b"], dtype=float)
        effective_w_u = np.asarray(result["model_bundle"]["effective_coefficients"]["W_u"], dtype=float)
        effective_w_in = np.asarray(result["model_bundle"]["effective_coefficients"]["W_in"], dtype=float)
        effective_b = np.asarray(result["model_bundle"]["effective_coefficients"]["b"], dtype=float)
        np.testing.assert_allclose(effective_w_u, collapse_operator @ raw_w_u, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(
            effective_w_in,
            collapse_operator @ raw_w_in + pass_through_operator,
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(effective_b, collapse_operator @ raw_b, atol=1e-10, rtol=1e-10)

    def test_predict_roundtrip_from_saved_bundle(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.cobre_dataset,
            test_fraction=0.2,
            random_seed=11,
        )
        result = run_cobre_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
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

    def test_rank_deficient_training_uses_bootstrap_uncertainty_in_auto_mode(self) -> None:
        params = self._tiny_params()
        dataset_splits = make_train_test_split(
            self.cobre_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_cobre_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        coefficient_inference = result["model_bundle"]["coefficient_inference"]
        self.assertEqual(coefficient_inference["method"], "bootstrap")
        self.assertTrue(bool(coefficient_inference["rank_deficient"]))
        self.assertIn("affine_core_prediction_standard_errors", result["test_report"])
        self.assertIn("prediction_uncertainty_summary", result["test_report"])
        self.assertEqual(
            result["test_report"]["uncertainty_metadata"].iloc[0]["prediction_target"],
            "affine_core_measured_prediction",
        )

        effective_uncertainty = result["model_bundle"]["effective_coefficient_uncertainty"]
        self.assertIn("W_u", effective_uncertainty)
        self.assertTrue(np.all(np.asarray(effective_uncertainty["W_u"]["standard_error"], dtype=float) >= 0.0))

    def test_rank_deficient_training_allows_forced_analytic_uncertainty(self) -> None:
        params = self._tiny_params(uncertainty_method="analytic")
        dataset_splits = make_train_test_split(
            self.cobre_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        result = run_cobre_pipeline(
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
        self.assertEqual(prediction_metadata["prediction_target"], "affine_core_measured_prediction")
        self.assertIn("affine_core_prediction_standard_errors", result["test_report"])
        self.assertIn("prediction_uncertainty_summary", result["test_report"])

    def test_full_rank_training_uses_analytic_uncertainty(self) -> None:
        synthetic_train, synthetic_test, a_matrix, composition_matrix = self._make_full_rank_synthetic_splits()
        params = self._tiny_params(bootstrap_samples=16)

        result = run_cobre_pipeline(
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
        self.assertEqual(list(uncertainty_frame.columns), ["AffineCoreSE_Out_Y1", "AffineCoreSE_Out_Y2"])
        self.assertTrue((uncertainty_frame.to_numpy(dtype=float) >= 0.0).all())

        affine_predictions = result["test_report"]["affine_predictions"]
        ci_lower = result["test_report"]["affine_core_prediction_confidence_interval_lower"]
        ci_upper = result["test_report"]["affine_core_prediction_confidence_interval_upper"]
        projected_values = affine_predictions.rename(columns=lambda value: str(value).removeprefix("Affine_"))
        lower_values = ci_lower.rename(columns=lambda value: str(value).removeprefix("AffineCoreCI95Lower_"))
        upper_values = ci_upper.rename(columns=lambda value: str(value).removeprefix("AffineCoreCI95Upper_"))
        self.assertTrue((lower_values.to_numpy(dtype=float) <= projected_values.to_numpy(dtype=float)).all())
        self.assertTrue((projected_values.to_numpy(dtype=float) <= upper_values.to_numpy(dtype=float)).all())

    def test_nonnegative_projection_uses_osqp_when_affine_prediction_is_negative(self) -> None:
        a_matrix = np.asarray([[1.0, 1.0]], dtype=float)
        raw_predictions = np.asarray([[-1.0, 2.0]], dtype=float)
        constraint_reference = np.asarray([[1.0, 1.0]], dtype=float)
        projection_matrix = build_projection_operator(a_matrix)
        projection_complement = np.eye(projection_matrix.shape[0], dtype=float) - projection_matrix

        projection_result = project_to_nonnegative_feasible_set(
            raw_predictions,
            constraint_reference,
            a_matrix,
            projection_operator=projection_matrix,
            projection_complement=projection_complement,
            projection_solver="osqp",
            constraint_tolerance=1e-8,
            nonnegativity_tolerance=1e-10,
            osqp_eps_abs=1e-8,
            osqp_eps_rel=1e-8,
            osqp_max_iter=10000,
            osqp_polish=True,
            osqp_verbose=False,
            osqp_warm_start=True,
        )

        np.testing.assert_allclose(projection_result["affine_predictions"], [[-0.5, 2.5]], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(projection_result["projected_predictions"], [[0.0, 2.0]], atol=1e-8, rtol=1e-8)
        self.assertFalse(bool(projection_result["raw_feasible_mask"][0]))
        self.assertFalse(bool(projection_result["affine_feasible_mask"][0]))
        self.assertTrue(bool(projection_result["qp_active_mask"][0]))
        self.assertEqual(str(projection_result["projection_stage"][0]), "qp_corrected")

    def test_nonnegative_projection_clips_when_invariant_matrix_is_trivial(self) -> None:
        raw_predictions = np.asarray([[1.0, -0.2, 0.3]], dtype=float)
        constraint_reference = np.asarray([[0.4, 0.5, 0.6]], dtype=float)
        a_matrix = np.zeros((0, 3), dtype=float)

        projection_result = project_to_nonnegative_feasible_set(
            raw_predictions,
            constraint_reference,
            a_matrix,
            projection_operator=np.zeros((3, 3), dtype=float),
            projection_complement=np.eye(3, dtype=float),
            projection_solver="osqp",
            constraint_tolerance=1e-8,
            nonnegativity_tolerance=1e-10,
            osqp_eps_abs=1e-8,
            osqp_eps_rel=1e-8,
            osqp_max_iter=10000,
            osqp_polish=True,
            osqp_verbose=False,
            osqp_warm_start=True,
        )

        np.testing.assert_allclose(projection_result["affine_predictions"], raw_predictions, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(projection_result["projected_predictions"], [[1.0, 0.0, 0.3]], atol=1e-10, rtol=1e-10)
        self.assertEqual(str(projection_result["projection_stage"][0]), "orthant_clip")

    def test_projected_ols_matches_explicit_kronecker_solution(self) -> None:
        params = self._tiny_params(ols_backend="numpy_lstsq")
        dataset_splits = make_train_test_split(
            self.cobre_dataset,
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
            composition_matrix=self.composition_matrix,
            training_options={"show_progress": False},
        )

        design_frame, _ = build_cobre_design_frame(
            train_split.features,
            list(train_split.constraint_reference.columns),
            include_bias_term=True,
        )
        projection_matrix = build_projection_operator(self.a_matrix)
        projection_complement = np.eye(projection_matrix.shape[0], dtype=float) - projection_matrix

        collapse_operator = self.composition_matrix @ projection_complement
        pass_through_operator = self.composition_matrix @ projection_matrix
        y_tilde = train_split.targets.to_numpy(dtype=float) - train_split.constraint_reference.to_numpy(dtype=float) @ pass_through_operator.T
        z_matrix = np.kron(collapse_operator, design_frame.to_numpy(dtype=float))
        beta, *_ = np.linalg.lstsq(
            z_matrix,
            y_tilde.reshape(-1, order="F"),
            rcond=None,
        )
        explicit_raw = beta.reshape(
            design_frame.shape[1],
            projection_complement.shape[0],
            order="F",
        )

        np.testing.assert_allclose(
            np.asarray(training_result["raw_parameter_matrix"], dtype=float),
            explicit_raw,
            atol=1e-8,
            rtol=1e-8,
        )

    @patch("src.models.ml.cobre.create_progress_bar")
    def test_train_enables_progress_by_default(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        dataset_splits = make_train_test_split(
            self.cobre_dataset,
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
            composition_matrix=self.composition_matrix,
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(progress_factory.call_args.kwargs["enabled"])

    @patch("src.models.ml.cobre.create_progress_bar")
    def test_train_supports_progress_opt_out(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        dataset_splits = make_train_test_split(
            self.cobre_dataset,
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
            composition_matrix=self.composition_matrix,
            training_options={"show_progress": False},
        )

        self.assertTrue(progress_factory.called)
        self.assertFalse(progress_factory.call_args.kwargs["enabled"])

    def _tiny_params(
        self,
        *,
        ols_backend: str = "numpy_lstsq",
        bootstrap_samples: int = 32,
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
                    "objective": "projected_ols",
                    "solver": "multivariate_lstsq",
                    "ols_backend": ols_backend,
                    "include_bias_term": True,
                    "lstsq_rcond": None,
                    "projection_solver": "osqp",
                    "constraint_tolerance": 1e-8,
                    "nonnegativity_tolerance": 1e-10,
                    "osqp_eps_abs": 1e-8,
                    "osqp_eps_rel": 1e-8,
                    "osqp_max_iter": 10000,
                    "osqp_polish": True,
                    "osqp_verbose": False,
                    "osqp_warm_start": True,
                    "uncertainty_method": uncertainty_method,
                    "confidence_level": 0.95,
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_random_seed": 11,
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
        composition_matrix = np.asarray([[1.0], [1.75]], dtype=float)
        a_matrix = np.zeros((0, 1), dtype=float)
        design_frame, _ = build_cobre_design_frame(
            features,
            list(constraint_reference.columns),
            include_bias_term=True,
        )
        coefficient_matrix = random_generator.normal(loc=0.0, scale=0.2, size=(design_frame.shape[1], 2))
        targets = pd.DataFrame(
            design_frame.to_numpy(dtype=float) @ coefficient_matrix + random_generator.normal(0.0, 0.01, size=(row_count, 2)),
            columns=["Out_Y1", "Out_Y2"],
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
