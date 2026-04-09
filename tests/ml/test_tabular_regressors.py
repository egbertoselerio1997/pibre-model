"""Minimal end-to-end tests for the requested measured-space tabular regressors."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

from src.models.ml.ann_deep_regressor import (
    load_ann_deep_regressor_params,
    predict_ann_deep_regressor_model,
    run_ann_deep_regressor_pipeline,
)
from src.models.ml.ann_medium_regressor import (
    load_ann_medium_regressor_params,
    predict_ann_medium_regressor_model,
    run_ann_medium_regressor_pipeline,
)
from src.models.ml.ann_shallow_regressor import (
    load_ann_shallow_regressor_params,
    predict_ann_shallow_regressor_model,
    run_ann_shallow_regressor_pipeline,
)
from src.models.ml.adaboost_regressor import (
    load_adaboost_regressor_params,
    predict_adaboost_regressor_model,
    run_adaboost_regressor_pipeline,
)
from src.models.ml.catboost_regressor import (
    load_catboost_regressor_params,
    predict_catboost_regressor_model,
    run_catboost_regressor_pipeline,
)
from src.models.ml.lightgbm_regressor import (
    load_lightgbm_regressor_params,
    predict_lightgbm_regressor_model,
    run_lightgbm_regressor_pipeline,
)
from src.models.ml.knn_regressor import load_knn_regressor_params, predict_knn_regressor_model, run_knn_regressor_pipeline
from src.models.ml.pls_regressor import load_pls_regressor_params, predict_pls_regressor_model, run_pls_regressor_pipeline
from src.models.ml.random_forest_regressor import (
    load_random_forest_regressor_params,
    predict_random_forest_regressor_model,
    run_random_forest_regressor_pipeline,
)
from src.models.ml.svr_regressor import load_svr_regressor_params, predict_svr_regressor_model, run_svr_regressor_pipeline
from src.models.ml.xgboost_regressor import (
    load_xgboost_regressor_params,
    predict_xgboost_regressor_model,
    run_xgboost_regressor_pipeline,
)
from src.models.simulation.asm2d_tsn_simulation import generate_asm2d_tsn_dataset
from src.utils.io import save_pickle_file
from src.utils.metrics import summarize_mass_balance_residuals
from src.utils.process import build_fractional_input_measured_output_dataset, has_active_projection, make_train_test_split


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


def _build_tiny_params(
    base_params: dict[str, object],
    *,
    iteration_key: str | None,
    fixed_iteration_key: str | None = None,
) -> dict[str, object]:
    params = copy.deepcopy(base_params)
    params["hyperparameters"]["random_seed"] = 11
    params["artifact_options"] = {
        "persist_model": True,
        "persist_metrics": True,
        "persist_optuna": True,
    }

    if "batch_size" in params["training_defaults"]:
        params["training_defaults"]["batch_size"] = 8
    if "hidden_layer_sizes" in params["training_defaults"]:
        params["training_defaults"]["learning_rate_init"] = 0.01
        params["training_defaults"]["tol"] = 0.01

    # Keep KNN feasible for tiny fixture splits where train rows can be < default neighbors.
    if "n_neighbors" in params["training_defaults"]:
        max_neighbors = min(5, int(params["training_defaults"]["n_neighbors"]))
        params["training_defaults"]["n_neighbors"] = max_neighbors
        if "n_neighbors" in params["search_space"]:
            params["search_space"]["n_neighbors"] = {
                "type": "int",
                "low": 1,
                "high": max_neighbors,
                "log": False,
            }

    if fixed_iteration_key is not None and fixed_iteration_key in params["training_defaults"]:
        if fixed_iteration_key == "max_iter":
            params["training_defaults"][fixed_iteration_key] = 50
        else:
            params["training_defaults"][fixed_iteration_key] = 12

    if iteration_key is not None and iteration_key in params["training_defaults"]:
        if iteration_key == "max_iter":
            params["training_defaults"][iteration_key] = 50
            params["search_space"][iteration_key] = {
                "type": "int",
                "low": 20,
                "high": 50,
                "log": False,
            }
        else:
            params["training_defaults"][iteration_key] = 12
            params["search_space"][iteration_key] = {
                "type": "int",
                "low": 8,
                "high": 12,
                "log": False,
            }

    return params


class TabularRegressorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm2d_tsn_dataset(n_samples=12, random_seed=19)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)
        cls.projection_active = has_active_projection(cls.a_matrix)
        cls.model_specs = [
            {
                "name": "knn_regressor",
                "load_params": load_knn_regressor_params,
                "run_pipeline": run_knn_regressor_pipeline,
                "predict_model": predict_knn_regressor_model,
                "iteration_key": None,
            },
            {
                "name": "pls_regressor",
                "load_params": load_pls_regressor_params,
                "run_pipeline": run_pls_regressor_pipeline,
                "predict_model": predict_pls_regressor_model,
                "iteration_key": "max_iter",
            },
            {
                "name": "ann_shallow_regressor",
                "load_params": load_ann_shallow_regressor_params,
                "run_pipeline": run_ann_shallow_regressor_pipeline,
                "predict_model": predict_ann_shallow_regressor_model,
                "iteration_key": None,
                "fixed_iteration_key": "max_iter",
            },
            {
                "name": "ann_medium_regressor",
                "load_params": load_ann_medium_regressor_params,
                "run_pipeline": run_ann_medium_regressor_pipeline,
                "predict_model": predict_ann_medium_regressor_model,
                "iteration_key": None,
                "fixed_iteration_key": "max_iter",
            },
            {
                "name": "ann_deep_regressor",
                "load_params": load_ann_deep_regressor_params,
                "run_pipeline": run_ann_deep_regressor_pipeline,
                "predict_model": predict_ann_deep_regressor_model,
                "iteration_key": None,
                "fixed_iteration_key": "max_iter",
            },
            {
                "name": "xgboost_regressor",
                "load_params": load_xgboost_regressor_params,
                "run_pipeline": run_xgboost_regressor_pipeline,
                "predict_model": predict_xgboost_regressor_model,
                "iteration_key": "n_estimators",
                "fixed_iteration_key": None,
            },
            {
                "name": "lightgbm_regressor",
                "load_params": load_lightgbm_regressor_params,
                "run_pipeline": run_lightgbm_regressor_pipeline,
                "predict_model": predict_lightgbm_regressor_model,
                "iteration_key": "n_estimators",
                "fixed_iteration_key": None,
            },
            {
                "name": "catboost_regressor",
                "load_params": load_catboost_regressor_params,
                "run_pipeline": run_catboost_regressor_pipeline,
                "predict_model": predict_catboost_regressor_model,
                "iteration_key": "iterations",
                "fixed_iteration_key": None,
            },
            {
                "name": "adaboost_regressor",
                "load_params": load_adaboost_regressor_params,
                "run_pipeline": run_adaboost_regressor_pipeline,
                "predict_model": predict_adaboost_regressor_model,
                "iteration_key": "n_estimators",
                "fixed_iteration_key": None,
            },
            {
                "name": "random_forest_regressor",
                "load_params": load_random_forest_regressor_params,
                "run_pipeline": run_random_forest_regressor_pipeline,
                "predict_model": predict_random_forest_regressor_model,
                "iteration_key": "n_estimators",
                "fixed_iteration_key": None,
            },
            {
                "name": "svr_regressor",
                "load_params": load_svr_regressor_params,
                "run_pipeline": run_svr_regressor_pipeline,
                "predict_model": predict_svr_regressor_model,
                "iteration_key": None,
                "fixed_iteration_key": None,
            },
        ]

    def test_requested_regressors_pipeline_and_roundtrip(self) -> None:
        benchmark_dataset = build_fractional_input_measured_output_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            benchmark_dataset,
            test_fraction=0.2,
            random_seed=11,
        )

        for spec in self.model_specs:
            with self.subTest(model=spec["name"]):
                params = _build_tiny_params(
                    spec["load_params"](),
                    iteration_key=spec["iteration_key"],
                    fixed_iteration_key=spec.get("fixed_iteration_key"),
                )
                result = spec["run_pipeline"](
                    dataset_splits.train,
                    dataset_splits.test,
                    self.a_matrix,
                    model_params=params,
                    show_progress=False,
                    persist_artifacts=False,
                )

                aggregate_metrics = result["test_report"]["aggregate_metrics"]
                expected_prediction_types = ["raw", "projected"] if self.projection_active else ["raw"]
                self.assertEqual(list(aggregate_metrics["prediction_type"]), expected_prediction_types)
                self.assertIn("report_metadata", result["test_report"])
                report_metadata = result["test_report"]["report_metadata"].iloc[0]
                self.assertEqual(bool(report_metadata["projection_active"]), self.projection_active)
                self.assertEqual(
                    str(report_metadata["constraint_status"]),
                    "active" if self.projection_active else "inactive_trivial_null_space",
                )
                self.assertEqual(
                    result["model_bundle"]["feature_space"],
                    "fractional_input",
                )
                self.assertEqual(bool(result["model_bundle"]["projection_active"]), self.projection_active)
                if self.projection_active:
                    self.assertIn("diagnostic_summary", result["test_report"])
                    self.assertIn("projection_diagnostics", result["test_report"])
                    diagnostic_summary = result["test_report"]["diagnostic_summary"]
                    projected_constraint_row = diagnostic_summary.loc[
                        (diagnostic_summary["diagnostic_name"] == "measured_constraint_residual")
                        & (diagnostic_summary["prediction_type"] == "projected")
                    ].iloc[0]
                    self.assertLess(float(projected_constraint_row["constraint_max_abs"]), 1e-7)
                    self.assertLess(float(projected_constraint_row["constraint_mean_l2"]), 1e-7)
                else:
                    self.assertNotIn("diagnostic_summary", result["test_report"])
                    self.assertNotIn("projection_diagnostics", result["test_report"])
                    self.assertNotIn("constraint_residuals", result["test_report"])
                    self.assertNotIn("projected_predictions", result["test_report"])
                self.assertIsNone(result["artifact_paths"]["model_bundle"])
                self.assertIsNone(result["artifact_paths"]["metrics"])
                self.assertIsNone(result["artifact_paths"]["optuna"])

                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model_path = Path(temp_dir_name) / f"{spec['name']}.pkl"
                    save_pickle_file(model_path, result["model_bundle"])
                    prediction_result = spec["predict_model"](
                        self.dataset.iloc[:6].copy(),
                        model_path,
                        metadata=self.metadata,
                        composition_matrix=self.composition_matrix,
                    )

                expected_output_dim = len(self.metadata["measured_output_columns"])
                self.assertEqual(prediction_result["raw_predictions"].shape, (6, expected_output_dim))
                self.assertEqual(bool(prediction_result["projection_active"]), self.projection_active)
                if self.projection_active:
                    self.assertEqual(prediction_result["projected_predictions"].shape, (6, expected_output_dim))
                    summary = summarize_mass_balance_residuals(
                        prediction_result["projected_predictions"].to_numpy(dtype=float),
                        prediction_result["constraint_reference"].to_numpy(dtype=float),
                        self.a_matrix,
                    )
                    self.assertLess(summary["constraint_max_abs"], 1e-7)
                else:
                    self.assertNotIn("projected_predictions", prediction_result)

    @patch("src.utils.train.create_progress_bar")
    def test_tabular_pipeline_enables_progress_by_default(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        benchmark_dataset = build_fractional_input_measured_output_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            benchmark_dataset,
            test_fraction=0.2,
            random_seed=11,
        )
        params = _build_tiny_params(load_adaboost_regressor_params(), iteration_key="n_estimators")

        run_adaboost_regressor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            model_params=params,
            persist_artifacts=False,
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(any(call.kwargs["enabled"] for call in progress_factory.call_args_list))

    @patch("src.utils.train.create_progress_bar")
    def test_tabular_pipeline_supports_progress_opt_out(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        benchmark_dataset = build_fractional_input_measured_output_dataset(
            self.dataset,
            self.metadata,
            self.composition_matrix,
        )
        dataset_splits = make_train_test_split(
            benchmark_dataset,
            test_fraction=0.2,
            random_seed=11,
        )
        params = _build_tiny_params(load_adaboost_regressor_params(), iteration_key="n_estimators")

        run_adaboost_regressor_pipeline(
            dataset_splits.train,
            dataset_splits.test,
            self.a_matrix,
            model_params=params,
            show_progress=False,
            persist_artifacts=False,
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(all(not call.kwargs["enabled"] for call in progress_factory.call_args_list))


if __name__ == "__main__":
    unittest.main()
