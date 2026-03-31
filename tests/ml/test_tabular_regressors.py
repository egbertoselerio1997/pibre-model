"""Minimal end-to-end tests for the requested measured-space tabular regressors."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

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
from src.models.simulation.asm2d_tcn_simulation import generate_asm2d_tcn_dataset
from src.utils.io import save_pickle_file
from src.utils.metrics import summarize_mass_balance_residuals
from src.utils.process import build_fractional_input_measured_output_dataset, make_train_test_split


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


def _build_tiny_params(base_params: dict[str, object], *, iteration_key: str | None) -> dict[str, object]:
    params = copy.deepcopy(base_params)
    params["hyperparameters"]["random_seed"] = 11
    params["artifact_options"] = {
        "persist_model": True,
        "persist_metrics": True,
        "persist_optuna": True,
    }

    if iteration_key is not None and iteration_key in params["training_defaults"]:
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
        dataset, metadata, matrix_bundle = generate_asm2d_tcn_dataset(n_samples=12, random_seed=19)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)
        cls.model_specs = [
            {
                "name": "xgboost_regressor",
                "load_params": load_xgboost_regressor_params,
                "run_pipeline": run_xgboost_regressor_pipeline,
                "predict_model": predict_xgboost_regressor_model,
                "iteration_key": "n_estimators",
            },
            {
                "name": "lightgbm_regressor",
                "load_params": load_lightgbm_regressor_params,
                "run_pipeline": run_lightgbm_regressor_pipeline,
                "predict_model": predict_lightgbm_regressor_model,
                "iteration_key": "n_estimators",
            },
            {
                "name": "catboost_regressor",
                "load_params": load_catboost_regressor_params,
                "run_pipeline": run_catboost_regressor_pipeline,
                "predict_model": predict_catboost_regressor_model,
                "iteration_key": "iterations",
            },
            {
                "name": "adaboost_regressor",
                "load_params": load_adaboost_regressor_params,
                "run_pipeline": run_adaboost_regressor_pipeline,
                "predict_model": predict_adaboost_regressor_model,
                "iteration_key": "n_estimators",
            },
            {
                "name": "random_forest_regressor",
                "load_params": load_random_forest_regressor_params,
                "run_pipeline": run_random_forest_regressor_pipeline,
                "predict_model": predict_random_forest_regressor_model,
                "iteration_key": "n_estimators",
            },
            {
                "name": "svr_regressor",
                "load_params": load_svr_regressor_params,
                "run_pipeline": run_svr_regressor_pipeline,
                "predict_model": predict_svr_regressor_model,
                "iteration_key": None,
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
                params = _build_tiny_params(spec["load_params"](), iteration_key=spec["iteration_key"])
                result = spec["run_pipeline"](
                    dataset_splits.train,
                    dataset_splits.test,
                    self.a_matrix,
                    model_params=params,
                    show_progress=False,
                    persist_artifacts=False,
                )

                aggregate_metrics = result["test_report"]["aggregate_metrics"]
                self.assertEqual(list(aggregate_metrics["prediction_type"]), ["raw", "projected"])
                self.assertIn("report_metadata", result["test_report"])
                self.assertIn("diagnostic_summary", result["test_report"])
                self.assertIn("projection_diagnostics", result["test_report"])
                self.assertEqual(
                    result["model_bundle"]["feature_space"],
                    "fractional_input",
                )
                diagnostic_summary = result["test_report"]["diagnostic_summary"]
                projected_constraint_row = diagnostic_summary.loc[
                    (diagnostic_summary["diagnostic_name"] == "measured_constraint_residual")
                    & (diagnostic_summary["prediction_type"] == "projected")
                ].iloc[0]
                self.assertLess(float(projected_constraint_row["constraint_max_abs"]), 1e-7)
                self.assertLess(float(projected_constraint_row["constraint_mean_l2"]), 1e-7)
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
                self.assertEqual(prediction_result["projected_predictions"].shape, (6, expected_output_dim))
                summary = summarize_mass_balance_residuals(
                    prediction_result["projected_predictions"].to_numpy(dtype=float),
                    prediction_result["constraint_reference"].to_numpy(dtype=float),
                    self.a_matrix,
                )
                self.assertLess(summary["constraint_max_abs"], 1e-7)

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