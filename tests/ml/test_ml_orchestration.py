"""Tests for notebook-managed ML orchestration helpers."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

from src.models.ml.adaboost_regressor import build_adaboost_regressor_model, load_adaboost_regressor_params
from src.models.simulation.asm2d_tsn_simulation import generate_asm2d_tsn_dataset
from src.utils.process import (
    apply_train_test_split_indices,
    build_icsor_supervised_dataset,
    build_fractional_input_measured_output_dataset,
    make_train_test_split,
    make_train_test_split_indices,
    sample_dataset_fraction,
    sample_dataset_split_indices,
)
from src.utils.simulation import load_params_config
from src.utils.train import tune_icsor_hyperparameters, tune_tabular_regressor_hyperparameters


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


def _compute_icsor_a_matrix(petersen_matrix: np.ndarray) -> np.ndarray:
    constraint_basis = null_space(petersen_matrix)
    a_matrix = constraint_basis.T
    a_matrix = np.round(a_matrix, 5)
    a_matrix[np.abs(a_matrix) < 1e-10] = 0.0

    for row_index in range(a_matrix.shape[0]):
        non_zero_entries = a_matrix[row_index, a_matrix[row_index, :] != 0]
        if len(non_zero_entries) > 0:
            a_matrix[row_index, :] = a_matrix[row_index, :] / non_zero_entries[0]

    return a_matrix


def _build_tiny_adaboost_params() -> dict[str, object]:
    params = copy.deepcopy(load_adaboost_regressor_params())
    params["hyperparameters"]["random_seed"] = 11
    params["training_defaults"]["n_estimators"] = 12
    params["search_space"] = {
        "n_estimators": {"type": "int", "low": 8, "high": 12, "log": False},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "loss": {"type": "categorical", "choices": ["linear", "square"]},
    }
    return params


def _build_tiny_icsor_params() -> dict[str, object]:
    return {
        "hyperparameters": {
            "random_seed": 11,
            "scale_features": False,
            "scale_targets": False,
        },
        "training_defaults": {
            "objective": "projected_ridge",
            "solver": "multivariate_lstsq",
            "affine_estimator": "ridge",
            "ols_backend": "numpy_lstsq",
            "ridge_alpha": 0.001,
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
            "uncertainty_method": "analytic",
            "confidence_level": 0.95,
        },
        "search_space": {
            "ridge_alpha": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        },
        "artifact_options": {
            "persist_model": False,
            "persist_metrics": False,
            "persist_optuna": False,
        },
    }


CLASSICAL_MODEL_NAMES = [
    "xgboost_regressor",
    "lightgbm_regressor",
    "catboost_regressor",
    "adaboost_regressor",
    "random_forest_regressor",
    "svr_regressor",
    "knn_regressor",
    "pls_regressor",
    "ann_shallow_regressor",
    "ann_medium_regressor",
    "ann_deep_regressor",
]

FIXED_HYPERPARAMETER_KEYS = {
    "xgboost_regressor": {"objective", "random_state", "n_jobs", "verbosity"},
    "lightgbm_regressor": {"objective", "random_state", "verbosity", "n_jobs"},
    "catboost_regressor": {"loss_function", "bootstrap_type", "random_seed", "verbose", "allow_writing_files"},
    "adaboost_regressor": {"random_state"},
    "random_forest_regressor": {"random_state", "n_jobs"},
    "svr_regressor": set(),
    "knn_regressor": {"n_jobs"},
    "pls_regressor": {"scale", "copy"},
    "ann_shallow_regressor": {"hidden_layer_sizes", "solver", "max_iter", "early_stopping", "random_state", "verbose"},
    "ann_medium_regressor": {"hidden_layer_sizes", "solver", "max_iter", "early_stopping", "random_state", "verbose"},
    "ann_deep_regressor": {"hidden_layer_sizes", "solver", "max_iter", "early_stopping", "random_state", "verbose"},
}


class MlOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm2d_tsn_dataset(n_samples=12, random_seed=23)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)
        cls.icsor_a_matrix = _compute_icsor_a_matrix(cls.petersen_matrix)
        cls.classical_benchmark_dataset = build_fractional_input_measured_output_dataset(
            dataset,
            metadata,
            cls.composition_matrix,
        )
        cls.icsor_dataset = build_icsor_supervised_dataset(dataset, metadata, cls.composition_matrix)

    def test_expanded_simulation_schema_is_excluded_from_ml_features_and_targets(self) -> None:
        state_columns = list(self.metadata["state_columns"])
        measured_output_columns = list(self.metadata["measured_output_columns"])
        operational_columns = list(self.metadata["operational_columns"])

        self.assertIn("In_COD", self.dataset.columns)
        self.assertIn("Out_S_A", self.dataset.columns)

        expected_feature_columns = operational_columns + [f"In_{name}" for name in state_columns]
        expected_target_columns = [f"Out_{name}" for name in measured_output_columns]

        self.assertEqual(list(self.classical_benchmark_dataset.features.columns), expected_feature_columns)
        self.assertEqual(list(self.classical_benchmark_dataset.targets.columns), expected_target_columns)
        self.assertNotIn("In_COD", self.classical_benchmark_dataset.features.columns)
        self.assertNotIn("Out_S_A", self.classical_benchmark_dataset.features.columns)
        self.assertNotIn("Out_S_A", self.classical_benchmark_dataset.targets.columns)

    def test_shared_split_indices_align_classical_benchmark_with_icsor(self) -> None:
        split_indices = make_train_test_split_indices(self.dataset.index, test_fraction=0.2, random_seed=17)
        classical_splits = apply_train_test_split_indices(self.classical_benchmark_dataset, split_indices)
        icsor_splits = apply_train_test_split_indices(self.icsor_dataset, split_indices)

        self.assertTrue(classical_splits.train.features.index.equals(icsor_splits.train.features.index))
        self.assertTrue(classical_splits.test.features.index.equals(icsor_splits.test.features.index))
        self.assertTrue(classical_splits.train.features.columns.equals(icsor_splits.train.features.columns))
        self.assertTrue(classical_splits.test.targets.columns.equals(icsor_splits.test.targets.columns))

    def test_optuna_subset_is_drawn_from_training_pool_only(self) -> None:
        main_splits = make_train_test_split(self.classical_benchmark_dataset, test_fraction=0.2, random_seed=17)
        tuning_indices = sample_dataset_split_indices(main_splits.train.features.index, fraction=0.5, random_seed=17)
        tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=17)
        tuning_splits = make_train_test_split(tuning_dataset, test_fraction=0.25, random_seed=17)

        self.assertTrue(set(main_splits.test.features.index).isdisjoint(tuning_dataset.features.index))
        self.assertTrue(set(main_splits.test.features.index).isdisjoint(tuning_splits.train.features.index))
        self.assertTrue(set(main_splits.test.features.index).isdisjoint(tuning_splits.test.features.index))
        self.assertLess(len(tuning_dataset.features), len(main_splits.train.features))
        self.assertEqual(set(tuning_dataset.features.index), set(tuning_indices))

    def test_icsor_optuna_subset_aligns_with_shared_indices(self) -> None:
        split_indices = make_train_test_split_indices(self.dataset.index, test_fraction=0.2, random_seed=17)
        tuning_indices = sample_dataset_split_indices(split_indices.train, fraction=0.5, random_seed=17)
        tuning_split_indices = make_train_test_split_indices(tuning_indices, test_fraction=0.25, random_seed=17)

        classical_tuning_splits = apply_train_test_split_indices(self.classical_benchmark_dataset, tuning_split_indices)
        icsor_tuning_splits = apply_train_test_split_indices(self.icsor_dataset, tuning_split_indices)

        self.assertTrue(classical_tuning_splits.train.features.index.equals(icsor_tuning_splits.train.features.index))
        self.assertTrue(classical_tuning_splits.test.features.index.equals(icsor_tuning_splits.test.features.index))
        self.assertTrue(set(split_indices.test).isdisjoint(icsor_tuning_splits.train.features.index))
        self.assertTrue(set(split_indices.test).isdisjoint(icsor_tuning_splits.test.features.index))

    def test_classical_search_space_covers_non_fixed_training_defaults(self) -> None:
        params = load_params_config()

        for model_name in CLASSICAL_MODEL_NAMES:
            with self.subTest(model=model_name):
                model_params = params[model_name]
                training_keys = set(model_params["training_defaults"])
                search_keys = set(model_params["search_space"])
                fixed_keys = FIXED_HYPERPARAMETER_KEYS[model_name]

                self.assertTrue(fixed_keys.issubset(training_keys))
                self.assertTrue(fixed_keys.isdisjoint(search_keys))

                tuned_keys = training_keys - fixed_keys
                self.assertTrue(tuned_keys)
                self.assertEqual(tuned_keys, tuned_keys & search_keys)

    def test_external_tabular_tuning_returns_hyperparameters(self) -> None:
        params = _build_tiny_adaboost_params()
        main_splits = make_train_test_split(self.classical_benchmark_dataset, test_fraction=0.2, random_seed=11)
        tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=11)
        tuning_splits = make_train_test_split(tuning_dataset, test_fraction=0.25, random_seed=11)

        best_hyperparameters, optuna_summary = tune_tabular_regressor_hyperparameters(
            "adaboost_regressor",
            build_adaboost_regressor_model,
            tuning_splits.train,
            tuning_splits.test,
            A_matrix=self.a_matrix,
            model_params=params,
            n_trials=1,
            show_progress_bar=False,
        )

        self.assertTrue(set(params["training_defaults"]).issubset(best_hyperparameters))
        self.assertEqual(optuna_summary["n_trials"], 1)

    def test_external_icsor_tuning_returns_ridge_hyperparameters(self) -> None:
        params = _build_tiny_icsor_params()
        split_indices = make_train_test_split_indices(self.dataset.index, test_fraction=0.2, random_seed=11)
        tuning_indices = sample_dataset_split_indices(split_indices.train, fraction=0.5, random_seed=11)
        tuning_split_indices = make_train_test_split_indices(tuning_indices, test_fraction=0.25, random_seed=11)
        tuning_splits = apply_train_test_split_indices(self.icsor_dataset, tuning_split_indices)

        best_hyperparameters, optuna_summary = tune_icsor_hyperparameters(
            tuning_splits.train,
            tuning_splits.test,
            A_matrix=self.icsor_a_matrix,
            composition_matrix=self.composition_matrix,
            model_params=params,
            n_trials=1,
            show_progress_bar=False,
        )

        self.assertEqual(best_hyperparameters["affine_estimator"], "ridge")
        self.assertIn("ridge_alpha", best_hyperparameters)
        self.assertGreaterEqual(float(best_hyperparameters["ridge_alpha"]), 1e-6)
        self.assertLessEqual(float(best_hyperparameters["ridge_alpha"]), 1e-2)
        self.assertEqual(optuna_summary["n_trials"], 1)
        self.assertEqual(optuna_summary["best_trial_number"], 0)

    @patch("src.utils.optuna.create_progress_bar")
    def test_tabular_tuning_enables_progress_by_default(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        params = _build_tiny_adaboost_params()
        main_splits = make_train_test_split(self.classical_benchmark_dataset, test_fraction=0.2, random_seed=11)
        tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=11)
        tuning_splits = make_train_test_split(tuning_dataset, test_fraction=0.25, random_seed=11)

        tune_tabular_regressor_hyperparameters(
            "adaboost_regressor",
            build_adaboost_regressor_model,
            tuning_splits.train,
            tuning_splits.test,
            A_matrix=self.a_matrix,
            model_params=params,
            n_trials=1,
        )

        self.assertTrue(progress_factory.called)
        self.assertTrue(progress_factory.call_args.kwargs["enabled"])
        self.assertIn("validation_mse", progress_factory.call_args.kwargs["desc"])

    @patch("src.utils.optuna.create_progress_bar")
    def test_tabular_tuning_supports_progress_opt_out(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        params = _build_tiny_adaboost_params()
        main_splits = make_train_test_split(self.classical_benchmark_dataset, test_fraction=0.2, random_seed=11)
        tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=11)
        tuning_splits = make_train_test_split(tuning_dataset, test_fraction=0.25, random_seed=11)

        tune_tabular_regressor_hyperparameters(
            "adaboost_regressor",
            build_adaboost_regressor_model,
            tuning_splits.train,
            tuning_splits.test,
            A_matrix=self.a_matrix,
            model_params=params,
            n_trials=1,
            show_progress_bar=False,
        )

        self.assertTrue(progress_factory.called)
        self.assertFalse(progress_factory.call_args.kwargs["enabled"])
        self.assertIn("validation_mse", progress_factory.call_args.kwargs["desc"])


if __name__ == "__main__":
    unittest.main()

