"""Tests for notebook-managed ML orchestration helpers."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

from src.models.ml.adaboost_regressor import build_adaboost_regressor_model, load_adaboost_regressor_params
from src.models.ml.pibre import load_pibre_params
from src.models.simulation.asm1_simulation import generate_asm1_dataset
from src.utils.process import build_measured_supervised_dataset, make_train_test_split, sample_dataset_fraction
from src.utils.train import tune_pibre_hyperparameters, tune_tabular_regressor_hyperparameters


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


def _build_tiny_pibre_params() -> dict[str, object]:
    params = copy.deepcopy(load_pibre_params())
    params["hyperparameters"]["random_seed"] = 11
    params["hyperparameters"]["batch_size"] = 8
    params["hyperparameters"]["log_interval"] = 2
    params["hyperparameters"]["training_epochs"] = 8
    params["search_space"] = {
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.02, "log": True},
        "lambda_l1": {"type": "float", "low": 0.00001, "high": 0.001, "log": True},
        "weight_decay": {"type": "float", "low": 0.00000001, "high": 0.0001, "log": True},
        "clip_max_norm": {"type": "float", "low": 0.5, "high": 2.0, "log": False},
        "bilinear_init_scale": {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    }
    params["pruner"] = {
        "type": "median",
        "n_startup_trials": 1,
        "n_warmup_steps": 1,
        "interval_steps": 1,
    }
    return params


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


class MlOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, metadata, matrix_bundle = generate_asm1_dataset(n_samples=32, random_seed=23)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)
        cls.measured_dataset = build_measured_supervised_dataset(dataset, metadata, cls.composition_matrix)

    def test_optuna_subset_is_drawn_from_training_pool_only(self) -> None:
        main_splits = make_train_test_split(self.measured_dataset, test_fraction=0.2, random_seed=17)
        tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=17)
        tuning_splits = make_train_test_split(tuning_dataset, test_fraction=0.25, random_seed=17)

        self.assertTrue(set(main_splits.test.features.index).isdisjoint(tuning_dataset.features.index))
        self.assertTrue(set(main_splits.test.features.index).isdisjoint(tuning_splits.train.features.index))
        self.assertTrue(set(main_splits.test.features.index).isdisjoint(tuning_splits.test.features.index))
        self.assertLess(len(tuning_dataset.features), len(main_splits.train.features))

    def test_external_pibre_tuning_returns_hyperparameters(self) -> None:
        params = _build_tiny_pibre_params()
        main_splits = make_train_test_split(self.measured_dataset, test_fraction=0.2, random_seed=11)
        tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=11)
        tuning_splits = make_train_test_split(tuning_dataset, test_fraction=0.25, random_seed=11)

        best_hyperparameters, optuna_summary = tune_pibre_hyperparameters(
            tuning_splits.train,
            tuning_splits.test,
            A_matrix=self.a_matrix,
            model_params=params,
            tuning_epochs=3,
            n_trials=1,
            show_progress_bar=False,
        )

        self.assertTrue(set(params["training_defaults"]).issubset(best_hyperparameters))
        self.assertEqual(optuna_summary["n_trials"], 1)

    def test_external_tabular_tuning_returns_hyperparameters(self) -> None:
        params = _build_tiny_adaboost_params()
        main_splits = make_train_test_split(self.measured_dataset, test_fraction=0.2, random_seed=11)
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

    @patch("src.utils.optuna.create_progress_bar")
    def test_tabular_tuning_enables_progress_by_default(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        params = _build_tiny_adaboost_params()
        main_splits = make_train_test_split(self.measured_dataset, test_fraction=0.2, random_seed=11)
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
    def test_pibre_tuning_supports_progress_opt_out(self, progress_factory: MagicMock) -> None:
        progress_factory.return_value = MagicMock()
        params = _build_tiny_pibre_params()
        main_splits = make_train_test_split(self.measured_dataset, test_fraction=0.2, random_seed=11)
        tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=11)
        tuning_splits = make_train_test_split(tuning_dataset, test_fraction=0.25, random_seed=11)

        tune_pibre_hyperparameters(
            tuning_splits.train,
            tuning_splits.test,
            A_matrix=self.a_matrix,
            model_params=params,
            tuning_epochs=3,
            n_trials=1,
            show_progress_bar=False,
        )

        self.assertTrue(progress_factory.called)
        self.assertFalse(progress_factory.call_args.kwargs["enabled"])
        self.assertIn("validation_loss", progress_factory.call_args.kwargs["desc"])


if __name__ == "__main__":
    unittest.main()