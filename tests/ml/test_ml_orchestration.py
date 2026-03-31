"""Tests for notebook-managed ML orchestration helpers."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.linalg import null_space

from src.models.ml.adaboost_regressor import build_adaboost_regressor_model, load_adaboost_regressor_params
from src.models.simulation.asm2d_tcn_simulation import generate_asm2d_tcn_dataset
from src.utils.process import (
    apply_train_test_split_indices,
    build_cobre_supervised_dataset,
    build_fractional_input_measured_output_dataset,
    make_train_test_split,
    make_train_test_split_indices,
    sample_dataset_fraction,
    sample_dataset_split_indices,
)
from src.utils.train import tune_tabular_regressor_hyperparameters


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
        dataset, metadata, matrix_bundle = generate_asm2d_tcn_dataset(n_samples=12, random_seed=23)
        cls.dataset = dataset
        cls.metadata = metadata
        cls.composition_matrix = matrix_bundle["composition_matrix"]
        cls.petersen_matrix = matrix_bundle["petersen_matrix"]
        cls.a_matrix = _compute_a_matrix(cls.petersen_matrix, cls.composition_matrix)
        cls.classical_benchmark_dataset = build_fractional_input_measured_output_dataset(
            dataset,
            metadata,
            cls.composition_matrix,
        )
        cls.cobre_dataset = build_cobre_supervised_dataset(dataset, metadata, cls.composition_matrix)

    def test_shared_split_indices_align_classical_benchmark_with_cobre(self) -> None:
        split_indices = make_train_test_split_indices(self.dataset.index, test_fraction=0.2, random_seed=17)
        classical_splits = apply_train_test_split_indices(self.classical_benchmark_dataset, split_indices)
        cobre_splits = apply_train_test_split_indices(self.cobre_dataset, split_indices)

        self.assertTrue(classical_splits.train.features.index.equals(cobre_splits.train.features.index))
        self.assertTrue(classical_splits.test.features.index.equals(cobre_splits.test.features.index))
        self.assertTrue(classical_splits.train.features.columns.equals(cobre_splits.train.features.columns))
        self.assertTrue(classical_splits.test.targets.columns.equals(cobre_splits.test.targets.columns))

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