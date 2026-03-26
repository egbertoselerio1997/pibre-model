"""Minimal tests for the extracted PIBRe model workflow."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
from scipy.linalg import null_space

from src.models.ml.pibre import (
    compute_mass_balance_violation,
    evaluate_pibre_model,
    predict_pibre,
    prepare_pibre_arrays,
    project_to_mass_balance,
    split_pibre_dataset,
    train_pibre_model,
    tune_pibre_hyperparameters,
)
from src.models.simulation.asm1_simulation import generate_asm1_dataset


class PibreModelTests(unittest.TestCase):
    def setUp(self) -> None:
        dataset, metadata, matrix_bundle = generate_asm1_dataset(n_samples=20, random_seed=11)
        self.dataset = dataset
        self.metadata = metadata
        self.composition_matrix = matrix_bundle["composition_matrix"]
        self.a_matrix = null_space(matrix_bundle["petersen_matrix"] @ self.composition_matrix.T).T
        self.a_matrix = np.round(self.a_matrix, 5)
        self.a_matrix[np.abs(self.a_matrix) < 1e-10] = 0.0
        for row_index in range(self.a_matrix.shape[0]):
            non_zero_entries = self.a_matrix[row_index, self.a_matrix[row_index, :] != 0]
            if len(non_zero_entries) > 0:
                self.a_matrix[row_index, :] = self.a_matrix[row_index, :] / non_zero_entries[0]

        self.hyperparameters = {
            "random_seed": 42,
            "test_size": 0.2,
            "validation_size": 0.25,
            "tuning_subset_size": 8,
            "optuna_trials": 1,
            "tuning_epochs": 4,
            "final_training_epochs": 6,
            "validation_frequency": 2,
            "pruner_startup_trials": 1,
            "pruner_warmup_steps": 1,
            "artifact_name": "unit_test_pibre",
            "save_model": True,
            "show_optuna_progress": False,
            "show_training_progress": False,
            "learning_rate_search": {"low": 1e-3, "high": 5e-3, "log": True},
            "lambda_l1_search": {"low": 1e-6, "high": 1e-4, "log": True},
            "weight_decay_search": {"low": 1e-8, "high": 1e-5, "log": True},
            "clip_max_norm_search": {"low": 0.5, "high": 1.0, "log": False},
        }

    def test_prepare_pibre_arrays_match_measured_output_space(self) -> None:
        prepared = prepare_pibre_arrays(self.dataset, self.metadata, self.composition_matrix)

        self.assertEqual(prepared["x_scaled"].shape[0], len(self.dataset))
        self.assertEqual(prepared["x_scaled"].shape[1], len(self.metadata["independent_columns"]))
        self.assertEqual(prepared["y"].shape[1], len(self.metadata["measured_output_columns"]))
        self.assertEqual(prepared["cin_measured"].shape[1], len(self.metadata["measured_output_columns"]))

    def test_projection_reduces_mass_balance_violation(self) -> None:
        prepared = prepare_pibre_arrays(self.dataset, self.metadata, self.composition_matrix)
        raw_predictions = prepared["y"] + 0.25
        projected_predictions = project_to_mass_balance(raw_predictions, prepared["cin_measured"], self.a_matrix)

        raw_violation = compute_mass_balance_violation(raw_predictions, prepared["cin_measured"], self.a_matrix)
        projected_violation = compute_mass_balance_violation(
            projected_predictions,
            prepared["cin_measured"],
            self.a_matrix,
        )

        if self.a_matrix.shape[0] == 0:
            self.assertAlmostEqual(projected_violation.mean(), raw_violation.mean())
            self.assertLessEqual(projected_violation.mean(), 1e-12)
        else:
            self.assertLess(projected_violation.mean(), raw_violation.mean())
            self.assertLessEqual(projected_violation.mean(), 1e-6)

    def test_tune_train_and_predict_round_trip(self) -> None:
        train_dataset, test_dataset = split_pibre_dataset(self.dataset, self.hyperparameters)
        tuning_result = tune_pibre_hyperparameters(
            train_dataset,
            self.hyperparameters,
            metadata=self.metadata,
            composition_matrix=self.composition_matrix,
            A_matrix=self.a_matrix,
            device="cpu",
        )

        self.assertEqual(tuning_result["completed_trials"], 1)
        tuned_hyperparameters = {**self.hyperparameters, **tuning_result["best_params"]}
        paths_config = {
            "ml_model_artifact_pattern": "results/{model_name}/{artifact_name}_{date_time}.pkl",
        }

        with tempfile.TemporaryDirectory() as temp_dir_name:
            trained_bundle, training_predictions = train_pibre_model(
                train_dataset,
                tuned_hyperparameters,
                metadata=self.metadata,
                composition_matrix=self.composition_matrix,
                A_matrix=self.a_matrix,
                device="cpu",
                repo_root=temp_dir_name,
                paths_config=paths_config,
            )

            self.assertTrue(trained_bundle["model_path"].is_file())
            self.assertEqual(len(training_predictions), len(train_dataset))

            predicted_frame = predict_pibre(test_dataset, trained_bundle["model_path"], device="cpu")
            expected_target_columns = [f"Out_{name}" for name in self.metadata["measured_output_columns"]]
            self.assertEqual(list(predicted_frame.columns), expected_target_columns)
            self.assertEqual(list(predicted_frame.index), list(test_dataset.index))

            evaluation = evaluate_pibre_model(test_dataset, trained_bundle["model_path"], device="cpu")
            report_df = evaluation["report_df"].set_index("model")
            self.assertLessEqual(report_df.loc["PIBRe projected", "mean_mb_error"], 1e-4)
            self.assertGreaterEqual(
                report_df.loc["PIBRe raw", "mean_mb_error"],
                report_df.loc["PIBRe projected", "mean_mb_error"],
            )
            self.assertEqual(len(evaluation["per_target_df"]), len(expected_target_columns))


if __name__ == "__main__":
    unittest.main()