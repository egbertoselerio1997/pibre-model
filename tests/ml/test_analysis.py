"""Tests for dataset-size sweep analysis helpers."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import pandas as pd

from src.utils.analysis import build_dataset_size_schedule, run_model_dataset_size_analysis
from src.utils.process import DatasetSplit, SupervisedDatasetFrames
from src.utils.test import evaluate_prediction_bundle


def _build_synthetic_dataset(n_samples: int = 18) -> SupervisedDatasetFrames:
	index = pd.Index(range(n_samples), name="sample_id")
	features = pd.DataFrame(
		{
			"feature_1": np.linspace(0.0, 1.0, n_samples),
			"feature_2": np.linspace(1.0, 2.0, n_samples),
		},
		index=index,
	)
	targets = pd.DataFrame(
		{
			"Out_A": np.linspace(2.0, 3.0, n_samples),
			"Out_B": np.linspace(4.0, 5.0, n_samples),
		},
		index=index,
	)
	constraint_reference = targets.copy()
	return SupervisedDatasetFrames(
		features=features,
		targets=targets,
		constraint_reference=constraint_reference,
	)


def _fake_runner(
	training_split: DatasetSplit,
	test_split: DatasetSplit,
	A_matrix: np.ndarray,
	**_: Any,
) -> dict[str, Any]:
	train_targets = training_split.targets.to_numpy(dtype=float)
	test_targets = test_split.targets.to_numpy(dtype=float)
	train_raw = train_targets + 0.05
	test_raw = test_targets + 0.08
	train_projected = train_targets.copy()
	test_projected = test_targets.copy()

	train_report = evaluate_prediction_bundle(
		train_targets,
		train_raw,
		train_projected,
		training_split.constraint_reference.to_numpy(dtype=float),
		A_matrix,
		training_split.targets.columns,
		index=training_split.targets.index,
	)
	test_report = evaluate_prediction_bundle(
		test_targets,
		test_raw,
		test_projected,
		test_split.constraint_reference.to_numpy(dtype=float),
		A_matrix,
		test_split.targets.columns,
		index=test_split.targets.index,
	)

	return {
		"best_hyperparameters": {"objective": "synthetic"},
		"optuna_summary": None,
		"artifact_paths": {"model_bundle": None, "metrics": None, "optuna": None},
		"train_report": train_report,
		"test_report": test_report,
		"model_bundle": {"model_name": "synthetic"},
		"dataset_splits": {"train": training_split, "test": test_split},
	}


class AnalysisHelperTests(unittest.TestCase):
	def test_dataset_size_schedule_includes_capped_maximum(self) -> None:
		schedule = build_dataset_size_schedule(
			18,
			min_total_samples=6,
			max_total_samples=17,
			total_sample_step=5,
		)

		self.assertEqual(schedule, [6, 11, 16, 17])

	def test_run_model_dataset_size_analysis_returns_expected_shapes(self) -> None:
		dataset = _build_synthetic_dataset()
		a_matrix = np.array([[1.0, -1.0]], dtype=float)

		result = run_model_dataset_size_analysis(
			"synthetic_model",
			dataset,
			a_matrix,
			_fake_runner,
			min_total_samples=6,
			max_total_samples=11,
			total_sample_step=5,
			n_repeats=2,
			test_fraction=0.25,
			random_seed=13,
			show_progress=False,
			show_runner_progress=False,
		)

		self.assertEqual(result["dataset_sizes"], [6, 11])
		self.assertEqual(len(result["run_metadata"]), 4)
		self.assertEqual(len(result["prediction_tables"]), 8)
		self.assertIn("projected_R2", result["per_target_metrics"].columns)
		self.assertEqual(set(result["per_target_metrics"]["split_name"]), {"train", "test"})
		self.assertEqual(set(result["aggregate_metrics"]["prediction_type"]), {"raw", "projected"})
		self.assertEqual(int(result["analysis_config"]["n_repeats"]), 2)

		first_prediction_table = result["prediction_tables"][0]
		self.assertIn("Actual_Out_A", first_prediction_table.columns)
		self.assertIn("Raw_Out_A", first_prediction_table.columns)
		self.assertIn("Projected_Out_A", first_prediction_table.columns)
		self.assertIn("ConstraintReference_Out_A", first_prediction_table.columns)

		train_sizes = set(result["run_metadata"]["train_size"])
		test_sizes = set(result["run_metadata"]["test_size"])
		self.assertEqual(train_sizes, {4, 8})
		self.assertEqual(test_sizes, {2, 3})


if __name__ == "__main__":
	unittest.main()