"""Tests for dataset-size sweep analysis helpers."""

from __future__ import annotations

import copy
import tempfile
import unittest
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import null_space

from src.models.ml.cobre import run_cobre_pipeline
from src.models.simulation.asm2d_tcn_simulation import generate_asm2d_tcn_dataset
from src.utils.analysis import (
	build_negative_prediction_tables,
	build_cobre_response_surface_prediction_data,
	build_dataset_size_schedule,
	run_model_dataset_size_analysis,
)
from src.utils.io import save_pickle_file
from src.utils.process import DatasetSplit, SupervisedDatasetFrames, build_cobre_supervised_dataset, make_train_test_split
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


def _tiny_cobre_params() -> dict[str, Any]:
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
				"ols_backend": "numpy_lstsq",
				"include_bias_term": True,
				"lstsq_rcond": None,
			},
			"artifact_options": {
				"persist_model": True,
				"persist_metrics": True,
			},
		}
	)


class AnalysisHelperTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		dataset, metadata, matrix_bundle = generate_asm2d_tcn_dataset(n_samples=12, random_seed=17)
		cls.cobre_dataset = build_cobre_supervised_dataset(dataset, metadata, matrix_bundle["composition_matrix"])
		cls.cobre_metadata = metadata
		cls.cobre_composition_matrix = matrix_bundle["composition_matrix"]
		cls.cobre_a_matrix = _compute_a_matrix(matrix_bundle["petersen_matrix"])

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
		self.assertIn("measured_adjustment_l2", first_prediction_table.columns)

		train_sizes = set(result["run_metadata"]["train_size"])
		test_sizes = set(result["run_metadata"]["test_size"])
		self.assertEqual(train_sizes, {4, 8})
		self.assertEqual(test_sizes, {2, 3})

	def test_run_model_dataset_size_analysis_omits_projected_outputs_when_projection_inactive(self) -> None:
		dataset = _build_synthetic_dataset()
		inactive_a_matrix = np.zeros((0, 2), dtype=float)

		result = run_model_dataset_size_analysis(
			"synthetic_model",
			dataset,
			inactive_a_matrix,
			_fake_runner,
			min_total_samples=6,
			max_total_samples=6,
			total_sample_step=5,
			n_repeats=1,
			test_fraction=0.25,
			random_seed=13,
			show_progress=False,
			show_runner_progress=False,
		)

		self.assertNotIn("projected_R2", result["per_target_metrics"].columns)
		self.assertEqual(set(result["aggregate_metrics"]["prediction_type"]), {"raw"})

		first_prediction_table = result["prediction_tables"][0]
		self.assertIn("Actual_Out_A", first_prediction_table.columns)
		self.assertIn("Raw_Out_A", first_prediction_table.columns)
		self.assertIn("ConstraintReference_Out_A", first_prediction_table.columns)
		self.assertNotIn("Projected_Out_A", first_prediction_table.columns)
		self.assertNotIn("measured_adjustment_l2", first_prediction_table.columns)

	def test_build_negative_prediction_tables_handles_active_and_inactive_reports(self) -> None:
		target_columns = ["Out_A", "Out_B"]
		index = pd.Index([0, 1], name="sample_id")
		active_a_matrix = np.array([[1.0, -1.0]], dtype=float)
		inactive_a_matrix = np.zeros((0, 2), dtype=float)

		active_report = evaluate_prediction_bundle(
			y_true=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
			raw_predictions=np.array([[-1.0, 2.0], [3.0, -0.5]], dtype=float),
			projected_predictions=np.array([[0.2, 2.0], [3.0, 0.1]], dtype=float),
			constraint_reference=np.zeros((2, 2), dtype=float),
			A_matrix=active_a_matrix,
			target_columns=target_columns,
			index=index,
		)
		inactive_report = evaluate_prediction_bundle(
			y_true=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
			raw_predictions=np.array([[0.1, -0.2], [1.0, 0.3]], dtype=float),
			projected_predictions=np.array([[0.1, -0.2], [1.0, 0.3]], dtype=float),
			constraint_reference=np.zeros((2, 2), dtype=float),
			A_matrix=inactive_a_matrix,
			target_columns=target_columns,
			index=index,
		)

		negative_prediction_tables = build_negative_prediction_tables(
			{
				"train": active_report,
				"test": inactive_report,
			}
		)

		summary = negative_prediction_tables["summary"]
		per_target = negative_prediction_tables["per_target"]

		self.assertEqual(len(summary), 3)
		self.assertEqual(
			list(summary["prediction_type"]),
			["raw", "projected", "raw"],
		)

		train_raw_row = summary.loc[
			(summary["split"] == "train") & (summary["prediction_type"] == "raw")
		].iloc[0]
		self.assertEqual(int(train_raw_row["negative_predictions"]), 2)
		self.assertEqual(int(train_raw_row["total_predictions"]), 4)
		self.assertAlmostEqual(float(train_raw_row["negative_prediction_rate_pct"]), 50.0)
		self.assertEqual(int(train_raw_row["samples_with_any_negative"]), 2)
		self.assertAlmostEqual(float(train_raw_row["sample_incidence_rate_pct"]), 100.0)
		self.assertAlmostEqual(float(train_raw_row["minimum_prediction"]), -1.0)
		self.assertAlmostEqual(float(train_raw_row["mean_negative_prediction"]), -0.75)
		self.assertAlmostEqual(float(train_raw_row["median_negative_prediction"]), -0.75)

		train_projected_row = summary.loc[
			(summary["split"] == "train") & (summary["prediction_type"] == "projected")
		].iloc[0]
		self.assertEqual(int(train_projected_row["negative_predictions"]), 0)
		self.assertTrue(np.isnan(float(train_projected_row["mean_negative_prediction"])))

		test_raw_row = summary.loc[
			(summary["split"] == "test") & (summary["prediction_type"] == "raw")
		].iloc[0]
		self.assertEqual(int(test_raw_row["negative_predictions"]), 1)
		self.assertAlmostEqual(float(test_raw_row["minimum_prediction"]), -0.2)

		self.assertEqual(len(per_target), 6)
		self.assertFalse(
			((per_target["split"] == "test") & (per_target["prediction_type"] == "projected")).any()
		)
		self.assertIn("Out_A", set(per_target["target"]))
		self.assertIn("Out_B", set(per_target["target"]))

	def test_build_cobre_response_surface_prediction_data_uses_midpoint_profile_and_extended_domain(self) -> None:
		dataset_splits = make_train_test_split(
			self.cobre_dataset,
			test_fraction=0.25,
			random_seed=11,
		)
		result = run_cobre_pipeline(
			dataset_splits.train,
			dataset_splits.test,
			self.cobre_a_matrix,
			composition_matrix=self.cobre_composition_matrix,
			model_params=_tiny_cobre_params(),
			show_progress=False,
			persist_artifacts=False,
		)

		with tempfile.TemporaryDirectory() as temp_dir_name:
			model_path = tempfile.NamedTemporaryFile(dir=temp_dir_name, suffix=".pkl", delete=False)
			model_path.close()
			save_pickle_file(model_path.name, result["model_bundle"])
			response_surface = build_cobre_response_surface_prediction_data(
				model_path.name,
				metadata=self.cobre_metadata,
				grid_points_per_axis=7,
			)

		self.assertEqual(response_surface["response_surface_config"]["fixed_influent_profile"], "midpoint")
		self.assertEqual(response_surface["response_surface_config"]["grid_points_per_axis"], 7)
		self.assertAlmostEqual(response_surface["training_domain"]["HRT"]["min"], 6.0)
		self.assertAlmostEqual(response_surface["training_domain"]["HRT"]["max"], 36.0)
		self.assertAlmostEqual(response_surface["extended_domain"]["HRT"]["min"], 0.0)
		self.assertAlmostEqual(response_surface["extended_domain"]["HRT"]["max"], 51.0)
		self.assertAlmostEqual(response_surface["extended_domain"]["Aeration"]["min"], 0.0)
		self.assertAlmostEqual(response_surface["extended_domain"]["Aeration"]["max"], 3.5)
		self.assertEqual(response_surface["operational_meshes"]["HRT"].shape, (7, 7))
		self.assertEqual(response_surface["operational_meshes"]["Aeration"].shape, (7, 7))
		self.assertEqual(len(response_surface["prediction_table"]), 49)
		self.assertGreaterEqual(float(response_surface["prediction_table"]["HRT"].min()), 0.0)
		self.assertGreaterEqual(float(response_surface["prediction_table"]["Aeration"].min()), 0.0)
		self.assertEqual(
			set(response_surface["per_target_surfaces"].keys()),
			set(f"Out_{name}" for name in self.cobre_metadata["measured_output_columns"]),
		)
		self.assertIn("Projected_Out_COD", response_surface["prediction_table"].columns)
		self.assertIn("ConstraintReference_S_A", response_surface["prediction_table"].columns)
		self.assertAlmostEqual(float(response_surface["fixed_influent_profile"].loc["S_A"]), 42.5)


if __name__ == "__main__":
	unittest.main()