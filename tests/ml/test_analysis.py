"""Tests for dataset-size sweep analysis helpers."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import null_space

from src.models.ml.icsor import run_icsor_pipeline
from src.models.simulation.asm2d_tsn_simulation import generate_asm2d_tsn_dataset
from src.utils.analysis import (
	add_effective_metric_columns,
	build_negative_prediction_tables,
	build_notebook_table_recorder,
	build_train_test_gap_summary,
	build_separated_negative_prediction_tables,
	build_icsor_response_surface_prediction_data,
	build_dataset_size_schedule,
	collate_model_analysis_results,
	load_latest_analysis_result,
	load_latest_classical_training_context,
	load_latest_icsor_training_context,
	persist_analysis_result_artifacts,
	persist_classical_training_context,
	persist_icsor_training_context,
	rank_metric_summary,
	run_model_dataset_size_analysis,
	summarize_metric_distribution,
)
from src.utils.io import save_pickle_file
from src.utils.process import DatasetSplit, SupervisedDatasetFrames, build_icsor_supervised_dataset, make_train_test_split
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


def _tiny_icsor_params() -> dict[str, Any]:
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
				"affine_estimator": "ols",
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
			"artifact_options": {
				"persist_model": True,
				"persist_metrics": True,
				"persist_optuna": False,
			},
		}
	)


def _write_temp_paths_config(repo_root: Path) -> None:
	(repo_root / "config").mkdir(parents=True, exist_ok=True)
	(repo_root / "results").mkdir(parents=True, exist_ok=True)
	paths_config = {
		"notebook_tabular_results_dir": "results/tabular_results",
		"notebook_plot_results_dir": "results/plot_results",
		"notebook_tabular_artifact_pattern": "results/tabular_results/{artifact_group}/{artifact_name}_{date_time}.csv",
		"notebook_plot_artifact_pattern": "results/plot_results/{artifact_group}/{artifact_name}_{date_time}.{extension}",
	}
	with (repo_root / "config" / "paths.json").open("w", encoding="utf-8") as handle:
		json.dump(paths_config, handle)
		handle.write("\n")


class AnalysisHelperTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		dataset, metadata, matrix_bundle = generate_asm2d_tsn_dataset(n_samples=12, random_seed=17)
		cls.icsor_dataset = build_icsor_supervised_dataset(dataset, metadata, matrix_bundle["composition_matrix"])
		cls.icsor_metadata = metadata
		cls.icsor_composition_matrix = matrix_bundle["composition_matrix"]
		cls.icsor_a_matrix = _compute_a_matrix(matrix_bundle["petersen_matrix"])

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

	def test_add_effective_metric_columns_prefers_projected_and_falls_back_to_raw(self) -> None:
		metric_frame = pd.DataFrame(
			[
				{
					"model_name": "projected_model",
					"projected_RMSE": 0.2,
					"raw_RMSE": 0.35,
				},
				{
					"model_name": "raw_model",
					"projected_RMSE": np.nan,
					"raw_RMSE": 0.45,
				},
			]
		)

		result = add_effective_metric_columns(metric_frame)

		self.assertEqual(list(result["effective_RMSE_source"]), ["projected", "raw"])
		self.assertAlmostEqual(float(result.loc[0, "effective_RMSE"]), 0.2)
		self.assertAlmostEqual(float(result.loc[1, "effective_RMSE"]), 0.45)

	def test_summarize_metric_distribution_rank_and_gap_helpers_align_for_lower_is_better_metric(self) -> None:
		metric_frame = pd.DataFrame(
			[
				{"model_label": "Model A", "split_name": "train", "train_size": 80, "effective_RMSE": 0.10},
				{"model_label": "Model A", "split_name": "train", "train_size": 80, "effective_RMSE": 0.11},
				{"model_label": "Model A", "split_name": "test", "train_size": 80, "effective_RMSE": 0.20},
				{"model_label": "Model A", "split_name": "test", "train_size": 80, "effective_RMSE": 0.22},
				{"model_label": "Model B", "split_name": "train", "train_size": 80, "effective_RMSE": 0.12},
				{"model_label": "Model B", "split_name": "train", "train_size": 80, "effective_RMSE": 0.13},
				{"model_label": "Model B", "split_name": "test", "train_size": 80, "effective_RMSE": 0.30},
				{"model_label": "Model B", "split_name": "test", "train_size": 80, "effective_RMSE": 0.35},
			]
		)

		summary = summarize_metric_distribution(
			metric_frame,
			metric_name="effective_RMSE",
			group_columns=["model_label", "split_name", "train_size"],
		)
		test_summary = summary.loc[summary["split_name"] == "test"].reset_index(drop=True)
		ranked = rank_metric_summary(
			test_summary,
			group_columns=["train_size"],
			metric_name="effective_RMSE",
		)
		gap_summary = build_train_test_gap_summary(
			summary,
			group_columns=["model_label", "train_size"],
			metric_name="effective_RMSE",
		)

		self.assertEqual(
			list(ranked.sort_values("metric_rank")["model_label"]),
			["Model A", "Model B"],
		)
		model_a_gap = gap_summary.loc[gap_summary["model_label"] == "Model A"].iloc[0]
		model_b_gap = gap_summary.loc[gap_summary["model_label"] == "Model B"].iloc[0]
		self.assertGreater(float(model_a_gap["generalization_gap"]), 0.0)
		self.assertGreater(float(model_b_gap["generalization_gap"]), float(model_a_gap["generalization_gap"]))

	def test_collate_model_analysis_results_builds_prediction_diagnostics_and_effective_metrics(self) -> None:
		dataset = _build_synthetic_dataset()
		active_a_matrix = np.array([[1.0, -1.0]], dtype=float)
		inactive_a_matrix = np.zeros((0, 2), dtype=float)

		projected_result = run_model_dataset_size_analysis(
			"projected_model",
			dataset,
			active_a_matrix,
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
		raw_result = run_model_dataset_size_analysis(
			"raw_model",
			dataset,
			inactive_a_matrix,
			_fake_runner,
			min_total_samples=6,
			max_total_samples=6,
			total_sample_step=5,
			n_repeats=1,
			test_fraction=0.25,
			random_seed=17,
			show_progress=False,
			show_runner_progress=False,
		)

		collated = collate_model_analysis_results(
			{
				"projected_model": projected_result,
				"raw_model": raw_result,
			},
			model_labels={
				"projected_model": "Projected Model",
				"raw_model": "Raw Model",
			},
			model_families={
				"projected_model": "Constrained",
				"raw_model": "Baseline",
			},
		)

		coverage = collated["coverage"]
		self.assertEqual(set(coverage["effective_metric_source"]), {"projected", "raw"})
		self.assertIn("effective_RMSE", collated["effective_aggregate_metrics"].columns)
		self.assertEqual(set(collated["effective_aggregate_metrics"]["effective_prediction_type"]), {"projected", "raw"})
		self.assertIn("effective_RMSE", collated["per_target_metrics"].columns)
		self.assertIn("effective_RMSE_source", collated["per_target_metrics"].columns)
		self.assertEqual(set(collated["prediction_diagnostics"]["effective_prediction_type"]), {"projected", "raw"})
		self.assertEqual(set(collated["prediction_target_diagnostics"]["target"]), {"Out_A", "Out_B"})

		projected_diagnostics = collated["prediction_diagnostics"].loc[
			collated["prediction_diagnostics"]["model_name"] == "projected_model"
		].iloc[0]
		self.assertGreater(float(projected_diagnostics["raw_to_effective_adjustment_mean_l2"]), 0.0)
		self.assertEqual(projected_diagnostics["model_label"], "Projected Model")

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

	def test_persist_and_load_latest_analysis_result_roundtrip(self) -> None:
		dataset = _build_synthetic_dataset()
		a_matrix = np.array([[1.0, -1.0]], dtype=float)
		analysis_result = run_model_dataset_size_analysis(
			"synthetic_model",
			dataset,
			a_matrix,
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

		with tempfile.TemporaryDirectory() as temp_dir:
			temp_root = Path(temp_dir)
			_write_temp_paths_config(temp_root)
			persist_analysis_result_artifacts(
				"synthetic_model",
				analysis_result,
				repo_root=temp_root,
				timestamp="20260406_010101",
			)
			updated_analysis_result = dict(analysis_result)
			updated_analysis_result["analysis_config"] = {
				**dict(analysis_result["analysis_config"]),
				"n_repeats": 7,
			}
			persist_analysis_result_artifacts(
				"synthetic_model",
				updated_analysis_result,
				repo_root=temp_root,
				timestamp="20260406_020202",
			)

			loaded_analysis_result = load_latest_analysis_result(
				"synthetic_model",
				repo_root=temp_root,
			)

			self.assertEqual(loaded_analysis_result["artifact_timestamp"], "20260406_020202")
			self.assertEqual(int(loaded_analysis_result["analysis_config"]["n_repeats"]), 7)
			self.assertEqual(len(loaded_analysis_result["prediction_tables"]), len(analysis_result["prediction_tables"]))
			self.assertEqual(
				list(loaded_analysis_result["aggregate_metrics"]["prediction_type"].unique()),
				["raw", "projected"],
			)

	def test_persist_and_load_latest_classical_training_context_roundtrip(self) -> None:
		dataset = _build_synthetic_dataset()
		dataset_splits = make_train_test_split(dataset, test_fraction=0.25, random_seed=5)
		classical_result = _fake_runner(
			dataset_splits.train,
			dataset_splits.test,
			np.array([[1.0, -1.0]], dtype=float),
		)
		classical_result["best_hyperparameters"] = {"depth": 3, "learning_rate": 0.1}
		classical_result["artifact_paths"] = {
			"model_bundle": Path("results/old_model.pkl"),
			"metrics": Path("results/old_metrics.json"),
			"optuna": None,
		}

		with tempfile.TemporaryDirectory() as temp_dir:
			temp_root = Path(temp_dir)
			_write_temp_paths_config(temp_root)
			persist_classical_training_context(
				"synthetic_regressor",
				classical_result,
				repo_root=temp_root,
				timestamp="20260406_010101",
			)
			updated_result = dict(classical_result)
			updated_result["best_hyperparameters"] = {"depth": 5, "learning_rate": 0.05}
			updated_result["artifact_paths"] = {
				"model_bundle": Path("results/new_model.pkl"),
				"metrics": Path("results/new_metrics.json"),
				"optuna": None,
			}
			persist_classical_training_context(
				"synthetic_regressor",
				updated_result,
				repo_root=temp_root,
				timestamp="20260406_020202",
			)

			loaded_context = load_latest_classical_training_context(
				"synthetic_regressor",
				repo_root=temp_root,
			)

			self.assertEqual(loaded_context["artifact_timestamp"], "20260406_020202")
			self.assertEqual(int(loaded_context["best_hyperparameters"]["depth"]), 5)
			self.assertEqual(
				loaded_context["training_artifact_paths"]["model_bundle"],
				Path("results/new_model.pkl"),
			)

	def test_persist_and_load_latest_icsor_training_context_roundtrip(self) -> None:
		dataset = _build_synthetic_dataset()
		dataset_splits = make_train_test_split(dataset, test_fraction=0.25, random_seed=7)
		prefixed = lambda frame, prefix: frame.rename(columns=lambda column_name: f"{prefix}{column_name}")
		icsor_result = {
			"best_hyperparameters": {"objective": "projected_ols"},
			"artifact_paths": {
				"model_bundle": Path("results/icsor/model.pkl"),
				"metrics": Path("results/icsor/metrics.json"),
				"optuna": None,
			},
			"train_report": {
				"raw_predictions": prefixed(dataset_splits.train.targets, "Raw_"),
				"affine_predictions": prefixed(dataset_splits.train.targets, "Affine_"),
				"projected_predictions": prefixed(dataset_splits.train.targets, "Projected_"),
				"raw_fractional_predictions": prefixed(dataset_splits.train.constraint_reference, "RawFractional_"),
				"affine_fractional_predictions": prefixed(dataset_splits.train.constraint_reference, "AffineFractional_"),
				"projected_fractional_predictions": prefixed(dataset_splits.train.constraint_reference, "ProjectedFractional_"),
			},
			"test_report": {
				"raw_predictions": prefixed(dataset_splits.test.targets, "Raw_"),
				"affine_predictions": prefixed(dataset_splits.test.targets, "Affine_"),
				"projected_predictions": prefixed(dataset_splits.test.targets, "Projected_"),
				"raw_fractional_predictions": prefixed(dataset_splits.test.constraint_reference, "RawFractional_"),
				"affine_fractional_predictions": prefixed(dataset_splits.test.constraint_reference, "AffineFractional_"),
				"projected_fractional_predictions": prefixed(dataset_splits.test.constraint_reference, "ProjectedFractional_"),
			},
			"model_bundle": {
				"effective_coefficients": {
					"W_u": np.arange(4, dtype=float).reshape(2, 2),
					"W_in": np.arange(4, 8, dtype=float).reshape(2, 2),
					"b": np.array([1.0, 2.0], dtype=float),
					"Theta_uu": np.arange(8, dtype=float).reshape(2, 2, 2),
					"Theta_cc": np.arange(8, 16, dtype=float).reshape(2, 2, 2),
					"Theta_uc": np.arange(16, 24, dtype=float).reshape(2, 2, 2),
				}
			},
		}

		with tempfile.TemporaryDirectory() as temp_dir:
			temp_root = Path(temp_dir)
			_write_temp_paths_config(temp_root)
			persist_icsor_training_context(
				icsor_result,
				{"train": dataset_splits.train, "test": dataset_splits.test},
				repo_root=temp_root,
				timestamp="20260406_030303",
			)

			loaded_context = load_latest_icsor_training_context(repo_root=temp_root)

			self.assertEqual(loaded_context["artifact_timestamp"], "20260406_030303")
			self.assertEqual(list(loaded_context["train_targets"].columns), ["Out_A", "Out_B"])
			self.assertIn("projected_predictions", loaded_context["train_report"])
			self.assertEqual(loaded_context["effective_coefficients"]["Theta_uu"].shape, (2, 2, 2))
			self.assertEqual(
				loaded_context["training_artifact_paths"]["model_bundle"],
				Path("results/icsor/model.pkl"),
			)

	def test_build_separated_negative_prediction_tables_splits_composite_and_fractional_families(self) -> None:
		index = pd.Index([0, 1], name="sample_id")
		reports_by_split = {
			"train": {
				"raw_predictions": pd.DataFrame(
					{
						"Raw_Out_A": [-1.0, 3.0],
						"Raw_Out_B": [2.0, -0.5],
					},
					index=index,
				),
				"affine_predictions": pd.DataFrame(
					{
						"Affine_Out_A": [0.2, 3.0],
						"Affine_Out_B": [2.0, 0.1],
					},
					index=index,
				),
				"projected_predictions": pd.DataFrame(
					{
						"Projected_Out_A": [0.3, 3.0],
						"Projected_Out_B": [1.5, 0.4],
					},
					index=index,
				),
				"raw_fractional_predictions": pd.DataFrame(
					{
						"RawFractional_S_A": [0.2, -0.4],
						"RawFractional_S_B": [-0.1, 0.3],
					},
					index=index,
				),
				"affine_fractional_predictions": pd.DataFrame(
					{
						"AffineFractional_S_A": [0.2, 0.4],
						"AffineFractional_S_B": [0.1, -0.05],
					},
					index=index,
				),
				"projected_fractional_predictions": pd.DataFrame(
					{
						"ProjectedFractional_S_A": [0.2, 0.4],
						"ProjectedFractional_S_B": [0.1, 0.05],
					},
					index=index,
				),
			},
			"test": {
				"raw_predictions": pd.DataFrame(
					{
						"Raw_Out_A": [0.1, 1.0],
						"Raw_Out_B": [-0.2, 0.3],
					},
					index=index,
				),
				"affine_predictions": pd.DataFrame(
					{
						"Affine_Out_A": [0.2, 1.0],
						"Affine_Out_B": [0.1, 0.3],
					},
					index=index,
				),
				"projected_predictions": pd.DataFrame(
					{
						"Projected_Out_A": [0.05, 1.0],
						"Projected_Out_B": [0.1, 0.3],
					},
					index=index,
				),
				"raw_fractional_predictions": pd.DataFrame(
					{
						"RawFractional_S_A": [0.1, 0.0],
						"RawFractional_S_B": [-0.2, 0.3],
					},
					index=index,
				),
				"affine_fractional_predictions": pd.DataFrame(
					{
						"AffineFractional_S_A": [0.1, 0.0],
						"AffineFractional_S_B": [0.2, 0.3],
					},
					index=index,
				),
				"projected_fractional_predictions": pd.DataFrame(
					{
						"ProjectedFractional_S_A": [0.1, 0.0],
						"ProjectedFractional_S_B": [0.2, 0.3],
					},
					index=index,
				),
			},
		}

		negative_prediction_tables = build_separated_negative_prediction_tables(reports_by_split)

		self.assertEqual(set(negative_prediction_tables), {"composite", "fractional"})
		self.assertEqual(
			list(negative_prediction_tables["composite"]),
			["raw", "affine", "projected"],
		)
		self.assertEqual(
			list(negative_prediction_tables["fractional"]),
			["raw", "affine", "projected"],
		)

		composite_raw_summary = negative_prediction_tables["composite"]["raw"]["summary"]
		self.assertEqual(list(composite_raw_summary["split"]), ["train", "test"])
		self.assertEqual(
			list(composite_raw_summary["negative_predictions"]),
			[2, 1],
		)

		fractional_affine_summary = negative_prediction_tables["fractional"]["affine"]["summary"]
		self.assertEqual(list(fractional_affine_summary["split"]), ["train", "test"])
		self.assertEqual(
			list(fractional_affine_summary["negative_predictions"]),
			[1, 0],
		)

		fractional_projected_summary = negative_prediction_tables["fractional"]["projected"]["summary"]
		self.assertTrue((fractional_projected_summary["negative_predictions"] == 0).all())

		fractional_raw_per_target = negative_prediction_tables["fractional"]["raw"]["per_target"]
		self.assertEqual(set(fractional_raw_per_target["target"]), {"S_A", "S_B"})
		train_s_a_row = fractional_raw_per_target.loc[
			(fractional_raw_per_target["split"] == "train")
			& (fractional_raw_per_target["target"] == "S_A")
		].iloc[0]
		self.assertEqual(int(train_s_a_row["negative_predictions"]), 1)
		self.assertAlmostEqual(float(train_s_a_row["minimum_prediction"]), -0.4)

		composite_projected_per_target = negative_prediction_tables["composite"]["projected"]["per_target"]
		self.assertEqual(len(composite_projected_per_target), 4)
		self.assertTrue((composite_projected_per_target["negative_predictions"] == 0).all())

	def test_build_icsor_response_surface_prediction_data_uses_midpoint_profile_and_extended_domain(self) -> None:
		dataset_splits = make_train_test_split(
			self.icsor_dataset,
			test_fraction=0.25,
			random_seed=11,
		)
		result = run_icsor_pipeline(
			dataset_splits.train,
			dataset_splits.test,
			self.icsor_a_matrix,
			composition_matrix=self.icsor_composition_matrix,
			model_params=_tiny_icsor_params(),
			show_progress=False,
			persist_artifacts=False,
		)

		with tempfile.TemporaryDirectory() as temp_dir_name:
			model_path = tempfile.NamedTemporaryFile(dir=temp_dir_name, suffix=".pkl", delete=False)
			model_path.close()
			save_pickle_file(model_path.name, result["model_bundle"])
			response_surface = build_icsor_response_surface_prediction_data(
				model_path.name,
				metadata=self.icsor_metadata,
				grid_points_per_axis=7,
				operational_extension_fraction=0.5,
			)

		self.assertEqual(response_surface["response_surface_config"]["fixed_influent_profile"], "midpoint")
		self.assertEqual(response_surface["response_surface_config"]["grid_points_per_axis"], 7)
		self.assertAlmostEqual(
			response_surface["response_surface_config"]["operational_extension_fraction"],
			0.5,
		)
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
			set(f"Out_{name}" for name in self.icsor_metadata["measured_output_columns"]),
		)
		self.assertIn("Projected_Out_COD", response_surface["prediction_table"].columns)
		self.assertIn("ConstraintReference_S_A", response_surface["prediction_table"].columns)
		self.assertAlmostEqual(float(response_surface["fixed_influent_profile"].loc["S_A"]), 42.5)


if __name__ == "__main__":
	unittest.main()

