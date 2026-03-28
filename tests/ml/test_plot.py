"""Tests for repository-standard analysis plots."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.plot import apply_pibre_plot_theme, plot_train_test_metric_boxplots


def _build_metric_frame() -> pd.DataFrame:
	rows = []
	for train_size in [80, 344, 608]:
		for split_name, offset in [("train", 0.0), ("test", 0.1)]:
			for repeat_index in range(4):
				rows.append(
					{
						"model_name": "synthetic_model",
						"dataset_size_total": train_size + 20,
						"repeat_index": repeat_index,
						"train_size": train_size,
						"test_size": 20,
						"run_seed": 10 + repeat_index,
						"split_name": split_name,
						"target": "Out_A",
						"projected_R2": 0.8 + offset + 0.01 * repeat_index,
						"projected_MSE": 0.2 + offset + 0.01 * repeat_index,
						"projected_RMSE": 0.4 + offset + 0.01 * repeat_index,
						"projected_MAE": 0.3 + offset + 0.01 * repeat_index,
						"projected_MAPE": 0.1 + offset + 0.01 * repeat_index,
					}
				)
	rows.append(
		{
			"model_name": "synthetic_model",
			"dataset_size_total": 100,
			"repeat_index": 99,
			"train_size": 80,
			"test_size": 20,
			"run_seed": 99,
			"split_name": "train",
			"target": "Out_A",
			"projected_R2": 1.9,
			"projected_MSE": 1.0,
			"projected_RMSE": 1.0,
			"projected_MAE": 1.0,
			"projected_MAPE": 1.0,
		}
	)
	return pd.DataFrame(rows)


class PlotHelperTests(unittest.TestCase):
	def tearDown(self) -> None:
		plt.close("all")

	def test_apply_pibre_plot_theme_sets_expected_defaults(self) -> None:
		apply_pibre_plot_theme()

		self.assertEqual(matplotlib.rcParams["figure.facecolor"], "#F7F4EA")
		self.assertEqual(matplotlib.rcParams["axes.facecolor"], "#FFFFFF")
		self.assertEqual(matplotlib.rcParams["image.cmap"], "cividis")
		self.assertEqual(matplotlib.rcParams["lines.linewidth"], 2.0)

	def test_plot_train_test_metric_boxplots_returns_mean_overlays_and_fliers(self) -> None:
		metric_frame = _build_metric_frame()

		figure, axis = plot_train_test_metric_boxplots(
			metric_frame,
			metric_name="projected_R2",
			target_name="Out_A",
			model_name="Synthetic Model",
		)

		artist_bundle = getattr(axis, "_pibre_metric_boxplot")
		self.assertIs(figure, axis.figure)
		self.assertEqual(artist_bundle["train_mean_line"].get_label(), "Train mean")
		self.assertEqual(artist_bundle["test_mean_line"].get_label(), "Test mean")
		self.assertTrue(any(flier.get_marker() == "o" for flier in artist_bundle["train"]["fliers"]))
		self.assertEqual(axis.get_xlabel(), "Number of training samples")

	def test_plot_train_test_metric_boxplots_rejects_unknown_metric(self) -> None:
		metric_frame = _build_metric_frame()

		with self.assertRaises(ValueError):
			plot_train_test_metric_boxplots(
				metric_frame,
				metric_name="raw_R2",
				target_name="Out_A",
			)


if __name__ == "__main__":
	unittest.main()