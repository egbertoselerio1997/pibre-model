"""Tests for repository-standard analysis plots."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plot import (
	apply_pibre_plot_theme,
	plot_coefficient_bar_chart,
	plot_coefficient_heatmap,
	plot_coefficient_tensor_heatmaps,
	plot_response_surface_contours,
	plot_train_test_parity_panels,
	plot_train_test_metric_boxplots,
)


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
						"raw_R2": 0.75 + offset + 0.01 * repeat_index,
						"raw_MSE": 0.25 + offset + 0.01 * repeat_index,
						"raw_RMSE": 0.45 + offset + 0.01 * repeat_index,
						"raw_MAE": 0.35 + offset + 0.01 * repeat_index,
						"raw_MAPE": 0.15 + offset + 0.01 * repeat_index,
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
			"raw_R2": 1.8,
			"raw_MSE": 1.1,
			"raw_RMSE": 1.1,
			"raw_MAE": 1.1,
			"raw_MAPE": 1.1,
			"projected_R2": 1.9,
			"projected_MSE": 1.0,
			"projected_RMSE": 1.0,
			"projected_MAE": 1.0,
			"projected_MAPE": 1.0,
		}
	)
	return pd.DataFrame(rows)


def _build_parity_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	train_index = pd.Index([0, 1, 2], name="sample_id")
	test_index = pd.Index([10, 11], name="sample_id")
	train_actual = pd.DataFrame(
		{
			"Out_COD": [100.0, 120.0, 140.0],
			"Out_TN": [20.0, 25.0, 30.0],
		},
		index=train_index,
	)
	train_predicted = pd.DataFrame(
		{
			"Out_COD": [102.0, 118.0, 143.0],
			"Out_TN": [19.5, 25.5, 29.5],
		},
		index=train_index,
	)
	test_actual = pd.DataFrame(
		{
			"Out_COD": [110.0, 150.0],
			"Out_TN": [22.0, 33.0],
		},
		index=test_index,
	)
	test_predicted = pd.DataFrame(
		{
			"Out_COD": [111.0, 147.0],
			"Out_TN": [21.5, 34.0],
		},
		index=test_index,
	)
	return train_actual, train_predicted, test_actual, test_predicted


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

	def test_plot_train_test_metric_boxplots_accepts_raw_metric(self) -> None:
		metric_frame = _build_metric_frame()

		figure, axis = plot_train_test_metric_boxplots(
			metric_frame,
			metric_name="raw_R2",
			target_name="Out_A",
			model_name="Synthetic Model",
		)

		artist_bundle = getattr(axis, "_pibre_metric_boxplot")
		self.assertIs(figure, axis.figure)
		self.assertEqual(artist_bundle["train_mean_line"].get_label(), "Train mean")

	def test_plot_train_test_parity_panels_returns_one_panel_per_column(self) -> None:
		train_actual, train_predicted, test_actual, test_predicted = _build_parity_frames()

		figure, axes = plot_train_test_parity_panels(
			train_actual,
			train_predicted,
			test_actual,
			test_predicted,
			title="COBRE projected parity plots",
			x_label="Actual value",
			y_label="Projected prediction",
		)

		artist_bundle = getattr(figure, "_pibre_train_test_parity")
		legend_text = [text.get_text() for text in figure.legends[0].texts]
		self.assertEqual(axes.shape, (1, 2))
		self.assertEqual(len(artist_bundle["axes"]), 2)
		self.assertEqual(len(artist_bundle["train_scatters"]), 2)
		self.assertEqual(len(artist_bundle["test_scatters"]), 2)
		self.assertEqual(len(artist_bundle["parity_lines"]), 2)
		self.assertEqual(legend_text, ["Train", "Test", "Parity line"])
		self.assertEqual(artist_bundle["parity_lines"][0].get_linestyle(), "--")
		self.assertEqual(artist_bundle["axes"][0].get_xlabel(), "Actual value")
		self.assertEqual(artist_bundle["axes"][0].get_ylabel(), "Projected prediction")
		self.assertEqual(artist_bundle["axes"][0].get_title(), "COD")
		self.assertEqual(artist_bundle["axes"][1].get_title(), "TN")

	def test_plot_train_test_metric_boxplots_rejects_unknown_metric(self) -> None:
		metric_frame = _build_metric_frame()

		with self.assertRaises(ValueError):
			plot_train_test_metric_boxplots(
				metric_frame,
				metric_name="constraint_R2",
				target_name="Out_A",
			)

	def test_plot_coefficient_heatmap_returns_colorbar_and_labels(self) -> None:
		figure, axis = plot_coefficient_heatmap(
			np.array([[0.2, -0.1, 0.0], [0.4, -0.3, 0.1]], dtype=float),
			row_labels=["Out_A", "Out_B"],
			column_labels=["Flow", "Aeration", "Recycle"],
			title="Effective operational coefficients",
			x_label="Operational variable",
			y_label="Measured target",
		)

		artist_bundle = getattr(axis, "_pibre_coefficient_heatmap")
		self.assertIs(figure, axis.figure)
		self.assertEqual(artist_bundle["values"].shape, (2, 3))
		self.assertEqual(artist_bundle["image"].origin, "lower")
		self.assertEqual(axis.get_xlabel(), "Operational variable")
		self.assertEqual(axis.get_ylabel(), "Measured target")
		self.assertEqual(len(figure.axes), 2)

	def test_plot_coefficient_bar_chart_returns_expected_number_of_bars(self) -> None:
		figure, axis = plot_coefficient_bar_chart(
			np.array([0.4, -0.2, 0.1], dtype=float),
			labels=["Out_A", "Out_B", "Out_C"],
			title="Effective bias coefficients",
			x_label="Measured target",
			y_label="Coefficient value",
		)

		artist_bundle = getattr(axis, "_pibre_coefficient_bar_chart")
		self.assertIs(figure, axis.figure)
		self.assertEqual(len(artist_bundle["bars"]), 3)
		self.assertEqual(axis.get_xlabel(), "Measured target")
		self.assertEqual(axis.get_ylabel(), "Coefficient value")

	def test_plot_coefficient_tensor_heatmaps_returns_one_subplot_per_target(self) -> None:
		figure, axes = plot_coefficient_tensor_heatmaps(
			np.array(
				[
					[[0.2, -0.1], [0.0, 0.3]],
					[[0.1, 0.4], [-0.2, -0.3]],
				],
				dtype=float,
			),
			target_labels=["Out_A", "Out_B"],
			row_labels=["Flow", "Aeration"],
			column_labels=["Flow", "Aeration"],
			title="Operational interaction coefficients",
			x_label="Operational variable",
			y_label="Operational variable",
		)

		artist_bundle = getattr(figure, "_pibre_coefficient_tensor_heatmaps")
		self.assertEqual(len(artist_bundle["axes"]), 2)
		self.assertEqual(len(figure.axes), 3)
		self.assertEqual(axes.shape, (1, 2))
		self.assertEqual(artist_bundle["axes"][0].images[0].origin, "lower")
		self.assertEqual(artist_bundle["axes"][0].get_title(), "Out_A")
		self.assertEqual(artist_bundle["axes"][1].get_title(), "Out_B")

	def test_plot_coefficient_tensor_heatmaps_rejects_target_label_mismatch(self) -> None:
		with self.assertRaises(ValueError):
			plot_coefficient_tensor_heatmaps(
				np.ones((2, 2, 2), dtype=float),
				target_labels=["Out_A"],
				row_labels=["Flow", "Aeration"],
				column_labels=["Flow", "Aeration"],
				title="Operational interaction coefficients",
				x_label="Operational variable",
				y_label="Operational variable",
			)

	def test_plot_response_surface_contours_returns_one_panel_per_target(self) -> None:
		hrt_mesh, aeration_mesh = np.meshgrid(
			np.linspace(-9.0, 51.0, 5, dtype=float),
			np.linspace(-0.5, 3.5, 4, dtype=float),
		)
		figure, axes = plot_response_surface_contours(
			hrt_mesh,
			aeration_mesh,
			{
				"Out_COD": hrt_mesh + 2.0 * aeration_mesh,
				"Out_TN": hrt_mesh - aeration_mesh,
			},
			title="COBRE operational response surfaces",
			x_label="HRT",
			y_label="Aeration",
			training_domain={
				"HRT": {"min": 6.0, "max": 36.0},
				"Aeration": {"min": 0.5, "max": 2.5},
			},
			contour_levels=9,
		)

		artist_bundle = getattr(figure, "_pibre_response_surface_contours")
		self.assertEqual(axes.shape, (1, 2))
		self.assertEqual(len(artist_bundle["axes"]), 2)
		self.assertEqual(len(artist_bundle["colorbars"]), 2)
		self.assertEqual(len(artist_bundle["contour_labels"]), 2)
		self.assertEqual(len(artist_bundle["training_patches"]), 2)
		self.assertEqual(artist_bundle["axes"][0].get_xlabel(), "HRT")
		self.assertEqual(artist_bundle["axes"][0].get_ylabel(), "Aeration")
		self.assertEqual(artist_bundle["axes"][0].get_title(), "COD")
		self.assertEqual(artist_bundle["colorbars"][0].ax.get_ylabel(), "COD")
		self.assertGreater(len(artist_bundle["contour_labels"][0]), 0)
		self.assertIsInstance(
			artist_bundle["axes"][0].xaxis.get_major_formatter(),
			matplotlib.ticker.FormatStrFormatter,
		)
		self.assertIsInstance(
			artist_bundle["axes"][0].yaxis.get_major_formatter(),
			matplotlib.ticker.FormatStrFormatter,
		)
		self.assertIsInstance(
			artist_bundle["colorbars"][0].ax.yaxis.get_major_formatter(),
			matplotlib.ticker.FormatStrFormatter,
		)
		self.assertEqual(artist_bundle["axes"][0].xaxis.get_major_formatter().fmt, "%.2f")
		self.assertEqual(artist_bundle["axes"][0].yaxis.get_major_formatter().fmt, "%.2f")
		self.assertEqual(artist_bundle["colorbars"][0].ax.yaxis.get_major_formatter().fmt, "%.2f")
		first_label = artist_bundle["contour_labels"][0][0].get_text()
		self.assertRegex(first_label, r"^-?\d+\.\d{2}$")
		self.assertEqual(len(figure.axes), 4)

	def test_plot_response_surface_contours_rejects_shape_mismatch(self) -> None:
		hrt_mesh, aeration_mesh = np.meshgrid(
			np.linspace(0.0, 1.0, 4, dtype=float),
			np.linspace(0.0, 1.0, 4, dtype=float),
		)

		with self.assertRaises(ValueError):
			plot_response_surface_contours(
				hrt_mesh,
				aeration_mesh,
				{"Out_COD": np.ones((3, 3), dtype=float)},
				title="Invalid response surfaces",
				x_label="HRT",
				y_label="Aeration",
			)


if __name__ == "__main__":
	unittest.main()