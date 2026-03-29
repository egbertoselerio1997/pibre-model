"""Reusable plotting helpers for repository-standard figures."""

from __future__ import annotations

import math
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PIBRE_THEME_TOKENS: dict[str, Any] = {
	"figure_background": "#F7F4EA",
	"axes_background": "#FFFFFF",
	"primary_text": "#22303C",
	"secondary_text": "#5B6770",
	"major_grid": "#D8DEE6",
	"minor_grid": "#EEF2F5",
	"qualitative_cycle": [
		"#0072B2",
		"#E69F00",
		"#009E73",
		"#D55E00",
		"#CC79A7",
		"#56B4E9",
		"#F0E442",
		"#4D4D4D",
	],
	"missing_color": "#B0B7BF",
}

PROJECTED_METRIC_COLUMNS = (
	"projected_R2",
	"projected_MSE",
	"projected_RMSE",
	"projected_MAE",
	"projected_MAPE",
)


def _build_diverging_colormap() -> LinearSegmentedColormap:
	return LinearSegmentedColormap.from_list(
		"pibre_blue_white_vermilion",
		["#0072B2", "#F7F7F7", "#D55E00"],
	)


def apply_pibre_plot_theme() -> dict[str, Any]:
	"""Apply the repository-wide Pibre Scientific plotting theme."""

	tokens = dict(PIBRE_THEME_TOKENS)
	mpl.rcParams.update(
		{
			"figure.facecolor": tokens["figure_background"],
			"figure.dpi": 140,
			"axes.facecolor": tokens["axes_background"],
			"axes.edgecolor": tokens["primary_text"],
			"axes.labelcolor": tokens["primary_text"],
			"axes.titlecolor": tokens["primary_text"],
			"axes.grid": True,
			"axes.axisbelow": True,
			"axes.spines.top": False,
			"axes.spines.right": False,
			"axes.linewidth": 0.8,
			"axes.prop_cycle": cycler(color=tokens["qualitative_cycle"]),
			"font.family": ["DejaVu Sans"],
			"font.sans-serif": ["DejaVu Sans"],
			"grid.color": tokens["major_grid"],
			"grid.alpha": 0.45,
			"grid.linewidth": 0.8,
			"image.cmap": "cividis",
			"legend.facecolor": tokens["axes_background"],
			"legend.edgecolor": tokens["major_grid"],
			"legend.frameon": True,
			"legend.fontsize": 10,
			"lines.linewidth": 2.0,
			"lines.markeredgewidth": 0.75,
			"savefig.dpi": 140,
			"savefig.facecolor": tokens["figure_background"],
			"text.color": tokens["primary_text"],
			"xtick.color": tokens["primary_text"],
			"ytick.color": tokens["primary_text"],
		}
	)
	tokens["diverging_colormap"] = _build_diverging_colormap()
	tokens["sequential_colormap"] = plt.get_cmap("cividis")
	return tokens


def _format_metric_label(metric_name: str) -> str:
	parts = metric_name.split("_", 1)
	if len(parts) == 2:
		prefix, metric = parts
		return f"{prefix.capitalize()} {metric}"
	return metric_name.replace("_", " ").title()


def _format_target_label(target_name: str) -> str:
	return target_name.replace("Out_", "", 1).replace("_", " ")


def _validate_label_count(labels: list[str], *, expected_size: int, label_name: str) -> list[str]:
	label_list = [str(label) for label in labels]
	if len(label_list) != expected_size:
		raise ValueError(f"{label_name} must contain exactly {expected_size} labels.")
	return label_list


def _validate_coefficient_array(
	coefficient_values: Any,
	*,
	expected_ndim: int,
	value_name: str,
) -> np.ndarray:
	coefficient_array = np.asarray(coefficient_values, dtype=float)
	if coefficient_array.ndim != expected_ndim:
		raise ValueError(f"{value_name} must be a {expected_ndim}D numeric array.")
	if not np.isfinite(coefficient_array).all():
		raise ValueError(f"{value_name} must contain only finite numeric values.")
	return coefficient_array


def _build_centered_diverging_norm(coefficient_values: np.ndarray) -> TwoSlopeNorm:
	max_magnitude = float(np.max(np.abs(coefficient_values)))
	if max_magnitude <= 0.0:
		max_magnitude = 1.0
	return TwoSlopeNorm(vmin=-max_magnitude, vcenter=0.0, vmax=max_magnitude)


def _resolve_subplot_grid(panel_count: int, *, max_columns: int) -> tuple[int, int]:
	if panel_count <= 0:
		raise ValueError("panel_count must be positive.")
	column_count = min(max_columns, max(1, math.ceil(math.sqrt(panel_count))))
	row_count = math.ceil(panel_count / column_count)
	return row_count, column_count


def plot_coefficient_heatmap(
	coefficient_values: Any,
	*,
	row_labels: list[str],
	column_labels: list[str],
	title: str,
	x_label: str,
	y_label: str,
	colorbar_label: str = "Coefficient value",
	ax: Any | None = None,
	figure_size: tuple[float, float] = (10.0, 6.0),
	x_tick_rotation: float = 45.0,
) -> tuple[Any, Any]:
	"""Plot a coefficient heatmap with repository-standard styling."""

	tokens = apply_pibre_plot_theme()
	coefficient_matrix = _validate_coefficient_array(
		coefficient_values,
		expected_ndim=2,
		value_name="coefficient_values",
	)
	row_label_list = _validate_label_count(
		row_labels,
		expected_size=coefficient_matrix.shape[0],
		label_name="row_labels",
	)
	column_label_list = _validate_label_count(
		column_labels,
		expected_size=coefficient_matrix.shape[1],
		label_name="column_labels",
	)

	if ax is None:
		figure, ax = plt.subplots(figsize=figure_size, dpi=140, constrained_layout=True)
	else:
		figure = ax.figure

	image = ax.imshow(
		coefficient_matrix,
		aspect="auto",
		cmap=tokens["diverging_colormap"],
		norm=_build_centered_diverging_norm(coefficient_matrix),
		origin="lower",
		interpolation="nearest",
	)
	colorbar = figure.colorbar(image, ax=ax)
	colorbar.set_label(colorbar_label)
	colorbar.ax.tick_params(colors=tokens["primary_text"])

	ax.set_xticks(np.arange(len(column_label_list), dtype=float))
	ax.set_xticklabels(column_label_list, rotation=x_tick_rotation, ha="right")
	ax.set_yticks(np.arange(len(row_label_list), dtype=float))
	ax.set_yticklabels(row_label_list)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_title(title)
	ax.set_facecolor(tokens["axes_background"])
	ax.grid(False)
	setattr(
		ax,
		"_pibre_coefficient_heatmap",
		{"image": image, "colorbar": colorbar, "values": coefficient_matrix},
	)
	return figure, ax


def plot_coefficient_bar_chart(
	coefficient_values: Any,
	*,
	labels: list[str],
	title: str,
	x_label: str,
	y_label: str,
	ax: Any | None = None,
	figure_size: tuple[float, float] = (10.0, 5.5),
	x_tick_rotation: float = 45.0,
) -> tuple[Any, Any]:
	"""Plot a coefficient bar chart with repository-standard styling."""

	tokens = apply_pibre_plot_theme()
	coefficient_vector = _validate_coefficient_array(
		coefficient_values,
		expected_ndim=1,
		value_name="coefficient_values",
	)
	label_list = _validate_label_count(
		labels,
		expected_size=coefficient_vector.shape[0],
		label_name="labels",
	)

	if ax is None:
		figure, ax = plt.subplots(figsize=figure_size, dpi=140, constrained_layout=True)
	else:
		figure = ax.figure

	positions = np.arange(len(label_list), dtype=float)
	bar_container = ax.bar(
		positions,
		coefficient_vector,
		color=tokens["qualitative_cycle"][0],
		edgecolor=tokens["primary_text"],
		alpha=0.82,
		linewidth=0.7,
	)
	ax.axhline(0.0, color=tokens["secondary_text"], linewidth=1.0, linestyle="--")
	ax.set_xticks(positions)
	ax.set_xticklabels(label_list, rotation=x_tick_rotation, ha="right")
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_title(title)
	ax.grid(axis="y", which="major", color=tokens["major_grid"], alpha=0.45)
	ax.grid(axis="x", which="major", visible=False)
	setattr(
		ax,
		"_pibre_coefficient_bar_chart",
		{"bars": list(bar_container), "values": coefficient_vector},
	)
	return figure, ax


def plot_coefficient_tensor_heatmaps(
	coefficient_values: Any,
	*,
	target_labels: list[str],
	row_labels: list[str],
	column_labels: list[str],
	title: str,
	x_label: str,
	y_label: str,
	colorbar_label: str = "Coefficient value",
	figure_size_per_panel: tuple[float, float] = (4.6, 3.8),
	max_columns: int = 3,
	x_tick_rotation: float = 45.0,
) -> tuple[Any, np.ndarray]:
	"""Plot one coefficient heatmap per target for a rank-3 tensor."""

	tokens = apply_pibre_plot_theme()
	coefficient_tensor = _validate_coefficient_array(
		coefficient_values,
		expected_ndim=3,
		value_name="coefficient_values",
	)
	target_label_list = _validate_label_count(
		target_labels,
		expected_size=coefficient_tensor.shape[0],
		label_name="target_labels",
	)
	row_label_list = _validate_label_count(
		row_labels,
		expected_size=coefficient_tensor.shape[1],
		label_name="row_labels",
	)
	column_label_list = _validate_label_count(
		column_labels,
		expected_size=coefficient_tensor.shape[2],
		label_name="column_labels",
	)
	row_count, column_count = _resolve_subplot_grid(coefficient_tensor.shape[0], max_columns=max_columns)
	figure, axes = plt.subplots(
		row_count,
		column_count,
		figsize=(figure_size_per_panel[0] * column_count, figure_size_per_panel[1] * row_count),
		dpi=140,
		constrained_layout=True,
		squeeze=False,
	)
	norm = _build_centered_diverging_norm(coefficient_tensor)
	active_axes: list[Any] = []
	last_image = None

	for axis_index, axis in enumerate(axes.flat):
		if axis_index >= coefficient_tensor.shape[0]:
			axis.set_visible(False)
			continue
		active_axes.append(axis)
		image = axis.imshow(
			coefficient_tensor[axis_index],
			aspect="auto",
			cmap=tokens["diverging_colormap"],
			norm=norm,
			origin="lower",
			interpolation="nearest",
		)
		last_image = image
		axis.set_xticks(np.arange(len(column_label_list), dtype=float))
		axis.set_xticklabels(column_label_list, rotation=x_tick_rotation, ha="right")
		axis.set_yticks(np.arange(len(row_label_list), dtype=float))
		axis.set_yticklabels(row_label_list)
		axis.set_xlabel(x_label)
		axis.set_ylabel(y_label)
		axis.set_title(target_label_list[axis_index])
		axis.set_facecolor(tokens["axes_background"])
		axis.grid(False)

	if last_image is None:
		raise ValueError("coefficient_values must contain at least one target panel.")

	colorbar = figure.colorbar(last_image, ax=active_axes, shrink=0.92, pad=0.02)
	colorbar.set_label(colorbar_label)
	colorbar.ax.tick_params(colors=tokens["primary_text"])
	figure.suptitle(title)
	setattr(
		figure,
		"_pibre_coefficient_tensor_heatmaps",
		{
			"axes": active_axes,
			"colorbar": colorbar,
			"values": coefficient_tensor,
		},
	)
	return figure, axes


def plot_train_test_metric_boxplots(
	metric_frame: pd.DataFrame,
	*,
	metric_name: str,
	target_name: str,
	model_name: str | None = None,
	ax: Any | None = None,
	figure_size: tuple[float, float] = (12.0, 6.5),
) -> tuple[Any, Any]:
	"""Plot train and test boxplots of one projected metric across training sizes."""

	if metric_name not in PROJECTED_METRIC_COLUMNS:
		raise ValueError(
			f"metric_name must be one of {', '.join(PROJECTED_METRIC_COLUMNS)}."
		)

	required_columns = {"target", "split_name", "train_size", metric_name}
	missing_columns = sorted(required_columns.difference(metric_frame.columns))
	if missing_columns:
		missing_display = ", ".join(missing_columns)
		raise KeyError(f"metric_frame is missing required columns: {missing_display}")

	apply_pibre_plot_theme()
	filtered_frame = metric_frame.loc[metric_frame["target"] == target_name].copy()
	if filtered_frame.empty:
		raise ValueError(f"No metric rows were found for target '{target_name}'.")

	train_sizes = sorted(int(value) for value in filtered_frame["train_size"].unique())
	position_index = np.arange(len(train_sizes), dtype=float)
	offset = 0.18
	box_width = 0.3

	train_values = [
		filtered_frame.loc[
			(filtered_frame["split_name"] == "train") & (filtered_frame["train_size"] == train_size),
			metric_name,
		].to_numpy(dtype=float)
		for train_size in train_sizes
	]
	test_values = [
		filtered_frame.loc[
			(filtered_frame["split_name"] == "test") & (filtered_frame["train_size"] == train_size),
			metric_name,
		].to_numpy(dtype=float)
		for train_size in train_sizes
	]
	if any(len(values) == 0 for values in train_values) or any(len(values) == 0 for values in test_values):
		raise ValueError("Train and test distributions must both be present for every training-size group.")

	tokens = apply_pibre_plot_theme()
	train_color = tokens["qualitative_cycle"][0]
	test_color = tokens["qualitative_cycle"][1]

	if ax is None:
		figure, ax = plt.subplots(figsize=figure_size, dpi=140, constrained_layout=True)
	else:
		figure = ax.figure

	common_kwargs = {
		"patch_artist": True,
		"showmeans": True,
		"showfliers": True,
		"whis": 1.5,
		"manage_ticks": False,
		"medianprops": {"color": tokens["primary_text"], "linewidth": 1.2},
		"whiskerprops": {"color": tokens["secondary_text"], "linewidth": 1.0},
		"capprops": {"color": tokens["secondary_text"], "linewidth": 1.0},
		"meanprops": {"marker": "D", "markeredgecolor": tokens["primary_text"], "markerfacecolor": tokens["axes_background"], "markersize": 6.0},
	}
	train_box = ax.boxplot(
		train_values,
		positions=position_index - offset,
		widths=box_width,
		boxprops={"facecolor": train_color, "alpha": 0.35, "edgecolor": train_color, "linewidth": 1.2},
		flierprops={"marker": "o", "markersize": 4.5, "markerfacecolor": train_color, "markeredgecolor": train_color, "alpha": 0.6},
		**common_kwargs,
	)
	test_box = ax.boxplot(
		test_values,
		positions=position_index + offset,
		widths=box_width,
		boxprops={"facecolor": test_color, "alpha": 0.35, "edgecolor": test_color, "linewidth": 1.2},
		flierprops={"marker": "o", "markersize": 4.5, "markerfacecolor": test_color, "markeredgecolor": test_color, "alpha": 0.6},
		**common_kwargs,
	)

	train_mean_line = ax.plot(
		position_index - offset,
		[np.mean(values) for values in train_values],
		color=train_color,
		linestyle="-",
		marker="s",
		markersize=5.0,
		label="Train mean",
	)[0]
	test_mean_line = ax.plot(
		position_index + offset,
		[np.mean(values) for values in test_values],
		color=test_color,
		linestyle="--",
		marker="s",
		markersize=5.0,
		label="Test mean",
	)[0]

	ax.set_xticks(position_index)
	ax.set_xticklabels([str(train_size) for train_size in train_sizes], rotation=45, ha="right")
	ax.set_xlabel("Number of training samples")
	ax.set_ylabel(_format_metric_label(metric_name))
	title_prefix = _format_target_label(target_name)
	if model_name is not None:
		title = f"{model_name} {_format_metric_label(metric_name)} across training sizes for {title_prefix}"
	else:
		title = f"{_format_metric_label(metric_name)} across training sizes for {title_prefix}"
	ax.set_title(title)
	ax.set_facecolor(tokens["axes_background"])
	ax.grid(axis="y", which="major", color=tokens["major_grid"], alpha=0.45)
	ax.grid(axis="y", which="minor", color=tokens["minor_grid"], alpha=0.35)
	ax.minorticks_on()
	legend_handles = [
		Patch(facecolor=train_color, edgecolor=train_color, alpha=0.35, label="Train distribution"),
		Patch(facecolor=test_color, edgecolor=test_color, alpha=0.35, label="Test distribution"),
		Line2D([], [], color=train_color, linestyle="-", marker="s", label="Train mean"),
		Line2D([], [], color=test_color, linestyle="--", marker="s", label="Test mean"),
	]
	ax.legend(handles=legend_handles, loc="best")
	setattr(
		ax,
		"_pibre_metric_boxplot",
		{
			"train": train_box,
			"test": test_box,
			"train_mean_line": train_mean_line,
			"test_mean_line": test_mean_line,
		},
	)

	return figure, ax


__all__ = [
	"PIBRE_THEME_TOKENS",
	"PROJECTED_METRIC_COLUMNS",
	"apply_pibre_plot_theme",
	"plot_coefficient_bar_chart",
	"plot_coefficient_heatmap",
	"plot_coefficient_tensor_heatmaps",
	"plot_train_test_metric_boxplots",
]