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
from matplotlib.patches import Rectangle


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

RAW_METRIC_COLUMNS = tuple(column_name.replace("projected_", "raw_", 1) for column_name in PROJECTED_METRIC_COLUMNS)
SUPPORTED_METRIC_COLUMNS = PROJECTED_METRIC_COLUMNS + RAW_METRIC_COLUMNS


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


def _format_panel_label(panel_name: str) -> str:
	panel_label = str(panel_name)
	if panel_label.startswith("Out_"):
		return panel_label.removeprefix("Out_")
	return panel_label


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


def _validate_surface_mesh(
	x_mesh: Any,
	y_mesh: Any,
	*,
	value_name: str,
) -> tuple[np.ndarray, np.ndarray]:
	x_array = _validate_coefficient_array(x_mesh, expected_ndim=2, value_name=f"{value_name}_x_mesh")
	y_array = _validate_coefficient_array(y_mesh, expected_ndim=2, value_name=f"{value_name}_y_mesh")
	if x_array.shape != y_array.shape:
		raise ValueError(f"{value_name} meshes must share the same shape.")
	return x_array, y_array


def _coerce_numeric_dataframe(
	frame: Any,
	*,
	frame_name: str,
) -> pd.DataFrame:
	if not isinstance(frame, pd.DataFrame):
		raise ValueError(f"{frame_name} must be a pandas DataFrame.")
	if frame.empty:
		raise ValueError(f"{frame_name} must contain at least one row.")

	try:
		resolved_frame = frame.astype(float).copy()
	except ValueError as exc:
		raise ValueError(f"{frame_name} must contain only numeric values.") from exc

	if not np.isfinite(resolved_frame.to_numpy(dtype=float)).all():
		raise ValueError(f"{frame_name} must contain only finite numeric values.")

	return resolved_frame


def _validate_parity_frames(
	train_actual: Any,
	train_predicted: Any,
	test_actual: Any,
	test_predicted: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
	train_actual_frame = _coerce_numeric_dataframe(train_actual, frame_name="train_actual")
	train_predicted_frame = _coerce_numeric_dataframe(train_predicted, frame_name="train_predicted")
	test_actual_frame = _coerce_numeric_dataframe(test_actual, frame_name="test_actual")
	test_predicted_frame = _coerce_numeric_dataframe(test_predicted, frame_name="test_predicted")

	if train_actual_frame.shape != train_predicted_frame.shape:
		raise ValueError("train_actual and train_predicted must share the same shape.")
	if test_actual_frame.shape != test_predicted_frame.shape:
		raise ValueError("test_actual and test_predicted must share the same shape.")
	if not train_actual_frame.index.equals(train_predicted_frame.index):
		raise ValueError("train_actual and train_predicted must share the same index.")
	if not test_actual_frame.index.equals(test_predicted_frame.index):
		raise ValueError("test_actual and test_predicted must share the same index.")
	if not train_actual_frame.columns.equals(train_predicted_frame.columns):
		raise ValueError("train_actual and train_predicted must share the same columns.")
	if not test_actual_frame.columns.equals(test_predicted_frame.columns):
		raise ValueError("test_actual and test_predicted must share the same columns.")
	if not train_actual_frame.columns.equals(test_actual_frame.columns):
		raise ValueError("Train and test parity frames must share the same columns.")

	return (
		train_actual_frame,
		train_predicted_frame,
		test_actual_frame,
		test_predicted_frame,
		[str(column_name) for column_name in train_actual_frame.columns],
	)


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


def plot_response_surface_contours(
	x_mesh: Any,
	y_mesh: Any,
	response_surfaces: dict[str, Any] | pd.Series,
	*,
	title: str,
	x_label: str,
	y_label: str,
	training_domain: dict[str, dict[str, float]] | None = None,
	contour_levels: int = 18,
	decimal_places: int = 2,
	figure_size_per_panel: tuple[float, float] = (4.8, 4.0),
	max_columns: int = 3,
) -> tuple[Any, np.ndarray]:
	"""Plot one filled contour response surface per target with repository-standard styling."""

	if contour_levels < 2:
		raise ValueError("contour_levels must be at least 2.")
	if decimal_places < 0:
		raise ValueError("decimal_places must be at least 0.")

	mesh_x, mesh_y = _validate_surface_mesh(x_mesh, y_mesh, value_name="response_surface")
	if isinstance(response_surfaces, pd.Series):
		surface_mapping = response_surfaces.to_dict()
	else:
		surface_mapping = dict(response_surfaces)
	if not surface_mapping:
		raise ValueError("response_surfaces must contain at least one target surface.")

	tokens = apply_pibre_plot_theme()
	target_labels = [str(target_name) for target_name in surface_mapping.keys()]
	row_count, column_count = _resolve_subplot_grid(len(target_labels), max_columns=max_columns)
	figure, axes = plt.subplots(
		row_count,
		column_count,
		figsize=(figure_size_per_panel[0] * column_count, figure_size_per_panel[1] * row_count),
		dpi=140,
		constrained_layout=True,
		squeeze=False,
	)
	active_axes: list[Any] = []
	colorbars: list[Any] = []
	filled_contours: list[Any] = []
	line_contours: list[Any] = []
	contour_labels: list[list[Any]] = []
	training_patches: list[Any] = []
	formatter_pattern = f"%.{decimal_places}f"

	for axis_index, axis in enumerate(axes.flat):
		if axis_index >= len(target_labels):
			axis.set_visible(False)
			continue

		target_label = target_labels[axis_index]
		surface_array = _validate_coefficient_array(
			surface_mapping[target_label],
			expected_ndim=2,
			value_name=f"response_surfaces['{target_label}']",
		)
		if surface_array.shape != mesh_x.shape:
			raise ValueError(
				f"response_surfaces['{target_label}'] must match the mesh shape {mesh_x.shape}."
			)

		active_axes.append(axis)
		filled = axis.contourf(
			mesh_x,
			mesh_y,
			surface_array,
			levels=int(contour_levels),
			cmap=tokens["sequential_colormap"],
		)
		lines = axis.contour(
			mesh_x,
			mesh_y,
			surface_array,
			levels=filled.levels,
			colors=tokens["primary_text"],
			linewidths=0.6,
			alpha=0.55,
		)
		labels = axis.clabel(
			lines,
			fmt=formatter_pattern,
			fontsize=8.0,
			inline=True,
			inline_spacing=3,
			colors=tokens["primary_text"],
		)
		filled_contours.append(filled)
		line_contours.append(lines)
		contour_labels.append(list(labels))
		colorbar = figure.colorbar(filled, ax=axis, shrink=0.9, pad=0.02)
		colorbar.set_label(_format_target_label(target_label))
		colorbar.formatter = mpl.ticker.FormatStrFormatter(formatter_pattern)
		colorbar.update_ticks()
		colorbar.ax.tick_params(colors=tokens["primary_text"])
		colorbars.append(colorbar)

		if training_domain is not None:
			training_patch = Rectangle(
				(
					float(training_domain["HRT"]["min"]),
					float(training_domain["Aeration"]["min"]),
				),
				float(training_domain["HRT"]["max"]) - float(training_domain["HRT"]["min"]),
				float(training_domain["Aeration"]["max"]) - float(training_domain["Aeration"]["min"]),
				fill=False,
				edgecolor=tokens["secondary_text"],
				linewidth=1.3,
				linestyle="--",
			)
			axis.add_patch(training_patch)
			training_patches.append(training_patch)

		axis.set_xlabel(x_label)
		axis.set_ylabel(y_label)
		axis.set_title(_format_target_label(target_label))
		axis.set_facecolor(tokens["axes_background"])
		axis.set_xlim(float(np.min(mesh_x)), float(np.max(mesh_x)))
		axis.set_ylim(float(np.min(mesh_y)), float(np.max(mesh_y)))
		axis.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(formatter_pattern))
		axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(formatter_pattern))
		axis.grid(which="major", color=tokens["major_grid"], alpha=0.35)
		axis.grid(which="minor", color=tokens["minor_grid"], alpha=0.2)
		axis.minorticks_on()

	figure.suptitle(title)
	setattr(
		figure,
		"_pibre_response_surface_contours",
		{
			"axes": active_axes,
			"colorbars": colorbars,
			"filled_contours": filled_contours,
			"line_contours": line_contours,
			"contour_labels": contour_labels,
			"training_patches": training_patches,
			"x_mesh": mesh_x,
			"y_mesh": mesh_y,
			"response_surfaces": {key: np.asarray(value, dtype=float) for key, value in surface_mapping.items()},
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
	"""Plot train and test boxplots of one raw or projected metric across training sizes."""

	if metric_name not in SUPPORTED_METRIC_COLUMNS:
		raise ValueError(
			f"metric_name must be one of {', '.join(SUPPORTED_METRIC_COLUMNS)}."
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


def plot_train_test_parity_panels(
	train_actual: Any,
	train_predicted: Any,
	test_actual: Any,
	test_predicted: Any,
	*,
	title: str,
	x_label: str = "Actual value",
	y_label: str = "Predicted value",
	train_label: str = "Train",
	test_label: str = "Test",
	parity_label: str = "Parity line",
	figure_size_per_panel: tuple[float, float] = (4.8, 4.1),
	max_columns: int = 3,
) -> tuple[Any, np.ndarray]:
	"""Plot one parity panel per column with shared train/test styling."""

	(
		train_actual_frame,
		train_predicted_frame,
		test_actual_frame,
		test_predicted_frame,
		column_names,
	) = _validate_parity_frames(
		train_actual,
		train_predicted,
		test_actual,
		test_predicted,
	)
	if max_columns < 1:
		raise ValueError("max_columns must be at least 1.")

	tokens = apply_pibre_plot_theme()
	row_count, column_count = _resolve_subplot_grid(len(column_names), max_columns=max_columns)
	figure, axes = plt.subplots(
		row_count,
		column_count,
		figsize=(figure_size_per_panel[0] * column_count, figure_size_per_panel[1] * row_count),
		dpi=140,
		constrained_layout=True,
		squeeze=False,
	)
	train_color = tokens["qualitative_cycle"][0]
	test_color = tokens["qualitative_cycle"][1]
	active_axes: list[Any] = []
	train_scatters: list[Any] = []
	test_scatters: list[Any] = []
	parity_lines: list[Any] = []

	for axis_index, axis in enumerate(axes.flat):
		if axis_index >= len(column_names):
			axis.set_visible(False)
			continue

		column_name = column_names[axis_index]
		active_axes.append(axis)
		train_x = train_actual_frame[column_name].to_numpy(dtype=float)
		train_y = train_predicted_frame[column_name].to_numpy(dtype=float)
		test_x = test_actual_frame[column_name].to_numpy(dtype=float)
		test_y = test_predicted_frame[column_name].to_numpy(dtype=float)
		combined_values = np.concatenate([train_x, train_y, test_x, test_y])
		minimum_value = float(np.min(combined_values))
		maximum_value = float(np.max(combined_values))
		value_span = maximum_value - minimum_value
		if value_span <= 0.0:
			padding = 1.0 if maximum_value == 0.0 else 0.05 * abs(maximum_value)
		else:
			padding = 0.04 * value_span
		lower_bound = minimum_value - padding
		upper_bound = maximum_value + padding

		train_scatter = axis.scatter(
			train_x,
			train_y,
			color=train_color,
			alpha=0.55,
			s=24.0,
			marker="o",
		)
		test_scatter = axis.scatter(
			test_x,
			test_y,
			color=test_color,
			alpha=0.72,
			s=26.0,
			marker="^",
		)
		parity_line = axis.plot(
			[lower_bound, upper_bound],
			[lower_bound, upper_bound],
			color=tokens["secondary_text"],
			linestyle="--",
			linewidth=1.2,
		)[0]

		axis.set_xlim(lower_bound, upper_bound)
		axis.set_ylim(lower_bound, upper_bound)
		axis.set_aspect("equal", adjustable="box")
		axis.set_xlabel(x_label)
		axis.set_ylabel(y_label)
		axis.set_title(_format_panel_label(column_name))
		axis.set_facecolor(tokens["axes_background"])
		axis.grid(which="major", color=tokens["major_grid"], alpha=0.45)
		axis.grid(which="minor", color=tokens["minor_grid"], alpha=0.3)
		axis.minorticks_on()

		train_scatters.append(train_scatter)
		test_scatters.append(test_scatter)
		parity_lines.append(parity_line)

	legend_handles = [
		Line2D([], [], color=train_color, marker="o", linestyle="None", markersize=6.0, label=train_label),
		Line2D([], [], color=test_color, marker="^", linestyle="None", markersize=6.0, label=test_label),
		Line2D([], [], color=tokens["secondary_text"], linestyle="--", linewidth=1.2, label=parity_label),
	]
	legend = figure.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
	figure.suptitle(title)
	setattr(
		figure,
		"_pibre_train_test_parity",
		{
			"axes": active_axes,
			"legend": legend,
			"train_scatters": train_scatters,
			"test_scatters": test_scatters,
			"parity_lines": parity_lines,
			"column_names": column_names,
		},
	)

	return figure, axes


__all__ = [
	"PIBRE_THEME_TOKENS",
	"PROJECTED_METRIC_COLUMNS",
	"RAW_METRIC_COLUMNS",
	"SUPPORTED_METRIC_COLUMNS",
	"apply_pibre_plot_theme",
	"plot_coefficient_bar_chart",
	"plot_coefficient_heatmap",
	"plot_coefficient_tensor_heatmaps",
	"plot_response_surface_contours",
	"plot_train_test_parity_panels",
	"plot_train_test_metric_boxplots",
]