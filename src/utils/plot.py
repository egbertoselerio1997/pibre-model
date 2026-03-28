"""Reusable plotting helpers for repository-standard figures."""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
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
	"plot_train_test_metric_boxplots",
]