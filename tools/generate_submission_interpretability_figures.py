from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "DCHE-D-26-00020"
FOOTER_TEXT = "Illustrative sample layout only; replace with final benchmark exports."

OPERATIONAL_LABELS = ["HRT", "Aeration"]

ASM_COMPONENT_LABELS = [
	"S_O",
	"S_F",
	"S_A",
	"S_NH4",
	"S_NO2",
	"S_NO3",
	"S_N2",
	"S_PO4",
	"S_I",
	"S_ALK",
	"X_I",
	"X_S",
	"X_H",
	"X_PAO",
	"X_PP",
	"X_PHA",
	"X_AOB",
	"X_NOB",
	"X_MeP",
	"X_MeOH",
]

MAIN_TEXT_FILE_NAME = "figure4_icsor_structure.pdf"

SUPPLEMENTARY_FILE_NAMES = {
	"COD": "figureS1_cod_icsor_structure.pdf",
	"TN": "figureS2_tn_icsor_structure.pdf",
	"TP": "figureS3_tp_icsor_structure.pdf",
	"TSS": "figureS4_tss_icsor_structure.pdf",
}

PROFILE_BY_TARGET = {
	"COD": np.array(
		[
			-0.18,
			0.62,
			0.54,
			-0.12,
			-0.08,
			-0.06,
			-0.02,
			-0.05,
			0.35,
			0.04,
			0.41,
			0.58,
			0.46,
			0.33,
			0.08,
			0.31,
			0.27,
			0.22,
			0.05,
			-0.01,
		],
		dtype=float,
	),
	"TN": np.array(
		[
			-0.05,
			0.11,
			0.08,
			0.71,
			0.56,
			0.61,
			0.18,
			0.02,
			0.07,
			-0.09,
			0.14,
			0.18,
			0.29,
			0.22,
			0.05,
			0.04,
			0.47,
			0.41,
			0.03,
			-0.01,
		],
		dtype=float,
	),
	"TP": np.array(
		[
			-0.04,
			0.06,
			0.07,
			0.04,
			0.02,
			0.02,
			0.01,
			0.74,
			0.03,
			-0.02,
			0.09,
			0.14,
			0.18,
			0.46,
			0.68,
			0.29,
			0.15,
			0.12,
			0.58,
			0.02,
		],
		dtype=float,
	),
	"TSS": np.array(
		[
			-0.02,
			0.02,
			0.03,
			0.01,
			0.00,
			0.01,
			0.00,
			0.03,
			0.06,
			0.00,
			0.63,
			0.71,
			0.69,
			0.42,
			0.39,
			0.34,
			0.28,
			0.26,
			0.57,
			0.48,
		],
		dtype=float,
	),
}

BIAS_BY_TARGET = {
	"COD": 0.14,
	"TN": 0.09,
	"TP": 0.07,
	"TSS": 0.12,
}

W_U_BY_TARGET = {
	"COD": np.array([0.28, -0.52], dtype=float),
	"TN": np.array([-0.36, -0.44], dtype=float),
	"TP": np.array([-0.18, -0.21], dtype=float),
	"TSS": np.array([0.17, 0.05], dtype=float),
}

THETA_UU_BY_TARGET = {
	"COD": np.array([[0.12, -0.09], [-0.09, -0.18]], dtype=float),
	"TN": np.array([[-0.11, -0.07], [-0.07, -0.15]], dtype=float),
	"TP": np.array([[-0.05, -0.03], [-0.03, -0.08]], dtype=float),
	"TSS": np.array([[0.06, 0.02], [0.02, 0.01]], dtype=float),
}

plt.rcParams.update(
	{
		"font.size": 9,
		"axes.titlesize": 10,
		"axes.labelsize": 9,
		"xtick.labelsize": 6,
		"ytick.labelsize": 6,
		"figure.facecolor": "white",
		"axes.facecolor": "white",
		"savefig.facecolor": "white",
		"axes.edgecolor": "#6B7280",
		"axes.linewidth": 0.6,
	}
)


def _build_theta_uc(target_name: str) -> np.ndarray:
	profile = PROFILE_BY_TARGET[target_name]
	w_u = W_U_BY_TARGET[target_name]
	component_axis = np.linspace(-0.35, 0.35, profile.size, dtype=float)
	theta_uc = 0.38 * np.outer(w_u, profile)
	theta_uc += 0.03 * np.outer(np.array([1.0, -1.0], dtype=float), component_axis)
	return theta_uc


def _build_theta_cc(target_name: str) -> np.ndarray:
	profile = PROFILE_BY_TARGET[target_name]
	component_axis = np.linspace(-1.0, 1.0, profile.size, dtype=float)
	oscillation = 0.02 * np.cos(np.subtract.outer(np.arange(profile.size), np.arange(profile.size)) / 3.0)
	theta_cc = 0.24 * np.outer(profile, profile)
	theta_cc += 0.03 * np.outer(component_axis, -component_axis)
	theta_cc += oscillation
	theta_cc = 0.5 * (theta_cc + theta_cc.T)
	return theta_cc


def _build_gamma(target_name: str) -> np.ndarray:
	profile = PROFILE_BY_TARGET[target_name]
	component_axis = np.linspace(-1.0, 1.0, profile.size, dtype=float)
	rolled_profile = np.roll(profile, 2)
	gamma = 0.16 * np.outer(profile, rolled_profile)
	gamma -= 0.05 * np.outer(component_axis, component_axis[::-1])
	gamma += 0.025 * np.sin(np.subtract.outer(np.arange(profile.size), np.arange(profile.size)) / 2.2)
	np.fill_diagonal(gamma, 0.0)
	return np.clip(gamma, -0.32, 0.32)


def build_target_blocks(target_name: str) -> dict[str, np.ndarray]:
	return {
		"b": np.array([[BIAS_BY_TARGET[target_name]]], dtype=float),
		"W_u": W_U_BY_TARGET[target_name][None, :],
		"W_in": PROFILE_BY_TARGET[target_name][None, :],
		"Theta_uu": THETA_UU_BY_TARGET[target_name],
		"Theta_uc": _build_theta_uc(target_name),
		"Theta_cc": _build_theta_cc(target_name),
		"Gamma": _build_gamma(target_name),
	}


def _resolve_global_scale(blocks: dict[str, np.ndarray]) -> TwoSlopeNorm:
	max_abs = max(float(np.max(np.abs(block_values))) for block_values in blocks.values())
	if max_abs <= 0.0:
		max_abs = 1.0
	return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _style_axis(axis: plt.Axes) -> None:
	axis.tick_params(length=0, pad=1.5, colors="#334155")
	for spine in axis.spines.values():
		spine.set_color("#94A3B8")
		spine.set_linewidth(0.6)


def _plot_heatmap(
	axis: plt.Axes,
	data: np.ndarray,
	*,
	title: str,
	x_labels: list[str],
	y_labels: list[str],
	norm: TwoSlopeNorm,
	annotate: bool = False,
	x_rotation: float = 55.0,
	aspect: str = "auto",
) -> any:
	image = axis.imshow(data, cmap="coolwarm", norm=norm, origin="lower", aspect=aspect)
	axis.set_title(title, pad=4)
	axis.set_xticks(np.arange(len(x_labels), dtype=float))
	axis.set_xticklabels(x_labels, rotation=x_rotation, ha="right")
	axis.set_yticks(np.arange(len(y_labels), dtype=float))
	axis.set_yticklabels(y_labels)
	_style_axis(axis)

	if annotate:
		for row_index in range(data.shape[0]):
			for column_index in range(data.shape[1]):
				axis.text(
					column_index,
					row_index,
					f"{data[row_index, column_index]:+.2f}",
					ha="center",
					va="center",
					fontsize=6.3,
					color="#0F172A",
				)

	return image


def _add_footer(figure: plt.Figure) -> None:
	figure.text(
		0.985,
		0.008,
		FOOTER_TEXT,
		ha="right",
		va="bottom",
		fontsize=7,
		color="#475569",
	)


def build_target_figure(target_name: str) -> plt.Figure:
	blocks = build_target_blocks(target_name)
	norm = _resolve_global_scale(blocks)
	figure = plt.figure(figsize=(8.4, 12.8), dpi=200)
	grid = figure.add_gridspec(
		nrows=5,
		ncols=4,
		height_ratios=[0.95, 1.05, 1.25, 3.2, 3.2],
		width_ratios=[1.0, 1.0, 1.0, 0.08],
		wspace=0.28,
		hspace=0.78,
	)

	ax_b = figure.add_subplot(grid[0, 0])
	ax_w_u = figure.add_subplot(grid[0, 1])
	ax_theta_uu = figure.add_subplot(grid[0, 2])
	ax_w_in = figure.add_subplot(grid[1, 0:3])
	ax_theta_uc = figure.add_subplot(grid[2, 0:3])
	ax_theta_cc = figure.add_subplot(grid[3, 0:3])
	ax_gamma = figure.add_subplot(grid[4, 0:3])
	colorbar_axis = figure.add_subplot(grid[:, 3])

	shared_labels = ASM_COMPONENT_LABELS

	image = _plot_heatmap(
		ax_b,
		blocks["b"],
		title="b",
		x_labels=[target_name],
		y_labels=[target_name],
		norm=norm,
		annotate=True,
		x_rotation=0.0,
	)
	_plot_heatmap(
		ax_w_u,
		blocks["W_u"],
		title=r"$W_u$",
		x_labels=OPERATIONAL_LABELS,
		y_labels=[target_name],
		norm=norm,
		annotate=True,
		x_rotation=0.0,
	)
	_plot_heatmap(
		ax_theta_uu,
		blocks["Theta_uu"],
		title=r"$\Theta_{uu}$",
		x_labels=OPERATIONAL_LABELS,
		y_labels=OPERATIONAL_LABELS,
		norm=norm,
		annotate=True,
		x_rotation=0.0,
	)
	_plot_heatmap(
		ax_w_in,
		blocks["W_in"],
		title=r"$W_{in}$",
		x_labels=shared_labels,
		y_labels=[target_name],
		norm=norm,
		x_rotation=62.0,
	)
	_plot_heatmap(
		ax_theta_uc,
		blocks["Theta_uc"],
		title=r"$\Theta_{uc}$",
		x_labels=shared_labels,
		y_labels=OPERATIONAL_LABELS,
		norm=norm,
		x_rotation=62.0,
	)
	_plot_heatmap(
		ax_theta_cc,
		blocks["Theta_cc"],
		title=r"$\Theta_{cc}$",
		x_labels=shared_labels,
		y_labels=shared_labels,
		norm=norm,
		x_rotation=62.0,
	)
	_plot_heatmap(
		ax_gamma,
		blocks["Gamma"],
		title=r"$\Gamma$",
		x_labels=shared_labels,
		y_labels=shared_labels,
		norm=norm,
		x_rotation=62.0,
	)

	colorbar = figure.colorbar(image, cax=colorbar_axis)
	colorbar.set_label("Illustrative signed coefficient value", fontsize=8)
	colorbar.ax.tick_params(labelsize=7, colors="#334155")

	figure.suptitle(
		f"Illustrative ICSOR coefficient atlas for {target_name}",
		fontsize=14,
		y=0.996,
	)
	figure.text(
		0.5,
		0.972,
		"Blocks shown per target: b, W_u, W_in, Theta_uu, Theta_uc, Theta_cc, and Gamma.\n"
		"Heatmaps are drawn from the lower origin so the operational axes begin at HRT then Aeration.",
		ha="center",
		va="top",
		fontsize=8.3,
		color="#334155",
	)
	_add_footer(figure)
	return figure


def build_theta_cc_figure(target_name: str) -> plt.Figure:
	theta_cc = build_target_blocks(target_name)["Theta_cc"]
	max_abs = float(np.max(np.abs(theta_cc)))
	if max_abs <= 0.0:
		max_abs = 1.0
	norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
	figure, axis = plt.subplots(figsize=(8.2, 7.2), dpi=200)
	image = _plot_heatmap(
		axis,
		theta_cc,
		title=rf"COD-only $\Theta_{{cc}}$ interaction surface",
		x_labels=ASM_COMPONENT_LABELS,
		y_labels=ASM_COMPONENT_LABELS,
		norm=norm,
		x_rotation=62.0,
	)
	colorbar = figure.colorbar(image, ax=axis, pad=0.02)
	colorbar.set_label(r"Illustrative COD $\Theta_{cc}$ coefficient value", fontsize=8)
	colorbar.ax.tick_params(labelsize=7, colors="#334155")
	figure.suptitle(
		"Illustrative COD cross-influent curvature summary",
		fontsize=13,
		y=0.98,
	)
	figure.text(
		0.5,
		0.945,
		"The heatmap is drawn from the lower origin so the ASM-component labels progress upward and rightward from the matrix origin.",
		ha="center",
		va="top",
		fontsize=8.0,
		color="#334155",
	)
	_add_footer(figure)
	return figure


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	main_figure = build_theta_cc_figure("COD")
	main_output_path = OUTPUT_DIR / MAIN_TEXT_FILE_NAME
	main_figure.savefig(main_output_path, format="pdf", bbox_inches="tight")
	plt.close(main_figure)
	print(f"Wrote {main_output_path}")

	for target_name, file_name in SUPPLEMENTARY_FILE_NAMES.items():
		figure = build_target_figure(target_name)
		output_path = OUTPUT_DIR / file_name
		figure.savefig(output_path, format="pdf", bbox_inches="tight")
		plt.close(figure)
		print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()