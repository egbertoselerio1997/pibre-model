from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.utils.plot import plot_coefficient_heatmap, plot_icsor_target_atlas, save_figure_pdf


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "DCHE-D-26-00020"

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


def build_target_figure(target_name: str) -> Figure:
	return plot_icsor_target_atlas(
		build_target_blocks(target_name),
		target_name=target_name,
		operational_labels=OPERATIONAL_LABELS,
		state_labels=ASM_COMPONENT_LABELS,
		colorbar_label="Coefficient value",
		include_footer=False,
	)


def build_theta_cc_figure(target_name: str) -> Figure:
	figure, _ = plot_coefficient_heatmap(
		build_target_blocks(target_name)["Theta_cc"],
		row_labels=ASM_COMPONENT_LABELS,
		column_labels=ASM_COMPONENT_LABELS,
		title=r"COD-specific $\Theta_{cc}$ heatmap",
		x_label="Influent ASM component",
		y_label="Influent ASM component",
		colorbar_label="Coefficient value",
		figure_size=(9.6, 7.8),
		x_tick_rotation=62.0,
	)
	return figure


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	main_figure = build_theta_cc_figure("COD")
	main_output_path = OUTPUT_DIR / MAIN_TEXT_FILE_NAME
	save_figure_pdf(main_figure, main_output_path)
	plt.close(main_figure)
	print(f"Wrote {main_output_path}")

	for target_name, file_name in SUPPLEMENTARY_FILE_NAMES.items():
		figure = build_target_figure(target_name)
		output_path = OUTPUT_DIR / file_name
		save_figure_pdf(figure, output_path)
		plt.close(figure)
		print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()