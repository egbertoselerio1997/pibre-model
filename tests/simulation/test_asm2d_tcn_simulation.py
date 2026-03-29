"""Workbook and runtime contract tests for the ASM2d-TCN reference model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from src.models.simulation.asm2d_tcn_simulation import (
    generate_asm2d_tcn_dataset,
    get_asm2d_tcn_matrices,
    create_asm2d_tcn_workbook,
    load_asm2d_tcn_simulation_params,
    resolve_asm2d_tcn_simulation_artifact_paths,
    resolve_asm2d_tcn_workbook_path,
    run_asm2d_tcn_simulation,
)


def _column_index_by_header(worksheet) -> dict[str, int]:
    return {
        str(cell.value): index
        for index, cell in enumerate(worksheet[1], start=1)
        if cell.value is not None
    }


def _row_index_by_value(worksheet, column_number: int) -> dict[str, int]:
    index: dict[str, int] = {}
    for row_number in range(2, worksheet.max_row + 1):
        value = worksheet.cell(row=row_number, column=column_number).value
        if value is not None:
            index[str(value)] = row_number
    return index


class Asm2dTcnWorkbookTests(unittest.TestCase):
    def test_resolve_workbook_path_uses_configured_location(self) -> None:
        workbook_path = resolve_asm2d_tcn_workbook_path()

        self.assertTrue(workbook_path.as_posix().endswith("data/asm2d-tcn/asm2d_tcn_workbook.xlsx"))

    def test_workbook_config_contains_expected_dimensions(self) -> None:
        params = load_asm2d_tcn_simulation_params()
        workbook_config = params["workbook"]

        self.assertEqual(
            workbook_config["sheets"],
            ["stoichiometric_matrix", "composition_matrix", "parameter_table"],
        )
        self.assertEqual(len(workbook_config["processes"]), 28)
        self.assertEqual(len(workbook_config["state_columns"]), 21)
        self.assertEqual(len(workbook_config["composite_variables"]), 6)

    def test_create_workbook_writes_required_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workbook_path = create_asm2d_tcn_workbook(Path(tmp_dir) / "asm2d_tcn.xlsx")
            workbook = load_workbook(workbook_path, data_only=False)

        self.assertEqual(
            workbook.sheetnames,
            ["stoichiometric_matrix", "composition_matrix", "parameter_table"],
        )

    def test_stoichiometric_matrix_contains_formula_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workbook_path = create_asm2d_tcn_workbook(Path(tmp_dir) / "asm2d_tcn.xlsx")
            workbook = load_workbook(workbook_path, data_only=False)

        worksheet = workbook["stoichiometric_matrix"]
        header_index = _column_index_by_header(worksheet)
        process_row_index = _row_index_by_value(worksheet, 2)

        aerobic_hydrolysis_row = process_row_index["Aerobic hydrolysis"]
        precipitation_row = process_row_index["Precipitation"]
        aob_growth_row = process_row_index["Aerobic growth of X_AOB"]

        self.assertIn("parameter_table", str(worksheet.cell(aerobic_hydrolysis_row, header_index["S_F"]).value))
        self.assertIn("parameter_table", str(worksheet.cell(aerobic_hydrolysis_row, header_index["S_NH4"]).value))
        self.assertIn("parameter_table", str(worksheet.cell(aob_growth_row, header_index["S_NO2"]).value))
        self.assertEqual(worksheet.cell(precipitation_row, header_index["S_PO4"]).value, "=-1")
        self.assertEqual(worksheet.cell(precipitation_row, header_index["X_TSS"]).value, "=1.42")

    def test_composition_matrix_contains_formula_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workbook_path = create_asm2d_tcn_workbook(Path(tmp_dir) / "asm2d_tcn.xlsx")
            workbook = load_workbook(workbook_path, data_only=False)

        worksheet = workbook["composition_matrix"]
        header_index = _column_index_by_header(worksheet)
        state_row_index = _row_index_by_value(worksheet, 2)

        self.assertIn("parameter_table", str(worksheet.cell(state_row_index["X_H"], header_index["TN"]).value))
        self.assertIn("parameter_table", str(worksheet.cell(state_row_index["X_MeP"], header_index["TP"]).value))
        self.assertEqual(worksheet.cell(state_row_index["X_TSS"], header_index["TSS"]).value, "=1")


class Asm2dTcnSimulationTests(unittest.TestCase):
    def test_resolve_simulation_artifact_paths_use_requested_folder(self) -> None:
        dataset_path, metadata_path, dataset_relative = resolve_asm2d_tcn_simulation_artifact_paths(
            timestamp="20260330_000000"
        )

        self.assertTrue(dataset_path.as_posix().endswith("data/asm2d-tcn/simulation/data_20260330_000000.csv"))
        self.assertTrue(metadata_path.as_posix().endswith("data/asm2d-tcn/simulation/metadata_20260330_000000.json"))
        self.assertEqual(dataset_relative, "data/asm2d-tcn/simulation/data_20260330_000000.csv")

    def test_numeric_matrices_have_expected_shapes(self) -> None:
        params = load_asm2d_tcn_simulation_params()
        matrix_bundle = get_asm2d_tcn_matrices(params)

        self.assertEqual(matrix_bundle["petersen_matrix"].shape, (28, 21))
        self.assertEqual(matrix_bundle["composition_matrix"].shape, (6, 21))
        self.assertEqual(matrix_bundle["measured_output_columns"], params["measured_output_columns"])

    def test_generate_dataset_reports_composites_only(self) -> None:
        params = load_asm2d_tcn_simulation_params()
        dataset, metadata, matrix_bundle = generate_asm2d_tcn_dataset(
            model_params=params,
            n_samples=12,
            random_seed=7,
            parallel_workers=1,
        )
        expected_independent = metadata["independent_columns"]
        expected_dependent = metadata["dependent_columns"]

        self.assertEqual(dataset.shape, (12, len(expected_independent) + len(expected_dependent)))
        self.assertEqual(list(dataset.columns), expected_independent + expected_dependent)
        self.assertEqual(expected_dependent, [f"Out_{name}" for name in params["measured_output_columns"]])
        self.assertFalse(any(column_name.startswith("Out_S_") or column_name.startswith("Out_X_") for column_name in expected_dependent))
        self.assertEqual(matrix_bundle["petersen_matrix"].shape, (28, 21))
        self.assertEqual(matrix_bundle["composition_matrix"].shape, (6, 21))

    def test_run_simulation_returns_notebook_facing_bundle(self) -> None:
        result = run_asm2d_tcn_simulation(
            save_artifacts=False,
            n_samples=8,
            random_seed=5,
            parallel_workers=1,
        )

        self.assertIn("dataset", result)
        self.assertIn("metadata", result)
        self.assertIn("petersen_matrix", result)
        self.assertIn("composition_matrix", result)
        self.assertIn("matrix_bundle", result)
        self.assertIn("artifact_paths", result)
        self.assertEqual(result["artifact_paths"]["dataset_csv"], None)
        self.assertEqual(result["artifact_paths"]["metadata_json"], None)
        self.assertEqual(result["metadata"]["measured_output_columns"], ["COD", "TN", "TKN", "TP", "TSS", "VSS"])


if __name__ == "__main__":
    unittest.main()