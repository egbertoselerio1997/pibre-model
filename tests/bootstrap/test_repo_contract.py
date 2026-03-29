"""Minimal bootstrap tests for repository structure and configuration contracts."""

from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DIRECTORIES = [
    "config",
    "src",
    "results",
    "data",
    "tests",
    "docs",
    "docs/ml",
    "docs/simulation",
    "src/utils",
    "src/models",
    "src/models/simulation",
    "src/models/ml",
]

EXPECTED_UTILITY_MODULES = [
    "analysis.py",
    "io.py",
    "metrics.py",
    "optuna.py",
    "plot.py",
    "process.py",
    "simulation.py",
    "test.py",
    "train.py",
]

EXPECTED_PATH_KEYS = {
    "root_dir",
    "config_dir",
    "src_dir",
    "results_dir",
    "data_dir",
    "tests_dir",
    "docs_dir",
    "docs_ml_dir",
    "docs_simulation_dir",
    "main_notebook",
    "src_utils_dir",
    "src_models_dir",
    "simulation_models_dir",
    "ml_models_dir",
    "ml_model_bundle_pattern",
    "ml_metrics_pattern",
    "ml_optuna_pattern",
    "asm2d_tcn_reference_workbook",
    "asm2d_tcn_simulation_data_pattern",
    "asm2d_tcn_simulation_metadata_pattern",
    "simulation_data_pattern",
    "simulation_metadata_pattern",
}


class BootstrapContractTests(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        for relative_path in EXPECTED_DIRECTORIES:
            directory = REPO_ROOT / relative_path
            self.assertTrue(directory.is_dir(), f"Missing required directory: {relative_path}")

    def test_required_utility_modules_exist(self) -> None:
        utils_dir = REPO_ROOT / "src" / "utils"
        for module_name in EXPECTED_UTILITY_MODULES:
            self.assertTrue((utils_dir / module_name).is_file(), f"Missing utility module: {module_name}")

    def test_paths_config_contains_expected_keys(self) -> None:
        config_path = REPO_ROOT / "config" / "paths.json"
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.assertTrue(EXPECTED_PATH_KEYS.issubset(data.keys()))

    def test_params_config_uses_model_namespaces(self) -> None:
        config_path = REPO_ROOT / "config" / "params.json"
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.assertTrue(data, "params.json must define at least one model namespace template")
        self.assertIn("ml_orchestration", data)
        for model_name, params in data.items():
            self.assertIsInstance(model_name, str)
            self.assertIsInstance(params, dict)
            self.assertIn("hyperparameters", params)

    def test_rules_require_training_progress_visibility(self) -> None:
        rules_path = REPO_ROOT / "CODEBASE_RULES.md"
        rules_text = rules_path.read_text(encoding="utf-8")

        self.assertIn("### 14.1 Training Progress Visibility", rules_text)
        self.assertIn("TQDM progress bars", rules_text)
        self.assertIn("enabled by default", rules_text)

    def test_rules_require_common_plot_theme(self) -> None:
        rules_path = REPO_ROOT / "CODEBASE_RULES.md"
        rules_text = rules_path.read_text(encoding="utf-8")

        self.assertIn("### 12.1 Common Plot Theme and Style", rules_text)
        self.assertIn("Pibre Scientific theme", rules_text)
        self.assertIn("default qualitative data cycle", rules_text)
        self.assertIn("default sequential colormap: cividis", rules_text)
        self.assertIn("blue-white-vermilion", rules_text)

    def test_pyproject_declares_tqdm_dependency(self) -> None:
        pyproject_path = REPO_ROOT / "pyproject.toml"
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)

        dependencies = data["project"]["dependencies"]
        self.assertTrue(any(str(dependency).startswith("tqdm>=") for dependency in dependencies))

    def test_pyproject_declares_openpyxl_dependency(self) -> None:
        pyproject_path = REPO_ROOT / "pyproject.toml"
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)

        dependencies = data["project"]["dependencies"]
        self.assertTrue(any(str(dependency).startswith("openpyxl>=") for dependency in dependencies))


if __name__ == "__main__":
    unittest.main()