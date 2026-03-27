"""Measured-space random forest regression pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils.process import DatasetSplit, ScalingBundle
from src.utils.simulation import load_model_params
from src.utils.train import (
    predict_tabular_regressor_model,
    run_tabular_regressor_pipeline,
    train_tabular_regressor,
    tune_tabular_regressor_hyperparameters,
)


MODEL_NAME = "random_forest_regressor"


def load_random_forest_regressor_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the random forest regressor."""

    return load_model_params(MODEL_NAME, repo_root)


def build_random_forest_regressor_model(model_hyperparameters: Mapping[str, Any]) -> RandomForestRegressor:
    """Build one multi-output random forest regressor from configured hyperparameters."""

    return RandomForestRegressor(**dict(model_hyperparameters))


def train_random_forest_regressor_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the random forest regressor on a prepared dataset."""

    return train_tabular_regressor(training_dataset, build_random_forest_regressor_model, model_hyperparameters)


def tune_random_forest_regressor_hyperparameters(
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    *,
    scaling_bundle: ScalingBundle,
    A_matrix: np.ndarray,
    model_params: Mapping[str, Any] | None = None,
    tuning_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune the random forest regressor with Optuna."""

    params = dict(model_params) if model_params is not None else load_random_forest_regressor_params()
    return tune_tabular_regressor_hyperparameters(
        MODEL_NAME,
        build_random_forest_regressor_model,
        train_split,
        validation_split,
        scaling_bundle=scaling_bundle,
        A_matrix=A_matrix,
        model_params=params,
        tuning_profile=tuning_profile,
    )


def predict_random_forest_regressor_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted random forest regressor bundle and generate predictions."""

    return predict_tabular_regressor_model(
        test_dataset,
        model_path,
        metadata=metadata,
        composition_matrix=composition_matrix,
    )


def run_random_forest_regressor_pipeline(
    dataset: pd.DataFrame,
    metadata: Mapping[str, Any],
    composition_matrix: np.ndarray,
    A_matrix: np.ndarray,
    *,
    repo_root: str | Path | None = None,
    model_params: Mapping[str, Any] | None = None,
    tuning_profile: str | None = None,
    persist_artifacts: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Tune, fit, evaluate, and optionally persist the random forest regressor."""

    params = dict(model_params) if model_params is not None else load_random_forest_regressor_params(repo_root)
    return run_tabular_regressor_pipeline(
        MODEL_NAME,
        build_random_forest_regressor_model,
        dataset,
        metadata,
        composition_matrix,
        A_matrix,
        repo_root=repo_root,
        model_params=params,
        tuning_profile=tuning_profile,
        persist_artifacts=persist_artifacts,
        timestamp=timestamp,
    )


__all__ = [
    "MODEL_NAME",
    "build_random_forest_regressor_model",
    "load_random_forest_regressor_params",
    "predict_random_forest_regressor_model",
    "run_random_forest_regressor_pipeline",
    "train_random_forest_regressor_model",
    "tune_random_forest_regressor_hyperparameters",
]