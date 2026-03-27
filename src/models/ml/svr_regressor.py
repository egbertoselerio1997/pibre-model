"""Measured-space support vector regression pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from src.utils.process import DatasetSplit, ScalingBundle
from src.utils.simulation import load_model_params
from src.utils.train import (
    predict_tabular_regressor_model,
    run_tabular_regressor_pipeline,
    train_tabular_regressor,
    tune_tabular_regressor_hyperparameters,
)


MODEL_NAME = "svr_regressor"


def load_svr_regressor_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the support vector regressor."""

    return load_model_params(MODEL_NAME, repo_root)


def build_svr_regressor_model(model_hyperparameters: Mapping[str, Any]) -> MultiOutputRegressor:
    """Build one multi-output support vector regressor from configured hyperparameters."""

    return MultiOutputRegressor(SVR(**dict(model_hyperparameters)))


def train_svr_regressor_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the support vector regressor on a prepared dataset."""

    return train_tabular_regressor(training_dataset, build_svr_regressor_model, model_hyperparameters)


def tune_svr_regressor_hyperparameters(
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    *,
    scaling_bundle: ScalingBundle,
    A_matrix: np.ndarray,
    model_params: Mapping[str, Any] | None = None,
    tuning_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune the support vector regressor with Optuna."""

    params = dict(model_params) if model_params is not None else load_svr_regressor_params()
    return tune_tabular_regressor_hyperparameters(
        MODEL_NAME,
        build_svr_regressor_model,
        train_split,
        validation_split,
        scaling_bundle=scaling_bundle,
        A_matrix=A_matrix,
        model_params=params,
        tuning_profile=tuning_profile,
    )


def predict_svr_regressor_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted support vector regressor bundle and generate predictions."""

    return predict_tabular_regressor_model(
        test_dataset,
        model_path,
        metadata=metadata,
        composition_matrix=composition_matrix,
    )


def run_svr_regressor_pipeline(
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
    """Tune, fit, evaluate, and optionally persist the support vector regressor."""

    params = dict(model_params) if model_params is not None else load_svr_regressor_params(repo_root)
    return run_tabular_regressor_pipeline(
        MODEL_NAME,
        build_svr_regressor_model,
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
    "build_svr_regressor_model",
    "load_svr_regressor_params",
    "predict_svr_regressor_model",
    "run_svr_regressor_pipeline",
    "train_svr_regressor_model",
    "tune_svr_regressor_hyperparameters",
]