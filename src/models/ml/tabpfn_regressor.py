"""Fractional-space TabPFN regression pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.utils.process import DatasetSplit
from src.utils.simulation import load_model_params
from src.utils.train import (
    predict_tabular_regressor_model,
    run_tabular_regressor_pipeline,
    train_tabular_regressor,
)


MODEL_NAME = "tabpfn_regressor"


def load_tabpfn_regressor_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the TabPFN regressor."""

    return load_model_params(MODEL_NAME, repo_root)


def _get_tabpfn_regressor_class() -> Any:
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:
        raise ImportError(
            "tabpfn is required for the TabPFN regressor. Install the project dependencies with `uv sync`."
        ) from exc

    return TabPFNRegressor


def _resolve_tabpfn_model_version(version_name: str | None) -> Any | None:
    if version_name is None:
        return None

    normalized_version = str(version_name).strip().lower().replace(".", "_")
    enum_names = {
        "v2": "V2",
        "v2_0": "V2",
        "v2_5": "V2_5",
        "v2_6": "V2_6",
    }
    enum_name = enum_names.get(normalized_version)
    if enum_name is None:
        raise ValueError(f"Unsupported TabPFN model_version: {version_name}")

    try:
        from tabpfn.constants import ModelVersion
    except ImportError:
        return None

    return getattr(ModelVersion, enum_name)


def _resolve_tabpfn_load_device(requested_device: Any) -> str:
    if requested_device is None:
        return "cpu"

    normalized_device = str(requested_device).strip().lower()
    if normalized_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized_device


def build_single_target_tabpfn_regressor_model(model_hyperparameters: Mapping[str, Any]) -> Any:
    """Build one single-target TabPFN regressor from configured hyperparameters."""

    regressor_class = _get_tabpfn_regressor_class()
    resolved_hyperparameters = dict(model_hyperparameters)
    model_version = _resolve_tabpfn_model_version(resolved_hyperparameters.pop("model_version", None))

    if model_version is not None and hasattr(regressor_class, "create_default_for_version"):
        return regressor_class.create_default_for_version(model_version, **resolved_hyperparameters)

    return regressor_class(**resolved_hyperparameters)


class TabPFNMultiOutputRegressor:
    """Independent per-target wrapper around single-target TabPFN regressors."""

    def __init__(self, model_hyperparameters: Mapping[str, Any]):
        self.model_hyperparameters = dict(model_hyperparameters)
        self.estimators_: list[Any] = []
        self.feature_columns_: list[str] = []
        self.target_columns_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray) -> TabPFNMultiOutputRegressor:
        """Fit one independent TabPFN regressor per target column."""

        feature_frame = pd.DataFrame(X)
        target_frame = pd.DataFrame(y)

        self.feature_columns_ = list(feature_frame.columns)
        self.target_columns_ = list(target_frame.columns)
        self.estimators_ = []

        for column_name in self.target_columns_:
            estimator = build_single_target_tabpfn_regressor_model(self.model_hyperparameters)
            estimator.fit(feature_frame, target_frame[column_name].to_numpy(dtype=float))
            self.estimators_.append(estimator)

        self.n_features_in_ = feature_frame.shape[1]
        self.n_outputs_ = len(self.target_columns_)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict all target columns and stack them into one output array."""

        if not self.estimators_:
            raise RuntimeError("TabPFNMultiOutputRegressor must be fitted before predict is called.")

        feature_frame = pd.DataFrame(X)
        if self.feature_columns_:
            if list(feature_frame.columns) != self.feature_columns_ and feature_frame.shape[1] == len(self.feature_columns_):
                feature_frame = pd.DataFrame(
                    feature_frame.to_numpy(dtype=float),
                    index=feature_frame.index,
                    columns=self.feature_columns_,
                )
            else:
                feature_frame = feature_frame.loc[:, self.feature_columns_]

        prediction_columns = [
            np.asarray(estimator.predict(feature_frame), dtype=float).reshape(-1, 1)
            for estimator in self.estimators_
        ]
        return np.hstack(prediction_columns)

    def __getstate__(self) -> dict[str, Any]:
        """Serialize fitted target-specific estimators through their fit-state archives."""

        state = dict(self.__dict__)
        serialized_estimators: list[bytes] = []
        for estimator_index, estimator in enumerate(state.pop("estimators_", [])):
            with tempfile.TemporaryDirectory() as temp_dir_name:
                estimator_path = Path(temp_dir_name) / f"target_{estimator_index}.tabpfn_fit"
                estimator.save_fit_state(estimator_path)
                serialized_estimators.append(estimator_path.read_bytes())

        state["_serialized_estimators"] = serialized_estimators
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore fitted target-specific estimators from saved fit-state archives."""

        serialized_estimators = list(state.get("_serialized_estimators", []))
        restored_state = dict(state)
        restored_state.pop("_serialized_estimators", None)
        self.__dict__.update(restored_state)

        regressor_class = _get_tabpfn_regressor_class()
        load_device = _resolve_tabpfn_load_device(self.model_hyperparameters.get("device", "auto"))
        self.estimators_ = []
        for estimator_index, estimator_payload in enumerate(serialized_estimators):
            with tempfile.TemporaryDirectory() as temp_dir_name:
                estimator_path = Path(temp_dir_name) / f"target_{estimator_index}.tabpfn_fit"
                estimator_path.write_bytes(bytes(estimator_payload))
                estimator = regressor_class.load_from_fit_state(estimator_path, device=load_device)
                self.estimators_.append(estimator)


def build_tabpfn_regressor_model(model_hyperparameters: Mapping[str, Any]) -> TabPFNMultiOutputRegressor:
    """Build one multi-output TabPFN regressor from configured hyperparameters."""

    return TabPFNMultiOutputRegressor(model_hyperparameters)


def train_tabpfn_regressor_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the TabPFN regressor on a prepared dataset."""

    return train_tabular_regressor(training_dataset, build_tabpfn_regressor_model, model_hyperparameters)


def predict_tabpfn_regressor_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted TabPFN regressor bundle and generate predictions."""

    return predict_tabular_regressor_model(
        test_dataset,
        model_path,
        metadata=metadata,
        composition_matrix=composition_matrix,
    )


def run_tabpfn_regressor_pipeline(
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    A_matrix: np.ndarray,
    *,
    composition_matrix: np.ndarray | None = None,
    measured_output_columns: list[str] | None = None,
    repo_root: str | Path | None = None,
    model_params: Mapping[str, Any] | None = None,
    model_hyperparameters: Mapping[str, Any] | None = None,
    optuna_summary: Mapping[str, Any] | None = None,
    show_progress: bool = True,
    persist_artifacts: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and optionally persist the TabPFN regressor."""

    params = dict(model_params) if model_params is not None else load_tabpfn_regressor_params(repo_root)
    return run_tabular_regressor_pipeline(
        MODEL_NAME,
        build_tabpfn_regressor_model,
        training_split,
        test_split,
        A_matrix,
        composition_matrix=composition_matrix,
        measured_output_columns=measured_output_columns,
        repo_root=repo_root,
        model_params=params,
        model_hyperparameters=model_hyperparameters,
        optuna_summary=optuna_summary,
        show_progress=show_progress,
        persist_artifacts=persist_artifacts,
        timestamp=timestamp,
    )


__all__ = [
    "MODEL_NAME",
    "TabPFNMultiOutputRegressor",
    "build_single_target_tabpfn_regressor_model",
    "build_tabpfn_regressor_model",
    "load_tabpfn_regressor_params",
    "predict_tabpfn_regressor_model",
    "run_tabpfn_regressor_pipeline",
    "train_tabpfn_regressor_model",
]