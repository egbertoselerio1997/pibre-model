"""Decoupled unconstrained bilinear regression in measured-output space."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from src.utils.optuna import create_progress_bar
from src.utils.process import (
    DatasetSplit,
    ScalingBundle,
    TrainTestDatasetSplits,
    build_measured_supervised_dataset,
    fit_scalers,
    inverse_transform_targets,
    project_to_mass_balance,
    transform_dataset_split,
)
from src.utils.simulation import load_model_params
from src.utils.test import evaluate_prediction_bundle
from src.utils.train import (
    load_model_bundle,
    persist_training_artifacts,
    resolve_model_hyperparameters,
    resolve_training_objective_label,
    serialize_report_frames,
    transform_feature_frame,
)


MODEL_NAME = "uncobre"


def load_uncobre_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the UNCOBRE model."""

    return load_model_params(MODEL_NAME, repo_root)


def _ensure_two_dimensional_predictions(values: Any) -> np.ndarray:
    prediction_array = np.asarray(values, dtype=float)
    if prediction_array.ndim == 1:
        return prediction_array.reshape(-1, 1)
    return prediction_array


def _resolve_training_options(
    training_options: Mapping[str, Any] | None,
    *,
    objective_name: str,
) -> dict[str, Any]:
    options = dict(training_options or {})
    options.setdefault("show_progress", True)
    options.setdefault("progress_description", "Training UNCOBRE")
    options.setdefault("objective_name", objective_name)
    return options


def build_uncobre_feature_expander(
    model_hyperparameters: Mapping[str, Any],
) -> PolynomialFeatures:
    """Build the configured degree-2 bilinear feature expander."""

    return PolynomialFeatures(
        degree=2,
        include_bias=bool(model_hyperparameters.get("include_bias", False)),
        interaction_only=bool(model_hyperparameters.get("interaction_only", False)),
    )


def build_uncobre_model(model_hyperparameters: Mapping[str, Any]) -> LinearRegression | Ridge:
    """Build one unconstrained estimator from configured OLS or Ridge hyperparameters."""

    regression_mode = str(model_hyperparameters.get("regression_mode", "ridge")).strip().lower()
    fit_intercept = bool(model_hyperparameters.get("fit_intercept", True))

    if regression_mode == "ols":
        return LinearRegression(
            fit_intercept=fit_intercept,
        )

    if regression_mode == "ridge":
        return Ridge(
            alpha=float(model_hyperparameters.get("ridge_alpha", 1.0)),
            fit_intercept=fit_intercept,
            copy_X=True,
            random_state=int(model_hyperparameters.get("random_state", 42)),
        )

    raise ValueError("regression_mode must be either 'ols' or 'ridge'.")


def train_uncobre_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, Any],
    *,
    training_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the UNCOBRE estimator on a prepared dataset."""

    feature_frame = pd.DataFrame(training_dataset["features"])
    target_frame = pd.DataFrame(training_dataset["targets"])
    objective_label = resolve_training_objective_label(
        model_hyperparameters,
        default=str(model_hyperparameters.get("regression_mode", "ridge")),
    )
    options = _resolve_training_options(
        training_options,
        objective_name=objective_label,
    )

    feature_expander = build_uncobre_feature_expander(model_hyperparameters)
    model = build_uncobre_model(model_hyperparameters)

    progress_bar = create_progress_bar(
        total=2,
        desc=str(options["progress_description"]),
        enabled=bool(options["show_progress"]),
        unit="stage",
    )
    try:
        progress_bar.set_postfix(stage="expand", objective=str(options["objective_name"]))
        expanded_training_features = feature_expander.fit_transform(feature_frame)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="fit", objective=str(options["objective_name"]))
        model.fit(expanded_training_features, target_frame.to_numpy(dtype=float))
        training_predictions = _ensure_two_dimensional_predictions(model.predict(expanded_training_features))
        train_mse = float(mean_squared_error(target_frame.to_numpy(dtype=float), training_predictions))
        progress_bar.set_postfix(
            stage="fit",
            objective=str(options["objective_name"]),
            objective_value=f"{train_mse:.6g}",
        )
        progress_bar.update(1)
    finally:
        progress_bar.close()

    return {
        "model": model,
        "feature_expander": feature_expander,
        "training_predictions": pd.DataFrame(
            training_predictions,
            index=target_frame.index,
            columns=target_frame.columns,
        ),
        "training_mse": train_mse,
    }


def _predict_unconstrained_split(
    model: LinearRegression | Ridge,
    feature_expander: PolynomialFeatures,
    dataset_split: DatasetSplit,
    *,
    scaling_bundle: ScalingBundle,
    A_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expanded_features = feature_expander.transform(dataset_split.features)
    raw_predictions_scaled = _ensure_two_dimensional_predictions(model.predict(expanded_features))
    raw_predictions = inverse_transform_targets(raw_predictions_scaled, scaling_bundle)
    projected_predictions = project_to_mass_balance(
        raw_predictions,
        dataset_split.constraint_reference.to_numpy(dtype=float),
        np.asarray(A_matrix, dtype=float),
    )
    return projected_predictions, raw_predictions


def _build_model_bundle(
    training_result: Mapping[str, Any],
    scaling_bundle: ScalingBundle,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    constraint_columns: list[str],
    A_matrix: np.ndarray,
    model_hyperparameters: Mapping[str, Any],
    training_options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model": training_result["model"],
        "feature_expander": training_result["feature_expander"],
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "constraint_columns": constraint_columns,
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "scaling_bundle": scaling_bundle,
        "model_hyperparameters": dict(model_hyperparameters),
        "training_options": dict(training_options),
    }


def predict_uncobre_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted unconstrained bundle and generate aligned predictions."""

    model_bundle = load_model_bundle(model_path)
    scaling_bundle: ScalingBundle = model_bundle["scaling_bundle"]

    constraint_columns = list(model_bundle["constraint_columns"])

    if isinstance(test_dataset, pd.DataFrame):
        if metadata is None or composition_matrix is None:
            raise ValueError("metadata and composition_matrix are required when predicting from a raw dataset.")
        measured_dataset = build_measured_supervised_dataset(
            test_dataset,
            dict(metadata),
            np.asarray(composition_matrix, dtype=float),
        )
        feature_frame = measured_dataset.features
        constraint_reference = measured_dataset.constraint_reference
    else:
        feature_frame = pd.DataFrame(test_dataset["features"], columns=scaling_bundle.feature_columns)
        constraint_reference = pd.DataFrame(test_dataset["constraint_reference"], columns=constraint_columns)

    selected_constraint_reference = pd.DataFrame(constraint_reference).loc[:, constraint_columns].copy()

    transformed_features = transform_feature_frame(feature_frame, scaling_bundle)

    prediction_split = DatasetSplit(
        features=transformed_features,
        targets=pd.DataFrame(
            np.zeros((len(feature_frame), len(model_bundle["target_columns"]))),
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        constraint_reference=selected_constraint_reference,
    )

    projected_predictions, raw_predictions = _predict_unconstrained_split(
        model_bundle["model"],
        model_bundle["feature_expander"],
        prediction_split,
        scaling_bundle=scaling_bundle,
        A_matrix=np.asarray(model_bundle["A_matrix"], dtype=float),
    )

    return {
        "raw_predictions": pd.DataFrame(raw_predictions, index=feature_frame.index, columns=model_bundle["target_columns"]),
        "projected_predictions": pd.DataFrame(
            projected_predictions,
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "constraint_reference": selected_constraint_reference,
    }


def run_uncobre_pipeline(
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    A_matrix: np.ndarray,
    *,
    repo_root: str | Path | None = None,
    model_params: Mapping[str, Any] | None = None,
    model_hyperparameters: Mapping[str, Any] | None = None,
    optuna_summary: Mapping[str, Any] | None = None,
    show_progress: bool = True,
    persist_artifacts: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and optionally persist one UNCOBRE bundle."""

    params = dict(model_params) if model_params is not None else load_uncobre_params(repo_root)
    split_params = params["hyperparameters"]
    selected_hyperparameters = resolve_model_hyperparameters(params, model_hyperparameters)
    objective_label = resolve_training_objective_label(
        selected_hyperparameters,
        default=str(selected_hyperparameters.get("regression_mode", "ridge")),
    )

    progress_bar = create_progress_bar(
        total=5,
        desc="Training UNCOBRE",
        enabled=show_progress,
        unit="stage",
    )

    try:
        progress_bar.set_postfix(stage="scaling", objective=objective_label)
        scaling_bundle = fit_scalers(
            training_split,
            scale_features=bool(split_params["scale_features"]),
            scale_targets=bool(split_params["scale_targets"]),
        )
        scaled_training_split = transform_dataset_split(training_split, scaling_bundle)
        scaled_test_split = transform_dataset_split(test_split, scaling_bundle)
        progress_bar.update(1)

        progress_bar.set_postfix(stage="fit", objective=objective_label)
        training_result = train_uncobre_model(
            {
                "features": scaled_training_split.features,
                "targets": scaled_training_split.targets,
            },
            selected_hyperparameters,
            training_options={
                "show_progress": False,
                "progress_description": "Training UNCOBRE",
                "objective_name": objective_label,
            },
        )
        final_model: LinearRegression | Ridge = training_result["model"]
        feature_expander: PolynomialFeatures = training_result["feature_expander"]
        progress_bar.update(1)

        progress_bar.set_postfix(stage="evaluate_train", objective=objective_label)
        train_projected, train_raw = _predict_unconstrained_split(
            final_model,
            feature_expander,
            scaled_training_split,
            scaling_bundle=scaling_bundle,
            A_matrix=np.asarray(A_matrix, dtype=float),
        )
        train_report = evaluate_prediction_bundle(
            training_split.targets.to_numpy(dtype=float),
            train_raw,
            train_projected,
            training_split.constraint_reference.to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
            training_split.targets.columns,
            index=training_split.targets.index,
        )
        progress_bar.update(1)

        progress_bar.set_postfix(stage="evaluate_test", objective=objective_label)
        test_projected, test_raw = _predict_unconstrained_split(
            final_model,
            feature_expander,
            scaled_test_split,
            scaling_bundle=scaling_bundle,
            A_matrix=np.asarray(A_matrix, dtype=float),
        )
        test_report = evaluate_prediction_bundle(
            test_split.targets.to_numpy(dtype=float),
            test_raw,
            test_projected,
            test_split.constraint_reference.to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
            test_split.targets.columns,
            index=test_split.targets.index,
        )
        progress_bar.update(1)

        model_bundle = _build_model_bundle(
            training_result,
            scaling_bundle,
            feature_columns=list(training_split.features.columns),
            target_columns=list(training_split.targets.columns),
            constraint_columns=list(training_split.constraint_reference.columns),
            A_matrix=np.asarray(A_matrix, dtype=float),
            model_hyperparameters=selected_hyperparameters,
            training_options={
                "objective_name": objective_label,
                "show_progress": show_progress,
            },
        )

        dataset_splits = TrainTestDatasetSplits(train=training_split, test=test_split)
        artifact_paths: dict[str, Path | None] = {
            "model_bundle": None,
            "metrics": None,
            "optuna": None,
        }

        progress_bar.set_postfix(stage="persist", objective=objective_label)
        artifact_options = dict(params.get("artifact_options", {}))
        if persist_artifacts and bool(artifact_options.get("persist_model", True)):
            metrics_payload = {
                "train": serialize_report_frames(train_report),
                "test": serialize_report_frames(test_report),
                "split_sizes": {
                    "train": int(len(dataset_splits.train.features)),
                    "test": int(len(dataset_splits.test.features)),
                },
            }
            optuna_payload = (
                dict(optuna_summary)
                if optuna_summary is not None and bool(artifact_options.get("persist_optuna", True))
                else None
            )
            metrics_summary = metrics_payload if bool(artifact_options.get("persist_metrics", True)) else None
            artifact_paths = persist_training_artifacts(
                MODEL_NAME,
                model_bundle,
                metrics_summary=metrics_summary,
                optuna_summary=optuna_payload,
                repo_root=repo_root,
                timestamp=timestamp,
            )
        progress_bar.update(1)
    finally:
        progress_bar.close()

    return {
        "best_hyperparameters": selected_hyperparameters,
        "optuna_summary": optuna_summary,
        "artifact_paths": artifact_paths,
        "train_report": train_report,
        "test_report": test_report,
        "model_bundle": model_bundle,
        "dataset_splits": dataset_splits,
    }


__all__ = [
    "MODEL_NAME",
    "build_uncobre_feature_expander",
    "build_uncobre_model",
    "load_uncobre_params",
    "predict_uncobre_model",
    "project_to_mass_balance",
    "run_uncobre_pipeline",
    "train_uncobre_model",
]
