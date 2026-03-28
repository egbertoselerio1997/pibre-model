"""Reusable training, device-selection, and artifact-persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from .io import load_pickle_file, save_json_file, save_pickle_file
from .optuna import create_optuna_study, create_progress_bar, make_study_summary, optimize_study, suggest_parameters
from .process import (
    DatasetSplit,
    ScalingBundle,
    TrainTestDatasetSplits,
    build_measured_supervised_dataset,
    fit_scalers,
    inverse_transform_targets,
    project_to_mass_balance,
    transform_dataset_split,
)
from .simulation import get_repo_root, load_model_params, load_paths_config, make_simulation_timestamp
from .test import evaluate_prediction_bundle


TabularEstimatorFactory = Callable[[Mapping[str, Any]], Any]


def resolve_training_objective_label(
    model_hyperparameters: Mapping[str, Any],
    *,
    default: str = "fit",
) -> str:
    """Resolve a human-readable training objective label for progress output."""

    for key in ("objective", "loss_function", "loss", "criterion"):
        value = model_hyperparameters.get(key)
        if value is not None:
            return str(value)
    return default


def get_training_device(*, prefer_directml: bool = True) -> tuple[Any, str]:
    """Return the best available PyTorch device and a human-readable label."""

    if prefer_directml:
        try:
            import torch_directml

            return torch_directml.device(), "directml"
        except ImportError:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    return torch.device("cpu"), "cpu"


def resolve_torch_runtime_options(model_params: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve configured runtime options for PyTorch-backed models."""

    runtime_params = dict(model_params.get("runtime", {}))
    return {
        "prefer_directml": bool(runtime_params.get("prefer_directml", True)),
        "adam_foreach": runtime_params.get("adam_foreach"),
    }


def resolve_torch_adam_options(*, device_label: str, foreach: Any = None) -> dict[str, Any]:
    """Resolve Adam keyword arguments that avoid backend fallbacks when possible."""

    if foreach is None:
        if device_label == "directml":
            return {"foreach": False}
        return {}

    return {"foreach": bool(foreach)}


def render_ml_artifact_paths(
    model_name: str,
    *,
    repo_root: str | Path | None = None,
    timestamp: str | None = None,
    paths_config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Resolve the configured model, metrics, and Optuna artifact paths."""

    root = get_repo_root(repo_root)
    config = dict(paths_config) if paths_config is not None else load_paths_config(root)
    date_time = make_simulation_timestamp(timestamp)

    return {
        "model_bundle": root / Path(config["ml_model_bundle_pattern"].format(model_name=model_name, date_time=date_time)),
        "metrics": root / Path(config["ml_metrics_pattern"].format(model_name=model_name, date_time=date_time)),
        "optuna": root / Path(config["ml_optuna_pattern"].format(model_name=model_name, date_time=date_time)),
    }


def persist_training_artifacts(
    model_name: str,
    model_bundle: Mapping[str, Any],
    *,
    metrics_summary: Mapping[str, Any] | None = None,
    optuna_summary: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    timestamp: str | None = None,
    paths_config: Mapping[str, Any] | None = None,
) -> dict[str, Path | None]:
    """Persist a trained model bundle and optional metrics and Optuna summaries."""

    artifact_paths = render_ml_artifact_paths(
        model_name,
        repo_root=repo_root,
        timestamp=timestamp,
        paths_config=paths_config,
    )
    save_pickle_file(artifact_paths["model_bundle"], dict(model_bundle))

    persisted_paths: dict[str, Path | None] = {
        "model_bundle": artifact_paths["model_bundle"],
        "metrics": None,
        "optuna": None,
    }
    if metrics_summary is not None:
        save_json_file(artifact_paths["metrics"], dict(metrics_summary))
        persisted_paths["metrics"] = artifact_paths["metrics"]
    if optuna_summary is not None:
        save_json_file(artifact_paths["optuna"], dict(optuna_summary))
        persisted_paths["optuna"] = artifact_paths["optuna"]

    return persisted_paths


def load_model_bundle(path: str | Path) -> Any:
    """Load a previously persisted model bundle."""

    return load_pickle_file(path)


def resolve_model_hyperparameters(
    model_params: Mapping[str, Any],
    model_hyperparameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the explicit or default hyperparameters for one model family."""

    resolved_hyperparameters = dict(model_params["training_defaults"])
    if model_hyperparameters is not None:
        resolved_hyperparameters.update(dict(model_hyperparameters))
    return resolved_hyperparameters


def serialize_report_frames(report: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Convert report dataframes into JSON-serializable records."""

    return {
        key: dataframe.reset_index(drop=True).to_dict(orient="records")
        for key, dataframe in report.items()
    }


def transform_feature_frame(feature_frame: pd.DataFrame, scaling_bundle: ScalingBundle) -> pd.DataFrame:
    """Apply a fitted feature scaler while preserving dataframe structure."""

    aligned_features = feature_frame.loc[:, scaling_bundle.feature_columns]
    if scaling_bundle.feature_scaler is None:
        return aligned_features.copy()

    transformed_values = scaling_bundle.feature_scaler.transform(aligned_features)
    return pd.DataFrame(transformed_values, index=aligned_features.index, columns=aligned_features.columns)


def _ensure_two_dimensional_predictions(values: Any) -> np.ndarray:
    prediction_array = np.asarray(values, dtype=float)
    if prediction_array.ndim == 1:
        return prediction_array.reshape(-1, 1)
    return prediction_array


def train_tabular_regressor(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    estimator_factory: TabularEstimatorFactory,
    model_hyperparameters: Mapping[str, Any],
    *,
    show_progress: bool = True,
    progress_description: str | None = None,
) -> dict[str, Any]:
    """Fit a scikit-compatible tabular regressor on the provided dataset."""

    feature_frame = pd.DataFrame(training_dataset["features"])
    target_frame = pd.DataFrame(training_dataset["targets"])
    estimator = estimator_factory(model_hyperparameters)

    objective_label = resolve_training_objective_label(model_hyperparameters)
    progress_bar = create_progress_bar(
        total=1,
        desc=progress_description or f"Train {estimator.__class__.__name__}",
        enabled=show_progress,
        unit="fit",
    )
    progress_bar.set_postfix(objective=objective_label)
    try:
        estimator.fit(feature_frame, target_frame)
        progress_bar.update(1)
        progress_bar.set_postfix(objective=objective_label, status="complete")
    finally:
        progress_bar.close()

    return {
        "model": estimator,
    }


def predict_tabular_regressor_split(
    model: Any,
    dataset_split: DatasetSplit,
    *,
    scaling_bundle: ScalingBundle,
    A_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate raw and projected predictions for one prepared dataset split."""

    raw_predictions = _ensure_two_dimensional_predictions(model.predict(dataset_split.features))
    raw_predictions = inverse_transform_targets(raw_predictions, scaling_bundle)
    projected_predictions = project_to_mass_balance(
        raw_predictions,
        dataset_split.constraint_reference.to_numpy(dtype=float),
        np.asarray(A_matrix, dtype=float),
    )
    return projected_predictions, raw_predictions


def tune_tabular_regressor_hyperparameters(
    model_name: str,
    estimator_factory: TabularEstimatorFactory,
    tuning_train_split: DatasetSplit,
    tuning_test_split: DatasetSplit,
    *,
    A_matrix: np.ndarray,
    model_params: Mapping[str, Any],
    n_trials: int,
    timeout: int | None = None,
    show_progress_bar: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune a scikit-compatible tabular regressor with Optuna."""

    base_hyperparameters = dict(model_params["training_defaults"])
    split_params = model_params["hyperparameters"]
    seed = int(split_params["random_seed"])
    scaling_bundle = fit_scalers(
        tuning_train_split,
        scale_features=bool(split_params["scale_features"]),
        scale_targets=bool(split_params["scale_targets"]),
    )
    scaled_tuning_train_split = transform_dataset_split(tuning_train_split, scaling_bundle)
    scaled_tuning_test_split = transform_dataset_split(tuning_test_split, scaling_bundle)

    study = create_optuna_study(
        model_name,
        seed=seed,
        pruner_config=model_params.get("pruner"),
    )

    def objective(trial: Any) -> float:
        hyperparameters = dict(base_hyperparameters)
        hyperparameters.update(suggest_parameters(trial, model_params["search_space"]))
        training_result = train_tabular_regressor(
            {
                "features": scaled_tuning_train_split.features,
                "targets": scaled_tuning_train_split.targets,
            },
            estimator_factory,
            hyperparameters,
            show_progress=False,
        )
        projected_predictions, _ = predict_tabular_regressor_split(
            training_result["model"],
            scaled_tuning_test_split,
            scaling_bundle=scaling_bundle,
            A_matrix=A_matrix,
        )
        tuning_targets = inverse_transform_targets(scaled_tuning_test_split.targets, scaling_bundle)
        return float(mean_squared_error(tuning_targets, projected_predictions))

    optimize_study(
        study,
        objective,
        n_trials=int(n_trials),
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        objective_name="validation_mse",
    )
    best_hyperparameters = dict(base_hyperparameters)
    best_hyperparameters.update(study.best_trial.params)
    return best_hyperparameters, make_study_summary(study)


def tune_cobre_hyperparameters(
    tuning_train_split: DatasetSplit,
    tuning_test_split: DatasetSplit,
    *,
    A_matrix: np.ndarray,
    model_params: Mapping[str, Any],
    tuning_epochs: int,
    n_trials: int,
    timeout: int | None = None,
    show_progress_bar: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune COBRE with notebook-managed tuning splits."""

    from src.models.ml.cobre import train_cobre_model

    params = dict(model_params)
    split_params = params["hyperparameters"]
    runtime_options = resolve_torch_runtime_options(params)
    seed = int(split_params["random_seed"])
    scaling_bundle = fit_scalers(
        tuning_train_split,
        scale_features=bool(split_params["scale_features"]),
        scale_targets=bool(split_params["scale_targets"]),
    )
    scaled_tuning_train_split = transform_dataset_split(tuning_train_split, scaling_bundle)
    scaled_tuning_test_split = transform_dataset_split(tuning_test_split, scaling_bundle)
    base_hyperparameters = dict(params["training_defaults"])

    study = create_optuna_study(
        "cobre",
        seed=seed,
        pruner_config=params.get("pruner"),
    )

    def objective(trial: Any) -> float:
        hyperparameters = dict(base_hyperparameters)
        hyperparameters.update(suggest_parameters(trial, params["search_space"]))
        tuned_batch_size = int(hyperparameters.get("batch_size", split_params["batch_size"]))
        result = train_cobre_model(
            {
                "features": scaled_tuning_train_split.features,
                "targets": scaled_tuning_train_split.targets,
                "constraint_reference": scaled_tuning_train_split.constraint_reference,
            },
            hyperparameters,
            A_matrix=A_matrix,
            training_options={
                "epochs": int(tuning_epochs),
                "batch_size": tuned_batch_size,
                "random_seed": seed + int(trial.number),
                "log_interval": int(split_params["log_interval"]),
                "early_stopping_patience_epochs": int(split_params["early_stopping_patience_epochs"]),
                "prefer_directml": runtime_options["prefer_directml"],
                "adam_foreach": runtime_options["adam_foreach"],
                "show_progress": False,
                "validation_dataset": {
                    "features": scaled_tuning_test_split.features,
                    "targets": scaled_tuning_test_split.targets,
                    "constraint_reference": scaled_tuning_test_split.constraint_reference,
                },
            },
            trial=trial,
        )
        return float(result["best_validation_loss"])

    optimize_study(
        study,
        objective,
        n_trials=int(n_trials),
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        objective_name="validation_loss",
    )
    best_hyperparameters = dict(base_hyperparameters)
    best_hyperparameters.update(study.best_trial.params)
    return best_hyperparameters, make_study_summary(study)


def tune_uncobre_hyperparameters(
    tuning_train_split: DatasetSplit,
    tuning_test_split: DatasetSplit,
    *,
    A_matrix: np.ndarray,
    model_params: Mapping[str, Any],
    n_trials: int,
    timeout: int | None = None,
    show_progress_bar: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune UNCOBRE with notebook-managed tuning splits."""

    from src.models.ml.uncobre import train_uncobre_model

    params = dict(model_params)
    split_params = params["hyperparameters"]
    seed = int(split_params["random_seed"])
    scaling_bundle = fit_scalers(
        tuning_train_split,
        scale_features=bool(split_params["scale_features"]),
        scale_targets=bool(split_params["scale_targets"]),
    )
    scaled_tuning_train_split = transform_dataset_split(tuning_train_split, scaling_bundle)
    scaled_tuning_test_split = transform_dataset_split(tuning_test_split, scaling_bundle)
    base_hyperparameters = dict(params["training_defaults"])

    study = create_optuna_study(
        "uncobre",
        seed=seed,
        pruner_config=params.get("pruner"),
    )

    def objective(trial: Any) -> float:
        hyperparameters = dict(base_hyperparameters)
        hyperparameters.update(suggest_parameters(trial, params["search_space"]))

        training_result = train_uncobre_model(
            {
                "features": scaled_tuning_train_split.features,
                "targets": scaled_tuning_train_split.targets,
            },
            hyperparameters,
            training_options={
                "show_progress": False,
                "progress_description": "Train UNCOBRE",
                "objective_name": str(hyperparameters.get("regression_mode", "ridge")),
            },
        )

        expanded_validation_features = training_result["feature_expander"].transform(
            scaled_tuning_test_split.features,
        )
        raw_predictions_scaled = np.asarray(training_result["model"].predict(expanded_validation_features), dtype=float)
        if raw_predictions_scaled.ndim == 1:
            raw_predictions_scaled = raw_predictions_scaled.reshape(-1, 1)

        raw_predictions = inverse_transform_targets(raw_predictions_scaled, scaling_bundle)
        projected_predictions = project_to_mass_balance(
            raw_predictions,
            scaled_tuning_test_split.constraint_reference.to_numpy(dtype=float),
            np.asarray(A_matrix, dtype=float),
        )
        tuning_targets = inverse_transform_targets(scaled_tuning_test_split.targets, scaling_bundle)
        return float(mean_squared_error(tuning_targets, projected_predictions))

    optimize_study(
        study,
        objective,
        n_trials=int(n_trials),
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        objective_name="validation_projected_mse",
    )
    best_hyperparameters = dict(base_hyperparameters)
    best_hyperparameters.update(study.best_trial.params)
    return best_hyperparameters, make_study_summary(study)


def build_tabular_model_bundle(
    model_name: str,
    fitted_model: Any,
    scaling_bundle: ScalingBundle,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    constraint_columns: list[str],
    A_matrix: np.ndarray,
    model_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble a persisted bundle for a scikit-compatible tabular regressor."""

    return {
        "model_name": model_name,
        "model": fitted_model,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "constraint_columns": constraint_columns,
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "scaling_bundle": scaling_bundle,
        "model_hyperparameters": dict(model_hyperparameters),
    }


def predict_tabular_regressor_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a persisted tabular regressor bundle and generate aligned predictions."""

    model_bundle = load_model_bundle(model_path)
    scaling_bundle: ScalingBundle = model_bundle["scaling_bundle"]

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
        constraint_reference = pd.DataFrame(
            test_dataset["constraint_reference"],
            columns=model_bundle["constraint_columns"],
        )

    transformed_features = transform_feature_frame(feature_frame, scaling_bundle)
    prediction_split = DatasetSplit(
        features=transformed_features,
        targets=pd.DataFrame(
            np.zeros((len(feature_frame), len(model_bundle["target_columns"]))),
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        constraint_reference=constraint_reference.loc[:, model_bundle["constraint_columns"]],
    )
    projected_predictions, raw_predictions = predict_tabular_regressor_split(
        model_bundle["model"],
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
        "constraint_reference": constraint_reference.loc[:, model_bundle["constraint_columns"]].copy(),
    }


def run_tabular_regressor_pipeline(
    model_name: str,
    estimator_factory: TabularEstimatorFactory,
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
    """Train, evaluate, and optionally persist one measured-space tabular regressor."""

    params = dict(model_params) if model_params is not None else load_model_params(model_name, repo_root)
    split_params = params["hyperparameters"]
    selected_hyperparameters = resolve_model_hyperparameters(params, model_hyperparameters)
    objective_label = resolve_training_objective_label(selected_hyperparameters)
    progress_bar = create_progress_bar(
        total=5,
        desc=f"Training {model_name}",
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
        final_training_result = train_tabular_regressor(
            {
                "features": scaled_training_split.features,
                "targets": scaled_training_split.targets,
            },
            estimator_factory,
            selected_hyperparameters,
            show_progress=False,
            progress_description=f"Train {model_name}",
        )
        progress_bar.update(1)

        final_model = final_training_result["model"]
        progress_bar.set_postfix(stage="evaluate_train", objective=objective_label)
        train_projected, train_raw = predict_tabular_regressor_split(
            final_model,
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
        test_projected, test_raw = predict_tabular_regressor_split(
            final_model,
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

        model_bundle = build_tabular_model_bundle(
            model_name,
            final_model,
            scaling_bundle,
            feature_columns=list(training_split.features.columns),
            target_columns=list(training_split.targets.columns),
            constraint_columns=list(training_split.constraint_reference.columns),
            A_matrix=np.asarray(A_matrix, dtype=float),
            model_hyperparameters=selected_hyperparameters,
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
            optuna_payload = dict(optuna_summary) if optuna_summary is not None and bool(artifact_options.get("persist_optuna", True)) else None
            metrics_summary = metrics_payload if bool(artifact_options.get("persist_metrics", True)) else None
            artifact_paths = persist_training_artifacts(
                model_name,
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
    "TabularEstimatorFactory",
    "build_tabular_model_bundle",
    "get_training_device",
    "load_model_bundle",
    "predict_tabular_regressor_model",
    "predict_tabular_regressor_split",
    "persist_training_artifacts",
    "render_ml_artifact_paths",
    "resolve_model_hyperparameters",
    "resolve_training_objective_label",
    "resolve_torch_adam_options",
    "resolve_torch_runtime_options",
    "run_tabular_regressor_pipeline",
    "serialize_report_frames",
    "tune_cobre_hyperparameters",
    "tune_uncobre_hyperparameters",
    "train_tabular_regressor",
    "transform_feature_frame",
    "tune_tabular_regressor_hyperparameters",
]