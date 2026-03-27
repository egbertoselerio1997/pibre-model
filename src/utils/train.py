"""Reusable training, device-selection, and artifact-persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import torch

from .io import load_pickle_file, save_json_file, save_pickle_file
from .optuna import create_optuna_study, make_study_summary, optimize_study, suggest_parameters
from .process import (
    DatasetSplit,
    ScalingBundle,
    SupervisedDatasetFrames,
    build_measured_supervised_dataset,
    combine_dataset_splits,
    fit_scalers,
    inverse_transform_targets,
    make_train_validation_test_splits,
    project_to_mass_balance,
    transform_dataset_split,
    transform_dataset_splits,
)
from .simulation import get_repo_root, load_model_params, load_paths_config, make_simulation_timestamp
from .test import evaluate_prediction_bundle


TabularEstimatorFactory = Callable[[Mapping[str, Any]], Any]


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


def resolve_model_tuning_profile(model_params: Mapping[str, Any], tuning_profile: str | None) -> dict[str, Any]:
    """Resolve a configured tuning profile for one model family."""

    profile_name = tuning_profile or str(model_params["hyperparameters"]["default_tuning_profile"])
    tuning_profiles = dict(model_params["tuning_profiles"])
    if profile_name not in tuning_profiles:
        raise KeyError(f"Unknown tuning profile: {profile_name}")
    return dict(tuning_profiles[profile_name])


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
) -> dict[str, Any]:
    """Fit a scikit-compatible tabular regressor on the provided dataset."""

    feature_frame = pd.DataFrame(training_dataset["features"])
    target_frame = pd.DataFrame(training_dataset["targets"])
    estimator = estimator_factory(model_hyperparameters)
    estimator.fit(feature_frame, target_frame)

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
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    *,
    scaling_bundle: ScalingBundle,
    A_matrix: np.ndarray,
    model_params: Mapping[str, Any],
    tuning_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune a scikit-compatible tabular regressor with Optuna."""

    profile = resolve_model_tuning_profile(model_params, tuning_profile)
    base_hyperparameters = dict(model_params["training_defaults"])
    seed = int(model_params["hyperparameters"]["random_seed"])

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
                "features": train_split.features,
                "targets": train_split.targets,
            },
            estimator_factory,
            hyperparameters,
        )
        projected_predictions, _ = predict_tabular_regressor_split(
            training_result["model"],
            validation_split,
            scaling_bundle=scaling_bundle,
            A_matrix=A_matrix,
        )
        validation_targets = inverse_transform_targets(validation_split.targets, scaling_bundle)
        return float(mean_squared_error(validation_targets, projected_predictions))

    optimize_study(
        study,
        objective,
        n_trials=int(profile["n_trials"]),
        timeout=profile.get("timeout_seconds"),
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
    """Tune, fit, evaluate, and optionally persist one measured-space tabular regressor."""

    params = dict(model_params) if model_params is not None else load_model_params(model_name, repo_root)
    measured_dataset: SupervisedDatasetFrames = build_measured_supervised_dataset(
        dataset,
        dict(metadata),
        np.asarray(composition_matrix, dtype=float),
    )
    split_params = params["hyperparameters"]
    dataset_splits = make_train_validation_test_splits(
        measured_dataset,
        test_fraction=float(split_params["test_fraction"]),
        validation_fraction=float(split_params["validation_fraction"]),
        random_seed=int(split_params["random_seed"]),
    )
    scaling_bundle = fit_scalers(
        dataset_splits.train,
        scale_features=bool(split_params["scale_features"]),
        scale_targets=bool(split_params["scale_targets"]),
    )
    scaled_splits = transform_dataset_splits(dataset_splits, scaling_bundle)
    best_hyperparameters, optuna_summary = tune_tabular_regressor_hyperparameters(
        model_name,
        estimator_factory,
        scaled_splits.train,
        scaled_splits.validation,
        scaling_bundle=scaling_bundle,
        A_matrix=np.asarray(A_matrix, dtype=float),
        model_params=params,
        tuning_profile=tuning_profile,
    )

    final_training_split = combine_dataset_splits(dataset_splits.train, dataset_splits.validation)
    final_scaling_bundle = fit_scalers(
        final_training_split,
        scale_features=bool(split_params["scale_features"]),
        scale_targets=bool(split_params["scale_targets"]),
    )
    scaled_final_training_split = transform_dataset_split(final_training_split, final_scaling_bundle)
    scaled_test_split = transform_dataset_split(dataset_splits.test, final_scaling_bundle)

    final_training_result = train_tabular_regressor(
        {
            "features": scaled_final_training_split.features,
            "targets": scaled_final_training_split.targets,
        },
        estimator_factory,
        best_hyperparameters,
    )

    final_model = final_training_result["model"]
    train_projected, train_raw = predict_tabular_regressor_split(
        final_model,
        scaled_final_training_split,
        scaling_bundle=final_scaling_bundle,
        A_matrix=np.asarray(A_matrix, dtype=float),
    )
    test_projected, test_raw = predict_tabular_regressor_split(
        final_model,
        scaled_test_split,
        scaling_bundle=final_scaling_bundle,
        A_matrix=np.asarray(A_matrix, dtype=float),
    )

    train_report = evaluate_prediction_bundle(
        final_training_split.targets.to_numpy(dtype=float),
        train_raw,
        train_projected,
        final_training_split.constraint_reference.to_numpy(dtype=float),
        np.asarray(A_matrix, dtype=float),
        final_training_split.targets.columns,
        index=final_training_split.targets.index,
    )
    test_report = evaluate_prediction_bundle(
        dataset_splits.test.targets.to_numpy(dtype=float),
        test_raw,
        test_projected,
        dataset_splits.test.constraint_reference.to_numpy(dtype=float),
        np.asarray(A_matrix, dtype=float),
        dataset_splits.test.targets.columns,
        index=dataset_splits.test.targets.index,
    )

    model_bundle = build_tabular_model_bundle(
        model_name,
        final_model,
        final_scaling_bundle,
        feature_columns=list(final_training_split.features.columns),
        target_columns=list(final_training_split.targets.columns),
        constraint_columns=list(final_training_split.constraint_reference.columns),
        A_matrix=np.asarray(A_matrix, dtype=float),
        model_hyperparameters=best_hyperparameters,
    )

    artifact_paths: dict[str, Path | None] = {
        "model_bundle": None,
        "metrics": None,
        "optuna": None,
    }
    artifact_options = dict(params.get("artifact_options", {}))
    if persist_artifacts and bool(artifact_options.get("persist_model", True)):
        metrics_payload = {
            "train": serialize_report_frames(train_report),
            "test": serialize_report_frames(test_report),
            "split_sizes": {
                "train": int(len(dataset_splits.train.features)),
                "validation": int(len(dataset_splits.validation.features)),
                "test": int(len(dataset_splits.test.features)),
            },
        }
        optuna_payload = optuna_summary if bool(artifact_options.get("persist_optuna", True)) else None
        metrics_summary = metrics_payload if bool(artifact_options.get("persist_metrics", True)) else None
        artifact_paths = persist_training_artifacts(
            model_name,
            model_bundle,
            metrics_summary=metrics_summary,
            optuna_summary=optuna_payload,
            repo_root=repo_root,
            timestamp=timestamp,
        )

    return {
        "best_hyperparameters": best_hyperparameters,
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
    "resolve_model_tuning_profile",
    "run_tabular_regressor_pipeline",
    "serialize_report_frames",
    "train_tabular_regressor",
    "transform_feature_frame",
    "tune_tabular_regressor_hyperparameters",
]