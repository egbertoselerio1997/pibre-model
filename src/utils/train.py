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
    build_fractional_input_measured_output_dataset,
    build_measured_supervised_dataset,
    collapse_fractional_states_to_measured_outputs,
    fit_scalers,
    has_active_projection,
    inverse_transform_targets,
    project_to_mass_balance,
    transform_dataset_split,
)
from .simulation import get_repo_root, load_model_params, load_paths_config, make_simulation_timestamp
from .test import evaluate_prediction_bundle


TabularEstimatorFactory = Callable[[Mapping[str, Any]], Any]


def infer_tabular_feature_space(dataset_split: DatasetSplit) -> str:
    """Infer the feature basis used by one classical training split."""

    if any(str(column_name).startswith("In_") for column_name in dataset_split.features.columns):
        return "fractional_input"
    return "measured_composite"


def _resolve_external_measured_output_columns(
    target_columns: list[str],
    *,
    composition_matrix: np.ndarray,
    measured_output_columns: list[str] | None = None,
) -> list[str]:
    composition_array = np.asarray(composition_matrix, dtype=float)
    if composition_array.ndim != 2:
        raise ValueError("composition_matrix must be two-dimensional for external collapse.")

    measured_count = int(composition_array.shape[0])
    if measured_output_columns is not None:
        resolved_columns = [str(column_name) for column_name in measured_output_columns]
        if len(resolved_columns) != measured_count:
            raise ValueError(
                "measured_output_columns length must match composition_matrix row count for external collapse."
            )
        return resolved_columns

    target_derived_columns = [
        str(column_name).replace("Out_", "", 1) if str(column_name).startswith("Out_") else str(column_name)
        for column_name in target_columns
    ]
    if len(target_derived_columns) == measured_count:
        return target_derived_columns

    return [f"Measured_{column_index + 1}" for column_index in range(measured_count)]


def _collapse_fractional_outputs_for_comparison(
    values: np.ndarray,
    *,
    target_columns: list[str],
    state_columns: list[str],
    composition_matrix: np.ndarray,
    measured_output_columns: list[str] | None = None,
) -> np.ndarray:
    resolved_measured_output_columns = _resolve_external_measured_output_columns(
        target_columns,
        composition_matrix=np.asarray(composition_matrix, dtype=float),
        measured_output_columns=measured_output_columns,
    )
    return collapse_fractional_states_to_measured_outputs(
        np.asarray(values, dtype=float),
        state_columns,
        np.asarray(composition_matrix, dtype=float),
        resolved_measured_output_columns,
        output_prefix="Out_",
    ).to_numpy(dtype=float)


def prepare_tabular_prediction_dataset(
    dataset: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    composition_matrix: np.ndarray,
    feature_space: str,
) -> DatasetSplit:
    """Rebuild the supervised inference frames required by one persisted classical bundle."""

    if feature_space == "measured_composite":
        prepared_dataset = build_measured_supervised_dataset(
            dataset,
            dict(metadata),
            np.asarray(composition_matrix, dtype=float),
        )
    elif feature_space == "fractional_input":
        prepared_dataset = build_fractional_input_measured_output_dataset(
            dataset,
            dict(metadata),
            np.asarray(composition_matrix, dtype=float),
        )
    else:
        raise ValueError(f"Unsupported tabular feature_space: {feature_space}")

    return DatasetSplit(
        features=prepared_dataset.features,
        targets=prepared_dataset.targets,
        constraint_reference=prepared_dataset.constraint_reference,
    )


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


def get_training_device() -> tuple[Any, str]:
    """Return the best available PyTorch device and a human-readable label."""

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    return torch.device("cpu"), "cpu"


def resolve_torch_runtime_options(model_params: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve configured runtime options for PyTorch-backed models."""

    runtime_params = dict(model_params.get("runtime", {}))
    return {
        "adam_foreach": runtime_params.get("adam_foreach"),
    }


def resolve_torch_adam_options(*, device_label: str, foreach: Any = None) -> dict[str, Any]:
    """Resolve Adam keyword arguments that avoid backend fallbacks when possible."""

    if foreach is None:
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
    if not has_active_projection(A_matrix):
        return raw_predictions.copy(), raw_predictions

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
    composition_matrix: np.ndarray | None = None,
    measured_output_columns: list[str] | None = None,
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
        projected_predictions, raw_predictions = predict_tabular_regressor_split(
            training_result["model"],
            scaled_tuning_test_split,
            scaling_bundle=scaling_bundle,
            A_matrix=A_matrix,
        )
        tuning_targets = inverse_transform_targets(scaled_tuning_test_split.targets, scaling_bundle)
        tuned_predictions = projected_predictions if has_active_projection(A_matrix) else raw_predictions
        if composition_matrix is not None:
            state_columns = list(tuning_test_split.constraint_reference.columns)
            target_columns = list(tuning_test_split.targets.columns)
            tuning_targets = _collapse_fractional_outputs_for_comparison(
                tuning_targets,
                target_columns=target_columns,
                state_columns=state_columns,
                composition_matrix=np.asarray(composition_matrix, dtype=float),
                measured_output_columns=measured_output_columns,
            )
            tuned_predictions = _collapse_fractional_outputs_for_comparison(
                tuned_predictions,
                target_columns=target_columns,
                state_columns=state_columns,
                composition_matrix=np.asarray(composition_matrix, dtype=float),
                measured_output_columns=measured_output_columns,
            )
        return float(mean_squared_error(tuning_targets, tuned_predictions))

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


def tune_icsor_hyperparameters(
    tuning_train_split: DatasetSplit,
    tuning_test_split: DatasetSplit,
    *,
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray,
    measured_output_columns: list[str] | None = None,
    model_params: Mapping[str, Any],
    model_hyperparameters: Mapping[str, Any] | None = None,
    n_trials: int,
    timeout: int | None = None,
    show_progress_bar: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune the ICSOR ridge regularization strength with notebook-managed splits."""

    from src.models.ml.icsor import run_icsor_pipeline

    base_hyperparameters = resolve_model_hyperparameters(model_params, model_hyperparameters)
    if str(base_hyperparameters.get("affine_estimator", "ols")).strip().lower() != "ridge":
        raise ValueError(
            "tune_icsor_hyperparameters requires affine_estimator='ridge'; Optuna does not choose the ICSOR solver family."
        )

    if not model_params.get("search_space"):
        raise ValueError("icsor search_space must be defined in config/params.json to tune ridge_alpha.")

    base_hyperparameters = dict(base_hyperparameters)
    base_hyperparameters["objective"] = "projected_ridge"
    seed = int(model_params["hyperparameters"]["random_seed"])
    study = create_optuna_study(
        "icsor",
        seed=seed,
        pruner_config=model_params.get("pruner"),
    )

    def objective(trial: Any) -> float:
        hyperparameters = dict(base_hyperparameters)
        hyperparameters.update(suggest_parameters(trial, model_params["search_space"]))
        hyperparameters["uncertainty_method"] = "none"
        tuning_result = run_icsor_pipeline(
            tuning_train_split,
            tuning_test_split,
            np.asarray(A_matrix, dtype=float),
            composition_matrix=np.asarray(composition_matrix, dtype=float),
            measured_output_columns=measured_output_columns,
            model_params=model_params,
            model_hyperparameters=hyperparameters,
            show_progress=False,
            persist_artifacts=False,
        )
        aggregate_metrics = tuning_result["test_report"]["aggregate_metrics"]
        projected_row = aggregate_metrics.loc[aggregate_metrics["prediction_type"] == "projected"].iloc[0]
        return float(projected_row["MSE"])

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


def build_tabular_model_bundle(
    model_name: str,
    fitted_model: Any,
    scaling_bundle: ScalingBundle,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    constraint_columns: list[str],
    A_matrix: np.ndarray,
    composition_matrix: np.ndarray | None,
    measured_output_columns: list[str] | None,
    model_hyperparameters: Mapping[str, Any],
    feature_space: str,
    target_space: str,
    constraint_space: str,
) -> dict[str, Any]:
    """Assemble a persisted bundle for a scikit-compatible tabular regressor."""

    return {
        "model_name": model_name,
        "model": fitted_model,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "constraint_columns": constraint_columns,
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "composition_matrix": None if composition_matrix is None else np.asarray(composition_matrix, dtype=float),
        "measured_output_columns": None if measured_output_columns is None else list(measured_output_columns),
        "projection_active": has_active_projection(A_matrix),
        "scaling_bundle": scaling_bundle,
        "model_hyperparameters": dict(model_hyperparameters),
        "feature_space": str(feature_space),
        "target_space": str(target_space),
        "constraint_space": str(constraint_space),
        "native_prediction_space": str(target_space),
        "comparison_target_space": (
            "external_measured_output" if composition_matrix is not None else str(target_space)
        ),
        "direct_comparison_scope": (
            "externally_collapsed_measured_output_metrics"
            if composition_matrix is not None
            else "native_prediction_metrics_only"
        ),
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
        prepared_dataset = prepare_tabular_prediction_dataset(
            test_dataset,
            metadata=dict(metadata),
            composition_matrix=np.asarray(composition_matrix, dtype=float),
            feature_space=str(model_bundle.get("feature_space", "fractional_input")),
        )
        feature_frame = prepared_dataset.features
        constraint_reference = prepared_dataset.constraint_reference
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

    prediction_result: dict[str, Any] = {
        "raw_predictions": pd.DataFrame(raw_predictions, index=feature_frame.index, columns=model_bundle["target_columns"]),
        "projection_active": bool(model_bundle.get("projection_active", has_active_projection(model_bundle["A_matrix"]))),
        "constraint_reference": constraint_reference.loc[:, model_bundle["constraint_columns"]].copy(),
    }
    if prediction_result["projection_active"]:
        prediction_result["projected_predictions"] = pd.DataFrame(
            projected_predictions,
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        )
    return prediction_result


def run_tabular_regressor_pipeline(
    model_name: str,
    estimator_factory: TabularEstimatorFactory,
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
            composition_matrix=None if composition_matrix is None else np.asarray(composition_matrix, dtype=float),
            state_columns=training_split.constraint_reference.columns,
            measured_output_columns=measured_output_columns,
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
            composition_matrix=None if composition_matrix is None else np.asarray(composition_matrix, dtype=float),
            state_columns=test_split.constraint_reference.columns,
            measured_output_columns=measured_output_columns,
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
            composition_matrix=None if composition_matrix is None else np.asarray(composition_matrix, dtype=float),
            measured_output_columns=measured_output_columns,
            model_hyperparameters=selected_hyperparameters,
            feature_space=infer_tabular_feature_space(training_split),
            target_space="fractional_component",
            constraint_space="fractional_component",
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
    "train_tabular_regressor",
    "transform_feature_frame",
    "tune_icsor_hyperparameters",
    "tune_tabular_regressor_hyperparameters",
]