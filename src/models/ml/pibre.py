"""Physics-informed bilinear regression helpers and model implementation."""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.simulation import (
    get_repo_root,
    load_model_params,
    load_paths_config,
    make_simulation_timestamp,
)


MODEL_NAME = "pibre"
DEFAULT_ARTIFACT_NAME = "trained_model"


def load_pibre_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured PIBRe parameter namespace."""

    return load_model_params(MODEL_NAME, repo_root)


def get_training_device(device: Any | None = None) -> tuple[Any, str]:
    """Resolve the requested device or the preferred runtime backend."""

    if device is not None:
        resolved = torch.device(device)
        return resolved, str(resolved)

    try:
        import torch_directml
    except ImportError:
        torch_directml = None

    if torch_directml is not None:
        try:
            directml_device = torch_directml.device()
            return cast(Any, directml_device), "DirectML"
        except RuntimeError:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"

    return torch.device("cpu"), "CPU"


def split_pibre_dataset(
    dataset: pd.DataFrame,
    hyperparameters: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train and test partitions for PIBRe."""

    test_size = float(hyperparameters["test_size"])
    random_seed = int(hyperparameters["random_seed"])
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=random_seed)
    return train_dataset.copy(), test_dataset.copy()


def get_feature_columns(metadata: Mapping[str, Any]) -> list[str]:
    """Return the configured independent-feature column order."""

    return list(metadata["independent_columns"])


def get_target_columns(metadata: Mapping[str, Any]) -> list[str]:
    """Return the measured-output target columns used by PIBRe."""

    return [f"Out_{name}" for name in metadata["measured_output_columns"]]


def get_influent_state_columns(metadata: Mapping[str, Any]) -> list[str]:
    """Return the influent state columns needed to derive measured composites."""

    return [f"In_{name}" for name in metadata["state_columns"]]


def build_measured_influent_composites(
    dataset: pd.DataFrame,
    metadata: Mapping[str, Any],
    composition_matrix: np.ndarray,
) -> np.ndarray:
    """Map influent states into measured-output composite space."""

    influent_state_columns = get_influent_state_columns(metadata)
    influent_states = dataset.loc[:, influent_state_columns].to_numpy(dtype=float)
    return influent_states @ np.asarray(composition_matrix, dtype=float).T


def prepare_pibre_arrays(
    dataset: pd.DataFrame,
    metadata: Mapping[str, Any],
    composition_matrix: np.ndarray,
    *,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> dict[str, Any]:
    """Build feature, target, and influent composite arrays for PIBRe."""

    feature_columns = get_feature_columns(metadata)
    target_columns = get_target_columns(metadata)
    x_array = dataset.loc[:, feature_columns].to_numpy(dtype=float)
    y_array = dataset.loc[:, target_columns].to_numpy(dtype=float)
    cin_measured = build_measured_influent_composites(dataset, metadata, composition_matrix)

    resolved_scaler = StandardScaler() if scaler is None else scaler
    if fit_scaler or scaler is None:
        x_scaled = resolved_scaler.fit_transform(x_array)
    else:
        x_scaled = resolved_scaler.transform(x_array)

    return {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "x": x_array,
        "x_scaled": x_scaled,
        "y": y_array,
        "cin_measured": cin_measured,
        "scaler": resolved_scaler,
    }


class ProjectedPIBRe_DML(nn.Module):
    """Bilinear regression model with mass-balance projection in output space."""

    def __init__(self, input_dim: int, output_dim: int, a_matrix: np.ndarray) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bilinear = nn.Bilinear(input_dim, input_dim, output_dim)

        a_tensor = torch.tensor(np.asarray(a_matrix), dtype=torch.float32)
        if a_tensor.ndim == 1:
            a_tensor = a_tensor.unsqueeze(0)

        self.register_buffer("A", a_tensor)

        if a_tensor.shape[0] == 0:
            projector = torch.zeros((output_dim, 0), dtype=torch.float32)
        else:
            gram = a_tensor @ a_tensor.T
            projector = a_tensor.T @ torch.linalg.pinv(gram)

        self.register_buffer("projector", projector)

    def forward(self, x_tensor: torch.Tensor, c_in_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_raw = self.linear(x_tensor) + self.bilinear(x_tensor, x_tensor)

        if self.A.shape[0] == 0:
            return c_raw, c_raw

        residual = torch.matmul(c_raw - c_in_tensor, self.A.T)
        correction = torch.matmul(residual, self.projector.T)
        c_projected = c_raw - correction
        return c_projected, c_raw


def project_to_mass_balance(c_raw: np.ndarray, c_in: np.ndarray, a_matrix: np.ndarray) -> np.ndarray:
    """Project raw predictions onto the affine mass-balance constraint set."""

    a_array = np.asarray(a_matrix, dtype=float)
    if a_array.shape[0] == 0:
        return np.asarray(c_raw, dtype=float).copy()

    gram = a_array @ a_array.T
    projector = a_array.T @ np.linalg.pinv(gram)
    residual = (np.asarray(c_raw, dtype=float) - np.asarray(c_in, dtype=float)) @ a_array.T
    correction = residual @ projector.T
    return np.asarray(c_raw, dtype=float) - correction


def compute_mass_balance_violation(c_out: np.ndarray, c_in: np.ndarray, a_matrix: np.ndarray) -> np.ndarray:
    """Compute the L2 norm of the mass-balance residual for each sample."""

    a_array = np.asarray(a_matrix, dtype=float)
    b_out = np.asarray(c_out, dtype=float) @ a_array.T
    b_in = np.asarray(c_in, dtype=float) @ a_array.T
    return np.linalg.norm(b_out - b_in, axis=1)


def tune_pibre_hyperparameters(
    training_dataset: pd.DataFrame,
    hyperparameters: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    composition_matrix: np.ndarray,
    A_matrix: np.ndarray,
    device: Any | None = None,
) -> dict[str, Any]:
    """Run Optuna tuning for PIBRe on a capped subset of the training data."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    random_seed = int(hyperparameters["random_seed"])
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    resolved_device, device_name = get_training_device(device)

    prepared = prepare_pibre_arrays(training_dataset, metadata, composition_matrix, fit_scaler=False)
    sample_count = prepared["x"].shape[0]
    tuning_subset_size = min(int(hyperparameters["tuning_subset_size"]), sample_count)
    validation_size = float(hyperparameters["validation_size"])
    rng = np.random.default_rng(random_seed)
    tuning_indices = rng.choice(sample_count, size=tuning_subset_size, replace=False)

    x_pool = prepared["x"][tuning_indices]
    y_pool = prepared["y"][tuning_indices]
    cin_pool = prepared["cin_measured"][tuning_indices]

    x_train, x_val, y_train, y_val, cin_train, cin_val = train_test_split(
        x_pool,
        y_pool,
        cin_pool,
        test_size=validation_size,
        random_state=random_seed,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    x_train_tensor = _to_tensor(x_train_scaled, resolved_device)
    y_train_tensor = _to_tensor(y_train, resolved_device)
    cin_train_tensor = _to_tensor(cin_train, resolved_device)
    x_val_tensor = _to_tensor(x_val_scaled, resolved_device)
    y_val_tensor = _to_tensor(y_val, resolved_device)
    cin_val_tensor = _to_tensor(cin_val, resolved_device)
    target_mean = _to_tensor(y_train.mean(axis=0), resolved_device)

    optuna_trials = int(hyperparameters["optuna_trials"])
    tuning_epochs = int(hyperparameters["tuning_epochs"])
    validation_frequency = int(hyperparameters["validation_frequency"])

    if optuna_trials <= 0:
        return {
            "best_params": {},
            "best_value": float("nan"),
            "completed_trials": 0,
            "pruned_trials": 0,
            "device_name": device_name,
            "tuning_subset_size": tuning_subset_size,
        }

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = {
            "learning_rate": _suggest_from_spec(trial, "learning_rate", hyperparameters["learning_rate_search"]),
            "lambda_l1": _suggest_from_spec(trial, "lambda_l1", hyperparameters["lambda_l1_search"]),
            "weight_decay": _suggest_from_spec(trial, "weight_decay", hyperparameters["weight_decay_search"]),
            "clip_max_norm": _suggest_from_spec(trial, "clip_max_norm", hyperparameters["clip_max_norm_search"]),
        }

        _, _, validation_metric = _fit_pibre_model(
            input_dim=x_train_tensor.shape[1],
            output_dim=y_train_tensor.shape[1],
            a_matrix=A_matrix,
            target_mean=target_mean,
            x_train_tensor=x_train_tensor,
            y_train_tensor=y_train_tensor,
            cin_train_tensor=cin_train_tensor,
            learning_rate=float(trial_params["learning_rate"]),
            lambda_l1=float(trial_params["lambda_l1"]),
            weight_decay=float(trial_params["weight_decay"]),
            clip_max_norm=float(trial_params["clip_max_norm"]),
            epochs=tuning_epochs,
            device=resolved_device,
            random_seed=random_seed,
            validation_tensors=(x_val_tensor, y_val_tensor, cin_val_tensor),
            validation_frequency=validation_frequency,
            trial=trial,
            show_progress=bool(hyperparameters.get("show_training_progress", False)),
        )

        return float(validation_metric)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(hyperparameters["pruner_startup_trials"]),
        n_warmup_steps=int(hyperparameters["pruner_warmup_steps"]),
        interval_steps=validation_frequency,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=bool(hyperparameters.get("show_optuna_progress", False)))

    pruned_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED])
    return {
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "completed_trials": len(study.trials),
        "pruned_trials": pruned_trials,
        "device_name": device_name,
        "tuning_subset_size": tuning_subset_size,
    }


def train_pibre_model(
    training_dataset: pd.DataFrame,
    hyperparameters: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    composition_matrix: np.ndarray,
    A_matrix: np.ndarray,
    device: Any | None = None,
    save_model: bool | None = None,
    repo_root: str | Path | None = None,
    paths_config: Mapping[str, Any] | None = None,
    artifact_name: str | None = None,
    timestamp: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train PIBRe on a training dataset and optionally persist a model artifact."""

    random_seed = int(hyperparameters["random_seed"])
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    resolved_device, device_name = get_training_device(device)
    prepared = prepare_pibre_arrays(training_dataset, metadata, composition_matrix, fit_scaler=True)

    x_train_tensor = _to_tensor(prepared["x_scaled"], resolved_device)
    y_train_tensor = _to_tensor(prepared["y"], resolved_device)
    cin_train_tensor = _to_tensor(prepared["cin_measured"], resolved_device)
    target_mean = _to_tensor(prepared["y"].mean(axis=0), resolved_device)

    model, optimizer, training_metric = _fit_pibre_model(
        input_dim=x_train_tensor.shape[1],
        output_dim=y_train_tensor.shape[1],
        a_matrix=A_matrix,
        target_mean=target_mean,
        x_train_tensor=x_train_tensor,
        y_train_tensor=y_train_tensor,
        cin_train_tensor=cin_train_tensor,
        learning_rate=float(hyperparameters["learning_rate"]),
        lambda_l1=float(hyperparameters["lambda_l1"]),
        weight_decay=float(hyperparameters["weight_decay"]),
        clip_max_norm=float(hyperparameters["clip_max_norm"]),
        epochs=int(hyperparameters["final_training_epochs"]),
        device=resolved_device,
        random_seed=random_seed,
        validation_tensors=None,
        validation_frequency=int(hyperparameters["validation_frequency"]),
        trial=None,
        show_progress=bool(hyperparameters.get("show_training_progress", False)),
    )

    raw_predictions, projected_predictions = _predict_arrays(
        model,
        prepared["x_scaled"],
        prepared["cin_measured"],
        resolved_device,
    )

    training_prediction_frame = _build_prediction_frame(
        training_dataset.index,
        prepared["target_columns"],
        raw_predictions,
        projected_predictions,
    )

    trained_bundle = {
        "model": model,
        "optimizer": optimizer,
        "scaler": prepared["scaler"],
        "feature_columns": prepared["feature_columns"],
        "target_columns": prepared["target_columns"],
        "state_columns": list(metadata["state_columns"]),
        "measured_output_columns": list(metadata["measured_output_columns"]),
        "composition_matrix": np.asarray(composition_matrix, dtype=float),
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "training_hyperparameters": dict(hyperparameters),
        "device_name": device_name,
        "training_metric": float(training_metric),
    }

    should_save = bool(hyperparameters.get("save_model", True)) if save_model is None else save_model
    if should_save:
        model_path = save_pibre_model(
            trained_bundle,
            artifact_name=artifact_name or str(hyperparameters.get("artifact_name", DEFAULT_ARTIFACT_NAME)),
            repo_root=repo_root,
            paths_config=paths_config,
            timestamp=timestamp,
        )
        trained_bundle["model_path"] = model_path

    return trained_bundle, training_prediction_frame


def save_pibre_model(
    trained_bundle: Mapping[str, Any],
    *,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    repo_root: str | Path | None = None,
    paths_config: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Persist a trained PIBRe model and preprocessing bundle as a pickle file."""

    artifact_path = _render_model_artifact_path(
        MODEL_NAME,
        artifact_name,
        repo_root=repo_root,
        paths_config=paths_config,
        timestamp=timestamp,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    model = trained_bundle["model"]
    state_dict = {
        parameter_name: parameter_value.detach().cpu()
        for parameter_name, parameter_value in model.state_dict().items()
    }
    payload = {
        "model_name": MODEL_NAME,
        "state_dict": state_dict,
        "scaler": trained_bundle["scaler"],
        "feature_columns": list(trained_bundle["feature_columns"]),
        "target_columns": list(trained_bundle["target_columns"]),
        "state_columns": list(trained_bundle["state_columns"]),
        "measured_output_columns": list(trained_bundle["measured_output_columns"]),
        "composition_matrix": np.asarray(trained_bundle["composition_matrix"], dtype=float),
        "A_matrix": np.asarray(trained_bundle["A_matrix"], dtype=float),
        "training_hyperparameters": dict(trained_bundle["training_hyperparameters"]),
        "input_dim": int(len(trained_bundle["feature_columns"])),
        "output_dim": int(len(trained_bundle["target_columns"])),
    }

    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    return artifact_path


def load_pibre_model(
    model_path: str | Path,
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    """Load a trained PIBRe model artifact."""

    with Path(model_path).open("rb") as handle:
        payload = pickle.load(handle)

    resolved_device, device_name = get_training_device(device)
    model = ProjectedPIBRe_DML(payload["input_dim"], payload["output_dim"], payload["A_matrix"]).to(resolved_device)
    state_dict = {
        parameter_name: parameter_value.to(resolved_device)
        if isinstance(parameter_value, torch.Tensor)
        else parameter_value
        for parameter_name, parameter_value in payload["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    payload["model"] = model
    payload["device"] = resolved_device
    payload["device_name"] = device_name
    return payload


def predict_pibre(
    test_dataset: pd.DataFrame,
    model_path: str | Path,
    *,
    device: Any | None = None,
) -> pd.DataFrame:
    """Predict physically projected measured outputs for a test dataset."""

    payload = load_pibre_model(model_path, device=device)
    raw_predictions, projected_predictions, _ = _predict_from_payload(test_dataset, payload)
    del raw_predictions
    return pd.DataFrame(projected_predictions, index=test_dataset.index, columns=payload["target_columns"])


def evaluate_pibre_model(
    test_dataset: pd.DataFrame,
    model_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluate raw and projected PIBRe predictions on a held-out dataset."""

    payload = load_pibre_model(model_path, device=device)
    raw_predictions, projected_predictions, cin_measured = _predict_from_payload(test_dataset, payload)
    y_true = test_dataset.loc[:, payload["target_columns"]].to_numpy(dtype=float)

    report_df = pd.DataFrame(
        [
            {
                "model": "PIBRe raw",
                **_calculate_standard_metrics(y_true, raw_predictions),
                "mean_mb_error": float(compute_mass_balance_violation(raw_predictions, cin_measured, payload["A_matrix"]).mean()),
            },
            {
                "model": "PIBRe projected",
                **_calculate_standard_metrics(y_true, projected_predictions),
                "mean_mb_error": float(compute_mass_balance_violation(projected_predictions, cin_measured, payload["A_matrix"]).mean()),
            },
        ]
    )

    per_target_df = pd.DataFrame({"target": payload["target_columns"]})
    raw_metrics = _calculate_per_target_metrics(y_true, raw_predictions)
    projected_metrics = _calculate_per_target_metrics(y_true, projected_predictions)
    for metric_name, metric_values in raw_metrics.items():
        per_target_df[f"raw_{metric_name.lower()}"] = metric_values
    for metric_name, metric_values in projected_metrics.items():
        per_target_df[f"projected_{metric_name.lower()}"] = metric_values

    prediction_frame = _build_prediction_frame(
        test_dataset.index,
        payload["target_columns"],
        raw_predictions,
        projected_predictions,
    )

    return {
        "report_df": report_df,
        "per_target_df": per_target_df,
        "prediction_frame": prediction_frame,
        "raw_predictions": raw_predictions,
        "projected_predictions": projected_predictions,
        "cin_measured": cin_measured,
    }


def _fit_pibre_model(
    *,
    input_dim: int,
    output_dim: int,
    a_matrix: np.ndarray,
    target_mean: torch.Tensor,
    x_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    cin_train_tensor: torch.Tensor,
    learning_rate: float,
    lambda_l1: float,
    weight_decay: float,
    clip_max_norm: float,
    epochs: int,
    device: torch.device,
    random_seed: int,
    validation_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    validation_frequency: int,
    trial: optuna.trial.Trial | None,
    show_progress: bool,
) -> tuple[ProjectedPIBRe_DML, Any, float]:
    torch.manual_seed(random_seed)
    model = ProjectedPIBRe_DML(input_dim, output_dim, a_matrix).to(device)

    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.copy_(target_mean)
        model.bilinear.weight.zero_()
        model.bilinear.bias.zero_()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # type: ignore[attr-defined]
    criterion = nn.MSELoss()
    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    metric_value = float("nan")

    progress_iterable = range(epochs)
    if show_progress:
        from tqdm.auto import tqdm

        progress_iterable = tqdm(progress_iterable, desc="PIBRe training", unit="epoch")

    for epoch in progress_iterable:
        model.train()
        optimizer.zero_grad()
        projected_train, _ = model(x_train_tensor, cin_train_tensor)
        mse_loss = criterion(projected_train, y_train_tensor)
        l1_penalty = torch.norm(model.linear.weight, p=1) + torch.norm(model.bilinear.weight, p=1)
        loss = mse_loss + lambda_l1 * l1_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step()

        metric_value = float(mse_loss.item())
        should_validate = validation_tensors is not None and (
            (epoch + 1) % max(validation_frequency, 1) == 0 or (epoch + 1) == epochs
        )
        if should_validate:
            assert validation_tensors is not None
            x_val_tensor, y_val_tensor, cin_val_tensor = validation_tensors
            model.eval()
            with torch.no_grad():
                projected_val, _ = model(x_val_tensor, cin_val_tensor)
                metric_value = float(criterion(projected_val, y_val_tensor).item())

            if metric_value < best_metric:
                best_metric = metric_value
                best_state = deepcopy(model.state_dict())

            if trial is not None:
                trial.report(metric_value, step=epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, optimizer, float(metric_value if best_state is None else best_metric)


def _predict_arrays(
    model: ProjectedPIBRe_DML,
    x_scaled: np.ndarray,
    cin_measured: np.ndarray,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    x_tensor = _to_tensor(x_scaled, device)
    cin_tensor = _to_tensor(cin_measured, device)
    model.eval()
    with torch.no_grad():
        _, raw_predictions = model(x_tensor, cin_tensor)
    raw_predictions_np = raw_predictions.detach().cpu().numpy()
    projected_predictions_np = project_to_mass_balance(
        raw_predictions_np,
        cin_measured,
        model.A.detach().cpu().numpy(),
    )
    return raw_predictions_np, projected_predictions_np


def _predict_from_payload(
    dataset: pd.DataFrame,
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata_like = {
        "independent_columns": payload["feature_columns"],
        "measured_output_columns": payload["measured_output_columns"],
        "state_columns": payload["state_columns"],
    }
    prepared = prepare_pibre_arrays(
        dataset,
        metadata_like,
        payload["composition_matrix"],
        scaler=payload["scaler"],
        fit_scaler=False,
    )
    raw_predictions, projected_predictions = _predict_arrays(
        payload["model"],
        prepared["x_scaled"],
        prepared["cin_measured"],
        payload["device"],
    )
    return raw_predictions, projected_predictions, prepared["cin_measured"]


def _to_tensor(array: np.ndarray, device: Any) -> torch.Tensor:
    return torch.tensor(np.asarray(array), dtype=torch.float32, device=device)


def _suggest_from_spec(
    trial: optuna.trial.Trial,
    name: str,
    spec: Mapping[str, Any],
) -> float:
    return float(
        trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    )


def _render_model_artifact_path(
    model_name: str,
    artifact_name: str,
    *,
    repo_root: str | Path | None = None,
    paths_config: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> Path:
    root = get_repo_root(repo_root)
    config = dict(paths_config) if paths_config is not None else load_paths_config(root)
    relative_path = Path(
        config["ml_model_artifact_pattern"].format(
            model_name=model_name,
            artifact_name=artifact_name,
            date_time=make_simulation_timestamp(timestamp),
        )
    )
    return root / relative_path


def _build_prediction_frame(
    index: pd.Index,
    target_columns: list[str],
    raw_predictions: np.ndarray,
    projected_predictions: np.ndarray,
) -> pd.DataFrame:
    raw_frame = pd.DataFrame(
        raw_predictions,
        index=index,
        columns=[f"Raw_{column_name}" for column_name in target_columns],
    )
    projected_frame = pd.DataFrame(
        projected_predictions,
        index=index,
        columns=[f"Projected_{column_name}" for column_name in target_columns],
    )
    return pd.concat([raw_frame, projected_frame], axis=1)


def _calculate_standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse_value = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "MSE": mse_value,
        "RMSE": float(np.sqrt(mse_value)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(_mean_absolute_percentage_error(y_true, y_pred)),
    }


def _calculate_per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    mse_values = np.asarray(mean_squared_error(y_true, y_pred, multioutput="raw_values"), dtype=float)
    return {
        "R2": np.asarray(r2_score(y_true, y_pred, multioutput="raw_values"), dtype=float),
        "MSE": mse_values,
        "RMSE": np.sqrt(mse_values),
        "MAE": np.asarray(mean_absolute_error(y_true, y_pred, multioutput="raw_values"), dtype=float),
        "MAPE": np.asarray(_mean_absolute_percentage_error(y_true, y_pred, per_target=True), dtype=float),
    }


def _mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    per_target: bool = False,
) -> float | np.ndarray:
    denominator = np.maximum(np.abs(np.asarray(y_true, dtype=float)), 1e-9)
    percentage_error = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) / denominator
    if per_target:
        return percentage_error.mean(axis=0) * 100.0
    return float(percentage_error.mean() * 100.0)


__all__ = [
    "MODEL_NAME",
    "ProjectedPIBRe_DML",
    "build_measured_influent_composites",
    "compute_mass_balance_violation",
    "evaluate_pibre_model",
    "get_feature_columns",
    "get_influent_state_columns",
    "get_target_columns",
    "get_training_device",
    "load_pibre_model",
    "load_pibre_params",
    "predict_pibre",
    "prepare_pibre_arrays",
    "project_to_mass_balance",
    "save_pibre_model",
    "split_pibre_dataset",
    "train_pibre_model",
    "tune_pibre_hyperparameters",
]