"""Physics-informed bilinear regression in measured-output space."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.utils.process import (
    DatasetSplit,
    ScalingBundle,
    TrainTestDatasetSplits,
    build_projection_operator,
    build_measured_supervised_dataset,
    fit_scalers,
    project_to_mass_balance,
    transform_dataset_split,
)
from src.utils.simulation import load_model_params
from src.utils.test import evaluate_prediction_bundle
from src.utils.train import (
    get_training_device,
    load_model_bundle,
    persist_training_artifacts,
    resolve_model_hyperparameters,
    resolve_torch_adam_options,
    resolve_torch_runtime_options,
    serialize_report_frames,
    transform_feature_frame,
)
from src.utils.optuna import create_progress_bar


MODEL_NAME = "cobre"


def load_cobre_params(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load the configured parameters for the COBRE model."""

    return load_model_params(MODEL_NAME, repo_root)


def _project_tensor_predictions(
    raw_predictions: torch.Tensor,
    constraint_reference: torch.Tensor,
    projection_operator: torch.Tensor,
) -> torch.Tensor:
    return raw_predictions - torch.matmul(raw_predictions - constraint_reference, projection_operator.T)


class DirectMLAdam(Optimizer):
    """Adam variant that avoids DirectML-unsupported lerp updates."""

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            amsgrad = bool(group["amsgrad"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("DirectMLAdam does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"])

                if weight_decay != 0.0:
                    gradient = gradient.add(parameter, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                denom_base = exp_avg_sq
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom_base = max_exp_avg_sq

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = denom_base.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                parameter.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def _build_optimizer(
    model: ProjectedCOBRE,
    model_hyperparameters: Mapping[str, float],
    *,
    device_label: str,
    adam_foreach: Any,
) -> Optimizer:
    learning_rate = float(model_hyperparameters["learning_rate"])
    weight_decay = float(model_hyperparameters["weight_decay"])

    if device_label == "directml" and adam_foreach is not True:
        return DirectMLAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        **resolve_torch_adam_options(
            device_label=device_label,
            foreach=adam_foreach,
        ),
    )


class ProjectedCOBRE(nn.Module):
    """Bilinear regressor with a differentiable measured-space projection layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        A_matrix: np.ndarray,
        *,
        bilinear_init_scale: float,
    ) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.bilinear_weight = nn.Parameter(torch.empty(output_dim, input_dim, input_dim))

        constraint_matrix = torch.as_tensor(np.asarray(A_matrix, dtype=np.float32))
        projection_operator = torch.as_tensor(
            build_projection_operator(np.asarray(A_matrix, dtype=float)).astype(np.float32)
        )
        self.register_buffer("constraint_matrix", constraint_matrix)
        self.register_buffer("projection_operator", projection_operator)
        self.reset_parameters(bilinear_init_scale)

    def reset_parameters(self, bilinear_init_scale: float) -> None:
        """Initialize linear and bilinear parameters."""

        nn.init.xavier_uniform_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)
        nn.init.normal_(self.bilinear_weight, mean=0.0, std=float(bilinear_init_scale))

    def set_output_bias(self, target_mean: np.ndarray) -> None:
        """Initialize the output bias to the mean target vector."""

        bias_tensor = torch.as_tensor(np.asarray(target_mean, dtype=np.float32), device=self.linear_layer.bias.device)
        with torch.no_grad():
            self.linear_layer.bias.copy_(bias_tensor)

    def forward(self, features: torch.Tensor, constraint_reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        linear_term = self.linear_layer(features)
        bilinear_term = torch.einsum("bi,oij,bj->bo", features, self.bilinear_weight, features)
        raw_predictions = linear_term + bilinear_term
        projected_predictions = _project_tensor_predictions(
            raw_predictions,
            constraint_reference,
            self.projection_operator,
        )
        return projected_predictions, raw_predictions


def _as_float_array(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _build_tensor_dataset(dataset_split: DatasetSplit) -> TensorDataset:
    features = torch.as_tensor(_as_float_array(dataset_split.features))
    targets = torch.as_tensor(_as_float_array(dataset_split.targets))
    constraint_reference = torch.as_tensor(_as_float_array(dataset_split.constraint_reference))
    return TensorDataset(features, targets, constraint_reference)


def _evaluate_model_predictions(
    model: ProjectedCOBRE,
    dataset_split: DatasetSplit,
    *,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.as_tensor(_as_float_array(dataset_split.features), device=device)
        constraint_tensor = torch.as_tensor(_as_float_array(dataset_split.constraint_reference), device=device)
        _, raw_predictions = model(feature_tensor, constraint_tensor)

    raw_prediction_array = raw_predictions.detach().cpu().numpy().astype(float)
    projected_prediction_array = project_to_mass_balance(
        raw_prediction_array,
        dataset_split.constraint_reference.to_numpy(dtype=float),
        model.constraint_matrix.detach().cpu().numpy().astype(float),
    )

    return (
        projected_prediction_array,
        raw_prediction_array,
    )


def _projected_mse(model: ProjectedCOBRE, dataset_split: DatasetSplit, *, device: Any) -> float:
    projected_predictions, _ = _evaluate_model_predictions(model, dataset_split, device=device)
    targets = np.asarray(dataset_split.targets, dtype=float)
    return float(np.mean((projected_predictions - targets) ** 2))


def _l1_penalty(model: ProjectedCOBRE) -> torch.Tensor:
    return model.linear_layer.weight.abs().sum() + model.bilinear_weight.abs().sum()


def _make_loader(
    dataset_split: DatasetSplit,
    *,
    batch_size: int,
    random_seed: int,
) -> DataLoader:
    tensor_dataset = _build_tensor_dataset(dataset_split)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    effective_batch_size = min(max(int(batch_size), 1), max(len(dataset_split.features), 1))
    return DataLoader(
        tensor_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=generator,
    )


def _resolve_training_options(
    training_options: Mapping[str, Any] | None,
    *,
    default_random_seed: int,
    default_log_interval: int,
    default_batch_size: int,
) -> dict[str, Any]:
    options = dict(training_options or {})
    options.setdefault("epochs", 100)
    options.setdefault("batch_size", default_batch_size)
    options.setdefault("random_seed", default_random_seed)
    options.setdefault("log_interval", default_log_interval)
    options.setdefault("prefer_directml", True)
    options.setdefault("adam_foreach", None)
    options.setdefault("show_progress", True)
    options.setdefault("progress_description", "Training COBRE")
    options.setdefault("objective_name", "projected_mse_plus_l1")
    options.setdefault("validation_dataset", None)
    return options


def train_cobre_model(
    training_dataset: Mapping[str, pd.DataFrame | np.ndarray],
    model_hyperparameters: Mapping[str, float],
    *,
    A_matrix: np.ndarray,
    training_options: Mapping[str, Any] | None = None,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    """Train COBRE on a prepared dataset and return the model and aligned predictions."""

    feature_frame = pd.DataFrame(training_dataset["features"])
    target_frame = pd.DataFrame(training_dataset["targets"])
    constraint_frame = pd.DataFrame(training_dataset["constraint_reference"])
    dataset_split = DatasetSplit(
        features=feature_frame,
        targets=target_frame,
        constraint_reference=constraint_frame,
    )

    default_random_seed = int(training_options.get("random_seed", 42)) if training_options else 42
    default_log_interval = int(training_options.get("log_interval", 25)) if training_options else 25
    default_batch_size = int(training_options.get("batch_size", 64)) if training_options else 64
    options = _resolve_training_options(
        training_options,
        default_random_seed=default_random_seed,
        default_log_interval=default_log_interval,
        default_batch_size=default_batch_size,
    )
    validation_dataset_raw = options["validation_dataset"]
    validation_dataset = None
    if validation_dataset_raw is not None:
        validation_dataset = DatasetSplit(
            features=pd.DataFrame(validation_dataset_raw["features"]),
            targets=pd.DataFrame(validation_dataset_raw["targets"]),
            constraint_reference=pd.DataFrame(validation_dataset_raw["constraint_reference"]),
        )

    device, device_label = get_training_device(prefer_directml=bool(options["prefer_directml"]))
    model = ProjectedCOBRE(
        input_dim=feature_frame.shape[1],
        output_dim=target_frame.shape[1],
        A_matrix=A_matrix,
        bilinear_init_scale=float(model_hyperparameters["bilinear_init_scale"]),
    ).to(device)
    model.set_output_bias(target_frame.mean(axis=0).to_numpy(dtype=float))

    optimizer = _build_optimizer(
        model,
        model_hyperparameters,
        device_label=device_label,
        adam_foreach=options["adam_foreach"],
    )
    criterion = nn.MSELoss()
    loader = _make_loader(
        dataset_split,
        batch_size=int(options["batch_size"]),
        random_seed=int(options["random_seed"]),
    )

    history: list[dict[str, float | int]] = []
    best_validation_loss = np.inf
    best_state_dict = copy.deepcopy(model.state_dict())

    progress_bar = create_progress_bar(
        total=int(options["epochs"]),
        desc=str(options["progress_description"]),
        enabled=bool(options["show_progress"]),
        unit="epoch",
    )
    progress_bar.set_postfix(objective=str(options["objective_name"]))
    try:
        for epoch in range(int(options["epochs"])):
            model.train()
            running_loss = 0.0
            sample_count = 0
            for batch_features, batch_targets, batch_constraints in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                batch_constraints = batch_constraints.to(device)

                optimizer.zero_grad(set_to_none=True)
                projected_predictions, _ = model(batch_features, batch_constraints)
                penalty = float(model_hyperparameters["lambda_l1"]) * _l1_penalty(model)
                loss = criterion(projected_predictions, batch_targets) + penalty
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=float(model_hyperparameters["clip_max_norm"]))
                optimizer.step()

                batch_size = int(batch_features.shape[0])
                running_loss += float(loss.detach().cpu()) * batch_size
                sample_count += batch_size

            train_loss = running_loss / max(sample_count, 1)
            validation_loss = None
            should_record = ((epoch + 1) % int(options["log_interval"]) == 0) or epoch == 0 or epoch == int(options["epochs"]) - 1
            if validation_dataset is not None:
                validation_loss = _projected_mse(model, validation_dataset, device=device)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

                if trial is not None:
                    trial.report(validation_loss, step=epoch + 1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            else:
                if train_loss < best_validation_loss:
                    best_validation_loss = train_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

            if should_record:
                history_row: dict[str, float | int] = {"epoch": epoch + 1, "train_loss": float(train_loss)}
                if validation_loss is not None:
                    history_row["validation_loss"] = float(validation_loss)
                history.append(history_row)

            progress_kwargs: dict[str, str] = {
                "objective": str(options["objective_name"]),
                "train_loss": f"{train_loss:.6g}",
            }
            if validation_loss is not None:
                progress_kwargs["validation_loss"] = f"{validation_loss:.6g}"
            progress_bar.set_postfix(**progress_kwargs)
            progress_bar.update(1)
    finally:
        progress_bar.close()

    model.load_state_dict(best_state_dict)
    training_projected, training_raw = _evaluate_model_predictions(model, dataset_split, device=device)

    return {
        "model": model,
        "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "training_predictions": training_projected,
        "training_raw_predictions": training_raw,
        "best_validation_loss": float(best_validation_loss),
        "history": history,
        "device": device_label,
    }


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
        "state_dict": dict(training_result["model_state_dict"]),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "constraint_columns": constraint_columns,
        "A_matrix": np.asarray(A_matrix, dtype=float),
        "scaling_bundle": scaling_bundle,
        "model_hyperparameters": dict(model_hyperparameters),
        "training_options": dict(training_options),
    }


def _load_model_from_bundle(model_bundle: Mapping[str, Any], *, device: Any) -> ProjectedCOBRE:
    model = ProjectedCOBRE(
        input_dim=len(model_bundle["feature_columns"]),
        output_dim=len(model_bundle["target_columns"]),
        A_matrix=np.asarray(model_bundle["A_matrix"], dtype=float),
        bilinear_init_scale=float(model_bundle["model_hyperparameters"]["bilinear_init_scale"]),
    ).to(device)
    model.load_state_dict(model_bundle["state_dict"])
    model.eval()
    return model


def predict_cobre_model(
    test_dataset: pd.DataFrame | Mapping[str, pd.DataFrame | np.ndarray],
    model_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate raw and projected COBRE predictions aligned to the supplied test dataset."""

    model_bundle = load_model_bundle(model_path)
    scaling_bundle: ScalingBundle = model_bundle["scaling_bundle"]
    runtime_options = dict(model_bundle.get("training_options", {}))

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
        constraint_reference = pd.DataFrame(test_dataset["constraint_reference"], columns=model_bundle["constraint_columns"])

    constraint_columns = list(model_bundle["constraint_columns"])
    transformed_features = transform_feature_frame(feature_frame, scaling_bundle)
    device, _ = get_training_device(prefer_directml=bool(runtime_options.get("prefer_directml", True)))
    model = _load_model_from_bundle(model_bundle, device=device)
    prediction_split = DatasetSplit(
        features=transformed_features,
        targets=pd.DataFrame(np.zeros((len(feature_frame), len(model_bundle["target_columns"]))), columns=model_bundle["target_columns"]),
        constraint_reference=constraint_reference.loc[:, constraint_columns],
    )
    projected_predictions, raw_predictions = _evaluate_model_predictions(model, prediction_split, device=device)

    return {
        "raw_predictions": pd.DataFrame(raw_predictions, index=feature_frame.index, columns=model_bundle["target_columns"]),
        "projected_predictions": pd.DataFrame(
            projected_predictions,
            index=feature_frame.index,
            columns=model_bundle["target_columns"],
        ),
        "constraint_reference": constraint_reference.loc[:, constraint_columns].copy(),
    }


def run_cobre_pipeline(
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
    """Train, evaluate, and optionally persist a COBRE model bundle."""

    params = dict(model_params) if model_params is not None else load_cobre_params(repo_root)
    split_params = params["hyperparameters"]
    runtime_options = resolve_torch_runtime_options(params)
    selected_hyperparameters = resolve_model_hyperparameters(params, model_hyperparameters)
    selected_batch_size = int(selected_hyperparameters.get("batch_size", split_params["batch_size"]))
    scaling_bundle = fit_scalers(
        training_split,
        scale_features=bool(split_params["scale_features"]),
        scale_targets=bool(split_params["scale_targets"]),
    )
    scaled_training_split = transform_dataset_split(training_split, scaling_bundle)
    scaled_test_split = transform_dataset_split(test_split, scaling_bundle)

    final_training_options = {
        "epochs": int(split_params["training_epochs"]),
        "batch_size": selected_batch_size,
        "random_seed": int(split_params["random_seed"]),
        "log_interval": int(split_params["log_interval"]),
        "prefer_directml": runtime_options["prefer_directml"],
        "adam_foreach": runtime_options["adam_foreach"],
        "show_progress": show_progress,
        "progress_description": "Training COBRE",
        "objective_name": "projected_mse_plus_l1",
    }
    final_training_result = train_cobre_model(
        {
            "features": scaled_training_split.features,
            "targets": scaled_training_split.targets,
            "constraint_reference": scaled_training_split.constraint_reference,
        },
        selected_hyperparameters,
        A_matrix=np.asarray(A_matrix, dtype=float),
        training_options=final_training_options,
    )

    final_model: ProjectedCOBRE = final_training_result["model"]
    device, _ = get_training_device(prefer_directml=bool(runtime_options["prefer_directml"]))
    train_projected, train_raw = _evaluate_model_predictions(final_model, scaled_training_split, device=device)
    test_projected, test_raw = _evaluate_model_predictions(final_model, scaled_test_split, device=device)

    train_report = evaluate_prediction_bundle(
        training_split.targets.to_numpy(dtype=float),
        train_raw,
        train_projected,
        training_split.constraint_reference.to_numpy(dtype=float),
        np.asarray(A_matrix, dtype=float),
        training_split.targets.columns,
        index=training_split.targets.index,
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

    model_bundle = _build_model_bundle(
        final_training_result,
        scaling_bundle,
        feature_columns=list(training_split.features.columns),
        target_columns=list(training_split.targets.columns),
        constraint_columns=list(training_split.constraint_reference.columns),
        A_matrix=np.asarray(A_matrix, dtype=float),
        model_hyperparameters=selected_hyperparameters,
        training_options=final_training_options,
    )

    dataset_splits = TrainTestDatasetSplits(train=training_split, test=test_split)

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
                "test": int(len(dataset_splits.test.features)),
            },
        }
        optuna_payload = dict(optuna_summary) if optuna_summary is not None and bool(artifact_options.get("persist_optuna", True)) else None
        metrics_summary = metrics_payload if bool(artifact_options.get("persist_metrics", True)) else None
        artifact_paths = persist_training_artifacts(
            MODEL_NAME,
            model_bundle,
            metrics_summary=metrics_summary,
            optuna_summary=optuna_payload,
            repo_root=repo_root,
            timestamp=timestamp,
        )

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
    "ProjectedCOBRE",
    "build_projection_operator",
    "load_cobre_params",
    "predict_cobre_model",
    "project_to_mass_balance",
    "run_cobre_pipeline",
    "train_cobre_model",
]