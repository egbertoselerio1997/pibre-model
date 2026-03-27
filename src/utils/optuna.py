"""Reusable Optuna study and parameter-suggestion helpers."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import TrialState


def build_pruner(pruner_config: Mapping[str, Any] | None) -> optuna.pruners.BasePruner:
	"""Build a pruner from configuration."""

	if not pruner_config:
		return NopPruner()

	pruner_type = str(pruner_config.get("type", "none")).lower()
	if pruner_type == "median":
		return MedianPruner(
			n_startup_trials=int(pruner_config.get("n_startup_trials", 5)),
			n_warmup_steps=int(pruner_config.get("n_warmup_steps", 20)),
			interval_steps=int(pruner_config.get("interval_steps", 1)),
		)

	return NopPruner()


def create_optuna_study(
	model_name: str,
	*,
	direction: str = "minimize",
	seed: int = 42,
	pruner_config: Mapping[str, Any] | None = None,
) -> Study:
	"""Create a seeded Optuna study for one model family."""

	return optuna.create_study(
		study_name=f"{model_name}_study",
		direction=direction,
		sampler=TPESampler(seed=seed),
		pruner=build_pruner(pruner_config),
	)


def suggest_parameters(trial: optuna.Trial, search_space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
	"""Suggest parameter values from a declarative search-space definition."""

	suggested: dict[str, Any] = {}
	for parameter_name, spec in search_space.items():
		parameter_type = str(spec["type"]).lower()
		if parameter_type == "float":
			suggested[parameter_name] = trial.suggest_float(
				parameter_name,
				float(spec["low"]),
				float(spec["high"]),
				log=bool(spec.get("log", False)),
			)
		elif parameter_type == "int":
			suggested[parameter_name] = trial.suggest_int(
				parameter_name,
				int(spec["low"]),
				int(spec["high"]),
				log=bool(spec.get("log", False)),
			)
		elif parameter_type == "categorical":
			suggested[parameter_name] = trial.suggest_categorical(
				parameter_name,
				list(spec["choices"]),
			)
		else:
			raise ValueError(f"Unsupported Optuna parameter type: {parameter_type}")

	return suggested


def optimize_study(
	study: Study,
	objective: Callable[[optuna.Trial], float],
	*,
	n_trials: int,
	timeout: int | None = None,
	show_progress_bar: bool = False,
) -> Study:
	"""Run an Optuna study and return the completed study."""

	study.optimize(
		objective,
		n_trials=n_trials,
		timeout=timeout,
		show_progress_bar=show_progress_bar,
	)
	return study


def make_study_summary(study: Study) -> dict[str, Any]:
	"""Convert an Optuna study into a JSON-serializable summary."""

	trial_state_counts: dict[str, int] = {
		trial_state.name.lower(): 0 for trial_state in TrialState
	}
	for trial in study.trials:
		trial_state_counts[trial.state.name.lower()] += 1

	best_trial = study.best_trial
	return {
		"best_value": float(best_trial.value),
		"best_params": dict(best_trial.params),
		"best_trial_number": int(best_trial.number),
		"n_trials": len(study.trials),
		"trial_state_counts": trial_state_counts,
	}


__all__ = [
	"build_pruner",
	"create_optuna_study",
	"make_study_summary",
	"optimize_study",
	"suggest_parameters",
]