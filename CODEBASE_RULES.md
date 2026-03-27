
# CODEBASE RULES

This document defines the mandatory operating rules for coding agents working in this repository. These rules apply to all implementation, testing, orchestration, and command-line workflows.

## 1. Repository Architecture

The repository uses the following root-level folders:

- config
- src
- results
- data
- tests
- docs

Agents must not create new code or files outside these folders, except for updates to existing root-level repository files that already exist, such as this document, the project metadata file, and the root notebook.

Agents must not write code that generates files outside these folders.

## 2. Path Management

All file and directory paths used by code must be declared in config/paths.json.

Paths must never be hardcoded in Python source files. Code must load the path configuration at runtime whenever a path reference is needed.

## 3. Parameter Management

All model parameters and hyperparameters must be declared in config/params.json.

The required structure is model_name.{...} where each model owns its parameter namespace.

Shared machine-learning orchestration parameters that are consumed by main.ipynb must be declared in a dedicated top-level namespace in config/params.json named ml_orchestration.{...}.

Parameters that control dataset splitting, Optuna dataset subsampling, global tuning_epochs, and global n_trials must be declared in ml_orchestration and must not be duplicated as per-model tuning-profile structures.

Parameter values must never be hardcoded in Python source files. Code must load the parameter configuration at runtime whenever parameters are needed.

## 4. Package Management

The preferred Python package manager for this repository is uv.

Agents must use uv whenever practical and only use another package-management approach when absolutely necessary.

## 5. Command-Line Preference

PowerShell commands are preferred whenever running commands in the CLI.

## 6. CLI Strategy Logging

Whenever a CLI command fails, produces errors, or gives ineffective results, that outcome must be documented in this document.

Whenever a CLI command works well for a specific case, that outcome must also be documented in this document.

Each note must briefly describe the case, the command strategy used, and whether it was effective or ineffective. The purpose is to improve future command selection.

## 7. Placement Uncertainty

If an agent is unsure where a new file belongs within the existing folders, the agent must ask the user before creating the file.

## 8. Revision Impact Analysis

When asked to revise a file, an agent must begin planning by identifying all upstream, downstream, and dependency files within the codebase that are affected by the change.

Any affected files that require corresponding updates to preserve correctness, consistency, documentation accuracy, or test validity must also be revised accordingly.

## 9. Testing Requirement

Minimal-scenario tests must always be carried out whenever code is developed.

Test scripts must be saved under the tests folder.

The tests folder must use subfolders that correspond to the code file or feature area being tested. Multiple test scripts may be required for a feature.

## 10. Dependency Declaration

Whenever new packages are needed, agents must first check pyproject.toml.

If a required package is missing, pyproject.toml must be updated and uv must be used to install or lock the dependency.

## 11. Source Layout

The src folder contains:

- utils
- models

The src/models folder contains:

- simulation
- ml

The simulation folder contains Python files used to generate datasets for machine learning models.

The ml folder contains Python files for machine learning models, and each filename should match the model name.

## 12. Utility Modules

The src/utils folder must contain reusable helper modules with the following intended responsibilities:

- optuna.py: hyperparameter optimization helpers
- process.py: preprocessing, processing, splitting, scaling, and related data helpers
- train.py: machine learning training helpers
- test.py: machine learning evaluation helpers
- simulation.py: simulation helpers
- metrics.py: metric calculation helpers
- plot.py: plotting helpers
- analysis.py: analysis and comparison helpers not covered elsewhere
- io.py: loading, storing, updating, and related input-output helpers

These helpers must be designed for maximum reuse, especially optuna.py, train.py, test.py, metrics.py, plot.py, and analysis.py.

## 13. Simulation Dataset Output Contract

All simulation-generated datasets for machine learning training must be saved under:

data/{simulation_name}/data_{date_time}.csv

Each generated dataset must also include:

data/{simulation_name}/metadata_{date_time}.json

The metadata JSON must define:

- dependent or target column headers
- independent variable column headers
- identifier column headers
- ignored column headers
- the file path to the corresponding CSV dataset

Machine learning training and testing pipelines must use the metadata JSON as the dataset-loading contract.

## 14. Machine Learning Module Contract

Each machine learning model file must contain:

- a model module that defines the model
- a train module
- a predict module

Optuna hyperparameter optimization must not be executed inside files under src/models/ml.

Reusable Optuna helpers may exist under src/utils, but they may only be executed from the orchestration notebook main.ipynb.

Dataset splitting for train, test, and Optuna-only subsets must not be executed inside files under src/models/ml.

Dataset splitting must be orchestrated only from main.ipynb, and notebook-prepared splits must be passed into the machine learning model helpers.

External Optuna helpers must output the optimal hyperparameters in a dictionary, along with any additional relevant Optuna outputs.

Across machine learning model files, the train and predict modules must share the same minimum interface.

Minimum train-module requirements:

- inputs: training dataset and model hyperparameters as a dictionary
- outputs: trained model and training predictions that can be mapped to the training dataset

Thin run modules may additionally accept notebook-prepared training and test splits plus explicit hyperparameters, but they must not create those splits internally.

Minimum predict-module requirements:

- inputs: test dataset and the path to the trained model .pkl file
- outputs: test predictions that can be mapped to the test dataset

Models may define additional inputs or outputs when required by the specific algorithm.

### 14.1 Training Progress Visibility

All machine learning model training paths must display TQDM progress bars while training is running.

The progress display must include the current optimization objective name and, whenever a live objective value is naturally available, the latest objective value.

Progress display must be enabled by default.

An explicit opt-out may be provided only for tests or other non-interactive automation contexts.

## 15. Standard Metrics

All machine learning models must calculate the following metrics:

- R2
- MSE
- RMSE
- MAE
- MAPE

## 16. Orchestration Entry Point

The orchestration of simulation-driven dataset generation and the machine learning pipeline is done through the root notebook main.ipynb.

Functions, helpers, modules, and related code in src are imported and used there.

The notebook is the only allowed execution point for machine-learning dataset splitting and Optuna hyperparameter optimization.

Any Optuna subset used for hyperparameter optimization must be drawn only from the notebook-managed training pool and must exclude the final holdout test split.

## 17. Model Documentation Requirement

Both machine learning models and simulation models must have corresponding Markdown documentation files under docs.

Machine learning model documentation must live in docs/ml.

Simulation model documentation must live in docs/simulation.

Documentation filenames should correspond to model filenames whenever practical.

Examples:

- src/models/ml/model_name.py should be documented by docs/ml/model_name.md
- src/models/simulation/simulation_name.py should be documented by docs/simulation/simulation_name.md

Each documentation file must be written as if the intended reader is an academic who is not a programmer and does not have access to the codebase.

The writing must be comprehensive, technically rigorous, and understandable without referring to source code.

### 17.1 Machine Learning Documentation

Each machine learning model document must thoroughly explain:

- the model background
- the rigorous mathematical definition of the model
- the exact implementation used in this repository
- the adopted model structure or architecture when relevant, especially for neural networks and deep learning models
- the standard name of the adopted architecture when such a standard name exists
- citations for sources used for the adopted architecture
- diagrams or other visualizations whenever they materially improve explanation of process flow, architecture, or another concept

Mermaid diagrams may be used whenever they help communicate structure, flow, or architecture.

Each machine learning documentation file must follow the same standard structure unless a specific model requires an additional section:

1. Title and model summary
2. Background and use case
3. Mathematical definition
4. Inputs, outputs, and assumptions
5. Implementation used in this repository
6. Architecture details and adopted standard architecture name, when relevant
7. Training or optimization notes, when relevant
8. Prediction workflow
9. Limitations and expected failure modes
10. References

The documentation must explain the exact implementation adopted in the repository, not only the generic textbook model.

If a visualization is omitted, the document should still explain the process or architecture clearly in prose.

### 17.2 Simulation Documentation

Each simulation model document must thoroughly explain:

- the simulation model background
- its rigorous mathematical definition
- its exact implementation used in this repository
- the structure, architecture, orchestration, or adopted approach when relevant
- the standard name of the adopted approach when such a standard name exists
- citations for sources used for the adopted approach
- diagrams or other visualizations whenever they materially improve explanation of process flow, architecture, orchestration, or another concept

Mermaid diagrams may be used whenever they help communicate process flow, architecture, orchestration, or another relevant concept.

Each simulation documentation file must follow the same standard structure unless a specific simulation requires an additional section:

1. Title and simulation summary
2. Background and system or process context
3. Mathematical definition and governing relations
4. Inputs, outputs, state variables, and assumptions
5. Implementation used in this repository
6. Architecture, orchestration, or adopted approach details and standard name, when relevant
7. Dataset-generation or execution workflow
8. Limitations and expected failure modes
9. References

The documentation must explain the exact simulation implementation adopted in the repository, not only the generic conceptual or mathematical model.

If a visualization is omitted, the document should still explain the flow, orchestration, or structure clearly in prose.

## CLI Command Log

Document command strategy outcomes here.

### Effective

- Case: repository inspection and bootstrap validation on Windows. Strategy: use PowerShell-oriented workspace inspection, including Get-ChildItem -Name and file reads before editing, then run the standard bootstrap repository test command. Result: effective for confirming the root layout, identifying scaffold gaps, and validating the repository contract.
- Case: policy and documentation contract validation on Windows. Strategy: after updates to CODEBASE_RULES.md, docs/ml, or docs/simulation, rerun the standard bootstrap repository test command. Result: effective for confirming that rule and documentation revisions remained consistent with the repository scaffold.
- Case: simulation workflow validation on Windows. Strategy: run the standard repository-and-simulation unittest command after simulation integration, renames, schema refactors, solver changes, or matrix exposure updates. Result: effective for validating repository compatibility, simulation metadata and schema updates, and behavior checks across ASM1 changes.
- Case: ASM1 public API smoke validation on Windows. Strategy: run a small in-memory Python smoke command after tests pass. Result: effective for confirming that the public simulation entry point still returns dataset and metadata outputs without writing artifacts.
- Case: reduced ASM1 measured-output schema validation on Windows. Strategy: after trimming the exported composite measures, run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation, then run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c "from src.models.simulation.asm1_simulation import run_asm1_simulation; result = run_asm1_simulation(save_artifacts=False, n_samples=4, random_seed=3); print(result['dataset'].shape); print(result['metadata']['schema_version']); print(result['metadata']['measured_output_columns'])". Result: effective for confirming the updated 8-output contract, new matrix shape, and public API metadata without writing artifacts.
- Case: PIBRe repository validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation tests.ml.test_pibre after adding measured-space ML utilities, the PIBRe model module, and the notebook orchestration cell. Result: effective for validating the repository contract, preserving ASM1 behavior, and checking minimal end-to-end PIBRe training and prediction behavior.
- Case: notebook-managed PIBRe and classical-regressor validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.ml.test_pibre tests.ml.test_tabular_regressors tests.ml.test_ml_orchestration, then run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c "import copy; import numpy as np; from scipy.linalg import null_space; from src.models.simulation import run_asm1_simulation; from src.models.ml import load_adaboost_regressor_params, load_pibre_params, run_adaboost_regressor_pipeline, run_pibre_pipeline; from src.models.ml.adaboost_regressor import build_adaboost_regressor_model; from src.utils.process import build_measured_supervised_dataset, make_train_test_split, sample_dataset_fraction; from src.utils.simulation import load_ml_orchestration_params; from src.utils.train import tune_pibre_hyperparameters, tune_tabular_regressor_hyperparameters; simulation_result = run_asm1_simulation(save_artifacts=False, n_samples=24, random_seed=29); a_matrix = null_space(simulation_result['petersen_matrix'] @ simulation_result['composition_matrix'].T).T; a_matrix = np.round(a_matrix, 5); a_matrix[np.abs(a_matrix) < 1e-10] = 0.0; [a_matrix.__setitem__((row_index, slice(None)), a_matrix[row_index, :] / a_matrix[row_index, a_matrix[row_index, :] != 0][0]) for row_index in range(a_matrix.shape[0]) if np.any(a_matrix[row_index, :] != 0)]; orchestration = load_ml_orchestration_params()['hyperparameters']; measured_dataset = build_measured_supervised_dataset(simulation_result['dataset'], simulation_result['metadata'], simulation_result['composition_matrix']); main_splits = make_train_test_split(measured_dataset, test_fraction=float(orchestration['test_fraction']), random_seed=int(orchestration['random_seed'])); tuning_dataset = sample_dataset_fraction(main_splits.train, fraction=0.5, random_seed=int(orchestration['random_seed'])); tuning_splits = make_train_test_split(tuning_dataset, test_fraction=float(orchestration['optuna_test_fraction']), random_seed=int(orchestration['random_seed'])); pibre_params = copy.deepcopy(load_pibre_params()); pibre_params['hyperparameters']['training_epochs'] = 12; pibre_params['hyperparameters']['batch_size'] = 8; pibre_params['hyperparameters']['log_interval'] = 2; pibre_best, pibre_optuna = tune_pibre_hyperparameters(tuning_splits.train, tuning_splits.test, A_matrix=a_matrix, model_params=pibre_params, tuning_epochs=3, n_trials=1); pibre_result = run_pibre_pipeline(main_splits.train, main_splits.test, a_matrix, model_params=pibre_params, model_hyperparameters=pibre_best, optuna_summary=pibre_optuna, persist_artifacts=True, timestamp='20260327_153500'); adaboost_params = copy.deepcopy(load_adaboost_regressor_params()); adaboost_params['training_defaults']['n_estimators'] = 12; adaboost_params['search_space']['n_estimators'] = {'type': 'int', 'low': 8, 'high': 12, 'log': False}; adaboost_best, adaboost_optuna = tune_tabular_regressor_hyperparameters('adaboost_regressor', build_adaboost_regressor_model, tuning_splits.train, tuning_splits.test, A_matrix=a_matrix, model_params=adaboost_params, n_trials=1); adaboost_result = run_adaboost_regressor_pipeline(main_splits.train, main_splits.test, a_matrix, model_params=adaboost_params, model_hyperparameters=adaboost_best, optuna_summary=adaboost_optuna, persist_artifacts=True, timestamp='20260327_153501'); print(pibre_result['artifact_paths']); print(adaboost_result['artifact_paths']); print(pibre_result['test_report']['aggregate_metrics'].to_string(index=False)); print(adaboost_result['test_report']['aggregate_metrics'].to_string(index=False))". Result: effective for confirming the new notebook-owned split flow, external Optuna execution, thin model runners, persisted artifact paths, and low projected constraint residuals; DirectML emitted an Adam CPU-fallback warning but the workflow still completed successfully.
- Case: dependency refresh for additional classical ML regressors on Windows. Strategy: after updating pyproject.toml with LightGBM and CatBoost, run uv sync from the repository root. Result: effective for updating uv.lock and installing the new model dependencies into the project environment without leaving the repository package-management workflow.
- Case: measured-space classical regressor validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation tests.ml.test_pibre tests.ml.test_tabular_regressors, and when the command returns no transcript in the terminal tool, confirm success with Write-Output $LASTEXITCODE. Result: effective for validating the repository contract, preserving the ASM1 and PIBRe paths, and exercising XGBoost, LightGBM, CatBoost, AdaBoost, Random Forest, and SVR end-to-end under the shared measured-space helper pipeline.
- Case: PIBRe DirectML optimizer regression validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.ml.test_pibre after replacing the DirectML optimizer path, then run a PowerShell here-string Python smoke command with single-quoted Python literals to train a tiny PIBRe model and capture warnings. Result: effective for confirming the DirectML training path executes without the previous Adam `lerp.Scalar_out` CPU-fallback warning and for checking the projected metrics on a small run.
- Case: ASM1 process-backed parallel generation validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation after switching the dataset generator to chunked ProcessPoolExecutor workers, then run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c "import time; from src.models.simulation.asm1_simulation import run_asm1_simulation; start = time.perf_counter(); result = run_asm1_simulation(save_artifacts=False, n_samples=240, random_seed=13); elapsed = time.perf_counter() - start; print({'shape': result['dataset'].shape, 'seconds': round(elapsed, 3)})". Result: effective for confirming the Windows process-worker path, the public API, and the shipped default parallel settings without writing artifacts.
- Case: TQDM dependency refresh and ML progress validation on Windows. Strategy: run uv sync after adding tqdm to pyproject.toml, then run uv run python -m unittest tests.bootstrap.test_repo_contract and PowerShell-safe direct unittest runners via uv run python -c "import sys, unittest; ..." for tests.ml.test_pibre, tests.ml.test_tabular_regressors, and tests.ml.test_ml_orchestration. Result: effective for installing tqdm into the project environment, validating the new repository rule, and confirming default-on plus opt-out ML progress behavior across PIBRe, tabular regressors, and notebook-managed tuning.
- Case: pibre_unconstrained initial implementation validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.ml.test_pibre_unconstrained tests.ml.test_ml_orchestration after adding the new unconstrained model module, notebook integration path, and Optuna tuning helper. Result: effective for confirming OLS/Ridge coverage, progress-bar defaults, persisted-bundle prediction roundtrip, and external notebook-managed tuning behavior for the new model namespace.
- Case: pibre_unconstrained regression safety check on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.ml.test_pibre tests.ml.test_tabular_regressors and verify completion with Write-Output $LASTEXITCODE when only progress dots are shown by the terminal tool. Result: effective for confirming repository-contract integrity and no regressions to existing PIBRe and measured-space tabular regressor pipelines.
- Case: pibre_unconstrained export smoke-check on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c "from src.models.ml import load_pibre_unconstrained_params, run_pibre_unconstrained_pipeline; print('ok')" when notebook imports appeared stale. Result: effective for separating notebook-kernel cache issues from source-code export correctness.
- Case: pibre_unconstrained post-fix retest on Windows. Strategy: rerun c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.ml.test_pibre_unconstrained tests.ml.test_ml_orchestration after fixing feature double-scaling in unconstrained evaluation paths. Result: effective for confirming the scaling fix and preserving Optuna orchestration behavior.
- Case: ASM1 config-driven measured-output refactor validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation immediately after moving measured-output composition definitions to config/params.json and removing hardcoded REQUIRED_* lists. Result: effective for confirming repository-contract integrity plus ASM1 matrix and metadata compatibility after the refactor.
- Case: ASM1 downstream measured-space compatibility check on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.ml.test_pibre tests.ml.test_pibre_unconstrained tests.ml.test_ml_orchestration, then run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c "from src.models.simulation.asm1_simulation import run_asm1_simulation; result = run_asm1_simulation(save_artifacts=False, n_samples=4, random_seed=17, parallel_workers=1); print(result['dataset'].shape); print(result['metadata']['measured_output_columns']); print(result['composition_matrix'].shape)". Result: effective for confirming measured-space ML consumers and the public ASM1 API remain stable under the default output contract.
- Case: ASM1 state-complete measured-output definition validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation after adding identity measured_output_definitions for every state column and adding overlap-safe dependent-column assembly. Result: effective for confirming optional state-name measured outputs can be configured while preserving unique dataset and metadata dependent columns.
- Case: ASM1 overlap-change downstream ML safety check on Windows. Strategy: rerun c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.ml.test_pibre tests.ml.test_pibre_unconstrained tests.ml.test_ml_orchestration after the overlap-safe dataset assembly update. Result: effective for confirming measured-space ML training and tuning helpers remain compatible with unchanged default ASM1 outputs.

### Ineffective

- Case: first-pass measurable-schema ASM1 behavior validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation immediately after refactor without oxygen-limited scaling. Result: ineffective because the aeration response assertion failed due zero dissolved oxygen in both low and high aeration scenarios.
- Case: first-pass mechanistic steady-state ASM1 validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.simulation.test_asm1_simulation immediately after the mechanistic solver refactor. Result: ineffective because config JSON errors and poor steady-state initialization caused repeated test failures until the parameter file, multistart initialization, and operating-domain settings were corrected.
- Case: first-pass reduced ASM1 measured-output schema validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation immediately after trimming the measured outputs. Result: ineffective because one test still asserted the old dataset width and composition-matrix shape until the stale expectation in test_run_asm1_simulation_without_artifacts_keeps_paths_empty was updated.
- Case: PIBRe DirectML smoke validation through PowerShell on Windows. Strategy: embed Python in a PowerShell here-string but keep Python output statements in double-quoted literals or nested f-strings. Result: ineffective because PowerShell stripped or distorted the Python quoting and produced `SyntaxError` failures before the validation code ran.
- Case: first-pass ASM1 process-worker probe on Windows. Strategy: call ProcessPoolExecutor.map from c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c against the keyword-only _generate_asm1_dataset_chunk worker. Result: ineffective because map passed a positional payload and raised a TypeError; executor.submit with keyword arguments worked.
- Case: small-batch ASM1 parallel timing check on Windows. Strategy: compare run_asm1_simulation(save_artifacts=False, n_samples=24, random_seed=13, parallel_workers=1, parallel_chunk_size=4) against the same call with parallel_workers=4 from c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c. Result: ineffective for judging throughput because process startup dominated the workload and the parallel run was slower than serial on that batch size.
- Case: combined ML unittest transcript capture through the terminal tool on Windows. Strategy: run uv run python -m unittest tests.ml.test_pibre tests.ml.test_tabular_regressors tests.ml.test_ml_orchestration directly from the terminal tool. Result: ineffective for logging because the tool returned only progress dots or truncated output even when the tests completed.
- Case: direct unittest runner with a single-quoted Python payload through PowerShell on Windows. Strategy: run uv run python -c 'import sys, unittest; ...'. Result: ineffective because PowerShell mangled the quoting and raised a `SyntaxError`; a double-quoted PowerShell command with single-quoted Python strings worked.

