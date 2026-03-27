
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
- an optuna module that executes hyperparameter optimization for the model
- a train module
- a predict module

The optuna module must output the optimal hyperparameters in a dictionary, along with any additional relevant Optuna outputs.

Across machine learning model files, the train and predict modules must share the same minimum interface.

Minimum train-module requirements:

- inputs: training dataset and model hyperparameters as a dictionary
- outputs: trained model and training predictions that can be mapped to the training dataset

Minimum predict-module requirements:

- inputs: test dataset and the path to the trained model .pkl file
- outputs: test predictions that can be mapped to the test dataset

Models may define additional inputs or outputs when required by the specific algorithm.

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
- Case: PIBRe end-to-end smoke validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c "import copy; import numpy as np; from scipy.linalg import null_space; from src.models.simulation import run_asm1_simulation; from src.models.ml import run_pibre_pipeline; from src.models.ml.pibre import load_pibre_params; simulation_result = run_asm1_simulation(save_artifacts=False, n_samples=24, random_seed=21); a_matrix = null_space(simulation_result['petersen_matrix'] @ simulation_result['composition_matrix'].T).T; a_matrix = np.round(a_matrix, 5); a_matrix[np.abs(a_matrix) < 1e-10] = 0.0; [a_matrix.__setitem__((row_index, slice(None)), a_matrix[row_index, :] / a_matrix[row_index, a_matrix[row_index, :] != 0][0]) for row_index in range(a_matrix.shape[0]) if np.any(a_matrix[row_index, :] != 0)]; params = copy.deepcopy(load_pibre_params()); params['hyperparameters']['default_tuning_profile'] = 'fast'; params['hyperparameters']['log_interval'] = 5; params['tuning_profiles']['fast'] = {'n_trials': 2, 'tuning_epochs': 8, 'final_epochs': 12, 'timeout_seconds': None}; result = run_pibre_pipeline(simulation_result['dataset'], simulation_result['metadata'], simulation_result['composition_matrix'], a_matrix, model_params=params, tuning_profile='fast', persist_artifacts=True, timestamp='20260327_040500'); print(result['artifact_paths']['model_bundle']); print(result['artifact_paths']['metrics']); print(result['artifact_paths']['optuna']); print(result['test_report']['aggregate_metrics'].to_string(index=False))". Result: effective for confirming the notebook-style ASM1-to-PIBRe flow, persisted results paths, and small projected constraint residuals; DirectML emitted an Adam CPU-fallback warning but the workflow still completed successfully.
- Case: dependency refresh for additional classical ML regressors on Windows. Strategy: after updating pyproject.toml with LightGBM and CatBoost, run uv sync from the repository root. Result: effective for updating uv.lock and installing the new model dependencies into the project environment without leaving the repository package-management workflow.
- Case: measured-space classical regressor validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation tests.ml.test_pibre tests.ml.test_tabular_regressors, and when the command returns no transcript in the terminal tool, confirm success with Write-Output $LASTEXITCODE. Result: effective for validating the repository contract, preserving the ASM1 and PIBRe paths, and exercising XGBoost, LightGBM, CatBoost, AdaBoost, Random Forest, and SVR end-to-end under the shared measured-space helper pipeline.

### Ineffective

- Case: first-pass measurable-schema ASM1 behavior validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation immediately after refactor without oxygen-limited scaling. Result: ineffective because the aeration response assertion failed due zero dissolved oxygen in both low and high aeration scenarios.
- Case: first-pass mechanistic steady-state ASM1 validation on Windows. Strategy: run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.simulation.test_asm1_simulation immediately after the mechanistic solver refactor. Result: ineffective because config JSON errors and poor steady-state initialization caused repeated test failures until the parameter file, multistart initialization, and operating-domain settings were corrected.
- Case: first-pass reduced ASM1 measured-output schema validation on Windows. Strategy: run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm1_simulation immediately after trimming the measured outputs. Result: ineffective because one test still asserted the old dataset width and composition-matrix shape until the stale expectation in test_run_asm1_simulation_without_artifacts_keeps_paths_empty was updated.

