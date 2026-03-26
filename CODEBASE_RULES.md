
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

## 8. Testing Requirement

Minimal-scenario tests must always be carried out whenever code is developed.

Test scripts must be saved under the tests folder.

The tests folder must use subfolders that correspond to the code file or feature area being tested. Multiple test scripts may be required for a feature.

## 9. Dependency Declaration

Whenever new packages are needed, agents must first check pyproject.toml.

If a required package is missing, pyproject.toml must be updated and uv must be used to install or lock the dependency.

## 10. Source Layout

The src folder contains:

- utils
- models

The src/models folder contains:

- simulation
- ml

The simulation folder contains Python files used to generate datasets for machine learning models.

The ml folder contains Python files for machine learning models, and each filename should match the model name.

## 11. Utility Modules

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

## 12. Simulation Dataset Output Contract

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

## 13. Machine Learning Module Contract

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

## 14. Standard Metrics

All machine learning models must calculate the following metrics:

- R2
- MSE
- RMSE
- MAE
- MAPE

## 15. Orchestration Entry Point

The orchestration of simulation-driven dataset generation and the machine learning pipeline is done through the root notebook main.ipynb.

Functions, helpers, modules, and related code in src are imported and used there.

## CLI Command Log

Document command strategy outcomes here.

### Effective

- Case: repository inspection on Windows. Strategy: use PowerShell-oriented workspace inspection and file reads before editing. Result: effective for confirming the repo was still a blank scaffold and identifying empty root files.
- Case: bootstrap scaffold validation on Windows. Strategy: use PowerShell Get-ChildItem -Name to confirm the root layout, then run uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract. Result: effective for verifying the required folders, config contracts, and utility-module scaffold with minimal tests.

### Ineffective

- Case: none recorded yet.

