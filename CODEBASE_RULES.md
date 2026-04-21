
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

CLI command guidance must be maintained as a compact decision aid, not as a chronological run-by-run history.

When a reusable lesson is discovered, update the CLI Command Log section with one of the following only:

- a canonical command pattern for a recurring use case
- a stable modifier or escalation rule for diagnosis
- a durable pitfall to avoid

Do not append per-invocation success or failure notes unless they introduce a new reusable lesson that is not already captured.

Each CLI Command Log entry must stay concise and include the use case plus the command pattern or rule that improves future command selection.

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

### 12.1 Common Plot Theme and Style

All generated plots, whether produced from Python modules, notebooks, tests that persist figures, or documentation-support scripts, must use one repository-wide manuscript style profile applied through shared helpers in src/utils/plot.py.

The repository standard profile is the manuscript earth-and-mineral profile aligned to docs/DCHE-D-26-00020/figure_style_guide.md and the retained figure set referenced by docs/DCHE-D-26-00020/manuscript.tex and docs/DCHE-D-26-00020/supplementary_material.tex. Style tokens must be semantic and centralized in src/utils/plot.py. Ad hoc notebook-local and script-local style stacks are not allowed for persisted manuscript assets.

Mandatory profile tokens:

- figure and canvas background: white #FFFFFF
- axes background: white #FFFFFF
- primary text: charcoal #22303C
- secondary text: muted slate #5B6770
- major grid color: #CAD2D9
- minor grid color: #E6EBEF
- qualitative cycle (ordered): #264653, #2A9D8F, #E9C46A, #F4A261, #E76F51, #6D597A, #577590, #BC4749, #8D99AE, #ADB5BD
- reserved emphasis color for ICSOR and focal manuscript series: #6D597A
- missing or masked values: neutral gray #ADB5BD
- default sequential colormap: cividis
- default diverging colormap: repository earth diverging map with explicit center at the reference value

Mandatory typography and geometry defaults:

- font family must be repository-wide sans-serif defaults from src/utils/plot.py
- default font size hierarchy must be manuscript-readable (base 10, title 13, labels 11, ticks 9)
- default line width must be 1.8 and marker edge width 0.75
- emphasized focal-series line width must be 2.7
- default uncertainty-band alpha must be 0.12 and emphasized focal-series band alpha 0.24
- grids must be low-ink dotted grids with alpha 0.35 by default
- top and right spines are disabled by default; visible spines must stay subtle

Required figure-size presets in src/utils/plot.py:

- learning curve: (10.2, 6.0)
- runtime curve: (10.2, 6.0)
- main heatmap: (8.8, 5.6)
- target atlas: (8.4, 12.8)

Theme application requirements:

- all plotting code must call shared helpers in src/utils/plot.py before drawing persisted figures
- matplotlib, seaborn, and pandas plotting paths must consume repository rcParams or shared equivalents exposed from src/utils/plot.py
- any library that cannot consume matplotlib rcParams directly must map the same semantic tokens through an adapter in src/utils/plot.py
- local style overrides are allowed only when required by accessibility or publication constraints and must preserve semantic color roles and focal-series emphasis behavior
- direct local plt.rcParams.update blocks, hard-coded color cycles, and notebook-local colormap contracts for persisted manuscript assets are not allowed

Manuscript-specific rules:

- any persisted figure for docs/DCHE-D-26-00020 must route through shared plotting and shared export helpers in src/utils/plot.py
- scalar-field plots (heatmaps, contour maps, atlases) must include a colorbar when color encodes magnitude
- legends for dense multi-model line plots should default to bottom-outside placement when this improves plot readability
- export paths for manuscript and supplementary assets must produce PDF outputs through shared helpers
- draft-only footer text may be used during illustration runs but must be removed from final submission outputs

Color-consistency and accessibility rules:

- recurring semantic pairs (train vs test, observed vs predicted, baseline vs model) must retain consistent colors across figures
- critical distinctions must not rely only on color when line style, marker shape, annotations, or labels can provide redundancy
- rainbow or jet colormaps, neon palettes, and arbitrary per-figure palette swaps are not allowed

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

Maintain this section as a compact command playbook for recurring Windows workflows. Update only when a new reusable lesson appears.

### Canonical Commands

- Repository inspection before edits: use PowerShell workspace inspection (`Get-ChildItem -Name`) plus targeted file reads.
- Bulk in-repo hard-cutover renames for modules, docs, and tests: use PowerShell `Move-Item` so the filesystem rename is explicit before patching content.
- Repository contract validation (after updates to CODEBASE_RULES.md, docs/ml, docs/simulation, scaffold, or path/parameter contracts): `uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract`.
- Simulation contract validation (after ASM2D-TSN integration, schema refactor, solver change, matrix exposure change, or measured-output definition change): `uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm2d_tsn_simulation`.
- Downstream ML compatibility validation after simulation contract changes: `uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.ml.test_ICSOR tests.ml.test_ml_orchestration`.
- Measured-space model and orchestration validation: `uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.ml.test_ICSOR tests.ml.test_tabular_regressors tests.ml.test_ml_orchestration`.
- Broad regression safety sweep: `uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m unittest tests.bootstrap.test_repo_contract tests.simulation.test_asm2d_tsn_simulation tests.ml.test_ICSOR tests.ml.test_tabular_regressors tests.ml.test_ml_orchestration`.
- ASM2D-TSN public API smoke check without artifacts: `uv run c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -c "from src.models.simulation.asm2d_tsn_simulation import run_asm2d_tsn_simulation; result = run_asm2d_tsn_simulation(save_artifacts=False, n_samples=4, random_seed=17, parallel_workers=1); print(result['dataset'].shape); print(result['metadata']['measured_output_columns']); print(result['composition_matrix'].shape)"`.
- Notebook JSON integrity validation after notebook edits: `c:/Users/eselerio/projects/pibre-model/.venv/Scripts/python.exe -m json.tool main.ipynb > $null`.
- Dependency refresh after pyproject.toml changes: `uv sync`.

### Modifiers and Escalation

- If unittest output is sparse or truncated in the terminal tool, immediately check exit status with `Write-Output $LASTEXITCODE`.
- If a focused test run fails and transcript detail is insufficient, rerun with `-v` on the same unittest module list.
- For PowerShell `python -c` commands, use a double-quoted PowerShell command and single-quoted Python string literals.
- Use small in-memory smoke checks (`save_artifacts=False`, small `n_samples`) after tests pass to validate public API shape and metadata quickly.

### Known Pitfalls to Avoid

- Do not use a single-quoted top-level PowerShell payload for `python -c` unittest runners; quoting may be mangled and raise `SyntaxError`.
- Do not use PowerShell here-strings that preserve double-quoted Python print literals or nested f-strings without verifying quote behavior first.
- Do not use `ProcessPoolExecutor.map` with keyword-only worker payloads; use `executor.submit` with keyword arguments.
- Do not use small-batch parallel timing runs to judge process-based throughput; process startup can dominate and mislead conclusions.
- Do not treat stale hard-coded schema expectations as command-strategy failures; update tests to derive expected measured-output shapes from metadata contracts.

### Maintenance Rules for This Section

- Keep this section short and reusable.
- Replace or merge overlapping entries instead of appending near-duplicate cases.
- Add a new bullet only when it introduces a new command pattern, escalation rule, or durable pitfall.


