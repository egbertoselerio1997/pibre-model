# Orthogonal Null-Space Projection in the Measured-Space ML Pipelines

## 1. Purpose

This note documents how the repository derives and applies the measured-space orthogonal projection used by the machine-learning models in `src/models/ml`.

It answers two implementation questions:

1. how the projected predictions shown in `main.ipynb` are generated
2. whether the projection is part of model training or an external post-processing step

The short answer is:

- for the classical regressors (`adaboost_regressor`, `catboost_regressor`, `lightgbm_regressor`, `random_forest_regressor`, `svr_regressor`, `xgboost_regressor`), projection is external to estimator fitting and is applied after the model produces raw predictions
- for `collapsed_cobre`, projection is folded analytically into the least-squares solution, and the repository still reports both raw and projected predictions derived from the fitted parameter matrices

## 2. Measured-space constraint construction in the notebook

The notebook first builds the macroscopic stoichiometric matrix in the measured-output basis and then computes its null space:

$$
S_{macro} = S_{Petersen} C^T
$$

where:

- $S_{Petersen}$ is the process-by-state stoichiometric matrix
- $C$ is the measured-output composition matrix

The null space basis is then obtained as:

$$
N = \operatorname{null\_space}(S_{macro})
$$

and the repository defines the measured-space constraint matrix as:

$$
A = N^T
$$

In `main.ipynb`, this is the code path used to create `A_matrix`:

```python
macroscopic_stoichiometric_matrix = petersen_matrix @ composition_matrix.T
constraint_basis = null_space(macroscopic_stoichiometric_matrix)
A_matrix = constraint_basis.T
```

Interpretation:

- each row of $A$ represents one measured-space invariant
- if an output vector $y$ satisfies the invariants relative to an influent reference $y_{ref}$, then

$$
A y = A y_{ref}
$$

The notebook rounds and normalizes the displayed matrix for readability. The downstream code does not assume exact orthonormality and therefore computes the projector with a pseudoinverse.

## 3. Orthogonal projection used in the repository

The core implementation lives in `src/utils/process.py`.

The repository first constructs the projection operator:

$$
P = A^T (A A^T)^+ A
$$

where $(\cdot)^+$ denotes the Moore-Penrose pseudoinverse.

This is implemented by `build_projection_operator(A_matrix)`.

The actual projection step is then:

$$
\hat{y}_{proj} = \hat{y}_{raw} - (\hat{y}_{raw} - y_{ref}) P^T
$$

implemented by `project_to_mass_balance(raw_predictions, constraint_reference, A_matrix)`.

Because $P$ projects onto the row space of $A$, the correction removes exactly the component of $\hat{y}_{raw} - y_{ref}$ that violates the invariants. The result is the closest feasible prediction in the Euclidean sense:

$$
A \hat{y}_{proj} = A y_{ref}
$$

up to numerical precision.

Why this is called an orthogonal projection:

- the correction is the minimum-norm adjustment needed to satisfy the linear constraints
- among all vectors that satisfy the invariants, $\hat{y}_{proj}$ is the one closest to $\hat{y}_{raw}$ under the standard inner product used by the implementation

## 4. What `constraint_reference` means

The projection is not onto the homogeneous null space alone. It is onto an affine constraint set anchored by the sample-specific influent measured composites.

The repository constructs a supervised measured-space dataset with:

- `features`: operating variables plus influent measured composites
- `targets`: effluent measured outputs
- `constraint_reference`: the influent measured composite vector in the same measured-output basis as the targets

This is implemented in `build_measured_supervised_dataset(...)` in `src/utils/process.py`.

Therefore, each sample has its own reference vector $y_{ref}$, and the projection enforces the invariants relative to that sample's influent conditions.

## 5. Classical regressors: training flow and projection flow

The six classical regressors share the same structure:

- `src/models/ml/adaboost_regressor.py`
- `src/models/ml/catboost_regressor.py`
- `src/models/ml/lightgbm_regressor.py`
- `src/models/ml/random_forest_regressor.py`
- `src/models/ml/svr_regressor.py`
- `src/models/ml/xgboost_regressor.py`

Each file is intentionally thin. It defines:

1. a parameter loader
2. a model builder
3. a pipeline entry point that forwards into the shared utility `run_tabular_regressor_pipeline(...)`

The important consequence is that the projection logic is not implemented separately in each model file. It is centralized in the shared utilities under `src/utils/train.py` and `src/utils/process.py`.

### 5.1 What happens during fitting

For classical regressors, fitting happens inside `train_tabular_regressor(...)` in `src/utils/train.py`:

```python
estimator.fit(feature_frame, target_frame)
```

This call receives only features and targets. It does not receive:

- `A_matrix`
- `constraint_reference`
- a projection operator

That means the estimator is trained as an ordinary unconstrained regressor in measured-output space.

Examples:

- AdaBoost uses `MultiOutputRegressor(AdaBoostRegressor(...))`
- CatBoost uses `CatBoostRegressor(...)`
- LightGBM uses `MultiOutputRegressor(LGBMRegressor(...))`
- Random Forest uses `RandomForestRegressor(...)`
- SVR uses `MultiOutputRegressor(SVR(...))`
- XGBoost uses `MultiOutputRegressor(XGBRegressor(...))`

None of those model objects has the mass-balance projector embedded inside the estimator itself.

### 5.2 What happens after fitting

After a classical regressor produces raw predictions, the shared helper `predict_tabular_regressor_split(...)` applies the projection:

1. call `model.predict(...)`
2. inverse-transform targets back into physical units if target scaling was used
3. call `project_to_mass_balance(...)`
4. return both projected and raw predictions

In simplified form:

```python
raw_predictions = model.predict(dataset_split.features)
raw_predictions = inverse_transform_targets(raw_predictions, scaling_bundle)
projected_predictions = project_to_mass_balance(
    raw_predictions,
    dataset_split.constraint_reference.to_numpy(dtype=float),
    np.asarray(A_matrix, dtype=float),
)
```

This establishes the core behavior:

- the trained classical model first generates unconstrained raw outputs
- the repository then projects those outputs to generate physically compliant projected outputs

So for the classical regressors, the null-space projection is an external process applied to predictions, not a constraint built into training.

## 6. Why the notebook shows both raw and projected results

The notebook displays results from the report object returned by each pipeline. Those reports are created by `evaluate_prediction_bundle(...)` in `src/utils/test.py`.

That evaluation function computes metrics twice:

1. once for `raw_predictions`
2. once for `projected_predictions`

It returns:

- aggregate metrics with one row for `raw` and one row for `projected`
- per-target metrics for both raw and projected outputs
- raw and projected prediction frames
- residual summaries comparing both variants against the invariance constraints

This is why the notebook can show projected performance and projected constraint residuals without the underlying classical model being trained with a projection layer.

The projected values shown in `main.ipynb` are therefore generated by shared post-prediction evaluation code, not by a special projected estimator class for AdaBoost, CatBoost, or the other classical regressors.

## 7. Hyperparameter tuning for classical regressors

There is one subtle but important detail.

For classical regressors, fitting remains unconstrained, but Optuna tuning uses projected validation performance as the objective.

Inside `tune_tabular_regressor_hyperparameters(...)` in `src/utils/train.py`, the workflow is:

1. fit the candidate regressor on the tuning-train split
2. generate raw predictions on the tuning-test split
3. project those predictions with `project_to_mass_balance(...)`
4. evaluate mean squared error against the projected predictions

So the tuning objective is effectively:

$$
\min_{\theta} \operatorname{MSE}(y_{true}, \hat{y}_{proj}(\theta))
$$

not:

$$
\min_{\theta} \operatorname{MSE}(y_{true}, \hat{y}_{raw}(\theta))
$$

This means:

- projection influences hyperparameter selection
- projection does not alter the estimator's internal fitting algorithm

That distinction matters. The repository selects hyperparameters based on projected outputs while still training a conventional unconstrained regressor.

## 8. Prediction-time behavior for persisted classical regressor bundles

When a saved classical regressor bundle is loaded through `predict_tabular_regressor_model(...)`, the bundle contains:

- the fitted estimator
- saved feature and target column order
- fitted scalers
- `A_matrix`
- constraint column names

Prediction proceeds as follows:

1. rebuild the measured-space input frame if raw simulation rows are supplied
2. transform features with the saved scaler
3. call the estimator for raw predictions
4. apply the same measured-space projection using the saved `A_matrix`
5. return both `raw_predictions` and `projected_predictions`

This confirms that even after persistence, the projection remains an explicit downstream step applied to raw predictions.

## 9. `collapsed_cobre`: projection collapsed into the OLS solve

`src/models/ml/collapsed_cobre.py` does not train an unconstrained predictor and then project it afterwards. It analytically folds the measured-space projector into the least-squares problem.

The training path:

1. builds a partitioned second-order design matrix over operational variables and influent measured composites
2. computes the measured-space projection matrix and its complement
3. projects the training targets into the feasible subspace during the OLS assembly step
4. solves for one raw parameter matrix and one effective projected parameter matrix

The stored bundle therefore contains both:

- `raw_parameter_matrix` for the unconstrained bilinear response
- `effective_parameter_matrix` for the projected measured-space response

Prediction then evaluates both parameter matrices on the same design frame. So `collapsed_cobre` is different from the classical regressors:

- projection is not a post-hoc call to `project_to_mass_balance(...)` at evaluation time
- projection is not a differentiable training loop inside a neural network
- projection is built directly into the closed-form linear-algebra solution used for fitting

## 11. Summary by model family

### 11.1 Classical regressors

Models:

- AdaBoost
- CatBoost
- LightGBM
- Random Forest
- SVR
- XGBoost

Projection placement:

- external post-processing step

Training target:

- unconstrained raw measured outputs

Tuning objective:

- projected validation MSE

Reported notebook outputs:

- both raw and projected metrics and residuals

### 11.2 Collapsed COBRE

Model:

- partitioned bilinear design with projected multivariate least squares

Projection placement:

- collapsed analytically into the fitted parameter matrices

Training target:

- targets projected through the measured-space projector complement during the OLS solve

Tuning objective:

- none in the current repository workflow

Reported notebook outputs:

- both raw and projected metrics and residuals

## 12. Practical interpretation of the notebook results

When reading the classical-regressor outputs in `main.ipynb`, keep the following interpretation in mind:

- `raw` rows show what the unconstrained estimator predicted directly
- `projected` rows show the physically corrected predictions after orthogonal projection into the measured-space invariant set
- `constraint_residuals` compare how strongly the raw and projected outputs violate the invariants

Therefore, if the projected metrics improve or the projected constraint residuals collapse toward zero, that does not mean the estimator itself learned the invariants. It means the repository's post-processing projection successfully corrected the raw outputs.

For `collapsed_cobre`, the interpretation is different: projected outputs are not just a post-hoc correction. They come from the effective parameter matrix obtained by analytically incorporating the projection into the regression solve.

## 13. Why the pseudoinverse form is used

The projector implementation uses:

$$
A^T (A A^T)^+ A
$$

instead of assuming that $(A A^T)^{-1}$ exists exactly.

This is numerically safer because:

- `A_matrix` originates from a floating-point null-space computation
- the notebook rounds and normalizes the displayed basis for readability
- future constraint sets may not remain perfectly orthonormal or perfectly independent under all preprocessing choices

Using the pseudoinverse makes the projector robust to mild rank deficiency or numerical perturbations while preserving the intended orthogonal projection behavior.

## 14. Repository locations relevant to this methodology

Core notebook construction:

- `main.ipynb`

Shared projection and measured-space preprocessing:

- `src/utils/process.py`

Shared classical-regressor training, tuning, and prediction flow:

- `src/utils/train.py`

Shared reporting of raw versus projected results:

- `src/utils/test.py`

Classical regressor wrappers:

- `src/models/ml/adaboost_regressor.py`
- `src/models/ml/catboost_regressor.py`
- `src/models/ml/lightgbm_regressor.py`
- `src/models/ml/random_forest_regressor.py`
- `src/models/ml/svr_regressor.py`
- `src/models/ml/xgboost_regressor.py`

Projected least-squares bilinear model:

- `src/models/ml/collapsed_cobre.py`

## 15. Final answer to the implementation question

For `src/models/ml/adaboost_regressor.py`, `src/models/ml/catboost_regressor.py`, and the other classical regressors, the orthogonal null-space projection is not embedded within model training. The estimator is trained first, raw predictions are generated, and those raw predictions are then projected to produce the projected outputs shown in `main.ipynb`.

For `src/models/ml/collapsed_cobre.py`, the projection is not a separate prediction-time correction step. It is incorporated analytically into the least-squares formulation that produces the effective projected parameter matrix.