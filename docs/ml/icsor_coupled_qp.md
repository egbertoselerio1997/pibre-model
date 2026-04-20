# ICSOR Coupled QP Model Summary

## 1. Title and scope

This document describes the repository implementation of `icsor_coupled_qp`, a coupled second-order ASM-component surrogate that supports either recursive block-coordinate training or an Adam-Lasso training phase, and is deployed with a HiGHS-based linear program.

The mathematical source of truth is [docs/article/ICSOR-LP_CoupledQP.md](docs/article/ICSOR-LP_CoupledQP.md). This file documents implementation details and repository contracts.

## 2. Native model contract

The model is trained and deployed in ASM fractional component space.

- Inputs: operational variables plus influent ASM fractional components.
- Native target: effluent ASM fractional components.
- External comparison/reporting: measured composites are produced only after prediction via the configured composition matrix.

The notebook comparison layer therefore remains externally collapsed measured-output metrics, while the model-native diagnostics stay in fractional space.

## 3. Objective and training structure

The implementation supports two coefficient-estimation mechanisms under one model contract:

1. `recursive_qp` (original): cyclic block-coordinate updates where `B`, `Gamma`, and `C_hat` are updated with closed-form and OSQP-backed subproblems.
2. `adam_lasso` (new default): a gradient phase over `B`, boxed zero-diagonal `Gamma`, and a positive parameterization of `C_hat`, with Lasso penalties on `B` and `Gamma`.

Both mechanisms keep the same deployment-time constrained projection stage.

The shared coupled objective uses three blocks:

1. driver coefficients `B`
2. coupling matrix `Gamma`
3. fitted nonnegative state matrix `C_hat`

When `training_method=recursive_qp`, the training loop uses cyclic block-coordinate updates with restarts:

1. `B` update by ridge-style linear solve
2. `Gamma` update by OSQP with minimal convex admissibility set
3. `C_hat` update by per-sample OSQP nonnegative QP with an invariant-residual penalty weighted by `lambda_inv`

The recursive-QP path uses a minimal admissible set for `Gamma`:

- diagonal fixed to zero
- off-diagonal box bounds `[-gamma_abs_bound, +gamma_abs_bound]`
- L2 regularization via `lambda_gamma`
- conditioning guard on `R = I - Gamma`

When `training_method=adam_lasso`, the optimizer keeps the same fit, invariant, and coupled-system terms from the article but changes the regularizer family:

- `lasso_lambda_B` weights an L1 penalty on `B`
- `lasso_lambda_gamma` weights an L1 penalty on `Gamma`
- the gradient phase uses a positive parameterization for provisional `C_hat`
- before the training result is returned, `Gamma` is passed through the conditioning guard and `C_hat` is re-solved with the exact nonnegative QP for the returned `(B, Gamma)` pair so the exposed state remains aligned with the prediction-space objective

## 4. Deployment inference

For each sample, the model builds the feature driver, forms the coupled affine predictor, and solves a nonnegative deployment LP in component space using SciPy HiGHS.

- Raw prediction: unconstrained coupled solve from `R c = d`
- Projected prediction: exact-invariant nonnegative LP correction around the affine predictor

The model returns both raw and projected outputs so notebook diagnostics and cross-model effective-metric logic remain consistent.

## 5. Configuration namespace

Model settings are loaded from `config/params.json` under `icsor_coupled_qp`.

Important keys include:

- `training_method` (`adam_lasso` or `recursive_qp`)
- `lambda_inv`, `lambda_sys`
- `lambda_B`, `lambda_gamma` for the recursive-QP ridge regularizers
- `lasso_lambda_B`, `lasso_lambda_gamma` for the Adam-Lasso L1 regularizers
- `gamma_abs_bound`
- `max_outer_iterations`, `n_restarts`
- `objective_tolerance`, `parameter_tolerance`, `conditioning_max`
- `adam_epochs`, `adam_learning_rate`, `adam_beta1`, `adam_beta2`, `adam_epsilon`
- `adam_clip_grad_norm`, `adam_log_interval`, `adam_foreach`
- `osqp_eps_abs`, `osqp_eps_rel`, `osqp_max_iter`, `osqp_polish`, `osqp_verbose`
- `enable_training_warm_start`, `enable_gamma_warm_start`, `enable_c_hat_warm_start`, `warm_start_clip_tolerance`
- `nonnegativity_tolerance`, `constraint_tolerance`
- `highs_presolve`, `highs_max_iter`, `highs_verbose`, `highs_retry_without_presolve`, `parallel_workers`

`scale_features` and `scale_targets` are required to remain `false` for this model.

## 6. Artifacts and bundle fields

The persisted model bundle includes:

- `B_matrix`, `Gamma_matrix`, and `R_matrix`
- `design_schema`, feature/target/constraint columns
- `A_matrix` and `composition_matrix`
- hyperparameters and training diagnostics
- scaling bundle and notebook comparison metadata fields

Artifacts are saved through the shared model artifact path patterns in `config/paths.json`.

## 7. Current first-pass limitations

The first implementation intentionally excludes:

- coefficient uncertainty and prediction interval outputs
- exact-equality invariant-constrained training variant
- Optuna tuning for CoupledQP-specific hyperparameters

Those can be added in a later revision without changing the current notebook and artifact contracts.
