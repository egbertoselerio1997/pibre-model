# ICSOR Coupled QP Model Summary

## 1. Title and scope

This document describes the repository implementation of `icsor_coupled_qp`, a coupled second-order ASM-component surrogate trained with block-coordinate updates and solved with OSQP for the convex quadratic-program subproblems.

The mathematical source of truth is [docs/article/ICSOR-LP_CoupledQP.md](docs/article/ICSOR-LP_CoupledQP.md). This file documents implementation details and repository contracts.

## 2. Native model contract

The model is trained and deployed in ASM fractional component space.

- Inputs: operational variables plus influent ASM fractional components.
- Native target: effluent ASM fractional components.
- External comparison/reporting: measured composites are produced only after prediction via the configured composition matrix.

The notebook comparison layer therefore remains externally collapsed measured-output metrics, while the model-native diagnostics stay in fractional space.

## 3. Objective and training structure

The implementation follows the coupled objective in the article with three blocks:

1. driver coefficients `B`
2. coupling matrix `Gamma`
3. fitted nonnegative state matrix `C_hat`

The training loop uses cyclic block-coordinate updates with restarts:

1. `B` update by ridge-style linear solve
2. `Gamma` update by OSQP with minimal convex admissibility set
3. `C_hat` update by per-sample OSQP nonnegative QP

The first pass uses a minimal admissible set for `Gamma`:

- diagonal fixed to zero
- off-diagonal box bounds `[-gamma_abs_bound, +gamma_abs_bound]`
- L2 regularization via `lambda_gamma`
- conditioning guard on `R = I - Gamma`

## 4. Deployment inference

For each sample, the model builds the feature driver and solves a nonnegative deployment QP in component space using OSQP.

- Raw prediction: unconstrained coupled solve from `R c = d`
- Projected prediction: nonnegative penalized invariant deployment QP

The model returns both raw and projected outputs so notebook diagnostics and cross-model effective-metric logic remain consistent.

## 5. Configuration namespace

Model settings are loaded from `config/params.json` under `icsor_coupled_qp`.

Important keys include:

- `lambda_inv`, `lambda_sys`, `lambda_B`, `lambda_gamma`, `lambda_pred`
- `gamma_abs_bound`
- `max_outer_iterations`, `n_restarts`
- `objective_tolerance`, `parameter_tolerance`, `conditioning_max`
- `osqp_eps_abs`, `osqp_eps_rel`, `osqp_max_iter`, `osqp_polish`, `osqp_verbose`, `warm_start`

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
- exact-equality invariant-constrained variant
- Optuna tuning for CoupledQP-specific hyperparameters

Those can be added in a later revision without changing the current notebook and artifact contracts.
