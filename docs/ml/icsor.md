# ICSOR Model Summary

## 1. Title and model summary

This document describes the repository implementation of non-negative ICSOR as a constrained second-order surrogate over ASM component-space influent states. The model predicts measured effluent composites while preserving stoichiometric invariants and enforcing non-negativity of the deployed ASM component prediction.

This file is an implementation companion, not the primary theory source. The authoritative mathematical specification is `docs/article/ICSOR-quadratic-programming.md`. If any notation or derivation in this file conflicts with that article, the article is the source of truth.

## 2. Background and use case

The repository uses ICSOR when an interpretable physics-informed surrogate is preferred over a generic black-box regressor. ICSOR separates two physically distinct input roles:

- operational controls $u \in \mathbb{R}^{M_{op}}$
- influent ASM component states $c_{in} \in \mathbb{R}^{F}$

This separation matters because the effluent must reflect both reactor operating conditions and the invariant structure implied by the influent component inventory and the adopted stoichiometric network.

## 3. Theory-aligned notation

The implementation follows the article notation in notebook displays and ICSOR-specific reports:

- $u$: operational control vector
- $c_{in}$: influent ASM component state
- $c_{raw}$: unconstrained component-space prediction
- $c_{aff}$: affine invariant projection of $c_{raw}$
- $c^*$: final non-negative deployed component prediction
- $A$: null-space invariant matrix derived from the Petersen matrix
- $P_{inv} = A^T (A A^T)^{-1} A$: projector onto the invariant row space
- $P_{adm} = I - P_{inv}$: projector onto the admissible change space
- $G = I_{comp} P_{adm}$: affine measured-space core operator
- $H = I_{comp} P_{inv}$: invariant carry-through operator

Some persisted bundle keys retain legacy names such as `effective_parameter_matrix` and `effective_coefficients` for notebook compatibility. In the current implementation those fields represent the affine measured-space core, not the final piecewise-affine deployed predictor.

## 4. Mathematical definition

The unconstrained component-space surrogate is

$$
c_{raw} = W_u u + W_{in} c_{in} + b + \Theta_{uu}(u \otimes u) + \Theta_{cc}(c_{in} \otimes c_{in}) + \Theta_{uc}(u \otimes c_{in})
$$

where the coefficient blocks are stored over an unsymmetrized second-order design basis.

The invariant matrix is built from the null space of the Petersen matrix $\nu$:

$$
A = \operatorname{null\_space}(\nu)^T
$$

The repository keeps the original affine ICSOR projector as a reference stage:

$$
c_{aff} = P_{adm} c_{raw} + P_{inv} c_{in}
$$

The final deployed state is the Euclidean projection of $c_{raw}$ onto the intersection of the invariant-consistent affine set and the nonnegative orthant. The repository solves that strictly convex quadratic program with OSQP only when it is needed:

$$
c^* = \operatorname{Proj}_{\{c : A c = A c_{in},\; c \ge 0\}}(c_{raw})
$$

Measured outputs are then obtained by the configured composition matrix $I_{comp}$:

$$
y^* = I_{comp} c^*
$$

When the affine projector is already non-negative, the deployed predictor reduces exactly to $c_{aff}$ and the OSQP stage is skipped.

## 5. Training objective and identifiable parameters

For $N$ samples, the implementation builds a row-oriented design matrix $\Phi \in \mathbb{R}^{N \times D}$ over the operational, influent, bias, and second-order interaction basis. Let $Y \in \mathbb{R}^{N \times K}$ be the measured effluent targets and let $C_{IN} \in \mathbb{R}^{N \times F}$ be the aligned influent component states.

The transformed measured target is

$$
\widetilde{Y} = Y - C_{IN} H^T
$$

and the identifiable affine measured-space operator $M \in \mathbb{R}^{K \times D}$ satisfies

$$
\widetilde{Y} = \Phi M^T
$$

The repository supports two manual affine-core estimators for this measured-space fit.

- `ols` uses the original closed-form least-squares solution for the identifiable affine core.
- `ridge` applies L2 shrinkage to the same identifiable affine core, with the bias block left unpenalized.

In both cases, one admissible raw component-space coefficient matrix is then reconstructed as a minimum-norm solution consistent with the collapsed affine objective. The non-negative correction is not part of coefficient fitting; it is a post-estimation deployment step. When notebook-managed Optuna tuning is enabled for ICSOR, the notebook keeps the estimator fixed to ridge and tunes only the regularization strength `ridge_alpha`.

## 6. Returned artifacts and persisted bundle fields

The ICSOR training pipeline returns:

- the fitted model bundle for serialization and later prediction
- raw component-space coefficient blocks for lower-level diagnostics
- affine measured-space coefficient blocks for notebook interpretation
- metric tables and staged ICSOR evaluation reports
- coefficient uncertainty summaries for the affine measured-space core

The persisted model bundle stores the data needed to reproduce predictions:

- $A$, $P_{inv}$, $P_{adm}$, $G$, and $H$
- the composition matrix and design schema
- the raw component-space parameter matrix
- the identifiable affine measured-space parameter matrix
- the affine-core coefficient blocks exposed through the legacy `effective_*` fields
- estimator metadata, including the selected affine-core estimator and ridge strength when applicable
- OSQP projection settings loaded from `config/params.json`
- coefficient-inference metadata for the affine measured-space core

## 7. Coefficient uncertainty

ICSOR returns uncertainty information for the affine measured-space core estimated during training.

The top-level training result includes:

- `coefficient_inference`: metadata describing the inference method, confidence level, rank diagnostics, and residual degrees of freedom
- `identifiable_coefficient_uncertainty`: intervals for the identifiable measured-space operator estimated from $\widetilde{Y} = \Phi M^T$
- `effective_coefficient_uncertainty`: the same intervals after the deterministic $H$ carry-through contribution is added to the linear influent block used for affine measured-space interpretation

Prediction-time uncertainty is reported with `affine_core_prediction_*` outputs. Those intervals describe the affine measured-space core only. They are not exact finite-sample intervals for the final deployed predictor once the non-negativity constraints become active.

The implementation chooses the inference method as follows:

- `auto` resolves to analytic affine-core inference. For OLS runs it uses the familiar covariance formulas, with a pseudoinverse-based warning when the design is rank deficient. For ridge runs it uses the conditional-on-fixed-penalty Gaussian approximation described in the article.
- `analytic` is the explicit version of the same analytic affine-core path
- `none` is used internally by the Optuna tuning helper to skip uncertainty calculation during trial scoring

The default confidence level and OSQP settings are loaded from `config/params.json`.

## 8. Prediction workflow and staged deployment

Prediction proceeds as follows:

1. Load the persisted ICSOR bundle.
2. Rebuild or align the feature frame to the saved operational and influent schema.
3. Rebuild the second-order design matrix.
4. Compute the raw component prediction $c_{raw}$.
5. If $c_{raw}$ already satisfies the invariant equalities and non-negativity, keep it.
6. Otherwise compute the affine reference state $c_{aff}$.
7. If $c_{aff}$ is non-negative, keep it.
8. Otherwise solve the reduced non-negative quadratic program with OSQP.
9. Collapse the affine and final component states into measured space.
10. If inference metadata are present, compute affine-core prediction uncertainty.

`predict_icsor_model()` returns both the affine reference prediction and the final deployed prediction:

- `affine_predictions` and `affine_fractional_predictions`
- `projected_predictions` and `projected_fractional_predictions`
- `projection_stage_diagnostics` and `projection_stage_summary`
- `affine_core_prediction_*` uncertainty outputs when available

This staged output makes it explicit when the affine projector was already sufficient and when OSQP had to activate.

## 9. Notebook and downstream compatibility

The repository keeps the notebook-facing affine-core coefficient fields stable so existing coefficient plots and bundle inspections continue to work.

Downstream behavior now distinguishes three prediction objects:

- raw component and measured predictions
- affine invariant-consistent reference predictions
- final non-negative deployed predictions

The notebook and analysis helpers now surface the affine stage when it exists, but the final measured prediction used for comparisons, response surfaces, and deployment remains `projected_predictions`.

## 10. Architecture details and limitations

The adopted architecture is a constrained second-order polynomial regressor with bilinear interaction terms, an affine invariant projector, and a sample-wise non-negative QP correction solved by OSQP. It is not a neural network and it is not a tree-based ensemble.

Important limitations remain:

- the model is specific to the configured ASM basis, stoichiometric matrix, and measured-output composition mapping
- the raw component-space parameter matrix is not uniquely identifiable from measured-space supervision alone
- the unsymmetrized second-order basis can become rank deficient, especially when quadratic terms introduce duplicate interaction columns
- the final deployed predictor is piecewise affine rather than one global affine measured-space map
- ridge intervals are conditional Gaussian approximations for the affine core and are not exact finite-sample $t$-intervals
- closed-form uncertainty formulas apply to the affine core, not in general to the final OSQP-corrected predictor
- extrapolation outside the simulated operating envelope remains risky even when the prediction surface looks smooth

Expected failure modes include feature-order mismatches, inconsistent composition-matrix dimensions, or a stoichiometric-schema revision that changes the null-space basis and invalidates previously saved bundles.

## 11. References

For the formal ICSOR derivation used as the repository gold standard, see `docs/article/ICSOR-quadratic-programming.md`.

Additional background references used by the repository include:

- Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.
- Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
- Golub, G. H., and Van Loan, C. F. Matrix Computations. Johns Hopkins University Press, 2013.
- Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.

