# COBRE Model Summary

## 1. Title and model summary

This document describes the repository implementation of COBRE as a constrained bilinear regression model over wastewater fractional states. The model predicts measured macroscopic effluent composites while enforcing stoichiometric invariants derived from the configured Petersen matrix.

This file is an implementation companion, not the primary theory source. The authoritative mathematical specification is `docs/article/theoretical_framework.md`. If any notation or derivation in this file conflicts with that article, the article is the source of truth.

## 2. Background and use case

The repository uses COBRE when an interpretable, closed-form surrogate is preferred over a generic black-box regressor. COBRE separates two physically distinct input roles:

- operational controls $u \in \mathbb{R}^{M_{op}}$
- influent fractional states $C_{in} \in \mathbb{R}^{F}$

This separation matters because the effluent must reflect both reactor operating conditions and the invariant structure implied by the influent fractional composition and stoichiometry.

## 3. Theory-aligned notation

The implementation now follows the article notation in notebook displays and COBRE-specific reports:

- $u$: operational control vector
- $C_{in}$: influent fractional state
- $C_{raw}$: unconstrained fractional prediction
- $C^*$: constrained fractional prediction
- $A$: null-space invariant matrix derived from the Petersen matrix
- $P_{inv} = A^T (A A^T)^{-1} A$: invariant-space projector
- $P_{adm} = I - P_{inv}$: admissible-space projector
- $G = I_{comp} P_{adm}$: collapsed admissible operator
- $H = I_{comp} P_{inv}$: invariant pass-through operator

Some persisted bundle keys retain legacy names such as `effective_coefficients` for backward compatibility with the notebook and analysis pipeline. Those keys now represent the measured-space coefficient blocks induced by $G$ and, for the linear influent block, the additional $H$ pass-through term.

## 4. Mathematical definition

The unconstrained fractional surrogate is

$$
C_{raw} = W_u u + W_{in} C_{in} + b + \Theta_{uu}(u \otimes u) + \Theta_{cc}(C_{in} \otimes C_{in}) + \Theta_{uc}(u \otimes C_{in})
$$

where the coefficient blocks are stored as 2D matrices over an unsymmetrized Kronecker-style second-order basis.

The invariant matrix is built from the null space of the Petersen matrix $\nu$:

$$
A = \operatorname{null\_space}(\nu)^T
$$

The constrained fractional prediction is obtained by orthogonal projection:

$$
C^* = P_{adm} C_{raw} + P_{inv} C_{in}
$$

The measured-space prediction uses the configured composition matrix $I_{comp}$:

$$
y^* = I_{comp} C^* = G C_{raw} + H C_{in}
$$

After substitution, the measured-space prediction remains a second-order polynomial in $u$ and $C_{in}$, with interpreted coefficient blocks:

- $W_{u,y} = G W_u$
- $W_{in,y} + H = G W_{in} + H$
- $b_y = G b$
- $\Theta_{uu,y} = G \Theta_{uu}$
- $\Theta_{cc,y} = G \Theta_{cc}$
- $\Theta_{uc,y} = G \Theta_{uc}$

## 5. Training objective and identifiable parameters

For $N$ samples, the implementation builds a row-oriented design matrix $\Phi \in \mathbb{R}^{N \times D}$ over the operational, influent, bias, and second-order interaction basis. Let $Y \in \mathbb{R}^{N \times K}$ be the measured effluent targets and let $C_{IN} \in \mathbb{R}^{N \times F}$ be the influent fractional states aligned to the same rows.

The transformed target is

$$
\widetilde{Y} = Y - C_{IN} H^T
$$

and the fitted identifiable measured-space parameter matrix $M \in \mathbb{R}^{K \times D}$ satisfies

$$
\widetilde{Y} = \Phi M^T
$$

The repository solves this projected least-squares problem analytically. The raw fractional parameter matrix is then reconstructed as a minimum-norm solution consistent with the collapsed objective, and the notebook-facing measured coefficient blocks are unpacked from the identified parameter matrix.

## 6. Returned artifacts and persisted bundle fields

The COBRE training pipeline returns a result bundle that includes:

- the fitted model bundle for serialization and later prediction
- measured-space coefficient blocks for notebook plots and interpretation
- raw fractional coefficient blocks for lower-level diagnostics
- metric tables and COBRE-specific evaluation reports
- coefficient uncertainty summaries when inference succeeds

The persisted model bundle stores the matrices and metadata required to reproduce predictions:

- $A$, $P_{inv}$, $P_{adm}$, $G$, and $H$
- the composition matrix and feature schema
- the raw fractional parameter matrix
- the identifiable measured-space parameter matrix
- unpacked coefficient blocks for backward-compatible downstream use
- inference metadata and, when available, bootstrap parameter samples

## 7. Coefficient uncertainty

COBRE now returns uncertainty information for estimated coefficients at training time.

The top-level training result includes:

- `coefficient_inference`: metadata describing the inference method, confidence level, rank diagnostics, and residual degrees of freedom
- `identifiable_coefficient_uncertainty`: standard errors and interval bounds for the identifiable measured-space parameter matrix estimated from $\widetilde{Y} = \Phi M^T$
- `effective_coefficient_uncertainty`: the same coefficient intervals after the deterministic $H$ pass-through contribution is added to the linear influent block used for measured-space interpretation

The implementation chooses the uncertainty method as follows:

- `auto` uses analytic Gaussian inference when the projected least-squares design is full rank and falls back to bootstrap percentile inference when the design is rank deficient or analytic covariance estimation is not reliable
- `analytic` forces the analytic covariance path even when the design is rank deficient
- forced analytic intervals from a rank-deficient design are returned with a caution note because the original design-basis coefficients are not uniquely identifiable coefficientwise

The default confidence level is 0.95, and the default bootstrap fallback is controlled from `config/params.json`.

## 8. Prediction workflow and prediction uncertainty

Prediction proceeds as follows:

1. Load the persisted COBRE bundle.
2. Rebuild or align the feature frame to the saved operational and influent schema.
3. Rebuild the second-order design matrix.
4. Compute the raw fractional prediction $C_{raw}$.
5. Compute the constrained fractional prediction $C^*$.
6. Map raw and constrained fractional predictions into measured space.
7. If inference metadata are present, compute measured-space uncertainty for the projected prediction.

When available, `predict_cobre_model()` returns a `prediction_uncertainty` payload with:

- standard errors for the mean projected prediction
- lower and upper confidence bounds for the mean projected prediction
- standard errors for a future projected observation
- lower and upper prediction bounds for a future projected observation

Analytic prediction uncertainty uses leverage and residual covariance from the projected least-squares fit. Bootstrap prediction uncertainty uses stored bootstrap parameter samples and residual resampling.

## 9. Notebook and downstream compatibility

The implementation changes preserve the existing downstream workflow while updating symbols to match the article.

The protected COBRE notebook sections continue to produce the same functional outputs:

- the COBRE training section still trains, evaluates, serializes, and displays report tables
- the coefficient-visualization section still renders six measured-space coefficient figures, but now titles them with theory-aligned symbols such as $W_{u,y}$, $W_{in,y} + H$, and $b_y$
- the COBRE response-surface section still builds the same contour plots and now can also expose prediction uncertainty summaries in the preview data
- later comparison sections can continue consuming the same report and prediction-table structure because the new uncertainty content is additive

## 10. Architecture details and limitations

The adopted architecture is a constrained second-order polynomial regressor with bilinear interaction terms and an analytically derived orthogonal projection. It is not a neural network and it is not a tree-based ensemble.

Important limitations remain:

- the model is specific to the configured ASM basis, stoichiometric matrix, and measured-output composition mapping
- the raw fractional parameter matrix is not uniquely identifiable from measured-space supervision alone
- the unsymmetrized second-order basis can become rank deficient, especially when quadratic terms introduce duplicate interaction columns
- extrapolation outside the simulated operating envelope remains risky even when the prediction surface looks smooth

Expected failure modes include feature-order mismatches, inconsistent composition-matrix dimensions, or a deliberate stoichiometric-schema revision that changes the null-space basis and invalidates previously saved bundles.

## 11. References

For the formal COBRE derivation used as the repository gold standard, see `docs/article/theoretical_framework.md`.

Additional background references used by the repository include:

- Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.
- Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
- Golub, G. H., and Van Loan, C. F. Matrix Computations. Johns Hopkins University Press, 2013.
- Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.
