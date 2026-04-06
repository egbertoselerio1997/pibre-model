# Orthogonal Null-Space Projection in the Repository ML Pipelines

## 1. Purpose

This note explains how orthogonal projection is used in the machine-learning pipelines in this repository after the ICSOR refactor to the strict fractional-space formulation.

The central distinction is now explicit:

- the classical regressors now benchmark on the same operational and influent fractional inputs used by ICSOR, while still training measured-output targets and applying projection afterward in measured space
- ICSOR now trains from a fractional-space raw model whose physically admissible state is collapsed into the measured-output basis analytically

Those are related ideas, but they are not the same implementation.

## 2. Two invariant matrices now exist in the notebook workflow

The notebook keeps two different invariant constructions because the repository currently supports two different model families.

### 2.1 Fractional invariant matrix for ICSOR

For ICSOR, the invariant matrix is derived directly from the Petersen matrix:

$$
A_{frac} = \operatorname{null\_space}(\nu)^T
$$

This is the strict matrix required by the fractional-space derivation, because the projection constraint is imposed on the fractional change itself.

### 2.2 Measured-space invariant matrix for the classical regressors

The classical regressors still use the older measured-space projector derived from the macroscopic stoichiometric matrix:

$$
S_{macro} = \nu I_{comp}^T
$$

$$
A_{measured} = \operatorname{null\_space}(S_{macro})^T
$$

This measured-space matrix is retained in the notebook because the classical regressors continue to project measured-output predictions directly.

That construction is not guaranteed to be non-trivial. By rank-nullity,

$$
\dim\bigl(\operatorname{null\_space}(S_{macro})\bigr) = n_{measured} - \operatorname{rank}(S_{macro})
$$

where $n_{measured}$ is the number of measured-output coordinates. If $S_{macro}$ has full column rank in the measured-output basis, then the null space is zero-dimensional and $A_{measured}$ has shape $(0, n_{measured})$.

In that case the classical measured-space projector is mathematically trivial. There is no non-zero measured-space invariant left to enforce after collapsing the full fractional-state system into the chosen measured-output coordinates. The repository therefore treats the classical projection feature as inactive rather than as a successful zero-violation result.

## 3. Shared projector formula

Both model families use the same projector construction once an invariant matrix $A$ has been chosen:

$$
P = A^T (A A^T)^+ A
$$

where $(\cdot)^+$ is the Moore-Penrose pseudoinverse. The complementary projector is:

$$
P_{\perp} = I - P
$$

The helper that builds this projector lives in `src/utils/process.py`.

## 4. Classical regressors: ICSOR-aligned inputs with measured-space post-projection

The classical regressors in `src/models/ml` still follow the measured-space path:

1. build the same operational-plus-fractional input feature frame used by ICSOR
2. keep the supervised target in measured-output space
3. keep the projection reference in measured composite space
4. fit an unconstrained regressor in measured-output space
5. generate raw measured-output predictions
6. project those predictions onto the affine measured-space invariant set using the measured-space constraint reference

This benchmark contract makes the direct comparison with ICSOR stricter at the input and split level, even though the classical models still remain measured-output regressors rather than fractional-state regressors.

For those models the projected prediction is:

$$
\hat{y}_{proj} = \hat{y}_{raw} - (\hat{y}_{raw} - y_{ref}) P^T
$$

with:

- $\hat{y}_{raw}$ in measured-output space
- $y_{ref}$ in measured-output space
- $A = A_{measured}$

This remains a post-processing step outside the estimator fitting itself.

## 5. ICSOR: strict fractional-space projection with measured-space collapse

ICSOR now uses a different path.

### 5.1 Raw model space

The notebook prepares:

- operational inputs $u$
- influent fractional states $C_{in}$
- measured effluent targets $Y$

The raw ICSOR model predicts fractional states:

$$
C_{raw} = W_u u + W_{in} C_{in} + b + \Theta_{uu}(u \otimes u) + \Theta_{cc}(C_{in} \otimes C_{in}) + \Theta_{uc}(u \otimes C_{in})
$$

### 5.2 Projection space

The fractional correction is staged rather than purely affine.

The affine invariant reference state is:

$$
C_{aff} = P_{\perp} C_{raw} + P C_{in}
$$

with $A = A_{frac}$.

If $C_{raw}$ already satisfies the invariant equalities and non-negativity, the repository keeps $C_{raw}$ directly.

If $C_{aff}$ is also non-negative, the repository keeps $C_{aff}$ directly.

Only when the affine reference state still violates componentwise non-negativity does the repository solve the final reduced quadratic program with OSQP:

$$
C^* = \operatorname{Proj}_{\{C : A C = A C_{in},\; C \ge 0\}}(C_{raw})
$$

The affine projector therefore remains part of the implementation, but it is now the reference stage and inactive-constraint special case rather than the universal deployed solution.

### 5.3 Measured-output collapse

The measured prediction used for reporting and training loss is:

$$
C_{comp}^* = I_{comp} C^*
$$

The repository also retains the affine measured-space core

$$
C_{comp,aff} = I_{comp} C_{aff}
$$

because the least-squares training objective is still built on that affine measured-space relation even though the raw model and the constraints live in fractional space.

### 5.4 Implemented transformed target

The row-oriented transformed target used in the repository is:

$$
\widetilde{Y} = Y - C_{IN} P^T I_{comp}^T
$$

The repository then solves the analytically collapsed least-squares problem without explicitly building the large Kronecker matrix during normal training.

That affine-core fit is followed at prediction time by the staged raw or affine or OSQP deployment logic described above.

## 6. Reporting differences between the model families

The reporting distinction is now important.

### 6.1 Classical regressors

For the classical regressors:

- regression metrics are measured in measured-output space
- when $A_{measured}$ is non-trivial, constraint residuals are also measured in measured-output space
- when $A_{measured}$ is non-trivial, projection-adjustment diagnostics summarize how far the raw measured prediction moved during post-projection
- when $A_{measured}$ is trivial, the notebook reports only the raw measured-output metrics and marks the classical projection path as inactive

### 6.2 ICSOR

For ICSOR:

- regression metrics are reported for raw, affine, and final projected measured-output predictions after mapping through the composition matrix
- constraint residuals are measured on the fractional raw, affine, and final projected states against the fractional influent reference
- projection-adjustment diagnostics now distinguish raw-to-affine, affine-to-projected, and raw-to-projected changes
- staged projection diagnostics report when the raw prediction was already feasible, when the affine reference was sufficient, and when OSQP had to activate

The repository now uses a unified high-level report shape with a direct comparison layer and a separate model-native diagnostic layer.

## 7. Notebook orchestration implications

The notebook remains the only place where train-test splitting and any Optuna-only subset creation occur. After the benchmark refactor, the notebook therefore carries two aligned supervised datasets:

- a classical benchmark dataset with ICSOR-aligned operational-plus-fractional inputs, measured-output targets, and measured-space projection references
- a ICSOR-specific dataset with fractional influent features and measured targets

The notebook also keeps both invariant matrices so later classical-regressor cells can continue to use the measured-space projector while the ICSOR cells use the strict fractional projector.

If the measured-space null space collapses to zero dimension, the notebook now prints that status explicitly and suppresses the classical projected-result and measured-space discrepancy tables. This avoids misreading a vacuous projector as evidence that an unconstrained classical regressor satisfied mass balance.

## 8. Limitations

The present repository design intentionally supports both projection strategies at once. That choice preserves backward compatibility for the classical regressors, but it means a reader must distinguish carefully between:

- the fractional ICSOR invariant matrix
- the measured-space classical-regressor invariant matrix

Confusing those two objects will lead to incorrect interpretations of the reported residuals or of the fitted coefficient matrices.

There is an additional limitation for the classical measured-space path: after the full fractional system is collapsed into a small measured-output basis, the retained coordinates may no longer preserve a non-trivial null space. When that happens, the repository cannot produce a meaningful measured-space projection or a meaningful measured-space post-projection discrepancy diagnostic for the classical models. Those diagnostics are therefore treated as unavailable, not as identically zero.
