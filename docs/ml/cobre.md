# COBRE Model Summary

## 1. Title and model summary

The cobre module implements a closed-form projected ordinary least squares surrogate in measured-output space for Constrained Bilinear Regression. It starts from the partitioned bilinear formulation

$$
C_{raw} = W_u u + W_{in} C_{in} + b + u^T \Theta_{uu} u + C_{in}^T \Theta_{cc} C_{in} + u^T \Theta_{uc} C_{in}
$$

and solves for the coefficients after analytically integrating the orthogonal projection into the regression problem. The repository implementation therefore trains a single projected least-squares model rather than first fitting an unconstrained model and then correcting it afterward.

## 2. Background and use case

COBRE provides a bilinear measured-space surrogate whose projection is handled analytically inside the regression formulation itself rather than by a separate post-processing step.

It uses the same measured-space feature contract and the same stoichiometric projection matrix, but the parameters are computed analytically by projected multivariate least squares. This is useful when:

- deterministic closed-form training is preferred over gradient descent
- the user wants directly reportable effective coefficients after projection
- the user wants the measured-space projection built into the fitted linear-algebra solution itself

## 3. Mathematical definition

Let the feature vector be partitioned as

$$
x = \begin{bmatrix} u \\ C_{in} \end{bmatrix}
$$

where $u \in \mathbb{R}^{M_{op}}$ contains the operational variables and $C_{in} \in \mathbb{R}^{K}$ contains the measured influent composites used both as model inputs and as the projection reference.

The raw bilinear model is written as

$$
C_{raw} = W_u u + W_{in} C_{in} + b + u^T \Theta_{uu} u + C_{in}^T \Theta_{cc} C_{in} + u^T \Theta_{uc} C_{in}
$$

The measured-space invariants are encoded by $A$, and the orthogonal projection is

$$
C^* = C_{raw} - A^T (A A^T)^{-1} A (C_{raw} - C_{in})
$$

Define the projection matrix

$$
P = A^T (A A^T)^{-1} A
$$

and the complementary projector

$$
P_{\perp} = I - P
$$

Then the constrained model becomes

$$
C^* = W_{u,eff} u + W_{in,eff} C_{in} + b_{eff} + u^T \Theta_{uu,eff} u + C_{in}^T \Theta_{cc,eff} C_{in} + u^T \Theta_{uc,eff} C_{in}
$$

with

$$
W_{u,eff} = P_{\perp} W_u
$$

$$
W_{in,eff} = P_{\perp} W_{in} + P
$$

$$
b_{eff} = P_{\perp} b
$$

and the same left multiplication by $P_{\perp}$ for the three bilinear tensors.

For $N$ samples, let $\Phi \in \mathbb{R}^{N \times D}$ be the engineered design matrix containing:

- the operational block $u$
- the influent block $C_{in}$
- an optional bias column
- the full operational quadratic block $u \otimes u$
- the full influent quadratic block $C_{in} \otimes C_{in}$
- the full interaction block $u \otimes C_{in}$

The repository solves the exact projected least-squares problem by fitting the projected target matrix

$$
Y_{\perp} = Y P_{\perp}
$$

with standard multivariate least squares:

$$
\min_M \lVert \Phi M - Y_{\perp} \rVert_F^2
$$

This is algebraically equivalent to the Kronecker formulation of the projected OLS problem, but it is solved without explicitly materializing the dense Kronecker matrix.

The implementation supports two numerical backends for this regression step:

- a baseline NumPy backend that applies `numpy.linalg.lstsq` directly to $(\Phi, Y_{\perp})$
- a DirectML-priority backend that offloads the matrix multiplications needed for the normal-equation terms $\Phi^T \Phi$ and $\Phi^T Y_{\perp}$ to a DirectML-backed torch device and then solves the resulting linear system on the CPU

The configured default is `ols_backend = "auto"`, which tries the DirectML path first and falls back to the baseline NumPy least-squares solve whenever DirectML is unavailable or fails.

## 4. Inputs, outputs, and assumptions

Inputs follow the existing measured-space repository contract:

- all non-`In_` measured-space feature columns are treated as operational variables $u$
- all `In_*` measured-space columns are treated as the influent composite block $C_{in}$

Outputs are the measured-space effluent targets already defined by the current ASM1 metadata. In the present repository configuration this basis is metadata-driven and currently contains 15 measured outputs rather than the older 8-output subset.

Assumptions:

- the feature order remains `[operational_columns, In_* measured composites]`
- the influent measured columns and `constraint_reference` columns describe the same measured basis and order
- feature and target scaling are disabled for this model because the exact constrained equations are expressed in physical measured-space coordinates

## 5. Implementation used in this repository

Implementation is in `src/models/ml/cobre.py`.

The exact repository workflow is:

1. build measured-space features with `build_measured_supervised_dataset`
2. keep notebook-managed train-test splits and pass them into the model runner
3. validate that feature scaling and target scaling are disabled
4. partition the existing measured-space features into $u$ and $C_{in}$ using the column contract
5. build the full second-order design matrix with explicit block ordering
6. compute $P$ and $P_{\perp}$ from the supplied $A$ matrix
7. resolve the configured OLS backend and try the DirectML matrix-multiplication path first when available
8. otherwise solve the projected multivariate least-squares system directly with `numpy.linalg.lstsq`
9. recover raw coefficient blocks for one minimum-norm representative of the unconstrained model and then construct the effective constrained coefficients by adding the $P$ pass-through contribution to the linear influent block
10. evaluate both raw and projected predictions with the shared reporting utilities
11. optionally persist a pickle bundle and metrics JSON under the configured results paths

The saved bundle includes the design-schema metadata, raw and effective parameter matrices, named coefficient blocks, the projection matrices, backend-selection metadata for the OLS solve, and the standard scaling and column-order metadata used elsewhere in the repository.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is a projected second-order polynomial regression in measured-output space. It is not a neural network.

In machine-learning terms, the implementation is a multivariate ordinary least squares model over a hand-built second-order feature map, with the physics constraints embedded analytically through the integrated projection algebra.

The repository deliberately keeps the quadratic blocks unsymmetrized. In other words, the operational, influent, and interaction tensors are stored over the full ordered outer-product basis rather than a deduplicated symmetric basis. This makes the mapping between the design matrix and the persisted coefficient tensors mechanically exact.

## 7. Training or optimization notes

There is no gradient descent, no early stopping, and no Optuna path for this model in the current repository implementation.

Training is fully deterministic once the notebook-managed dataset split is fixed. The baseline numerical solver is the NumPy least-squares routine used to solve the projected multivariate regression system. When `ols_backend` is set to `auto`, the repository first attempts a DirectML-assisted normal-equations path that accelerates the cross-product matrix multiplications before falling back to the baseline NumPy solve.

Because the raw coefficient matrix is only identifiable up to components annihilated by $P_{\perp}$, the repository stores both:

- a minimum-norm raw representative used for reporting raw predictions
- the effective constrained coefficients used for direct interpretation of the physically compliant model

The effective coefficients are the physically meaningful quantities.

## 8. Prediction workflow

Prediction in this repository follows the contract below:

1. load the persisted COBRE bundle
2. rebuild measured-space features if a raw simulation dataframe is supplied
3. align the feature frame to the saved measured-space column order
4. rebuild the partitioned second-order design matrix from the saved schema
5. compute raw predictions from the stored minimum-norm raw parameter matrix
6. compute projected predictions from the stored effective parameter matrix
7. return aligned raw predictions, projected predictions, and the measured-space constraint reference

## 9. Limitations and expected failure modes

Important limitations:

- the model is exact only for the current measured-space basis and column contract used by the repository
- coefficient interpretation assumes features remain in physical coordinates, so feature scaling is intentionally disabled
- the full second-order feature map can become large if the measured basis is expanded substantially
- the minimum-norm raw coefficients are not unique in a physical sense; the effective constrained coefficients are the quantities that should be interpreted

Expected failure modes:

- inconsistent column ordering between features and constraint references will break the partitioned design assumptions
- weak extrapolation outside the simulation envelope, as with any polynomial surrogate
- numerical ill-conditioning if the design matrix becomes nearly rank-deficient for a particular dataset split
- slightly different numerical answers between the baseline and DirectML-assisted paths because the DirectML matrix multiplications use torch tensors before the final CPU solve

## 10. References

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

Gujer, W. Systems Analysis for Water Technology. Springer, 2008.

Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.
