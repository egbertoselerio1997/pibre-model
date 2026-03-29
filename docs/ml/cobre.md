# COBRE Model Summary

## 1. Title and model summary

The COBRE module in this repository implements a strict fractional-space Constrained Bilinear Regression model whose final prediction target is the measured macroscopic effluent space. The raw surrogate is trained over operational inputs and influent ASM1 fractions, the physical invariants are derived from the null space of the Petersen matrix, and the physically admissible effluent is collapsed into the measured-output basis through the configured composition matrix.

The implementation is not a neural network and it is not a generic measured-space polynomial regressor. It is a closed-form constrained bilinear surrogate whose training objective is the analytically collapsed measured-output projection derived from the underlying fractional stoichiometric model.

## 2. Background and use case

Biological wastewater process data contain two distinct classes of explanatory variables:

- operational controls, such as hydraulic retention time and aeration
- influent fraction states, such as soluble substrate, ammonium nitrogen, nitrate nitrogen, phosphate phosphorus, dissolved oxygen, alkalinity, and particulate fractions

The COBRE model separates those two roles explicitly. The operational controls drive the reactor regime, while the influent fractions define the entering material state that must also anchor the conservation constraints. A purely data-driven regression can fit measured outputs, but without additional structure it can violate invariants implied by microbiology and stoichiometry. COBRE addresses that weakness by combining:

- a bilinear second-order surrogate in the fractional state space
- a null-space projector derived from the Petersen matrix
- a composition mapping from fractional states to measured macroscopic outputs
- a closed-form least-squares calculation of the fitted coefficients

The repository uses COBRE when the goal is an interpretable analytical surrogate that predicts measured effluent composites while remaining consistent with the invariant fractional subspace implied by the configured ASM1 stoichiometry.

## 3. Mathematical definition

Let:

- $u \in \mathbb{R}^{M_{op}}$ denote the operational input vector
- $C_{in} \in \mathbb{R}^{F}$ denote the influent fractional state vector
- $C_{raw} \in \mathbb{R}^{F}$ denote the raw fractional prediction
- $I_{comp} \in \mathbb{R}^{K \times F}$ denote the composition matrix that maps fractions to measured composites
- $\nu \in \mathbb{R}^{R \times F}$ denote the Petersen matrix

The unconstrained raw fractional model is:

$$
C_{raw} = W_u u + W_{in} C_{in} + b + \Theta_{uu}(u \otimes u) + \Theta_{cc}(C_{in} \otimes C_{in}) + \Theta_{uc}(u \otimes C_{in})
$$

where:

- $W_u \in \mathbb{R}^{F \times M_{op}}$
- $W_{in} \in \mathbb{R}^{F \times F}$
- $b \in \mathbb{R}^{F}$
- $\Theta_{uu} \in \mathbb{R}^{F \times M_{op}^2}$
- $\Theta_{cc} \in \mathbb{R}^{F \times F^2}$
- $\Theta_{uc} \in \mathbb{R}^{F \times (M_{op}F)}$

The invariant matrix $A$ is built from the null space of the Petersen matrix:

$$
A = \operatorname{null\_space}(\nu)^T
$$

The orthogonal projector onto the invariant-error subspace is:

$$
P = A^T (A A^T)^+ A
$$

and the complementary projector is:

$$
P_{\perp} = I - P
$$

The physically admissible fractional state is therefore:

$$
C^* = P_{\perp} C_{raw} + P C_{in}
$$

The measured composite prediction is:

$$
C_{comp}^* = I_{comp} C^* = I_{comp} P_{\perp} C_{raw} + I_{comp} P C_{in}
$$

Substituting the raw bilinear model yields the collapsed form:

$$
C_{comp}^* = W_{u,eff} u + W_{in,eff} C_{in} + b_{eff} + \Theta_{uu,eff}(u \otimes u) + \Theta_{cc,eff}(C_{in} \otimes C_{in}) + \Theta_{uc,eff}(u \otimes C_{in})
$$

with:

$$
W_{u,eff} = I_{comp} P_{\perp} W_u
$$

$$
W_{in,eff} = I_{comp}(P_{\perp} W_{in} + P)
$$

$$
b_{eff} = I_{comp} P_{\perp} b
$$

and the same left multiplication by $I_{comp} P_{\perp}$ for the three bilinear coefficient tensors.

For $N$ samples the repository uses row-oriented matrices:

- $\Phi \in \mathbb{R}^{N \times D}$ for the engineered design matrix
- $Y \in \mathbb{R}^{N \times K}$ for the measured targets
- $C_{IN} \in \mathbb{R}^{N \times F}$ for the influent fractional states

The transformed target used in the implementation is:

$$
\widetilde{Y} = Y - C_{IN} P^T I_{comp}^T
$$

and the training equation is:

$$
\widetilde{Y} = \Phi B^T P_{\perp}^T I_{comp}^T
$$

where $B \in \mathbb{R}^{F \times D}$ is the raw fractional coefficient matrix assembled from all linear, bias, and bilinear blocks.

The repository does not materialize the full dense Kronecker matrix during normal training. Instead it uses a two-stage least-squares calculation that is mathematically equivalent to the explicit Kronecker solve:

1. solve for the collapsed parameter proxy $M = B^T P_{\perp}^T I_{comp}^T$
2. solve $I_{comp} P_{\perp} B = M^T$ for a minimum-norm raw fractional coefficient matrix

The equivalence to the explicit Kronecker formulation is covered by the COBRE unit tests in this repository.

## 4. Inputs, outputs, and assumptions

The strict COBRE dataset contract in this repository is:

- feature inputs: the configured operational columns followed by the influent fractional state columns `In_*`
- targets: the measured effluent outputs `Out_*` in the configured measured-output order
- constraint reference: the influent fractional state vector, stored without the `In_` prefix so its columns align directly with the fractional $A$ matrix

Important assumptions:

- feature scaling is disabled
- target scaling is disabled
- the feature order must remain `[operational_columns, In_state_columns]`
- the constraint reference columns must match the fractional state ordering used by the Petersen matrix
- the composition matrix row order must match the measured target ordering used for training and reporting

## 5. Implementation used in this repository

The implementation lives in `src/models/ml/cobre.py`, and the notebook orchestrates the required data preparation in `main.ipynb`.

The exact repository flow is:

1. run the ASM1 simulator to generate the dataset, Petersen matrix, and composition matrix
2. derive a strict COBRE invariant matrix from `null_space(petersen_matrix)`
3. build a COBRE-specific supervised dataset whose features stay in fractional space while the targets stay in measured-output space
4. build the explicit unsymmetrized second-order design matrix over operational inputs and influent fractions
5. compute $P$, $P_{\perp}$, the collapsed operator $I_{comp} P_{\perp}$, and the pass-through operator $I_{comp} P$
6. form the transformed measured target $\widetilde{Y}$
7. solve the two-stage least-squares problem using NumPy
8. recover raw fractional coefficient blocks and the direct effective measured-space coefficient blocks
9. report raw and projected measured-output metrics together with raw and projected fractional-space constraint residuals
10. optionally persist the model bundle and metrics using the repository path patterns from `config/paths.json`

The persisted COBRE bundle stores:

- the fractional invariant matrix $A$
- the composition matrix $I_{comp}$
- the projector $P$ and complement $P_{\perp}$
- the collapsed and pass-through operators
- the raw fractional parameter matrix
- the raw measured parameter matrix
- the effective measured parameter matrix
- named raw and effective coefficient blocks
- the design schema and scaling metadata
- the selected hyperparameters and training metadata

### 5.1 Effective coefficient visualization in the notebook

After the COBRE training cell runs in `main.ipynb`, the notebook renders six figures directly from `cobre_result["model_bundle"]["effective_coefficients"]`:

- a heatmap for $W_{u,eff}$ with measured targets on the y-axis and operational variables on the x-axis
- a heatmap for $W_{in,eff}$ with measured targets on the y-axis and influent fractional variables on the x-axis
- a bar plot for $b_{eff}$ with measured targets on the x-axis and coefficient value on the y-axis
- a composite heatmap figure for $\Theta_{uu,eff}$ with one operational-by-operational subplot per measured target
- a composite heatmap figure for $\Theta_{cc,eff}$ with one influent-by-influent subplot per measured target
- a composite heatmap figure for $\Theta_{uc,eff}$ with one operational-by-influent subplot per measured target

Those tensor figures preserve the repository's unsymmetrized Kronecker-style ordering. The notebook does not symmetrize, reorder, or aggregate the stored interaction blocks before plotting them, so each heatmap axis corresponds exactly to the ordered design basis used during training.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is a constrained second-order polynomial regression with bilinear interaction terms and an analytically collapsed orthogonal projector. In machine-learning terms it is a multivariate ordinary least-squares regressor over a hand-built second-order feature map.

It is not a deep-learning architecture and it is not a tree-based ensemble. The closest standard description is:

- multivariate polynomial regression of degree two over a structured feature map
- augmented with a linear equality-constrained projection derived from null-space stoichiometry

The repository deliberately keeps the quadratic and interaction blocks unsymmetrized so that the stored tensor coefficients correspond exactly to the ordered Kronecker-style feature basis used in the design matrix.

## 7. Training or optimization notes

Training is fully analytical in the current repository implementation.

There is:

- no gradient descent
- no epoch schedule
- no early stopping
- no Optuna tuning path for COBRE

The configured backend is `ols_backend = "numpy_lstsq"`. The implementation uses NumPy least squares twice to avoid explicitly constructing the dense Kronecker matrix while preserving the same mathematical objective.

Progress bars remain enabled by default to comply with the repository-wide machine-learning training visibility rules.

## 8. Prediction workflow

Prediction proceeds as follows:

1. load the persisted bundle
2. rebuild the strict COBRE dataset if a raw simulation dataframe is supplied
3. align the feature frame to the saved operational-plus-fractional feature order
4. rebuild the design matrix from the saved schema
5. generate the raw fractional prediction $C_{raw}$
6. generate the projected fractional prediction $C^* = P_{\perp} C_{raw} + P C_{in}$
7. map both variants into measured-output space with the composition matrix
8. return measured raw predictions, measured projected predictions, raw fractional predictions, projected fractional predictions, and the aligned fractional constraint reference

## 9. Limitations and expected failure modes

Important limitations in this repository implementation are:

- the model is specific to the configured ASM1 fractional basis and measured-output composition mapping
- the raw fractional coefficient matrix is not uniquely identifiable; the repository stores a minimum-norm solution consistent with the collapsed least-squares objective
- extrapolation beyond the simulated operating envelope remains risky, as with any polynomial surrogate
- the design matrix can become ill-conditioned if the operational inputs and influent fractions do not sufficiently excite the bilinear basis

Expected failure modes include:

- mismatched column ordering between notebook-prepared features, targets, and constraint references
- a composition matrix whose row and column dimensions no longer match the metadata contract
- a Petersen matrix whose null-space basis changes because of a deliberate simulation-schema revision, requiring notebook reruns and fresh bundles

## 10. References

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

Gujer, W. Systems Analysis for Water Technology. Springer, 2008.

Golub, G. H., and Van Loan, C. F. Matrix Computations. Johns Hopkins University Press, 2013.

Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.