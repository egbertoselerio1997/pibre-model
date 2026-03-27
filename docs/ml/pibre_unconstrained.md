# PIBRe Unconstrained Model Summary

## 1. Title and model summary

The pibre_unconstrained module implements a decoupled measured-space bilinear surrogate. It first trains an unconstrained predictor and only afterwards applies the measured-space projection as a post-processing step when physically consistent outputs are needed.

In this repository, the model remains in the same measured-output basis used by ASM1-derived targets: COD, TSS, TN, TP, NH4-N, NO3-N, PO4-P, and alkalinity.

## 2. Background and use case

The original PIBRe workflow combines a learned predictor and a projection layer in one training graph. The unconstrained variant separates these concerns so that:

1. the regression step can use standard closed-form or convex estimators such as OLS and Ridge
2. physical projection can be switched on or off at inference time without retraining the predictor

This is useful when comparing purely statistical fit quality against projected physical consistency, or when evaluating alternative constraint matrices over the same trained surrogate.

## 3. Mathematical definition

Let $x \in \mathbb{R}^{M}$ be the measured-space feature vector and let $y \in \mathbb{R}^{K}$ be the effluent measured-output target.

The unconstrained predictor is built as linear regression over a degree-2 expansion:

$$
\phi(x) = [x_1, \dots, x_M, x_1^2, x_1x_2, \dots, x_M^2]
$$

with model

$$
\hat{y}_{raw} = W\phi(x) + b
$$

where $W$ and $b$ are estimated either by ordinary least squares (OLS) or by Ridge regression.

Projection is then applied externally using the measured-space constraint matrix $A$ and influent reference $y_{in}$:

$$
\hat{y}_{proj} = \hat{y}_{raw} - A^T(AA^T)^{-1}A(\hat{y}_{raw}-y_{in})
$$

which enforces

$$
A\hat{y}_{proj} = Ay_{in}
$$

up to numerical precision.

## 4. Inputs, outputs, and assumptions

Inputs:

- HRT
- Aeration
- In_COD
- In_TSS
- In_TN
- In_TP
- In_NH4_N
- In_NO3_N
- In_PO4_P
- In_Alkalinity

Outputs:

- Out_COD
- Out_TSS
- Out_TN
- Out_TP
- Out_NH4_N
- Out_NO3_N
- Out_PO4_P
- Out_Alkalinity

Assumptions:

- training and evaluation are performed in measured-output space
- the projection basis and target basis remain aligned
- unconstrained regression is allowed to violate invariants before projection

## 5. Implementation used in this repository

Implementation is in src/models/ml/pibre_unconstrained.py.

Repository workflow:

1. main.ipynb builds measured-space features and notebook-managed splits
2. optional Optuna tuning is executed externally through src/utils/train.py
3. the model runner fits feature scaling on the train split
4. features are expanded with degree-2 polynomial/bilinear terms
5. OLS or Ridge is trained on unconstrained targets
6. raw predictions are generated
7. projection is applied only as post-processing for evaluation and reporting
8. artifacts are optionally persisted as model bundle and JSON summaries

## 6. Architecture details and adopted standard architecture name

The adopted architecture is a quadratic feature map plus linear estimator. This corresponds to a second-order polynomial regression view of bilinear modeling.

Two configurable estimators are provided:

1. OLS (ordinary least squares)
2. Ridge (L2-regularized linear regression)

Projection is not part of the learned estimator; it is an external deterministic correction step.

## 7. Training or optimization notes

Hyperparameter tuning is executed from the notebook orchestration flow, not inside the model module.

Tuned parameters may include:

- regression_mode (ols or ridge)
- ridge_alpha
- fit_intercept
- include_bias for feature expansion

Training progress is reported with TQDM. The optimization objective for tuning can be evaluated on projected predictions so hyperparameter selection respects physical consistency while keeping estimator fitting unconstrained.

## 8. Prediction workflow

Prediction flow:

1. load persisted model bundle
2. rebuild measured-space features when raw simulation rows are supplied
3. transform features using the saved scaler
4. apply saved degree-2 feature expansion
5. compute unconstrained raw predictions
6. optionally compute projected predictions with the saved $A$ matrix
7. return aligned dataframes for raw and projected outputs

## 9. Limitations and expected failure modes

Limitations:

- unconstrained raw predictions may violate mass-balance invariants
- projection enforces only the encoded invariants, not all process physics
- polynomial expansion can overfit when dataset size is small relative to expanded dimensionality

Expected failure modes:

- unstable extrapolation outside the simulated operating envelope
- weak generalization if Ridge regularization is too small
- biased fit if feature scaling or column ordering is inconsistent between train and inference

## 10. References

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

Gujer, W. Systems Analysis for Water Technology. Springer, 2008.

Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.
