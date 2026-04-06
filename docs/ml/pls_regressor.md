# Partial Least Squares Regressor Summary

## 1. Title and model summary

The pls_regressor module implements a measured-space surrogate based on sklearn.cross_decomposition.PLSRegression. The repository fits a multi-output latent-factor regression in the shared benchmark feature basis, uses the repository-wide scaling helpers, disables estimator-side scaling to avoid double normalization, and then projects raw measured-output predictions with the notebook-derived A matrix.

## 2. Background and use case

Partial least squares regression is a latent-variable method designed for settings where predictors are numerous, correlated, or both. It is valuable here because the operational variables and fractional influent variables can have strong dependency structure, and PLS provides a compact linear baseline that explicitly models predictor-response covariance rather than only predictor variance.

## 3. Mathematical definition

PLSRegression seeks latent scores and loadings such that

$$
X \approx T P^\top, \qquad Y \approx U Q^\top
$$

while each latent component is chosen to maximize covariance between the projected predictor scores and response scores. The resulting regression can be written as

$$
\hat{Y} = X B
$$

where $B$ is the coefficient matrix assembled from the learned latent factors. In this repository the raw measured-output prediction $\hat{Y}$ is then projected onto the measured invariant subspace whenever the A matrix is active.

## 4. Inputs, outputs, and assumptions

Inputs are the notebook-managed operational-plus-fractional benchmark features. Outputs are the measured effluent composites used for direct comparison with ICSOR and the other classical regressors.

The important implementation assumption is that estimator-side scaling remains false. All scaling is handled centrally in src/utils/train.py so the split governance, inverse transforms, and persistence contract stay uniform across models.

## 5. Implementation used in this repository

Implementation is in src/models/ml/pls_regressor.py.

The model module contributes the PLS estimator builder and the standard wrapper functions, while the shared training utilities handle scaling, evaluation, projection, artifact persistence, and notebook-driven Optuna optimization.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is multi-output partial least squares regression with a configurable number of latent components. The central model-design choice is n_components, which controls the rank of the latent representation used to couple the feature space and the measured-output space.

## 7. Training or optimization notes

The notebook-owned Optuna path tunes all non-compatibility PLS controls in this benchmark: number of latent components together with the iterative solver controls max_iter and tol. The scale and copy flags remain fixed in config/params.json because preprocessing is already governed by the shared repository scaling contract.

## 8. Prediction workflow

Prediction loads the persisted bundle, rebuilds the benchmark feature matrix, applies the stored feature scaler, produces raw measured-output predictions through the fitted latent-factor operator, inverse-transforms the targets if target scaling was used, and finally applies the invariant-space projection when active.

## 9. Limitations and expected failure modes

PLS is fundamentally linear after the latent projection, so it can miss strongly nonlinear interactions that tree ensembles or neural networks may recover. It can also underfit if too few latent components are retained and overreact to noise if too many are used relative to the information content of the measured outputs.

## 10. References

Wold, H. Partial Least Squares. In: Kotz, S., and Johnson, N. L., Encyclopedia of Statistical Sciences, Volume 6, Wiley, 1985.

de Jong, S. SIMPLS: An Alternative Approach to Partial Least Squares Regression. Chemometrics and Intelligent Laboratory Systems, 18(3), 251-263, 1993.

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.
