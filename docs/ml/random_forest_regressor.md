# Random Forest Regressor Summary

## 1. Title and model summary

The random_forest_regressor module implements a measured-space surrogate based on sklearn.ensemble.RandomForestRegressor. In this repository the random forest is trained directly on the multi-target measured-output problem and its raw predictions are then projected onto the measured-space invariant subspace before the final projected outputs are reported.

## 2. Background and use case

Random forests combine many decorrelated regression trees through bootstrap aggregation. They are often strong, stable baselines for structured data because they capture nonlinear interactions while reducing the variance of individual trees.

For the present repository the method provides a non-boosting tree ensemble benchmark for the ASM1 surrogate problem. It is useful when a robust tabular baseline is needed without the sequential optimization structure of boosting.

## 3. Mathematical definition

For one input vector $x$, a random forest prediction is the average of tree predictions,

$$
\hat{y}(x) = \frac{1}{T} \sum_{m=1}^{T} h_m(x),
$$

where each $h_m$ is trained on a bootstrap resample with randomized split selection. The scikit-learn implementation used here supports multi-output regression directly, so a single forest predicts all measured outputs simultaneously. The raw multi-output vector is then projected using

$$
C^{*} = C_{raw} - A^T (A A^T)^{-1} A (C_{raw} - C_{in}).
$$

## 4. Inputs, outputs, and assumptions

Inputs are HRT, aeration, and the ASM1-derived measured influent composites.

Outputs are the eight measured effluent composites defined by the repository contract.

The method assumes that the training dataset is representative of the operating space of interest and that the measured-output basis is unchanged between training and inference.

## 5. Implementation used in this repository

Implementation is in src/models/ml/random_forest_regressor.py.

The exact workflow mirrors the shared machine-learning contract in this repository: measured-space preprocessing, notebook-managed train-test splitting, optional feature scaling, optional external Optuna tuning on a notebook-managed subset, training on the provided training split, projection-aware evaluation, and optional artifact persistence.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is bootstrap-aggregated decision-tree regression, specifically the scikit-learn RandomForestRegressor implementation. Unlike the wrapped single-target models, this estimator handles all measured targets directly in one multi-output forest.

## 7. Training or optimization notes

The Optuna search space now tunes all non-infrastructure random-forest controls used in this benchmark: number of trees, maximum tree depth, minimum split size, minimum leaf size, feature fraction per split, and bootstrap sampling mode. Only infrastructure constants (random seed and worker count) remain fixed in config/params.json, while the shared Optuna trial count stays notebook-owned.

## 8. Prediction workflow

Prediction loads the saved random forest bundle, reconstructs the measured-space feature matrix, applies any configured feature scaling, computes raw multi-output predictions from the forest, projects those predictions to the invariant subspace, and returns aligned raw and projected dataframes.

## 9. Limitations and expected failure modes

Random forests do not extrapolate well outside the sampled domain and may require many trees for stable performance on complex response surfaces. Even though the forest predicts all targets jointly, it still does not natively enforce the measured-space invariants and therefore relies on the final projection stage.

## 10. References

Breiman, L. Random Forests. Machine Learning, 45, 5-32, 2001.

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

The repository reference note in references/doc.md describes the measured-space projection framework used after raw regression.