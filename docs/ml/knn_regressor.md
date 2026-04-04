# K-Nearest Neighbors Regressor Summary

## 1. Title and model summary

The knn_regressor module implements a measured-space surrogate based on sklearn.neighbors.KNeighborsRegressor. The repository fits the model directly in the shared operational-plus-fractional feature basis, applies repository-managed scaling before training, and then projects the raw measured-output predictions onto the invariant subspace defined by the notebook-derived A matrix.

## 2. Background and use case

K-nearest neighbors regression is a non-parametric baseline that predicts one output vector from the local neighborhood of similar samples. It is useful in this repository because it tests whether the ASM2d-TCN mapping can be recovered from local geometric similarity alone, without fitting an explicit global parametric relationship.

## 3. Mathematical definition

Given a query point $x$, the model finds the set $\\mathcal{N}_k(x)$ containing the $k$ nearest training samples under the configured Minkowski distance. For one target vector, the raw prediction is

$$
\hat{y}(x) = \frac{\sum_{i \in \\mathcal{N}_k(x)} w_i y_i}{\sum_{i \in \\mathcal{N}_k(x)} w_i}
$$

where $w_i = 1$ for uniform weighting or a distance-based weight when the configured scheme is distance weighting. In this repository the resulting raw measured-output vector is projected with the same A-matrix projection used by the other classical regressors.

## 4. Inputs, outputs, and assumptions

Inputs are the shared benchmark features built in main.ipynb: operational variables together with fractional influent-state variables. Outputs are the measured effluent composites used throughout the classical benchmark.

The key assumption is that feature scaling must remain active, because KNN is distance-based and would otherwise be dominated by the largest-magnitude features.

## 5. Implementation used in this repository

Implementation is in src/models/ml/knn_regressor.py.

The repository relies on shared helpers for notebook-managed splitting, scaling, tuning, evaluation, projection, and persistence. The model-specific module only defines the estimator builder plus the standard load, train, predict, and pipeline wrappers required by the repository contract.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is distance-weighted multi-output k-nearest-neighbors regression. The default benchmark configuration uses a Minkowski metric with configurable $p$, a direct multi-output estimator, and repository-managed post-projection in measured-output space.

## 7. Training or optimization notes

The notebook-owned Optuna path tunes neighborhood size, weighting mode, search algorithm, leaf size, and the Minkowski exponent. Because the estimator is instance-based rather than gradient-trained, there is no internal epoch loop. The repository progress bars still report the shared training stages so the model participates in the same orchestration contract as the other regressors.

## 8. Prediction workflow

Prediction loads the saved bundle, rebuilds the benchmark feature matrix, standardizes the features with the persisted scaler, produces raw measured-output predictions from the trained KNN model, and then applies the measured-space invariant projection whenever the A matrix is active.

## 9. Limitations and expected failure modes

KNN can degrade in high-dimensional spaces because nearest-neighbor distances become less informative. It also scales poorly in memory and latency as the training set grows, and it extrapolates weakly outside the observed operating region because predictions are assembled from nearby historical samples rather than a learned global law.

## 10. References

Altman, N. S. An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression. The American Statistician, 46(3), 175-185, 1992.

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

The repository reference note in references/doc.md describes the measured-space projection framework used after raw regression.