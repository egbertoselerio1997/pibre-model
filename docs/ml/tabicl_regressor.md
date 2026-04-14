# TabICL Regressor Summary

## 1. Title and model summary

The tabicl_regressor module implements a measured-space surrogate based on the open-source TabICLRegressor foundation model. The repository wraps the single-target external estimator as one independent regressor per measured output, keeps dataset splitting in the notebook, disables the repository's external scaling layer, and still applies the notebook-derived measured-space projection through the shared benchmark pipeline.

## 2. Background and use case

TabICL is a tabular foundation model built around in-context learning. Instead of optimizing a task-specific network from scratch, it reuses a pre-trained transformer that conditions on the training rows during prediction. In this repository it serves as a foundation-model benchmark against ICSOR and the classical regressors under the same operational-plus-fractional input basis and the same measured-output targets.

## 3. Mathematical definition

For one measured target, TabICL approximates the regression operator through a pre-trained transformer that consumes the training context and test features jointly:

$$
\hat{y}_{test} = f_{\theta}(X_{train}, y_{train}, X_{test})
$$

where $\theta$ is fixed by large-scale synthetic pre-training rather than notebook-time optimization. The repository fits one such predictor per measured output and stacks their outputs into a multi-target prediction matrix before the shared measured-space projection step is applied when the A matrix is active.

## 4. Inputs, outputs, and assumptions

Inputs are the notebook-managed operational variables plus fractional influent-state variables. Outputs are the measured effluent composites used for direct cross-model comparison.

The implementation assumes that TabICL's own preprocessing stack should remain in control, so external feature scaling and target scaling are disabled in config/params.json. The repository also assumes a one-target-at-a-time wrapper because the upstream TabICL regressor is single-output.

## 5. Implementation used in this repository

Implementation is in src/models/ml/tabicl_regressor.py.

The model file builds one TabICLRegressor per target inside sklearn.multioutput.MultiOutputRegressor. Shared utilities in src/utils/train.py still govern notebook-owned split handling, evaluation, projection, artifact persistence, and optional Optuna orchestration.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is an independent-target TabICL ensemble regressor. Each target equation uses the same configured TabICL hyperparameters, including the number of ensemble members, normalization strategy, feature-shuffling strategy, and runtime batching controls.

## 7. Training or optimization notes

TabICL does not perform repository-local gradient training in the usual sense. Its fit step prepares encoders, target scaling internal to TabICL, ensemble views, and optional caches for in-context inference. The notebook-owned Optuna path therefore tunes only inference-facing controls such as n_estimators, normalization mode, feature shuffling, outlier threshold, and batch size.

## 8. Prediction workflow

Prediction rebuilds the shared benchmark feature matrix from metadata, passes the features through the stored per-target TabICL estimators, stacks the resulting measured-output predictions, and then applies the shared measured-space projection when the A matrix is active.

## 9. Limitations and expected failure modes

Because the upstream regressor is single-output, inference cost grows linearly with the number of measured outputs. TabICL can also be memory-intensive on large datasets, especially when the feature count or number of ensemble members grows. Checkpoint download is automatic by default, so offline or firewalled environments must pre-populate the requested checkpoint path.

## 10. References

Qu, J., Holzmueller, D., Varoquaux, G., and Le Morvan, M. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. ICML, 2025.

Qu, J., Holzmueller, D., Varoquaux, G., and Le Morvan, M. TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model. arXiv, 2026.