# TabPFN Regressor Summary

## 1. Title and model summary

The tabpfn_regressor module implements a measured-space surrogate based on the TabPFNRegressor foundation model. The repository wraps the upstream single-target regressor as one independent predictor per measured output, keeps all split management in the notebook, disables external repository scaling, and preserves the shared post-prediction measured-space projection contract.

## 2. Background and use case

TabPFN is a pre-trained tabular foundation model designed to deliver strong accuracy on small and medium-sized regression datasets without task-specific optimization loops. In this repository it provides a foundation-model comparison point against ICSOR, the classical regressors, and TabICL under the same benchmark inputs, train-test rows, and measured-output targets.

## 3. Mathematical definition

For one measured target, TabPFN maps the training context and test rows through a pre-trained transformer prior:

$$
\hat{y}_{test} = g_{\theta}(X_{train}, y_{train}, X_{test})
$$

with $\theta$ fixed by prior training on synthetic tasks. The repository fits one TabPFN regressor per measured target and stacks the targetwise predictions before applying the shared measured-space projection when the A matrix is active.

## 4. Inputs, outputs, and assumptions

Inputs are the shared operational-plus-fractional benchmark features. Outputs are the measured effluent composites used for direct comparison with the rest of the notebook benchmark.

The implementation assumes that the upstream TabPFN regressor remains single-output, so a repository-side independent-target wrapper is necessary. It also assumes external feature and target scaling should stay disabled because TabPFN already contains its own learned preprocessing path. The repository defaults to the open-weight V2 checkpoint rather than the newer default checkpoints so the notebook can rely on a non-browser-gated model source.

## 5. Implementation used in this repository

Implementation is in src/models/ml/tabpfn_regressor.py.

The module defines a custom multi-output wrapper that fits one TabPFNRegressor per target and serializes each fitted estimator through TabPFN's own fit-state archive API before the shared repository pickle step. Shared utilities in src/utils/train.py still handle evaluation, projection, persistence of the surrounding bundle, and optional Optuna orchestration.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is an independent-target TabPFN ensemble regressor with open V2 weights. The main configurable controls are the number of ensemble members and the softmax aggregation behavior, while infrastructure controls such as fit_mode and preprocessing caching remain fixed for persistence stability.

## 7. Training or optimization notes

TabPFN does not optimize model weights during notebook execution. The fit step prepares the target-specific regression context for the pre-trained foundation model. The notebook-owned Optuna path therefore tunes only a narrow set of inference-facing controls, namely n_estimators, softmax_temperature, and average_before_softmax.

## 8. Prediction workflow

Prediction reloads the persisted repository bundle, reconstructs the saved target-specific TabPFN fit states, regenerates the shared benchmark feature matrix from metadata, stacks the measured-output predictions across targets, and finally applies the measured-space projection when the A matrix is active.

## 9. Limitations and expected failure modes

Prediction cost scales with both the number of measured outputs and the number of TabPFN ensemble members because the upstream regressor is single-output. TabPFN can also be slow on CPU relative to simpler baselines. If the requested checkpoint is not cached locally, the first fit may trigger a model download, and environments without network access must pre-populate the cache or point model_path at a local checkpoint.

## 10. References

Hollmann, N., Mueller, S., Eggensperger, K., and Hutter, F. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. ICLR, 2023.

Hollmann, N., Mueller, S., Purucker, L., Krishnakumar, A., Koerfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. Accurate Predictions on Small Data with a Tabular Foundation Model. Nature, 2025.