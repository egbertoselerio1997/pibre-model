# LightGBM Regressor Summary

## 1. Title and model summary

The lightgbm_regressor module implements a measured-space surrogate based on lightgbm.LGBMRegressor. The repository fits one LightGBM model per measured effluent target through a multi-output wrapper and then projects the raw predictions onto the A-matrix invariant subspace used throughout the measured-space workflow.

## 2. Background and use case

LightGBM is a gradient-boosting framework built around histogram-based tree learning and leaf-wise tree growth. It is designed for efficient training on structured tabular data and can model strong nonlinear behavior without explicit feature engineering.

In this repository it serves as an alternative surrogate model for ASM1-derived measured outputs. The statistical learner approximates the input-output map, while a separate projection step restores compliance with the measured conservation relations.

## 3. Mathematical definition

For each target, the predictor is an additive tree ensemble

$$
\hat{y}(x) = \sum_{m=1}^{T} f_m(x),
$$

where the weak learners $f_m$ are decision trees trained by gradient boosting. In LightGBM the tree-building process uses histogram binning and leaf-wise growth to reduce computational cost and often improve accuracy on tabular data.

The repository again applies the measured-space projection

$$
C^{*} = C_{raw} - A^T (A A^T)^{-1} A (C_{raw} - C_{in}),
$$

so the returned projected prediction satisfies the configured measured-space constraints.

## 4. Inputs, outputs, and assumptions

Inputs are HRT, aeration, and the measured influent composites derived from the ASM1 composition matrix.

Outputs are the eight measured effluent composites defined by the repository contract: COD, TSS, TN, TP, NH4_N, NO3_N, PO4_P, and Alkalinity.

The main assumptions are that the training data are simulation-driven, the measured-output basis is fixed, and physical consistency is imposed after prediction by projection rather than by embedding the constraints inside the boosting algorithm.

## 5. Implementation used in this repository

Implementation is in src/models/ml/lightgbm_regressor.py.

The repository uses shared preprocessing, notebook-managed splitting, scaling, external tuning, evaluation, and persistence helpers. The model file contributes only the LightGBM estimator builder and the model-specific train, predict, and thin pipeline wrappers that satisfy the common machine-learning interface.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is histogram-based gradient-boosted decision trees as implemented by LightGBM. The repository wraps LGBMRegressor in sklearn.multioutput.MultiOutputRegressor, so each target receives its own LightGBM ensemble. Raw multi-target predictions are coupled only by the final projection stage.

## 7. Training or optimization notes

The Optuna search space now covers all non-infrastructure LightGBM controls defined for this benchmark: number of boosting rounds, learning rate, number of leaves, depth control, minimum child samples, row subsampling, column subsampling, and L1 and L2 penalties. Only infrastructure constants stay fixed in config/params.json (objective, random seed, verbosity, and worker count), while the notebook-owned orchestration block sets the shared Optuna budget.

## 8. Prediction workflow

Prediction proceeds by loading the saved model bundle, preparing measured-space features, applying any saved feature scaling, generating raw predictions from the wrapped LightGBM model, projecting them into the invariant subspace, and returning both raw and projected outputs aligned to the supplied rows.

## 9. Limitations and expected failure modes

Leaf-wise growth can overfit very small datasets if depth-related controls are too loose. Independent target fitting can also miss shared target structure before the projection step. Accuracy will degrade if predictions are requested far outside the simulated operating region used for training.

## 10. References

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems, 2017.

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

The repository reference note in references/doc.md describes the measured-space projection framework used after raw regression.