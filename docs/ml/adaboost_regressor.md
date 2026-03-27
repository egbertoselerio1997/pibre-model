# AdaBoost Regressor Summary

## 1. Title and model summary

The adaboost_regressor module implements a measured-space surrogate based on sklearn.ensemble.AdaBoostRegressor. The repository fits one AdaBoost regressor per measured target by means of a multi-output wrapper and then projects the raw predictions onto the measured-space invariant subspace before reporting the physically consistent outputs.

## 2. Background and use case

AdaBoost regression builds an additive ensemble of weak learners, reweighting the training problem so that later learners focus more strongly on difficult examples. It is a simpler ensemble than gradient-boosted tree methods and therefore provides a useful lower-complexity benchmark in this repository.

## 3. Mathematical definition

For one target, the final prediction can be written as a weighted sum of weak learners,

$$
\hat{y}(x) = \sum_{m=1}^{T} \alpha_m h_m(x),
$$

where $h_m$ is the $m$th weak regressor and $\alpha_m$ is its weight. The repository trains one such ensemble per measured target through sklearn.multioutput.MultiOutputRegressor and then projects the raw target vector with

$$
C^{*} = C_{raw} - A^T (A A^T)^{-1} A (C_{raw} - C_{in}).
$$

## 4. Inputs, outputs, and assumptions

Inputs are the operating variables HRT and aeration together with the measured influent composites generated from the ASM1 state space.

Outputs are the eight measured effluent composites defined by the repository contract.

The implementation assumes that the surrogate target space remains the same measured basis used to derive the A matrix and that physical consistency can be recovered by post-prediction projection.

## 5. Implementation used in this repository

Implementation is in src/models/ml/adaboost_regressor.py.

The repository uses reusable helpers for measured-space dataset construction, train-validation-test splitting, optional scaling, Optuna tuning, projection-aware evaluation, and artifact persistence. The model module provides the AdaBoost estimator builder and the common train, tune, predict, and run wrappers.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is adaptive boosting for regression, using the AdaBoostRegressor implementation from scikit-learn. Because the estimator is used through a multi-output wrapper, one boosted regression ensemble is trained per measured target.

## 7. Training or optimization notes

Optuna tunes the number of weak learners, the boosting learning rate, and the selected AdaBoost loss formulation. The repository keeps these choices in config/params.json so the source module remains free of hardcoded parameter values.

## 8. Prediction workflow

Prediction loads the persisted bundle, reconstructs the prepared feature frame, applies any saved scaling, generates raw outputs from the wrapped AdaBoost models, projects those outputs to the invariant subspace, and returns aligned raw and projected dataframes.

## 9. Limitations and expected failure modes

AdaBoost is less expressive than deeper boosted-tree frameworks on highly nonlinear problems and can become unstable if the learning rate and number of learners are not balanced well. Independent target fitting also means target coupling appears only in the post-processing projection stage.

## 10. References

Drucker, H. Improving Regressors using Boosting Techniques. Proceedings of the Fourteenth International Conference on Machine Learning, 1997.

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

The repository reference note in references/doc.md describes the measured-space projection framework used after raw regression.