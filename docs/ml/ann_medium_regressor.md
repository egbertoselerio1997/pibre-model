# Medium Neural-Network Regressor Summary

## 1. Title and model summary

The ann_medium_regressor module implements a measured-space surrogate based on sklearn.neural_network.MLPRegressor with hidden layers (96, 48). This is the repository's recommended ANN architecture. It is designed to give the neural benchmark enough nonlinear capacity to model coupled operational and influent effects without introducing the optimization instability and overcapacity risk of the deeper alternative.

## 2. Background and use case

The repository compares three ANN shapes: shallow, medium, and deep. The medium network is the preferred architecture because the benchmark feature space is moderate in size, the target space contains only a small set of measured outputs, and the training workflow must remain compatible with the repository's shared classical-regressor pipeline. A two-stage compression from 96 to 48 hidden units is a practical middle ground between expressiveness and stability.

## 3. Mathematical definition

For input vector $x$, the medium network computes

$$
h_1 = \phi(W_1 x + b_1)
$$

$$
h_2 = \phi(W_2 h_1 + b_2)
$$

$$
\hat{y}_{raw} = W_3 h_2 + b_3
$$

with ReLU activation $\phi(\cdot)$ and Adam optimization during training. The repository then projects $\hat{y}_{raw}$ onto the measured invariant subspace defined by the notebook-derived A matrix whenever projection is active.

## 4. Inputs, outputs, and assumptions

Inputs are the shared benchmark features built from operational variables plus fractional influent-state variables. Outputs are the measured effluent composites used for direct benchmark comparison.

The architecture assumes repository-managed scaling for both features and targets. It also assumes early_stopping remains false, because the repository rules reserve train-test and Optuna subset management for main.ipynb rather than estimator-owned validation logic.

## 5. Implementation used in this repository

Implementation is in src/models/ml/ann_medium_regressor.py.

The model file only defines the MLP builder and the standard wrapper functions. Shared utilities in src/utils/train.py handle scaling, training orchestration, evaluation, projection, and artifact persistence, while Optuna execution remains notebook-owned.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is a two-hidden-layer multilayer perceptron with hidden_layer_sizes = (96, 48), ReLU activation, and Adam optimization.

This is the recommended ANN shape for four reasons.

First, the benchmark input space is large enough to benefit from nonlinear mixing, but not so large that a very deep network is necessary.

Second, the first hidden layer with 96 units gives the model room to represent cross-effects among HRT, Aeration, and the fractional influent states.

Third, the second hidden layer with 48 units forces a controlled compression before the final measured-output map, which acts as a regularizing bottleneck.

Fourth, the shape remains small enough that the shared sklearn-based training path is still practical for the notebook-managed Optuna workflow and the repeated dataset-size analysis sweeps.

## 7. Training or optimization notes

The notebook-owned Optuna path tunes regularization strength, batch size, learning rate, iteration budget, and tolerance while leaving the architecture fixed. The shape is fixed intentionally so architecture comparison remains interpretable across the shallow, medium, and deep ANN entries.

The repository uses sklearn MLPRegressor rather than a custom torch training loop because the current classical-regressor contract already provides compatible scaling, persistence, Optuna integration, and reporting for sklearn estimators. That keeps the ANN family inside the same benchmark machinery as the other classical models.

## 8. Prediction workflow

Prediction loads the saved bundle, rebuilds the benchmark feature matrix, applies the stored feature scaler, produces raw measured-output predictions from the fitted MLP, inverse-transforms the targets when target scaling was enabled, and finally applies the measured-space projection when active.

## 9. Limitations and expected failure modes

Even the recommended medium network can still be sensitive to learning-rate and regularization choices, and sklearn MLPRegressor does not expose the same level of training-loop control as a custom torch implementation. Because early stopping is disabled to preserve notebook-owned split governance, convergence behavior depends heavily on the configured max_iter, tol, and learning_rate_init values.

## 10. References

Rumelhart, D. E., Hinton, G. E., and Williams, R. J. Learning Representations by Back-Propagating Errors. Nature, 323, 533-536, 1986.

Kingma, D. P., and Ba, J. Adam: A Method for Stochastic Optimization. International Conference on Learning Representations, 2015.

Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. MIT Press, 2016.