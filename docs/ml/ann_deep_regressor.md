# Deep Neural-Network Regressor Summary

## 1. Title and model summary

The ann_deep_regressor module implements a measured-space surrogate based on sklearn.neural_network.MLPRegressor with hidden layers (128, 64, 32). It is the highest-capacity ANN benchmark in the repository and exists to test whether a deeper hierarchical nonlinear map improves measured-output prediction beyond the recommended medium architecture.

## 2. Background and use case

This deep network is not the recommended default. Its purpose is to act as a controlled high-capacity comparison point against the shallow and medium ANN variants under the same benchmark contract, the same scaling path, and the same invariant-space projection rules.

## 3. Mathematical definition

For input vector $x$, the deep network computes

$$
h_1 = \phi(W_1 x + b_1), \qquad h_2 = \phi(W_2 h_1 + b_2), \qquad h_3 = \phi(W_3 h_2 + b_3)
$$

$$
\hat{y}_{raw} = W_4 h_3 + b_4
$$

with ReLU activation and Adam optimization during training. The repository then applies the shared A-matrix projection to obtain the projected measured-output prediction when the invariant constraint space is active.

## 4. Inputs, outputs, and assumptions

Inputs are the notebook-managed operational-plus-fractional benchmark features. Outputs are the measured effluent composites.

As with the other ANN variants, the model assumes repository-managed feature and target scaling and keeps early_stopping false so estimator-owned validation splitting does not violate the notebook-owned orchestration rules.

## 5. Implementation used in this repository

Implementation is in src/models/ml/ann_deep_regressor.py.

The module only supplies the estimator builder and the standard wrapper functions. Shared training utilities handle scaling, evaluation, projection, persistence, and notebook-driven Optuna tuning.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is a three-hidden-layer multilayer perceptron with hidden_layer_sizes = (128, 64, 32), ReLU activation, and Adam optimization. The descending width pattern is intended to let the network first expand representational capacity and then progressively compress toward the measured-output space.

## 7. Training or optimization notes

The notebook-owned Optuna path now tunes all non-fixed deep-ANN controls used in this benchmark: activation function, regularization strength, batch size, learning rate, tolerance, and shuffle behavior. The deep shape stays fixed to preserve clean architecture-class comparison with shallow and medium ANN variants, and max_iter remains fixed as the repository epoch exception. Solver, early-stopping flag, random seed, and verbosity remain fixed compatibility or infrastructure constants.

The deeper architecture has more room to capture nonlinear structure, but it also has higher optimization cost and a larger risk of unstable convergence or unnecessary capacity relative to the size of the benchmark target space.

## 8. Prediction workflow

Prediction rebuilds the benchmark feature matrix, applies the stored scalers, produces raw measured-output predictions from the fitted MLP, inverse-transforms the targets when needed, and then applies the shared measured-space projection whenever the A matrix is active.

## 9. Limitations and expected failure modes

The deep network can overfit or converge less reliably than the medium architecture, especially because the repository intentionally avoids estimator-owned validation splitting. It is best interpreted as a stress-test of higher ANN capacity rather than the preferred production architecture.

## 10. References

Rumelhart, D. E., Hinton, G. E., and Williams, R. J. Learning Representations by Back-Propagating Errors. Nature, 323, 533-536, 1986.

Kingma, D. P., and Ba, J. Adam: A Method for Stochastic Optimization. International Conference on Learning Representations, 2015.

Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. MIT Press, 2016.