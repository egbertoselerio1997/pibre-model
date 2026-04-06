# Shallow Neural-Network Regressor Summary

## 1. Title and model summary

The ann_shallow_regressor module implements a measured-space surrogate based on sklearn.neural_network.MLPRegressor with one hidden layer of width 64. The repository treats it as the smallest neural benchmark in the ANN family, applies repository-managed scaling, keeps notebook-owned split governance intact, and projects raw measured-output predictions with the notebook-derived A matrix.

## 2. Background and use case

This shallow network is the low-capacity ANN baseline for the repository. Its role is to test whether a single nonlinear hidden layer is already sufficient to model the dominant operational and influent interactions without the additional optimization burden of deeper networks.

## 3. Mathematical definition

For input vector $x$, the shallow network computes

$$
h = \phi(W_1 x + b_1), \qquad \hat{y}_{raw} = W_2 h + b_2
$$

where $\phi$ is the ReLU activation. The repository then applies the shared measured-space projection to obtain the final projected prediction whenever the A matrix is active.

## 4. Inputs, outputs, and assumptions

Inputs are the shared benchmark features built from operational variables and fractional influent-state variables. Outputs are the measured effluent composites.

The network assumes repository-managed feature and target scaling. It also assumes early_stopping remains false, because the repository rules require train-test and Optuna subset management to remain in the notebook rather than inside the model estimator.

## 5. Implementation used in this repository

Implementation is in src/models/ml/ann_shallow_regressor.py.

The model file defines the estimator builder and standard wrapper functions, while the shared training utilities handle scaling, evaluation, projection, artifact persistence, and Optuna execution from the notebook.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is a single-hidden-layer multilayer perceptron with hidden_layer_sizes = (64,), ReLU activation, and Adam optimization. This is the lightest ANN benchmark in the repository and is intended to probe whether modest nonlinear capacity is enough for the benchmark mapping.

## 7. Training or optimization notes

The notebook-owned Optuna branch now tunes all non-fixed shallow-ANN controls used in this benchmark: activation function, regularization strength, batch size, learning rate, tolerance, and shuffle behavior. The hidden-layer shape stays fixed by design for architecture comparability, and max_iter remains fixed as the repository epoch exception. Solver, early-stopping flag, random seed, and verbosity also remain fixed compatibility or infrastructure constants.

## 8. Prediction workflow

Prediction rebuilds the benchmark feature matrix, applies the stored scalers, generates raw measured-output predictions from the fitted MLP, inverse-transforms the targets when needed, and then applies the invariant projection if the measured-output null space is non-trivial.

## 9. Limitations and expected failure modes

The shallow network can underfit if the operational-plus-fractional mapping requires richer hierarchical interactions than one hidden layer can express. It is also sensitive to learning-rate choice and iteration budget, especially because the repository deliberately avoids estimator-owned validation splitting.

## 10. References

Rumelhart, D. E., Hinton, G. E., and Williams, R. J. Learning Representations by Back-Propagating Errors. Nature, 323, 533-536, 1986.

Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. MIT Press, 2016.

The repository reference note in references/doc.md describes the measured-space projection framework used after raw regression.