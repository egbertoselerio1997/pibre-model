# Support Vector Regressor Summary

## 1. Title and model summary

The svr_regressor module implements a measured-space surrogate based on sklearn.svm.SVR. The repository fits one support vector regressor per measured target using a multi-output wrapper, applies feature scaling through the shared preprocessing helpers, and then projects the raw predictions onto the invariant subspace defined by the notebook-derived A matrix.

## 2. Background and use case

Support Vector Regression, or SVR, is a kernel-based method that seeks a function with controlled complexity while tolerating small prediction errors inside an epsilon-insensitive tube. It is valuable in this repository because it provides a fundamentally different nonlinear surrogate from tree ensembles and from PIBRe.

## 3. Mathematical definition

For one target the SVR problem minimizes a regularized objective of the form

$$
\min_{w, b, \xi_i, \xi_i^{*}} \frac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^{N} (\xi_i + \xi_i^{*})
$$

subject to the epsilon-insensitive constraints

$$
\begin{aligned}
y_i - \langle w, \phi(x_i) \rangle - b &\le \varepsilon + \xi_i, \\
\langle w, \phi(x_i) \rangle + b - y_i &\le \varepsilon + \xi_i^{*},
\end{aligned}
$$

with $\xi_i, \xi_i^{*} \ge 0$. The mapping $\phi$ is induced by the selected kernel. In this repository one wrapped SVR is trained per target and the raw measured-output vector is then projected with the same A-matrix projection used elsewhere.

## 4. Inputs, outputs, and assumptions

Inputs are the measured-space operating and influent variables used throughout the repository. Outputs are the eight measured effluent composites.

An important assumption is that feature scaling is enabled, because distance-based kernel methods are highly sensitive to differences in feature magnitude. The implementation also assumes that the measured-output basis remains aligned with the A matrix used in projection.

## 5. Implementation used in this repository

Implementation is in src/models/ml/svr_regressor.py.

The repository relies on shared helpers for dataset construction, splitting, scaling, tuning, evaluation, and persistence. The SVR module contributes the model-specific estimator builder and the standard train, tune, predict, and pipeline wrappers required by the repository contract.

## 6. Architecture details and adopted standard architecture name

The adopted architecture is epsilon-insensitive support vector regression with configurable kernel choice. The repository wraps sklearn.svm.SVR in sklearn.multioutput.MultiOutputRegressor so one support vector model is trained per measured target.

## 7. Training or optimization notes

Optuna tunes the regularization parameter $C$, the epsilon tube width, the kernel choice, the gamma setting, and the polynomial degree parameter when relevant. Because SVR is scale-sensitive, feature standardization is enabled by default for this model in config/params.json.

## 8. Prediction workflow

Prediction loads the saved bundle, reconstructs the measured-space feature matrix, standardizes the features with the stored scaler, generates raw outputs from the wrapped SVR models, projects those outputs to the invariant subspace, and returns aligned raw and projected predictions.

## 9. Limitations and expected failure modes

SVR can become computationally expensive as the sample count grows, and prediction quality is sensitive to kernel hyperparameters and scaling choices. Because targets are fit independently before projection, any shared target structure must be recovered indirectly through the final measured-space projection.

## 10. References

Smola, A. J., and Scholkopf, B. A Tutorial on Support Vector Regression. Statistics and Computing, 14, 199-222, 2004.

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

The repository reference note in references/doc.md describes the measured-space projection framework used after raw regression.