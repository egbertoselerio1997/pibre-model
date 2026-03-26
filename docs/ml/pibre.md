# PIBRe Model Summary

Physics-Informed Bilinear Regression (PIBRe) is a surrogate-modeling workflow that combines an unconstrained bilinear regression layer with a hard projection step that enforces conservation laws in measured-output space. In this repository the model is trained against the measured ASM1 outputs produced by the simulation notebook and is constrained by the matrix $A$ computed from the null space of the macroscopic stoichiometric matrix.

## Background And Use Case

Activated-sludge surrogate models must balance two competing goals. They should be expressive enough to fit nonlinear relationships between operating variables, influent conditions, and effluent measurements, but they should also respect physical conservation laws. A purely statistical regressor can fit held-out data well while still violating mass-balance relationships. PIBRe addresses that gap by learning an unconstrained prediction and then projecting it onto the physically admissible affine subspace.

This repository uses PIBRe for the measured-output layer of the ASM1 simulation workflow. The measured outputs are the composite measures COD, TSS, VSS, TN, and TP. The model predicts those outputs only, because the conservation matrix $A$ is defined in that measured-output space.

## Mathematical Definition

Let $x \in \mathbb{R}^{M}$ denote the feature vector assembled from the configured independent columns, and let $C_{in} \in \mathbb{R}^{K}$ denote the influent concentration vector expressed in the same measured-output space as the targets. The raw PIBRe prediction is

$$
C_{raw} = W_L x + b + x^T W_B x,
$$

where $W_L \in \mathbb{R}^{K \times M}$ is a linear weight matrix, $b \in \mathbb{R}^{K}$ is a bias vector, and $W_B \in \mathbb{R}^{K \times M \times M}$ is a bilinear tensor implemented in the repository through a bilinear PyTorch layer.

The conservation matrix $A \in \mathbb{R}^{C \times K}$ is computed from the null space of the macroscopic stoichiometric matrix. The physically admissible predictions satisfy

$$
A C^{*} = A C_{in}.
$$

The repository enforces this relation by orthogonally projecting $C_{raw}$ onto the constraint set:

$$
C^{*} = C_{raw} - A^T (A A^T)^{-1} A (C_{raw} - C_{in}).
$$

In practice the inverse is implemented as a pseudo-inverse to preserve numerical robustness.

## Inputs, Outputs, And Assumptions

The extracted PIBRe module consumes the following inputs:

1. A training or test dataset following the metadata contract emitted by the ASM1 simulation workflow.
2. The metadata dictionary containing independent columns, state columns, and measured-output columns.
3. The ASM1 composition matrix used to map influent state variables into measured-output space.
4. The conservation matrix $A$ derived from the current simulation setup.
5. A hyperparameter dictionary loaded from `config/params.json`.

The model outputs measured effluent predictions aligned to the dataset rows. During training and evaluation the repository reports both raw and projected predictions so the effect of the conservation enforcement can be inspected explicitly.

The main assumption is that the physically relevant conservation laws are sufficiently represented by the measured-output-space null-space basis. The implementation therefore constrains only the measured outputs, not the full latent ASM1 state vector.

## Implementation Used In This Repository

The implementation lives in `src/models/ml/pibre.py`. The file contains the model definition, Optuna tuning routine, training routine, serialization logic, prediction entry point, and evaluation helper needed by the repository’s machine-learning contract.

The workflow is as follows:

1. Select the PIBRe features from `metadata["independent_columns"]`.
2. Select the PIBRe targets from the measured-output subset `Out_{name}` for each configured measured output.
3. Construct the measured influent composite vector by multiplying the influent state matrix by the transpose of the ASM1 composition matrix.
4. Split the dataset into train and test partitions.
5. Use Optuna on a capped subset of the training rows to tune the learning rate, $L_1$ penalty, weight decay, and gradient-clipping threshold.
6. Refit the model on the full training partition using the best tuned parameters.
7. Save the trained model, scaler, column order, composition matrix, and $A$ matrix to a pickle artifact under the configured results pattern.
8. Evaluate both the raw and projected predictions on the held-out set using the standard metrics required by the repository.

## Architecture Details And Adopted Standard Architecture Name

The repository implementation is best described as a physics-informed bilinear regression network. It is not a deep multilayer architecture. Instead, it consists of two parallel algebraic components:

1. A linear map from features to measured outputs.
2. A bilinear map that captures pairwise feature interactions.

The projection layer is deterministic and parameter free once $A$ has been specified. The layer is implemented inside the forward pass of the PyTorch model so that both raw and projected predictions are available from one call.

## Training Or Optimization Notes

The model is trained with mean-squared-error loss evaluated on the projected output, augmented by an $L_1$ penalty on both the linear and bilinear weights. The projection ensures that the optimization sees a physically compliant prediction, while the $L_1$ term promotes a sparse bilinear representation.

Optuna uses a Tree-structured Parzen Estimator sampler and a median-based pruner. This reduces unnecessary trials while keeping the notebook workflow tractable. The repository caps the tuning subset size explicitly so tuning remains practical inside `main.ipynb`.

During evaluation the model reports the repository-standard metrics:

1. $R^2$
2. MSE
3. RMSE
4. MAE
5. MAPE

The evaluation also reports the mean mass-balance violation norm for both the raw and projected predictions.

## Prediction Workflow

Prediction starts from a saved `.pkl` model artifact. The artifact contains the fitted scaler, learned network weights, column order, composition matrix, and conservation matrix. The predict entry point restores those objects, rebuilds the network with the saved input and output dimensions, scales the test features using the saved scaler, reconstructs the measured influent composites from the test dataset, and returns projected measured-output predictions aligned to the input rows.

## Limitations And Expected Failure Modes

Several limitations should be kept in mind.

1. The model constrains measured outputs only. It does not enforce conservation directly on the latent ASM1 states.
2. The bilinear model remains a low-capacity surrogate relative to a full mechanistic solver, so highly nonlinear regimes may still be approximated poorly even when the prediction is physically projected.
3. If the measured-output composition mapping is changed, the saved model artifact becomes incompatible with earlier notebook runs unless the model is retrained.
4. If the measured-output null-space basis is ill-conditioned, the projection can amplify numerical noise even though the pseudo-inverse stabilizes the computation.
5. Because the model artifact stores the column order explicitly, prediction will fail if the required input columns are missing from the supplied test dataset.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. C. M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3.
2. Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. Optuna: A Next-generation Hyperparameter Optimization Framework.
3. Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library.