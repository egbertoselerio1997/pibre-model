# Non-Negative Invariant-Constrained Second-Order Regression (ICSOR) with Coupled Quadratic Programs for Activated Sludge Component Prediction

## Abstract

This article presents a coupled mathematical-programming formulation of invariant-constrained second-order regression (ICSOR) for steady-state activated-sludge surrogate modeling. The model accepts operational variables and influent activated-sludge-model (ASM) component states expressed on a common concentration-equivalent linear basis, and it predicts effluent ASM component states in that same basis. ICSOR remains defined natively in ASM component space. It is trained to predict ASM component states directly, and only those component states. If measured composite variables such as total COD, total nitrogen, total phosphorus, TSS, or VSS are needed, they are computed afterward by an external composition matrix. The collapse into measured-output space is therefore not part of the model itself.

The surrogate is split into two layers. A second-order feature map produces a feature-driven driver vector $r(u, c_{in}) = B \phi(u, c_{in})$. An estimated coupling matrix $\Gamma$ then links ASM components through the coupled system matrix $R = I_F - \Gamma$. The present article does not constrain $B$ to be entrywise non-negative. Instead, fitted predictions during training and deployed predictions at inference are obtained by solving optimization problems that enforce componentwise nonnegativity directly in prediction space. Because both $B$ and $\Gamma$ are estimated, the training problem is jointly nonconvex but retains blockwise convex structure. Conditional on trained parameters, deployment-time inference remains a convex linear program in the predicted component vector and its auxiliary correction variable.

Stoichiometric conservation remains central, and in this article it is handled through invariant-residual penalties in the fitted-prediction program used during training and as a hard linear equality in the deployed inference program. The resulting ICSOR formulation is a coupled component-space surrogate with optimization-based training, optimization-based inference, nonnegative fitted and deployed predictions, invariant-aware training, exact invariant preservation inside the deployed inference problem, and external measurement collapse. This article develops that theory carefully, states what it guarantees, and revises the discussion, estimation logic, uncertainty treatment, and limitations around that contract.

In this repository implementation, coefficient estimation now supports two selectable training mechanisms under the same model contract: the original recursive coupled-QP block-update routine and an Adam-based optimizer that keeps the same fit, invariant, and coupled-system terms while using Lasso regularization for $B$ and $\Gamma$. The deployment-time constrained inference stage remains unchanged. In the Adam path, the returned training state is reconciled back to the prediction-space formulation before persistence by conditioning $\Gamma$ into the admissible deployed set and re-solving the fitted nonnegative $\widehat C$ subproblem for the exposed $(B, \Gamma, \widehat C)$ triple. This means the training algorithm can be switched without changing the native prediction space, the projection contract, or the external measured-output comparison protocol.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to represent a quasi-steady operating condition: the operating variables, influent composition, and effluent response are treated as effectively time-invariant over the control volume being modeled for the sampling window of interest.

The usual difficulty is that wastewater physics is written in an ASM component basis, whereas reporting and plant supervision are often performed in measured aggregate variables. The stoichiometric matrix acts on component states such as substrate, biomass, ammonium, nitrate, phosphate, dissolved oxygen, or alkalinity. Plant dashboards, however, often monitor total COD, total nitrogen, total phosphorus, TSS, or VSS. A model trained only on those aggregates can fit measured outputs while implying an impossible redistribution of the underlying ASM components.

ICSOR is formulated here to remove that mismatch at the model-definition level. The model input is split into

1. an operational block $u$, and
2. an influent ASM component-state block $c_{in}$.

The model output is the effluent ASM component-state vector $\hat c$ in the same ASM basis as the input. This means the native prediction target is not a measured composite. It is the component vector itself. If measured composites are needed for reporting, optimization, or comparison to plant observations, they are obtained afterward by an external composition matrix:

$$
\hat y_{ext} = I_{comp} \hat c.
$$

That external collapse is deliberate. It keeps the learned map in the same space in which stoichiometric invariants are defined and in which the non-negativity claim is made.

This article answers one precise question:

> Given a steady-state influent ASM component-state vector and a steady-state operating condition, what effluent ASM component-state vector should be predicted if the surrogate is second order, its ASM components are explicitly coupled through coefficients that must be estimated from data, training is posed as a coupled mathematical program, deployment itself is posed as a constrained inference program, and non-negative predictions must be enforced directly in prediction space while stoichiometric invariants are penalized during training and enforced exactly at deployment?

The theory in this article is restricted to steady-state reactor-block prediction. It does not replace a dynamic activated-sludge simulator. It also does not claim that non-negative ASM component predictions are automatically kinetically or biologically realizable. Its narrower claim is that the surrogate should live in component space, be fitted in component space, produce deployed predictions through a constrained mathematical program, and treat measured-output collapse as an external downstream calculation.

Two additional consequences follow immediately from this choice.

1. Training requires effluent ASM component-state targets, either observed directly from a simulator or reconstructed upstream from measured data.
2. Deployment requires solving a small constrained optimization problem for each new sample rather than only evaluating a polynomial map in closed form.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit represented by quasi-steady samples. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes bypass streams, gas stripping, chemical dosing, sludge wastage, or any other transport mechanism that changes the component inventory across the chosen boundary.

Throughout the article, the ASM state vectors are treated as non-negative component-state coordinates on a common concentration-equivalent linear basis. The invariant relations and stoichiometric algebra below are written for that linear basis. They are not claimed to remain unchanged under arbitrary normalization, literal fractions, or other nonlinear reparameterizations of the ASM states.

The theory therefore applies only after the modeler has fixed the following items:

1. the reactor or process block being represented,
2. the ASM component basis used to describe material composition,
3. the stoichiometric matrix associated with that basis, and
4. the composition matrix used later for external reporting in measured-output space.

### 2.2 Why two spaces are still relevant

ICSOR is trained and deployed in ASM component space, but two spaces still matter conceptually.

1. **ASM component space.** This is the native model space. Stoichiometric invariants, componentwise non-negativity, coefficient estimation, and scientific interpretation all live here.
2. **Measured composite space.** This is an external reporting space obtained by applying a fixed composition matrix after prediction.

The distinction is important because two different ASM component states can collapse to the same measured aggregate outputs. If the model is trained only in measured-output space, that ambiguity is hidden. If the model is trained in component space and measured collapse is treated externally, that ambiguity remains visible and auditable.

### 2.3 Notation

Single-sample vectors are written as column vectors. Dataset matrices are defined later with samples stored by rows.

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $u$ | $\mathbb{R}_{+}^{M_{op}}$ | Operational input vector, for example hydraulic retention time, aeration intensity, recycle ratio, or other manipulated variables, expressed on a positive coordinate system |
| $c_{in}$ | $\mathbb{R}_{+}^{F}$ | Influent ASM component-state vector on the adopted concentration-equivalent basis |
| $c_{out}$ | $\mathbb{R}_{+}^{F}$ | True effluent ASM component-state vector on the adopted concentration-equivalent basis |
| $\hat c$ | $\mathbb{R}_{+}^{F}$ | Predicted effluent ASM component-state vector |
| $y_{ext}$ | $\mathbb{R}^{K}$ | External measured composite vector computed from a component vector |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM components to measured composite variables |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction-progress vector expressed on the same concentration-equivalent basis as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank invariant matrix satisfying $A \nu^T = 0$ |
| $\phi(u, c_{in})$ | $\mathbb{R}_{+}^{D}$ | Non-negative second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $B$ | $\mathbb{R}^{F \times D}$ | Coefficient matrix of the second-order feature-driven driver |
| $\Gamma$ | $\mathbb{R}^{F \times F}$ | Estimated component-coupling matrix; estimation constrains $R = I_F - \Gamma$ to remain nonsingular |
| $R$ | $\mathbb{R}^{F \times F}$ | Coupled-system matrix $I_F - \Gamma$ |
| $r(u, c_{in})$ | $\mathbb{R}^{F}$ | Feature-driven driver vector defined by $r(u, c_{in}) = B \phi(u, c_{in})$ |
| $\widehat C$ | $\mathbb{R}_{+}^{N \times F}$ | Fitted prediction matrix used in the training mathematical program |
| $\lambda_{inv}$ | $\mathbb{R}_{+}$ | Invariant-residual penalty weight in the training objective |
| $\lambda_{sys}$ | $\mathbb{R}_{+}$ | Coupled-system consistency penalty weight in the training objective |

The external measured variables are defined by the linear map

$$
y_{ext} = I_{comp} c.
$$

When this map is applied to the true effluent state, it yields the true measured composites. When it is applied to $\hat c$, it yields the externally reported prediction $\hat y_{ext} = I_{comp} \hat c$. This reporting step is outside the model: ICSOR itself predicts only $\hat c$.

## 3. Modeling Assumptions

The framework rests on the following assumptions. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a quasi-steady input-output condition rather than a dynamic trajectory.
2. **Fixed ASM basis.** The ASM component basis and the associated stoichiometric matrix are fixed before estimation begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement.
4. **Non-negative input coordinates.** The operational variables and influent ASM component states are supplied on a non-negative, physically meaningful coordinate system, so that every monomial used in the feature map is non-negative.
5. **Direct component-space targets.** The model is trained against effluent ASM component states, not against measured composite outputs.
6. **Second-order driver class.** The feature-driven driver is a partitioned second-order polynomial model with linear, quadratic, and operation-loading interaction terms.
7. **Estimated coupling matrix.** The ASM component equations are coupled by a matrix $\Gamma$ that is estimated from data under structural restrictions that keep $R = I_F - \Gamma$ nonsingular.
8. **No coefficient sign restriction.** The coefficient matrix $B$ is not constrained to be entrywise non-negative.
9. **Coupled mathematical-programming training.** Parameter estimation is posed as a jointly nonconvex but blockwise structured mathematical program in the driver coefficients, the coupling coefficients, and the fitted component predictions.
10. **Mathematical-programming inference.** Deployment-time prediction is itself posed as a constrained optimization problem in the predicted component vector, conditional on the trained parameters.
11. **Prediction-space nonnegativity.** Non-negativity is imposed directly on fitted and deployed predictions rather than on coefficients.
12. **Invariant-aware training with exact deployed conservation.** Stoichiometric invariants are handled through residual penalties in ASM component space during training and through hard linear equality constraints during deployment-time inference.
13. **External composition map.** The collapse from ASM component states to measured composites is an external calculation performed after prediction through $I_{comp}$.
14. **Composite-sign scope.** Non-negative component predictions imply non-negative measured composites only when the relevant rows of $I_{comp}$ are entrywise non-negative.
15. **Target availability.** The effluent ASM component states used for training are assumed available directly or through an upstream state-reconstruction step outside the present model.

These assumptions matter because they narrow the scientific claim. The model guarantees non-negative fitted predictions and non-negative deployed predictions because the corresponding optimization problems impose those constraints directly in prediction space. It also guarantees exact preservation of the adopted stoichiometric invariants inside the deployed inference program. During training, invariant mismatch is penalized in ASM component space rather than eliminated by a hard equality constraint at every fitted sample. The model does not claim that the unconstrained driver $B \phi(u, c_{in})$ is non-negative, and it does not by itself guarantee exact kinetics, thermodynamic feasibility, or global convexity of the joint estimation problem in $(B, \Gamma, \widehat C)$.

## 4. Stoichiometric Structure and Conserved Quantities

### 4.1 From stoichiometric reactions to component-state change

Let $\nu \in \mathbb{R}^{R \times F}$ be the stoichiometric matrix written in the adopted ASM component basis. For one steady-state sample, define the net reaction-progress vector $\xi \in \mathbb{R}^{R}$ so that

$$
c_{out} - c_{in} = \nu^T \xi.
$$

This equation says that the net change in the effluent ASM component state is a linear combination of the reaction stoichiometries. The entries of $\xi$ need not be observed. They summarize the net progression of the modeled reactions over the chosen control volume after scaling into the same concentration-equivalent basis used for $c_{out} - c_{in}$.

### 4.2 Invariant relations implied by the stoichiometric matrix

Introduce a full-row-rank matrix $A \in \mathbb{R}^{q \times F}$ whose rows span the invariant space:

$$
A \nu^T = 0.
$$

Multiplying the stoichiometric change relation by $A$ gives

$$
A(c_{out} - c_{in}) = A \nu^T \xi = 0,
$$

so the conserved quantities satisfy

$$
A c_{out} = A c_{in}.
$$

Each row of $A$ represents one independent conserved combination of ASM components under the adopted stoichiometric model and system boundary. Depending on the basis, these may correspond to COD-equivalent, nitrogen-equivalent, phosphorus-equivalent, charge-related, or other conserved pools. The meaning comes from the chosen stoichiometric model; it is not created by the regression model.

### 4.3 Why the basis of $A$ is not unique

The matrix $A$ is not unique. If $R_A \in \mathbb{R}^{q \times q}$ is invertible, then $\widetilde A = R_A A$ defines the same invariant set because

$$
\widetilde A c = \widetilde A c_{in}
\quad \Longleftrightarrow \quad
R_A A c = R_A A c_{in}
\quad \Longleftrightarrow \quad
A c = A c_{in}.
$$

Thus, the physics is carried by the row space of $A$, not by one particular numerical basis.

### 4.4 Minimal worked example

Consider two components, $c_1$ and $c_2$, and one reaction that converts $c_1$ into $c_2$ without net loss:

$$
\nu = \begin{bmatrix}
-1 & 1
\end{bmatrix}.
$$

Then one admissible invariant matrix is

$$
A = \begin{bmatrix} 1 & 1 \end{bmatrix},
$$

so the invariant relation is

$$
c_{out,1} + c_{out,2} = c_{in,1} + c_{in,2}.
$$

The feasible effluent states lie on the non-negative line segment defined by this equality. The purpose of ICSOR is to learn a coupled prediction rule whose fitted and deployed solutions select a non-negative point directly in component space, not to predict a measured aggregate first and only later infer where the underlying components should have been.

### 4.5 ASM-flavored miniature example before external composition

Suppose the component vector is

$$
c = \begin{bmatrix} S_S \\ X_S \\ S_{NH} \end{bmatrix},
$$

where $S_S$ is soluble substrate, $X_S$ is particulate substrate, and $S_{NH}$ is ammonium. Let one simplified reaction convert soluble substrate into particulate substrate without changing ammonium:

$$
\nu = \begin{bmatrix}
-1 & 1 & 0
\end{bmatrix}.
$$

One admissible invariant matrix is therefore

$$
A = \begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Suppose the influent state is

$$
c_{in} = \begin{bmatrix} 10 \\ 10 \\ 5 \end{bmatrix}
$$

and the constrained deployment solve predicts

$$
\hat c = \begin{bmatrix} 8 \\ 12 \\ 5 \end{bmatrix}.
$$

This prediction is componentwise non-negative. If one wishes to report total COD and ammonium externally, the composition matrix may be chosen as

$$
I_{comp} = \begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix},
$$

which yields

$$
\hat y_{ext} = I_{comp} \hat c = \begin{bmatrix} 20 \\ 5 \end{bmatrix}.
$$

The key point is not the arithmetic. It is the ordering. The model predicts the ASM component states first. Only after that step are measured aggregates obtained. If another reporting convention is later desired, one changes $I_{comp}$ externally; the trained component-space model does not change.

## 5. Coupled Second-Order Surrogate in ASM Component Space

### 5.1 Why the input is partitioned

In activated-sludge systems, operating conditions and influent component states play different physical roles.

1. Operating variables such as hydraulic retention time, dissolved-oxygen setpoint, or recycle settings alter the process environment.
2. Influent ASM component states describe the material inventory entering that environment.

Treating those two groups as interchangeable predictors hides an important engineering distinction. ICSOR therefore partitions the input into an operational block $u$ and an influent ASM component block $c_{in}$.

### 5.2 Feature map

We define the second-order feature map

$$
\phi(u, c_{in}) =
\begin{bmatrix}
1 \\
u \\
c_{in} \\
u \otimes u \\
c_{in} \otimes c_{in} \\
u \otimes c_{in}
\end{bmatrix}
\in \mathbb{R}_{+}^{D},
$$

where $\otimes$ denotes the Kronecker product. We use the conventions

$$
u \otimes u = \operatorname{vec}(u u^T), \qquad
c_{in} \otimes c_{in} = \operatorname{vec}(c_{in} c_{in}^T), \qquad
u \otimes c_{in} = \operatorname{vec}(u c_{in}^T).
$$

The resulting feature dimension is

$$
D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F.
$$

Because the inputs are non-negative and every feature is either a positive constant or a product of non-negative inputs, the full feature vector is non-negative componentwise.

### 5.3 Feature-driven driver and coupled component system

The second-order feature block does not produce the final component prediction directly. Instead, it produces a feature-driven driver vector

$$
r(u, c_{in}) = B \phi(u, c_{in}),
$$

where $B \in \mathbb{R}^{F \times D}$ is unrestricted in sign.

Cross-component dependence is introduced by an estimated coupling matrix $\Gamma \in \mathbb{R}^{F \times F}$. During estimation, $\Gamma$ is restricted to an admissible set that preserves interpretability and keeps the coupled system usable; common examples include $\operatorname{diag}(\Gamma) = 0$ together with additional conditioning or stability restrictions that keep $R = I_F - \Gamma$ nonsingular. Define the coupled-system matrix

$$
R = I_F - \Gamma.
$$

The ideal noiseless coupled relation is then

$$
R c = r(u, c_{in}).
$$

For any admissible $\Gamma$ with nonsingular $R$, this may be written formally as

$$
c = R^{-1} r(u, c_{in}).
$$

For training and deployment, however, ICSOR does not rely on this unconstrained expression as the final prediction rule. Instead, it treats the coupled relation as a structural target inside mathematical programs that also enforce prediction nonnegativity, penalize invariant mismatch during training, and enforce exact invariant equalities during deployment-time inference. Because $\Gamma$ is estimated jointly with $B$, the coupled relation enters training through a residual term rather than as a fixed known map.

At dataset level, the fitted prediction matrix $\widehat C$ and the feature-driven driver matrix $\Phi B^T$ are linked through the same coupled structure:

$$
\widehat C (I_F - \Gamma)^T \approx \Phi B^T.
$$

If the coupled residual vanishes, this relation becomes exact.

### 5.4 Why non-negativity is enforced in prediction space

Once coefficient sign restrictions are removed, the nonnegative feature map no longer implies nonnegative drivers or nonnegative unconstrained component predictions. Even when $\phi(u, c_{in}) \ge 0$, the vector $r(u, c_{in}) = B \phi(u, c_{in})$ can have mixed signs, and the unconstrained coupled solution $R^{-1} r(u, c_{in})$ can also have mixed signs.

For that reason, the final ICSOR predictor is not defined as direct evaluation of $B \phi(u, c_{in})$ or even as direct evaluation of $R^{-1} B \phi(u, c_{in})$. Non-negativity is instead enforced where it is scientifically needed: on the fitted component predictions during training and on the deployed component predictions during inference. In this article, those two objects are produced by constrained mathematical programs with explicit nonnegativity constraints.

### 5.5 Interpretation of the driver and the coupling matrix

The coupled formulation separates two roles that were previously folded into one coefficient matrix.

1. The matrix $B$ maps operating conditions and influent composition into a feature-driven driver vector.
2. The matrix $\Gamma$ mediates direct coupling among ASM components.
3. The invariant-aware training term and the deployed invariant equality constraints add a second structural layer tied to stoichiometric conservation.

This separation changes interpretation. The blocks of $B$ still describe how the second-order feature basis pushes the latent driver, but the final component prediction is mediated through the coupled system and, at deployment, through the constrained inference solve. Because $\Gamma$ is estimated jointly with $B$, interpretation must also account for partial confounding between driver effects and coupling effects: without structural restrictions or regularization on $\Gamma$, some behavior attributed to direct coupling can be absorbed by $B$ and vice versa. Because $B$ is not sign-constrained, the revised model is no longer monotone by construction in the native feature basis. That loss of monotonicity is intentional: it trades the old sign-restricted guarantee for a more expressive but harder estimation problem while moving the non-negativity guarantee into prediction space.

## 6. Invariant-Aware Coupled Mathematical Programming

### 6.1 Dataset-level component-space regression objects

Let $N$ steady-state samples be available, and store samples by rows:

$$
\Phi =
\begin{bmatrix}
\phi(u_1, c_{in,1})^T \\
\phi(u_2, c_{in,2})^T \\
\vdots \\
\phi(u_N, c_{in,N})^T
\end{bmatrix}
\in \mathbb{R}_{+}^{N \times D},
$$

$$
C_{in} =
\begin{bmatrix}
c_{in,1}^T \\
c_{in,2}^T \\
\vdots \\
c_{in,N}^T
\end{bmatrix}
\in \mathbb{R}_{+}^{N \times F},
$$

$$
C_{out} =
\begin{bmatrix}
c_{out,1}^T \\
c_{out,2}^T \\
\vdots \\
c_{out,N}^T
\end{bmatrix}
\in \mathbb{R}_{+}^{N \times F}.
$$

For a given coefficient matrix $B$, the feature-driven driver matrix is

$$
D(B) = \Phi B^T.
$$

The training mathematical program also introduces a fitted prediction matrix $\widehat C \in \mathbb{R}_{+}^{N \times F}$ and estimates a coupling matrix $\Gamma$ in an admissible set $\mathcal G \subseteq \mathbb{R}^{F \times F}$.

### 6.2 Coupled training objective in prediction space

When $\Gamma$ is estimated, the natural component-space fit minimizes squared component error, penalizes invariant residuals, penalizes mismatch between the fitted prediction matrix and the coupled driver relation, and regularizes the driver and coupling terms enough to control identifiability and conditioning. Let $\lambda_{inv}, \lambda_B, \lambda_\Gamma \ge 0$, and let $\mathcal R_B$ and $\mathcal R_\Gamma$ denote optional regularizers on $B$ and $\Gamma$.

$$
(\widehat B, \widehat \Gamma, \widehat C)
=
\arg\min_{B \in \mathbb{R}^{F \times D},\; \Gamma \in \mathcal G,\; \widehat C \in \mathbb{R}_{+}^{N \times F}}
J(B, \Gamma, \widehat C),
$$

with

$$
J(B, \Gamma, \widehat C)
=
\left\|
C_{out} - \widehat C
\right\|_F^2
+ \lambda_{inv}
\left\|
(\widehat C - C_{in}) A^T
\right\|_F^2
+ \lambda_{sys}
\left\|
\widehat C (I_F - \Gamma)^T - \Phi B^T
\right\|_F^2
+ \lambda_B \mathcal R_B(B)
+ \lambda_\Gamma \mathcal R_\Gamma(\Gamma).
$$

The first term fits the effluent ASM component states directly. The second term penalizes violations of the stoichiometric invariant relations in the fitted prediction matrix. The third term enforces consistency between the fitted prediction matrix and the coupled second-order driver. The last two terms stabilize the driver and coupling estimates when regularization is used. The hard nonnegativity constraint acts directly on $\widehat C$, while the admissible-set restriction acts on $\Gamma$.

Repository note. The `recursive_qp` implementation instantiates $\mathcal R_B$ and $\mathcal R_\Gamma$ as ridge penalties weighted by `lambda_B` and `lambda_gamma`. The `adam_lasso` implementation keeps the same first three terms but uses L1 penalties weighted by `lasso_lambda_B` and `lasso_lambda_gamma` during the gradient phase, then recomputes the returned fitted prediction matrix with the exact nonnegative $\widehat C$ subproblem for the final admissible $\Gamma$.

### 6.3 Blockwise optimization when $\Gamma$ is estimated

When $\Gamma$ is estimated jointly with $B$ and $\widehat C$, the coupled residual term is

$$
\widehat C (I_F - \Gamma)^T - \Phi B^T
=
(\widehat C - \Phi B^T) - \widehat C \Gamma^T,
$$

which is bilinear in $(\widehat C, \Gamma)$. After squaring, the full objective is therefore jointly nonconvex. The useful structural fact is instead blockwise convexity.

For fixed $(\Gamma, \widehat C)$, the $B$-subproblem is

$$
\widehat B(\Gamma, \widehat C)
=
\arg\min_{B \in \mathbb{R}^{F \times D}}
\lambda_{sys}
\left\|
\widehat C (I_F - \Gamma)^T - \Phi B^T
\right\|_F^2
+ \lambda_B \mathcal R_B(B),
$$

which is a convex least-squares or ridge-type problem when $\mathcal R_B$ is convex.

For fixed $(B, \widehat C)$, define $g = \operatorname{vec}(\Gamma^T)$. Using

$$
\operatorname{vec}(\widehat C \Gamma^T)
=
(I_F \otimes \widehat C) g,
$$

the $\Gamma$-subproblem becomes

$$
\widehat g(B, \widehat C)
=
\arg\min_{g \in \mathcal G_{vec}}
\lambda_{sys}
\left\|
\operatorname{vec}(\widehat C - \Phi B^T)
- (I_F \otimes \widehat C) g
\right\|_2^2
+ \lambda_\Gamma \widetilde{\mathcal R}_\Gamma(g),
$$

which is a convex quadratic program whenever the admissible set $\mathcal G$ and the regularizer are convex after vectorization.

For fixed $(B, \Gamma)$, the $\widehat C$-subproblem is

$$
\widehat C(B, \Gamma)
=
\arg\min_{\widehat C \in \mathbb{R}_{+}^{N \times F}}
\left\|
C_{out} - \widehat C
\right\|_F^2
+ \lambda_{inv}
\left\|
(\widehat C - C_{in}) A^T
\right\|_F^2
+ \lambda_{sys}
\left\|
\widehat C (I_F - \Gamma)^T - \Phi B^T
\right\|_F^2,
$$

which is a convex quadratic program in the fitted predictions. Thus the appropriate training statement is not global convexity but blockwise convexity together with local stationarity of the final estimate.

One explicit implementation is cyclic block-coordinate descent with restarts. Choose a regression-window size $W \ge 2$, a slope tolerance $\varepsilon_{reg} \ge 0$, a maximum outer-iteration count $T_{max}$, and a restart count $S$. For restart index $s = 1, \ldots, S$, initialize an admissible coupling matrix $\Gamma^{(0,s)} \in \mathcal G$ such as the zero matrix or a sparse prior guess, initialize a nonnegative fitted prediction matrix $\widehat C^{(0,s)}$ such as $\max(C_{out}, 0)$ or the solution of the $\widehat C$-subproblem under that pilot coupling, and obtain $B^{(0,s)}$ from the corresponding $B$-update. Then, for outer iteration $t = 0, 1, \ldots, T_{max}-1$, perform the cyclic updates

$$
B^{(t+1,s)}
=
\arg\min_{B} J(B, \Gamma^{(t,s)}, \widehat C^{(t,s)}),
$$

$$
\Gamma^{(t+1,s)}
=
\arg\min_{\Gamma \in \mathcal G} J(B^{(t+1,s)}, \Gamma, \widehat C^{(t,s)}),
$$

$$
\widehat C^{(t+1,s)}
=
\arg\min_{\widehat C \in \mathbb{R}_{+}^{N \times F}} J(B^{(t+1,s)}, \Gamma^{(t+1,s)}, \widehat C).
$$

After each full cycle, compute the objective value

$$
J^{(t+1,s)} = J(B^{(t+1,s)}, \Gamma^{(t+1,s)}, \widehat C^{(t+1,s)}).
$$

Also record the running minimum objective value

$$
m^{(t+1,s)} = \min_{0 \le i \le t+1} J^{(i,s)}.
$$

Once at least $W$ objective values have been recorded, fit a simple linear regression to the trailing window of running minimum objective values,

$$
m^{(t-W+2+j,s)} \approx \alpha^{(t+1,s)} + \beta^{(t+1,s)} j,
\qquad j = 0, \ldots, W-1.
$$

Stop the restart when the regression slope is sufficiently flat relative to the best objective attained so far,

$$
\left|\widehat \beta^{(t+1,s)}\right| \le \varepsilon_{reg},
$$

or when the outer-iteration cap is reached. Across all restarts, retain the feasible stationary point associated with the smallest recorded running minimum objective value and acceptable conditioning of $I_F - \Gamma$. Because each exact block update cannot increase $J$ and because $J \ge 0$, the objective sequence within each restart is monotone nonincreasing and converges in value, although the final point need not be globally optimal.

### 6.4 Deployment-time inference as a mathematical program

Training determines the driver coefficients $\widehat B$ and the coupling matrix $\widehat \Gamma$. Deployment then predicts a new component vector by solving a second mathematical program. For a new sample $(u_*, c_{in,*})$, define

$$
\phi_* = \phi(u_*, c_{in,*}),
\qquad
d_* = \widehat B \phi_*,
\qquad
\widehat R = I_F - \widehat \Gamma.
$$

Whenever $\widehat R$ is nonsingular, define the coupled affine predictor by

$$
c_{aff} = \widehat R^{-1} d_*.
$$

Let $w_c \in \mathbb{R}_{+}^{F}$ denote a user-chosen componentwise correction-weight vector, and introduce an auxiliary vector $r \in \mathbb{R}^{F}$ that bounds the componentwise absolute correction away from $c_{aff}$.

The deployed prediction is defined by the convex linear program

$$
\min_{c \in \mathbb{R}^{F},\, r \in \mathbb{R}^{F}} \; w_c^T r
$$

subject to

$$
A c = A c_{in,*},
\qquad
c \ge 0,
$$

$$
-r \le c - c_{aff} \le r,
\qquad
r \ge 0.
$$

This is the deployed inference rule of the model. It is not merely the direct evaluation of $\widehat B \phi_*$. The objective minimizes a weighted absolute correction away from the coupled affine predictor while preserving the stoichiometric invariants exactly and enforcing componentwise nonnegative predictions. The auxiliary variable $r$ linearizes the absolute correction term component by component.

Because the objective is linear and every constraint is linear, the deployment problem is a convex linear program conditional on the trained parameters. Under the adopted non-negative concentration-equivalent state basis, the feasible set contains $(c, r) = (c_{in,*}, |c_{in,*} - c_{aff}|)$ and is therefore not empty.

### 6.5 What this formulation guarantees and what it does not

The formulation guarantees the following.

1. **Nonnegative fitted predictions.** The fitted training predictions are nonnegative because the training program imposes $\widehat C \ge 0$ directly.
2. **Non-negative deployed predictions.** The deployed prediction vector is nonnegative because the inference program imposes $c \ge 0$ directly.
3. **Component-space training target.** The model is trained on ASM component states directly on the adopted concentration-equivalent basis.
4. **Explicit learned coupling.** Cross-component dependence enters the surrogate through the estimated coupling matrix $\Gamma$ and the coupled-system matrix $R = I_F - \Gamma$.
5. **Invariant-aware fitted predictions and exact deployed conservation.** Conservation enters training through invariant residual terms in ASM component space and enters deployment through hard linear equality constraints.
6. **Conditional convex deployment.** Once $(\widehat B, \widehat \Gamma)$ have been trained, deployment-time inference is a convex linear program in the predicted component vector and its auxiliary correction variable.
7. **External measurement collapse.** Conversion to measured aggregates remains outside the model.

It does not guarantee the following.

1. **Nonnegative unconstrained drivers.** The vector $B \phi(u, c_{in})$ need not be nonnegative, because $B$ is not sign-constrained.
2. **Nonnegative unconstrained coupled solutions.** The formal unconstrained solution $R^{-1} B \phi(u, c_{in})$ need not be nonnegative.
3. **Global convexity of training.** Joint estimation of $(B, \Gamma, \widehat C)$ is not a one-shot convex program.
4. **Unique identification of driver and coupling.** Without sufficient excitation, structural restrictions, and regularization, some effects of $B$ and $\Gamma$ can be partially confounded.
5. **Exact fitted conservation at training points.** With finite $\lambda_{inv}$, the training objective encourages but does not guarantee exact invariant satisfaction in $\widehat C$.
6. **Full biological realizability.** Non-negative component states are necessary but not sufficient for full process feasibility.
7. **Monotonicity in the feature basis.** Because the coefficients are unrestricted in sign, the revised model is not monotone by construction in the native second-order feature coordinates.

If exact equality-constrained fitting is required as well, the training penalty term must be replaced by the corresponding fitted-state equality constraints and feasibility must be checked against the adopted invariant matrix, system boundary, and state basis.

## 7. External Composition-Matrix Collapse

### 7.1 External measured output equation

The ICSOR model predicts only ASM component states, and those predictions are obtained from the deployment-time inference program in Section 6.4. If measured composite outputs are needed for reporting, one computes them externally through

$$
\hat y_{ext} = I_{comp}\hat c.
$$

This equation is not part of the regression model itself. It is a downstream linear transformation of the model output.

### 7.2 Why the collapse is external

This separation is substantive, not cosmetic.

1. The learned object is the coupled ASM component-space predictor $\hat c(u, c_{in})$ produced by the driver and the constrained inference solve.
2. The composition matrix $I_{comp}$ is an external reporting operator.
3. Changing the reporting convention changes $I_{comp}$, not the trained ICSOR model.

This is useful because different studies may want different measured aggregates while still using the same ASM component predictor. One study may collapse to COD, TN, and TP; another may include TSS and VSS as well. Those are downstream choices.

### 7.3 When non-negative component predictions imply non-negative composites

If every row of $I_{comp}$ is entrywise non-negative, then non-negative component predictions imply non-negative reported composites. For output index $k$,

$$
\hat y_{ext,k}
= \sum_{f=1}^{F} (I_{comp})_{k f} \hat c_f.
$$

If $(I_{comp})_{k f} \ge 0$ for all $f$ and $\hat c_f \ge 0$ for all $f$, then

$$
\hat y_{ext,k} \ge 0.
$$

This is the common case for composite definitions built as sums of COD-bearing, nitrogen-bearing, phosphorus-bearing, or solids-bearing fractions with non-negative conversion factors.

### 7.4 Why measured-space reporting should remain external

Two different ASM component states can collapse to the same measured composite vector. Therefore measured-output agreement alone is not enough to characterize the internal ASM state. By making measurement collapse external, the model keeps the scientific interpretation attached to the component-space prediction rather than to an aggregate that may hide multiple plausible internal redistributions.

## 8. Estimation, Identifiability, and Practical Interpretation in Component Space

### 8.1 What the data identify now

Because the model is trained directly on $C_{out}$, the primary inferential objects are the driver matrix $B$, the coupling matrix $\Gamma$, and the fitted nonnegative prediction matrix $\widehat C$, all under the admissible-set restrictions and regularization choices adopted during training. The training targets live in ASM component space from the start.

This means the data identify a structured driver-coupling-prediction architecture rather than a sign-restricted coefficient matrix alone. In particular, the article no longer interprets $B$ as a direct map from features to final deployed component predictions. Instead, $B$ determines the driver, $\Gamma$ determines how components are coupled in that driver-to-state relation, and the deployed prediction is the optimizer of the inference program.

### 8.2 Rank deficiency, coupling conditioning, and coefficient interpretation

Direct component-space training with estimated coupling still leaves several identifiability issues.

1. If $N < D$, the feature design cannot have full column rank and the driver coefficients are not unique.
2. Even when $N \ge D$, the operating domain may weakly excite some directions of the second-order basis, making the driver identification ill conditioned.
3. If $R = I_F - \Gamma$ is poorly conditioned or nearly singular, small driver or coupling perturbations can be amplified before the deployment-time inference program corrects them.
4. The unsymmetrized second-order basis contains symmetric duplicates such as $u_i u_j$ and $u_j u_i$ unless the basis is compressed, which enlarges $D$ and can degrade coefficient interpretability.
5. Because $\Gamma$ is estimated jointly with $B$, some direct-coupling effects and feature-driven effects can be partially confounded unless $\Gamma$ is structurally restricted and regularized.

Under these conditions different parameter triples $(B, \Gamma, \widehat C)$ can produce similar driver fields and similar deployed predictions. The new formulation restores sign flexibility in $B$ and learns coupling explicitly, but it does not create uniqueness when the feature space is poorly excited or when the coupled system is ill conditioned.

Therefore interpretation should focus on the following objects in order of reliability:

1. deployed ASM component predictions produced by the inference program,
2. fitted prediction matrices and invariant-residual diagnostics,
3. coupling patterns in $\Gamma$ that remain stable across admissible initializations and regularization choices,
4. blockwise driver contributions of $B$,
5. individual coefficients only when the design matrix is sufficiently informative and the coupling estimate is stable.

### 8.3 Practical QP solvers for training and inference

Training is no longer a one-shot convex quadratic program. It is a structured block-coordinate estimation problem with three main subproblems:

1. a convex least-squares or ridge-type update for $B$,
2. a convex quadratic program or constrained least-squares update for $\Gamma$, and
3. a convex quadratic program with nonnegativity constraints and invariant-residual penalties for $\widehat C$.

Standard convex machinery therefore still applies within each block, including active-set methods, interior-point methods, operator-splitting methods, and augmented-Lagrangian variants. In the $\widehat C$-step, the invariant penalty preserves convexity without requiring additional linear equality constraints.

In implementation terms, the training loop is a Gauss-Seidel sweep in the order $B \rightarrow \Gamma \rightarrow \widehat C$, with warm starts passed from one outer iteration to the next. Each restart should record the running minimum objective value, the final objective value, the conditioning of $I_F - \Gamma$, and any active admissibility constraints on $\Gamma$; the retained estimate is the best feasible restart judged by the smallest attained objective value rather than merely the last one run.

Deployment-time inference is substantially smaller. For each new sample, one solves a convex linear program in the $2F$ variables $(c, r)$. That means the additional online cost is typically modest relative to the cost of plant simulation, while still giving a hard nonnegativity guarantee and exact invariant preservation on the deployed prediction.

Three practical points matter.

1. Training and inference are both coupled across ASM components; solving $F$ independent scalar prediction problems is not equivalent once $\Gamma$ and the invariant terms are present.
2. Warm starts are natural in both phases: training can reuse nearby solutions across refits, and inference can reuse nearby solutions across similar operating conditions.
3. Multiple initializations and conditioning diagnostics for $\Gamma$ are prudent because the joint training problem is nonconvex even though each block subproblem is convex.
4. Feature scaling should preserve the intended physical coordinate system. Coefficient sign freedom means arbitrary centering no longer breaks the positivity guarantee itself, but it can still harm conditioning and obscure the engineering meaning of the driver coefficients.
5. If tuning weights or admissible-set hyperparameters are selected by validation, that outer selection loop must rerun the entire block-coordinate estimator rather than reuse one previously trained factorization.

### 8.4 Deployment after training

Once $(\widehat B, \widehat \Gamma)$ have been estimated, the deployed predictor is obtained in two steps.

1. Build the feature-driven driver $d_* = \widehat B \phi(u_*, c_{in,*})$, the learned coupled-system matrix $\widehat R = I_F - \widehat \Gamma$, and the coupled affine predictor $c_{aff} = \widehat R^{-1} d_*$.
2. Solve the deployment-time inference program from Section 6.4 to obtain $\hat c_*$.

This optimizer is the native output of ICSOR. If an application requires measured composites, the external reporting vector is then

$$
\hat y_{ext}(u_*, c_{in,*}) = I_{comp} \hat c_*.
$$

That second equation is a reporting formula, not a redefinition of the model target.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Local regimes in training and inference

The revised formulation has several possible regime changes.

1. In training, the nonnegativity constraints act on the entries of the fitted prediction matrix $\widehat C$, while the invariant residual penalty remains active for every sample.
2. If the admissible set $\mathcal G$ imposes inequality, sign, magnitude, or sparsity restrictions on $\Gamma$, those coupling constraints can also become active during training.
3. In deployment, the nonnegativity constraints act on the entries of the inferred component vector $\hat c_*$, while the invariant equalities remain active for every solve.

These regime changes matter because the final deployed predictor is piecewise smooth rather than globally affine, even after the feature map has been fixed.

### 9.2 Bootstrap refitting as the recommended default

For the final deployed predictor, bootstrap refitting is the recommended default uncertainty method. Each bootstrap replicate should repeat the full modeling pipeline:

1. resample the steady-state samples,
2. refit $(B, \Gamma, \widehat C)$ under the same admissible-set restrictions, regularization, and tuning policy,
3. re-solve the deployment-time inference program for each prediction point of interest, and
4. push the resulting component predictions through $I_{comp}$ if measured-space uncertainty is required.

This procedure captures driver-estimation uncertainty, coupling-estimation uncertainty, active-set changes in the fitted prediction step, active-set changes in deployment-time inference, and sensitivity to the conditioning of $R = I_F - \Gamma$.

### 9.3 What a local analytic approximation would require

If one nevertheless wants a local covariance approximation, it must be stated explicitly as conditional on a fixed local optimum, fixed active sets, fixed admissibility regime for $\Gamma$, fixed regularization weights, and a chosen parameterization of the coupling matrix. Let

$$
x =
\begin{bmatrix}
\operatorname{vec}(B^T) \\
\operatorname{vec}(\Gamma^T) \\
\operatorname{vec}(\widehat C)
\end{bmatrix}.
$$

A local approximation would use the free-coordinate block of the Hessian of the training objective together with the Jacobian of any active equality constraints. Such formulas are model- and constraint-dependent and do not provide a clean exact uncertainty description for the final deployed predictor, because they ignore regime changes and condition on one local solution of a nonconvex training problem.

### 9.4 Prediction uncertainty for ASM components and for external composites

Whether uncertainty is quantified by bootstrap refitting or by a local approximation, the logical order of propagation remains the same: uncertainty is first characterized in native ASM component space and only then pushed forward through the external composition matrix. If $\operatorname{Var}(\hat c_*)$ is available, the corresponding measured-space covariance follows by

$$
\operatorname{Var}(\hat y_{ext,*}) \approx I_{comp}\operatorname{Var}(\hat c_*) I_{comp}^T.
$$

This equation remains useful because the measurement collapse is external to the deployed component-space predictor.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct ASM component prediction now uses a latent driver and a constrained predictor

The model still targets the effluent ASM component states directly, so the scientific target remains close to the mechanistic state description. What changes is the deployed prediction rule: the feature basis drives a latent signal together with an estimated coupling structure, and the final ASM component prediction is obtained from a constrained inference program rather than from direct closed-form regression evaluation.

### 10.2 Non-negativity is an optimization property, not a coefficient-sign property

The non-negativity guarantee is no longer claimed as a property of the coefficient matrix. It comes instead from the optimization problems that define fitted and deployed predictions. This restores sign flexibility in $B$, allows decreasing effects to be represented directly in the driver, and moves the hard sign guarantee to the place where it matters physically: the predicted ASM component states.

### 10.3 Invariants act during training and deployment in different ways

Stoichiometric invariants remain structural guidance, and in this formulation they appear in two places. They shape the fitted prediction matrix during training through an invariant-residual penalty, and they also shape the deployed prediction during inference through hard linear equality constraints. The physics is therefore enforced through optimization rather than through a closed-form algebraic projector.

### 10.4 Measurement collapse remains a reporting decision

Because the model still predicts ASM component states only, the composition matrix remains a downstream reporting choice. This is useful in practice because one can analyze the same trained driver-and-inference architecture under different measured-output reporting conventions without retraining the surrogate.

## 11. Limitations

ICSOR is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state in the quasi-steady-sample sense and does not represent temporal dynamics or path dependence.
2. It requires effluent ASM component-state targets for training, either directly from a simulator or from an upstream reconstruction step.
3. Joint estimation of $(B, \Gamma, \widehat C)$ is nonconvex, so practical algorithms generally return local solutions rather than a guaranteed global optimum.
4. The non-negativity guarantee applies to the fitted and deployed outputs of the optimization programs, not to the unconstrained driver $B \phi(u, c_{in})$ or to the unconstrained coupled solution $R^{-1} B \phi(u, c_{in})$.
5. Deployment requires solving a linear program for every prediction point, so runtime depends on solver choice, conditioning, and tolerances.
6. Identification of the coupling matrix depends on admissible-set restrictions, regularization, and operating-domain excitation; without them, coupling effects and driver effects can be partially confounded.
7. If $R = I_F - \Gamma$ is poorly conditioned or nearly singular, the coupled architecture can become numerically sensitive.
8. The second-order feature basis can be statistically fragile when it is weakly excited or highly collinear.
9. The unsymmetrized quadratic basis contains duplicated monomials unless it is compressed, so it can inflate $D$ and reduce individual coefficient interpretability.
10. Because $B$ is unrestricted in sign, the revised model no longer carries a monotonicity guarantee in the native feature basis.
11. Uncertainty for the final deployed predictor is most defensibly handled by bootstrap refitting rather than by simple closed-form covariance formulas.
12. Non-negative component predictions imply non-negative externally reported composites only when the relevant rows of $I_{comp}$ are entrywise non-negative.
13. The invariant theory is written on a concentration-equivalent linear component basis and does not transfer unchanged to arbitrary normalized fractions or other nonlinear state parameterizations.
14. A misspecified stoichiometric matrix, incorrect system boundary, poorly chosen admissible set for $\Gamma$, or uncertain composition matrix can yield mathematically consistent optimization problems for the wrong physical system, particularly because the deployed inference problem enforces the adopted invariant equalities exactly.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its guarantees correctly.

## 12. Conclusion

ICSOR is formulated here as a direct ASM component-space surrogate with an explicit separation between feature-driven forcing and final deployed prediction. It takes operational variables and influent ASM component states as input, builds a second-order driver $B \phi(u, c_{in})$, estimates a coupling matrix $\Gamma$, and predicts effluent ASM component states through optimization problems posed directly in component space.

Training is cast as a jointly nonconvex but blockwise structured mathematical program. It learns the driver coefficients, coupling coefficients, and fitted nonnegative prediction matrix while penalizing invariant mismatch and coupled-system mismatch. Deployment is cast as a convex linear program conditional on the trained parameters: for each new sample, the final component prediction is the optimizer of a constrained inference problem that enforces both componentwise nonnegativity and the invariant equalities while minimizing a weighted absolute correction away from the coupled affine predictor. Measured composites remain external calculations through the composition matrix.

Under that reading, ICSOR is best understood as a coupled, second-order, component-space surrogate whose scientific target remains the ASM component-state vector itself, whose coupling structure is learned rather than prescribed, whose measurement layer remains external, and whose non-negativity guarantee is created by mathematical programming at both training and inference rather than by coefficient sign restrictions.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Lawson, C. L., and Hanson, R. J. Solving Least Squares Problems. SIAM, 1995.
4. Bates, D. M., and Watts, D. G. Nonlinear Regression Analysis and Its Applications. Wiley, 1988.
5. Nocedal, J., and Wright, S. J. Numerical Optimization. 2nd ed. Springer, 2006.
6. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
7. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.
