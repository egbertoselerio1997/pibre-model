# Non-Negative Invariant-Constrained Second-Order Regression (ICSOR) with NNLS for Activated Sludge Component Prediction

## Abstract

This article presents a non-negative formulation of invariant-constrained second-order regression (ICSOR) for steady-state activated-sludge surrogate modeling. The model accepts operational variables and influent activated-sludge-model (ASM) component fractions, and it predicts effluent ASM component fractions in the same component basis. ICSOR is defined natively in ASM component space. It is trained to predict ASM component fractions directly, and only those component fractions. If measured composite variables such as total COD, total nitrogen, total phosphorus, TSS, or VSS are needed, they are computed afterward by an external composition matrix. The collapse into measured-output space is therefore not part of the model itself.

The non-negativity guarantee is built directly into coefficient estimation. The second-order feature vector is assembled only from positive inputs and positive monomials, and every learnable coefficient is constrained to be positive. Coefficient estimation is executed entirely by nonlinear least squares through a positive parameterization of the coefficient matrix. In coefficient space, this is equivalent to a non-negative least-squares objective; in implementation, it is solved as nonlinear least squares over unconstrained parameters mapped into the positive orthant. Because the feature vector and the learned coefficients are both positive, the predicted ASM component fractions are non-negative componentwise.

Stoichiometric conservation remains central, but it is handled during fitting in component space through explicit invariant residuals. ICSOR is a direct component-space surrogate with positive coefficients, positive predictions, and external measurement collapse. This article develops that theory carefully, states what it guarantees, and revises the discussion, estimation logic, uncertainty treatment, and limitations around that contract.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to represent a quasi-steady operating condition: the operating variables, influent composition, and effluent response are treated as effectively time-invariant over the control volume being modeled for the sampling window of interest.

The usual difficulty is that wastewater physics is written in an ASM component basis, whereas reporting and plant supervision are often performed in measured aggregate variables. The stoichiometric matrix acts on component fractions such as substrate, biomass, ammonium, nitrate, phosphate, dissolved oxygen, or alkalinity. Plant dashboards, however, often monitor total COD, total nitrogen, total phosphorus, TSS, or VSS. A model trained only on those aggregates can fit measured outputs while implying an impossible redistribution of the underlying ASM components.

ICSOR is formulated here to remove that mismatch at the model-definition level. The model input is split into

1. an operational block $u$, and
2. an influent ASM component-fraction block $c_{in}$.

The model output is the effluent ASM component-fraction vector $\hat c$ in the same ASM basis as the input. This means the native prediction target is not a measured composite. It is the component vector itself. If measured composites are needed for reporting, optimization, or comparison to plant observations, they are obtained afterward by an external composition matrix:

$$
\hat y_{ext} = I_{comp} \hat c.
$$

That external collapse is deliberate. It keeps the learned map in the same space in which stoichiometric invariants are defined and in which the non-negativity claim is made.

This article answers one precise question:

> Given a steady-state influent ASM component-fraction vector and a steady-state operating condition, what effluent ASM component-fraction vector should be predicted if the learned surrogate is second order, its coefficients are constrained to be positive, its predictions must therefore remain non-negative, and stoichiometric invariants must be respected during estimation?

The theory in this article is restricted to steady-state reactor-block prediction. It does not replace a dynamic activated-sludge simulator. It also does not claim that non-negative ASM component predictions are automatically kinetically or biologically realizable. Its narrower claim is that the surrogate should live in component space, be fitted in component space, remain non-negative by construction, and treat measured-output collapse as an external downstream calculation.

Two additional consequences follow immediately from this choice.

1. Training requires effluent ASM component-fraction targets, either observed directly from a simulator or reconstructed upstream from measured data.
2. Non-negativity is a property of the fitted model class itself rather than a deployment-time repair step.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit represented by quasi-steady samples. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes bypass streams, gas stripping, chemical dosing, sludge wastage, or any other transport mechanism that changes the component inventory across the chosen boundary.

Throughout the article, the ASM state vectors are treated as non-negative component fractions on a common basis. In some implementations those coordinates may be literal fractions; in others they may be concentration-equivalent normalized states. The algebra below is unchanged as long as the same non-negative ASM basis is used consistently for $c_{in}$ and $c_{out}$.

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
| $c_{in}$ | $\mathbb{R}_{+}^{F}$ | Influent ASM component-fraction vector |
| $c_{out}$ | $\mathbb{R}_{+}^{F}$ | True effluent ASM component-fraction vector |
| $\hat c$ | $\mathbb{R}_{+}^{F}$ | Predicted effluent ASM component-fraction vector |
| $y_{ext}$ | $\mathbb{R}^{K}$ | External measured composite vector computed from a component vector |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM components to measured composite variables |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction-progress vector expressed on the same concentration-equivalent basis as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank invariant matrix satisfying $A \nu^T = 0$ |
| $\phi(u, c_{in})$ | $\mathbb{R}_{+}^{D}$ | Positive second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $B_{+}$ | $\mathbb{R}_{++}^{F \times D}$ | Positive coefficient matrix of the second-order component model |
| $\Gamma$ | $\mathbb{R}^{F \times D}$ | Unconstrained parameter matrix used to generate $B_{+}$ through a positive elementwise map |
| $s(\cdot)$ | scalar map | Positive elementwise map, for example softplus, used to enforce $B_{+} = s(\Gamma)$ |
| $\lambda_{inv}$ | $\mathbb{R}_{+}$ | Invariant-residual penalty weight in the estimation objective |
| $r_n(\Gamma)$ | $\mathbb{R}^{F+q}$ | Stacked component-fit and invariant residual for sample $n$ |
| $J(\widehat\Gamma)$ | matrix | Jacobian of the stacked nonlinear least-squares residual at the fitted solution |

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
4. **Positive input coordinates.** The operational variables and influent ASM component fractions are supplied on a non-negative, physically meaningful coordinate system, so that every monomial used in the feature map is non-negative.
5. **Direct component-space targets.** The model is trained against effluent ASM component fractions, not against measured composite outputs.
6. **Second-order surrogate class.** The predictor is a partitioned second-order polynomial model with linear, quadratic, and operation-loading interaction terms.
7. **Positive coefficient parameterization.** Every learnable coefficient is constrained to be positive through an explicit parameterization $B_{+} = s(\Gamma)$.
8. **Nonlinear least-squares estimation only.** All coefficient estimation is carried out by nonlinear least squares on $\Gamma$.
9. **Equivalent NNLS interpretation.** In the coefficient matrix $B_{+}$, the estimation problem is a non-negative least-squares problem augmented with invariant residuals; the practical implementation uses the equivalent nonlinear least-squares parameterization.
10. **Invariant-aware training objective.** Stoichiometric invariants are handled inside the estimation objective through residual terms in ASM component space.
11. **External composition map.** The collapse from ASM component fractions to measured composites is an external calculation performed after prediction through $I_{comp}$.
12. **Composite-sign scope.** Non-negative component predictions imply non-negative measured composites only when the relevant rows of $I_{comp}$ are entrywise non-negative.
13. **Target availability.** The effluent ASM component fractions used for training are assumed available directly or through an upstream state-reconstruction step outside the present model.

These assumptions matter because they narrow the scientific claim. The model guarantees non-negative component predictions by construction, but that guarantee is purchased by a positive-coefficient model class and by component-space supervision. It does not by itself guarantee exact kinetics, thermodynamic feasibility, or universal exact conservation for every unseen operating point.

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

The feasible effluent states lie on the non-negative line segment defined by this equality. The purpose of ICSOR is to learn a non-negative point on that segment directly in component space, not to predict a measured aggregate first and only later infer where the underlying components should have been.

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

and the trained positive-coefficient surrogate predicts

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

The key point is not the arithmetic. It is the ordering. The model predicts the ASM component fractions first. Only after that step are measured aggregates obtained. If another reporting convention is later desired, one changes $I_{comp}$ externally; the trained component-space model does not change.

## 5. Positive Second-Order Surrogate in ASM Component Space

### 5.1 Why the input is partitioned

In activated-sludge systems, operating conditions and influent component fractions play different physical roles.

1. Operating variables such as hydraulic retention time, dissolved-oxygen setpoint, or recycle settings alter the process environment.
2. Influent ASM component fractions describe the material inventory entering that environment.

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

### 5.3 Positive coefficient parameterization

The effluent ASM component-fraction prediction is defined directly by

$$
\hat c = B_{+} \phi(u, c_{in}),
$$

where $B_{+} \in \mathbb{R}_{++}^{F \times D}$ is an entrywise positive coefficient matrix. To enforce positivity, we introduce an unconstrained parameter matrix $\Gamma \in \mathbb{R}^{F \times D}$ and a positive elementwise map $s(\cdot)$, for example

$$
s(z) = \log(1 + e^z),
$$

and set

$$
B_{+} = s(\Gamma).
$$

Blockwise, the same model may be written as

$$
\hat c
= b_{+}
+ W_{u,+} u
+ W_{in,+} c_{in}
+ \Theta_{uu,+}(u \otimes u)
+ \Theta_{cc,+}(c_{in} \otimes c_{in})
+ \Theta_{uc,+}(u \otimes c_{in}),
$$

with

$$
b_{+},\; W_{u,+},\; W_{in,+},\; \Theta_{uu,+},\; \Theta_{cc,+},\; \Theta_{uc,+}
$$

all entrywise positive.

### 5.4 Why this guarantees non-negative predictions

For the $f$-th ASM component,

$$
\hat c_f = \sum_{d=1}^{D} (B_{+})_{f d} \, \phi_d(u, c_{in}).
$$

Every term in that sum is non-negative because

$$
(B_{+})_{f d} > 0
\qquad \text{and} \qquad
\phi_d(u, c_{in}) \ge 0.
$$

Therefore

$$
\hat c_f \ge 0
\qquad \text{for every component } f.
$$

If the bias feature is retained and its coefficient is strictly positive, then every predicted component is strictly positive. The weaker and more important guarantee is the non-negative one:

$$
\hat c \in \mathbb{R}_{+}^{F}.
$$

This is the non-negativity mechanism of ICSOR. It does not rely on clipping or any downstream repair step after the regression has been evaluated.

### 5.5 Componentwise interpretation and induced monotonicity

Each predicted ASM component is a positive weighted sum of non-negative monomials. This makes the model easy to audit:

1. $W_{u,+} u$ captures first-order operating contributions.
2. $W_{in,+} c_{in}$ captures first-order influent carry-through and composition effects.
3. $\Theta_{uu,+}(u \otimes u)$ captures operating curvature.
4. $\Theta_{cc,+}(c_{in} \otimes c_{in})$ captures influent curvature.
5. $\Theta_{uc,+}(u \otimes c_{in})$ captures operation-loading interactions.

The same sign structure also has an important consequence: in the native positive coordinate system, the model is monotone non-decreasing in every feature coordinate. That monotonicity is the price of the hard non-negativity guarantee. If a physically decreasing effect must be represented, it must be encoded through the choice of inputs or feature transformations rather than through negative coefficients.

## 6. Invariant-Aware NNLS and Nonlinear Least-Squares Estimation

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

For a given positive coefficient matrix $B_{+}$, define the prediction matrix

$$
\widehat C(B_{+}) = \Phi B_{+}^T.
$$

### 6.2 Equivalent NNLS objective in coefficient space

The natural component-space fit minimizes squared component error while also penalizing invariant residuals:

$$
\widehat B_{+}
= \arg\min_{B \in \mathbb{R}_{+}^{F \times D}}
\left\|
C_{out} - \Phi B^T
\right\|_F^2
+ \lambda_{inv}
\left\|
\Phi B^T A^T - C_{in} A^T
\right\|_F^2.
$$

This is the coefficient-space NNLS view of the model. The first term fits the effluent ASM component fractions directly. The second term penalizes violations of the stoichiometric invariant relations in the predicted component space. The non-negativity constraint $B \ge 0$ is the mechanism that guarantees non-negative predictions.

If one wishes to place different importance on different ASM components or different invariants, diagonal weighting matrices may be inserted into the two norms. The article omits those extra weights for clarity because they do not change the central structure of the estimator.

### 6.3 Implemented nonlinear least-squares estimator

Although the objective above is naturally described as NNLS in $B_{+}$, the implementation in this article uses nonlinear least squares on the unconstrained parameter matrix $\Gamma$ through the positive map

$$
B_{+} = s(\Gamma).
$$

For sample $n$, define the stacked residual

$$
r_n(\Gamma)
=
\begin{bmatrix}
c_{out,n} - s(\Gamma)\phi_n \\
\sqrt{\lambda_{inv}}\left(A s(\Gamma)\phi_n - A c_{in,n}\right)
\end{bmatrix}
\in \mathbb{R}^{F+q},
$$

where $\phi_n = \phi(u_n, c_{in,n})$. The fitted parameter matrix is then

$$
\widehat\Gamma
= \arg\min_{\Gamma \in \mathbb{R}^{F \times D}}
\sum_{n=1}^{N} \left\| r_n(\Gamma) \right\|_2^2.
$$

All coefficient estimation is executed through this nonlinear least-squares problem. After estimation,

$$
\widehat B_{+} = s(\widehat\Gamma)
$$

and the deployed model is

$$
\hat c(u, c_{in}) = \widehat B_{+}\phi(u, c_{in}).
$$

This formulation satisfies both requirements of the present model:

1. the learned coefficients are positive because they are generated by a positive map, and
2. the fitting procedure is nonlinear least squares from end to end.

### 6.4 Deployment sequence

For a new sample, the deployed prediction is evaluated in the following order:

1. form the positive second-order feature vector $\phi(u, c_{in})$,
2. evaluate the positive coefficient matrix $\widehat B_{+}$,
3. compute the ASM component-fraction prediction $\hat c = \widehat B_{+}\phi(u, c_{in})$,
4. return $\hat c$ as the model output.

If a measured composite vector is later needed, it is computed externally through

$$
\hat y_{ext} = I_{comp}\hat c.
$$

The deployed output is returned directly from the positive second-order regression model.

### 6.5 What this formulation guarantees and what it does not

The formulation guarantees the following.

1. **Positive coefficients.** The learned coefficient matrix is positive because $B_{+} = s(\Gamma)$.
2. **Non-negative ASM component predictions.** Since $\phi(u, c_{in}) \ge 0$, the predicted component vector $\hat c = B_{+}\phi(u, c_{in})$ is non-negative.
3. **Component-space training target.** The model is trained on ASM component fractions directly.
4. **Invariant-aware estimation.** Conservation enters estimation through the invariant residual term rather than through a downstream correction rule.
5. **External measurement collapse.** Conversion to measured aggregates is outside the model.

It does not guarantee the following.

1. **Universal exact conservation at every unseen point.** With a finite penalty $\lambda_{inv}$, invariants are enforced through the estimation objective and are not automatically exact at every unseen point.
2. **Full biological realizability.** Non-negative component fractions are necessary but not sufficient for full process feasibility.
3. **Monotonic flexibility in both directions.** Because the coefficients are positive, decreasing effects are not represented through negative slopes in the native feature basis.

If a particular application requires exact equality-constrained fitting of the invariant residual at the estimation stage, one may replace the quadratic penalty by equality constraints inside the same nonlinear least-squares solve. The positive-coefficient logic and the external reporting logic are unchanged.

## 7. External Composition-Matrix Collapse

### 7.1 External measured output equation

The ICSOR model predicts only ASM component fractions. If measured composite outputs are needed for reporting, one computes them externally through

$$
\hat y_{ext} = I_{comp}\hat c.
$$

This equation is not part of the regression model itself. It is a downstream linear transformation of the model output.

### 7.2 Why the collapse is external

This separation is substantive, not cosmetic.

1. The learned object is the ASM component-fraction map $\hat c(u, c_{in})$.
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

Because the model is trained directly on $C_{out}$, the primary inferential object is the positive component-space coefficient matrix $B_{+}$ itself. The training targets live in ASM component space from the start.

The model assumes component-space targets are available and fits the component map directly.

### 8.2 Rank deficiency, duplicated monomials, and coefficient interpretation

Direct component-space training removes one identifiability problem, but it does not remove all of them.

1. If $N < D$, the design matrix cannot have full column rank.
2. Even when $N \ge D$, the feature map can be rank deficient because the operating domain does not excite all directions of the second-order basis.
3. The full Kronecker basis contains symmetric duplicates such as $u_i u_j$ and $u_j u_i$ unless the basis is compressed.

Under these conditions, different positive coefficient matrices can produce similar fitted predictions. The non-negative coefficient restriction stabilizes the problem by removing sign cancellations, but it does not create uniqueness when the feature space itself is poorly excited.

Therefore interpretation should focus on the following objects in order of reliability:

1. predicted ASM component fractions,
2. blockwise contribution patterns of $W_{u,+}$, $W_{in,+}$, $\Theta_{uu,+}$, $\Theta_{cc,+}$, and $\Theta_{uc,+}$,
3. individual coefficients only when the design matrix is sufficiently informative.

### 8.3 Practical nonlinear least-squares solvers

The present article is solver-agnostic. Any nonlinear least-squares method that can optimize $\Gamma$ may be used, provided it is applied to the positive parameterization $B_{+} = s(\Gamma)$. Common choices include Gauss-Newton, Levenberg-Marquardt variants on the reparameterized variables, and trust-region methods.

Three practical points matter more than the specific solver brand.

1. The residual vector should stack both component-fit residuals and invariant residuals.
2. Initialization matters because the problem is nonlinear after the positive map is introduced.
3. Feature scaling must preserve the intended positive coordinate system; arbitrary centering that creates negative feature coordinates breaks the direct positivity argument of the model.

### 8.4 Deployment after training

Once $\widehat\Gamma$ has been estimated, the deployed component-space predictor is

$$
\hat c(u, c_{in}) = s(\widehat\Gamma)\phi(u, c_{in}).
$$

This is the native output of ICSOR. If an application requires measured composites, the external reporting vector is

$$
\hat y_{ext}(u, c_{in}) = I_{comp} s(\widehat\Gamma)\phi(u, c_{in}).
$$

That second equation is a reporting formula, not a redefinition of the model target.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Residual vector and local Jacobian

Stack the samplewise residuals into one global residual vector

$$
r(\Gamma) =
\begin{bmatrix}
r_1(\Gamma) \\
r_2(\Gamma) \\
\vdots \\
r_N(\Gamma)
\end{bmatrix}.
$$

Let

$$
\widehat\theta = \operatorname{vec}(\widehat\Gamma)
$$

and let

$$
J(\widehat\Gamma)
= \frac{\partial r(\Gamma)}{\partial \operatorname{vec}(\Gamma)^T}
\Bigg|_{\Gamma = \widehat\Gamma}
$$

be the Jacobian of the stacked residual at the fitted solution.

### 9.2 Local covariance approximation

Exact ordinary-least-squares covariance formulas do not apply because the fitted model is nonlinear in $\Gamma$. A standard local approximation is the Gauss-Newton covariance

$$
\operatorname{Cov}(\widehat\theta)
\approx \widehat\sigma^2
\left(J(\widehat\Gamma)^T J(\widehat\Gamma)\right)^{-1},
$$

provided the Jacobian has full column rank in the neighborhood of the solution. Here

$$
\widehat\sigma^2
= \frac{\|r(\widehat\Gamma)\|_2^2}{m-p},
$$

where $m$ is the total residual dimension and $p$ is the number of free parameters in $\Gamma$.

This approximation is local and asymptotic. It is most defensible when the residual surface is well behaved and the fitted solution is not too close to a flat or poorly identified region.

### 9.3 Prediction uncertainty for ASM components and for external composites

For a new sample $(u_*, c_{in,*})$, define

$$
\hat c_* = s(\widehat\Gamma)\phi(u_*, c_{in,*}).
$$

Let

$$
G_* =
\frac{\partial \hat c_*}{\partial \operatorname{vec}(\Gamma)^T}
\Bigg|_{\Gamma = \widehat\Gamma}.
$$

Then a local delta-method approximation gives

$$
\operatorname{Var}(\hat c_*)
\approx
G_* \operatorname{Cov}(\widehat\theta) G_*^T.
$$

If external measured composites are later computed, their approximate covariance follows directly:

$$
\operatorname{Var}(\hat y_{ext,*})
\approx
I_{comp}\operatorname{Var}(\hat c_*) I_{comp}^T.
$$

This equation is useful because it preserves the logical order of the formulation: uncertainty is computed first in the native ASM component space and only then pushed forward through the external composition matrix.

### 9.4 Why resampling is still preferred

Despite the local Jacobian formulas above, bootstrap refitting remains the more defensible default for the deployed model. The reasons are straightforward.

1. The fitted map is nonlinear because of the positive coefficient parameterization.
2. The positive orthant restriction can create boundary-like behavior in coefficient space.
3. The invariant penalty introduces additional curvature and dependence across outputs.
4. Rank deficiency and duplicated monomials can make local covariance estimates numerically fragile.

Therefore the recommended uncertainty treatment for the final deployed predictor is bootstrap refitting in component space, followed by optional external collapse through $I_{comp}$ when measured-space summaries are required.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct ASM component prediction changes the inferential target

The model targets the effluent ASM component fractions directly. This makes the surrogate scientifically closer to the mechanistic state description, but it also requires training data in that same ASM basis.

### 10.2 Positive coefficients build sign safety into the model class

The non-negativity guarantee is not a numerical afterthought. It comes from the model class itself: positive features and positive coefficients imply non-negative predicted components. The tradeoff is expressive restriction. In the native coordinate system, the model is additive and monotone in the selected features. Any effect that should behave like a decrease must be represented through the design of the input coordinates rather than through negative coefficients.

### 10.3 Invariants are handled during estimation rather than after prediction

Stoichiometric invariants act as training-time structural guidance. The invariant residuals are part of the nonlinear least-squares objective, so conservation influences the fitted coefficients directly.

### 10.4 Measurement collapse becomes a reporting decision

Because the model predicts ASM component fractions only, the composition matrix becomes a downstream reporting choice. This is useful in practice because one can analyze the same trained model under different measured-output reporting conventions without retraining the surrogate.

## 11. Limitations

ICSOR is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state in the quasi-steady-sample sense and does not represent temporal dynamics or path dependence.
2. It requires effluent ASM component-fraction targets for training, either directly from a simulator or from an upstream reconstruction step.
3. The non-negativity guarantee depends on maintaining a non-negative input coordinate system and a non-negative monomial feature map.
4. Positive coefficients imply a monotone additive model in the native feature coordinates, which can underrepresent genuinely decreasing or sign-changing physical effects unless those are encoded through feature design.
5. If a strictly positive bias term is retained, exact zero predictions are not produced; the model produces non-negative values and typically strictly positive ones.
6. With a finite invariant penalty $\lambda_{inv}$, stoichiometric conservation is enforced through estimation pressure and is not exact by construction for every deployed sample.
7. If exact equality-constrained fitting of the invariants is mandatory, the nonlinear least-squares solver must be upgraded accordingly.
8. The second-order feature basis can be statistically fragile when it is weakly excited or highly collinear.
9. The full Kronecker quadratic basis contains duplicated monomials unless it is compressed, so individual quadratic coefficients are not always uniquely interpretable.
10. Local Jacobian-based covariance formulas are only approximations for the fitted nonlinear model.
11. Non-negative component predictions imply non-negative externally reported composites only when the relevant rows of $I_{comp}$ are entrywise non-negative.
12. A misspecified stoichiometric matrix or incorrect system boundary leads to an invariant term that is mathematically consistent with the wrong physical system.
13. The composition matrix itself is treated as known and fixed; uncertainty in $I_{comp}$ is outside the present error model.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its guarantees correctly.

## 12. Conclusion

ICSOR is formulated here as a direct ASM component-space surrogate. It takes operational variables and influent ASM component fractions as input, and it predicts effluent ASM component fractions in the same basis. The model is non-negative by construction because its feature vector is non-negative and its coefficient matrix is constrained to be positive. All coefficient estimation is executed by nonlinear least squares through a positive parameterization, which is the implemented form of the underlying NNLS idea.

Stoichiometric invariants enter the estimation problem as explicit component-space residuals. Measured composites are obtained afterward as an external calculation through the composition matrix. Under that reading, ICSOR is best understood as a positive, second-order, component-space surrogate whose primary output is the ASM component-fraction vector itself and whose measured-output reporting layer sits outside the model.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Lawson, C. L., and Hanson, R. J. Solving Least Squares Problems. SIAM, 1995.
4. Bates, D. M., and Watts, D. G. Nonlinear Regression Analysis and Its Applications. Wiley, 1988.
5. Nocedal, J., and Wright, S. J. Numerical Optimization. 2nd ed. Springer, 2006.
6. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
