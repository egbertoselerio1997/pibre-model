# Non-Negative Invariant-Constrained Second-Order Regression (ICSOR) with NNLS for Activated Sludge Component Prediction

## Abstract

This article presents a non-negative formulation of invariant-constrained second-order regression (ICSOR) for steady-state activated-sludge surrogate modeling. The model accepts operational variables and influent activated-sludge-model (ASM) component fractions, and it predicts effluent ASM component fractions in the same component basis. ICSOR is defined natively in ASM component space. It is trained to predict ASM component fractions directly, and only those component fractions. If measured composite variables such as total COD, total nitrogen, total phosphorus, TSS, or VSS are needed, they are computed afterward by an external composition matrix. The collapse into measured-output space is therefore not part of the model itself.

The non-negativity guarantee is built directly into coefficient estimation. The second-order feature vector is assembled only from non-negative inputs and non-negative monomials, and every learnable coefficient is constrained to be non-negative. Coefficient estimation is posed directly as a coupled non-negative least-squares problem in the coefficient matrix itself. Because the feature vector and the learned coefficients are both non-negative, the predicted ASM component fractions are non-negative componentwise, with exact zeros allowed.

Stoichiometric conservation remains central, but it is handled during fitting in component space through explicit invariant residuals. ICSOR is a direct component-space surrogate with nonnegative coefficients, nonnegative predictions, and external measurement collapse. This article develops that theory carefully, states what it guarantees, and revises the discussion, estimation logic, uncertainty treatment, and limitations around that contract.

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

> Given a steady-state influent ASM component-fraction vector and a steady-state operating condition, what effluent ASM component-fraction vector should be predicted if the learned surrogate is second order, its coefficients are constrained to be non-negative, its predictions must therefore remain non-negative, and stoichiometric invariants must be respected during estimation?

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
| $\phi(u, c_{in})$ | $\mathbb{R}_{+}^{D}$ | Non-negative second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $B$ | $\mathbb{R}_{+}^{F \times D}$ | Nonnegative coefficient matrix of the second-order component model |
| $\lambda_{inv}$ | $\mathbb{R}_{+}$ | Invariant-residual penalty weight in the estimation objective |

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
4. **Non-negative input coordinates.** The operational variables and influent ASM component fractions are supplied on a non-negative, physically meaningful coordinate system, so that every monomial used in the feature map is non-negative.
5. **Direct component-space targets.** The model is trained against effluent ASM component fractions, not against measured composite outputs.
6. **Second-order surrogate class.** The predictor is a partitioned second-order polynomial model with linear, quadratic, and operation-loading interaction terms.
7. **Nonnegative coefficient constraint.** Every learnable coefficient is constrained to be non-negative, so the coefficient matrix satisfies $B \ge 0$ componentwise.
8. **Coupled NNLS estimation.** All coefficient estimation is posed directly as one coupled non-negative least-squares problem in the coefficient matrix.
9. **Output coupling through invariants.** The invariant residual couples the output components through $A$, so the estimator is one multi-output NNLS problem rather than $F$ independent per-output NNLS fits.
10. **Invariant-aware training objective.** Stoichiometric invariants are handled inside the estimation objective through residual terms in ASM component space.
11. **External composition map.** The collapse from ASM component fractions to measured composites is an external calculation performed after prediction through $I_{comp}$.
12. **Composite-sign scope.** Non-negative component predictions imply non-negative measured composites only when the relevant rows of $I_{comp}$ are entrywise non-negative.
13. **Target availability.** The effluent ASM component fractions used for training are assumed available directly or through an upstream state-reconstruction step outside the present model.

These assumptions matter because they narrow the scientific claim. The model guarantees non-negative component predictions by construction, but that guarantee is purchased by a nonnegative-coefficient model class and by component-space supervision. It does not by itself guarantee exact kinetics, thermodynamic feasibility, or universal exact conservation for every unseen operating point.

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

### 5.3 Nonnegative coefficient matrix

The effluent ASM component-fraction prediction is defined directly by

$$
\hat c = B \phi(u, c_{in}),
$$

where $B \in \mathbb{R}_{+}^{F \times D}$ is an entrywise nonnegative coefficient matrix.

Blockwise, the same model may be written as

$$
\hat c
= b
+ W_u u
+ W_{in} c_{in}
+ \Theta_{uu}(u \otimes u)
+ \Theta_{cc}(c_{in} \otimes c_{in})
+ \Theta_{uc}(u \otimes c_{in}),
$$

with

$$
b,\; W_u,\; W_{in},\; \Theta_{uu},\; \Theta_{cc},\; \Theta_{uc}
$$

all entrywise nonnegative.

### 5.4 Why this guarantees non-negative predictions

For the $f$-th ASM component,

$$
\hat c_f = \sum_{d=1}^{D} B_{f d} \, \phi_d(u, c_{in}).
$$

Every term in that sum is non-negative because

$$
 B_{f d} \ge 0
\qquad \text{and} \qquad
\phi_d(u, c_{in}) \ge 0.
$$

Therefore

$$
\hat c_f \ge 0
\qquad \text{for every component } f.
$$

Exact zeros are admissible because the coefficients are required to be non-negative rather than strictly positive. The guarantee delivered by the model class is therefore the non-negative one:

$$
\hat c \in \mathbb{R}_{+}^{F}.
$$

This is the non-negativity mechanism of ICSOR. It does not rely on clipping or any downstream repair step after the regression has been evaluated.

### 5.5 Componentwise interpretation and induced monotonicity

Each predicted ASM component is a nonnegative weighted sum of non-negative monomials. This makes the model easy to audit:

1. $W_u u$ captures first-order operating contributions.
2. $W_{in} c_{in}$ captures first-order influent carry-through and composition effects.
3. $\Theta_{uu}(u \otimes u)$ captures operating curvature.
4. $\Theta_{cc}(c_{in} \otimes c_{in})$ captures influent curvature.
5. $\Theta_{uc}(u \otimes c_{in})$ captures operation-loading interactions.

The same sign structure also has an important consequence: in the native positive coordinate system, the model is monotone non-decreasing in every feature coordinate. That monotonicity is the price of the hard non-negativity guarantee. If a physically decreasing effect must be represented, it must be encoded through the choice of inputs or feature transformations rather than through negative coefficients.

## 6. Invariant-Aware NNLS Estimation

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

For a given nonnegative coefficient matrix $B$, define the prediction matrix

$$
\widehat C(B) = \Phi B^T.
$$

### 6.2 Equivalent NNLS objective in coefficient space

The natural component-space fit minimizes squared component error while also penalizing invariant residuals:

$$
\widehat B
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

For algebraic analysis, this estimator can still be vectorized into one coupled bound-constrained problem. For implementation, however, explicit Kronecker-expanded least-squares systems are usually an unnecessary computational burden. A better realization works directly with the coefficient matrix and applies projected first-order updates to the same smooth convex objective.

### 6.3 Matrix-form convex objective and gradient

For computation it is convenient to transpose the coefficient matrix and work with

$$
W = B^T \in \mathbb{R}_{+}^{D \times F}.
$$

Then the estimator in Section 6.2 becomes

$$
\widehat W
=
\arg\min_{W \in \mathbb{R}_{+}^{D \times F}}
f(W),
\qquad
f(W)
=
\left\|
C_{out} - \Phi W
\right\|_F^2
+
\lambda_{inv}
\left\|
(\Phi W - C_{in}) A^T
\right\|_F^2.
$$

Define the Gram and coupling matrices

$$
G_\Phi = \Phi^T \Phi \in \mathbb{R}^{D \times D},
\qquad
G_A = A^T A \in \mathbb{R}^{F \times F},
\qquad
M = I_F + \lambda_{inv} G_A,
$$

and the cross-term matrix

$$
H
=
\Phi^T C_{out}
+
\lambda_{inv}\Phi^T C_{in} G_A
\in \mathbb{R}^{D \times F}.
$$

Then the objective is a smooth convex quadratic in $W$, and its gradient is

$$
\nabla f(W)
=
2\left(
G_\Phi W M - H
\right).
$$

Equivalently, if one defines $w = \operatorname{vec}(W)$, the same quadratic has Hessian

$$
Q
=
2\left(
M \otimes G_\Phi
\right)
=
2\left(
(I_F + \lambda_{inv} A^T A)\otimes(\Phi^T \Phi)
\right)
\succeq 0.
$$

This shows that the estimator is convex, but it also makes clear why the explicit stacked least-squares realization is unnecessary for implementation. The physics remains exactly the same, yet the optimization can be carried out directly in the native $D \times F$ coefficient matrix rather than through a Kronecker-expanded residual operator with $N(F+q)$ rows.

The solve is still one coupled multi-output NNLS problem. The outputs remain linked through $M$ and therefore through $A^T A$. Solving $F$ independent per-target NNLS problems is not equivalent unless the invariant coupling vanishes.

### 6.4 Practical matrix-form coefficient estimation

The recommended implementation is a projected first-order method applied directly to $W$. A basic projected-gradient step is

$$
W^{(k+1)}
=
\Pi_{\mathbb{R}_{+}^{D \times F}}
\left(
W^{(k)} - \alpha_k \nabla f(W^{(k)})
\right)
=
\max\left(
0,\;
W^{(k)} - \alpha_k \nabla f(W^{(k)})
\right),
$$

where the projection is taken componentwise onto the non-negative orthant. An accelerated projected-gradient variant may be used when faster convergence is desired, but the estimator itself is unchanged.

A practical implementation proceeds as follows.

1. Build $\Phi$, $C_{in}$, $C_{out}$, and $A$.
2. Precompute $G_\Phi = \Phi^T \Phi$, $G_A = A^T A$, $M = I_F + \lambda_{inv} G_A$, and $H = \Phi^T C_{out} + \lambda_{inv}\Phi^T C_{in} G_A$.
3. Choose any nonnegative initialization $W^{(0)}$. A zero start is admissible, and warm starts are natural when the estimator is refit repeatedly.
4. Iterate projected updates using $\nabla f(W) = 2(G_\Phi W M - H)$.
5. Stop when a projected-gradient optimality residual, a KKT residual, or a relative objective decrease falls below the desired tolerance.
6. Return $\widehat B = \widehat W^T$.

At any minimizer $\widehat W$, the coordinatewise Karush-Kuhn-Tucker conditions are

$$
\widehat W \ge 0,
\qquad
\nabla f(\widehat W) \ge 0,
\qquad
\widehat W \odot \nabla f(\widehat W) = 0,
$$

where $\odot$ denotes the Hadamard product. Because the feasible set is the non-negative orthant and $f$ is convex, these conditions characterize global optimality.

A safe fixed step size satisfies

$$
0 < \alpha_k \le \frac{1}{L},
\qquad
L = 2\,\lambda_{\max}(G_\Phi)\,\lambda_{\max}(M),
$$

because $L$ is a Lipschitz constant for $\nabla f$ under the Frobenius norm. Backtracking or other monotone line-search rules may also be used.

For the basic projected-gradient method, any fixed step size $\alpha \in (0, 1/L]$ gives a descent iteration. Since the problem is convex and $\nabla f$ is $L$-Lipschitz, the projected-gradient iterates converge to a global minimizer of the coefficient-estimation problem. Accelerated projected methods can improve convergence speed, but the plain projected-gradient method is the cleanest baseline because its optimality conditions and descent property are transparent.

This matrix-form implementation is computationally attractive because it stores only $G_\Phi \in \mathbb{R}^{D \times D}$, $M \in \mathbb{R}^{F \times F}$, $H \in \mathbb{R}^{D \times F}$, the current iterate, and a few work arrays. It does not materialize the tall Kronecker-expanded design with $N(F+q)$ rows and $FD$ columns. After the one-time precomputations, each iteration is dominated by matrix multiplications of order $O(D^2F + DF^2)$, which is usually much cheaper than repeatedly operating on an explicit stacked system when $N$ is large.

### 6.5 What this formulation guarantees and what it does not

The formulation guarantees the following.

1. **Nonnegative coefficients.** The learned coefficient matrix is nonnegative because the estimator imposes $B \ge 0$ directly.
2. **Non-negative ASM component predictions.** Since $\phi(u, c_{in}) \ge 0$, the predicted component vector $\hat c = B\phi(u, c_{in})$ is non-negative.
3. **Component-space training target.** The model is trained on ASM component fractions directly.
4. **Invariant-aware estimation.** Conservation enters estimation through the invariant residual term rather than through a downstream correction rule.
5. **External measurement collapse.** Conversion to measured aggregates is outside the model.

These guarantees depend on the estimator itself, not on whether it is solved by projected gradient, an accelerated projected method, or another equivalent convex optimizer.

It does not guarantee the following.

1. **Universal exact conservation at every unseen point.** With a finite penalty $\lambda_{inv}$, invariants are enforced through the estimation objective and are not automatically exact at every unseen point.
2. **Full biological realizability.** Non-negative component fractions are necessary but not sufficient for full process feasibility.
3. **Monotonic flexibility in both directions.** Because the coefficients are nonnegative, decreasing effects are not represented through negative slopes in the native feature basis.

If exact equality-constrained fitting of the invariant residual is required at the estimation stage, one must replace the penalized objective by a different estimator with explicit linear equality constraints. The matrix-form projected solve described above computes the penalized nonnegative problem stated in Section 6.2; it does not turn that soft penalty into a hard equality constraint.

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

Because the model is trained directly on $C_{out}$, the primary inferential object is the nonnegative component-space coefficient matrix $B$ itself. The training targets live in ASM component space from the start.

The model assumes component-space targets are available and fits the component map directly.

### 8.2 Rank deficiency, duplicated monomials, and coefficient interpretation

Direct component-space training removes one identifiability problem, but it does not remove all of them.

1. If $N < D$, the feature design cannot have full column rank and $G_\Phi = \Phi^T \Phi$ is singular.
2. Even when $N \ge D$, the operating domain may weakly excite some directions of the second-order basis, making $G_\Phi$ ill conditioned.
3. The unsymmetrized second-order basis contains symmetric duplicates such as $u_i u_j$ and $u_j u_i$ unless the basis is compressed, which enlarges $D$ and therefore enlarges the coefficient matrix and the Gram matrix used by the projected solve.

Under these conditions different nonnegative coefficient matrices can produce similar fitted predictions. The non-negative coefficient restriction stabilizes the problem by removing sign cancellations, but it does not create uniqueness when the feature space itself is poorly excited. The same conditioning issues also matter computationally: ill-conditioned $G_\Phi$ can slow first-order convergence, and duplicated monomials increase both memory use and arithmetic cost without adding new physics.

Therefore interpretation should focus on the following objects in order of reliability:

1. predicted ASM component fractions,
2. blockwise contribution patterns of $W_u$, $W_{in}$, $\Theta_{uu}$, $\Theta_{cc}$, and $\Theta_{uc}$,
3. individual coefficients only when the design matrix is sufficiently informative.

### 8.3 Practical matrix-form NNLS solvers

The recommended implementation is a matrix-form projected-gradient method, optionally accelerated by an extrapolated projected-gradient variant, applied directly to $W = B^T$. This solves the same coupled NNLS estimator as Section 6.2 without explicitly building the Kronecker-expanded least-squares system.

The reason for that recommendation is computational rather than conceptual. The explicit stacked design is algebraically valid, but it scales with $N(F+q)$ rows and $FD$ columns. In contrast, the matrix-form implementation precomputes $G_\Phi = \Phi^T\Phi$, $G_A = A^TA$, and $H = \Phi^T C_{out} + \lambda_{inv}\Phi^T C_{in} G_A$, then iterates only on $W \in \mathbb{R}^{D \times F}$. That usually gives a much smaller memory footprint and a cleaner path to warm starts and repeated refits.

Three practical points matter.

1. The solve is coupled across outputs; solving $F$ independent per-target NNLS problems is not equivalent unless the invariant coupling disappears.
2. Efficiency comes from staying in matrix form; explicit Kronecker assembly is useful for derivation but not recommended as the runtime path.
3. Feature scaling must preserve the intended non-negative coordinate system; arbitrary centering that creates negative feature coordinates breaks the direct non-negativity argument of the model.

If an application later introduces additional linear equality or inequality constraints beyond simple nonnegativity, a more general constrained optimizer may become appropriate. That is a change in computational machinery or even in the estimator, not a reason to abandon the matrix-form realization of the present problem.

### 8.4 Deployment after training

Once $\widehat B$ has been estimated, the deployed component-space predictor is

$$
\hat c(u, c_{in}) = \widehat B\phi(u, c_{in}).
$$

This is the native output of ICSOR. If an application requires measured composites, the external reporting vector is

$$
\hat y_{ext}(u, c_{in}) = I_{comp}\widehat B\phi(u, c_{in}).
$$

That second equation is a reporting formula, not a redefinition of the model target.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Active set and local free coefficients

Let

$$
\widehat W = \widehat B^T,
\qquad
\widehat w = \operatorname{vec}(\widehat W)
$$

be the fitted NNLS solution. Define the active index set

$$
\mathcal{A} = \{ j : \widehat w_j > 0 \},
$$

and let $\widehat w_{\mathcal A}$ denote the subvector of coefficients whose fitted values are strictly above zero.

For the vectorized quadratic objective, define the Hessian

$$
Q
=
2\left(
(I_F + \lambda_{inv} A^T A)\otimes(\Phi^T \Phi)
\right).
$$

Let $Q_{\mathcal A}$ denote the principal submatrix associated with the active coordinates. The bound-constrained estimator is convex, but it is not ordinary unconstrained least squares. Its local behavior depends both on which coefficients remain free at the solution and on the curvature carried by the free-coordinate Hessian block $Q_{\mathcal A}$.

### 9.2 Local covariance approximation conditional on an active set

Standard OLS covariance formulas do not transfer directly because the nonnegativity constraints can pin some coefficients at zero. A common local approximation treats the active set as fixed in a neighborhood of the solution and uses only the free-coordinate curvature:

$$
\operatorname{Cov}(\widehat w_{\mathcal{A}} \mid \mathcal{A}\ \text{fixed})
\approx
2\widehat\sigma^2 Q_{\mathcal{A}}^{-1},
$$

provided $Q_{\mathcal{A}}$ is nonsingular.

Here

$$
\widehat\sigma^2
=
\frac{
\left\|
C_{out} - \Phi \widehat W
\right\|_F^2
+
\lambda_{inv}
\left\|
(\Phi \widehat W - C_{in}) A^T
\right\|_F^2
}{m - |\mathcal{A}|},
$$

and $m = NF + Nq$ is the dimension of the conceptual stacked residual vector associated with the two Frobenius-norm terms, even though the implementation need not form that stacked system explicitly.

This approximation is conditional and local. It ignores uncertainty from active-set changes and becomes fragile when the free-coordinate Hessian is ill conditioned or nearly singular.

### 9.3 Prediction uncertainty for ASM components and for external composites

For a new sample $(u_*, c_{in,*})$, define $\phi_* = \phi(u_*, c_{in,*})$. The component prediction can be written as

$$
\hat c_* = H_* \widehat w,
\qquad
H_* = I_F \otimes \phi_*^T.
$$

Conditioning again on a fixed active set gives the local approximation

$$
\operatorname{Var}(\hat c_* \mid \mathcal{A}\ \text{fixed})
\approx
H_{*,\mathcal{A}}
\left(
2\widehat\sigma^2 Q_{\mathcal{A}}^{-1}
\right)
H_{*,\mathcal{A}}^T,
$$

If external measured composites are later computed, their approximate covariance follows directly:

$$
\operatorname{Var}(\hat y_{ext,*} \mid \mathcal{A}\ \text{fixed})
\approx
I_{comp}\operatorname{Var}(\hat c_*) I_{comp}^T.
$$

This equation is useful because it preserves the logical order of the formulation: uncertainty is computed first in the native ASM component space and only then pushed forward through the external composition matrix.

### 9.4 Why resampling is still preferred

Despite the local active-set formulas above, bootstrap refitting remains the more defensible default for the deployed model. The reasons are straightforward.

1. The fitted solution depends on an active set induced by the orthant projection, so the estimator changes regime when coefficients move on or off the boundary.
2. The nonnegative orthant restriction creates boundary effects in coefficient space even though the objective is smooth in the interior.
3. The invariant penalty couples the outputs through $A^T A$, so perturbations in one target can alter the fitted coefficients of the others.
4. Rank deficiency, duplicated monomials, and ill-conditioned Gram matrices can make local covariance estimates numerically fragile.

Therefore the recommended uncertainty treatment for the final deployed predictor is bootstrap refitting in component space, followed by optional external collapse through $I_{comp}$ when measured-space summaries are required.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct ASM component prediction changes the inferential target

The model targets the effluent ASM component fractions directly. This makes the surrogate scientifically closer to the mechanistic state description, but it also requires training data in that same ASM basis.

### 10.2 Nonnegative coefficients build sign safety into the model class

The non-negativity guarantee is not a numerical afterthought. It comes from the model class itself: non-negative features and nonnegative coefficients imply non-negative predicted components. Exact zeros remain admissible because the coefficient bounds are nonnegative rather than strictly positive. The tradeoff is expressive restriction. In the native coordinate system, the model is additive and monotone in the selected features. Any effect that should behave like a decrease must be represented through the design of the input coordinates rather than through negative coefficients.

### 10.3 Invariants are handled during estimation rather than after prediction

Stoichiometric invariants act as training-time structural guidance. The invariant residuals are part of the coupled NNLS objective, so conservation influences the fitted coefficients directly.

### 10.4 Measurement collapse becomes a reporting decision

Because the model predicts ASM component fractions only, the composition matrix becomes a downstream reporting choice. This is useful in practice because one can analyze the same trained model under different measured-output reporting conventions without retraining the surrogate.

## 11. Limitations

ICSOR is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state in the quasi-steady-sample sense and does not represent temporal dynamics or path dependence.
2. It requires effluent ASM component-fraction targets for training, either directly from a simulator or from an upstream reconstruction step.
3. The non-negativity guarantee depends on maintaining a non-negative input coordinate system and a non-negative monomial feature map.
4. Nonnegative coefficients imply a monotone additive model in the native feature coordinates, which can underrepresent genuinely decreasing or sign-changing physical effects unless those are encoded through feature design.
5. Exact zero predictions are allowed by the model class, but whether they appear in practice depends on the fitted active set and the evaluated feature pattern rather than on a separate sparsity mechanism.
6. With a finite invariant penalty $\lambda_{inv}$, stoichiometric conservation is enforced through estimation pressure and is not exact by construction for every deployed sample.
7. If exact equality-constrained fitting of the invariants is mandatory, the present penalized NNLS estimator must be replaced by a distinct equality-constrained formulation.
8. The second-order feature basis can be statistically fragile when it is weakly excited or highly collinear, and the same ill-conditioning can slow first-order convergence.
9. The unsymmetrized quadratic basis contains duplicated monomials unless it is compressed, so it can inflate $D$, enlarge the Gram matrix, and reduce individual coefficient interpretability.
10. Local active-set covariance formulas are only approximations for the fitted box-constrained estimator.
11. Non-negative component predictions imply non-negative externally reported composites only when the relevant rows of $I_{comp}$ are entrywise non-negative.
12. A misspecified stoichiometric matrix or incorrect system boundary leads to an invariant term that is mathematically consistent with the wrong physical system.
13. The composition matrix itself is treated as known and fixed; uncertainty in $I_{comp}$ is outside the present error model.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its guarantees correctly.

## 12. Conclusion

ICSOR is formulated here as a direct ASM component-space surrogate. It takes operational variables and influent ASM component fractions as input, and it predicts effluent ASM component fractions in the same basis. The model is non-negative by construction because its feature vector is non-negative and its coefficient matrix is constrained to be nonnegative. Coefficient estimation remains one coupled NNLS problem, and a practical realization works directly with the native coefficient matrix and solves the smooth convex penalized problem by projected matrix-form updates rather than by explicitly assembling a Kronecker-expanded least-squares system.

Stoichiometric invariants enter the estimation problem as explicit component-space residuals. Measured composites are obtained afterward as an external calculation through the composition matrix. Under that reading, ICSOR is best understood as a nonnegative, second-order, component-space surrogate whose primary output is the ASM component-fraction vector itself, whose measurement layer remains external, and whose coefficient estimator should be implemented in the smallest native matrix form that preserves the coupled physics.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Lawson, C. L., and Hanson, R. J. Solving Least Squares Problems. SIAM, 1995.
4. Bates, D. M., and Watts, D. G. Nonlinear Regression Analysis and Its Applications. Wiley, 1988.
5. Nocedal, J., and Wright, S. J. Numerical Optimization. 2nd ed. Springer, 2006.
6. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
