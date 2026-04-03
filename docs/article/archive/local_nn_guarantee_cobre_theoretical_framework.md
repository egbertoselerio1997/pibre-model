# Constrained Orthogonal Bilinear Regression (COBRE) for Activated Sludge Surrogate Modeling

## Abstract

This article presents a revised theoretical framework for Constrained Orthogonal Bilinear Regression (COBRE), a physics-informed surrogate model for steady-state activated-sludge systems. The purpose of COBRE is to predict measured effluent variables from operating conditions and influent activated-sludge-model (ASM) component concentrations while preserving the stoichiometric invariants implied by the adopted reaction network. The key difficulty is that the conservation laws are defined in ASM component space, whereas plant observations are usually reported as composite variables such as total COD, total nitrogen, total phosphorus, or suspended solids. A regression model built only in measured-output space can fit those aggregates while still implying an impossible redistribution of the underlying ASM components. COBRE resolves that mismatch in two stages. First, a partitioned second-order surrogate produces an unconstrained prediction of the effluent ASM component state. Second, an orthogonal projection maps that raw prediction onto the affine set of states that satisfy the invariant relations induced by the stoichiometric matrix. The projected component state is then collapsed into measured output space through a linear composition map.

The framework is written as a self-contained theory section for readers in chemical engineering, wastewater process modeling, and machine learning. All symbols are defined before use. The physical assumptions are stated explicitly. The invariant constraint is derived from the stoichiometric change relation rather than asserted heuristically. The projection is derived step by step. The distinction between identifiable measured-space coefficients and non-identifiable latent component-space coefficients is made explicit. The article also formulates an end-to-end constrained variant designed to preserve non-negativity of intermediate ASM component predictions on the fitted support without resorting to post-hoc clipping. The result is a precise and reproducible formulation that clarifies both what COBRE guarantees and what it does not guarantee.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to be a steady-state operating condition: the operating variables, influent composition, and effluent response are treated as time-invariant over the control volume being modeled. The usual difficulty is that a generic data-driven regressor can fit observed effluent data while still violating fundamental conservation structure. In activated-sludge modeling, that failure is not a minor technical detail. It undermines the physical credibility of the surrogate because it can imply component inventories that are inconsistent with the adopted reaction network even when the measured aggregates appear plausible.

The source of the problem is a mismatch between two spaces.

1. The mechanistic stoichiometric model is written in an ASM component basis, such as soluble substrate, ammonium, nitrate, autotrophic biomass, particulate organics, phosphate, dissolved oxygen, and alkalinity.
2. The plant or simulator often reports outputs in measured composite variables, such as total COD, total nitrogen, total phosphorus, TSS, or VSS.

These two spaces are related, but they are not the same. Conservation laws are naturally expressed in the ASM component basis because the stoichiometric matrix acts on individual components. Observations, however, are usually available only after those components have been aggregated into measurable composites. A surrogate that learns only in measured-output space may reproduce the observed aggregates while obscuring physically impossible changes in the underlying component inventory.

COBRE is designed to address exactly that mismatch. Given operating conditions and an influent ASM component state, it predicts the steady-state effluent in a way that is flexible enough to capture nonlinear operating-loading interactions and structured enough to preserve the stoichiometric invariants implied by the chosen reaction network. The model is constructed to answer one precise question:

> Given a steady-state influent state and a steady-state operating condition, what measured effluent state should be predicted if the underlying effluent ASM component state must remain consistent with the conserved quantities implied by the adopted stoichiometric model?

The theory in this article is restricted to steady-state reactor-block prediction. It does not aim to replace a dynamic activated-sludge simulator. Rather, it provides an analytically constrained surrogate that preserves the most important stoichiometric structure while remaining simple enough to estimate directly from data. The discussion proceeds from physical scope and notation, to derivation of the invariant constraint, to projection, to collapse into measured space, and finally to estimation, uncertainty, and an end-to-end non-negativity-constrained extension.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit operated at steady state. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes transport or removal mechanisms such as bypass streams, gas stripping, chemical dosing, or sludge wastage if they cross the chosen boundary and are not encoded in the stoichiometric description. The theory therefore applies only after the modeler has fixed the following items:

1. the reactor or process block being represented,
2. the ASM component basis used to describe material composition,
3. the stoichiometric matrix associated with that basis, and
4. the measurement map used to aggregate component concentrations into observed composite variables.

The framework is steady-state. It does not represent settling dynamics, sludge age dynamics, sensor dynamics, start-up transients, or time-varying trajectories. Changing the system boundary changes the admissible stoichiometric change space and therefore changes the invariant projector itself.

### 2.2 Why two state spaces are needed

To make the distinction concrete, suppose the underlying component basis contains soluble biodegradable substrate, particulate biodegradable substrate, ammonium, nitrate, phosphate, dissolved oxygen, alkalinity, and biomass fractions. A plant rarely measures all of those components directly. Instead, it may report total COD, total nitrogen, total phosphorus, TSS, and VSS. Those measured variables are linear combinations of the component concentrations under a chosen analytical convention.

The surrogate must therefore operate across two linked spaces:

1. ASM component space, where stoichiometry and conserved quantities are defined.
2. Measured composite space, where prediction targets are observed and evaluated.

COBRE learns and constrains the prediction in component space and only then maps the result to measured space. That order is essential. The conservation structure originates in the component basis, not in the aggregated measurement basis.

Before introducing symbols, it is useful to separate three distinct objects that will appear repeatedly. The first is the component state, which is the detailed ASM description used by the stoichiometric model. The second is the measured state, which is the aggregate laboratory or simulator output actually reported to the engineer. The third is the admissible change space, which contains only those component-state changes that can be generated by the adopted stoichiometric matrix. COBRE first predicts in the component basis, then removes the part of that prediction that violates the admissible change structure, and only then converts the result into measured variables.

### 2.3 Notation

Single-sample vectors are written as column vectors. Dataset matrices are defined later with samples stored by rows.

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $u$ | $\mathbb{R}^{M_{op}}$ | Operational input vector, for example hydraulic retention time, aeration intensity, recycle ratio, or other manipulated or design variables |
| $c_{in}$ | $\mathbb{R}^{F}$ | Influent ASM component concentration vector |
| $c_{out}$ | $\mathbb{R}^{F}$ | True steady-state effluent ASM component concentration vector |
| $c_{raw}$ | $\mathbb{R}^{F}$ | Unconstrained surrogate prediction of the effluent ASM component concentration vector |
| $c^*$ | $\mathbb{R}^{F}$ | Projected effluent ASM component prediction that satisfies the invariant relations |
| $y$ | $\mathbb{R}^{K}$ | Measured effluent composite vector |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM component concentrations to measured composite variables |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction progress vector expressed in concentration-equivalent units so that $\nu^T \xi$ has the same units as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank matrix whose rows form a basis of $\operatorname{null}(\nu)$ |
| $P_{inv}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the row space of $A$ |
| $P_{adm}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the admissible change space, $I_F - P_{inv}$ |
| $\phi(u, c_{in})$ | $\mathbb{R}^{D}$ | Engineered second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $B$ | $\mathbb{R}^{F \times D}$ | Raw component-space coefficient matrix |
| $G$ | $\mathbb{R}^{K \times F}$ | Measured-space constrained operator, $G = I_{comp} P_{adm}$ |
| $H$ | $\mathbb{R}^{K \times F}$ | Measured-space invariant carry-through operator, $H = I_{comp} P_{inv}$ |
| $M$ | $\mathbb{R}^{K \times D}$ | Effective identifiable measured-space coefficient matrix, $M = G B$ |

The measured effluent variables are defined by the linear map

$$
y = I_{comp} c_{out}
$$

This linear composition map is standard in activated-sludge modeling when measured variables are aggregates of ASM components. For example, total COD or total nitrogen is formed by summing the relevant component concentrations with appropriate conversion factors.

With the spaces and notation fixed, the next step is to derive exactly which combinations of ASM components must be preserved by the adopted reaction network.

## 3. Modeling Assumptions

The framework rests on the following assumptions. These are not optional preferences left to the reader. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a steady-state input-output condition. The model is not a dynamic state estimator.
2. **Fixed component basis.** The ASM component basis and the associated stoichiometric matrix are fixed before regression begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement. Any external source or sink outside that boundary is outside the present model.
4. **Linear composition map.** Measured effluent variables are linear combinations of the underlying ASM component concentrations through $I_{comp}$.
5. **Direct effluent-state parameterization.** The surrogate is parameterized to predict the effluent ASM component state $c_{out}$ directly rather than the component change $c_{out} - c_{in}$. This keeps the learned target aligned with the final quantity of practical interest while letting the projection restore the invariant component explicitly.
6. **Second-order surrogate class.** The raw surrogate is a partitioned second-order polynomial model that includes linear, quadratic, and operation-loading interaction terms.
7. **Euclidean correction metric.** The physical correction is defined as the smallest Euclidean adjustment in ASM component concentration space. This choice yields a unique closed-form projector but is not physically neutral to coordinate scaling.
8. **Constraint scope.** The projection enforces only the stoichiometric invariants implied by the chosen basis and system boundary. It does not enforce nonnegativity, upper bounds, kinetic feasibility, or thermodynamic admissibility beyond those invariants.
9. **Statistical scope.** Classical coefficient and prediction intervals are interpreted under independent-sample multivariate linear-model assumptions, and exact finite-sample $t$-based intervals require Gaussian errors and an identifiable coefficient parameterization.

These assumptions matter because each one narrows the scientific claim. A prediction that satisfies the invariant relations is invariant-consistent under the chosen model. It is not automatically guaranteed to be fully physically realizable in every operating regime.

## 4. Stoichiometric Structure and Conserved Quantities

### 4.1 From stoichiometric reactions to component-state change

Let $\nu \in \mathbb{R}^{R \times F}$ be the stoichiometric matrix written in the adopted ASM component basis. For one steady-state sample, define the net reaction progress vector $\xi \in \mathbb{R}^{R}$ so that

$$
c_{out} - c_{in} = \nu^T \xi
$$

This equation is the starting point of the theory. It says that the net change in the effluent component state is a linear combination of reaction stoichiometries. The entries of $\xi$ need not be observed individually. They collect the net progression of each modeled reaction over the chosen control volume after scaling into concentration-equivalent units. For example, if reaction $i$ has steady-state rate $r_i$ in units of concentration per time and the relevant hydraulic time scale of the control volume is $\tau$, then one admissible definition is $\xi_i = r_i \tau$, which has concentration units. More generally, $\xi_i$ may be interpreted as the net integrated reaction extent over the control volume after whatever normalization is required so that $\nu^T \xi$ is expressed in the same units as $c_{out} - c_{in}$.

That definition makes the statement dimensionally coherent: both $c_{out} - c_{in}$ and $\nu^T \xi$ live in the same ASM component concentration space. Without that scaling convention, the conservation equation would be ambiguous and different readers could implement different, incompatible normalizations.

### 4.2 Invariant relations implied by the stoichiometric matrix

The reaction progress vector $\xi$ is not part of the surrogate model and is usually not observed. To eliminate it, we introduce a matrix $A \in \mathbb{R}^{q \times F}$ whose rows form a basis of $\operatorname{null}(\nu)$. By construction,

$$
\nu a^T = 0
$$

for every row vector $a$ of $A$, and therefore

$$
A \nu^T = 0
$$

Multiplying the stoichiometric change relation by $A$ gives

$$
A(c_{out} - c_{in}) = A \nu^T \xi = 0
$$

which implies the affine invariant relation

$$
A c_{out} = A c_{in}
$$

Each row of $A$ represents one independent conserved combination of ASM components under the adopted stoichiometric model and system boundary. The exact physical interpretation depends on the chosen basis and stoichiometric matrix. In activated-sludge applications, the conserved combinations often correspond to pools such as COD equivalents, nitrogen equivalents, phosphorus equivalents, or charge-related balances, provided those balances are preserved by the modeled reaction network and the selected boundary.

### 4.3 Why the basis of $A$ is not unique

The matrix $A$ is not unique. If $R_A \in \mathbb{R}^{q \times q}$ is invertible, then $\widetilde A = R_A A$ generates the same constraint set because

$$
\widetilde A c = \widetilde A c_{in}
\quad \Longleftrightarrow \quad
R_A A c = R_A A c_{in}
\quad \Longleftrightarrow \quad
A c = A c_{in}
$$

Thus, the physics is carried by the row space of $A$, not by one particular numerical basis. This matters because the final orthogonal projector should depend only on the subspace being enforced, not on an arbitrary basis used to represent it.

### 4.4 Minimal worked example

Consider two components, $c_1$ and $c_2$, and one reaction that converts $c_1$ into $c_2$ without net loss:

$$
\nu = \begin{bmatrix}
-1 & 1
\end{bmatrix}
$$

Then

$$
\operatorname{null}(\nu) = \operatorname{span}\left\{ \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right\}
$$

so one admissible choice is

$$
A = \begin{bmatrix} 1 & 1 \end{bmatrix}
$$

The invariant relation becomes

$$
c_{out,1} + c_{out,2} = c_{in,1} + c_{in,2}
$$

The reaction may redistribute material between the two components, but it cannot change the conserved total represented by $c_1 + c_2$. This is the intuition behind the null-space construction: the stoichiometric model allows motion in some directions of component space and forbids motion in others.

## 5. Unconstrained Surrogate in ASM Component Space

### 5.1 Why the input is partitioned

In activated-sludge systems, operating conditions and influent component concentrations play different physical roles.

1. Operating variables such as hydraulic retention time, dissolved-oxygen setpoint, or recycle settings alter the process environment.
2. Influent component concentrations describe the material inventory entering that environment.

Treating those two groups as interchangeable predictors hides an important engineering distinction. COBRE therefore partitions the input into an operational block $u$ and an influent component block $c_{in}$.

### 5.2 Feature map

We define the second-order feature map

$$
\phi(u, c_{in}) = \begin{bmatrix}
1 \\
u \\
c_{in} \\
u \otimes u \\
c_{in} \otimes c_{in} \\
u \otimes c_{in}
\end{bmatrix} \in \mathbb{R}^{D}
$$

where $\otimes$ denotes the Kronecker product. The quadratic blocks are retained in explicit vectorized form rather than reduced to a symmetry-compressed basis. That choice keeps the algebra transparent and fixes one unambiguous design basis for estimation.

Throughout the chapter, we use the convention $u \otimes u = \operatorname{vec}(u u^T)$, $c_{in} \otimes c_{in} = \operatorname{vec}(c_{in} c_{in}^T)$, and $u \otimes c_{in} = \operatorname{vec}(u c_{in}^T)$ under column-wise vectorization. The resulting feature dimension is therefore

$$
D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F
$$

Retaining the full vectorized quadratic blocks avoids hidden indexing conventions and makes the later estimation problem unambiguous.

### 5.3 Raw effluent-state surrogate

The unconstrained surrogate is defined by

$$
c_{raw} = B \phi(u, c_{in})
$$

or, blockwise,

$$
c_{raw} = b + W_u u + W_{in} c_{in} + \Theta_{uu}(u \otimes u) + \Theta_{cc}(c_{in} \otimes c_{in}) + \Theta_{uc}(u \otimes c_{in})
$$

with parameter blocks

$$
W_u \in \mathbb{R}^{F \times M_{op}}, \quad
W_{in} \in \mathbb{R}^{F \times F}, \quad
b \in \mathbb{R}^{F}
$$

$$
\Theta_{uu} \in \mathbb{R}^{F \times M_{op}^2}, \quad
\Theta_{cc} \in \mathbb{R}^{F \times F^2}, \quad
\Theta_{uc} \in \mathbb{R}^{F \times (M_{op}F)}
$$

Each block has a physical interpretation.

1. $W_u u$ captures first-order operating effects.
2. $W_{in} c_{in}$ captures direct carry-through and first-order dependence on influent composition.
3. $\Theta_{uu}(u \otimes u)$ captures nonlinear interactions among operating variables.
4. $\Theta_{cc}(c_{in} \otimes c_{in})$ captures nonlinear dependence on influent composition.
5. $\Theta_{uc}(u \otimes c_{in})$ captures the operation-loading interaction that motivates the COBRE name.

The model is therefore not purely bilinear in the strict algebraic sense because it includes self-quadratic terms as well as cross terms. The name COBRE is retained because the operational-loading interaction remains the defining structural idea, but the mathematical class should be understood precisely as a partitioned second-order regression model with bilinear cross interactions.

The raw surrogate is flexible enough to capture curvature and interaction, but it is data-driven and unconstrained. There is no reason for $c_{raw}$ to satisfy the invariant relation $A c_{raw} = A c_{in}$ without an additional correction step. The next section therefore introduces the constrained correction that separates learned variation from invariant carry-through.

## 6. Orthogonal Projection onto the Invariant-Consistent Affine Set

### 6.1 Feasible set

For a fixed influent state $c_{in}$, define the feasible affine set

$$
\mathcal{S}(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in} \}
$$

This set contains exactly those effluent component states whose conserved combinations match the influent conserved combinations under the adopted stoichiometric model. Because the right-hand side depends on $c_{in}$, the feasible set is affine rather than linear.

### 6.2 Projection problem

COBRE corrects the raw surrogate by solving

$$
\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2} \lVert c - c_{raw} \rVert_2^2
$$

subject to

$$
A c = A c_{in}
$$

This optimization asks for the smallest Euclidean adjustment in ASM component concentration space that restores the invariant relations. The metric matters: by choosing the Euclidean norm, we declare that distance is measured directly in the adopted component coordinates. If the components are rescaled, the correction changes. That dependence is not a flaw in the derivation; it is a consequence of the chosen metric.

Because the objective has Hessian $I_F$, the optimization problem is strictly convex. Combined with the affine equality constraint, that guarantees a unique projected state $c^*$ for each pair $(c_{raw}, c_{in})$.

### 6.3 Lagrange-multiplier derivation

Introduce a multiplier vector $\lambda \in \mathbb{R}^{q}$ and define the Lagrangian

$$
\mathcal{L}(c, \lambda) = \frac{1}{2}(c - c_{raw})^T(c - c_{raw}) + \lambda^T(A c - A c_{in})
$$

Differentiating with respect to $c$ and setting the gradient to zero gives

$$
\nabla_c \mathcal{L} = c - c_{raw} + A^T \lambda = 0
$$

so that

$$
c = c_{raw} - A^T \lambda
$$

This equation already has a geometric meaning: the correction from $c_{raw}$ to the projected state lies in the span of the constraint normals, namely the row space of $A$.

Substituting into the constraint yields

$$
A(c_{raw} - A^T \lambda) = A c_{in}
$$

which simplifies to

$$
A A^T \lambda = A(c_{raw} - c_{in})
$$

Because $A$ has full row rank, $A A^T$ is symmetric positive definite and therefore invertible. Hence

$$
\lambda = (A A^T)^{-1} A(c_{raw} - c_{in})
$$

Substituting back gives

$$
c^* = c_{raw} - A^T(A A^T)^{-1} A(c_{raw} - c_{in})
$$

This is the unique Euclidean projection of $c_{raw}$ onto the feasible affine set $\mathcal{S}(c_{in})$.

### 6.4 Projector form and interpretation

Define the orthogonal projector onto the row space of $A$ by

$$
P_{inv} = A^T(A A^T)^{-1} A
$$

and define the complementary projector by

$$
P_{adm} = I_F - P_{inv}
$$

Then the projected state can be written compactly as

$$
c^* = P_{adm} c_{raw} + P_{inv} c_{in}
$$

This formula is the core expression of COBRE.

1. $P_{adm} c_{raw}$ keeps the component of the raw prediction that lies in the admissible change directions.
2. $P_{inv} c_{in}$ restores the conserved component that must match the influent state.

The formula also makes basis invariance transparent. If $A$ is replaced by another full-row-rank basis with the same row space, then $P_{inv}$ and $P_{adm}$ remain unchanged. The projection depends on the invariant subspace itself, not on a particular basis used to represent it.

The earlier two-component example makes this correction concrete. If $c_{raw}$ predicts a total $c_{raw,1} + c_{raw,2}$ that is too large relative to $c_{in,1} + c_{in,2}$, the projector removes exactly the violating component in the direction $[1, 1]^T$ while leaving the admissible redistribution direction $[-1, 1]^T$ unchanged. In that simple case, the projection can be interpreted as removing only the part of the raw prediction that changes the conserved total.

### 6.5 What the projection guarantees and what it does not

The projection guarantees that

$$
A c^* = A c_{in}
$$

for every sample. It does not guarantee the following.

1. It does not guarantee nonnegative component concentrations.
2. It does not guarantee realistic upper bounds or process-feasible operating ranges.
3. It does not impose kinetic feasibility beyond the chosen invariant relations.
4. It does not correct errors introduced by a misspecified stoichiometric basis or an incorrect system boundary.

These are not derivation errors. They are the exact consequences of the constraint set being enforced. A rigorous remedy for non-negativity requires changing the estimator itself, not clipping the projected state after fitting.

## 7. Collapse from ASM Component Space to Measured Output Space

### 7.1 Measured output equation

Practical prediction targets are usually measured composite variables rather than ASM component concentrations. The projected measured output is therefore

$$
y^* = I_{comp} c^*
$$

Substituting the projected state gives

$$
y^* = I_{comp}(P_{adm} c_{raw} + P_{inv} c_{in})
$$

and substituting the raw surrogate gives

$$
y^* = I_{comp} P_{adm} B \phi(u, c_{in}) + I_{comp} P_{inv} c_{in}
$$

Define

$$
G = I_{comp} P_{adm}, \qquad H = I_{comp} P_{inv}
$$

Then the final measured-output model becomes

$$
y^* = G B \phi(u, c_{in}) + H c_{in}
$$

To make the measurement map concrete, consider a reduced ASM-like component basis

$$
c = \begin{bmatrix}
S_{COD} \\
X_{COD} \\
S_{NH_4} \\
S_{NO_3} \\
S_{PO_4}
\end{bmatrix}
$$

in which COD-bearing and nutrient-bearing components are already expressed in their reporting units. If the measured outputs are total COD, total nitrogen, and total phosphorus, one admissible composition matrix is

$$
I_{comp} = \begin{bmatrix}
1 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 1 & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

so that $y = [\text{COD}_{tot}, \text{TN}, \text{TP}]^T = I_{comp} c$. In a full ASM implementation, the same construction is used but some rows may include conversion factors rather than only zeros and ones. The important point is that $I_{comp}$ is fixed by the measurement convention before regression begins.

### 7.2 Effective measured-space coefficients

Only the product $G B$ acts on the engineered features in measured-output space. We therefore define the effective measured-space coefficient matrix

$$
M = G B \in \mathbb{R}^{K \times D}
$$

so that

$$
y^* = M \phi(u, c_{in}) + H c_{in}
$$

This form clarifies the division of roles.

1. $M \phi(u, c_{in})$ captures the identifiable data-driven contribution after stoichiometric correction and measurement collapse.
2. $H c_{in}$ carries the invariant component of the influent state directly into the predicted measured output.

The term $H c_{in}$ is not an ad hoc adjustment. It is the unavoidable algebraic consequence of enforcing the invariant relations before collapsing to measured space. For example, if one row of $A$ represents a conserved nitrogen pool, then the corresponding part of $H c_{in}$ simply transfers that invariant nitrogen content of the influent into the predicted measured output, while $M \phi(u, c_{in})$ explains how the non-invariant part of the effluent responds to operating condition and influent composition.

### 7.3 Why projection must occur before collapse

The order of operations matters. Conservation relations originate in component space because the stoichiometric matrix acts on ASM components. If the prediction were first collapsed to measured space and only then corrected, part of the invariant structure could be lost or become unobservable. Enforcing the constraint in component space is therefore stronger than attempting to repair only the measured-output prediction. With the measured-space model now fixed, the next step is to ask what the available data can actually identify.

### 7.4 Blockwise measured-output model and interpretation

The compact form

$$
y^* = G B \phi(u, c_{in}) + H c_{in}
$$

is algebraically convenient, but it compresses the block structure that made the raw component-space model easy to read. That structure can be restored in measured-output space by partitioning the raw coefficient matrix $B$ conformably with the feature map. Using the same blocks introduced in Section 5.3,

$$
B = \begin{bmatrix}
b & W_u & W_{in} & \Theta_{uu} & \Theta_{cc} & \Theta_{uc}
\end{bmatrix}
$$

and therefore

$$
M = G B = \begin{bmatrix}
b_y & W_{u,y} & W_{in,y} & \Theta_{uu,y} & \Theta_{cc,y} & \Theta_{uc,y}
\end{bmatrix}
$$

with effective measured-space blocks defined by

$$
b_y = G b, \quad
W_{u,y} = G W_u, \quad
W_{in,y} = G W_{in}
$$

$$
\Theta_{uu,y} = G \Theta_{uu}, \quad
\Theta_{cc,y} = G \Theta_{cc}, \quad
\Theta_{uc,y} = G \Theta_{uc}
$$

Substituting these blocks into the measured-output model gives

$$
y^* = b_y + W_{u,y} u + W_{in,y} c_{in} + \Theta_{uu,y}(u \otimes u) + \Theta_{cc,y}(c_{in} \otimes c_{in}) + \Theta_{uc,y}(u \otimes c_{in}) + H c_{in}
$$

or, after collecting the two first-order influent terms,

$$
y^* = b_y + W_{u,y} u + (W_{in,y} + H)c_{in} + \Theta_{uu,y}(u \otimes u) + \Theta_{cc,y}(c_{in} \otimes c_{in}) + \Theta_{uc,y}(u \otimes c_{in})
$$

This blockwise expression restores the same interpretive separation that was available in component space, but now directly at the level of measured effluent variables. The block $b_y$ is the baseline measured-output offset. The block $W_{u,y} u$ captures first-order operating effects on the measured outputs. The block $W_{in,y} c_{in}$ captures the learned first-order dependence of measured outputs on influent composition that is not fixed by the invariant constraint. The block $H c_{in}$ is different in kind: it is not learned from measured-output regression coefficients but is the fixed invariant carry-through implied by the stoichiometric projection. The three quadratic blocks preserve the same interpretation as in the raw surrogate, now in measured-output space: $\Theta_{uu,y}(u \otimes u)$ captures curvature and interaction among operating variables, $\Theta_{cc,y}(c_{in} \otimes c_{in})$ captures curvature in influent composition, and $\Theta_{uc,y}(u \otimes c_{in})$ captures operation-loading interactions.

This form is useful because it decomposes each predicted measured output into additive term contributions for any given sample. For one operating point $(u, c_{in})$, the prediction can be written as

$$
y^* = y_{bias} + y_{u,lin} + y_{c,lin} + y_{inv} + y_{uu} + y_{cc} + y_{uc}
$$

where

$$
y_{bias} = b_y, \quad
y_{u,lin} = W_{u,y} u, \quad
y_{c,lin} = W_{in,y} c_{in}, \quad
y_{inv} = H c_{in}
$$

$$
y_{uu} = \Theta_{uu,y}(u \otimes u), \quad
y_{cc} = \Theta_{cc,y}(c_{in} \otimes c_{in}), \quad
y_{uc} = \Theta_{uc,y}(u \otimes c_{in})
$$

For output component $k$, this gives the scalar decomposition

$$
y_k^* = [y_{bias}]_k + [y_{u,lin}]_k + [y_{c,lin}]_k + [y_{inv}]_k + [y_{uu}]_k + [y_{cc}]_k + [y_{uc}]_k
$$

which explains why the predicted value takes its final level for that sample. Variable-level contributions can then be read from the same blocks. For example,

$$
[y_{u,lin}]_k = \sum_{j=1}^{M_{op}} (W_{u,y})_{k j} u_j
$$

and

$$
[y_{c,lin} + y_{inv}]_k = \sum_{f=1}^{F} \left[(W_{in,y})_{k f} + H_{k f}\right] (c_{in})_f
$$

so the linear contribution of each operating variable or influent component is explicit. Likewise, the interaction block expands as

$$
[y_{uc}]_k = \sum_{j=1}^{M_{op}} \sum_{f=1}^{F} (\Theta_{uc,y})_{k,(j,f)} u_j (c_{in})_f
$$

and analogous expansions hold for $y_{uu}$ and $y_{cc}$. These decompositions are local rather than global: the contribution of a variable depends on the current sample because the quadratic and interaction terms depend on products of input values. The model therefore supports interpretation of which terms and variables are driving a particular prediction without claiming that one fixed effect size applies uniformly over the entire operating domain.

This blockwise rewriting does not change the identifiability question. The quantities that measured composite data identify are the effective measured-space blocks that make up $M = G B$, together with the known carry-through operator $H$. By contrast, the latent component-space blocks inside $B$ remain generally non-unique unless extra structure is imposed. The next section makes that distinction precise at the dataset level.

## 8. Estimation and Identifiability from Measured Composite Data

### 8.1 Dataset-level model

Let $N$ steady-state samples be available, and store samples by rows:

$$
\Phi = \begin{bmatrix}
\phi(u_1, c_{in,1})^T \\
\phi(u_2, c_{in,2})^T \\
\vdots \\
\phi(u_N, c_{in,N})^T
\end{bmatrix} \in \mathbb{R}^{N \times D}
$$

$$
C_{in} = \begin{bmatrix}
c_{in,1}^T \\
c_{in,2}^T \\
\vdots \\
c_{in,N}^T
\end{bmatrix} \in \mathbb{R}^{N \times F}
$$

$$
Y = \begin{bmatrix}
y_1^T \\
y_2^T \\
\vdots \\
y_N^T
\end{bmatrix} \in \mathbb{R}^{N \times K}
$$

The measured-output model is

$$
Y = \Phi M^T + C_{in} H^T + E
$$

where $E \in \mathbb{R}^{N \times K}$ collects model and measurement errors. The formulation assumes that the influent state $c_{in}$ is available in ASM component coordinates for every sample. In a simulation study, that information may be available directly from the mechanistic model. In plant applications, it may instead come from a soft sensor, a prior state estimator, or a reconstruction from measured aggregate influent variables. The present theory treats that component-space influent state as given; uncertainty in its reconstruction is outside the current error model and should be handled separately if material.

Define the transformed target

$$
\widetilde Y = Y - C_{in} H^T
$$

Then the estimation equation becomes

$$
\widetilde Y = \Phi M^T + E
$$

This is the correct regression equation for estimation from measured composite data when $A$ and $I_{comp}$ are treated as known.

The phrase measured composite data should therefore be read carefully. In this framework, the measured outputs $Y$ live in composite space, while the influent input $C_{in}$ is assumed to have already been expressed in ASM component space.

### 8.2 What the data identify

Measured composite data generally identify the effective matrix $M = G B$, not the latent raw component-space coefficient matrix $B$ uniquely. The reason is simple. If $N_B \in \mathbb{R}^{F \times D}$ satisfies

$$
G N_B = 0
$$

then

$$
G(B + N_B) = G B = M
$$

Thus, infinitely many different component-space matrices can generate the same measured-space model. The practical implication is important:

1. the primary inferential target is $M$ and its block structure in measured-output space,
2. any reconstructed $B$ is one admissible representative unless extra structure is imposed.

Failing to separate those two objects leads to overinterpretation of non-identifiable latent coefficients.

### 8.3 Least-squares estimator of the effective coefficients

The natural estimator of $M$ is the least-squares solution of

$$
\widehat M^T = \arg\min_{Q \in \mathbb{R}^{D \times K}} \lVert \widetilde Y - \Phi Q \rVert_F^2
$$

The Moore-Penrose solution is

$$
\widehat M^T = \Phi^+ \widetilde Y
$$

or equivalently

$$
\widehat M = \widetilde Y^T (\Phi^+)^T
$$

If $\Phi$ has full column rank, then

$$
\Phi^+ = (\Phi^T \Phi)^{-1} \Phi^T
$$

and therefore

$$
\widehat M^T = (\Phi^T \Phi)^{-1} \Phi^T \widetilde Y
$$

The pseudoinverse expression is the general statement. The explicit inverse is only a full-rank special case.

### 8.4 Rank deficiency and interpretability

Second-order feature maps can be high dimensional, and real wastewater datasets may not excite all directions of that design space. If $\Phi$ is rank deficient, the fitted values remain well defined through the pseudoinverse, but the individual coefficients of $M$ are not uniquely identified. In that case, the minimum-norm pseudoinverse returns one representative coefficient matrix, not a unique physical truth. Under rank deficiency, interpretation should focus on fitted predictions or on estimable linear functionals rather than on individual coefficients. In practical terms, the modeler then has three defensible options: reduce the feature basis, regularize the estimation problem, or shift the inferential emphasis from coefficients to predicted outputs and their uncertainty.

### 8.5 Reconstructing one admissible component-space coefficient matrix

If one needs a component-space coefficient matrix after estimating $M$, it must satisfy

$$
G B = M
$$

The minimum-Frobenius-norm solution is

$$
\widehat B_{min} = G^+ M
$$

and the full solution set is

$$
B = G^+ M + (I_F - G^+ G) Z
$$

for arbitrary $Z \in \mathbb{R}^{F \times D}$. The free matrix $Z$ is the algebraic statement of non-identifiability.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Error model

For statistical inference, suppose the row errors of $E$ are independent across samples and satisfy

$$
\mathbb{E}[E \mid \Phi] = 0
$$

and

$$
\operatorname{Var}(\operatorname{vec}(E) \mid \Phi) = \Omega \otimes I_N
$$

where $\Omega \in \mathbb{R}^{K \times K}$ is the within-sample covariance across measured outputs. This allows, for example, COD and total nitrogen errors to be correlated within the same sample.

### 9.2 Full-rank coefficient covariance

If $\Phi$ has full column rank $D$, then

$$
\widehat M^T = (\Phi^T \Phi)^{-1} \Phi^T \widetilde Y
$$

and the coefficient covariance is

$$
\operatorname{Var}(\operatorname{vec}(\widehat M^T) \mid \Phi) = \Omega \otimes (\Phi^T \Phi)^{-1}
$$

Therefore, for the coefficient $\widehat M_{k,j}$,

$$
SE(\widehat M_{k,j}) = \sqrt{\Omega_{kk} \left[(\Phi^T \Phi)^{-1}\right]_{jj}}
$$

This formula shows two sources of uncertainty: intrinsic output noise through $\Omega_{kk}$ and poor excitation of feature direction $j$ through the design-conditioning term.

### 9.3 Estimating the output covariance

Let the fitted residual matrix be

$$
\widehat E = \widetilde Y - \Phi \widehat M^T
$$

If $\Phi$ has full column rank and $N > D$, the usual unbiased estimator of the within-sample output covariance is

$$
\widehat \Omega = \frac{\widehat E^T \widehat E}{N - D}
$$

If $\Phi$ is rank deficient, coefficientwise covariance formulas require additional care because individual coefficients are not uniquely identified. In that setting, interval estimation is better focused on fitted predictions, bootstrap distributions, or regularized estimands rather than on raw coefficient entries.

### 9.4 Mean-prediction uncertainty

For a new operating point with feature vector $\phi_* = \phi(u_*, c_{in,*})$, the fitted mean output is

$$
\widehat y_* = \widehat M \phi_* + H c_{in,*}
$$

If $\Phi$ has full column rank, define the leverage factor

$$
s_* = \phi_*^T (\Phi^T \Phi)^{-1} \phi_*
$$

Then

$$
\operatorname{Var}(\widehat y_* \mid \phi_*, c_{in,*}, \Phi) = s_* \Omega
$$

and the standard error of the fitted mean for output $k$ is

$$
SE_{mean,k}(\phi_*) = \sqrt{s_* \, \Omega_{kk}}
$$

Under Gaussian errors, the corresponding coefficientwise $(1-\alpha)$ confidence interval is

$$
\widehat y_{*,k} \pm t_{1-\alpha/2,\, N-D} \sqrt{s_* \, [\widehat \Omega]_{kk}}
$$

### 9.5 Prediction intervals for a future observation

If a future observed output satisfies

$$
y_{future,*} = M \phi_* + H c_{in,*} + e_*
$$

with

$$
\mathbb{E}[e_* \mid \phi_*, c_{in,*}] = 0, \qquad
\operatorname{Var}(e_* \mid \phi_*, c_{in,*}) = \Omega
$$

and $e_*$ independent of the training data, then

$$
\operatorname{Var}(y_{future,*} - \widehat y_* \mid \phi_*, c_{in,*}, \Phi) = (1 + s_*) \Omega
$$

Hence the prediction standard error for output $k$ is

$$
SE_{pred,k}(\phi_*) = \sqrt{(1 + s_*) \, \Omega_{kk}}
$$

and the corresponding Gaussian prediction interval is

$$
\widehat y_{*,k} \pm t_{1-\alpha/2,\, N-D} \sqrt{(1 + s_*) \, [\widehat \Omega]_{kk}}
$$

Prediction intervals are wider than mean-confidence intervals because they include both parameter-estimation uncertainty and irreducible sample-to-sample variability.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct effluent-state parameterization

The surrogate is parameterized on the effluent component state rather than on the net change. This keeps the learned target aligned with the quantity ultimately used for reporting and decision support. The cost is that the role of the influent state enters twice: once inside the feature map and once inside the invariant carry-through term $P_{inv} c_{in}$. That is not redundancy. The first role captures empirical dependence; the second role enforces the conserved component of the state exactly.

The distinction can be stated operationally. If operating conditions change while the invariant component of the influent stays fixed, the term inside the feature map allows the model to change the predicted effluent response. By contrast, $P_{inv} c_{in}$ cannot be learned away by the regression because it is the exact part of the state that the stoichiometric constraint says must persist regardless of how the unconstrained fit behaves.

### 10.2 Partitioned second-order feature structure

The partitioned feature map separates operating effects, influent-composition effects, and operation-loading interactions in a way that is interpretable to process engineers. The price of that interpretability is rapid feature growth, which can create multicollinearity, unstable coefficients, and weakly identified directions if the dataset does not adequately excite the design space. Since $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$, even moderate values of $M_{op}$ and $F$ can produce a feature basis that is large relative to the sample count. The resulting model can still interpolate training data through the pseudoinverse while leaving individual coefficients poorly determined.

### 10.3 Euclidean projection metric

The Euclidean projection gives a unique closed-form correction and keeps the constrained step analytically simple. At the same time, it is sensitive to the scaling of component coordinates. If one component is numerically much larger or measured in a much different effective scale than another, the Euclidean metric may assign disproportionate influence to that direction. A weighted projection may be preferable in some applications, but that would be a different model from the one defined here.

### 10.4 Projection before measurement collapse

Enforcing the invariant relations before collapsing to measured space is a substantive modeling decision, not a notational convenience. It preserves the constraints in the space where the stoichiometric matrix is actually defined. Once the state is collapsed into measured composites, some physically meaningful directions may no longer be separately visible.

### 10.5 Effective coefficients versus latent component-space coefficients

The effective coefficients $M$ are the correct objects for direct engineering interpretation because they act directly on observed outputs. The latent component-space coefficients $B$ are useful only when accompanied by a clear statement that they are generally not unique. Presenting one reconstructed $B$ without that warning would overstate what the data actually determine.

## 11. End-to-End Non-Negativity-Constrained COBRE

### 11.1 Why non-negativity must be handled during training

The baseline COBRE construction in Sections 5 through 10 is exact with respect to the affine invariant relation

$$
A c^* = A c_{in}
$$

but it does not guarantee that the projected ASM component vector $c^*$ is elementwise nonnegative. That gap cannot be repaired cleanly by post-hoc clipping. If a clipped state $	ilde c$ is defined by replacing negative entries of $c^*$ with zero, then in general

$$
A \tilde c \neq A c_{in}
$$

so exact invariant preservation is lost for that sample. The resulting clipped model is therefore no longer the model defined in Sections 4 through 10. If non-negativity is scientifically required, it must be built into the estimation problem itself.

This section formulates an end-to-end constrained COBRE variant whose coefficients are learned directly against the final projected measured outputs while simultaneously enforcing non-negativity of the intermediate projected ASM component predictions over the fitted support.

### 11.2 When nonnegative ASM components imply nonnegative measured composites

The final measured prediction is

$$
y^* = I_{comp} c^*
$$

so non-negativity of $c^*$ alone is not sufficient unless the relevant rows of the composition map are monotone in the component concentrations.

Let $\mathcal{K}_+ \subseteq \{1, \dots, K\}$ denote the set of measured outputs whose composition rows are elementwise nonnegative:

$$
k \in \mathcal{K}_+ \quad \Longleftrightarrow \quad (I_{comp})_{k f} \ge 0 \text{ for all } f = 1, \dots, F
$$

Then the following implication is immediate.

**Proposition 1 (Monotone-collapse sufficiency).** If $c^* \ge 0$ elementwise, then for every $k \in \mathcal{K}_+$,

$$
y_k^* = \sum_{f=1}^{F} (I_{comp})_{k f} c_f^* \ge 0
$$

because each summand is nonnegative. In particular, if all rows of $I_{comp}$ are elementwise nonnegative, then

$$
c^* \ge 0 \quad \Longrightarrow \quad y^* \ge 0
$$

elementwise.

This proposition identifies the exact additional condition needed to transfer a component-space non-negativity guarantee into measured-output space. The guarantee is therefore output-specific whenever some rows of $I_{comp}$ contain negative coefficients.

### 11.3 Admissible-state parameterization with exact invariants

To preserve the affine invariant relation exactly while avoiding post-hoc repair, it is convenient to parameterize the predicted effluent state directly inside the admissible affine family. Introduce an unconstrained matrix $\widetilde B \in \mathbb{R}^{F \times D}$ and define the admissible coefficient matrix

$$
B_{adm} = P_{adm} \widetilde B
$$

Then define the constrained effluent-state predictor by

$$
c_{\theta}^*(u, c_{in}) = B_{adm} \phi(u, c_{in}) + P_{inv} c_{in}
$$

where $\theta$ denotes the free parameters inside $\widetilde B$. Since

$$
A P_{adm} = 0, \qquad A P_{inv} = A
$$

it follows immediately that

$$
A c_{\theta}^*(u, c_{in}) = A B_{adm} \phi(u, c_{in}) + A P_{inv} c_{in} = A c_{in}
$$

for every input pair $(u, c_{in})$. Thus the invariant relation is built into the parameterization itself rather than restored afterward.

The same partitioned second-order structure is retained because $B_{adm}$ can be written blockwise as

$$
B_{adm} = \begin{bmatrix}
b_{adm} & W_{u,adm} & W_{in,adm} & \Theta_{uu,adm} & \Theta_{cc,adm} & \Theta_{uc,adm}
\end{bmatrix}
$$

so the engineering interpretation of operational, influent, quadratic, and interaction contributions is preserved.

### 11.4 End-to-end constrained estimation against the final projected outputs

For sample $i$, define

$$
c_{\theta,i}^* = P_{adm} \widetilde B \, \phi(u_i, c_{in,i}) + P_{inv} c_{in,i}
$$

and

$$
y_{\theta,i}^* = I_{comp} c_{\theta,i}^*
$$

The end-to-end constrained estimator is then defined by the optimization problem

$$
\widehat{\widetilde B}
= \arg\min_{\widetilde B}
\sum_{i=1}^{N} \left\| y_i - I_{comp}\left(P_{adm} \widetilde B \, \phi(u_i, c_{in,i}) + P_{inv} c_{in,i}\right) \right\|_2^2
+ \lambda \| \widetilde B \|_F^2
$$

subject to the samplewise inequalities

$$
P_{adm} \widetilde B \, \phi(u_i, c_{in,i}) + P_{inv} c_{in,i} \ge \varepsilon \mathbf{1}_F,
\qquad i = 1, \dots, N
$$

where $\lambda \ge 0$ is a regularization parameter and $\varepsilon \ge 0$ is an optional positivity margin. The case $\varepsilon = 0$ enforces non-negativity exactly, while $\varepsilon > 0$ imposes a strictly positive safety buffer on the fitted support.

This is an end-to-end training objective in the precise sense relevant here: the loss is evaluated on the final projected measured outputs $y_{\theta,i}^*$, but the inequality constraints are enforced on the corresponding intermediate projected ASM component states $c_{\theta,i}^*$. The coefficients are therefore learned against the final nonnegative composite targets through the same optimization that imposes the component-space feasibility condition.

The problem is a convex quadratic program. The objective is quadratic in the free coefficients of $\widetilde B$, and each componentwise inequality constraint is affine in those same coefficients because $\phi(u_i, c_{in,i})$, $P_{adm}$, and $P_{inv} c_{in,i}$ are fixed for sample $i$. Consequently, exact affine invariants and fitted-support non-negativity can be enforced simultaneously without introducing a post-hoc stage.

### 11.5 Identifiability, interpretability, and certification under the constrained estimator

The identifiability distinction from Section 8 remains essential. Measured composite data still identify the measured-space action of the constrained model more directly than they identify one unique component-space coefficient matrix. The regularization term $\lambda \| \widetilde B \|_F^2$ should therefore be interpreted as part of the model definition: it selects one admissible representative from a generally non-unique family while stabilizing estimation in high-dimensional second-order feature spaces.

Measured-space interpretation is preserved. Define

$$
M_{adm} = I_{comp} B_{adm}
$$

Then the constrained measured-output model is

$$
y_{\theta}^* = M_{adm} \phi(u, c_{in}) + H c_{in}
$$

which has exactly the same additive block structure as Section 7.4. The baseline offset, first-order operating effects, first-order influent effects, invariant carry-through, and second-order interaction terms can therefore still be decomposed and interpreted sample by sample.

What changes is the guarantee scope. If the fitted solution satisfies

$$
c_{\theta,i}^* \ge 0
\qquad \text{for every fitted sample } i
$$

then the model is certified to be nonnegative on that fitted support. If, in addition, the relevant output rows of $I_{comp}$ belong to $\mathcal{K}_+$, then Proposition 1 certifies non-negativity of those final measured outputs on the same support. This is a rigorous data-domain guarantee, not a universal guarantee over all possible inputs. Extending the guarantee beyond the observed support would require an additional bounded-domain certification analysis that is outside the present article.

The practical consequence is clear. If interpretability and exact invariant preservation are both to be retained, then the principled route is not to clip projected predictions after fitting. The principled route is to learn an admissible-state parameterization directly from the final measured targets under hard component-space non-negativity constraints.

## 12. Limitations

COBRE is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state and does not represent temporal dynamics or path dependence.
2. It enforces only the invariant relations encoded by the chosen stoichiometric basis and system boundary.
3. In the baseline projection-after-surrogate formulation, nonnegative ASM component concentrations are not guaranteed after projection. Section 11 gives an end-to-end constrained remedy on the fitted support, but that remedy changes the estimator and should not be conflated with post-hoc clipping.
4. Even under the constrained estimator, non-negativity of final measured composites follows from component non-negativity only for outputs whose composition-map rows are elementwise nonnegative.
5. The constrained estimator provides a fitted-support guarantee, not a global guarantee over all possible operating and influent conditions.
6. It depends on the assumed linear composition map from ASM component space to measured-output space.
7. It can be statistically fragile if the second-order feature basis is weakly excited or highly collinear.
8. Classical coefficient and prediction intervals require identifiable coefficients and the usual multivariate linear-model assumptions. Under constrained estimation, bootstrap or other resampling-based uncertainty quantification may be more appropriate than closed-form least-squares formulas.
9. A misspecified stoichiometric matrix or incorrect system boundary leads to a formally correct projection onto the wrong physical constraint set.
10. If the influent ASM component state is reconstructed from measured aggregate variables rather than observed directly, reconstruction error enters upstream of the regression and is not represented by the output-noise covariance formulas derived here.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its claims correctly.

## 13. Conclusion

COBRE combines a partitioned second-order surrogate with an orthogonal projection derived from stoichiometric invariants. The framework is useful for wastewater applications because it preserves the distinction between operating conditions and influent composition, enforces conservation structure where that structure naturally lives, and returns predictions in the measured variables used by plant operators and simulation studies.

The central theoretical point is that measured composite data identify the effective measured-space operator $M$, not the latent component-space coefficient matrix $B$ uniquely. Once that distinction is made explicit, the model becomes conceptually cleaner, the estimation problem becomes more precise, and the uncertainty analysis becomes easier to interpret. Under that reading, baseline COBRE is best understood as an analytically constrained steady-state surrogate for activated-sludge prediction: more physically disciplined than a generic black-box regressor, but narrower in scope than a full dynamic mechanistic simulator.

When non-negativity of intermediate ASM components is also required, the clean extension is not a post-hoc correction. The clean extension is the end-to-end constrained formulation of Section 11, which parameterizes the admissible state directly, regresses coefficients against the final projected measured outputs, and preserves the additive block interpretation while enforcing fitted-support non-negativity of the projected ASM state. For outputs whose composition-map rows are elementwise nonnegative, that component-space guarantee transfers directly to the final measured predictions.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
4. Seber, G. A. F., and Lee, A. J. Linear Regression Analysis. 2nd ed. Wiley, 2003.
5. Rao, C. R., and Mitra, S. Generalized Inverse of Matrices and Its Applications. Wiley, 1971.
6. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.