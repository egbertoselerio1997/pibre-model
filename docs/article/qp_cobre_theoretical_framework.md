# Non-Negative Constrained Orthogonal Bilinear Regression (COBRE) for Activated Sludge Surrogate Modeling

## Abstract

This article presents a non-negative formulation of Constrained Orthogonal Bilinear Regression (COBRE), a physics-informed surrogate model for steady-state activated-sludge systems. The purpose of COBRE is to predict measured effluent variables from operating conditions and influent activated-sludge-model (ASM) component concentrations while preserving the stoichiometric invariants implied by the adopted reaction network and enforcing non-negativity of the predicted effluent ASM component state. The key difficulty is that conservation laws are defined in ASM component space, whereas plant observations are usually reported as composite variables such as total COD, total nitrogen, total phosphorus, or suspended solids. A regression model built only in measured-output space can fit those aggregates while still implying an impossible redistribution of the underlying ASM components. An affine invariant projection resolves only part of that mismatch because it restores stoichiometric consistency but can still produce negative component predictions. The non-negative COBRE formulation therefore uses a two-stage correction in component space. First, a partitioned second-order surrogate produces an unconstrained prediction of the effluent ASM component state. Second, a convex projection maps that raw prediction onto the intersection of the invariant-consistent affine set and the nonnegative orthant. The corrected component state is then collapsed into measured output space through a linear composition map.

The framework is written as a self-contained theory section for readers in chemical engineering, wastewater process modeling, and machine learning. All symbols are defined before use. The invariant constraint is derived from the stoichiometric change relation rather than asserted heuristically. The non-negative correction is formulated as a strictly convex quadratic program, and the role of the earlier orthogonal affine projector is retained explicitly as a reference solution and inactive-constraint special case. In deployment, that quadratic program is needed only when the raw prediction is infeasible and the closed-form affine projector still violates componentwise non-negativity. The distinction between identifiable affine measured-space coefficients and non-identifiable latent component-space coefficients is preserved, while the limits of exact closed-form uncertainty analysis for the final inequality-constrained predictor are stated explicitly. The result is a precise formulation of what non-negative COBRE guarantees, what remains estimated by least squares, and what must instead be handled through convex post-estimation correction.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to be a steady-state operating condition: the operating variables, influent composition, and effluent response are treated as time-invariant over the control volume being modeled. The usual difficulty is that a generic data-driven regressor can fit observed effluent data while still violating fundamental conservation structure. In activated-sludge modeling, that failure is not a minor technical detail. It undermines the physical credibility of the surrogate because it can imply component inventories that are inconsistent with the adopted reaction network even when the measured aggregates appear plausible.

The source of the problem is a mismatch between two spaces.

1. The mechanistic stoichiometric model is written in an ASM component basis, such as soluble substrate, ammonium, nitrate, autotrophic biomass, particulate organics, phosphate, dissolved oxygen, and alkalinity.
2. The plant or simulator often reports outputs in measured composite variables, such as total COD, total nitrogen, total phosphorus, TSS, or VSS.

These two spaces are related, but they are not the same. Conservation laws are naturally expressed in the ASM component basis because the stoichiometric matrix acts on individual components. Observations, however, are usually available only after those components have been aggregated into measurable composites. A surrogate that learns only in measured-output space may reproduce the observed aggregates while obscuring physically impossible changes in the underlying component inventory.

Classical COBRE addresses that mismatch by predicting in component space and projecting the raw surrogate output onto the affine set consistent with the stoichiometric invariants. That construction repairs invariant violations, but it does not ensure that the projected component concentrations are non-negative. Negative component predictions are a serious defect in the present setting because component concentrations are themselves physical quantities and because negative components can propagate into implausible reported composites. The non-negative COBRE formulation developed here therefore strengthens the correction step. The affine invariant projection remains part of the derivation, but the deployed predictor is the closest point to the raw surrogate that satisfies both the invariant relations and componentwise non-negativity.

The model is constructed to answer one precise question:

> Given a steady-state influent state and a steady-state operating condition, what measured effluent state should be predicted if the underlying effluent ASM component state must satisfy the conserved quantities implied by the adopted stoichiometric model and must remain non-negative componentwise?

The theory in this article is restricted to steady-state reactor-block prediction. It does not aim to replace a dynamic activated-sludge simulator. Rather, it provides an analytically structured surrogate that preserves stoichiometric structure, enforces non-negativity at the deployed component-state level, and remains simple enough that its affine core can still be estimated directly from data by least squares. In deployment, the correction is evaluated in stages: no correction when the raw component state is already feasible, closed-form affine projection when only the invariant equalities are violated, and quadratic-program correction only when non-negativity remains violated after the affine step. The discussion proceeds from physical scope and notation, to derivation of the invariant relations, to convex non-negative projection, to collapse into measured space, and finally to estimation and uncertainty.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit operated at steady state. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes transport or removal mechanisms such as bypass streams, gas stripping, chemical dosing, or sludge wastage if they cross the chosen boundary and are not encoded in the stoichiometric description. The theory therefore applies only after the modeler has fixed the following items:

1. the reactor or process block being represented,
2. the ASM component basis used to describe material composition,
3. the stoichiometric matrix associated with that basis, and
4. the measurement map used to aggregate component concentrations into observed composite variables.

The framework is steady-state. It does not represent settling dynamics, sludge age dynamics, sensor dynamics, start-up transients, or time-varying trajectories. Changing the system boundary changes the admissible stoichiometric change space and therefore changes the invariant and non-negative feasible sets.

### 2.2 Why two state spaces are needed

To make the distinction concrete, suppose the underlying component basis contains soluble biodegradable substrate, particulate biodegradable substrate, ammonium, nitrate, phosphate, dissolved oxygen, alkalinity, and biomass fractions. A plant rarely measures all of those components directly. Instead, it may report total COD, total nitrogen, total phosphorus, TSS, and VSS. Those measured variables are linear combinations of the component concentrations under a chosen analytical convention.

The surrogate must therefore operate across two linked spaces:

1. ASM component space, where stoichiometry, invariants, and non-negativity are defined.
2. Measured composite space, where prediction targets are observed and evaluated.

COBRE learns and constrains the prediction in component space and only then maps the result to measured space. That order remains essential in the non-negative formulation. Stoichiometric invariants originate in the component basis, and the componentwise non-negativity claim is also made in that basis. A measured-space-only correction would generally be too weak to control the underlying ASM state.

### 2.3 Notation

Single-sample vectors are written as column vectors. Dataset matrices are defined later with samples stored by rows.

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $u$ | $\mathbb{R}^{M_{op}}$ | Operational input vector, for example hydraulic retention time, aeration intensity, recycle ratio, or other manipulated or design variables |
| $c_{in}$ | $\mathbb{R}^{F}$ | Influent ASM component concentration vector |
| $c_{out}$ | $\mathbb{R}^{F}$ | True steady-state effluent ASM component concentration vector |
| $c_{raw}$ | $\mathbb{R}^{F}$ | Unconstrained surrogate prediction of the effluent ASM component concentration vector |
| $c_{aff}$ | $\mathbb{R}^{F}$ | Affine orthogonal projection of $c_{raw}$ onto the invariant-consistent set |
| $c^*$ | $\mathbb{R}^{F}$ | Final non-negative projected ASM component prediction |
| $y$ | $\mathbb{R}^{K}$ | Measured effluent composite vector |
| $y_{aff}$ | $\mathbb{R}^{K}$ | Measured output induced by the affine projection $c_{aff}$ |
| $y^*$ | $\mathbb{R}^{K}$ | Final measured output induced by $c^*$ |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM component concentrations to measured composite variables |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction progress vector expressed in concentration-equivalent units so that $\nu^T \xi$ has the same units as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank matrix whose rows form a basis of $\operatorname{null}(\nu)$ |
| $P_{inv}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the row space of $A$ |
| $P_{adm}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the admissible change space, $I_F - P_{inv}$ |
| $N$ | $\mathbb{R}^{F \times (F-q)}$ | Matrix whose columns form an orthonormal basis of $\operatorname{null}(A)$ |
| $\phi(u, c_{in})$ | $\mathbb{R}^{D}$ | Engineered second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $B$ | $\mathbb{R}^{F \times D}$ | Raw component-space coefficient matrix |
| $G$ | $\mathbb{R}^{K \times F}$ | Measured-space affine constrained operator, $G = I_{comp} P_{adm}$ |
| $H$ | $\mathbb{R}^{K \times F}$ | Measured-space invariant carry-through operator, $H = I_{comp} P_{inv}$ |
| $M$ | $\mathbb{R}^{K \times D}$ | Effective identifiable affine measured-space coefficient matrix, $M = G B$ |

The measured effluent variables are defined by the linear map

$$
y = I_{comp} c_{out}
$$

This linear composition map is standard in activated-sludge modeling when measured variables are aggregates of ASM components. For example, total COD or total nitrogen is formed by summing the relevant component concentrations with appropriate conversion factors. In the non-negative COBRE setting, the sign structure of $I_{comp}$ matters because non-negative component predictions imply non-negative measured composites only when the corresponding measurement map is entrywise non-negative.

## 3. Modeling Assumptions

The framework rests on the following assumptions. These are not optional preferences left to the reader. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a steady-state input-output condition. The model is not a dynamic state estimator.
2. **Fixed component basis.** The ASM component basis and the associated stoichiometric matrix are fixed before regression begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement. Any external source or sink outside that boundary is outside the present model.
4. **Linear composition map.** Measured effluent variables are linear combinations of the underlying ASM component concentrations through $I_{comp}$.
5. **Direct effluent-state parameterization.** The surrogate is parameterized to predict the effluent ASM component state $c_{out}$ directly rather than the component change $c_{out} - c_{in}$.
6. **Second-order surrogate class.** The raw surrogate is a partitioned second-order polynomial model that includes linear, quadratic, and operation-loading interaction terms.
7. **Euclidean correction metric.** The deployed correction is defined as the smallest Euclidean adjustment in ASM component concentration space that restores the required constraints.
8. **Constraint scope.** The final projection enforces the stoichiometric invariants implied by the chosen basis and system boundary together with componentwise non-negativity. It does not enforce upper bounds, kinetic feasibility, or thermodynamic admissibility beyond those conditions.
9. **Influent feasibility.** The influent reference state is assumed non-negative in component space. Under that assumption the non-negative feasible set is non-empty because the influent state itself satisfies the invariant equalities.
10. **Composite-sign scope.** Non-negative component predictions imply non-negative measured composites only when the relevant rows of $I_{comp}$ are entrywise non-negative. If the measurement convention uses negative coefficients, extra output-space sign constraints would be required for a composite non-negativity guarantee.
11. **Statistical scope.** Classical least-squares coefficient and affine-core prediction intervals are interpreted under independent-sample multivariate linear-model assumptions. Exact finite-sample $t$-based intervals do not generally extend to the final inequality-constrained deployed predictor.

These assumptions matter because each one narrows the scientific claim. A prediction that satisfies the invariant relations and componentwise non-negativity is physically better disciplined than the affine-only formulation, but it is still not automatically guaranteed to be fully process-realizable in every operating regime.

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

Thus, the physics is carried by the row space of $A$, not by one particular numerical basis. This matters because both the affine orthogonal projector and the later non-negative feasible set depend only on the invariant subspace being enforced, not on an arbitrary basis used to represent it.

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

The reaction may redistribute material between the two components, but it cannot change the conserved total represented by $c_1 + c_2$. In the non-negative COBRE setting, the feasible effluent set is therefore the non-negative line segment satisfying this equality. The later correction step will select the point on that segment closest to the raw surrogate prediction.

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

The raw surrogate is flexible enough to capture curvature and interaction, but it is data-driven and unconstrained. There is no reason for $c_{raw}$ to satisfy the invariant relation $A c_{raw} = A c_{in}$ or the componentwise non-negativity requirement without an additional correction step. The next section therefore introduces the constrained correction that separates learned variation from physically required structure.

## 6. Projection onto the Invariant-Consistent Non-Negative Set

### 6.1 Affine invariant set

For a fixed influent state $c_{in}$, define the invariant-consistent affine set

$$
\mathcal{S}(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in} \}
$$

This set contains exactly those effluent component states whose conserved combinations match the influent conserved combinations under the adopted stoichiometric model. Because the right-hand side depends on $c_{in}$, the set is affine rather than linear.

### 6.2 Orthogonal affine projection as reference solution

Before imposing non-negativity, COBRE can correct the raw surrogate by solving

$$
\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2} \lVert c - c_{raw} \rVert_2^2
$$

subject to

$$
A c = A c_{in}
$$

The solution is the orthogonal affine projection

$$
c_{aff} = P_{adm} c_{raw} + P_{inv} c_{in}
$$

with

$$
P_{inv} = A^T(A A^T)^{-1} A,
\qquad
P_{adm} = I_F - P_{inv}
$$

This expression remains important in the present article for two reasons. First, it provides the simplest invariant-consistent reference solution. Second, it is the exact solution of the final non-negative formulation whenever the orthogonal affine projection already lies in the nonnegative orthant. It is therefore retained as a derivational benchmark rather than discarded.

### 6.3 Non-negative feasible set

The deployed non-negative COBRE predictor is defined on the smaller feasible set

$$
\mathcal{S}_+(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in}, \; c \ge 0 \}
$$

where $c \ge 0$ is understood componentwise. This set intersects the invariant-consistent affine space with the nonnegative orthant. In geometric terms, the affine set removes changes that violate the stoichiometric invariants, while the orthant removes candidate states with negative ASM component concentrations.

### 6.4 Non-negative correction problem

Non-negative COBRE corrects the raw surrogate by solving

$$
\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2} \lVert c - c_{raw} \rVert_2^2
$$

subject to

$$
A c = A c_{in},
\qquad
c \ge 0
$$

The objective asks for the smallest Euclidean adjustment in ASM component concentration space that restores both the invariant relations and componentwise non-negativity. Relative to the affine-only correction, the feasible set is smaller but still convex. The problem is therefore a strictly convex quadratic program with linear equality and inequality constraints.

This formulation can be recentered around the orthogonal affine projection without changing the minimizer. For every feasible $c \in \mathcal{S}_+(c_{in}) \subseteq \mathcal{S}(c_{in})$,

$$
c - c_{raw} = (c - c_{aff}) + (c_{aff} - c_{raw})
$$

and the defining property of the orthogonal projection implies

$$
(c - c_{aff})^T(c_{aff} - c_{raw}) = 0
$$

because $c - c_{aff}$ lies in the admissible subspace parallel to $\mathcal{S}(c_{in})$ whereas $c_{aff} - c_{raw}$ lies in its orthogonal complement. Therefore,

$$
\lVert c - c_{raw} \rVert_2^2 = \lVert c - c_{aff} \rVert_2^2 + \lVert c_{aff} - c_{raw} \rVert_2^2
$$

and minimizing over $\mathcal{S}_+(c_{in})$ is equivalent to solving

$$
\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2} \lVert c - c_{aff} \rVert_2^2
$$

subject to

$$
A c = A c_{in},
\qquad
c \ge 0
$$

The affine projector is therefore not only a theoretical reference solution. It is also the natural center of the residual non-negativity correction in the cases where a quadratic program remains necessary.

### 6.5 Feasibility, existence, and uniqueness

The first question is whether the feasible set can be empty. Under the present modeling assumptions, it is not. If the influent component state is non-negative, then $c = c_{in}$ is feasible because it satisfies

$$
A c_{in} = A c_{in},
\qquad
c_{in} \ge 0
$$

Hence

$$
c_{in} \in \mathcal{S}_+(c_{in})
$$

and the feasible set is non-empty.

Because $\mathcal{S}_+(c_{in})$ is a closed convex set and the objective is continuous, a minimizer exists. Because the Hessian of the objective is $I_F$, the objective is strictly convex, and therefore the minimizer is unique. We denote that unique point by $c^*$.

Thus, for every pair $(c_{raw}, c_{in})$ with $c_{in} \ge 0$, non-negative COBRE produces one and only one corrected component-state prediction $c^*$.

### 6.6 KKT characterization

Introduce multipliers $\lambda \in \mathbb{R}^{q}$ for the equality constraints and $\mu \in \mathbb{R}^{F}$ for the non-negativity constraints. The Lagrangian is

$$
\mathcal{L}(c, \lambda, \mu) = \frac{1}{2}(c - c_{raw})^T(c - c_{raw}) + \lambda^T(A c - A c_{in}) - \mu^T c
$$

with dual feasibility condition $\mu \ge 0$. The Karush-Kuhn-Tucker conditions for the unique minimizer are

$$
c - c_{raw} + A^T \lambda - \mu = 0
$$

$$
A c = A c_{in}
$$

$$
c \ge 0,
\qquad
\mu \ge 0
$$

$$
\mu_i c_i = 0, \qquad i = 1, \dots, F
$$

The stationarity condition shows that the correction from $c_{raw}$ to $c^*$ lies in the sum of two directions: the row space of $A$, which enforces the stoichiometric invariants, and the cone generated by active non-negativity constraints, which prevents components from crossing below zero. Complementary slackness identifies which non-negativity constraints are active at the solution.

### 6.7 Relation to the orthogonal affine projector

The orthogonal affine projection remains embedded in the new formulation as a special case. If the affine projector already satisfies non-negativity, then the non-negative formulation returns exactly the same point.

Indeed, suppose

$$
c_{aff} \ge 0
$$

componentwise. Then $c_{aff} \in \mathcal{S}_+(c_{in})$. But $c_{aff}$ is already the unique minimizer of the same strictly convex objective over the larger set $\mathcal{S}(c_{in})$. Since $\mathcal{S}_+(c_{in}) \subseteq \mathcal{S}(c_{in})$ and contains $c_{aff}$, the minimizer over the smaller set is also $c_{aff}$. Hence

$$
c^* = c_{aff}
\qquad \text{whenever} \qquad
c_{aff} \ge 0
$$

This result provides continuity with the earlier COBRE theory. The non-negative formulation does not replace the orthogonal affine projector arbitrarily; it extends it to the cases where the affine projector would otherwise leave the physically admissible orthant.

### 6.8 Efficient deployment sequence

The deployed correction should be evaluated in a staged order so that the quadratic program is solved only when it is actually needed.

1. Form the raw component prediction $c_{raw}$.
2. If $c_{raw}$ already satisfies $A c_{raw} = A c_{in}$ and $c_{raw} \ge 0$, return $c_{raw}$ directly.
3. Otherwise compute the closed-form affine projection $c_{aff} = P_{adm} c_{raw} + P_{inv} c_{in}$.
4. If $c_{aff} \ge 0$, return $c_{aff}$.
5. Solve the non-negative quadratic program only when one or more components of $c_{aff}$ remain negative.

This order follows directly from the geometry developed above. The first check avoids unnecessary correction when the raw surrogate already lies in the feasible set. The second check uses the orthogonal affine projector as the cheapest exact repair of invariant violations. The quadratic program is therefore a residual correction step for the subset of samples in which the affine projector still leaves the nonnegative orthant.

One degenerate case is worth noting. If $A$ has zero rows, then there is no non-trivial invariant equality to enforce, $c_{aff} = c_{raw}$, and the correction reduces to Euclidean projection onto the nonnegative orthant, namely componentwise clipping at zero. In that case no quadratic program is required.

### 6.9 Reduced null-space formulation

The affine-centered formulation also leads to a reduced problem that is often easier to solve numerically. Let $N \in \mathbb{R}^{F \times (F-q)}$ have orthonormal columns spanning $\operatorname{null}(A)$. Every point in the affine set can then be written as

$$
c = c_{aff} + N z
$$

for some reduced coordinate vector $z \in \mathbb{R}^{F-q}$, because $A c_{aff} = A c_{in}$ and $A N = 0$. The equality constraints are therefore built into the parameterization. Substituting into the affine-centered objective gives

$$
\min_{z \in \mathbb{R}^{F-q}} \; \frac{1}{2} \lVert N z \rVert_2^2
$$

subject to

$$
N z \ge -c_{aff}
$$

Since $N$ has orthonormal columns,

$$
\lVert N z \rVert_2^2 = z^T N^T N z = z^T z
$$

and the reduced problem becomes

$$
\min_{z \in \mathbb{R}^{F-q}} \; \frac{1}{2} z^T z
$$

subject to

$$
N z \ge -c_{aff}
$$

If a non-orthonormal null-space basis is used instead, the reduced Hessian becomes $N^T N$. The feasible set is unchanged, but the orthonormal basis is preferable because it preserves the Euclidean metric and improves conditioning.

This reduced form makes the computational role of the quadratic program more transparent. The solver is no longer enforcing the affine invariant equalities; those have already been resolved exactly by the affine projector. It is only finding the smallest admissible displacement inside the null-space directions that restores non-negativity.

Implementation note: in a fixed model with fixed $A$, the reduced Hessian and constraint matrix are constant across samples, while the lower bound $-c_{aff}$ changes from sample to sample. That structure is convenient for operator-splitting quadratic-program solvers such as OSQP, especially when warm-started from $z = 0$, which corresponds exactly to the affine projector.

### 6.10 What the projection guarantees and what it does not

The non-negative projection guarantees that

$$
A c^* = A c_{in}
$$

and

$$
c^* \ge 0
$$

for every sample. It does not guarantee the following.

1. It does not guarantee realistic upper bounds or process-feasible operating ranges.
2. It does not impose kinetic feasibility beyond the chosen invariant relations and non-negativity.
3. It does not guarantee thermodynamic admissibility.
4. It does not correct errors introduced by a misspecified stoichiometric basis or an incorrect system boundary.

These are not derivation errors. They are the exact consequences of the constraint set being enforced.

## 7. Collapse from ASM Component Space to Measured Output Space

### 7.1 Final measured output equation

Practical prediction targets are usually measured composite variables rather than ASM component concentrations. The component-space correction is applied before any measurement collapse, and the final measured output is therefore

$$
y^* = I_{comp} c^*
$$

This is the deployed prediction reported by non-negative COBRE.

### 7.2 When component non-negativity implies composite non-negativity

If every row of $I_{comp}$ is entrywise non-negative, then component non-negativity implies measured-output non-negativity. The proof is immediate. For any output index $k$,

$$
y_k^* = \sum_{f=1}^{F} (I_{comp})_{k f} c_f^*
$$

and if both $(I_{comp})_{k f} \ge 0$ and $c_f^* \ge 0$ for all $f$, then $y_k^* \ge 0$.

This sufficient condition is satisfied for many standard composite definitions such as sums of COD-bearing, nitrogen-bearing, phosphorus-bearing, or solids-bearing components with non-negative conversion factors. If the chosen measurement convention contains negative coefficients, then component non-negativity alone does not imply composite non-negativity. In that case, one must either accept only the component-level guarantee or enlarge the feasible set to include additional constraints of the form

$$
I_{comp} c \ge 0
$$

for the relevant measured variables. That enlarged formulation is conceptually straightforward but is outside the baseline model analyzed here.

### 7.3 The affine measured-space core

Before positivity constraints activate, the measured prediction induced by the affine projector is

$$
y_{aff} = I_{comp} c_{aff}
$$

Substituting the affine projection gives

$$
y_{aff} = I_{comp}(P_{adm} c_{raw} + P_{inv} c_{in})
$$

and substituting the raw surrogate gives

$$
y_{aff} = I_{comp} P_{adm} B \phi(u, c_{in}) + I_{comp} P_{inv} c_{in}
$$

Define

$$
G = I_{comp} P_{adm}, \qquad H = I_{comp} P_{inv}
$$

Then the affine measured-space model is

$$
y_{aff} = G B \phi(u, c_{in}) + H c_{in}
$$

and, with

$$
M = G B,
$$

it becomes

$$
y_{aff} = M \phi(u, c_{in}) + H c_{in}
$$

This affine relation remains central because it is the identifiable linear core estimated from measured composite data. It is also the exact deployed predictor whenever the non-negativity inequalities are inactive, so the nonlinear correction is a post-estimation deployment step rather than part of the affine-core fit.

### 7.4 Final deployed prediction as affine core plus positivity correction

The final deployed prediction is generally not equal to $y_{aff}$. Define the positivity-correction term

$$
\delta_+(u, c_{in}) = I_{comp}(c^* - c_{aff})
$$

Then

$$
y^* = y_{aff} + \delta_+(u, c_{in})
$$

When the affine projector is already non-negative, $\delta_+(u, c_{in}) = 0$. When one or more non-negativity constraints are active, $\delta_+(u, c_{in})$ is a sample-specific correction induced by the active inequality set. This decomposition is useful because it separates the globally affine, data-identifiable part of the model from the local correction needed to enforce physical admissibility.

### 7.5 Blockwise affine-core interpretation

The compact form

$$
y_{aff} = G B \phi(u, c_{in}) + H c_{in}
$$

compresses the block structure that made the raw component-space model easy to read. That structure can be restored in measured-output space by partitioning the raw coefficient matrix $B$ conformably with the feature map. Using the same blocks introduced in Section 5.3,

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

with effective affine measured-space blocks defined by

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

Substituting these blocks into the affine-core model gives

$$
y_{aff} = b_y + W_{u,y} u + W_{in,y} c_{in} + \Theta_{uu,y}(u \otimes u) + \Theta_{cc,y}(c_{in} \otimes c_{in}) + \Theta_{uc,y}(u \otimes c_{in}) + H c_{in}
$$

or, after collecting the two first-order influent terms,

$$
y_{aff} = b_y + W_{u,y} u + (W_{in,y} + H)c_{in} + \Theta_{uu,y}(u \otimes u) + \Theta_{cc,y}(c_{in} \otimes c_{in}) + \Theta_{uc,y}(u \otimes c_{in})
$$

This blockwise expression remains useful for interpretation, but it must now be interpreted correctly. It decomposes the affine core $y_{aff}$, not necessarily the final deployed prediction $y^*$. The final prediction is obtained by adding the sample-specific positivity correction $\delta_+(u, c_{in})$.

## 8. Estimation and Identifiability from Measured Composite Data

### 8.1 Dataset-level affine-core model

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

The affine measured-space core satisfies

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

This is the correct regression equation for least-squares estimation from measured composite data when $A$ and $I_{comp}$ are treated as known. It identifies the affine-core operator that would generate the deployed prediction whenever the non-negativity constraints are inactive.

### 8.2 What the data identify

Measured composite data generally identify the effective affine matrix $M = G B$, not the latent raw component-space coefficient matrix $B$ uniquely. The reason is simple. If $N_B \in \mathbb{R}^{F \times D}$ satisfies

$$
G N_B = 0
$$

then

$$
G(B + N_B) = G B = M
$$

Thus, infinitely many different component-space matrices can generate the same affine measured-space core. The practical implication is important:

1. the primary inferential target available from measured composite data is $M$ and its block structure in measured-output space,
2. any reconstructed $B$ is one admissible representative unless extra structure is imposed.

Failing to separate those two objects leads to overinterpretation of non-identifiable latent coefficients.

### 8.3 Least-squares estimator of the affine-core coefficients

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

This least-squares stage is unchanged by the introduction of non-negativity. The non-negative COBRE article does not replace coefficient estimation with a nonlinear estimator. It keeps least-squares estimation for the identifiable affine core and applies the non-negative correction after that stage.

### 8.4 Rank deficiency and interpretability

Second-order feature maps can be high dimensional, and real wastewater datasets may not excite all directions of that design space. If $\Phi$ is rank deficient, the fitted affine-core values remain well defined through the pseudoinverse, but the individual coefficients of $M$ are not uniquely identified. In that case, the minimum-norm pseudoinverse returns one representative coefficient matrix, not a unique physical truth. Under rank deficiency, interpretation should focus on fitted affine-core predictions or on estimable linear functionals rather than on individual coefficients. In practical terms, the modeler then has three defensible options: reduce the feature basis, regularize the estimation problem, or shift the inferential emphasis from coefficients to predicted outputs and their uncertainty.

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

### 8.6 Final deployed predictor after estimation

Once an admissible raw coefficient matrix has been chosen, the estimated raw component prediction for a new sample is

$$
\widehat c_{raw} = \widehat B \phi(u, c_{in})
$$

The final deployed non-negative component prediction is then obtained by the same staged component-space logic introduced in Section 6: keep $\widehat c_{raw}$ if it is already feasible, otherwise apply the affine projector, and solve the quadratic program only when the affine projector still violates non-negativity. In compact notation,

$$
\widehat c^* = \operatorname{Proj}_{\mathcal{S}_+(c_{in})}(\widehat c_{raw})
$$

and the final deployed measured prediction is

$$
\widehat y^* = I_{comp} \widehat c^*
$$

This last map is deterministic, but it is not globally affine in $\phi(u, c_{in})$. The least-squares coefficients therefore characterize the identifiable affine core of non-negative COBRE, while the final prediction is produced by augmenting that core with a sample-specific convex correction applied after estimation and before collapse to measured space.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Error model for the affine core

For statistical inference, suppose the row errors of $E$ are independent across samples and satisfy

$$
\mathbb{E}[E \mid \Phi] = 0
$$

and

$$
\operatorname{Var}(\operatorname{vec}(E) \mid \Phi) = \Omega \otimes I_N
$$

where $\Omega \in \mathbb{R}^{K \times K}$ is the within-sample covariance across measured outputs. This allows, for example, COD and total nitrogen errors to be correlated within the same sample.

### 9.2 Full-rank affine-core coefficient covariance

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

### 9.3 Affine-core mean-prediction uncertainty

For a new operating point with feature vector $\phi_* = \phi(u_*, c_{in,*})$, the fitted affine-core mean output is

$$
\widehat y_{aff,*} = \widehat M \phi_* + H c_{in,*}
$$

If $\Phi$ has full column rank, define the leverage factor

$$
s_* = \phi_*^T (\Phi^T \Phi)^{-1} \phi_*
$$

Then

$$
\operatorname{Var}(\widehat y_{aff,*} \mid \phi_*, c_{in,*}, \Phi) = s_* \Omega
$$

and the standard error of the fitted affine-core mean for output $k$ is

$$
SE_{mean,k}^{aff}(\phi_*) = \sqrt{s_* \, \Omega_{kk}}
$$

Under Gaussian errors, the corresponding affine-core confidence interval is

$$
\widehat y_{aff,*,k} \pm t_{1-\alpha/2,\, N-D} \sqrt{s_* \, [\widehat \Omega]_{kk}}
$$

Likewise, the standard prediction error for a future affine-core observation is

$$
SE_{pred,k}^{aff}(\phi_*) = \sqrt{(1 + s_*) \, \Omega_{kk}}
$$

with prediction interval

$$
\widehat y_{aff,*,k} \pm t_{1-\alpha/2,\, N-D} \sqrt{(1 + s_*) \, [\widehat \Omega]_{kk}}
$$

### 9.4 Why these formulas do not globally extend to the final non-negative predictor

The final deployed prediction is

$$
\widehat y_*^* = I_{comp} \, \operatorname{Proj}_{\mathcal{S}_+(c_{in,*})}(\widehat c_{raw,*})
$$

This map is generally piecewise affine rather than globally affine because the active set of non-negativity constraints can change from sample to sample. At points where the active set changes, the mapping is not described by one global coefficient matrix. Consequently, the closed-form affine-core variance formulas above are not exact finite-sample formulas for the final deployed predictor in general.

Two special cases are simpler.

1. If the non-negativity constraints are inactive at the prediction point, then $\widehat y_*^* = \widehat y_{aff,*}$ and the affine-core formulas apply exactly.
2. If the active set is locally stable, the final predictor is locally affine and the affine-core formulas can sometimes be adapted as a local approximation.

Neither of those special cases justifies a global exact interval formula for the final non-negative predictor.

### 9.5 Recommended uncertainty treatment for the final predictor

For the final deployed non-negative predictor, the most defensible default approach is resampling-based uncertainty quantification, such as bootstrap refitting or residual bootstrap, because it propagates uncertainty through both stages of the model: least-squares coefficient estimation and the sample-specific convex correction. In applications where only a fast approximation is needed, the affine-core intervals may still be reported as intervals for the linear core, provided they are labeled accordingly and not misrepresented as exact intervals for the final constrained predictor.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct effluent-state parameterization

The surrogate is parameterized on the effluent component state rather than on the net change. This keeps the learned target aligned with the quantity ultimately used for reporting and decision support. The cost is that the role of the influent state enters twice: once inside the feature map and once inside the constrained correction. That is not redundancy. The first role captures empirical dependence; the second role enforces the physically required part of the state.

### 10.2 Partitioned second-order feature structure

The partitioned feature map separates operating effects, influent-composition effects, and operation-loading interactions in a way that is interpretable to process engineers. The price of that interpretability is rapid feature growth, which can create multicollinearity, unstable coefficients, and weakly identified directions if the dataset does not adequately excite the design space. Since $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$, even moderate values of $M_{op}$ and $F$ can produce a feature basis that is large relative to the sample count. The resulting affine core can still interpolate training data through the pseudoinverse while leaving individual coefficients poorly determined.

### 10.3 Euclidean convex projection metric

The Euclidean correction gives a unique convex projection and keeps the constrained step mathematically clean. At the same time, it is sensitive to the scaling of component coordinates. If one component is numerically much larger or measured in a much different effective scale than another, the Euclidean metric may assign disproportionate influence to that direction. A weighted projection may be preferable in some applications, but that would be a different model from the one defined here.

### 10.4 Projection before measurement collapse

Enforcing the invariant relations and non-negativity before collapsing to measured space is a substantive modeling decision, not a notational convenience. It preserves the constraints in the space where the stoichiometric matrix is actually defined and where the non-negativity claim is physically meaningful. Once the state is collapsed into measured composites, some physically meaningful directions may no longer be separately visible.

### 10.5 Affine-core coefficients versus the final deployed predictor

The affine-core coefficients $M$ are the correct objects for direct engineering interpretation because they act directly on observed outputs through the identifiable least-squares stage. The latent component-space coefficients $B$ remain generally non-unique unless extra structure is imposed. The final deployed predictor $y^*$ adds one more layer: even when $M$ is well estimated, the final sample-specific output also depends on which non-negativity constraints are active. That means coefficient interpretation is clearest for the affine core, whereas the final prediction should be read as affine signal plus convex feasibility correction.

## 11. Limitations

Non-negative COBRE is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state and does not represent temporal dynamics or path dependence.
2. It enforces only the invariant relations encoded by the chosen stoichiometric basis and system boundary together with componentwise non-negativity.
3. Non-negative component concentrations do not guarantee full kinetic, biological, or thermodynamic feasibility.
4. Non-negative component predictions imply non-negative measured composites only when the adopted composition matrix has the appropriate sign structure.
5. The correction depends on the chosen metric and is therefore sensitive to component scaling.
6. The final deployed predictor is not globally representable by one affine measured-space coefficient matrix once non-negativity constraints become active.
7. Exact closed-form prediction intervals are available for the affine core under the usual linear-model assumptions, but not in general for the final non-negative deployed predictor.
8. The second-order feature basis can be statistically fragile if it is weakly excited or highly collinear.
9. A misspecified stoichiometric matrix or incorrect system boundary leads to a formally correct projection onto the wrong physical constraint set.
10. If the influent ASM component state is reconstructed from measured aggregate variables rather than observed directly, reconstruction error enters upstream of the regression and is not represented by the affine-core output-noise covariance formulas derived here.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its claims correctly.

## 12. Conclusion

Non-negative COBRE combines a partitioned second-order surrogate with a convex projection derived from stoichiometric invariants and componentwise non-negativity. The framework is useful for wastewater applications because it preserves the distinction between operating conditions and influent composition, enforces conservation structure where that structure naturally lives, removes negative deployed component predictions, and returns predictions in the measured variables used by plant operators and simulation studies. In deployment, the correction should be evaluated hierarchically so that the quadratic program is reserved for the subset of samples not already repaired by the closed-form affine projector.

The central theoretical point remains that measured composite data identify the affine measured-space operator $M$, not the latent component-space coefficient matrix $B$ uniquely. The non-negative extension does not alter that identifiability fact. Instead, it changes the deployment map: after estimating the affine core by least squares, the final prediction is obtained by projecting the raw component-space state onto the invariant-consistent non-negative set. Under that reading, non-negative COBRE is best understood as an analytically structured steady-state surrogate for activated-sludge prediction: more physically disciplined than a generic black-box regressor, more realistic than affine-only invariant correction when negative component states would otherwise occur, but still narrower in scope than a full dynamic mechanistic simulator.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
4. Seber, G. A. F., and Lee, A. J. Linear Regression Analysis. 2nd ed. Wiley, 2003.
5. Rao, C. R., and Mitra, S. Generalized Inverse of Matrices and Its Applications. Wiley, 1971.
6. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.
