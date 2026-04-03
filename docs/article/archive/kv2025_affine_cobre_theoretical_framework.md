# Constrained Orthogonal Bilinear Regression (COBRE) for Activated Sludge Surrogate Modeling

## Abstract

This article presents a revised theoretical framework for Constrained Orthogonal Bilinear Regression (COBRE), a physics-informed surrogate model for steady-state activated-sludge systems. The purpose of COBRE is to predict measured effluent variables from operating conditions and influent activated-sludge-model (ASM) component concentrations while preserving the stoichiometric invariants implied by the adopted reaction network and guaranteeing nonnegative deployed component concentrations. The key difficulty is that the conservation laws are defined in ASM component space, whereas plant observations are usually reported as composite variables such as total COD, total nitrogen, total phosphorus, or suspended solids. A regression model built only in measured-output space can fit those aggregates while still implying an impossible redistribution of the underlying ASM components. COBRE resolves that mismatch in two stages. First, a partitioned second-order surrogate produces an unconstrained prediction of the effluent ASM component state. Second, a state-weighted convex projection maps that raw prediction onto the intersection of the invariant-consistent affine set and the nonnegative orthant. The corrected component state is then collapsed into measured output space through a linear composition map.

The framework is written as a self-contained theory section for readers in chemical engineering, wastewater process modeling, and machine learning. All symbols are defined before use. The physical assumptions are stated explicitly. The invariant constraint is derived from the stoichiometric change relation rather than asserted heuristically. The positivity-preserving correction is formulated as a strictly convex quadratic program under a state-dependent metric inspired by Kircher and Votsmeier (2025). The distinction between global component-space fit parameters and locally interpretable measured-space sensitivities is made explicit. The result is a precise and reproducible formulation that clarifies which properties COBRE guarantees by construction, which objects remain only locally interpretable, and why end-to-end estimation is necessarily nonlinear in the positivity-preserving setting.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to be a steady-state operating condition: the operating variables, influent composition, and effluent response are treated as time-invariant over the control volume being modeled. The usual difficulty is that a generic data-driven regressor can fit observed effluent data while still violating fundamental conservation structure. In activated-sludge modeling, that failure is not a minor technical detail. It undermines the physical credibility of the surrogate because it can imply component inventories that are inconsistent with the adopted reaction network even when the measured aggregates appear plausible.

The source of the problem is a mismatch between two spaces.

1. The mechanistic stoichiometric model is written in an ASM component basis, such as soluble substrate, ammonium, nitrate, autotrophic biomass, particulate organics, phosphate, dissolved oxygen, and alkalinity.
2. The plant or simulator often reports outputs in measured composite variables, such as total COD, total nitrogen, total phosphorus, TSS, or VSS.

These two spaces are related, but they are not the same. Conservation laws are naturally expressed in the ASM component basis because the stoichiometric matrix acts on individual components. Observations, however, are usually available only after those components have been aggregated into measurable composites. A surrogate that learns only in measured-output space may reproduce the observed aggregates while obscuring physically impossible changes in the underlying component inventory.

COBRE is designed to address exactly that mismatch. Given operating conditions and an influent ASM component state, it predicts the steady-state effluent in a way that is flexible enough to capture nonlinear operating-loading interactions and structured enough to preserve the stoichiometric invariants implied by the chosen reaction network while preventing physically impossible negative deployed component concentrations. The model is constructed to answer one precise question:

> Given a steady-state influent state and a steady-state operating condition, what measured effluent state should be predicted if the underlying effluent ASM component state must remain consistent with the conserved quantities implied by the adopted stoichiometric model and must remain nonnegative componentwise after correction?

The theory in this article is restricted to steady-state reactor-block prediction. It does not aim to replace a dynamic activated-sludge simulator. Rather, it provides an analytically structured surrogate that preserves the most important stoichiometric structure, removes negative deployed component states, and remains explicit enough that its optimization problem, identifiability limits, and uncertainty approximations can be analyzed directly. The discussion proceeds from physical scope and notation, to derivation of the invariant constraint, to weighted positivity-preserving projection, to collapse into measured space, and finally to nonlinear estimation and uncertainty.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit operated at steady state. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes transport or removal mechanisms such as bypass streams, gas stripping, chemical dosing, or sludge wastage if they cross the chosen boundary and are not encoded in the stoichiometric description. The theory therefore applies only after the modeler has fixed the following items:

1. the reactor or process block being represented,
2. the ASM component basis used to describe material composition,
3. the stoichiometric matrix associated with that basis, and
4. the measurement map used to aggregate component concentrations into observed composite variables.

The framework is steady-state. It does not represent settling dynamics, sludge age dynamics, sensor dynamics, start-up transients, or time-varying trajectories. Changing the system boundary changes the admissible stoichiometric change space and therefore changes the invariant-consistent and nonnegative feasible sets themselves.

### 2.2 Why two state spaces are needed

To make the distinction concrete, suppose the underlying component basis contains soluble biodegradable substrate, particulate biodegradable substrate, ammonium, nitrate, phosphate, dissolved oxygen, alkalinity, and biomass fractions. A plant rarely measures all of those components directly. Instead, it may report total COD, total nitrogen, total phosphorus, TSS, and VSS. Those measured variables are linear combinations of the component concentrations under a chosen analytical convention.

The surrogate must therefore operate across two linked spaces:

1. ASM component space, where stoichiometry, conserved quantities, and componentwise nonnegativity are defined.
2. Measured composite space, where prediction targets are observed and evaluated.

COBRE learns and constrains the prediction in component space and only then maps the result to measured space. That order is essential. The conservation structure originates in the component basis, and the nonnegativity claim is also made in that basis, not in the aggregated measurement basis.

Before introducing symbols, it is useful to separate three distinct objects that will appear repeatedly. The first is the component state, which is the detailed ASM description used by the stoichiometric model. The second is the measured state, which is the aggregate laboratory or simulator output actually reported to the engineer. The third is the feasible correction set, which contains only those component states that satisfy the invariant relations and nonnegativity conditions chosen for deployment. COBRE first predicts in the component basis, then corrects that prediction in component space, and only then converts the result into measured variables.

### 2.3 Notation

Single-sample vectors are written as column vectors. Dataset matrices are defined later with samples stored by rows.

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $u$ | $\mathbb{R}^{M_{op}}$ | Operational input vector, for example hydraulic retention time, aeration intensity, recycle ratio, or other manipulated or design variables |
| $c_{in}$ | $\mathbb{R}^{F}$ | Influent ASM component concentration vector |
| $c_{out}$ | $\mathbb{R}^{F}$ | True steady-state effluent ASM component concentration vector |
| $c_{raw}$ | $\mathbb{R}^{F}$ | Unconstrained surrogate prediction of the effluent ASM component concentration vector |
| $c_{aff}$ | $\mathbb{R}^{F}$ | Euclidean affine projection of $c_{raw}$ onto the invariant-consistent set, retained as the original COBRE reference |
| $c_{w,aff}$ | $\mathbb{R}^{F}$ | State-weighted affine reference projection of $c_{raw}$ onto the invariant-consistent set |
| $c^*$ | $\mathbb{R}^{F}$ | Final weighted nonnegative projected effluent ASM component prediction |
| $y$ | $\mathbb{R}^{K}$ | Measured effluent composite vector |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM component concentrations to measured composite variables |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction progress vector expressed in concentration-equivalent units so that $\nu^T \xi$ has the same units as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank matrix whose rows form a basis of $\operatorname{null}(\nu)$ |
| $P_{inv}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the row space of $A$ in the original Euclidean affine COBRE reference |
| $P_{adm}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the admissible change space, $I_F - P_{inv}$, in the original Euclidean affine COBRE reference |
| $\phi(u, c_{in})$ | $\mathbb{R}^{D}$ | Engineered second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $B$ | $\mathbb{R}^{F \times D}$ | Raw component-space coefficient matrix |
| $W(c_{raw})$ | $\mathbb{R}^{F \times F}$ | Positive diagonal state-weight matrix used in the deployment projection objective |
| $S(c_{raw})$ | $\mathbb{R}^{F \times F}$ | Inverse metric matrix, $S(c_{raw}) = W(c_{raw})^{-2}$ |

The measured effluent variables are defined by the linear map

$$
y = I_{comp} c_{out}
$$

This linear composition map is standard in activated-sludge modeling when measured variables are aggregates of ASM components. For example, total COD or total nitrogen is formed by summing the relevant component concentrations with appropriate conversion factors. In the revised positivity-preserving formulation, nonnegative component predictions imply nonnegative measured composites only when the relevant rows of $I_{comp}$ are entrywise nonnegative.

With the spaces and notation fixed, the next step is to derive exactly which combinations of ASM components must be preserved by the adopted reaction network.

## 3. Modeling Assumptions

The framework rests on the following assumptions. These are not optional preferences left to the reader. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a steady-state input-output condition. The model is not a dynamic state estimator.
2. **Fixed component basis.** The ASM component basis and the associated stoichiometric matrix are fixed before regression begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement. Any external source or sink outside that boundary is outside the present model.
4. **Linear composition map.** Measured effluent variables are linear combinations of the underlying ASM component concentrations through $I_{comp}$.
5. **Direct effluent-state parameterization.** The surrogate is parameterized to predict the effluent ASM component state $c_{out}$ directly rather than the component change $c_{out} - c_{in}$. This keeps the learned target aligned with the final quantity of practical interest while letting the constrained correction act directly on the deployed state.
6. **Second-order surrogate class.** The raw surrogate is a partitioned second-order polynomial model that includes linear, quadratic, and operation-loading interaction terms.
7. **State-weighted correction metric.** The deployed correction is defined as the smallest adjustment under a positive diagonal state-dependent metric $W(c_{raw})$. This metric penalizes corrections to depleted components more strongly than corrections to abundant components and is not physically neutral to coordinate scaling.
8. **Constraint scope.** The final deployment projection enforces both the stoichiometric invariants implied by the chosen basis and system boundary and componentwise nonnegativity of the deployed ASM component state. It does not enforce upper bounds, kinetic feasibility, or thermodynamic admissibility beyond those conditions.
9. **Influent feasibility.** The influent ASM component state is assumed nonnegative in component space. Under that assumption the invariant-consistent nonnegative feasible set is non-empty because the influent state itself satisfies the invariant equalities.
10. **Composite-sign scope.** Componentwise nonnegative deployed states imply nonnegative measured composites only for those measured variables whose rows of $I_{comp}$ are entrywise nonnegative. If the measurement convention uses negative coefficients, extra output-space constraints would be required for a composite nonnegativity guarantee.
11. **Statistical scope.** Because the deployed predictor is state-dependent and inequality-constrained, estimation is formulated as a nonlinear optimization problem in the global component-space parameter matrix $B$. Exact finite-sample $t$-based intervals do not generally exist for the final predictor; local Jacobian approximations and bootstrap procedures are the appropriate default uncertainty tools.

These assumptions matter because each one narrows the scientific claim. A prediction that satisfies the invariant relations and componentwise nonnegativity is more physically disciplined than the earlier affine-only formulation, but it is still not automatically guaranteed to be fully process-realizable in every operating regime.

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

Thus, the physics is carried by the row space of $A$, not by one particular numerical basis. This matters because both the original affine reference projector and the revised invariant-consistent feasible set should depend only on the subspace being enforced, not on an arbitrary basis used to represent it.

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

The raw surrogate is flexible enough to capture curvature and interaction, but it is data-driven and unconstrained. There is no reason for $c_{raw}$ to satisfy the invariant relation $A c_{raw} = A c_{in}$ or the componentwise nonnegativity requirement without an additional correction step. The next section therefore introduces the weighted constrained correction that separates learned variation from invariant and positivity enforcement.

## 6. Weighted Convex Projection onto the Invariant-Consistent Nonnegative Set

### 6.1 Feasible sets

For a fixed influent state $c_{in}$, define the invariant-consistent affine set

$$
\mathcal{S}(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in} \}
$$

and the deployed invariant-consistent nonnegative feasible set

$$
\mathcal{S}_+(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in}, \; c \ge 0 \}
$$

where $c \ge 0$ is understood componentwise. The first set restores only the stoichiometric invariants. The second set adds the physical requirement that the deployed ASM component concentrations remain nonnegative.

### 6.2 State-weighted metric and weighted affine reference projection

The positivity-preserving extension adopts a state-dependent metric that penalizes corrections to depleted components more strongly than corrections to abundant components. For concreteness, define the diagonal weight matrix

$$
W(c_{raw}) = \operatorname{Diag}\left(\frac{1}{\max(c_{raw,1}, \epsilon)}, \dots, \frac{1}{\max(c_{raw,F}, \epsilon)}\right)
$$

with fixed floor $\epsilon > 0$, and define the inverse metric matrix

$$
S(c_{raw}) = W(c_{raw})^{-2} = \operatorname{Diag}\left(\max(c_{raw,1}, \epsilon)^2, \dots, \max(c_{raw,F}, \epsilon)^2\right)
$$

The hard floor keeps the metric strictly positive definite for every sample. A smooth positive approximation could be used instead, but the theory below requires only that $W(c_{raw})$ remain positive diagonal.

Before imposing nonnegativity, the weighted affine reference state is defined by

$$
c_{w,aff} = \arg\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2}(c - c_{raw})^T W(c_{raw})^2 (c - c_{raw})
$$

subject to

$$
A c = A c_{in}
$$

The associated Lagrange-multiplier calculation is the weighted analogue of the earlier Euclidean derivation and gives

$$
c_{w,aff} = c_{raw} - S(c_{raw}) A^T\left(A S(c_{raw}) A^T\right)^{-1} A(c_{raw} - c_{in})
$$

This state-weighted affine correction is an important reference object because it shows how the metric reallocates invariant-restoring adjustments across components. It does not, by itself, guarantee nonnegativity.

### 6.3 Final positivity-preserving correction problem

The deployed positivity-preserving COBRE predictor is defined by the weighted convex projection

$$
c^* = \arg\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2}(c - c_{raw})^T W(c_{raw})^2 (c - c_{raw})
$$

subject to

$$
A c = A c_{in},
\qquad
c \ge 0
$$

The metric determines how the correction is distributed across component directions. The explicit inequality constraint $c \ge 0$ is what guarantees nonnegative deployed component concentrations. Weighting alone does not provide that guarantee.

### 6.4 Feasibility, existence, and uniqueness

Under the present assumptions, the feasible set is non-empty. If $c_{in} \ge 0$, then $c = c_{in}$ satisfies both

$$
A c_{in} = A c_{in}
\qquad \text{and} \qquad
c_{in} \ge 0
$$

and is therefore feasible. Because $\mathcal{S}_+(c_{in})$ is closed and convex and because the objective has Hessian $W(c_{raw})^2 \succ 0$, the deployment problem has a unique minimizer for every pair $(c_{raw}, c_{in})$.

### 6.5 KKT characterization

Introduce multipliers $\lambda \in \mathbb{R}^{q}$ for the equality constraints and $\mu \in \mathbb{R}^{F}$ for the nonnegativity constraints. The Lagrangian is

$$
\mathcal{L}(c, \lambda, \mu) = \frac{1}{2}(c - c_{raw})^T W(c_{raw})^2 (c - c_{raw}) + \lambda^T(A c - A c_{in}) - \mu^T c
$$

with dual feasibility condition $\mu \ge 0$. The Karush-Kuhn-Tucker conditions for the unique deployed minimizer are

$$
W(c_{raw})^2 (c - c_{raw}) + A^T \lambda - \mu = 0
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

These conditions show that the final correction combines invariant-restoring directions with the conic effect of the active nonnegativity constraints. When the active set changes, the local sensitivity of the deployed map changes as well.

### 6.6 Relation to weighted affine and original affine COBRE

The deployed positivity-preserving predictor contains both the weighted affine reference and the original Euclidean affine COBRE model as special cases.

1. If $c_{w,aff} \ge 0$ componentwise, then $c_{w,aff}$ is feasible for the deployed problem and therefore

$$
c^* = c_{w,aff}
$$

2. If $W(c_{raw}) = I_F$ and the nonnegativity constraints are inactive, then the deployed predictor reduces to the original affine COBRE projector

$$
c_{aff} = P_{adm} c_{raw} + P_{inv} c_{in}
$$

3. If $W(c_{raw}) = I_F$ but one or more nonnegativity constraints are active, then the deployed predictor reduces to the Euclidean nonnegative COBRE correction.

These recovery statements explain why the COBRE name is retained historically even though the deployed correction is no longer an orthogonal projector in the Euclidean sense.

### 6.7 What the projection guarantees and what it does not

The deployed projection guarantees that

$$
A c^* = A c_{in}
$$

and

$$
c^* \ge 0
$$

for every sample. It does not guarantee the following.

1. It does not guarantee realistic upper bounds or process-feasible operating ranges.
2. It does not impose kinetic feasibility beyond the chosen invariant relations and nonnegativity conditions.
3. It does not guarantee thermodynamic admissibility.
4. It does not correct errors introduced by a misspecified stoichiometric basis or an incorrect system boundary.

These are not derivation errors. They are the exact consequences of the constraint set being enforced.

## 7. Collapse from ASM Component Space to Measured Output Space

### 7.1 Final measured-output equation

Practical prediction targets are usually measured composite variables rather than ASM component concentrations. The deployed measured output is therefore

$$
y^*(u, c_{in}; B) = I_{comp} c^*(u, c_{in}; B)
$$

where the final deployed state is obtained from the constrained projection described in Section 6 after forming the raw state

$$
c_{raw}(u, c_{in}; B) = B \phi(u, c_{in})
$$

Because both the weights and the active inequality set depend on the raw state, the deployed map from $(u, c_{in})$ to $y^*$ is generally nonlinear even though the raw surrogate is linear in the engineered features.

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

If every row of $I_{comp}$ is entrywise nonnegative, then $c^* \ge 0$ implies $y^* \ge 0$ componentwise. If the measurement convention contains negative coefficients, then componentwise nonnegativity of $c^*$ alone does not imply nonnegative measured composites.

### 7.2 Weighted affine reference output and special cases

The weighted affine reference output is defined by

$$
y_{w,aff}(u, c_{in}; B) = I_{comp} c_{w,aff}(u, c_{in}; B)
$$

This reference output is useful for two reasons. First, it isolates the effect of the weighted metric without the additional geometry of the nonnegative orthant. Second, it identifies when the nonnegativity constraints are inactive, because

$$
y^*(u, c_{in}; B) = y_{w,aff}(u, c_{in}; B)
\qquad \text{whenever} \qquad
c_{w,aff}(u, c_{in}; B) \ge 0
$$

If the weights are further specialized to $W(c_{raw}) = I_F$, then the weighted affine reference collapses to the original Euclidean affine COBRE predictor.

### 7.3 Local sensitivity and interpretability

The final deployed predictor is not globally representable by one measured-space coefficient matrix. Interpretation therefore shifts from one global affine operator to local sensitivities of the deployed map.

Define the constrained solution map

$$
\Pi(c_{raw}, c_{in}) = c^*
$$

When the active set is locally stable and the chosen weighting rule is differentiable at the operating point of interest, the local Jacobians

$$
J_{raw}(c_{raw}, c_{in}) = \frac{\partial \Pi(c_{raw}, c_{in})}{\partial c_{raw}}
\qquad \text{and} \qquad
J_{in}(c_{raw}, c_{in}) = \frac{\partial \Pi(c_{raw}, c_{in})}{\partial c_{in}}
$$

exist locally. These Jacobians induce local measured-space sensitivities. For example,

$$
\frac{\partial y^*}{\partial B} = I_{comp} J_{raw}(c_{raw}, c_{in}) \frac{\partial c_{raw}}{\partial B}
$$

and the local sensitivity with respect to the original predictors is

$$
\frac{\partial y^*}{\partial u} = I_{comp} J_{raw}(c_{raw}, c_{in}) \frac{\partial c_{raw}}{\partial u}
$$

$$
\frac{\partial y^*}{\partial c_{in}} = I_{comp} \left[J_{raw}(c_{raw}, c_{in}) \frac{\partial c_{raw}}{\partial c_{in}} + J_{in}(c_{raw}, c_{in})\right]
$$

These identities explain the revised interpretability claim. The partitioned second-order raw surrogate still determines how operating variables and influent components enter the model, but the final measured response is filtered through the local Jacobian of the constrained deployment map. The correct local sensitivity is therefore not a state-dependent projector multiplied by $B$ alone; it is the full chain rule through the weighted constrained solution.

### 7.4 Why projection must occur before collapse

The order of operations still matters. Conservation relations and nonnegativity are defined in ASM component space because the stoichiometric matrix acts on ASM components and because nonnegativity is physically meaningful at the component level. If the prediction were first collapsed to measured space and only then corrected, both the invariant structure and the componentwise positivity claim could be lost or weakened. The next section therefore formulates estimation directly through the deployed component-space correction followed by measured-space collapse.

## 8. Nonlinear Estimation and Identifiability from Measured Composite Data

### 8.1 Dataset-level nonlinear model

Let $N$ steady-state samples be available and define

$$
\phi_i = \phi(u_i, c_{in,i}), \qquad i = 1, \dots, N
$$

For a given coefficient matrix $B$, the raw component prediction for sample $i$ is

$$
c_{raw,i}(B) = B \phi_i
$$

The deployed projected component state is then defined implicitly by

$$
c_i^*(B) = \arg\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2}(c - c_{raw,i}(B))^T W(c_{raw,i}(B))^2 (c - c_{raw,i}(B))
$$

subject to

$$
A c = A c_{in,i},
\qquad
c \ge 0
$$

and the deployed measured prediction is

$$
f_i(B) = I_{comp} c_i^*(B)
$$

The samplewise observed-output model is therefore

$$
y_i = f_i(B) + e_i
$$

where $e_i \in \mathbb{R}^{K}$ collects model and measurement errors. As in the earlier formulations, the influent state $c_{in,i}$ is assumed available in ASM component coordinates for every sample.

### 8.2 End-to-end estimator

Because the deployed prediction depends on $B$ through the raw state, the weights, and the active inequality set, the positivity-preserving COBRE estimator is nonlinear. The natural primary estimator is penalized nonlinear least squares,

$$
\widehat B = \arg\min_{B \in \mathbb{R}^{F \times D}} \; \sum_{i=1}^{N} \lVert y_i - f_i(B) \rVert_2^2 + \lambda_{reg} \lVert B \rVert_F^2
$$

with tuning parameter $\lambda_{reg} \ge 0$. If an estimate of within-sample output covariance is available, the Euclidean sample loss may be replaced by a covariance-weighted multivariate loss, but the estimator remains nonlinear either way.

This estimation problem nests one strictly convex projection problem per sample inside the outer fit over $B$. In practice, the inner projection is computationally manageable because the equality structure is fixed and the component dimension is moderate, while the outer problem can be handled by gradient-based optimization with differentiation through the constrained solution map.

### 8.3 What the data identify

Measured composite data no longer identify one global affine measured-space operator of the form $M = G B$ because the deployed predictor is not globally affine in the engineered features. The primary fitted object is now the global component-space parameter matrix $B$ under the nonlinear deployed map $f_i(B)$. Even so, exact identifiability of $B$ is not automatic.

If two parameter matrices $B_1$ and $B_2$ satisfy

$$
f_i(B_1) = f_i(B_2), \qquad i = 1, \dots, N
$$

then they are observationally indistinguishable on the available dataset. This can occur because measured outputs are composites of component states and because the positivity-preserving projection can collapse distinct raw states onto the same deployed constrained state. The practical implications are the following.

1. the primary empirical targets are deployed predictions and their local sensitivities, not the raw entries of $B$ interpreted in isolation,
2. weakly excited feature directions and heavily active constraint regions can make parts of $B$ only weakly identified,
3. regularization selects a stable representative within the set of parameter matrices that fit the deployed map similarly well, but it does not create information that is absent from the data.

### 8.4 Interpretation of fitted parameters and local sensitivities

The block structure of $B$ introduced in Section 5.3 still matters because it determines how operations, influent components, quadratic terms, and interaction terms enter the raw surrogate. What changes is the level at which interpretation is defensible. In the positivity-preserving mainline formulation, direct engineering interpretation is strongest for

1. deployed predictions $f_i(\widehat B)$,
2. active-set patterns of the constrained correction,
3. local sensitivities obtained from the Jacobians in Section 7.3.

By contrast, one should not read the entries of $\widehat B$ as globally valid measured-space effect sizes. They are global fit parameters for a nonlinear deployed map whose measured-space response is state-dependent.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Error model for the nonlinear deployed predictor

For statistical inference, write the deployed predictor as

$$
f(u, c_{in}; B) = I_{comp} \Pi(B \phi(u, c_{in}), c_{in})
$$

and suppose the samplewise observations satisfy

$$
y_i = f(u_i, c_{in,i}; B_0) + e_i
$$

with independent row errors obeying

$$
\mathbb{E}[e_i \mid u_i, c_{in,i}] = 0,
\qquad
\operatorname{Var}(e_i \mid u_i, c_{in,i}) = \Omega
$$

for some within-sample covariance matrix $\Omega \in \mathbb{R}^{K \times K}$.

### 9.2 Local asymptotic parameter covariance

Let $\theta = \operatorname{vec}(B) \in \mathbb{R}^{FD}$ and define the stacked mean map

$$
F(\theta) = \begin{bmatrix}
f(u_1, c_{in,1}; \theta) \\
\vdots \\
f(u_N, c_{in,N}; \theta)
\end{bmatrix} \in \mathbb{R}^{N K}
$$

Let $\mathcal{J}(\theta) = \partial F(\theta) / \partial \theta^T$ be the Jacobian of the stacked deployed predictor. Under standard nonlinear regression regularity conditions, local identifiability, and a locally stable active set, the nonlinear least-squares estimator is asymptotically normal with covariance approximation

$$
\operatorname{Var}(\widehat \theta) \approx \left(\mathcal{J}(\widehat \theta)^T (\Omega^{-1} \otimes I_N) \mathcal{J}(\widehat \theta)\right)^{-1}
$$

when the fit is interpreted as generalized nonlinear least squares and the model is correctly specified. If regularization is active or model misspecification is a concern, a sandwich covariance or bootstrap estimate is preferable.

### 9.3 Local prediction uncertainty

For a new operating point $(u_*, c_{in,*})$, define the deployed prediction

$$
\widehat y_*^* = f(u_*, c_{in,*}; \widehat \theta)
$$

and let

$$
J_*(\widehat \theta) = \frac{\partial f(u_*, c_{in,*}; \theta)}{\partial \theta^T}\Bigg|_{\theta = \widehat \theta}
$$

denote the local Jacobian of the deployed predictor with respect to the fitted parameter vector. A first-order delta-method approximation gives

$$
\operatorname{Var}(\widehat y_*^*) \approx J_*(\widehat \theta) \, \operatorname{Var}(\widehat \theta) \, J_*(\widehat \theta)^T
$$

and the corresponding future-observation covariance approximation is

$$
\operatorname{Var}(y_{future,*} - \widehat y_*^*) \approx \Omega + J_*(\widehat \theta) \, \operatorname{Var}(\widehat \theta) \, J_*(\widehat \theta)^T
$$

These formulas are local approximations. They are most trustworthy when the active set of the constrained projection is stable near both the fitted solution and the prediction point.

### 9.4 Recommended uncertainty treatment

Because the deployed predictor is piecewise smooth rather than globally affine, exact finite-sample $t$-based formulas for the final predictor are not generally available. Two practical consequences follow.

1. When the active set is locally stable, Jacobian-based approximations are useful for fast local uncertainty summaries.
2. When active-set changes are common or when a more global uncertainty description is needed, bootstrap refitting or residual bootstrap is the more defensible default because it propagates uncertainty through both the nonlinear fit and the constrained deployment map.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct effluent-state parameterization

The surrogate is parameterized on the effluent component state rather than on the net change. This keeps the learned target aligned with the quantity ultimately used for reporting and decision support. In the positivity-preserving formulation, the influent state affects the deployed prediction through two distinct channels: it enters the raw surrogate through the feature map, and it also defines the right-hand side of the invariant constraint $A c = A c_{in}$. That is not redundancy. The first channel is empirical and learned through the fitted parameter matrix $B$. The second channel is normative and enforced by the constrained deployment map.

### 10.2 Partitioned second-order feature structure

The partitioned feature map still separates operating effects, influent-composition effects, and operation-loading interactions in a way that is meaningful to process engineers. The price of that structure is now twofold. First, the feature dimension can still be large relative to the available sample count, which weakens statistical identification. Second, the deployed measured-space effect of those features is no longer globally affine because the constrained correction acts after the raw surrogate has been formed. The feature blocks therefore retain mechanistic meaning at the raw-surrogate level, but their measured-space influence must be read through local sensitivities rather than one global coefficient matrix.

### 10.3 State-weighted positivity-preserving metric

The state-weighted metric is the mechanism that biases invariant-restoring corrections away from depleted components. It is therefore substantively part of the model, not a harmless numerical detail. The choice of weighting rule and floor parameter affects the geometry of the deployment correction, the stability of local sensitivities, and the extent to which large positive components absorb the constraint-enforcement burden. For the same reason, the deployed correction is no longer orthogonal in the Euclidean sense. The COBRE name is retained historically because the raw surrogate class and the invariant-correction philosophy remain continuous with the earlier model.

### 10.4 Projection before measurement collapse

Enforcing the invariant and nonnegativity relations before collapsing to measured space is a substantive modeling decision, not a notational convenience. It preserves the constraints in the space where the stoichiometric matrix is actually defined and where the sign restriction is physically meaningful. Once the state is collapsed into measured composites, some physically meaningful component directions may no longer be separately visible.

### 10.5 Global fit parameters versus local measured-space sensitivities

In the positivity-preserving mainline formulation, the global fit parameters are the entries of $B$, but the directly interpretable measured-space objects are local Jacobians and deployed predictions. This is a sharper distinction than in affine COBRE. The entries of $B$ determine the raw component-space surface, yet the measured-space response also depends on the state-weighted constrained projection. Direct engineering interpretation is therefore strongest for deployed outputs, active-set behavior, and local sensitivities evaluated at operating points of interest rather than for raw entries of $B$ viewed as globally valid measured-space effects.

## 11. Limitations

COBRE is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state and does not represent temporal dynamics or path dependence.
2. It enforces only the invariant relations encoded by the chosen stoichiometric basis and system boundary together with componentwise nonnegativity.
3. Componentwise nonnegative deployed states do not guarantee nonnegative measured composites unless the relevant rows of the adopted composition matrix are entrywise nonnegative.
4. The deployed correction depends on the chosen weighting rule and floor parameter and is therefore sensitive to component scaling and metric design.
5. End-to-end estimation is computationally more demanding than the affine-only formulation because it requires differentiation through a samplewise constrained projection problem.
6. The final deployed predictor is not globally affine; local sensitivities can change when the active inequality set changes.
7. Exact finite-sample closed-form prediction intervals are not generally available for the final deployed predictor.
8. The second-order feature basis can still be statistically fragile if it is weakly excited or highly collinear.
9. A misspecified stoichiometric matrix or incorrect system boundary leads to a formally correct projection onto the wrong physical constraint set.
10. If the influent ASM component state is reconstructed from measured aggregate variables rather than observed directly, reconstruction error enters upstream of the nonlinear fit and is not represented by the output-noise covariance formulas derived here.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its claims correctly.

## 12. Conclusion

COBRE now combines a partitioned second-order surrogate with a state-weighted convex projection derived from stoichiometric invariants and componentwise nonnegativity. The framework is useful for wastewater applications because it preserves the distinction between operating conditions and influent composition, enforces conservation structure where that structure naturally lives, removes negative deployed component states, and returns predictions in the measured variables used by plant operators and simulation studies.

The central theoretical point of the revised formulation is that positivity preservation changes the mathematical character of the model. The deployed predictor is no longer one global affine measured-space map. Instead, the global component-space coefficient matrix $B$ is fitted through a nonlinear end-to-end optimization, while direct engineering interpretation is carried by deployed predictions and local measured-space sensitivities of the constrained map. Under that reading, positivity-preserving COBRE is best understood as an analytically structured steady-state surrogate for activated-sludge prediction: more physically disciplined than the affine-only formulation when negative component states would otherwise occur, but also more nonlinear in its estimation and uncertainty structure.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
4. Seber, G. A. F., and Lee, A. J. Linear Regression Analysis. 2nd ed. Wiley, 2003.
5. Rao, C. R., and Mitra, S. Generalized Inverse of Matrices and Its Applications. Wiley, 1971.
6. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.
7. Kircher, K., and Votsmeier, M. 2025. [Full bibliographic details to be completed by the author].