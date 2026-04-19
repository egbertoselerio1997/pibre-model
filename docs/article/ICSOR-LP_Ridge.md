# Non-Negative Invariant-Constrained Second-Order Regression (ICSOR) for Activated Sludge Surrogate Modeling

## Abstract

This article presents a non-negative formulation of invariant-constrained second-order regression (ICSOR), a physics-informed surrogate model for steady-state activated-sludge systems. The model accepts operational variables and influent activated-sludge-model (ASM) component fractions, and it predicts effluent ASM component fractions in the same ASM basis. ICSOR is therefore trained and deployed natively in ASM component space. If measured composite variables such as total COD, total nitrogen, total phosphorus, TSS, or VSS are needed, they are computed afterward by an external composition matrix. The collapse into measured-output space is not part of the model itself.

Stoichiometric conservation and non-negativity are handled by a staged component-space deployment rule. First, a partitioned second-order surrogate produces an unconstrained prediction of the effluent ASM component-fraction state. Second, that raw state is repaired to the invariant-consistent affine reference. Third, if the affine reference still violates componentwise non-negativity, a single linear program selects a non-negative invariant-consistent state by minimizing a weighted component-space $L_1$ deviation from the affine reference. The regression coefficients of the raw surrogate are estimated directly from component-space training pairs by ridge regression, so the primary inferential object is the ASM component-space coefficient matrix itself, stabilized by shrinkage rather than inferred through a measured-space affine core.

The framework is written as a self-contained theory section for readers in chemical engineering, wastewater process modeling, and machine learning. All symbols are defined before use. The invariant constraint is derived from the stoichiometric change relation rather than asserted heuristically. The non-negative correction is formulated as a single component-space linear program with one objective and one shared feasible set, and the role of the orthogonal affine projector is retained explicitly as the equality-only reference solution and zero-correction special case. In deployment, the LP stage is needed only when the raw prediction is infeasible and the closed-form affine projector still violates componentwise non-negativity. Measured composites are treated only as optional downstream reporting quantities. The result is a precise formulation of what component-space ICSOR with ridge regression guarantees, what is estimated directly from ASM component-fraction data, and what must instead be handled through LP-based post-estimation correction.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to represent a quasi-steady operating condition: the operating variables, influent composition, and effluent response are treated as effectively time-invariant over the control volume being modeled for the sampling window of interest. The usual difficulty is that a generic data-driven regressor can fit observed effluent data while still violating fundamental conservation structure. In activated-sludge modeling, that failure is not a minor technical detail. It undermines the physical credibility of the surrogate because it can imply component inventories that are inconsistent with the adopted reaction network even when the measured aggregates appear plausible.

The source of the problem is a mismatch between the space in which wastewater stoichiometry is defined and the space in which plant variables are often reported.

1. The mechanistic stoichiometric model is written in an ASM component basis, such as soluble substrate, ammonium, nitrate, autotrophic biomass, particulate organics, phosphate, dissolved oxygen, and alkalinity.
2. Plant or simulator dashboards often report composite variables such as total COD, total nitrogen, total phosphorus, TSS, or VSS.

Those two spaces are related, but they should not be confused. Conservation laws are naturally expressed in the ASM component basis because the stoichiometric matrix acts on individual components. Composite outputs are downstream aggregates of those components. In the formulation developed here, that downstream aggregation is treated explicitly as an external reporting step rather than as the model target.

The earlier affine-only ICSOR formulation already predicted in component space and repaired invariant violations by orthogonal projection. The present formulation strengthens that design rule in two ways. First, the regression stage itself is written directly in component space with effluent ASM component fractions as the training targets. Second, the deployed correction step now enforces both invariant consistency and componentwise non-negativity before any optional reporting collapse is applied. The resulting model therefore learns and corrects only ASM component-fraction states.

The model is constructed to answer one precise question:

> Given a steady-state influent ASM component-fraction vector and a steady-state operating condition, what effluent ASM component-fraction vector should be predicted if the surrogate is second order, must respect the conserved quantities implied by the adopted stoichiometric model, and must remain non-negative componentwise after deployment correction?

The theory in this article is restricted to steady-state reactor-block prediction. It does not aim to replace a dynamic activated-sludge simulator. Rather, it provides an analytically structured surrogate that preserves stoichiometric structure, enforces non-negativity at the deployed component-state level, and remains simple enough that its raw component-space coefficient matrix can still be estimated directly from component-space data by ridge regression. Ridge stabilization is useful here because the second-order design can be high dimensional and weakly conditioned. In deployment, the correction is evaluated in stages: no correction when the raw component state is already feasible, closed-form affine projection when only the invariant equalities are violated, and a single component-space LP correction only when non-negativity remains violated after the affine step. If measured composites are needed later, they are obtained by an external composition matrix. The discussion proceeds from physical scope and notation, to derivation of the invariant relations, to LP-based non-negative selection, to external reporting collapse, and finally to component-space estimation and uncertainty with explicit attention to ridge-induced bias and conditioning.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit represented by quasi-steady samples. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes transport or removal mechanisms such as bypass streams, gas stripping, chemical dosing, or sludge wastage if they cross the chosen boundary and are not encoded in the stoichiometric description. Throughout the article, the ASM states are treated as non-negative component fractions on one common basis. If a particular implementation uses concentration-equivalent normalized fractions rather than literal fractions, that preprocessing is part of the model definition rather than an implicit algebraic detail. The theory therefore applies only after the modeler has fixed the following items:

1. the reactor or process block being represented,
2. the ASM component basis used to describe material composition,
3. the stoichiometric matrix associated with that basis, and
4. the composition matrix used later for optional external reporting in measured-output space.

The framework is steady-state in the sense of quasi-steady samples rather than full dynamic trajectories. It does not represent settling dynamics, sludge age dynamics, sensor dynamics, start-up transients, or time-varying trajectories. In plant applications, the samples would typically correspond to stable operating windows or time-averaged periods rather than literal mathematical equilibria. Changing the system boundary changes the admissible stoichiometric change space and therefore changes the invariant and non-negative feasible sets.

### 2.2 Why two state spaces are still relevant

To make the distinction concrete, suppose the underlying component basis contains soluble biodegradable substrate, particulate biodegradable substrate, ammonium, nitrate, phosphate, dissolved oxygen, alkalinity, and biomass fractions. A plant rarely measures all of those components directly. Instead, it may report total COD, total nitrogen, total phosphorus, TSS, and VSS. Those measured variables are linear combinations of the component concentrations under a chosen analytical convention.

The surrogate must therefore operate across two linked spaces:

1. ASM component space, where stoichiometry, invariants, componentwise non-negativity, and the learned input-output map are defined.
2. Measured composite space, which is retained only as an external reporting space obtained by applying a fixed composition matrix after prediction.

ICSOR learns and constrains the prediction in component space and only afterward, if desired, maps the result to measured space. That order remains essential in the non-negative formulation. Stoichiometric invariants originate in the component basis, and the componentwise non-negativity claim is also made in that basis. A measured-space-only training target would generally be too weak to control the underlying ASM state.

### 2.3 Notation

Single-sample vectors are written as column vectors. Dataset matrices are defined later with samples stored by rows.

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $u$ | $\mathbb{R}^{M_{op}}$ | Operational input vector, for example hydraulic retention time, aeration intensity, recycle ratio, or other manipulated or design variables |
| $c_{in}$ | $\mathbb{R}_{+}^{F}$ | Influent ASM component-fraction vector |
| $c_{out}$ | $\mathbb{R}_{+}^{F}$ | True steady-state effluent ASM component-fraction vector |
| $c_{raw}$ | $\mathbb{R}^{F}$ | Unconstrained surrogate prediction of the effluent ASM component-fraction vector |
| $c_{aff}$ | $\mathbb{R}^{F}$ | Affine reference state obtained by orthogonal projection of $c_{raw}$ onto the invariant-consistent set |
| $c^*$ | $\mathbb{R}_{+}^{F}$ | Any optimal ASM component-fraction state of the component-space non-negative LP |
| $y_{ext}$ | $\mathbb{R}^{K}$ | External measured composite vector computed from a component vector |
| $r$ | $\mathbb{R}^{F}$ | Non-negative auxiliary vector bounding absolute component-state deviation from $c_{aff}$ in the component-space LP |
| $w_c$ | $\mathbb{R}_{++}^{F}$ | Positive weights on component-state absolute deviations in the component-space LP |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM component fractions to measured composite variables for external reporting |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction progress vector expressed on the same fraction-equivalent basis as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank matrix whose transposed rows form a basis of $\operatorname{null}(\nu)$, equivalently $A \nu^T = 0$ |
| $P_{inv}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the row space of $A$ |
| $P_{adm}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the admissible change space, $I_F - P_{inv}$ |
| $N_A$ | $\mathbb{R}^{F \times (F-q)}$ | Matrix whose columns form an orthonormal basis of $\operatorname{null}(A)$ |
| $\phi(u, c_{in})$ | $\mathbb{R}^{D}$ | Engineered second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $\lambda$ | scalar | Ridge penalty parameter used in component-space estimation |
| $\Gamma$ | $\mathbb{R}^{D \times D}$ | Symmetric positive-semidefinite ridge penalty matrix, commonly diagonal with the intercept left unpenalized |
| $B$ | $\mathbb{R}^{F \times D}$ | Raw component-space coefficient matrix estimated from component-space training targets |

Throughout the chapter, $w_c$, $\lambda$, and $\Gamma$ are treated as fixed model-definition quantities unless data-driven penalty selection is being discussed explicitly.

The external measured variables are defined by the linear map

$$
y_{ext} = I_{comp} c
$$

When this map is applied to the true effluent state, it yields the true measured composites. When it is applied to the deployed component prediction $c^*$, it yields the externally reported prediction $y_{ext} = I_{comp} c^*$. This reporting step is outside the model: ICSOR itself predicts only ASM component fractions.

## 3. Modeling Assumptions

The framework rests on the following assumptions. These are not optional preferences left to the reader. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a quasi-steady input-output condition, typically a stable operating epoch or time-averaged window. The model is not a dynamic state estimator.
2. **Fixed component basis.** The ASM component basis and the associated stoichiometric matrix are fixed before regression begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement. Any external source or sink outside that boundary is outside the present model.
4. **Common fraction basis.** The vectors $c_{in}$ and $c_{out}$ are expressed on one consistent non-negative ASM component-fraction basis.
5. **Direct component-space supervision.** The training targets are effluent ASM component fractions, not measured composite outputs.
6. **External composition map.** The collapse from ASM component fractions to measured composites is an external calculation performed after prediction through a known fixed matrix $I_{comp}$.
7. **Direct effluent-state parameterization.** The surrogate is parameterized to predict the effluent ASM component state $c_{out}$ directly rather than the component change $c_{out} - c_{in}$.
8. **Second-order surrogate class.** The raw surrogate is a partitioned second-order polynomial model that includes linear, quadratic, and operation-loading interaction terms.
9. **Component-space LP objective.** After the affine reference state is formed, the deployed correction minimizes a weighted component-space $L_1$ deviation from $c_{aff}$ over the invariant-consistent non-negative set.
10. **Weighting convention and optimizer scope.** The positive weight vector $w_c$ is fixed as part of the model definition. The baseline formulation allows any optimizer of the resulting LP to be used for deployment.
11. **Constraint scope.** The final LP correction enforces the stoichiometric invariants implied by the chosen basis and system boundary together with componentwise non-negativity. It does not enforce upper bounds, kinetic feasibility, or thermodynamic admissibility beyond those conditions.
12. **Influent feasibility.** The influent reference state is assumed non-negative in component space. Under that assumption the non-negative feasible set is non-empty because the influent state itself satisfies the invariant equalities.
13. **Ridge regularization.** The raw component-space coefficient matrix is estimated by multivariate ridge regression with fixed penalty parameter $\lambda$ and penalty matrix $\Gamma$.
14. **Composite-sign scope.** Non-negative component predictions imply non-negative externally reported composites only when the relevant rows of $I_{comp}$ are entrywise non-negative. If the reporting convention uses negative coefficients, extra output-space sign constraints would be required for a composite non-negativity guarantee.
15. **Statistical scope.** Ridge coefficient and prediction uncertainty formulas are interpreted conditionally on fixed $\lambda$ and $\Gamma$; exact finite-sample Student-$t$ formulas do not generally apply even before the LP-corrected deployed predictor is considered.

These assumptions matter because each one narrows the scientific claim. A prediction that satisfies the invariant relations and componentwise non-negativity is physically better disciplined than the affine-only formulation, but it is still not automatically guaranteed to be fully process-realizable in every operating regime.

## 4. Stoichiometric Structure and Conserved Quantities

### 4.1 From stoichiometric reactions to component-state change

Let $\nu \in \mathbb{R}^{R \times F}$ be the stoichiometric matrix written in the adopted ASM component basis. Its entries are treated here as fixed stoichiometric coefficients in that basis; any scaling needed to express component-state change in concentration units is absorbed into the definition of the reaction-progress vector. For one steady-state sample, define the net reaction progress vector $\xi \in \mathbb{R}^{R}$ so that

$$
c_{out} - c_{in} = \nu^T \xi
$$

This equation is the starting point of the theory. It says that the net change in the effluent component state is a linear combination of reaction stoichiometries. The entries of $\xi$ need not be observed individually. They collect the net progression of each modeled reaction over the chosen control volume after scaling into concentration-equivalent units. For example, if reaction $i$ has steady-state rate $r_i$ in units of concentration per time and the relevant hydraulic time scale of the control volume is $\tau$, then one admissible definition is $\xi_i = r_i \tau$, which has concentration units. More generally, $\xi_i$ may be interpreted as the net integrated reaction extent over the control volume after whatever normalization is required so that $\nu^T \xi$ is expressed in the same units as $c_{out} - c_{in}$.

That definition makes the statement dimensionally coherent: both $c_{out} - c_{in}$ and $\nu^T \xi$ live in the same ASM component concentration space. Without that scaling convention, the conservation equation would be ambiguous and different readers could implement different, incompatible normalizations.

### 4.2 Invariant relations implied by the stoichiometric matrix

The reaction progress vector $\xi$ is not part of the surrogate model and is usually not observed. To eliminate it, we introduce a full-row-rank matrix $A \in \mathbb{R}^{q \times F}$ whose transposed rows form a basis of $\operatorname{null}(\nu)$. Equivalently, each row $a$ of $A$, viewed as a vector in $\mathbb{R}^{F}$, satisfies

$$
\nu a^T = 0
$$

and therefore

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

Each row of $A$ represents one independent conserved combination of ASM components under the adopted stoichiometric model and system boundary. The exact physical interpretation depends on the chosen basis and stoichiometric matrix. In activated-sludge applications, the conserved combinations may correspond to pools such as COD equivalents, nitrogen equivalents, phosphorus equivalents, or charge-related balances, but only when those balances are actually implied by the adopted reaction network and boundary. The invariants are algebraic consequences of $\nu$ and the boundary definition; they do not acquire physical meaning automatically.

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

The reaction may redistribute material between the two components, but it cannot change the conserved total represented by $c_1 + c_2$. In the non-negative ICSOR setting, the feasible effluent set is therefore the non-negative line segment satisfying this equality. The later correction step will choose any point on that segment that minimizes the documented weighted component-space deviation from the affine reference.

### 4.5 ASM-flavored miniature example

A slightly more ASM-flavored toy example shows why both training and correction should remain in component space even if measured composites are later reported externally. Suppose the component vector is

$$
c = \begin{bmatrix} S_S \\ X_S \\ S_{NH} \end{bmatrix}
$$

where $S_S$ is soluble substrate, $X_S$ is particulate substrate, and $S_{NH}$ is ammonium. Let one simplified reaction convert soluble substrate into particulate substrate without changing ammonium,

$$
\nu = \begin{bmatrix}
-1 & 1 & 0
\end{bmatrix}
$$

so one admissible invariant basis is

$$
A = \begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

which preserves total COD equivalents in the first two components and preserves ammonium in the third. Suppose an external reporting convention later collapses the components to total COD and ammonium,

$$
I_{comp} = \begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

If

$$
c_{in} = \begin{bmatrix} 10 \\ 10 \\ 5 \end{bmatrix}
\qquad \text{and} \qquad
c_{raw} = \begin{bmatrix} -2 \\ 22 \\ 5 \end{bmatrix},
$$

the invariant equalities already hold, so $c_{aff} = c_{raw}$. The non-negative LP therefore reduces to selecting the feasible component state that minimizes the weighted component-space deviation from $c_{aff}$. Because $c_{aff} = [-2, 22, 5]^T$, the optimizer is $[0, 20, 5]^T$. If measured composites are later computed externally, both $c_{raw}$ and $[0, 20, 5]^T$ collapse to the same aggregate vector $[20, 5]^T$. The example is still toy-sized, but it captures the key logic of the full framework: invariants and non-negativity are enforced where the stoichiometric state actually lives, while measured composites remain a downstream reporting view that cannot by itself distinguish all component-space corrections.

## 5. Unconstrained Surrogate in ASM Component Space

### 5.1 Why the input is partitioned

In activated-sludge systems, operating conditions and influent component concentrations play different physical roles.

1. Operating variables such as hydraulic retention time, dissolved-oxygen setpoint, or recycle settings alter the process environment.
2. Influent component concentrations describe the material inventory entering that environment.

Treating those two groups as interchangeable predictors hides an important engineering distinction. ICSOR therefore partitions the input into an operational block $u$ and an influent component block $c_{in}$.

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

Retaining the full vectorized quadratic blocks avoids hidden indexing conventions and makes the later estimation problem unambiguous. This second-order polynomial basis is a modeling choice rather than a theorem. It is used here because it retains linear terms, self-quadratic curvature, and operation-loading interactions in one explicit design matrix. The cost is rapid growth of $D$; when the sample count is not large relative to $D$, the later estimation problem can become ill-conditioned and motivates ridge regularization, as discussed in Section 8.4.

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
5. $\Theta_{uc}(u \otimes c_{in})$ captures the operation-loading interaction that remains central to the ICSOR design.

The model is therefore a partitioned second-order regression model. For that reason, the framework is referred to here as invariant-constrained second-order regression (ICSOR): second-order because the surrogate includes linear, self-quadratic, and cross-interaction terms, and invariant-constrained because the deployed correction is defined by stoichiometric invariant equalities together with non-negativity.

The raw surrogate is flexible enough to capture curvature and interaction, but it is data-driven and unconstrained. There is no reason for $c_{raw}$ to satisfy the invariant relation $A c_{raw} = A c_{in}$ or the componentwise non-negativity requirement without an additional correction step. At this stage, however, $B$ is already the component-space coefficient matrix to be estimated directly from component-space training pairs. Section 8 shows that ridge regression targets $B$ itself rather than a measured-space affine core. Rank deficiency and duplicated second-order features can still weaken uniqueness, but the earlier measured-space representability ambiguity disappears because the training targets are ASM component fractions rather than collapsed composites. The next section therefore introduces the constrained correction that separates learned variation from physically required structure.

## 6. Linear-Program Selection on the Invariant-Consistent Non-Negative Set

### 6.1 Affine invariant set

For a fixed influent state $c_{in}$, define the invariant-consistent affine set

$$
\mathcal{S}(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in} \}
$$

This set contains exactly those effluent component states whose conserved combinations match the influent conserved combinations under the adopted stoichiometric model. Because the right-hand side depends on $c_{in}$, the set is affine rather than linear.

### 6.2 Orthogonal affine projection as reference solution

Before imposing non-negativity, ICSOR can correct the raw surrogate by solving

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

This expression remains important in the present article for three reasons. First, it provides the simplest invariant-consistent reference solution. Second, it defines the component-space anchor $c_{aff}$ used by the later LP objective. Third, it is the exact output of the full LP deployment rule whenever the orthogonal affine projection already lies in the nonnegative orthant. It is therefore retained as a derivational benchmark rather than discarded.

### 6.3 Non-negative feasible set

The deployed non-negative ICSOR predictor is defined on the smaller feasible set

$$
\mathcal{S}_+(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in}, \; c \ge 0 \}
$$

where $c \ge 0$ is understood componentwise. This set intersects the invariant-consistent affine space with the nonnegative orthant. In geometric terms, the affine set removes changes that violate the stoichiometric invariants, while the orthant removes candidate states with negative ASM component concentrations.

### 6.4 Non-negative correction as a single linear program

After the affine reference state has been formed, the non-negative deployment step is defined by one LP in component space only. Choose a positive weight vector

$$
w_c \in \mathbb{R}_{++}^{F}.
$$

Introduce an auxiliary vector $r \in \mathbb{R}^{F}$ that bounds the componentwise absolute deviations from $c_{aff}$. The LP is

$$
\min_{c \in \mathbb{R}^{F},\, r \in \mathbb{R}^{F}} \; w_c^T r
$$

subject to

$$
A c = A c_{in},
\qquad
c \ge 0,
$$

$$
-r \le c - c_{aff} \le r,
\qquad
r \ge 0.
$$

Because $r_f$ bounds $\lvert c_f - c_{aff,f} \rvert$, the objective is the weighted component-space $L_1$ deviation from the affine reference. The feasible set is polyhedral and the objective is linear, so the correction problem is a single linear program.

We denote any optimizer of this LP by $c^*$. The role of $c_{raw}$ is indirect but essential: it determines $c_{aff}$, and $c_{aff}$ in turn anchors the component-space correction objective.

### 6.5 Feasibility, existence, and possible non-uniqueness

The first question is whether the feasible set can be empty. This is where Assumption 10 matters. Under the present modeling assumptions, it is not. If the influent component state is non-negative, then $c = c_{in}$ is feasible because it satisfies

$$
A c_{in} = A c_{in},
\qquad
c_{in} \ge 0.
$$

Hence

$$
c_{in} \in \mathcal{S}_+(c_{in}).
$$

Moreover, this feasible point yields a finite objective value after choosing the admissible auxiliary vector

$$
r = \lvert c_{in} - c_{aff} \rvert.
$$

Because the objective is bounded below by zero, the LP has a finite optimum, and standard linear-program theory implies that at least one optimizer exists.

That optimizer need not be unique. The baseline formulation in this article does not impose an additional deterministic selection rule beyond the documented objective and constraints. Accordingly, any optimizer may be used as the deployed component state. If a software implementation requires a unique deployed state, that extra convention should be stated separately as an implementation choice rather than as part of the baseline mathematical model.

### 6.6 Primal-dual characterization of the combined LP

Introduce multipliers $\pi \in \mathbb{R}^{q}$ for the equality constraints, $\gamma^+, \gamma^- \in \mathbb{R}_{\ge 0}^{F}$ for the upper and lower component-space absolute-deviation bounds, $\mu \in \mathbb{R}_{\ge 0}^{F}$ for componentwise non-negativity, and $\delta \in \mathbb{R}_{\ge 0}^{F}$ for $r \ge 0$. The Lagrangian of the component-space LP is

$$
\begin{aligned}
\mathcal{L}(c, r, \pi, \gamma^+, \gamma^-, \mu, \delta)
= {} & w_c^T r + \pi^T(A c - A c_{in}) \\
& + (\gamma^+)^T(c - c_{aff} - r)
+ (\gamma^-)^T(-c + c_{aff} - r) \\
& - \mu^T c - \delta^T r.
\end{aligned}
$$

The corresponding primal-dual optimality conditions are

$$
A^T \pi + (\gamma^+ - \gamma^-) - \mu = 0,
$$

$$
w_c - \gamma^+ - \gamma^- - \delta = 0,
$$

$$
A c = A c_{in},
\qquad
c \ge 0,
$$

$$
-r \le c - c_{aff} \le r,
\qquad
r \ge 0,
$$

$$
\gamma^+, \gamma^-, \mu, \delta \ge 0,
$$

together with complementary slackness for every active inequality:

\gamma_f^+ \big((c - c_{aff})_f - r_f\big) = 0,
$$

$$
\gamma_f^- \big(-(c - c_{aff})_f - r_f\big) = 0,
$$

$$
\mu_f c_f = 0,
\qquad
\delta_f r_f = 0.
$$

The first stationarity relation shows that the selected component state balances three geometric objects: the invariant normals in the row space of $A$, the component-space discrepancy normals anchored at $c_{aff}$, and the active faces of the non-negative orthant. The remaining stationarity relation allocates the documented component weights across the signed absolute-deviation faces and any inactive slack.

### 6.7 Relation to the orthogonal affine projector

The orthogonal affine projection remains embedded in the new formulation as a special case. If the affine projector already satisfies non-negativity, then the combined LP returns exactly the same point.

Indeed, suppose

$$
c_{aff} \ge 0
$$

componentwise. Then $c_{aff} \in \mathcal{S}_+(c_{in})$. At that point,

$$
c_{aff} - c_{aff} = 0,
$$

so the LP can choose $r = 0$ and attain objective value zero. Because $r \ge 0$ and $w_c > 0$, no feasible point can improve on zero. Moreover, objective value zero forces $r = 0$, and $r = 0$ implies $c = c_{aff}$. Therefore every optimizer satisfies

$$
c^* = c_{aff}
\qquad \text{whenever} \qquad
c_{aff} \ge 0.
$$

This result preserves continuity with the earlier affine-only ICSOR theory. The non-negative formulation does not discard the affine projector; it uses the affine projector as the equality-consistent reference and returns it exactly whenever no further correction is needed.

### 6.8 Efficient deployment sequence

The deployed correction should still be evaluated in a staged order so that the LP machinery is activated only when it is actually needed.

1. Form the raw component prediction $c_{raw}$.
2. If $c_{raw}$ already satisfies $A c_{raw} = A c_{in}$ and $c_{raw} \ge 0$, return $c_{raw}$ directly. In that case $c_{aff} = c_{raw}$ and the component-space LP has objective value zero at $c_{raw}$.
3. Otherwise compute the closed-form affine projection $c_{aff} = P_{adm} c_{raw} + P_{inv} c_{in}$.
4. If $c_{aff} \ge 0$, return $c_{aff}$. Section 6.7 shows that this is exactly the optimizer of the component-space LP.
5. Otherwise solve the single component-space LP.
6. If measured composites are needed for reporting, compute them externally as $y_{ext} = I_{comp} c^*$.

This order follows directly from the construction above. The first check avoids unnecessary correction when the raw surrogate already lies in the feasible set. The second check uses the orthogonal affine projector as the cheapest exact repair of invariant violations. The LP stage is therefore a residual correction step for the subset of samples in which the affine projector still leaves the non-negative orthant.

One degenerate case is worth noting. If $A$ has zero rows, then there is no non-trivial invariant equality to enforce and $c_{aff} = c_{raw}$. The correction problem then becomes a separable non-negativity LP that minimizes a weighted component-space deviation from $c_{raw}$. Because the objective decouples coordinatewise when the equality constraints disappear, the optimizer reduces to componentwise clipping: $c_f^* = \max\{c_{raw,f}, 0\}$ for every component $f$.

### 6.9 Reduced null-space LP formulation

The affine reference state also yields a reduced formulation that removes the equality constraints exactly. Let $N_A \in \mathbb{R}^{F \times (F-q)}$ have orthonormal columns spanning $\operatorname{null}(A)$. Every point in the affine set can then be written as

$$
c = c_{aff} + N_A z
$$

for some reduced coordinate vector $z \in \mathbb{R}^{F-q}$, because $A c_{aff} = A c_{in}$ and $A N_A = 0$.

Substituting this parameterization into the component-space LP gives

$$
\min_{z \in \mathbb{R}^{F-q},\, r \in \mathbb{R}^{F}} \; w_c^T r
$$

subject to

$$
N_A z \ge -c_{aff},
$$

$$
-r \le N_A z \le r,
\qquad
r \ge 0.
$$

The equality constraints are now absorbed into the affine parameterization, and the component-space residual is simply

$$
(c_{aff} + N_A z) - c_{aff} = N_A z.
$$

The reduced formulation therefore remains linear throughout. There is no reduced Hessian, no Euclidean metric preservation argument, and no solver structure tied specifically to quadratic programming. The fixed matrix is $N_A$, and sample dependence enters through the right-hand side $-c_{aff}$. An orthonormal null-space basis is still numerically convenient because it keeps the reduced coordinates well scaled, but its role is now computational rather than metric.

### 6.10 What the LP correction guarantees and what it does not

The single combined LP guarantees that every optimizer satisfies

$$
A c^* = A c_{in}
$$

and

$$
c^* \ge 0.
$$

It also guarantees that no feasible component state achieves a smaller value of

$$
w_c^T \lvert c - c_{aff} \rvert
$$

when the absolute values are interpreted componentwise through the auxiliary-variable formulation above. What it does not guarantee is equally important.

1. It does not guarantee realistic upper bounds or process-feasible operating ranges.
2. It does not impose kinetic feasibility beyond the chosen invariant relations and non-negativity.
3. It does not guarantee thermodynamic admissibility.
4. It does not correct errors introduced by a misspecified stoichiometric basis or an incorrect system boundary.
5. It does not by itself guarantee non-negative external measured composites unless the composition map has the sign structure discussed in Section 7.3.
6. It does not guarantee a unique deployed state unless extra selection rules are added beyond the baseline LP.

These are not derivation errors. They are the exact consequences of the feasible set and combined objective being enforced.

## 7. External Composition-Matrix Collapse

### 7.1 External measured output equation

The ICSOR model predicts only ASM component fractions. If measured composite outputs are needed for reporting, one computes them externally through

$$
y_{ext} = I_{comp} c^*.
$$

This equation is not part of the regression model itself. It is a downstream linear transformation of the deployed component prediction.

### 7.2 Why the collapse is external

This separation is substantive, not cosmetic.

1. The learned object is the ASM component-fraction map $c^*(u, c_{in})$.
2. The composition matrix $I_{comp}$ is an external reporting operator.
3. Changing the reporting convention changes $I_{comp}$, not the trained ICSOR model.

This is useful because different studies may want different measured aggregates while still using the same ASM component predictor. One study may collapse to COD, TN, and TP; another may include TSS and VSS as well. Those are downstream choices.

### 7.3 When component non-negativity implies non-negative composites

If every row of $I_{comp}$ is entrywise non-negative, then non-negative component predictions imply non-negative reported composites. For output index $k$,

$$
y_{ext,k}
= \sum_{f=1}^{F} (I_{comp})_{k f} c_f^*.
$$

If $(I_{comp})_{k f} \ge 0$ for all $f$ and $c_f^* \ge 0$ for all $f$, then

$$
y_{ext,k} \ge 0.
$$

This is the common case for composite definitions built as sums of COD-bearing, nitrogen-bearing, phosphorus-bearing, or solids-bearing fractions with non-negative conversion factors.

### 7.4 Final component prediction and optional reporting map

The native deployed output of non-negative ICSOR is the corrected ASM component-fraction vector $c^*$. If an application later requires measured composites, the externally reported vector is

$$
y_{ext}(u, c_{in}) = I_{comp} c^*(u, c_{in}).
$$

The prediction target, the correction logic, and the primary scientific interpretation all remain attached to $c^*$ rather than to $y_{ext}$.

## 8. Estimation and Interpretation from Component-Space Data

At the estimation stage, the primary inferential object is the component-space coefficient matrix $B$. Because the training targets live in ASM component space from the start, there is no separate measured-space affine core to identify and no representability step is required before deployment. Ridge regularization is introduced here to stabilize the high-dimensional second-order design, not to compensate for an imposed measurement collapse.

### 8.1 Dataset-level component regression model

Let $N$ steady-state samples be available, and store samples by rows:

$$
\Phi = \begin{bmatrix}
\phi(u_1, c_{in,1})^T \\
\phi(u_2, c_{in,2})^T \\
\vdots \\
\phi(u_N, c_{in,N})^T
\end{bmatrix} \in \mathbb{R}^{N \times D},
$$

$$
C_{out} = \begin{bmatrix}
c_{out,1}^T \\
c_{out,2}^T \\
\vdots \\
c_{out,N}^T
\end{bmatrix} \in \mathbb{R}^{N \times F}.
$$

The raw component-space model satisfies

$$
C_{out} = \Phi B^T + E_c,
$$

where $E_c \in \mathbb{R}^{N \times F}$ collects component-space model and measurement errors. The formulation assumes that effluent ASM component-fraction targets are available for training, either directly from a simulator or through an upstream reconstruction step outside the present model.

### 8.2 What the data identify now

Because the model is trained directly on $C_{out}$, the primary inferential target is the component-space coefficient matrix $B$ itself. The earlier ambiguity between an identifiable measured-space operator and a non-identifiable latent component-space representative disappears once the training targets are ASM component fractions.

That does not mean every individual coefficient is automatically unique. Weak excitation of the feature basis, duplicated quadratic monomials, or small sample size can still make the fitted coefficients unstable or non-unique. Ridge shrinkage is used here precisely because it stabilizes those weakly supported directions at the price of bias.

### 8.3 Ridge estimator of the component-space coefficients

The ridge estimator of $B$ is

$$
\widehat B_\lambda^T = \arg\min_{Q \in \mathbb{R}^{D \times F}} \left\{ \lVert C_{out} - \Phi Q \rVert_F^2 + \lambda \lVert \Gamma^{1/2} Q \rVert_F^2 \right\}.
$$

Here $\lambda \ge 0$ controls the amount of shrinkage and $\Gamma \succeq 0$ defines which coefficient directions are penalized. A common choice is

$$
\Gamma = \operatorname{diag}(0, 1, \ldots, 1),
$$

which leaves the intercept unpenalized and shrinks the remaining feature directions. Because ridge shrinkage is scale dependent, the non-intercept columns of $\Phi$ should either be centered and scaled before tuning $\lambda$ or the intended scaling should be encoded explicitly in $\Gamma$.

When the penalized normal matrix is invertible, the solution is

$$
\widehat B_\lambda^T = (\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T C_{out},
$$

with the Moore-Penrose pseudoinverse used in place of the inverse if the penalized normal matrix is singular. The OLS estimator is recovered as the special case $\lambda = 0$ when the unpenalized normal equations are well posed.

### 8.4 Ridge regularization, conditioning, and interpretation

Second-order feature maps can be high dimensional, and real wastewater datasets may not excite all directions of that design space. If $N < D$, full column rank is impossible from the outset; even when $N \ge D$, the design may still be rank deficient because some feature directions are weakly or redundantly excited. Ridge regularization is introduced precisely because those conditions make unpenalized estimation unstable or non-unique.

That numerical stability does not restore unique identification of the unpenalized coefficient vector. Instead, $\widehat B_\lambda$ is a penalty-dependent shrinkage estimate of the component-space coefficient matrix. As $\lambda$ increases, coefficient magnitudes contract toward the null directions favored by $\Gamma$, and the fit trades variance for bias. Interpretation should therefore focus on the following objects in order of reliability:

1. predicted ASM component fractions,
2. blockwise contribution patterns that remain stable across a reasonable range of penalties,
3. individual penalized coefficients only when the design matrix is sufficiently informative.

### 8.5 Final deployed predictor after estimation

Once $\widehat B_\lambda$ has been estimated, the raw component prediction for a new sample is

$$
\widehat c_{raw,\lambda} = \widehat B_\lambda \phi(u, c_{in}).
$$

The final deployed non-negative component prediction is then obtained by the same staged component-space logic introduced in Section 6: keep $\widehat c_{raw,\lambda}$ if it is already feasible, otherwise apply the affine projector, and if non-negativity still fails solve the single component-space LP with $\widehat c_{raw,\lambda}$ in place of $c_{raw}$. We denote any optimizer of that LP by $\widehat c_\lambda^*$.

If an application later requires measured composites, the externally reported prediction is

$$
\widehat y_{ext,\lambda} = I_{comp} \widehat c_\lambda^*.
$$

That last equation is a reporting formula, not a redefinition of the model target.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Error model and conditioning on the ridge penalty

For statistical inference, suppose the row errors of $E_c$ are independent across samples and satisfy

$$
\mathbb{E}[E_c \mid \Phi] = 0
$$

and

$$
\operatorname{Var}(\operatorname{vec}(E_c) \mid \Phi) = \Sigma_c \otimes I_N
$$

where $\Sigma_c \in \mathbb{R}^{F \times F}$ is the within-sample covariance across ASM component outputs.

Throughout this section, the ridge penalty parameter $\lambda$ and penalty matrix $\Gamma$ are treated as fixed. If $\lambda$ is selected by cross-validation, generalized cross-validation, or another data-adaptive rule, the formulas below are conditional on the selected penalty and therefore omit tuning uncertainty unless that selection step is repeated inside resampling.

Define

$$
A_\lambda = (\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T
$$

so that

$$
\widehat B_\lambda^T = A_\lambda C_{out}.
$$

Also define the ridge fitted-value matrix

$$
W_\lambda = \Phi A_\lambda = \Phi (\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T
$$

and the fitted residual matrix

$$
\widehat E_{c,\lambda} = C_{out} - \Phi \widehat B_\lambda^T = (I_N - W_\lambda) C_{out}.
$$

A common plug-in estimator of the within-sample component-output covariance is

$$
\widehat \Sigma_{c,\lambda} = \frac{1}{N - \operatorname{df}_\lambda} \widehat E_{c,\lambda}^T \widehat E_{c,\lambda},
\qquad
\operatorname{df}_\lambda = \operatorname{tr}(W_\lambda),
$$

where $\operatorname{df}_\lambda$ is the effective degrees of freedom of the ridge fit. Unlike the OLS covariance estimator, this is best interpreted as a working plug-in estimate rather than an exact unbiased finite-sample formula because ridge estimation is biased.

### 9.2 Ridge coefficient bias and covariance

Define the coefficient-space shrinkage matrix

$$
S_\lambda = A_\lambda \Phi = (\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T \Phi.
$$

Then

$$
\mathbb{E}[\widehat B_\lambda^T \mid \Phi] = S_\lambda B^T
$$

and the conditional bias is

$$
\operatorname{Bias}(\widehat B_\lambda^T \mid \Phi) = (S_\lambda - I_D) B^T = -\lambda (\Phi^T \Phi + \lambda \Gamma)^{-1} \Gamma B^T.
$$

The conditional covariance is

$$
\operatorname{Var}(\operatorname{vec}(\widehat B_\lambda^T) \mid \Phi) = \Sigma_c \otimes (A_\lambda A_\lambda^T),
$$

or, equivalently,

$$
\operatorname{Var}(\operatorname{vec}(\widehat B_\lambda^T) \mid \Phi) = \Sigma_c \otimes \left[(\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T \Phi (\Phi^T \Phi + \lambda \Gamma)^{-1}\right]
$$

when $\Gamma$ is symmetric. Therefore, for the coefficient $\widehat B_{\lambda,fj}$,

$$
\operatorname{Var}(\widehat B_{\lambda,fj} \mid \Phi) = (\Sigma_c)_{ff} [A_\lambda A_\lambda^T]_{jj}.
$$

Unlike OLS, the conditional covariance does not fully characterize ridge coefficient uncertainty because the estimator is biased. A more honest coefficientwise summary is the conditional mean-squared error

$$
\operatorname{MSE}(\widehat B_{\lambda,fj} \mid \Phi) = (\Sigma_c)_{ff} [A_\lambda A_\lambda^T]_{jj} + \left[\operatorname{Bias}(\widehat B_{\lambda,fj} \mid \Phi)\right]^2.
$$

If $\lambda \to 0$ and $\Phi$ has full column rank, these expressions reduce to the familiar OLS covariance formulas.

### 9.3 Ridge raw and affine component-prediction uncertainty

For a new operating point with feature vector $\phi_* = \phi(u_*, c_{in,*})$, the fitted raw component-space mean prediction is

$$
\widehat c_{raw,\lambda,*} = \widehat B_\lambda \phi_*.
$$

Define the ridge leverage analogue

$$
s_{\lambda,*} = \phi_*^T A_\lambda A_\lambda^T \phi_*
$$

and the conditional bias vector of the raw component predictor

$$
b_{raw,\lambda,*} = B (S_\lambda^T - I_D) \phi_*.
$$

Then

$$
\operatorname{Var}(\widehat c_{raw,\lambda,*} \mid \phi_*, \Phi) = s_{\lambda,*} \Sigma_c
$$

and the conditional mean-squared-error matrix of the raw component prediction is

$$
\operatorname{MSE}(\widehat c_{raw,\lambda,*} \mid \phi_*, \Phi) = s_{\lambda,*} \Sigma_c + b_{raw,\lambda,*} b_{raw,\lambda,*}^T.
$$

If the deployed prediction stops at the affine projector, then

$$
\widehat c_{aff,\lambda,*} = P_{adm} \widehat c_{raw,\lambda,*} + P_{inv} c_{in,*},
$$

so

$$
\operatorname{Var}(\widehat c_{aff,\lambda,*} \mid \phi_*, c_{in,*}, \Phi) = s_{\lambda,*} P_{adm} \Sigma_c P_{adm}^T
$$

and

$$
\operatorname{MSE}(\widehat c_{aff,\lambda,*} \mid \phi_*, c_{in,*}, \Phi) = s_{\lambda,*} P_{adm} \Sigma_c P_{adm}^T + (P_{adm} b_{raw,\lambda,*})(P_{adm} b_{raw,\lambda,*})^T.
$$

The variance term plays the role of a ridge analogue of the OLS leverage formula, but the bias term has no OLS counterpart. Because the bias depends on the unknown population matrix $B$, fully closed-form finite-sample intervals require either a plug-in bias estimate, an asymptotic approximation, or resampling.

### 9.4 Why these formulas do not globally extend to the final non-negative predictor

The final deployed prediction is obtained by solving the single LP from Section 6 with $\widehat c_{raw,\lambda,*}$ in place of $c_{raw}$. This map is generally piecewise affine over polyhedral regions determined by the active non-negativity faces and the signed absolute-deviation faces of the component-space LP. At points where the optimal LP basis changes, the deployed mapping is not described by one global coefficient matrix. If the LP is non-unique, different optimizer selections can also produce different deployed outputs. Even away from those transitions, the upstream component-space fit is ridge biased and conditional on the chosen penalty. Consequently, the bias and covariance formulas above are informative local descriptors, not global exact finite-sample interval formulas, for the final deployed predictor.

Two special cases are simpler.

1. If the raw predictor is already invariant-consistent and non-negative at the prediction point, then the final deployed state equals $\widehat c_{raw,\lambda,*}$ and the raw ridge formulas apply exactly.
2. If only the affine projector is active and $\widehat c_{aff,\lambda,*} \ge 0$, then the affine ridge formulas above apply exactly.

Neither of those special cases yields a global exact interval formula for the final non-negative predictor, and neither removes tuning uncertainty when $\lambda$ is selected from the data.

### 9.5 Recommended uncertainty treatment for the final predictor

For the final deployed non-negative predictor, the most defensible default approach is resampling-based uncertainty quantification, such as bootstrap refitting or residual bootstrap, because it propagates uncertainty through both stages of the model: ridge coefficient estimation and the sample-specific LP correction. Each replicate should rebuild the affine reference and rerun the component-space LP when needed. If the ridge penalty is tuned from the data, that tuning step should be repeated inside each replicate if one wants the resampling interval to include penalty-selection uncertainty. If measured composites are later reported, they should be computed from each bootstrap component prediction by the same external composition matrix.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct effluent-state parameterization

The surrogate is parameterized on the effluent ASM component-fraction state rather than on measured composites or on the net change alone. This keeps the learned target aligned with the mechanistic state description and makes the primary scientific interpretation live in component space.

### 10.2 Partitioned second-order feature structure

The partitioned feature map separates operating effects, influent-composition effects, and operation-loading interactions in a way that is interpretable to process engineers. The price of that interpretability is rapid feature growth, which can create multicollinearity, unstable coefficients, and weakly identified directions if the dataset does not adequately excite the design space. Ridge regularization is the deliberate response to that high-dimensional geometry.

### 10.3 Component-space LP correction

The deployed correction is not defined by Euclidean nearest-point geometry alone. Instead, it uses the orthogonal affine projector as a reference and, when needed, solves a component-space $L_1$ LP on the invariant-consistent non-negative set. The weights $w_c$ determine how strongly each ASM component is protected against deviation from the affine reference.

### 10.4 External measurement collapse

Because the model is trained and corrected entirely in component space, the composition matrix becomes a downstream reporting choice. This keeps stoichiometric reasoning and non-negativity claims attached to the ASM state rather than to an aggregate that can hide multiple internal redistributions.

### 10.5 Ridge-stabilized component-space coefficients

Ridge regression now estimates the component-space coefficient matrix $B$ directly from component-fraction targets. That removes the earlier need to infer a latent component representative from a measured-space fit, although weak design excitation and duplicated features can still limit coefficient uniqueness and ridge shrinkage introduces deliberate bias.

## 11. Limitations

Non-negative ICSOR with ridge regression is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state in the quasi-steady-sample sense and does not represent temporal dynamics or path dependence.
2. It requires effluent ASM component-fraction targets for training, either directly from a simulator or from an upstream reconstruction step.
3. It enforces only the invariant relations encoded by the chosen stoichiometric basis and system boundary together with componentwise non-negativity.
4. Non-negative component fractions do not guarantee full kinetic, biological, or thermodynamic feasibility.
5. The LP correction depends on the chosen component-space weights $w_c$.
6. The final deployed predictor is not globally affine once LP-regime changes become active.
7. The baseline single-LP formulation need not be single-valued; if a unique deployed component state is required, an additional optimizer-selection convention must be added.
8. Exact closed-form ridge bias-variance formulas are available for the raw and affine linear stages only conditionally on a fixed penalty, but not in general for the final LP-corrected deployed predictor.
9. The second-order feature basis can be statistically fragile if it is weakly excited or highly collinear.
10. Ridge stabilization introduces shrinkage bias, and penalty tuning itself can materially affect the fitted component-space coefficients.
11. A misspecified stoichiometric matrix or incorrect system boundary leads to a formally correct LP-optimal state on the wrong physical constraint set.
12. Non-negative external measured composites are supported most naturally only when the composition matrix has non-negative rows.
13. If influent or effluent ASM component fractions are reconstructed upstream rather than observed directly, reconstruction error enters before the regression stage and is not represented by the ridge formulas above.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its claims correctly.

## 12. Conclusion

Non-negative ICSOR with ridge regression is formulated here as a direct ASM component-space surrogate. It takes operational variables and influent ASM component fractions as input, and it predicts effluent ASM component fractions in the same basis. The raw coefficient matrix is estimated directly by ridge regression from component-space training data, which stabilizes ill-conditioned second-order designs by shrinking weakly supported directions. Stoichiometric invariants and componentwise non-negativity are then enforced at deployment by a staged component-space correction consisting of a raw-feasibility shortcut, an affine invariant projection, and, when necessary, a single component-space LP.

Measured composites are no longer part of the model target or the correction objective. They are obtained only afterward as an external calculation through the composition matrix. Under that reading, ICSOR is best understood as a ridge-stabilized, second-order, component-space surrogate whose primary output is the ASM component-fraction vector itself and whose measured-output reporting layer sits outside the model.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
4. Seber, G. A. F., and Lee, A. J. Linear Regression Analysis. 2nd ed. Wiley, 2003.
5. Rao, C. R., and Mitra, S. Generalized Inverse of Matrices and Its Applications. Wiley, 1971.
6. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.
7. Hoerl, A. E., and Kennard, R. W. Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics 12(1), 55-67, 1970.
8. Hastie, T., Tibshirani, R., and Friedman, J. The Elements of Statistical Learning. 2nd ed. Springer, 2009.
