# Non-Negative Invariant-Constrained Second-Order Regression (ICSOR) for Activated Sludge Surrogate Modeling

## Abstract

This article presents a non-negative formulation of invariant-constrained second-order regression (ICSOR), a physics-informed surrogate model for steady-state activated-sludge systems. The purpose of ICSOR is to predict measured effluent variables from operating conditions and influent activated-sludge-model (ASM) component concentrations while preserving the stoichiometric invariants implied by the adopted reaction network and enforcing non-negativity of the predicted effluent ASM component state. The key difficulty is that conservation laws are defined in ASM component space, whereas plant observations are usually reported as composite variables such as total COD, total nitrogen, total phosphorus, or suspended solids. A regression model built only in measured-output space can fit those aggregates while still implying an impossible redistribution of the underlying ASM components. An affine invariant projection resolves only part of that mismatch because it restores stoichiometric consistency but can still produce negative component predictions. The non-negative ICSOR formulation therefore uses a two-stage correction in component space. First, a partitioned second-order surrogate produces an unconstrained prediction of the effluent ASM component state. Second, a convex projection maps that raw prediction onto the intersection of the invariant-consistent affine set and the nonnegative orthant. The corrected component state is then collapsed into measured output space through a linear composition map.

The framework is written as a self-contained theory section for readers in chemical engineering, wastewater process modeling, and machine learning. All symbols are defined before use. The invariant constraint is derived from the stoichiometric change relation rather than asserted heuristically. The non-negative correction is formulated as a strictly convex quadratic program, and the role of the earlier orthogonal affine projector is retained explicitly as a reference solution and inactive-constraint special case. In deployment, that quadratic program is needed only when the raw prediction is infeasible and the closed-form affine projector still violates componentwise non-negativity. The distinction between identifiable affine measured-space coefficients and non-identifiable latent component-space coefficients is preserved. Accordingly, the affine core is identifiable from measured composite data, whereas the final inequality-constrained deployed predictor is fully specified only after a latent component-space representative has been fixed or additional component-space information has been supplied. The limits of exact closed-form uncertainty analysis for that deployed predictor are stated explicitly. The result is a precise formulation of what non-negative ICSOR guarantees, what is stabilized by ridge estimation of the affine core, how that shrinkage changes uncertainty quantification, and what must instead be handled through convex post-estimation correction.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to represent a quasi-steady operating condition: the operating variables, influent composition, and effluent response are treated as effectively time-invariant over the control volume being modeled for the sampling window of interest. The usual difficulty is that a generic data-driven regressor can fit observed effluent data while still violating fundamental conservation structure. In activated-sludge modeling, that failure is not a minor technical detail. It undermines the physical credibility of the surrogate because it can imply component inventories that are inconsistent with the adopted reaction network even when the measured aggregates appear plausible.

The source of the problem is a mismatch between two spaces.

1. The mechanistic stoichiometric model is written in an ASM component basis, such as soluble substrate, ammonium, nitrate, autotrophic biomass, particulate organics, phosphate, dissolved oxygen, and alkalinity.
2. The plant or simulator often reports outputs in measured composite variables, such as total COD, total nitrogen, total phosphorus, TSS, or VSS.

These two spaces are related, but they are not the same. Conservation laws are naturally expressed in the ASM component basis because the stoichiometric matrix acts on individual components. Observations, however, are usually available only after those components have been aggregated into measurable composites. A surrogate that learns only in measured-output space may reproduce the observed aggregates while obscuring physically impossible changes in the underlying component inventory.

The earlier affine-only ICSOR formulation addresses that mismatch by predicting in component space and projecting the raw surrogate output onto the affine set consistent with the stoichiometric invariants. That construction repairs invariant violations, but it does not ensure that the projected component concentrations are non-negative. Negative component predictions are a serious defect in the present setting because component concentrations are themselves physical quantities and because negative components can propagate into implausible reported composites. The non-negative ICSOR formulation developed here therefore strengthens the correction step. The affine invariant projection remains part of the derivation, but the deployed predictor is the closest point to the raw surrogate that satisfies both the invariant relations and componentwise non-negativity.

The model is constructed to answer one precise question:

> Given a steady-state influent state and a steady-state operating condition, what measured effluent state should be predicted if the underlying effluent ASM component state must satisfy the conserved quantities implied by the adopted stoichiometric model and must remain non-negative componentwise?

The theory in this article is restricted to steady-state reactor-block prediction. It does not aim to replace a dynamic activated-sludge simulator. Rather, it provides an analytically structured surrogate that preserves stoichiometric structure, enforces non-negativity at the deployed component-state level, and remains simple enough that its affine core can still be estimated directly from data by ridge regression. That last point requires care: measured composite data identify the affine core, ridge stabilization improves the conditioning of the high-dimensional second-order design at the price of shrinkage bias, and any component-space inequality correction in deployment requires either a chosen latent representative or additional component-space information. In deployment, the correction is evaluated in stages: no correction when the raw component state is already feasible, closed-form affine projection when only the invariant equalities are violated, and quadratic-program correction only when non-negativity remains violated after the affine step. The discussion proceeds from physical scope and notation, to derivation of the invariant relations, to convex non-negative projection, to collapse into measured space, and finally to estimation and uncertainty with explicit attention to the ridge-induced implications for uncertainty quantification.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit represented by quasi-steady samples. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes transport or removal mechanisms such as bypass streams, gas stripping, chemical dosing, or sludge wastage if they cross the chosen boundary and are not encoded in the stoichiometric description. The theory therefore applies only after the modeler has fixed the following items:

1. the reactor or process block being represented,
2. the ASM component basis used to describe material composition,
3. the stoichiometric matrix associated with that basis, and
4. the measurement map used to aggregate component concentrations into observed composite variables.

The framework is steady-state in the sense of quasi-steady samples rather than full dynamic trajectories. It does not represent settling dynamics, sludge age dynamics, sensor dynamics, start-up transients, or time-varying trajectories. In plant applications, the samples would typically correspond to stable operating windows or time-averaged periods rather than literal mathematical equilibria. Changing the system boundary changes the admissible stoichiometric change space and therefore changes the invariant and non-negative feasible sets.

### 2.2 Why two state spaces are needed

To make the distinction concrete, suppose the underlying component basis contains soluble biodegradable substrate, particulate biodegradable substrate, ammonium, nitrate, phosphate, dissolved oxygen, alkalinity, and biomass fractions. A plant rarely measures all of those components directly. Instead, it may report total COD, total nitrogen, total phosphorus, TSS, and VSS. Those measured variables are linear combinations of the component concentrations under a chosen analytical convention.

The surrogate must therefore operate across two linked spaces:

1. ASM component space, where stoichiometry, invariants, and non-negativity are defined.
2. Measured composite space, where prediction targets are observed and evaluated.

ICSOR learns and constrains the prediction in component space and only then maps the result to measured space. That order remains essential in the non-negative formulation. Stoichiometric invariants originate in the component basis, and the componentwise non-negativity claim is also made in that basis. A measured-space-only correction would generally be too weak to control the underlying ASM state.

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
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank matrix whose transposed rows form a basis of $\operatorname{null}(\nu)$, equivalently $A \nu^T = 0$ |
| $P_{inv}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the row space of $A$ |
| $P_{adm}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the admissible change space, $I_F - P_{inv}$ |
| $N_A$ | $\mathbb{R}^{F \times (F-q)}$ | Matrix whose columns form an orthonormal basis of $\operatorname{null}(A)$ |
| $\phi(u, c_{in})$ | $\mathbb{R}^{D}$ | Engineered second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $\lambda$ | scalar | Ridge penalty parameter used in affine-core estimation |
| $\Gamma$ | $\mathbb{R}^{D \times D}$ | Symmetric positive-semidefinite ridge penalty matrix, commonly diagonal with the intercept left unpenalized |
| $B$ | $\mathbb{R}^{F \times D}$ | Raw component-space coefficient matrix |
| $G$ | $\mathbb{R}^{K \times F}$ | Measured-space affine constrained operator, $G = I_{comp} P_{adm}$ |
| $H$ | $\mathbb{R}^{K \times F}$ | Measured-space invariant carry-through operator, $H = I_{comp} P_{inv}$ |
| $M$ | $\mathbb{R}^{K \times D}$ | Effective identifiable affine measured-space coefficient matrix, $M = G B$ |

Throughout the estimation and uncertainty sections, $\lambda$ and $\Gamma$ are treated as fixed unless data-driven penalty selection is being discussed explicitly.

The measured effluent variables are defined by the linear map

$$
y = I_{comp} c_{out}
$$

This linear composition map is standard in activated-sludge modeling when measured variables are aggregates of ASM components. For example, total COD or total nitrogen is formed by summing the relevant component concentrations with appropriate conversion factors. In the common case of total COD, total nitrogen, total phosphorus, TSS, or VSS built as sums with non-negative conversion factors, the rows of $I_{comp}$ are entrywise non-negative. In the non-negative ICSOR setting, that sign structure matters because non-negative component predictions imply non-negative measured composites only under that condition. Throughout the chapter, $I_{comp}$ is treated as known and fixed; if its coefficients are themselves estimated, that uncertainty lies outside the present error model.

## 3. Modeling Assumptions

The framework rests on the following assumptions. These are not optional preferences left to the reader. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a quasi-steady input-output condition, typically a stable operating epoch or time-averaged window. The model is not a dynamic state estimator.
2. **Fixed component basis.** The ASM component basis and the associated stoichiometric matrix are fixed before regression begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement. Any external source or sink outside that boundary is outside the present model.
4. **Linear composition map.** Measured effluent variables are linear combinations of the underlying ASM component concentrations through a known fixed matrix $I_{comp}$.
5. **Direct effluent-state parameterization.** The surrogate is parameterized to predict the effluent ASM component state $c_{out}$ directly rather than the component change $c_{out} - c_{in}$.
6. **Second-order surrogate class.** The raw surrogate is a partitioned second-order polynomial model that includes linear, quadratic, and operation-loading interaction terms.
7. **Euclidean correction metric.** The deployed correction is defined as the smallest Euclidean adjustment in ASM component concentration space that restores the required constraints.
8. **Constraint scope.** The final projection enforces the stoichiometric invariants implied by the chosen basis and system boundary together with componentwise non-negativity. It does not enforce upper bounds, kinetic feasibility, or thermodynamic admissibility beyond those conditions.
9. **Influent feasibility.** The influent reference state is assumed non-negative in component space. Under that assumption the non-negative feasible set is non-empty because the influent state itself satisfies the invariant equalities.
10. **Composite-sign scope.** Non-negative component predictions imply non-negative measured composites only when the relevant rows of $I_{comp}$ are entrywise non-negative. If the measurement convention uses negative coefficients, extra output-space sign constraints would be required for a composite non-negativity guarantee.
11. **Statistical scope.** Ridge-based uncertainty statements are interpreted conditional on a fixed penalty parameter $\lambda$ and penalty matrix $\Gamma$ under independent-sample Gaussian multivariate linear-model assumptions. Closed-form bias and covariance expressions are available for the affine core, but classical unbiased OLS $t$-based intervals do not apply unchanged, and the final inequality-constrained deployed predictor generally requires resampling that repeats any data-driven tuning of $\lambda$.

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

The reaction may redistribute material between the two components, but it cannot change the conserved total represented by $c_1 + c_2$. In the non-negative ICSOR setting, the feasible effluent set is therefore the non-negative line segment satisfying this equality. The later correction step will select the point on that segment closest to the raw surrogate prediction.

### 4.5 ASM-flavored miniature example

A slightly more ASM-flavored toy example shows why projection must happen before measurement collapse. Suppose the component vector is

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

which preserves total COD equivalents in the first two components and preserves ammonium in the third. Suppose the measured outputs are total COD and ammonium,

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
c_{raw} = \begin{bmatrix} -1 \\ 21 \\ 5 \end{bmatrix},
$$

then the measured output is already

$$
y_{raw} = I_{comp} c_{raw} = \begin{bmatrix} 20 \\ 5 \end{bmatrix},
$$

which is exactly the same measured output produced by the feasible state $[0, 20, 5]^T$. A measured-space-only correction cannot distinguish those two component states, because both collapse to the same aggregates. The component-space non-negative projection, however, detects that $c_{raw}$ violates $c \ge 0$ and moves the prediction to the nearest point on the feasible set $A c = A c_{in}$, $c \ge 0$, namely $[0, 20, 5]^T$ in this simple case. The example is still toy-sized, but it captures the key logic of the full framework: invariants and non-negativity are enforced where the stoichiometric state actually lives, then the corrected state is collapsed to measured outputs.

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

The raw surrogate is flexible enough to capture curvature and interaction, but it is data-driven and unconstrained. There is no reason for $c_{raw}$ to satisfy the invariant relation $A c_{raw} = A c_{in}$ or the componentwise non-negativity requirement without an additional correction step. At this stage, however, $B$ should be read as a latent component-space parameterization rather than as an identified empirical object. Section 8 shows that measured composite data identify the affine measured-space core $M = G B$, not $B$ uniquely. That distinction is harmless for the affine core but becomes consequential once the positivity correction is executed in component space. The next section therefore introduces the constrained correction that separates learned variation from physically required structure.

## 6. Projection onto the Invariant-Consistent Non-Negative Set

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

This expression remains important in the present article for two reasons. First, it provides the simplest invariant-consistent reference solution. Second, it is the exact solution of the final non-negative formulation whenever the orthogonal affine projection already lies in the nonnegative orthant. It is therefore retained as a derivational benchmark rather than discarded.

### 6.3 Non-negative feasible set

The deployed non-negative ICSOR predictor is defined on the smaller feasible set

$$
\mathcal{S}_+(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in}, \; c \ge 0 \}
$$

where $c \ge 0$ is understood componentwise. This set intersects the invariant-consistent affine space with the nonnegative orthant. In geometric terms, the affine set removes changes that violate the stoichiometric invariants, while the orthant removes candidate states with negative ASM component concentrations.

### 6.4 Non-negative correction problem

Non-negative ICSOR corrects the raw surrogate by solving

$$
\min_{c \in \mathbb{R}^{F}} \; \frac{1}{2} \lVert c - c_{raw} \rVert_2^2
$$

subject to

$$
A c = A c_{in},
\qquad
c \ge 0
$$

The objective asks for the smallest Euclidean adjustment in ASM component concentration space that restores both the invariant relations and componentwise non-negativity. Because that metric is Euclidean in raw concentration coordinates, component scaling influences the magnitude and direction of the correction; a weighted metric would define a different model. Relative to the affine-only correction, the feasible set is smaller but still convex. The problem is therefore a strictly convex quadratic program with linear equality and inequality constraints.

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

The first question is whether the feasible set can be empty. This is where Assumption 9 matters. Under the present modeling assumptions, it is not. If the influent component state is non-negative, then $c = c_{in}$ is feasible because it satisfies

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

Thus, for every pair $(c_{raw}, c_{in})$ with $c_{in} \ge 0$, non-negative ICSOR produces one and only one corrected component-state prediction $c^*$. If an upstream reconstruction were to produce negative influent components, that guarantee would fail and feasibility would have to be checked separately.

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

For this feasible strictly convex quadratic program with affine constraints, any primal-dual point satisfying the KKT system is the unique minimizer $c^*$.

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

This result provides continuity with the earlier affine-only ICSOR theory. The non-negative formulation does not replace the orthogonal affine projector arbitrarily; it extends it to the cases where the affine projector would otherwise leave the physically admissible orthant.

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

The affine-centered formulation also leads to a reduced problem that is often easier to solve numerically. Let $N_A \in \mathbb{R}^{F \times (F-q)}$ have orthonormal columns spanning $\operatorname{null}(A)$. Every point in the affine set can then be written as

$$
c = c_{aff} + N_A z
$$

for some reduced coordinate vector $z \in \mathbb{R}^{F-q}$, because $A c_{aff} = A c_{in}$ and $A N_A = 0$. This parameterization therefore satisfies the equality constraint automatically. Substituting into the affine-centered objective gives

$$
\min_{z \in \mathbb{R}^{F-q}} \; \frac{1}{2} \lVert N_A z \rVert_2^2
$$

subject to

$$
N_A z \ge -c_{aff}
$$

Since $N_A$ has orthonormal columns,

$$
\lVert N_A z \rVert_2^2 = z^T N_A^T N_A z = z^T z
$$

and the reduced problem becomes

$$
\min_{z \in \mathbb{R}^{F-q}} \; \frac{1}{2} z^T z
$$

subject to

$$
N_A z \ge -c_{aff}
$$

If a non-orthonormal null-space basis is used instead, the reduced Hessian becomes $N_A^T N_A$. The feasible set is unchanged, but the orthonormal basis is preferable because it preserves the Euclidean metric and improves conditioning.

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

This is the deployed prediction reported by non-negative ICSOR.

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

When the affine projector is already non-negative, $\delta_+(u, c_{in}) = 0$. When one or more non-negativity constraints are active, $\delta_+(u, c_{in})$ is a sample-specific correction induced by the active inequality set. This decomposition is useful because it separates the globally affine, data-identifiable part of the model from the local correction needed to enforce physical admissibility. It also exposes a key boundary: once inequalities are active, $\delta_+(u, c_{in})$ depends on the chosen component-space raw prediction and therefore is not identified from measured composite data alone unless a latent representative has already been fixed.

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

At the estimation stage, three objects must be kept distinct: the latent component-space coefficient matrix $B$, the affine measured-space core, and the final deployed predictor $y^*$. Measured composite data identify an affine-core fit in measured space directly, and in the present formulation that fit is estimated by ridge regression to stabilize the high-dimensional second-order design. They do not identify $B$ uniquely without additional component-space structure, and a further representability step may be needed before that measured-space fit can be used in a component-space deployment map of the form $M = G B$.

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

This is the correct regression equation for ridge estimation from measured composite data when $A$, $I_{comp}$, $\lambda$, and $\Gamma$ are treated as known. It identifies the affine-core operator that would generate the deployed prediction whenever the non-negativity constraints are inactive.

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

### 8.3 Ridge estimator of the affine-core coefficients

The affine-core estimator used here is the multivariate ridge solution

$$
\widehat M_\lambda^T = \arg\min_{Q \in \mathbb{R}^{D \times K}} \left\{ \lVert \widetilde Y - \Phi Q \rVert_F^2 + \lambda \lVert \Gamma^{1/2} Q \rVert_F^2 \right\}
$$

where $\lambda \ge 0$ controls the amount of shrinkage and $\Gamma \succeq 0$ defines which coefficient directions are penalized. A common choice is

$$
\Gamma = \operatorname{diag}(0, 1, \ldots, 1),
$$

which leaves the intercept unpenalized and shrinks the remaining feature directions. Because ridge shrinkage is scale dependent, the non-intercept columns of $\Phi$ should either be centered and scaled before tuning $\lambda$ or the intended scaling should be encoded explicitly in $\Gamma$.

When the penalized normal matrix is invertible, the solution is

$$
\widehat M_\lambda^T = (\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T \widetilde Y,
$$

with the Moore-Penrose pseudoinverse used in place of the inverse if the penalized normal matrix is singular. The OLS estimator is recovered as the special case $\lambda = 0$ when the unpenalized normal equations are well posed.

This ridge stage is still a linear estimation step in measured space and is unchanged by the later introduction of non-negativity at deployment. The non-negative ICSOR formulation therefore does not replace coefficient estimation with a nonlinear estimator. It replaces the earlier OLS affine-core fit with a ridge-stabilized affine-core fit and applies the non-negative correction after that stage. If one intends to reconstruct a component-space surrogate, Section 8.5 shows that a representability step may still be required before deployment.

### 8.4 Ridge regularization, conditioning, and interpretability

Second-order feature maps can be high dimensional, and real wastewater datasets may not excite all directions of that design space. If $N < D$, full column rank is impossible from the outset; even when $N \ge D$, the design may still be rank deficient because some feature directions are weakly or redundantly excited. Ridge regularization is introduced precisely because those conditions make unpenalized estimation unstable or non-unique. With $\lambda > 0$ and a suitable penalty matrix $\Gamma$, the penalized affine-core fit remains well defined and numerically stable even when $\Phi$ is ill conditioned.

That numerical stability does not restore unique identification of the unpenalized coefficient vector. Instead, $\widehat M_\lambda$ is a penalty-dependent shrinkage estimate of the structurally identifiable affine core $M$. As $\lambda$ increases, coefficient magnitudes contract toward the null directions favored by $\Gamma$, and the fit trades variance for bias. Interpretation should therefore focus on fitted affine-core predictions, on coefficient patterns that remain stable across a reasonable range of penalties, or on grouped engineering effects rather than on one penalized coefficient in isolation. In practical terms, the modeler then has three defensible levers: reduce the feature basis, retune $\lambda$ and feature scaling, or shift the inferential emphasis from coefficients to predicted outputs and their uncertainty.

### 8.5 Reconstructing one admissible component-space coefficient matrix

If one seeks an exact component-space representative for a known affine-core matrix $M$ satisfying

$$
G B = M
$$

then the minimum-Frobenius-norm exact representative is

$$
B_{min} = G^+ M
$$

and the full solution set is

$$
B = G^+ M + (I_F - G^+ G) Z
$$

for arbitrary $Z \in \mathbb{R}^{F \times D}$. The free matrix $Z$ is the algebraic statement of non-identifiability.

After estimation, however, the available object is the penalized matrix $\widehat M_\lambda$ rather than an exact population matrix $M$. Define

$$
\widehat B_\lambda = G^+ \widehat M_\lambda
$$

and the induced representable affine-core operator

$$
\widehat M_{\lambda,G} = G \widehat B_\lambda = G G^+ \widehat M_\lambda.
$$

The matrix $\widehat M_{\lambda,G}$ is the orthogonal projection of $\widehat M_\lambda$ onto $\operatorname{range}(G)$, so it equals $\widehat M_\lambda$ only when $\widehat M_\lambda$ already lies in $\operatorname{range}(G)$. Therefore $\widehat B_\lambda$ should be interpreted as a chosen latent representative for deployment, not as an identified physical coefficient matrix, and $\widehat M_{\lambda,G}$ is the corresponding representable affine core. Any other representative obtained by adding $(I_F - G^+ G) Z$ leaves that representable affine core unchanged but can alter the component-space raw state used by the later inequality correction.

### 8.6 Final deployed predictor after estimation

Once a component-space representative $\widehat B_\lambda$ has been fixed, the estimated raw component prediction for a new sample is

$$
\widehat c_{raw,\lambda} = \widehat B_\lambda \phi(u, c_{in})
$$

The final deployed non-negative component prediction is then obtained by the same staged component-space logic introduced in Section 6: keep $\widehat c_{raw,\lambda}$ if it is already feasible, otherwise apply the affine projector, and solve the quadratic program only when the affine projector still violates non-negativity. In compact notation,

$$
\widehat c_\lambda^* = \operatorname{Proj}_{\mathcal{S}_+(c_{in})}(\widehat c_{raw,\lambda})
$$

and the final deployed measured prediction is

$$
\widehat y_\lambda^* = I_{comp} \widehat c_\lambda^*.
$$

This last map is deterministic conditional on the chosen representative $\widehat B_\lambda$, but it is not globally affine in $\phi(u, c_{in})$. If the non-negativity constraints are inactive, then the deployed prediction collapses to the representable affine core

$$
\widehat y_{\lambda,aff,G} = \widehat M_{\lambda,G} \phi(u, c_{in}) + H c_{in},
$$

which equals the unconstrained ridge affine-core fit only when $\widehat M_\lambda$ already lies in $\operatorname{range}(G)$ or representability was enforced during estimation. If one or more inequalities activate, different admissible representatives can generate different $\widehat c_{raw,\lambda}$ and therefore different projected outputs. The ridge coefficients therefore characterize the penalized measured-space affine-core fit, while the final deployed predictor becomes a fully specified ICSOR model only after one of two additional steps is taken: select a representative such as $\widehat B_\lambda = G^+ \widehat M_\lambda$, together with its compatible affine core $\widehat M_{\lambda,G} = G G^+ \widehat M_\lambda$, or impose extra component-space structure or data that identify $B$ more tightly.

## 9. Statistical Inference and Predictive Uncertainty

### 9.1 Error model and conditioning on the ridge penalty

For statistical inference, suppose the row errors of $E$ are independent across samples and satisfy

$$
\mathbb{E}[E \mid \Phi] = 0
$$

and

$$
\operatorname{Var}(\operatorname{vec}(E) \mid \Phi) = \Omega \otimes I_N
$$

where $\Omega \in \mathbb{R}^{K \times K}$ is the within-sample covariance across measured outputs. This allows, for example, COD and total nitrogen errors to be correlated within the same sample.

Throughout this section, the ridge penalty parameter $\lambda$ and penalty matrix $\Gamma$ are treated as fixed. If $\lambda$ is selected by cross-validation, generalized cross-validation, or another data-adaptive rule, the formulas below are conditional on the selected penalty and therefore omit tuning uncertainty unless that selection step is repeated inside resampling.

Define

$$
A_\lambda = (\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T
$$

so that

$$
\widehat M_\lambda^T = A_\lambda \widetilde Y.
$$

Also define the ridge fitted-value matrix

$$
W_\lambda = \Phi A_\lambda = \Phi (\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T
$$

and the fitted residual matrix

$$
\widehat E_\lambda = \widetilde Y - \Phi \widehat M_\lambda^T = (I_N - W_\lambda) \widetilde Y.
$$

A common plug-in estimator of the within-sample output covariance is

$$
\widehat \Omega_\lambda = \frac{1}{N - \operatorname{df}_\lambda} \widehat E_\lambda^T \widehat E_\lambda,
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
\mathbb{E}[\widehat M_\lambda^T \mid \Phi] = S_\lambda M^T
$$

and the conditional bias is

$$
\operatorname{Bias}(\widehat M_\lambda^T \mid \Phi) = (S_\lambda - I_D) M^T = -\lambda (\Phi^T \Phi + \lambda \Gamma)^{-1} \Gamma M^T.
$$

The conditional covariance is

$$
\operatorname{Var}(\operatorname{vec}(\widehat M_\lambda^T) \mid \Phi) = \Omega \otimes (A_\lambda A_\lambda^T),
$$

or, equivalently,

$$
\operatorname{Var}(\operatorname{vec}(\widehat M_\lambda^T) \mid \Phi) = \Omega \otimes \left[(\Phi^T \Phi + \lambda \Gamma)^{-1} \Phi^T \Phi (\Phi^T \Phi + \lambda \Gamma)^{-1}\right]
$$

when $\Gamma$ is symmetric. Therefore, for the coefficient $\widehat M_{\lambda,kj}$,

$$
\operatorname{Var}(\widehat M_{\lambda,kj} \mid \Phi) = \Omega_{kk} [A_\lambda A_\lambda^T]_{jj}.
$$

Unlike OLS, the conditional covariance does not fully characterize ridge coefficient uncertainty because the estimator is biased. A more honest coefficientwise summary is the conditional mean-squared error

$$
\operatorname{MSE}(\widehat M_{\lambda,kj} \mid \Phi) = \Omega_{kk} [A_\lambda A_\lambda^T]_{jj} + \left[\operatorname{Bias}(\widehat M_{\lambda,kj} \mid \Phi)\right]^2.
$$

If $\lambda \to 0$ and $\Phi$ has full column rank, these expressions reduce to the familiar OLS covariance formulas.

### 9.3 Ridge affine-core mean-prediction uncertainty

For a new operating point with feature vector $\phi_* = \phi(u_*, c_{in,*})$, the fitted affine-core mean output is

$$
\widehat y_{\lambda,aff,*} = \widehat M_\lambda \phi_* + H c_{in,*}.
$$

Define the ridge leverage analogue

$$
s_{\lambda,*} = \phi_*^T A_\lambda A_\lambda^T \phi_*
$$

and the conditional bias vector

$$
b_{\lambda,*} = M (S_\lambda^T - I_D) \phi_*.
$$

Then

$$
\operatorname{Var}(\widehat y_{\lambda,aff,*} \mid \phi_*, c_{in,*}, \Phi) = s_{\lambda,*} \Omega
$$

and the conditional mean-squared-error matrix of the fitted affine-core mean is

$$
\operatorname{MSE}(\widehat y_{\lambda,aff,*} \mid \phi_*, c_{in,*}, \Phi) = s_{\lambda,*} \Omega + b_{\lambda,*} b_{\lambda,*}^T.
$$

For a future affine-core observation at the same operating point, the corresponding prediction mean-squared-error matrix is

$$
\operatorname{MSE}_{pred}(\phi_*) = (1 + s_{\lambda,*}) \Omega + b_{\lambda,*} b_{\lambda,*}^T.
$$

The variance term plays the role of a ridge analogue of the OLS leverage formula, but the bias term has no OLS counterpart. Because $b_{\lambda,*}$ depends on the unknown population matrix $M$, fully closed-form finite-sample intervals require either a plug-in bias estimate, an asymptotic approximation, or resampling. There is therefore no exact Student-$t$ analogue for the ridge affine core.

### 9.4 Why these formulas do not globally extend to the final non-negative predictor

The final deployed prediction is

$$
\widehat y_{\lambda,*}^* = I_{comp} \, \operatorname{Proj}_{\mathcal{S}_+(c_{in,*})}(\widehat c_{raw,\lambda,*})
$$

with

$$
\widehat c_{raw,\lambda,*} = \widehat B_\lambda \phi_*.
$$

This map is generally piecewise affine rather than globally affine because the active set of non-negativity constraints can change from sample to sample. At points where the active set changes, the deployed mapping is not described by one global coefficient matrix. Even away from those transitions, the upstream affine-core fit is ridge biased and conditional on the chosen penalty and latent representative. Consequently, the bias and covariance formulas above are informative local descriptors, not global exact finite-sample interval formulas, for the final deployed predictor.

Two special cases are simpler.

1. If the non-negativity constraints are inactive at the prediction point, then $\widehat y_{\lambda,*}^* = \widehat y_{\lambda,aff,*}$ and the ridge affine-core bias and covariance expressions from Section 9.3 apply.
2. If the active set is locally stable, the final predictor is locally affine in the raw component state and delta-method or local linearization arguments can sometimes be built around that fixed active set.

Neither of those special cases yields a global exact interval formula for the final non-negative predictor, and neither removes tuning uncertainty when $\lambda$ is selected from the data.

### 9.5 Recommended uncertainty treatment for the final predictor

For the final deployed non-negative predictor, the most defensible default approach is resampling-based uncertainty quantification that repeats the full estimation and deployment pipeline. A bootstrap or residual-bootstrap replicate should

1. resample the data or fitted residuals,
2. rebuild $\Phi$ and $\widetilde Y$ for that replicate,
3. reselect $\lambda$ if the penalty was tuned from the data,
4. refit $\widehat M_\lambda$,
5. reconstruct the chosen component-space representative $\widehat B_\lambda$, and
6. rerun the affine and, when needed, quadratic-program projection before recording the resulting prediction.

This procedure propagates the two main uncertainty mechanisms that matter after the ridge revision: shrinkage-dependent affine-core estimation and sample-specific convex correction. In applications where only a fast approximation is needed, a conditional-on-$\lambda$ Gaussian approximation based on the Section 9.3 covariance formulas may still be reported for the affine core, preferably with an explicit note that it ignores penalty-selection uncertainty and is not an exact interval for the final constrained predictor.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct effluent-state parameterization

The surrogate is parameterized on the effluent component state rather than on the net change. This keeps the learned target aligned with the quantity ultimately used for reporting and decision support. The cost is that the role of the influent state enters twice: once inside the feature map and once inside the constrained correction. That is not redundancy. The first role captures empirical dependence; the second role enforces the physically required part of the state.

### 10.2 Partitioned second-order feature structure

The partitioned feature map separates operating effects, influent-composition effects, and operation-loading interactions in a way that is interpretable to process engineers. The price of that interpretability is rapid feature growth, which can create multicollinearity, unstable unpenalized coefficients, and weakly identified directions if the dataset does not adequately excite the design space. Since $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$, even moderate values of $M_{op}$ and $F$ can produce a feature basis that is large relative to the sample count. Ridge regression is therefore not a cosmetic estimation choice here. It is the device that stabilizes the affine-core fit in the presence of that collinearity. The tradeoff is that the reported coefficients now depend on feature scaling and on the chosen penalty pair $(\lambda, \Gamma)$, so raw coefficient magnitudes should not be interpreted independently of the regularization convention.

### 10.3 Euclidean convex projection metric

The Euclidean correction gives a unique convex projection and keeps the constrained step mathematically clean. At the same time, it is sensitive to the scaling of component coordinates. If one component is numerically much larger or measured in a much different effective scale than another, the Euclidean metric may assign disproportionate influence to that direction. A weighted projection may be preferable in some applications, but that would be a different model from the one defined here.

More explicitly, replacing $\lVert c - c_{raw} \rVert_2^2$ by $(c - c_{raw})^T W (c - c_{raw})$ with positive-definite $W$ would change both the geometry of the correction and the deployed predictor. The present article therefore treats metric choice as part of the model definition rather than as a numerical afterthought.

### 10.4 Projection before measurement collapse

Enforcing the invariant relations and non-negativity before collapsing to measured space is a substantive modeling decision, not a notational convenience. It preserves the constraints in the space where the stoichiometric matrix is actually defined and where the non-negativity claim is physically meaningful. Once the state is collapsed into measured composites, some physically meaningful directions may no longer be separately visible.

### 10.5 Affine-core coefficients versus the final deployed predictor

The affine-core coefficients $M$ remain the structurally identifiable objects for direct engineering interpretation because they act directly on observed outputs. The reported estimates $\widehat M_\lambda$ are ridge-shrunken versions of that identifiable operator, so interpretation of fitted coefficients should always be conditioned on the chosen penalty and feature scaling. The latent component-space coefficients $B$ remain generally non-unique unless extra structure is imposed. The final deployed predictor $y^*$ adds one more layer: even when $M$ is well estimated, the final sample-specific output depends on which non-negativity constraints are active and, when those constraints do activate, on the chosen latent representative used to form $c_{raw}$. That means coefficient interpretation is clearest for the affine core after stating the ridge convention, whereas the final prediction should be read as penalized affine signal plus a representative-dependent convex feasibility correction.

## 11. Limitations

Non-negative ICSOR is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state in the quasi-steady-sample sense and does not represent temporal dynamics or path dependence.
2. It enforces only the invariant relations encoded by the chosen stoichiometric basis and system boundary together with componentwise non-negativity.
3. Non-negative component concentrations do not guarantee full kinetic, biological, or thermodynamic feasibility.
4. Non-negative component predictions imply non-negative measured composites only when the adopted composition matrix has the appropriate sign structure.
5. The correction depends on the chosen metric and is therefore sensitive to component scaling.
6. The final deployed predictor is not globally representable by one affine measured-space coefficient matrix once non-negativity constraints become active.
7. When positivity constraints activate, the final deployed predictor is not identified from measured composite data alone unless a component-space representative is chosen or extra component-space information is supplied.
8. For the ridge affine core, closed-form conditional bias and covariance expressions are available only when the penalty is treated as fixed, but exact finite-sample $t$-based prediction intervals are not available in general, and the final non-negative deployed predictor still requires resampling-based uncertainty quantification.
9. The second-order feature basis can be statistically fragile if it is weakly excited or highly collinear; ridge regularization mitigates that fragility but does not remove dependence on feature scaling or penalty choice.
10. A misspecified stoichiometric matrix or incorrect system boundary leads to a formally correct projection onto the wrong physical constraint set.
11. If the influent ASM component state is reconstructed from measured aggregate variables rather than observed directly, reconstruction error enters upstream of the regression and is not represented by the affine-core uncertainty formulas derived here.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its claims correctly.

## 12. Conclusion

Non-negative ICSOR combines a partitioned second-order surrogate with a convex projection derived from stoichiometric invariants and componentwise non-negativity. The framework is useful for wastewater applications because it preserves the distinction between operating conditions and influent composition, enforces conservation structure where that structure naturally lives, removes negative deployed component predictions, and returns predictions in the measured variables used by plant operators and simulation studies. In deployment, the correction should be evaluated hierarchically so that the quadratic program is reserved for the subset of samples not already repaired by the closed-form affine projector.

The central theoretical point remains that measured composite data identify the affine measured-space operator $M$, not the latent component-space coefficient matrix $B$ uniquely. The non-negative extension does not alter that identifiability fact. Instead, it changes the deployment map: after estimating the affine core by ridge regression, the final prediction is obtained by projecting a chosen raw component-space state onto the invariant-consistent non-negative set. When inequalities are inactive, that deployed prediction collapses exactly to the identifiable affine core up to the chosen representability convention. When inequalities are active, deployment additionally requires a chosen component-space representative or extra component-space information, because measured composite data alone do not identify the corrected component-space map uniquely. The corresponding uncertainty analysis is likewise conditional and shrinkage aware at the affine-core stage, with resampling preferred for the final constrained predictor, especially when $\lambda$ is tuned from the data. Under that reading, non-negative ICSOR is best understood as an analytically structured steady-state surrogate for activated-sludge prediction: more physically disciplined than a generic black-box regressor, more realistic than affine-only invariant correction when negative component states would otherwise occur, but still narrower in scope than a full dynamic mechanistic simulator.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
4. Seber, G. A. F., and Lee, A. J. Linear Regression Analysis. 2nd ed. Wiley, 2003.
5. Rao, C. R., and Mitra, S. Generalized Inverse of Matrices and Its Applications. Wiley, 1971.
6. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.
7. Hoerl, A. E., and Kennard, R. W. Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics 12(1), 55-67, 1970.
8. Hastie, T., Tibshirani, R., and Friedman, J. The Elements of Statistical Learning. 2nd ed. Springer, 2009.
