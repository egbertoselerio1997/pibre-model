# Non-Negative Invariant-Constrained Second-Order Regression (ICSOR) for Activated Sludge Surrogate Modeling

## Abstract

This article presents a non-negative formulation of invariant-constrained second-order regression (ICSOR), a physics-informed surrogate model for steady-state activated-sludge systems. The purpose of ICSOR is to predict measured effluent variables from operating conditions and influent activated-sludge-model (ASM) component concentrations while preserving the stoichiometric invariants implied by the adopted reaction network and enforcing non-negativity of the predicted effluent ASM component state. The key difficulty is that conservation laws are defined in ASM component space, whereas plant observations are usually reported as composite variables such as total COD, total nitrogen, total phosphorus, or suspended solids. A regression model built only in measured-output space can fit those aggregates while still implying an impossible redistribution of the underlying ASM components. An affine invariant projection resolves only part of that mismatch because it restores stoichiometric consistency but can still produce negative component predictions. The non-negative ICSOR formulation therefore uses a staged component-space deployment rule. First, a partitioned second-order surrogate produces an unconstrained prediction of the effluent ASM component state. Second, the raw state is repaired to the invariant-consistent affine reference. Third, a linear-program selector searches the invariant-consistent non-negative set for the state whose measured composites remain as close as possible to the affine measured reference. A secondary linear tie-break then chooses, among those primary-optimal states, the one that stays closest to the affine component state, with a fixed lexicographic component ordering resolving any remaining ties. The selected component state is then collapsed into measured output space through a linear composition map.

The framework is written as a self-contained theory section for readers in chemical engineering, wastewater process modeling, and machine learning. All symbols are defined before use. The invariant constraint is derived from the stoichiometric change relation rather than asserted heuristically. The non-negative correction is formulated as a lexicographic sequence of linear programs, and the role of the earlier orthogonal affine projector is retained explicitly as the equality-only reference solution and zero-correction special case. In deployment, the LP stage is needed only when the raw prediction is infeasible and the closed-form affine projector still violates componentwise non-negativity. The distinction between identifiable affine measured-space coefficients and non-identifiable latent component-space coefficients is preserved. Accordingly, the affine core is identifiable from measured composite data, whereas the final inequality-constrained deployed predictor is fully specified only after a latent component-space representative, a composite-space weighting convention, and a deterministic tie-break rule have been fixed. The limits of exact closed-form uncertainty analysis for that LP-selected deployed predictor are stated explicitly. The result is a precise formulation of what non-negative ICSOR guarantees, what remains estimated by least squares, and what must instead be handled through linear-program post-estimation correction.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to represent a quasi-steady operating condition: the operating variables, influent composition, and effluent response are treated as effectively time-invariant over the control volume being modeled for the sampling window of interest. The usual difficulty is that a generic data-driven regressor can fit observed effluent data while still violating fundamental conservation structure. In activated-sludge modeling, that failure is not a minor technical detail. It undermines the physical credibility of the surrogate because it can imply component inventories that are inconsistent with the adopted reaction network even when the measured aggregates appear plausible.

The source of the problem is a mismatch between two spaces.

1. The mechanistic stoichiometric model is written in an ASM component basis, such as soluble substrate, ammonium, nitrate, autotrophic biomass, particulate organics, phosphate, dissolved oxygen, and alkalinity.
2. The plant or simulator often reports outputs in measured composite variables, such as total COD, total nitrogen, total phosphorus, TSS, or VSS.

These two spaces are related, but they are not the same. Conservation laws are naturally expressed in the ASM component basis because the stoichiometric matrix acts on individual components. Observations, however, are usually available only after those components have been aggregated into measurable composites. A surrogate that learns only in measured-output space may reproduce the observed aggregates while obscuring physically impossible changes in the underlying component inventory.

The earlier affine-only ICSOR formulation addresses that mismatch by predicting in component space and projecting the raw surrogate output onto the affine set consistent with the stoichiometric invariants. That construction repairs invariant violations, but it does not ensure that the projected component concentrations are non-negative. Negative component predictions are a serious defect in the present setting because component concentrations are themselves physical quantities and because negative components can propagate into implausible reported composites. The non-negative ICSOR formulation developed here therefore strengthens the correction step. The affine invariant projection remains part of the derivation, but the deployed predictor is now defined by a linear-program selector that first minimizes measured-composite deviation from the affine reference and then applies a deterministic component-space tie-break inside that primary-optimal set.

The model is constructed to answer one precise question:

> Given a steady-state influent state and a steady-state operating condition, what measured effluent state should be predicted if the underlying effluent ASM component state must satisfy the conserved quantities implied by the adopted stoichiometric model and must remain non-negative componentwise?

The theory in this article is restricted to steady-state reactor-block prediction. It does not aim to replace a dynamic activated-sludge simulator. Rather, it provides an analytically structured surrogate that preserves stoichiometric structure, enforces non-negativity at the deployed component-state level, and remains simple enough that its affine core can still be estimated directly from data by least squares. That last point requires care: measured composite data identify the affine core, whereas any component-space deployment correction requires either a chosen latent representative or additional component-space information. In deployment, the correction is evaluated in stages: no correction when the raw component state is already feasible, closed-form affine projection when only the invariant equalities are violated, and a lexicographic LP correction only when non-negativity remains violated after the affine step. The discussion proceeds from physical scope and notation, to derivation of the invariant relations, to LP-based non-negative selection, to collapse into measured space, and finally to estimation and uncertainty.

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
| $c_{aff}$ | $\mathbb{R}^{F}$ | Affine reference state obtained by orthogonal projection of $c_{raw}$ onto the invariant-consistent set |
| $c^*$ | $\mathbb{R}^{F}$ | Final ASM component state selected by the lexicographic LP deployment rule |
| $y$ | $\mathbb{R}^{K}$ | Measured effluent composite vector |
| $y_{aff}$ | $\mathbb{R}^{K}$ | Affine measured reference, $I_{comp} c_{aff}$ |
| $y^*$ | $\mathbb{R}^{K}$ | Final measured output induced by $c^*$ |
| $s$ | $\mathbb{R}^{K}$ | Non-negative auxiliary vector bounding absolute measured-composite deviation from $y_{aff}$ in the primary LP |
| $r$ | $\mathbb{R}^{F}$ | Non-negative auxiliary vector bounding absolute component-state deviation from $c_{aff}$ in the secondary LP |
| $w_y$ | $\mathbb{R}_{++}^{K}$ | Positive weights on primary measured-composite absolute deviations |
| $w_c$ | $\mathbb{R}_{++}^{F}$ | Positive weights on secondary component-state absolute deviations |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM component concentrations to measured composite variables |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction progress vector expressed in concentration-equivalent units so that $\nu^T \xi$ has the same units as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank matrix whose transposed rows form a basis of $\operatorname{null}(\nu)$, equivalently $A \nu^T = 0$ |
| $P_{inv}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the row space of $A$ |
| $P_{adm}$ | $\mathbb{R}^{F \times F}$ | Orthogonal projector onto the admissible change space, $I_F - P_{inv}$ |
| $N_A$ | $\mathbb{R}^{F \times (F-q)}$ | Matrix whose columns form an orthonormal basis of $\operatorname{null}(A)$ |
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

This linear composition map is standard in activated-sludge modeling when measured variables are aggregates of ASM components. For example, total COD or total nitrogen is formed by summing the relevant component concentrations with appropriate conversion factors. In the common case of total COD, total nitrogen, total phosphorus, TSS, or VSS built as sums with non-negative conversion factors, the rows of $I_{comp}$ are entrywise non-negative. In the non-negative ICSOR setting, that sign structure matters because non-negative component predictions imply non-negative measured composites only under that condition. Throughout the chapter, $I_{comp}$ is treated as known and fixed; if its coefficients are themselves estimated, that uncertainty lies outside the present error model.

## 3. Modeling Assumptions

The framework rests on the following assumptions. These are not optional preferences left to the reader. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a quasi-steady input-output condition, typically a stable operating epoch or time-averaged window. The model is not a dynamic state estimator.
2. **Fixed component basis.** The ASM component basis and the associated stoichiometric matrix are fixed before regression begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement. Any external source or sink outside that boundary is outside the present model.
4. **Linear composition map.** Measured effluent variables are linear combinations of the underlying ASM component concentrations through a known fixed matrix $I_{comp}$.
5. **Direct effluent-state parameterization.** The surrogate is parameterized to predict the effluent ASM component state $c_{out}$ directly rather than the component change $c_{out} - c_{in}$.
6. **Second-order surrogate class.** The raw surrogate is a partitioned second-order polynomial model that includes linear, quadratic, and operation-loading interaction terms.
7. **Primary composite-space LP objective.** After the affine reference state is formed, the deployed correction first minimizes a weighted $L_1$ deviation between the deployed measured composites and the affine measured reference $y_{aff}$.
8. **Deterministic tie-break convention.** Among all primary-optimal states, the model applies a secondary weighted $L_1$ component-space tie-break toward $c_{aff}$, with a fixed lexicographic component ordering used only if the secondary LP is still tied.
9. **Constraint scope.** The final LP selector enforces the stoichiometric invariants implied by the chosen basis and system boundary together with componentwise non-negativity. It does not enforce upper bounds, kinetic feasibility, or thermodynamic admissibility beyond those conditions.
10. **Influent feasibility.** The influent reference state is assumed non-negative in component space. Under that assumption the non-negative feasible set is non-empty because the influent state itself satisfies the invariant equalities.
11. **Composite-sign scope.** Non-negative component predictions imply non-negative measured composites only when the relevant rows of $I_{comp}$ are entrywise non-negative. If the measurement convention uses negative coefficients, extra output-space sign constraints would be required for a composite non-negativity guarantee.
12. **Statistical scope.** Classical least-squares coefficient and affine-core prediction intervals are interpreted under independent-sample Gaussian multivariate linear-model assumptions. Exact finite-sample $t$-based intervals require a full-rank affine-core design and do not generally extend through the LP-selected deployed predictor.

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

The reaction may redistribute material between the two components, but it cannot change the conserved total represented by $c_1 + c_2$. In the non-negative ICSOR setting, the feasible effluent set is therefore the non-negative line segment satisfying this equality. The later correction step will select the point on that segment whose measured composites stay as close as possible to the affine reference and, if several states are tied in measured space, the one favored by the documented component-space tie-break.

### 4.5 ASM-flavored miniature example

A slightly more ASM-flavored toy example shows why the component-space correction must happen before measurement collapse. Suppose the component vector is

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

which is exactly the same measured output produced by the feasible state $[0, 20, 5]^T$. A measured-space-only correction cannot distinguish those two component states, because both collapse to the same aggregates. The component-space LP selector, however, still detects that $c_{raw}$ violates $c \ge 0$. In this example the invariant equalities already hold, so $c_{aff} = c_{raw}$ and $y_{aff} = I_{comp} c_{raw} = [20, 5]^T$. Every feasible state with those same composites is primary-optimal because it attains zero composite deviation, and the secondary component-space tie-break then selects $[0, 20, 5]^T$ because it is the non-negative invariant-consistent state closest to $c_{aff}$ in weighted $L_1$ deviation. The example is still toy-sized, but it captures the key logic of the full framework: invariants and non-negativity are enforced where the stoichiometric state actually lives, measured-composite closeness is judged relative to the affine reference, and only then is the corrected state collapsed to measured outputs.

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

Retaining the full vectorized quadratic blocks avoids hidden indexing conventions and makes the later estimation problem unambiguous. This second-order polynomial basis is a modeling choice rather than a theorem. It is used here because it retains linear terms, self-quadratic curvature, and operation-loading interactions in one explicit design matrix. The cost is rapid growth of $D$; when the sample count is not large relative to $D$, the later coefficient interpretation becomes fragile, as discussed in Section 8.4.

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

This expression remains important in the present article for three reasons. First, it provides the simplest invariant-consistent reference solution. Second, it defines the affine measured reference $y_{aff} = I_{comp} c_{aff}$ that anchors the later LP objective. Third, it is the exact output of the full LP deployment rule whenever the orthogonal affine projection already lies in the nonnegative orthant. It is therefore retained as a derivational benchmark rather than discarded.

### 6.3 Non-negative feasible set

The deployed non-negative ICSOR predictor is defined on the smaller feasible set

$$
\mathcal{S}_+(c_{in}) = \{ c \in \mathbb{R}^{F} : A c = A c_{in}, \; c \ge 0 \}
$$

where $c \ge 0$ is understood componentwise. This set intersects the invariant-consistent affine space with the nonnegative orthant. In geometric terms, the affine set removes changes that violate the stoichiometric invariants, while the orthant removes candidate states with negative ASM component concentrations.

### 6.4 Non-negative correction as a lexicographic linear program

After the affine reference state has been formed, the non-negative deployment step is defined by a primary LP in which optimality is judged in measured composite space rather than by Euclidean distance in component space. Let

$$
y_{aff} = I_{comp} c_{aff}
$$

and choose positive weight vectors

$$
w_y \in \mathbb{R}_{++}^{K},
\qquad
w_c \in \mathbb{R}_{++}^{F}.
$$

Introduce an auxiliary vector $s \in \mathbb{R}^{K}$ that bounds the componentwise absolute composite deviation from the affine measured reference. The primary LP is

$$
\min_{c \in \mathbb{R}^{F},\, s \in \mathbb{R}^{K}} \; w_y^T s
$$

subject to

$$
A c = A c_{in},
\qquad
c \ge 0,
$$

$$
-s \le I_{comp} c - y_{aff} \le s,
\qquad
s \ge 0.
$$

Because $s_k$ bounds $\lvert (I_{comp} c - y_{aff})_k \rvert$, the objective is the weighted $L_1$ distance between the deployed measured composites and the affine measured reference. The feasible set is polyhedral and the objective is linear, so the primary correction problem is a linear program.

The primary LP can have multiple optima when several non-negative component states produce the same best composite deviation. To turn the deployed correction into a function, define $\eta^*$ as the optimal value of the primary LP and introduce a second auxiliary vector $r \in \mathbb{R}^{F}$ that bounds the componentwise absolute deviation from the affine component reference. The secondary LP is

$$
\min_{c \in \mathbb{R}^{F},\, s \in \mathbb{R}^{K},\, r \in \mathbb{R}^{F}} \; w_c^T r
$$

subject to

$$
A c = A c_{in},
\qquad
c \ge 0,
$$

$$
-s \le I_{comp} c - y_{aff} \le s,
\qquad
s \ge 0,
\qquad
w_y^T s = \eta^*,
$$

$$
-r \le c - c_{aff} \le r,
\qquad
r \ge 0.
$$

This second stage is only a tie-break. It is not allowed to sacrifice the primary composite-space objective. It merely chooses, among the primary-optimal component states, the one with the smallest weighted $L_1$ deviation from $c_{aff}$. If the secondary LP is still tied, the deployed state is defined to be the lexicographically smallest component vector under the fixed ASM component ordering. That final clause can itself be implemented by a finite sequence of LPs, so the full deployment map remains completely linear-program based.

We denote the output of this full lexicographic LP procedure by

$$
c^* = \operatorname{LexLP}_{\mathcal{S}_+}(c_{raw}, c_{in}).
$$

The role of $c_{raw}$ is indirect but essential: it determines $c_{aff}$, and $c_{aff}$ in turn determines both the primary measured reference $y_{aff}$ and the secondary component-space tie-break.

### 6.5 Feasibility, existence, and deterministic selection

The first question is whether the feasible set can be empty. This is where Assumption 10 matters. Under the present modeling assumptions, it is not. If the influent component state is non-negative, then $c = c_{in}$ is feasible because it satisfies

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

Because $\mathcal{S}_+(c_{in})$ is a non-empty polyhedron and the primary objective $w_y^T s$ is bounded below by zero, the primary LP attains at least one optimum. Its optimal face is therefore non-empty. The secondary LP is solved over that non-empty primary-optimal face, and its objective $w_c^T r$ is also bounded below by zero, so it too attains at least one optimum. If that second stage is still tied, each step of the final lexicographic component-ordering convention is another LP over a non-empty optimal face and therefore attains a solution.

Thus, for every pair $(c_{raw}, c_{in})$ with $c_{in} \ge 0$, the full documented LP procedure produces at least one admissible deployed component state, and the fixed lexicographic tie-break makes the deployed map single-valued by definition. Uniqueness is therefore a property of the full selection rule, not of the primary LP in isolation. If an upstream reconstruction were to produce negative influent components, that guarantee would fail and feasibility would have to be checked separately.

### 6.6 Primal-dual characterization of the primary LP

Introduce multipliers $\lambda \in \mathbb{R}^{q}$ for the equality constraints, $\alpha^+, \alpha^- \in \mathbb{R}_{\ge 0}^{K}$ for the upper and lower absolute-deviation bounds, $\mu \in \mathbb{R}_{\ge 0}^{F}$ for componentwise non-negativity, and $\beta \in \mathbb{R}_{\ge 0}^{K}$ for $s \ge 0$. The Lagrangian of the primary LP is

$$
\mathcal{L}(c, s, \lambda, \alpha^+, \alpha^-, \mu, \beta)
= w_y^T s
+ \lambda^T(A c - A c_{in})
+ (\alpha^+)^T(I_{comp} c - y_{aff} - s)
+ (\alpha^-)^T(-I_{comp} c + y_{aff} - s)
- \mu^T c
- \beta^T s.
$$

The corresponding primal-dual optimality conditions are

$$
A^T \lambda + I_{comp}^T(\alpha^+ - \alpha^-) - \mu = 0,
$$

$$
w_y - \alpha^+ - \alpha^- - \beta = 0,
$$

$$
A c = A c_{in},
\qquad
c \ge 0,
$$

$$
-s \le I_{comp} c - y_{aff} \le s,
\qquad
s \ge 0,
$$

$$
\alpha^+, \alpha^-, \mu, \beta \ge 0,
$$

together with complementary slackness for every active inequality:

$$
\alpha_k^+ \big((I_{comp} c - y_{aff})_k - s_k\big) = 0,
$$

$$
\alpha_k^- \big(-(I_{comp} c - y_{aff})_k - s_k\big) = 0,
$$

$$
\mu_f c_f = 0,
\qquad
\beta_k s_k = 0.
$$

The first stationarity relation shows that the selected component state balances three geometric objects: the invariant normals in the row space of $A$, the measured-space discrepancy normals pulled back through $I_{comp}^T$, and the active faces of the non-negative orthant. The second stationarity relation allocates each output weight $w_{y,k}$ across the two signed composite-deviation faces and any inactive slack. The secondary LP has an analogous primal-dual structure, but with the additional constraints that lock the primary optimum and bound absolute deviation from $c_{aff}$.

### 6.7 Relation to the orthogonal affine projector

The orthogonal affine projection remains embedded in the new formulation as a special case. If the affine projector already satisfies non-negativity, then the lexicographic LP rule returns exactly the same point.

Indeed, suppose

$$
c_{aff} \ge 0
$$

componentwise. Then $c_{aff} \in \mathcal{S}_+(c_{in})$. At that point,

$$
I_{comp} c_{aff} - y_{aff} = 0,
$$

so the primary LP can choose $s = 0$ and attain objective value zero. Because $s \ge 0$ and $w_y > 0$, no feasible point can do better than zero, hence the primary optimum is $\eta^* = 0$. The same point also satisfies

$$
c_{aff} - c_{aff} = 0,
$$

so the secondary LP can choose $r = 0$ and attain objective value zero. Because $r \ge 0$ and $w_c > 0$, no primary-optimal point can improve on that value. Therefore the full lexicographic LP selector returns

$$
c^* = c_{aff}
\qquad \text{whenever} \qquad
c_{aff} \ge 0.
$$

This result preserves continuity with the earlier affine-only ICSOR theory. The non-negative formulation does not discard the affine projector; it uses the affine projector as the equality-consistent reference and returns it exactly whenever no further correction is needed.

### 6.8 Efficient deployment sequence

The deployed correction should still be evaluated in a staged order so that the LP machinery is activated only when it is actually needed.

1. Form the raw component prediction $c_{raw}$.
2. If $c_{raw}$ already satisfies $A c_{raw} = A c_{in}$ and $c_{raw} \ge 0$, return $c_{raw}$ directly. In that case $c_{aff} = c_{raw}$, $y_{aff} = I_{comp} c_{raw}$, and both LP objectives are already zero.
3. Otherwise compute the closed-form affine projection $c_{aff} = P_{adm} c_{raw} + P_{inv} c_{in}$ and the affine measured reference $y_{aff} = I_{comp} c_{aff}$.
4. If $c_{aff} \ge 0$, return $c_{aff}$. Again the primary and secondary LP objectives are zero at that point.
5. Otherwise solve the primary composite-space LP.
6. Solve the secondary component-space tie-break LP over the primary-optimal set.
7. If the second stage is still tied, apply the fixed lexicographic component-ordering convention via additional LPs.

This order follows directly from the construction above. The first check avoids unnecessary correction when the raw surrogate already lies in the feasible set. The second check uses the orthogonal affine projector as the cheapest exact repair of invariant violations. The LP stage is therefore a residual correction step for the subset of samples in which the affine projector still leaves the non-negative orthant.

One degenerate case is worth noting. If $A$ has zero rows, then there is no non-trivial invariant equality to enforce and $c_{aff} = c_{raw}$. The correction problem does not collapse generically to componentwise clipping. Instead, it becomes a non-negativity LP that keeps the measured composites as close as possible to $I_{comp} c_{raw}$, followed by the same component-space tie-break and lexicographic selection. Componentwise clipping is recovered only for special composition maps and should not be treated as the generic no-invariant case.

### 6.9 Reduced null-space LP formulation

The affine reference state also yields a reduced formulation that removes the equality constraints exactly. Let $N_A \in \mathbb{R}^{F \times (F-q)}$ have orthonormal columns spanning $\operatorname{null}(A)$. Every point in the affine set can then be written as

$$
c = c_{aff} + N_A z
$$

for some reduced coordinate vector $z \in \mathbb{R}^{F-q}$, because $A c_{aff} = A c_{in}$ and $A N_A = 0$.

Substituting this parameterization into the primary LP gives

$$
\min_{z \in \mathbb{R}^{F-q},\, s \in \mathbb{R}^{K}} \; w_y^T s
$$

subject to

$$
N_A z \ge -c_{aff},
$$

$$
-s \le I_{comp} N_A z \le s,
\qquad
s \ge 0.
$$

The equality constraints are now absorbed into the affine parameterization, and the affine measured reference has disappeared from the residual expression because

$$
I_{comp}(c_{aff} + N_A z) - y_{aff} = I_{comp} N_A z.
$$

Once the primary optimum $\eta^*$ has been found, the reduced secondary LP becomes

$$
\min_{z \in \mathbb{R}^{F-q},\, s \in \mathbb{R}^{K},\, r \in \mathbb{R}^{F}} \; w_c^T r
$$

subject to

$$
N_A z \ge -c_{aff},
$$

$$
-s \le I_{comp} N_A z \le s,
\qquad
s \ge 0,
\qquad
w_y^T s = \eta^*,
$$

$$
-r \le N_A z \le r,
\qquad
r \ge 0.
$$

If the secondary LP is still tied, the fixed lexicographic component-ordering convention can be implemented by minimizing the relevant coordinate of $c_{aff} + N_A z$ over the current optimal face, one component at a time. The reduced formulation therefore remains linear throughout. There is no reduced Hessian, no Euclidean metric preservation argument, and no solver structure tied specifically to quadratic programming. The fixed matrices are $N_A$ and $I_{comp} N_A$; sample dependence enters through the right-hand side $-c_{aff}$. An orthonormal null-space basis is still numerically convenient because it keeps these reduced matrices well scaled, but its role is now computational rather than metric.

### 6.10 What the LP selector guarantees and what it does not

The full lexicographic LP selector guarantees that

$$
A c^* = A c_{in}
$$

and

$$
c^* \ge 0
$$

for every sample. It also guarantees that no feasible component state achieves a smaller weighted $L_1$ deviation from the affine measured reference $y_{aff}$, and that among all such primary-optimal states the deployed state minimizes the documented weighted $L_1$ deviation from $c_{aff}$. If those two stages are still tied, the fixed lexicographic component ordering chooses one state without changing the earlier optimality priorities.

It does not guarantee the following.

1. It does not guarantee realistic upper bounds or process-feasible operating ranges.
2. It does not impose kinetic feasibility beyond the chosen invariant relations and non-negativity.
3. It does not guarantee thermodynamic admissibility.
4. It does not correct errors introduced by a misspecified stoichiometric basis or an incorrect system boundary.
5. It does not by itself guarantee non-negative measured composites unless the composition map has the sign structure discussed in Section 7.2.

These are not derivation errors. They are the exact consequences of the feasible set and lexicographic objectives being enforced.

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

### 7.4 Final deployed prediction as affine core plus LP correction

The final deployed prediction is generally not equal to $y_{aff}$. Define the LP-correction term

$$
\delta_{LP}(u, c_{in}) = I_{comp}(c^* - c_{aff})
$$

Then

$$
y^* = y_{aff} + \delta_{LP}(u, c_{in})
$$

When the affine projector is already non-negative, $\delta_{LP}(u, c_{in}) = 0$. When it is not, $\delta_{LP}(u, c_{in})$ is the measured effect of the lexicographic LP selector: the primary LP keeps the deployed measured composites as close as possible to $y_{aff}$, while the secondary tie-break and any residual lexicographic convention choose which component-space state among the primary-optimal set is actually deployed. This decomposition is useful because it separates the globally affine, data-identifiable part of the model from the local LP-selected correction needed to enforce physical admissibility. It also exposes a key boundary: once the LP selector activates, $\delta_{LP}(u, c_{in})$ depends on the chosen component-space raw prediction, the documented weighting convention, and the fixed tie-break rule, so it is not identified from measured composite data alone unless a latent representative has already been fixed.

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

This blockwise expression remains useful for interpretation, but it must now be interpreted correctly. It decomposes the affine core $y_{aff}$, not necessarily the final deployed prediction $y^*$. The final prediction is obtained by adding the sample-specific LP correction $\delta_{LP}(u, c_{in})$.

## 8. Estimation and Identifiability from Measured Composite Data

At the estimation stage, three objects must be kept distinct: the latent component-space coefficient matrix $B$, the affine measured-space core, and the final deployed predictor $y^*$. Measured composite data identify an affine-core fit in measured space directly. They do not identify $B$ uniquely without additional component-space structure, and a further representability step may be needed before that measured-space fit can be used in a component-space deployment map of the form $M = G B$.

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

This least-squares stage is unchanged by the introduction of non-negativity. The non-negative ICSOR article does not replace coefficient estimation with a nonlinear estimator. It keeps least-squares estimation for the affine core and applies the non-negative correction after that stage. If one intends to reconstruct a component-space surrogate, Section 8.5 shows that a representability step may still be required before deployment.

### 8.4 Rank deficiency and interpretability

Second-order feature maps can be high dimensional, and real wastewater datasets may not excite all directions of that design space. If $N < D$, full column rank is impossible from the outset; even when $N \ge D$, the design may still be rank deficient because some feature directions are weakly or redundantly excited. In those cases, the fitted affine-core values remain well defined through the pseudoinverse, but the individual coefficients of $M$ are not uniquely identified. The minimum-norm pseudoinverse then returns one representative coefficient matrix, not a unique physical truth. Under rank deficiency, interpretation should focus on fitted affine-core predictions or on estimable linear functionals rather than on individual coefficients. In practical terms, the modeler then has three defensible options: reduce the feature basis, regularize the estimation problem, or shift the inferential emphasis from coefficients to predicted outputs and their uncertainty.

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

After estimation, however, the available object is $\widehat M$ rather than an exact population matrix $M$. The matrix

$$
\widehat B_{ls} = G^+ \widehat M
$$

is always defined and gives the minimum-Frobenius-norm least-squares representative because

$$
G \widehat B_{ls} = G G^+ \widehat M,
$$

which is the orthogonal projection of $\widehat M$ onto $\operatorname{range}(G)$. Define the compatible affine-core operator

$$
\widehat M_G = G \widehat B_{ls} = G G^+ \widehat M.
$$

It satisfies $\widehat M_G = \widehat M$ only when $\widehat M$ already lies in $\operatorname{range}(G)$. Therefore $\widehat B_{ls}$ should be interpreted as a chosen latent representative for deployment, not as an identified physical coefficient matrix, and $\widehat M_G$ is the corresponding representable affine core. Any other representative obtained by adding $(I_F - G^+ G) Z$ leaves that representable affine core unchanged but can alter the component-space raw state used by the later LP selection.

### 8.6 Final deployed predictor after estimation

Once a component-space representative $\widehat B$ has been fixed, the estimated raw component prediction for a new sample is

$$
\widehat c_{raw} = \widehat B \phi(u, c_{in})
$$

The final deployed non-negative component prediction is then obtained by the same staged component-space logic introduced in Section 6: keep $\widehat c_{raw}$ if it is already feasible, otherwise apply the affine projector, then solve the primary composite-space LP and the component-space tie-break LP, with the fixed lexicographic component ordering used only if those two stages are still tied. In compact notation,

$$
\widehat c^* = \operatorname{LexLP}_{\mathcal{S}_+}(\widehat c_{raw}, c_{in})
$$

and the final deployed measured prediction is

$$
\widehat y^* = I_{comp} \widehat c^*
$$

This last map is deterministic conditional on the chosen representative $\widehat B$, the documented weighting convention, and the fixed tie-break rule, but it is not globally affine in $\phi(u, c_{in})$. If the non-negativity constraints are inactive, then the deployed prediction collapses to the representable affine core

$$
\widehat y_{aff,G} = \widehat M_G \phi(u, c_{in}) + H c_{in},
$$

which equals the unconstrained least-squares affine-core fit only when $\widehat M$ already lies in $\operatorname{range}(G)$ or representability was enforced during estimation. If one or more LP stages activate, different admissible representatives can generate different $\widehat c_{raw}$ and therefore different LP-selected outputs. The least-squares coefficients therefore characterize the measured-space affine-core fit, while the final deployed predictor becomes a fully specified ICSOR model only after one of two additional steps is taken: select a representative such as $\widehat B_{ls} = G^+ \widehat M$, together with its compatible affine core $\widehat M_G = G G^+ \widehat M$, or impose extra component-space structure or data that identify $B$ more tightly.

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

Whenever finite-sample interval formulas are invoked below, the intended regime is the usual full-rank Gaussian multivariate linear model. In particular, $N > D$ and $\Phi$ has full column rank $D$. Define the fitted residual matrix

$$
\widehat E = \widetilde Y - \Phi \widehat M^T
$$

and the usual covariance estimator

$$
\widehat \Omega = \frac{1}{N-D} \widehat E^T \widehat E.
$$

### 9.2 Full-rank affine-core coefficient covariance

Under those full-rank conditions,

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
\widehat y_*^* = I_{comp} \, \operatorname{LexLP}_{\mathcal{S}_+}(\widehat c_{raw,*}, c_{in,*})
$$

This map is generally piecewise affine over polyhedral regions determined by the active non-negativity faces, the signed absolute-deviation faces of the primary LP, and the tie-break regime used inside the primary-optimal set. At boundaries where the optimal LP basis or the final lexicographic selection changes, the mapping is not described by one global coefficient matrix. Consequently, the closed-form affine-core variance formulas above are not exact finite-sample formulas for the final deployed predictor in general.

Two special cases are simpler.

1. If the affine projector is already non-negative at the prediction point, then $\widehat y_*^* = \widehat y_{aff,*}$ and the affine-core formulas apply exactly.
2. If the optimal LP basis and tie-break regime are locally stable, the final predictor is locally affine and the affine-core formulas can sometimes be adapted as a local approximation.

Neither of those special cases justifies a global exact interval formula for the final non-negative predictor.

### 9.5 Recommended uncertainty treatment for the final predictor

For the final deployed non-negative predictor, the most defensible default approach is resampling-based uncertainty quantification, such as bootstrap refitting or residual bootstrap, because it propagates uncertainty through both stages of the model: least-squares coefficient estimation and the sample-specific LP selection rule. Each replicate should rebuild the affine reference and rerun the primary LP, the secondary component-space tie-break LP, and the final lexicographic component-ordering convention when needed. In applications where only a fast approximation is needed, the affine-core intervals may still be reported as intervals for the linear core, provided they are labeled accordingly and not misrepresented as exact intervals for the final LP-selected predictor.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct effluent-state parameterization

The surrogate is parameterized on the effluent component state rather than on the net change. This keeps the learned target aligned with the quantity ultimately used for reporting and decision support. The cost is that the role of the influent state enters twice: once inside the feature map and once inside the constrained correction. That is not redundancy. The first role captures empirical dependence; the second role enforces the physically required part of the state.

### 10.2 Partitioned second-order feature structure

The partitioned feature map separates operating effects, influent-composition effects, and operation-loading interactions in a way that is interpretable to process engineers. The price of that interpretability is rapid feature growth, which can create multicollinearity, unstable coefficients, and weakly identified directions if the dataset does not adequately excite the design space. Since $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$, even moderate values of $M_{op}$ and $F$ can produce a feature basis that is large relative to the sample count. The resulting affine core can still interpolate training data through the pseudoinverse while leaving individual coefficients poorly determined.

### 10.3 Composite-space $L_1$ objective and component-space tie-break

The deployed correction is not defined by Euclidean nearest-point geometry. Its primary objective is the weighted $L_1$ deviation of the deployed measured composites from $y_{aff}$, while its secondary objective is the weighted $L_1$ deviation of the component state from $c_{aff}$ over the primary-optimal set. The output weights $w_y$ therefore define the main physical prioritization of the deployed model. Rescaling total COD relative to total nitrogen, for example, changes the model because it changes which composite discrepancies are treated as equally costly.

The component-space tie-break weights $w_c$ play a different role. They do not compete with the primary composite objective. They matter only after primary optimality has been fixed, and they determine which latent component state is selected when multiple feasible states share the same best composite fit. The final lexicographic component ordering is an even narrower convention: it exists only to make the deployed map single-valued if both earlier LP stages are still tied. Alternative weights, norms, or tie-break conventions are different model definitions rather than numerical afterthoughts.

### 10.4 Correction before measurement collapse

Enforcing the invariant relations and componentwise non-negativity in ASM component space before accepting a measured-space correction is a substantive modeling decision, not a notational convenience. The primary LP judges closeness in measured space, but it does so over component states that already respect the stoichiometric model. Once the state is collapsed into measured composites, some physically meaningful directions are no longer separately visible, so moving the entire correction into measured space would generally lose control over the latent ASM inventory.

### 10.5 Affine-core coefficients versus the final deployed predictor

The affine-core coefficients $M$ are the correct objects for direct engineering interpretation because they act directly on observed outputs through the identifiable least-squares stage. The latent component-space coefficients $B$ remain generally non-unique unless extra structure is imposed. The final deployed predictor $y^*$ adds one more layer: even when $M$ is well estimated, the final sample-specific output depends on the chosen latent representative used to form $c_{raw}$, on the composite-space weighting convention, and on the fixed tie-break rule used once the LP selector activates. That means coefficient interpretation is clearest for the affine core, whereas the final prediction should be read as affine signal plus a representative-dependent LP-selected feasibility correction.

## 11. Limitations

Non-negative ICSOR is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state in the quasi-steady-sample sense and does not represent temporal dynamics or path dependence.
2. It enforces only the invariant relations encoded by the chosen stoichiometric basis and system boundary together with componentwise non-negativity.
3. Non-negative component concentrations do not guarantee full kinetic, biological, or thermodynamic feasibility.
4. Non-negative component predictions imply non-negative measured composites only when the adopted composition matrix has the appropriate sign structure.
5. The correction depends on the chosen composite-output weights, component-space tie-break weights, and any scaling or normalization used to define them.
6. The final deployed predictor is not globally representable by one affine measured-space coefficient matrix once LP-regime changes become active.
7. Determinism of the deployed component state comes from the documented lexicographic LP selection rule rather than from intrinsic uniqueness of the primary feasible correction problem.
8. When the LP selector activates, the final deployed predictor is not identified from measured composite data alone unless a component-space representative is chosen or extra component-space information is supplied.
9. Exact closed-form prediction intervals are available for the affine core under the usual linear-model assumptions, but not in general for the final LP-selected deployed predictor.
10. The second-order feature basis can be statistically fragile if it is weakly excited or highly collinear.
11. A misspecified stoichiometric matrix or incorrect system boundary leads to a formally correct LP-selected state on the wrong physical constraint set.
12. If the influent ASM component state is reconstructed from measured aggregate variables rather than observed directly, reconstruction error enters upstream of the regression and is not represented by the affine-core output-noise covariance formulas derived here.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its claims correctly.

## 12. Conclusion

Non-negative ICSOR combines a partitioned second-order surrogate with a lexicographic LP selector derived from stoichiometric invariants, componentwise non-negativity, and measured-composite closeness to the affine reference. The framework is useful for wastewater applications because it preserves the distinction between operating conditions and influent composition, enforces conservation structure where that structure naturally lives, removes negative deployed component predictions, and returns predictions in the measured variables used by plant operators and simulation studies. In deployment, the correction should be evaluated hierarchically so that the LP stage is reserved for the subset of samples not already repaired by the raw-feasible and affine-feasible shortcuts.

The central theoretical point remains that measured composite data identify the affine measured-space operator $M$, not the latent component-space coefficient matrix $B$ uniquely. The non-negative extension does not alter that identifiability fact. Instead, it changes the deployment map: after estimating the affine core by least squares, the final prediction is obtained by applying a fully linear, lexicographic selection rule on the invariant-consistent non-negative set. When the affine reference is already non-negative, that deployed prediction collapses exactly to the identifiable affine core. When the LP selector activates, deployment additionally requires a chosen component-space representative, a composite-space weighting convention, and the fixed tie-break rule, because measured composite data alone do not identify the corrected component-space map uniquely. Under that reading, non-negative ICSOR is best understood as an analytically structured steady-state surrogate for activated-sludge prediction: more physically disciplined than a generic black-box regressor, more realistic than affine-only invariant correction when negative component states would otherwise occur, but still narrower in scope than a full dynamic mechanistic simulator.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
4. Seber, G. A. F., and Lee, A. J. Linear Regression Analysis. 2nd ed. Wiley, 2003.
5. Rao, C. R., and Mitra, S. Generalized Inverse of Matrices and Its Applications. Wiley, 1971.
6. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.
