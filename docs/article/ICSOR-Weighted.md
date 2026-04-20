# Weighted-Projection Invariant-Constrained Second-Order Regression (WP-ICSOR) for Activated Sludge Surrogate Modeling

## Abstract

This article presents weighted-projection invariant-constrained second-order regression (WP-ICSOR), a physics-informed surrogate model for steady-state activated-sludge systems. The model accepts operational variables and influent activated-sludge-model (ASM) component fractions, and it predicts effluent ASM component fractions in the same ASM basis. WP-ICSOR is therefore trained and deployed natively in ASM component space. If measured composite variables such as total COD, total nitrogen, total phosphorus, or suspended solids are needed, they are computed afterward by an external composition matrix. The collapse into measured-output space is not part of the model itself.

The formulation has two connected parts. First, the second-order component model predicts the effluent ASM component state directly as

$$
\hat c = \Pi \phi(u, c_{in}).
$$

Second, the raw component-space change $\hat c - c_{in}$ is passed through an embedded weighted projection onto the invariant-consistent change space. The projection uses prediction-dependent diagonal weights that make the conservation correction metric sensitive to components whose raw predicted values are near zero. Exact stoichiometric conservation is therefore enforced by construction, while the way the correction is distributed across components is biased by the selected weighting law. If measured composites are needed later, they are obtained only after this raw component prediction and embedded weighted projection have both been completed.

Each forward pass therefore consists of direct second-order component prediction followed by a reduced weighted-projection solve, but the parameter-estimation problem still has no closed-form solution because the deployed map depends on prediction-dependent weights. The coefficients are therefore estimated end to end with Adam and $L_1$ regularization for inherent feature selection on component-space targets. The framework is written for readers in chemical engineering, wastewater process modeling, and machine learning, with the structural guarantee of exact conservation separated explicitly from softer claims about near-boundary behavior.

## 1. Introduction and Modeling Objective

Surrogate models are valuable in wastewater engineering because they replace repeated numerical simulation or repeated plant-wide optimization with a direct input-output map. That speed matters when screening operating scenarios, embedding a reactor model in a larger optimization loop, or performing sensitivity studies over many influent conditions. In this article, each sample is assumed to represent a quasi-steady operating condition: the operating variables, influent composition, and effluent response are treated as effectively time-invariant over the control volume being modeled for the sampling window of interest. The usual difficulty is that a generic data-driven regressor can fit observed effluent data while still violating fundamental conservation structure. In activated-sludge modeling, that failure is not a minor technical detail. It undermines the physical credibility of the surrogate because it can imply component inventories that are inconsistent with the adopted reaction network even when the measured aggregates appear plausible.

The source of the problem is a mismatch between the space in which wastewater stoichiometry is defined and the space in which plant variables are often reported.

1. The mechanistic stoichiometric model is written in an ASM component basis, such as soluble substrate, ammonium, nitrate, autotrophic biomass, particulate organics, phosphate, dissolved oxygen, and alkalinity.
2. Plant or simulator dashboards often report composite variables, such as total COD, total nitrogen, total phosphorus, TSS, or VSS.

Those two spaces are related, but they should not be confused. Conservation laws are naturally expressed in the ASM component basis because the stoichiometric matrix acts on individual components. Composite outputs are downstream aggregates of those components. In the formulation developed here, that downstream aggregation is treated explicitly as an external reporting step rather than as the model target.

WP-ICSOR embeds the physical structure directly in the forward model. The effluent ASM component targets are predicted directly from the shared second-order feature vector, and the predicted component-space change is passed through a weighted projection layer that is active during both coefficient estimation and deployment. The projection preserves stoichiometric invariants exactly because it operates only on admissible changes. Its weighting law does not impose hard inequality constraints; instead, it changes the metric of the conservation correction so that components with small raw predicted values are altered reluctantly during projection.

This article answers one precise question:

> Given a steady-state influent ASM component-fraction vector and a steady-state operating condition, what effluent ASM component-fraction vector should be predicted if the underlying effluent ASM component state is governed by a second-order component model, must satisfy the conserved quantities implied by the adopted stoichiometric model, and is corrected in component space by an embedded weighted projection?

The theory in this article is restricted to steady-state reactor-block prediction. It does not aim to replace a dynamic activated-sludge simulator. Rather, it provides an analytically structured surrogate that preserves stoichiometric structure, predicts component targets directly, embeds a conservation-preserving weighted projection in the model, and remains trainable with gradient-based optimization. Because the weighted projection is part of the model itself, the final deployed component predictor is not globally affine in the parameters and cannot be estimated by ordinary least squares. The discussion proceeds from physical scope and notation, to the invariant relations, to the direct second-order component model, to the embedded weighted projection, to external reporting collapse, and finally to end-to-end estimation and uncertainty. Throughout, exact guarantees are separated from modeling preferences and implementation choices.

## 2. Physical Scope, State Spaces, and Notation

### 2.1 Control volume and modeling scope

We consider a fixed reactor block or fixed process unit represented by quasi-steady samples. The system boundary is the same boundary used to define the influent and effluent state vectors. External sources or sinks that cross that boundary must either be represented explicitly in the adopted stoichiometric model or be excluded from the claim of invariant preservation. This includes transport or removal mechanisms such as bypass streams, gas stripping, chemical dosing, or sludge wastage if they cross the chosen boundary and are not encoded in the stoichiometric description. In addition, the component states entering the surrogate are assumed to be expressed on a common non-negative component-fraction basis. If flow scaling, phase partitioning, or residence-time normalization are needed to make $c_{out} - c_{in}$ commensurate with stoichiometric change, that preprocessing is part of the model definition rather than a detail left implicit. The theory therefore applies only after the modeler has fixed the following items:

1. the reactor or process block being represented,
2. the ASM component basis used to describe material composition,
3. the stoichiometric matrix associated with that basis, and
4. the composition matrix used later for optional external reporting in measured-output space.

The framework is steady-state in the sense of quasi-steady samples rather than full dynamic trajectories. It does not represent settling dynamics, sludge age dynamics, sensor dynamics, start-up transients, or time-varying trajectories. In plant applications, the samples would typically correspond to stable operating windows or time-averaged periods rather than literal mathematical equilibria. Changing the system boundary changes the admissible stoichiometric change space and therefore changes both the conservation relations and the embedded weighted projection layer.

### 2.2 Why two state spaces are still needed

To make the distinction concrete, suppose the underlying component basis contains soluble biodegradable substrate, particulate biodegradable substrate, ammonium, nitrate, phosphate, dissolved oxygen, alkalinity, and biomass fractions. A plant rarely measures all of those components directly. Instead, it may report total COD, total nitrogen, total phosphorus, TSS, and VSS. Those measured variables are linear combinations of the component concentrations under a chosen analytical convention.

The surrogate must therefore operate across two linked spaces:

1. ASM component space, where stoichiometry, invariants, direct component prediction, and prediction-dependent weighting are defined.
2. Measured composite space, which is retained only as an external reporting space obtained by applying a fixed composition matrix after prediction.

WP-ICSOR learns and constrains the prediction in component space and only afterward, if desired, maps the result to measured space. That order is essential in the formulation. Stoichiometric invariants originate in the component basis, the raw second-order predictor acts on component targets, and the weighted projection layer is defined on component-space changes. A measured-space-only training target would generally be too weak to control the unobserved ASM component inventory.

### 2.3 Notation

Single-sample vectors are written as column vectors. Dataset matrices are defined later with samples stored by rows.

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $u$ | $\mathbb{R}^{M_{op}}$ | Operational input vector, for example hydraulic retention time, aeration intensity, recycle ratio, or other manipulated or design variables |
| $c_{in}$ | $\mathbb{R}_{+}^{F}$ | Influent ASM component-fraction vector |
| $c_{out}$ | $\mathbb{R}_{+}^{F}$ | True steady-state effluent ASM component-fraction vector |
| $\hat c$ | $\mathbb{R}^{F}$ | Raw component prediction obtained directly from the second-order feature map |
| $c^*$ | $\mathbb{R}^{F}$ | Projected component prediction used for deployment |
| $y_{ext}$ | $\mathbb{R}^{K}$ | External measured composite vector computed from a component vector |
| $I_{comp}$ | $\mathbb{R}^{K \times F}$ | Composition matrix mapping ASM component fractions to measured composite variables for external reporting |
| $\nu$ | $\mathbb{R}^{R \times F}$ | Stoichiometric matrix with $R$ reactions and $F$ ASM components |
| $\xi$ | $\mathbb{R}^{R}$ | Net reaction progress vector expressed in concentration-equivalent units so that $\nu^T \xi$ has the same units as $c_{out} - c_{in}$ |
| $A$ | $\mathbb{R}^{q \times F}$ | Full-row-rank matrix whose rows span the invariant space, equivalently $A \nu^T = 0$ |
| $N_A$ | $\mathbb{R}^{F \times (F-q)}$ | Matrix whose columns form an orthonormal basis of $\operatorname{null}(A)$, the admissible change space |
| $\phi(u, c_{in})$ | $\mathbb{R}^{D}$ | Engineered second-order feature map |
| $D$ | scalar | Feature dimension, $D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F$ |
| $\Pi$ | $\mathbb{R}^{F \times D}$ | Second-order coefficient matrix mapping the feature map directly to raw effluent component predictions |
| $I_F$ | $\mathbb{R}^{F \times F}$ | Identity matrix in ASM component space |
| $\Delta \hat c$ | $\mathbb{R}^{F}$ | Raw predicted component-space change, $\Delta \hat c = \hat c - c_{in}$ |
| $\alpha$ | $\mathbb{R}^{F-q}$ | Coordinates of an admissible stoichiometric change in the basis $N_A$ |
| $W_{\beta}(\hat c)$ | $\mathbb{R}^{F \times F}$ | Diagonal prediction-dependent weight matrix defining the metric of the embedded weighted projection |
| $\varepsilon$ | $\mathbb{R}_{++}$ | Small positive floor used to stabilize the weights |
| $\beta$ | $\mathbb{R}_{++}$ | Weight-sharpening exponent controlling how aggressively the projection resists the zero boundary |
| $\vartheta$ | collection | Trainable parameter set, $\vartheta = \{\Pi\}$ |

The external measured variables are defined by the linear map

$$
y_{ext} = I_{comp} c.
$$

When this map is applied to the true effluent state, it yields the true measured composites. When it is applied to the deployed component prediction $c^*$, it yields the externally reported prediction $y_{ext} = I_{comp} c^*$. This reporting step is outside the model: WP-ICSOR itself predicts only ASM component fractions.

## 3. Modeling Assumptions

The framework rests on the following assumptions. These are not optional preferences left to the reader. They define the exact model analyzed in this article.

1. **Steady-state scope.** Each sample represents a quasi-steady input-output condition, typically a stable operating epoch or time-averaged window. The model is not a dynamic state estimator.
2. **Fixed component basis.** The ASM component basis and the associated stoichiometric matrix are fixed before regression begins.
3. **Consistent system boundary.** The same physical boundary is used to define $c_{in}$, $c_{out}$, and the conservation statement. Any external source or sink outside that boundary is outside the present model.
4. **Common component-fraction basis.** The vectors $c_{in}$ and $c_{out}$ are expressed on a common basis so that $c_{out} - c_{in}$ and $\nu^T \xi$ are dimensionally commensurate.
5. **Direct component-space supervision.** The training targets are effluent ASM component fractions, not measured composite outputs.
6. **External composition map.** The collapse from ASM component fractions to measured composites is an external calculation performed after prediction through a known fixed matrix $I_{comp}$.
7. **Direct effluent-state parameterization.** The surrogate is parameterized to predict the effluent ASM component state directly rather than the component change alone.
8. **Second-order surrogate class.** The base predictor is a partitioned second-order polynomial model that includes linear, quadratic, and operation-loading interaction terms.
9. **Direct component prediction.** The raw effluent ASM component prediction is parameterized directly as $\hat c = \Pi \phi(u, c_{in})$.
10. **Embedded weighted projection.** Stoichiometric conservation is enforced by projecting the predicted change onto $\operatorname{null}(A)$ inside the model architecture rather than by a separate post-training correction stage.
11. **Near-zero-sensitive weighting, not hard non-negativity constraints.** Prediction-dependent diagonal weights with exponent $\beta > 1$ make the projection metric highly sensitive to components whose raw predicted values are small. This influences how the conservation correction is distributed across coordinates, but it does not impose the inequality constraint $c^* \ge 0$.
12. **Influent feasibility.** The influent reference state is assumed non-negative in component space.
13. **Conditional composite-sign interpretation.** If $c^*$ is componentwise non-negative and the relevant rows of $I_{comp}$ are entrywise non-negative, then the corresponding externally reported composites are non-negative. The model guarantees the second premise only if the measurement convention satisfies it; it does not guarantee the first premise.
14. **Optimization-based estimation.** Because the forward model depends nonlinearly on $\Pi$ through a prediction-dependent weighted projection, the coefficients are estimated by gradient-based optimization rather than ordinary least squares.
15. **Lasso regularization.** $L_1$ penalties are used to encourage sparse second-order coefficients as an inherent feature-selection device.

These assumptions matter because each one narrows the scientific claim. WP-ICSOR embeds conservation exactly and introduces a near-zero-sensitive weighted correction inside the model, but it is still not automatically guaranteed to be componentwise non-negative or fully process-realizable in every operating regime.

## 4. Stoichiometric Structure and Conserved Quantities

### 4.1 From stoichiometric reactions to component-state change

Let $\nu \in \mathbb{R}^{R \times F}$ be the stoichiometric matrix written in the adopted ASM component basis. Its entries are treated here as fixed stoichiometric coefficients in that basis; any scaling needed to express component-state change in concentration units is absorbed into the definition of the reaction-progress vector. For one steady-state sample, define the net reaction progress vector $\xi \in \mathbb{R}^{R}$ so that

$$
c_{out} - c_{in} = \nu^T \xi.
$$

This equation is the starting point of the theory. It says that the net change in the effluent component state is a linear combination of reaction stoichiometries. The entries of $\xi$ need not be observed individually. They collect the net progression of each modeled reaction over the chosen control volume after scaling into concentration-equivalent units. For example, if reaction $i$ has steady-state rate $r_i$ in units of concentration per time and the relevant hydraulic time scale of the control volume is $\tau$, then one admissible definition is $\xi_i = r_i \tau$, which has concentration units. More generally, $\xi_i$ may be interpreted as the net integrated reaction extent over the control volume after whatever normalization is required so that $\nu^T \xi$ is expressed in the same units as $c_{out} - c_{in}$. The relation should therefore be read as a reduced boundary-consistent closure in the adopted component basis, not as a replacement for a full flow-resolved dynamic mass-balance model.

### 4.2 Invariant relations implied by the stoichiometric matrix

The reaction progress vector $\xi$ is not part of the surrogate model and is usually not observed. To eliminate it, we introduce a full-row-rank matrix $A \in \mathbb{R}^{q \times F}$ whose rows span the invariant space. Equivalently,

$$
A \nu^T = 0.
$$

Multiplying the stoichiometric change relation by $A$ gives

$$
A(c_{out} - c_{in}) = A \nu^T \xi = 0,
$$

which implies the affine invariant relation

$$
A c_{out} = A c_{in}.
$$

Each row of $A$ represents one independent conserved combination of ASM components under the adopted stoichiometric model and system boundary. The exact physical interpretation depends on the chosen basis and stoichiometric matrix. In activated-sludge applications, the conserved combinations may correspond to pools such as COD equivalents, nitrogen equivalents, phosphorus equivalents, or charge-related balances, but only when those balances are actually implied by the adopted reaction network and boundary.

### 4.3 Why the basis of $A$ is not unique

The matrix $A$ is not unique. If $R_A \in \mathbb{R}^{q \times q}$ is invertible, then $\widetilde A = R_A A$ generates the same constraint set because

$$
\widetilde A c = \widetilde A c_{in}
\quad \Longleftrightarrow \quad
R_A A c = R_A A c_{in}
\quad \Longleftrightarrow \quad
A c = A c_{in}.
$$

Thus, the physics is carried by the row space of $A$, not by one particular numerical basis. This matters because the admissible change basis $N_A$ used later in the weighted projection depends only on $\operatorname{null}(A)$ and therefore only on the invariant subspace being enforced.

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

so the invariant relation becomes

$$
c_{out,1} + c_{out,2} = c_{in,1} + c_{in,2}.
$$

The admissible change space is the null space of $A$, namely

$$
\operatorname{null}(A) = \operatorname{span}\left\{ \begin{bmatrix} 1 \\ -1 \end{bmatrix} \right\}.
$$

If the raw model proposes a change that violates this relation, the weighted projection replaces it with the closest change, under the chosen prediction-dependent weights, that stays on this line. Conservation is therefore exact after projection, while the weights bias the correction against moving mass into a component that is already near zero.

### 4.5 ASM-flavored miniature example with external reporting

Suppose the component vector is

$$
c = \begin{bmatrix} S_S \\ X_S \\ S_{NH} \end{bmatrix},
$$

where $S_S$ is soluble substrate, $X_S$ is particulate substrate, and $S_{NH}$ is ammonium. Let one simplified reaction convert soluble substrate into particulate substrate without changing ammonium,

$$
\nu = \begin{bmatrix}
-1 & 1 & 0
\end{bmatrix},
$$

so one admissible invariant basis is

$$
A = \begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

If one later wishes to report total COD and ammonium externally, one may use

$$
I_{comp} = \begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Now let the second-order predictor produce

$$
\Pi \phi(u, c_{in}) = \begin{bmatrix} 8 \\ 14 \\ 5 \end{bmatrix}
$$

The raw component prediction is therefore read directly from the second-order regressor. The embedded weighted projection is then applied to $\Delta \hat c = \hat c - c_{in}$ so that the corrected change lies in $\operatorname{null}(A)$. If measured composites are later needed, they are computed externally as $y_{ext} = I_{comp} c^*$. This ordering is the core design rule of WP-ICSOR: component prediction and invariant-preserving correction both happen in component space first, and measured composites are computed only afterward.

## 5. Second-Order Surrogate in ASM Component Space

### 5.1 Why the input is partitioned

In activated-sludge systems, operating conditions and influent component concentrations play different physical roles.

1. Operating variables such as hydraulic retention time, dissolved-oxygen setpoint, or recycle settings alter the process environment.
2. Influent component concentrations describe the material inventory entering that environment.

Treating those two groups as interchangeable predictors hides an important engineering distinction. WP-ICSOR therefore partitions the input into an operational block $u$ and an influent component block $c_{in}$.

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
\end{bmatrix} \in \mathbb{R}^{D},
$$

where $\otimes$ denotes the Kronecker product, so that $u \otimes u \in \mathbb{R}^{M_{op}^2}$, $c_{in} \otimes c_{in} \in \mathbb{R}^{F^2}$, and $u \otimes c_{in} \in \mathbb{R}^{M_{op}F}$. The quadratic blocks are retained in explicit vectorized form rather than reduced to a symmetry-compressed basis. This choice keeps the algebra transparent, but it is not a minimal basis: symmetric cross-terms such as $u_i u_j$ and $u_j u_i$ appear twice. As a result, individual coefficients attached to duplicated monomials should be interpreted only through their symmetrized combined effect, not as uniquely identifiable physical quantities.

Throughout the chapter, we use the convention $u \otimes u = \operatorname{vec}(u u^T)$, $c_{in} \otimes c_{in} = \operatorname{vec}(c_{in} c_{in}^T)$, and $u \otimes c_{in} = \operatorname{vec}(u c_{in}^T)$ under column-wise vectorization. The resulting feature dimension is therefore

$$
D = 1 + M_{op} + F + M_{op}^2 + F^2 + M_{op}F.
$$

### 5.3 Direct second-order component model

The raw effluent ASM component prediction is defined directly by

$$
\hat c = \Pi \phi(u, c_{in}).
$$

Blockwise, this may be written as

$$
\hat c = b + W_u u + W_{in} c_{in} + \Theta_{uu}(u \otimes u) + \Theta_{cc}(c_{in} \otimes c_{in}) + \Theta_{uc}(u \otimes c_{in}),
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
\Theta_{uc} \in \mathbb{R}^{F \times (M_{op}F)}.
$$

Each block has a direct physical interpretation: first-order operating effects, first-order influent effects, operating curvature, influent curvature, and operation-loading interactions. This blockwise signal is the raw component prediction presented to the projection layer.

### 5.4 Componentwise interpretation

Componentwise, the $f$-th raw prediction equation is

$$
\hat c_f = \pi_f^T \phi(u, c_{in}),
$$

where $\pi_f^T$ is the $f$-th row of $\Pi$. The model is therefore multi-output because all component equations share the same feature vector and coefficient matrix. Dependence among the predicted components enters through the shared predictors and through the common invariant-preserving projection applied afterward.

### 5.5 Matrix form and samplewise evaluation

The raw component predictor is already in direct matrix form,

$$
\hat c = \Pi \phi(u, c_{in}).
$$

For fixed $\Pi$, evaluating $\hat c$ requires only matrix-vector multiplication. The deployed predictor is nevertheless not linear in the parameters because the weighted projection introduced in the next section depends on the current raw prediction. Hence the coefficient fit is not available from a single closed-form least-squares solve.

Two observations are important here.

1. The raw component prediction happens before any external reporting collapse because $\Pi$ acts in component space.
2. The only reduced linear solve in each forward pass comes from the weighted projection.

### 5.6 Interpretation of $\Pi$

The matrix $\Pi$ is not a stoichiometric matrix and it is not a measurement map. It parameterizes an unconstrained data-driven guess for the effluent ASM component state before conservation correction is applied. This separation is intentional: the raw predictor absorbs empirical input-output structure, while the embedded projection enforces the invariant relations. The tradeoff is that $\Pi$ alone does not guarantee admissibility; that guarantee is introduced only after projection.

## 6. Embedded Weighted Projection on the Invariant-Consistent Change Space

### 6.1 Why the projection is embedded instead of appended

The purpose of the weighted projection layer is to enforce stoichiometric conservation during coefficient estimation as well as during deployment. The weighted projection is part of the forward map used inside the loss, so every gradient update sees the conservation structure.

The projection acts on the raw predicted change

$$
\Delta \hat c = \hat c - c_{in},
$$

not directly on the measured outputs. This keeps the physical correction in component space, where the invariants are defined.

### 6.2 Admissible stoichiometric change space

Because

$$
A c_{out} = A c_{in},
$$

any admissible change must satisfy

$$
A(c_{out} - c_{in}) = 0.
$$

Therefore every invariant-consistent component-space change lies in

$$
\operatorname{null}(A).
$$

Let $N_A \in \mathbb{R}^{F \times (F-q)}$ have orthonormal columns spanning this null space. Then any admissible change can be written as

$$
\Delta c = N_A \alpha
$$

for some coefficient vector $\alpha \in \mathbb{R}^{F-q}$. This admissible change space is part of the model architecture itself.

Two edge cases are worth stating explicitly. If $q = 0$, no independent invariant equations are imposed and one may take $N_A = I_F$, so the projection reduces to the identity map on changes. If $q = F$, then $\operatorname{null}(A) = \{0\}$ and the only admissible change is zero, which forces $c^* = c_{in}$.

### 6.3 Prediction-dependent near-zero-sensitive weights

The raw component prediction $\hat c$ may still place some components close to zero or below the intended operating regime. To make the conservation correction sensitive to that regime, the projection is weighted componentwise with a sharpened inverse-power law. Define the diagonal matrix

$$
W_{\beta}(\hat c) = \operatorname{diag}\left((\max(\hat c_1, \varepsilon))^{-\beta}, \ldots, (\max(\hat c_F, \varepsilon))^{-\beta}\right),
$$

where $\varepsilon > 0$ is a small stabilization floor and $\beta > 1$ controls the near-boundary penalty. Small or non-positive raw component predictions therefore induce very large weights. The key point is interpretive: these weights do not penalize a negative component directly. They penalize deviations between the projected change and the raw proposed change more heavily in coordinates whose raw predicted values are small. The weighted projection therefore tends to leave those coordinates closer to the raw component prediction while shifting the conservation correction into less heavily weighted coordinates.

The exact formula above uses a hard floor for clarity. In differentiable software implementations, one may replace $\max(\hat c_i, \varepsilon)$ by a smooth positive proxy with the same limiting behavior. Doing so makes the forward map fully smooth and avoids a kink at $\hat c_i = \varepsilon$. The inverse-power form is used throughout this article.

### 6.4 Weighted projection objective and closed-form solution

The embedded weighted projection solves

$$
\alpha^* = \arg\min_{\alpha \in \mathbb{R}^{F-q}} \left\| W_{\beta}(\hat c) \left( N_A \alpha - \Delta \hat c \right) \right\|_2^2.
$$

This is a weighted least-squares projection of the raw predicted change onto the admissible change space. Because the problem is quadratic in $\alpha$, the first-order optimality condition is

$$
N_A^T W_{\beta}(\hat c)^2 N_A \, \alpha
= N_A^T W_{\beta}(\hat c)^2 \Delta \hat c.
$$

When $N_A^T W_{\beta}(\hat c)^2 N_A$ is invertible, the optimizer therefore has the closed form

$$
\alpha^* = \left( N_A^T W_{\beta}(\hat c)^2 N_A \right)^{-1} N_A^T W_{\beta}(\hat c)^2 \Delta \hat c.
$$

The projected component state is therefore

$$
c^* = c_{in} + N_A \left( N_A^T W_{\beta}(\hat c)^2 N_A \right)^{-1} N_A^T W_{\beta}(\hat c)^2 (\hat c - c_{in}).
$$

It is useful to define the weighted projector

$$
P_{N_A}^{W_{\beta}(\hat c)} = N_A \left( N_A^T W_{\beta}(\hat c)^2 N_A \right)^{-1} N_A^T W_{\beta}(\hat c)^2,
$$

so that

$$
c^* = c_{in} + P_{N_A}^{W_{\beta}(\hat c)} (\hat c - c_{in}).
$$

The same projected state may also be written in the constraint-space form

$$
c^* = \hat c - W_{\beta}(\hat c)^{-2} A^T \left( A W_{\beta}(\hat c)^{-2} A^T \right)^{-1} A(\hat c - c_{in}),
$$

provided $A W_{\beta}(\hat c)^{-2} A^T$ is invertible.

The null-space expression aligns directly with the admissible-change interpretation, whereas the constraint-space expression works directly with the invariant equations. Computationally, neither form dominates unconditionally: the preferable implementation depends on whether $F-q$ or $q$ is smaller and on which reduced system is better conditioned in the application at hand.

### 6.5 What the embedded projection guarantees

The projection guarantees exact conservation. Since $A N_A = 0$,

$$
A c^* = A c_{in} + A N_A \alpha^* = A c_{in}.
$$

Thus the stoichiometric invariants are enforced by construction for every forward pass, every optimization iterate, and every deployed sample.

The weighting changes how the exact conservation correction is allocated across components. Components with small raw predicted values receive sharply increasing weights, so the projection is reluctant to alter those coordinates in order to satisfy the invariant equations. This can reduce the risk that the projection itself creates additional distortion in depleted components, but it is not equivalent to the hard constraint $c^* \ge 0$. In particular, if the raw component prediction already places a component below zero, the weighted projection does not in general repair that violation. Exact conservation is a structural guarantee; non-negativity is not. If hard non-negativity is required, the projection step must be replaced by an inequality-constrained quadratic program or another explicitly constrained parameterization.

### 6.6 Relation to the standard orthogonal projector

If $W_{\beta}(\hat c) = I_F$, the weighted projector reduces to the standard orthogonal projector onto the admissible change space,

$$
P_{N_A}^{I_F} = N_A (N_A^T N_A)^{-1} N_A^T = N_A N_A^T,
$$

because the columns of $N_A$ are orthonormal. The formulation therefore generalizes the classical invariant-consistent orthogonal projection by replacing Euclidean distance with a prediction-dependent weighted metric. In physical terms, the choice of metric decides which components absorb the conservation correction required to bring $\Delta \hat c$ back into the admissible stoichiometric change space.

### 6.7 Embedded forward-pass sequence

For each sample, the deployed forward map is evaluated in the following order:

1. Form the second-order feature vector $\phi(u, c_{in})$.
2. Evaluate the raw component predictor $\hat c = \Pi \phi(u, c_{in})$.
3. Form the raw change $\Delta \hat c = \hat c - c_{in}$.
4. Build the sharpened prediction-dependent weight matrix $W_{\beta}(\hat c)$.
5. Solve the weighted projection for $\alpha^*$ and obtain $c^*$.
6. If measured composites are needed, compute them afterward as the external report $y_{ext} = I_{comp} c^*$.

Because this sequence sits inside the training objective, conservation and the chosen correction metric influence coefficient estimation directly. There is no separate projection layer attached after training is complete.

## 7. External Composition-Matrix Collapse

### 7.1 External reporting equation

If measured composites are needed for reporting, they are obtained after the model prediction has already been completed in ASM component space. The external reporting equation is

$$
y_{ext} = I_{comp} c^*.
$$

This map is not part of WP-ICSOR itself. The model's native output is the component-space prediction $c^*$.

### 7.2 Prediction and correction remain in component space

The order of operations is essential.

1. The base second-order signal is formed in component space.
2. Raw component targets are predicted directly as $\hat c = \Pi \phi(u, c_{in})$.
3. The raw component-space change is projected onto the invariant-consistent change space with prediction-dependent weighting.
4. The corrected component state $c^*$ is the deployed model output.
5. Any measured composite is obtained only afterward by applying the external composition matrix.

This ordering prevents the loss of physical information that would occur if one attempted to train or correct only the measured aggregates. Two different ASM component states can collapse to the same measured COD or nitrogen total while implying different stoichiometric feasibility. WP-ICSOR resolves that ambiguity by learning and enforcing structure in component space before any reporting collapse is applied.

### 7.3 When componentwise non-negativity implies non-negative reported composites

If every row of $I_{comp}$ is entrywise non-negative, then componentwise non-negativity in the component state transfers naturally to the measured outputs. For output index $k$,

$$
(y_{ext})_k = \sum_{f=1}^{F} (I_{comp})_{k f} c_f^*.
$$

If $(I_{comp})_{k f} \ge 0$ for all $f$ and $c_f^* \ge 0$ for all $f$, then $(y_{ext})_k \ge 0$. This is the common case for total COD, total nitrogen, total phosphorus, TSS, or VSS defined as sums with non-negative conversion factors.

If a chosen measurement convention uses negative coefficients, extra output-space treatment would be required to make a measured-space sign claim. Conversely, even with non-negative rows in $I_{comp}$, the present architecture makes no unconditional non-negativity guarantee because the weighted projection does not impose $c^* \ge 0$. Both conditions must be stated explicitly.

### 7.4 End-to-end component predictor and optional external report

Combining the direct component predictor and the weighted projection gives the full deployed component predictor

$$
c^*(u, c_{in}; \vartheta)
= c_{in} + N_A \left( N_A^T W_{\beta}(\hat c)^2 N_A \right)^{-1} N_A^T W_{\beta}(\hat c)^2 \left( \Pi \phi(u, c_{in}) - c_{in} \right),
$$

where

$$
\hat c = \Pi \phi(u, c_{in}).
$$

This expression defines the deployed predictor as a single end-to-end nonlinear map produced by direct component prediction followed by an embedded weighted projection. If measured composites are needed afterward, they are computed externally as $y_{ext}(u, c_{in}; \vartheta) = I_{comp} c^*(u, c_{in}; \vartheta)$.

## 8. Estimation by Adam with Lasso Regularization

### 8.1 Why Closed-Form Coefficient Estimation Does Not Apply

It is important to distinguish between the samplewise forward evaluation and the global coefficient-estimation problem.

1. **Forward prediction.** For fixed $\Pi$, the raw component state $\hat c$ is obtained directly by evaluating $\Pi \phi(u, c_{in})$, and the projected state $c^*$ is obtained by a reduced weighted least-squares solve.
2. **Coefficient estimation.** The parameters themselves are not available from a closed-form least-squares or pseudoinverse formula.

The second point follows from the structure of the forward map. The embedded weighted projection depends on the current prediction through $W_{\beta}(\hat c)$, and $\hat c$ itself depends on $\Pi$. The deployed component predictor is therefore nonlinear in the parameters even though the raw second-order model is linear in the features and coefficients. Coefficient estimation therefore proceeds through gradient-based optimization.

### 8.2 Dataset-level training objective

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
C_{in} = \begin{bmatrix}
c_{in,1}^T \\
c_{in,2}^T \\
\vdots \\
c_{in,N}^T
\end{bmatrix} \in \mathbb{R}^{N \times F},
$$

$$
C_{out} = \begin{bmatrix}
c_{out,1}^T \\
c_{out,2}^T \\
\vdots \\
c_{out,N}^T
\end{bmatrix} \in \mathbb{R}^{N \times F}.
$$

For sample $n$, the deployed prediction is $c_n^*(\vartheta)$ as defined in Section 7.4. A natural component-space training objective is

$$
\min_{\Pi} \; \mathcal{J}(\Pi)
$$

where

$$
\mathcal{J}(\Pi)
= \frac{1}{N} \sum_{n=1}^{N} \left\| c_{out,n} - c_n^*(\Pi) \right\|_2^2
+ \lambda_{\Pi} \lVert \Pi \rVert_1.
$$

The first term fits effluent ASM component fractions after the raw component prediction and embedded weighted projection have both been applied. The $L_1$ term is a lasso penalty that encourages sparsity in the second-order coefficients.

If component scaling is needed because some ASM fractions are estimated more reliably or are judged more important than others, the squared loss above may be replaced by a weighted component-space norm without changing the architecture.

### 8.3 Adam optimization and inherent feature selection

With the hard floor in $W_{\beta}(\hat c)$, the optimization problem is piecewise differentiable and is well suited to Adam or a comparable first-order optimizer. At each iteration, the training loop executes the same physically structured forward pass used in deployment:

1. build the second-order features,
2. evaluate the raw component predictor,
3. apply the embedded weighted projection,
4. evaluate the component-space loss, and
5. add the lasso penalties.

Gradients are then propagated back through the reduced projection solve. In this way, conservation and the chosen projection metric affect the coefficient updates directly. The model is not first trained as a generic regressor and then repaired afterward.

The lasso penalties serve as inherent feature selection. If a quadratic interaction, influent term, or operating term does not materially improve the component prediction after the embedded projection is taken into account, the corresponding entry of $\Pi$ can be driven toward zero.

### 8.4 Regularization in the direct model

The direct architecture narrows the parameter space and eliminates the need to maintain an additional invertible operator during training. Regularization is therefore concentrated on the second-order coefficient matrix $\Pi$ and on the hyperparameters that control the weighted projection. This simplification improves interpretability, but it also means that residual cross-component structure must be represented through the shared features and the common projection step.

### 8.5 Deployment after training

Once $\widehat \Pi$ has been estimated, a new sample is evaluated by the same embedded forward map:

$$
\widehat c^*(u, c_{in}) = c^*(u, c_{in}; \widehat \Pi),
$$

where $\widehat c^*$ is obtained from the direct component predictor and weighted projection using the learned parameters. If measured composites are needed, they are computed afterward as $\widehat y_{ext}(u, c_{in}) = I_{comp} \widehat c^*(u, c_{in})$. The learned model is evaluated through the same embedded forward map used during training.

## 9. Identifiability, Well-Posedness, and Uncertainty

### 9.1 What the data identify in the uncoupled model

Component-space training data identify the end-to-end prediction map induced by the projected second-order component model. They still do not, in general, determine a unique second-order coefficient matrix.

This has three consequences.

1. Different coefficient matrices $\Pi$ can yield similar component predictions once the embedded projection is applied.
2. The lasso penalty selects one sparse, well-posed representative rather than recovering a unique coefficient matrix in an underdetermined or weakly excited design.
3. If the full Kronecker basis is retained, symmetric duplicate monomials further weaken uniqueness of individual quadratic coefficients; the induced predictor is identifiable more readily than any single coefficient attached to a duplicated feature coordinate.

This behavior is an intrinsic consequence of the nonlinear projected architecture, the embedded physical projection, and the duplicated structure of the full second-order basis.

### 9.2 Existence and numerical stability of the forward pass

The forward pass is well defined when the weighted projection inverse exists.

The weighted projection requires

$$
N_A^T W_{\beta}(\hat c)^2 N_A
$$

to be invertible. This holds when $N_A$ has full column rank and $W_{\beta}(\hat c)$ is positive definite. The stabilization floor $\varepsilon > 0$ ensures that every diagonal weight is finite and strictly positive, so $W_{\beta}(\hat c)^2$ is positive definite. Because the columns of $N_A$ are linearly independent, the reduced matrix is then positive definite as well.

Under these conditions, the forward map is continuous in the parameters away from the chosen weight-floor transitions or their smooth approximations.

### 9.3 Uncertainty assessment

Exact ordinary-least-squares covariance formulas and $t$-based prediction intervals are not available for WP-ICSOR because the deployed predictor is not globally affine in the parameters and because the parameters are estimated by nonconvex gradient-based optimization. The most defensible default strategy is therefore resampling-based uncertainty quantification in component space.

Recommended options include the following.

1. **Bootstrap refitting.** Resample the training set, refit the model with Adam, and recompute component predictions.
2. **Ensemble refitting.** Train multiple models from different initializations and summarize predictive spread.
3. **Validation-based sensitivity analysis.** Examine how predictions change under perturbations of $\lambda_{\Pi}$, $\varepsilon$, $\beta$, and any smoothing or stabilization choices used in the weight definition.

These procedures quantify uncertainty in the deployed nonlinear component predictor itself. If measured composites are later reported, they should be computed from each resampled component prediction by the same external composition matrix.

## 10. Implications of the Main Modeling Choices

### 10.1 Direct effluent-state parameterization

The surrogate is parameterized on the effluent component state rather than on the net change alone. This keeps the learned target aligned with the ASM state that the stoichiometric model actually constrains. Each component target is predicted directly from the shared feature vector through one row of $\Pi$. Dependence among the predicted components therefore enters through the common predictors and the shared projection step. This simplifies the architecture and removes the need for an additional stability condition, but it also means that any residual cross-component structure not captured by the features must be absorbed indirectly.

### 10.2 Embedded Weighted Projection in the Forward Model

The weighted projection layer makes conservation part of the forward map that the optimizer sees during training. This means the estimated coefficients are shaped by the conservation structure throughout learning. The same is true of the prediction-dependent weighting, which determines how the exact conservation correction is allocated across coordinates during coefficient estimation.

### 10.3 Direct evaluation in prediction but not in estimation

The model has an explicit component prediction formula but no closed-form parameter estimator. That distinction is central. Once the coefficients are known, each sample is evaluated by direct component prediction and weighted projection. But because the projection weights themselves depend on the learnable raw prediction, the global estimation problem is nonlinear and requires gradient-based optimization.

### 10.4 Lasso as a structural sparsity device

The second-order feature map can be high dimensional. Lasso regularization therefore does more than prevent overfitting. It expresses a modeling preference for sparse polynomial structure. The resulting model is easier to interpret because inactive feature blocks are explicitly shrunk toward zero. That said, when the full Kronecker basis is used, interpretability should be attached to active monomials or symmetrized coefficient summaries rather than to any one duplicated quadratic coordinate.

### 10.5 External reporting after component correction

Component prediction and conservation-preserving correction are both executed in component space, and measured outputs are computed only afterward. This keeps the physically meaningful decisions in the same space in which stoichiometry operates. It also decouples model training from reporting convention: the same trained component-space model can be post-processed into different measured composite outputs by different external composition matrices without redefining the fitted surrogate.

## 11. Limitations

WP-ICSOR is deliberately narrower than a full mechanistic reactor model. Its main limitations are the following.

1. It is steady-state in the quasi-steady-sample sense and does not represent temporal dynamics or path dependence.
2. It requires effluent ASM component-fraction targets for training, either directly from a simulator or from an upstream reconstruction step.
3. It enforces only the invariant relations encoded by the chosen stoichiometric basis and system boundary.
4. The embedded weighted projection preserves conservation exactly, but it does not impose the hard inequality constraint $c^* \ge 0$ and does not in general repair a negative raw component prediction.
5. The effect of the weighting depends on the chosen metric design, the stabilization floor $\varepsilon$, the sharpening exponent $\beta$, and the operating regime.
6. Different coefficient matrices can still induce similar deployed component predictions, especially when the second-order design is weakly excited or highly collinear.
7. The direct architecture can underfit if important inter-component dependence is not captured by the shared feature basis and projection.
8. The full Kronecker quadratic basis contains symmetric duplicate monomials unless it is compressed, so individual quadratic coefficients are not uniquely interpretable without additional symmetrization conventions.
9. The second-order feature basis can be statistically fragile if it is weakly excited or highly collinear.
10. Non-negative external measured composites are supported most naturally only when the composition matrix has non-negative rows and the component prediction itself is non-negative.
11. Exact closed-form inference formulas for the final deployed predictor are not available because the estimation problem is nonlinear and the predictor includes prediction-dependent weights.
12. A misspecified stoichiometric matrix or incorrect system boundary leads to a formally consistent projection on the wrong physical change space.
13. If the influent ASM component state is reconstructed from measured aggregate variables rather than observed directly, reconstruction error enters upstream of the surrogate and is not represented by the component-space loss alone.
14. The relation $c_{out} - c_{in} = \nu^T \xi$ is a reduced concentration-equivalent closure; flow-resolved dynamics, phase partitioning, and unmodeled transport mechanisms remain outside the present surrogate.

These limitations should be stated explicitly in any application. Doing so does not weaken the model. It defines the scope of its claims correctly.

## 12. Conclusion

Weighted-projection invariant-constrained second-order regression (WP-ICSOR) combines a partitioned second-order surrogate and an embedded weighted projection derived from stoichiometric invariants. The effluent ASM component targets are first predicted directly as

$$
\hat c = \Pi \phi(u, c_{in}),
$$

and the raw change is then projected onto the invariant-consistent change space with prediction-dependent weights that preserve conservation exactly and determine how the required correction is distributed across components. The native output of the model is the corrected ASM component-fraction vector itself.

The central methodological point is therefore twofold. First, the component targets are predicted directly in component space and supervised in that same space during training. Second, the physical structure is embedded in the learned model: conservation is exact by construction, the projection metric is specified explicitly, and the parameters are estimated end to end with Adam and lasso regularization. If measured composites are needed, they are computed only afterward as an external reporting calculation through the composition matrix. Under that reading, WP-ICSOR is an analytically structured steady-state surrogate for activated-sludge prediction that preserves stoichiometric structure in every forward pass and remains narrower in scope than a full dynamic mechanistic simulator. Its strongest hard claim is exact invariant preservation; any claim about non-negativity remains conditional on additional constraints not included in the baseline formulation.

## References

1. Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Publishing, 2000.
2. Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
3. Golub, G. H., and Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
4. Boyd, S., and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.
5. Tibshirani, R. Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267-288, 1996.
6. Kingma, D. P., and Ba, J. Adam: A Method for Stochastic Optimization. 3rd International Conference on Learning Representations, 2015.