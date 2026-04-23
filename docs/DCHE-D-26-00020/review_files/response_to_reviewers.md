# Response to Reviewers

## General Manuscript-Wide Revisions

Before responding point by point, we would like to clarify that the revised manuscript is not a line-by-line adjustment of the previous version, but a substantial redesign motivated by the reviewers' technical concerns and by several additional changes that were necessary to restore internal consistency, physical rigor, and reproducibility.

The most fundamental revision is the transition from CoBRE to ICSOR. In the previous manuscript, CoBRE was presented as a coupled bilinear surrogate intended to balance interpretability and predictive performance, but the reviewers correctly identified several issues concerning the physical interpretation of the bilinear terms, the lack of hard mass-conservation guarantees, the treatment of non-negativity, the ambiguity of the left-hand-side coupling structure, and the risk of overstating what the surrogate physically enforces. In response, the revised manuscript no longer presents the work as a bilinear CoBRE model. Instead, it introduces Invariant-Constrained Second-Order Regression (ICSOR), whose name was changed to reflect the actual scientific contribution of the revised article: the model is now centered on a second-order interpretable surrogate coupled to explicit invariant conservation and componentwise non-negativity enforcement. The new name therefore serves a substantive purpose rather than a cosmetic one. "Invariant-Constrained" identifies the central physical guarantee introduced in the revised manuscript, while "Second-Order Regression" more accurately describes the retained feature structure than the earlier bilinear label.

Closely related to that name change, the revised manuscript shifts the scientific objective of the article. The earlier version primarily emphasized interpretability and predictive benchmarking of CoBRE against standard machine-learning baselines. The revised manuscript broadens the objective to ask a different and more rigorous question: whether an interpretable surrogate can remain competitively accurate while also guaranteeing exact stoichiometric conservation and non-negativity at deployment. This was motivated directly by the reviewers' repeated concerns that interpretability claims are technically weak if the predicted states remain physically inadmissible.

The mathematical formulation was therefore rebuilt at the root level. Rather than retaining the earlier implicit CoBRE equation and only adjusting its wording, the revised manuscript now formulates the surrogate in ASM component space, introduces the null-space invariant operator, defines the coupled driver and coupling matrix explicitly, and describes deployment as a staged procedure consisting of direct coupled evaluation, affine invariant projection, and a final linear program when non-negativity remains active. This redesign was motivated by the need to replace soft or verbal physical interpretations with exact mathematical guarantees that can be stated, analyzed, and tested directly.

The modeling scope was also deliberately narrowed. The previous manuscript combined a standalone CSTR discussion with a more elaborate plant-wide five-reactor-plus-clarifier simulation workflow. Several reviewer comments exposed that this broader setup created avoidable ambiguity regarding hydrodynamics, washout, initialization, data validity, dataset construction, and the interpretation of clarifier results. The revised manuscript therefore focuses on a standalone steady-state CSTR benchmark. This narrowing was adopted to ensure that the paper answers one question cleanly and defensibly instead of treating multiple unit-operation settings with uneven methodological depth.

Another major revision is that the revised manuscript no longer uses QSDsan as the dataset-generation platform. This change was motivated partly by the reviewer comments on steady-state validity, initialization quality, and physical plausibility, and partly by a broader methodological need that was not fully captured in any single reviewer item. In the revised study, the steady-state solver, initialization rules, acceptance criteria, and invariant construction all need to be aligned tightly with the exact stoichiometric structure used later by ICSOR. That level of control required a custom benchmark-generation workflow rather than a more general simulation wrapper. The revised manuscript therefore adopts a custom steady-state sampling and acceptance pipeline so that initialization logic, residual tolerances, admissibility screening, and the null-space conservation operator are all fully consistent with one another.

The data representation was also changed in a fundamental way. The previous manuscript mixed composite indicators with selected biological fractions and relied heavily on reported composite targets such as BOD, COD, TN, TKN, TSS, and VSS. Reviewer comments correctly pointed out the resulting logical tension between sensor-oriented justification and the inclusion of internal biomass states, as well as the special ambiguity surrounding BOD. The revised manuscript resolves this by making ASM component space the native prediction space for every model. Reported composites are now obtained afterward through an external composition matrix, and the reported output set has been reduced to COD, TN, TP, and TSS. This change was motivated not only by the reviewer comments, but also by the recognition that both conservation and non-negativity are meaningful first in component space, not in an already collapsed reporting space.

The revised manuscript also replaces the earlier transformed-target non-negativity argument with an explicit hard-constraint strategy. In the previous version, non-negativity was discussed through logarithmic transformation and normalization. The reviewers correctly noted that this does not guarantee non-negative final physical predictions. The revised manuscript therefore removes that claim entirely and instead enforces non-negativity directly in the deployed component prediction. This was a necessary conceptual correction, not merely a local wording fix.

The introduction was expanded substantially beyond the specific wording comments raised by the reviewers. In addition to clarifying abbreviations, transitions, and objectives, the revised manuscript now contains a broader literature discussion on conservation-preserving and positivity-preserving scientific machine learning, explains why exact joint enforcement of both constraints is nontrivial, and positions the revised contribution relative to that broader literature rather than only to standard wastewater-surrogate papers. This broader framing was added because, once the article was re-centered on physical admissibility, the original introduction no longer gave an adequate research context.

The theoretical exposition was also expanded beyond the original reviewer requests. In addition to the new Nomenclature section, the revised manuscript now includes a clearer definition of the system boundary, explicit discussion of what is and is not claimed physically, a blockwise interpretation of the coefficient structure, pseudocode for training and deployment, and an explanation of why deterministic recursive coupled-QP estimation was adopted instead of the earlier stochastic training logic. These additions were motivated by the need to make the revised model reproducible and mathematically legible after the transition away from the earlier CoBRE training formulation.

The benchmarking protocol was likewise redesigned. The revised manuscript adopts a one-shot 80--20 benchmark together with a repeated dataset-size analysis based on repeated random train-test evaluations at each retained size. This change was motivated by two goals: first, to provide a cleaner comparison in the revised single-benchmark setting; and second, to make generalization, sample efficiency, runtime scaling, and physical-admissibility behavior visible in a way that one aggregate cross-validation table could not. The baseline tuning protocol was also strengthened through a common Optuna design with 100 trials per retained model and explicit reporting of the MLP architecture and final selected settings.

The revised manuscript further adds analyses that were not present in the previous version but became necessary after the redesign. These include physical-admissibility diagnostics reported directly on the component-space predictions, deployment-stage diagnostics showing how often the affine presolve alone is sufficient versus when the final LP is required, a more careful discussion of training-time cost as an offline calibration issue rather than an operational-time issue, and an interpretability section that explicitly distinguishes sparse second-order driver structure from the denser coupling structure. These additions were motivated by the need to support the revised claims with diagnostics that match the new formulation.

Finally, the revised manuscript now states its boundaries more explicitly than the earlier version. The article no longer suggests that the surrogate fully captures washout physics, fully enforces mechanistic feasibility, or replaces dynamic process simulation. Instead, it states more carefully that the present contribution is a steady-state, component-space, physically admissible surrogate within the sampled operating domain. This clarification was motivated both by the reviewers' concerns and by the broader redesign needed to make the revised manuscript technically defensible. Revision highlight in manuscript: blue.

## Reviewer 1

### R1-1
> Please define all abbreviations at first use (e.g., PLS, KNN, SVR).

Response: Thank you for this suggestion. Compared with the previous manuscript, the revised manuscript now introduces the benchmark families in expanded form at first mention, including Partial Least Squares, $k$-Nearest Neighbors, Support Vector Regression, and the multilayer perceptron benchmark. This revision is reflected in the Abstract and in the benchmark overview in the Introduction. Revision highlight in manuscript: blue.

### R1-2
> Lines 31-32: The sentence "...was trained on this dataset, then benchmarked against PLS, KNN, SVR..." should be revised. Please rewrite a clearer version.

Response: Thank you. The benchmark sentence has been fully rewritten as part of the new Abstract. Compared with the previous version's compressed benchmark list, the revised Abstract now states the comparison in a clearer form by identifying the full benchmark set and the specific role of ICSOR within that comparison. The revision is located in the Abstract. Revision highlight in manuscript: blue.

### R1-3
> Lines 8-11: "In response to this complexity, research has increasingly focused on developing surrogate models to approximate the behavior of these systems (Bhosekar & Ierapetritou, 2018; Durkin et al., 2024; Tsochatzidi et al., 2025)." Since this starts a new paragraph, avoid using ambiguous phrases like "this complexity" and "these systems." Please explicitly restate what specific complexity and which systems you are referring to so the reader doesn't lose the context.

Response: Thank you for noting the ambiguity. In the previous manuscript, this transition relied on shorthand references. In the revised manuscript, the opening transition now explicitly states that the issue is the computational and mathematical complexity of non-linear Activated Sludge Models and that the surrogate models are intended for biological wastewater treatment systems. This revision appears in the second paragraph of the Introduction. Revision highlight in manuscript: blue.

### R1-4
> Please avoid using "On the other hand" twice in such close proximity. Furthermore, "On the other hand" implies a direct contrast, but here you are listing a progression of alternative methods. Consider using transitions that show alternatives instead.

Response: Thank you. This wording issue has been removed in the revised manuscript because the Introduction was rewritten and the earlier repeated transition phrases are no longer used. The revised literature progression is now presented as a structured movement from linear and polynomial surrogates to modern machine-learning baselines and then to the proposed constrained framework. The changes are located throughout the Introduction. Revision highlight in manuscript: blue.

### R1-5
> P. 3, lines 19-32: Please avoid using "On the other hand" twice in such close proximity. Furthermore, "On the other hand" implies a direct contrast, but here you are listing a progression of alternative methods. Consider using transitions that show alternatives or progression instead.

Response: Thank you. This duplicated stylistic issue has likewise been resolved through the rewritten Introduction. The revised manuscript no longer uses the repeated "On the other hand" phrasing in the affected literature-review sequence. The revision is located in the Introduction. Revision highlight in manuscript: blue.

### R1-6
> P. 4, line 6: Please avoid first-person ("we"). Please rephrase to use passive voice or third person.

Response: Thank you. Compared with the previous manuscript, the revised manuscript removes the earlier first-person phrasing in the corresponding theoretical discussion and adopts an impersonal scientific style throughout the Introduction and Theory and Calculation sections. The relevant revision is visible in the Introduction and throughout Section 2. Revision highlight in manuscript: blue.

### R1-7
> While the advantages of PLS are well-explained, the transition to the Machine Learning paragraph is somewhat abrupt. Please explicitly state the fundamental limitations of traditional PLS to clearly justify the shift towards ML and your proposed CoBRE framework.

Response: Thank you for this point. In the revised manuscript, the transition has been rebuilt so that the limitations of linear methods are stated explicitly before introducing broader ML baselines and then the proposed ICSOR framework. In particular, the Introduction now explains that linear methods such as PLS struggle to capture severe non-linearities without extensive feature engineering, and it then motivates the need for a constrained, interpretable alternative. This revision is located in the second through seventh paragraphs of the Introduction. Revision highlight in manuscript: blue.

### R1-8
> The text strongly highlights CoBRE's "parsimony and 'interpretability" as its main advantages over black-box ML models. However, the final paragraph states that the primary objective is to compare "predictive performance." To fully support your premise, please expand the stated objectives.

Response: Thank you. Compared with the previous manuscript, the revised objectives now extend beyond predictive accuracy alone. The Introduction and Abstract state that the revised study evaluates accuracy together with exact mass conservation, componentwise non-negativity, interpretability, and generalization behavior. These revisions are located in the Abstract, the final two paragraphs of the Introduction, and the evaluation criteria described in Section 3.5. Revision highlight in manuscript: blue.

### R1-9
> Briefly noting how bilinear terms naturally approximate the multiplicative interactions inherent in the biological process (such as biomass-substrate coupling in Monod kinetics) would provide a stronger phenomenological justification.

Response: Thank you. The revised manuscript no longer relies on the original CoBRE bilinear formulation, but the underlying phenomenological motivation has been retained in updated form. The new framework uses a partitioned second-order feature map and explicitly explains how operating-condition, influent, and interaction terms are tied to biologically meaningful process effects, including nitrification, denitrification, phosphorus storage, and loading-interaction behavior. This new rationale is presented in Sections 2.3 and 2.5. Revision highlight in manuscript: blue.

### R1-10
> Please include a separate Nomenclature section listing all variables and symbols, rather than introducing them only within the Theory and Calculation section. At present, some variables, such as K_L, a, do not appear in the current list (and others).

Response: Thank you. This has been addressed directly. The revised manuscript now contains a separate Nomenclature section immediately after the front matter, where the principal variables, operators, matrices, and symbols used in the revised ICSOR formulation are listed explicitly. The revision is located in the standalone Nomenclature section. Revision highlight in manuscript: teal.

### R1-11
> The final paragraph is repetitive, reiterating the same concepts as the end of the preceding paragraph (e.g., combinatorial explosion, overfitting, parsimony, and bilinear terms). Please merge these two paragraphs to state your justification once and concisely.

Response: Thank you. The repetitive passage in the previous manuscript is no longer present because the theoretical exposition has been rewritten around the ICSOR formulation. The revised Section 2 now presents the model construction, deployment logic, interpretability, and estimation strategy in separate non-redundant subsections. The relevant revisions are located throughout Section 2. Revision highlight in manuscript: blue.

### R1-12
> Review this section to eliminate first-person pronouns. Please change phrases to passive voice or third-person constructs.

Response: Thank you. This stylistic revision has been implemented in the rewritten theory section. Compared with the previous manuscript, the revised Section 2 consistently uses an impersonal technical style instead of first-person narration. The change is visible throughout Section 2. Revision highlight in manuscript: blue.

### R1-13
> Regarding Equation 4: While the bilinear term (y_k * x_m) introduces useful hydrodynamic modulation, be cautious about explicitly claiming it models "washout". Physically, washout is driven by hydraulic retention time (V/Q), making the system highly non-linear with respect to the inverse of the flow rate (1/x_m). A direct multiplication by flow rate (x_m) is a local linear approximation that will likely fail to capture true asymptotic washout behavior at extreme flows. Consider softening this claim to state that the term "approximates the effect of varying flow rates" rather than fully capturing washout mechanics.

Response: Thank you for this important clarification. This concern applied to the earlier CoBRE interpretation. In the revised manuscript, that claim has been removed because the prior bilinear washout discussion is no longer used; instead, the study is reformulated around a standalone CSTR benchmark with hydraulic retention time as an explicit operating variable and with physically admissible steady states enforced through invariant and non-negativity constraints rather than a verbal washout claim. The corresponding revisions are located in Sections 2.1-2.4 and Section 3.1. Revision highlight in manuscript: blue.

### R1-14
> There is a mathematical contradiction in the text describing Equation 5. You state that dependent variables are on the left-hand side and independent forcing functions are on the right. However, the term on the left contains x_m, which is an independent function (influent flow rate). Please rephrase this to clarify that the left-hand side groups all terms containing the state variables (y), rather than claiming a strict separation of dependent and independent variables.

Response: Thank you. This inconsistency has been removed in the revised manuscript because the old CoBRE Equation 5 is no longer part of the paper. The revised mathematical formulation now defines the coupled system through the matrices $R = I_F - \Gamma$, the latent driver $r(u, c_{in}) = B\phi(u, c_{in})$, and the staged deployment correction, without the earlier left-hand-side versus right-hand-side wording. The replacement formulation is given in Sections 2.3 and 2.4. Revision highlight in manuscript: blue.

### R1-15
> Since Equation 5 is implicit and couples the state variable y_k with other state variables y_p, predicting the outputs requires solving a system of simultaneous algebraic equations. For technical rigor, it would be beneficial to briefly mention that the application of the CoBRE for prediction involves a matrix inversion (or simultaneous resolution step) rather than a direct feedforward calculation.

Response: Thank you. This point is explicitly addressed in the revised manuscript. The deployment subsection now states the coupled relation, defines $\widehat{R} = I_F - \widehat{\Gamma}$, and shows that deployment begins from the raw coupled state $c_{raw} = \widehat{R}^{-1} d_*$. The manuscript then explains the subsequent affine projection and LP correction steps. These revisions are located in Section 2.4, with additional discussion of conditioning and invertibility in Section 2.6. Revision highlight in manuscript: blue.

### R1-16
> In section 3.1, you mention running the simulation for 180 days to reach a "stable dynamic steady state." If the influent conditions and parameters generated by the LHS are constant for each run, the system reaches a standard "steady state." The term "dynamic steady state" is usually reserved for systems with periodic forcing (e.g. influent variations). Please clarify if the inputs were dynamically varied over time or held constant to extract steady-state data.

Response: Thank you. The revised manuscript now resolves this terminology issue by reformulating the study around a standalone steady-state CSTR benchmark. Section 3.2 explicitly states that the accepted dataset consists of steady states obtained by solving the steady-state ASM balance equations, with dynamic relaxation used only as a fallback mechanism to supply an improved steady-state initialization. This clarification is located in Section 3.2. Revision highlight in manuscript: orange.

### R1-17
> By randomly assigning volumes and flow rates via LHS, there is a risk of generating mathematically valid but physically absurd scenarios (e.g., extremely low hydraulic retention times leading to total biomass washout in all 1000 runs). Could you add a brief sentence reassuring the reader that the LHS bounds in Table 1 were constrained to ensure a biologically viable sludge age (SRT) for the majority of the simulations?

Response: Thank you. Compared with the previous manuscript's plant-wide randomization, the revised manuscript now defines a physically bounded standalone CSTR domain in terms of hydraulic retention time, aeration, and influent ASM-component ranges. It further states that sampling continued until 10,000 valid steady states had been accepted and that only converged steady states satisfying the residual-acceptance threshold were retained, so the final benchmark contains biologically viable operating points only. These revisions are located in Sections 3.1 and 3.2. Revision highlight in manuscript: orange.

### R1-18
> Looking at Table 1, the independent LHS sampling of variables could lead to physically non-viable systems. For instance, pairing the maximum influent flow (Qinf = 20,000 m³/d) with the minimum total volume (1,500 m³) results in a Hydraulic Retention Time (HRT) of under 2 hours. Coupled with a high wastage rate, this will lead to severe biological washout (especially for nitrifiers) in a significant portion of the 1,000 runs. Please clarify in the text if any joint constraints were applied during the LHS generation to guarantee a biologically viable Sludge Retention Time (SRT), or if these washout scenarios were intentionally kept in the dataset to train the models on system failures.

Response: Thank you. This concern has been addressed by redesigning the benchmark domain and acceptance protocol. The revised manuscript no longer relies on independently sampled plant-wide volumes and flows; instead, it uses a bounded steady-state CSTR design in which sampling continues until valid states are accepted, infeasible draws can be replaced, and only biologically viable steady states are retained in the final dataset. The relevant revisions are in Section 3.1 and especially Section 3.2. Revision highlight in manuscript: orange.

### R1-19
> Table 2: Please review the values for the particulate components in CSTR 2. The inert particulate organic matter (XI) drops abruptly to 0 mg/L, and heterotrophic biomass (XH) drops to 207 mg/L, whereas CSTR 1 and CSTR 3 have values exceeding 2200 mg/L and 3700 mg/L, respectively. While initial conditions are ultimately washed out after a 180-day dynamic simulation to reach a steady state, inputting XI = 0 in the middle of a continuous reactor train appears to be a typographical error or an anomalous initialization.

Response: Thank you for identifying that inconsistency. The problematic plant-wide initialization table from the previous manuscript has been removed. In the revised manuscript, initialization is now described as a reproducible set of solver heuristics for the standalone CSTR benchmark, summarized in the initialization table and accompanying text in Section 3.2. This revision replaces the earlier unit-by-unit initial-condition table altogether. Revision highlight in manuscript: orange.

### R1-20
> It is excellent that you filtered out runs resulting in physical infeasibility (washout) or numerical divergence. However, it is crucial to report the final size of the dataset. Because the LHS sampling was completely random, a large portion of the 1000 initial runs might have resulted in washout. Please state exactly how many valid runs remained for training and testing.

Response: Thank you. The revised manuscript now states this explicitly. Section 3.2 reports that sampling continued until 10,000 valid steady states had been accepted, and Section 3.5 then states that the one-shot benchmark used an 80--20 split, yielding 8,000 training rows and 2,000 test rows. This is a direct clarification relative to the previous manuscript. Revision highlight in manuscript: orange.

### R1-21
> In section 3.2.1, you state that the surrogate model learns the input-output mapping based on local state variables V, KL, a, and influent concentration. However, hydrodynamics dictate that the flow rate (Q) or Hydraulic Retention Time is critical for determining the effluent. Please clarify in the text if the local flow rate traversing each specific CSTR was also explicitly included as an input feature for the ML models and the CoBRE.

Response: Thank you. The revised manuscript addresses this directly by redefining the operating inputs for the benchmark. The model input vector now explicitly includes hydraulic retention time and aeration level, together with the 20 influent ASM component concentrations, for all model families. This clarification appears in Sections 3.1 and 3.3. Revision highlight in manuscript: blue.

### R1-22
> In Section 2.2, the decision to use composite indicators (like COD and TN) rather than granular ASM states was explicitly justified by the 'practical necessity of aligning with sensor capabilities and regulatory standards.' However, in this section, you include specific biological fractions (XH, XPAO, etc.) as target variables. Since these specific biomass fractions cannot be directly measured by standard online plant sensors, this creates a logical contradiction. Please clarify why these specific internal states are included alongside the measurable composites, perhaps by noting that they are retained for fundamental process analysis rather than purely for sensor-based predictive control.

Response: Thank you. This logical inconsistency has been resolved by changing the modeling scope. The revised manuscript states that ICSOR is formulated natively in ASM component space and that measured composites are obtained only afterward through an external composition matrix. It also explains that the scientific object is the component-level effluent state itself and that composite outputs are reporting quantities derived from that state. These clarifications are located in Sections 2.1, 2.5, and 3.1-3.3. Revision highlight in manuscript: blue.

### R1-23
> BOD is an empirically defined parameter (typically a 5-day assay) rather than a strict state variable, making its calculation from ASM fractions notoriously sensitive to assumed decay rates and biodegradable fractions. While you mention using QSDsan's default functions, it would add rigor to briefly state or cite the specific mapping used to convert ASM2d components.

Response: Thank you. This issue no longer applies to the revised manuscript because BOD is no longer part of the reported target space. The revised study reports COD, TN, TP, and TSS only, and the manuscript explains that these reported outputs are computed through a common external composition matrix applied to the predicted ASM component states. The relevant revisions are in Sections 2.1, 2.5, and 3.1. Revision highlight in manuscript: blue.

### R1-24
> 3.4 subsection: While 20 trials might be sufficient for simpler models like PLS or KNN, may be inadequate for ANNs, where the search space is massive. If 20 was a strict computational limit, consider justifying it explicitly.

Response: Thank you. This concern has been addressed in the revised benchmarking protocol. Section 3.4 now states that all retained models were tuned with Optuna under a common design using 100 trials per model, no pruning, and no wall-clock timeout, and the selected settings are reported in the Optuna design and model-specific hyperparameter tables. These revisions are located in Section 3.4 and the associated Optuna tables. Revision highlight in manuscript: forest green.

### R1-25
> The term ANN is a term that describes an entire field rather than a specific architecture. To ensure reproducibility, please specify the exact architecture used in the benchmark (MLP or LSTM network).

Response: Thank you. The revised manuscript now specifies the ANN benchmark exactly. It identifies the retained neural baseline as a fully connected multilayer perceptron, gives the hidden-layer widths, activation, solver, and training settings, and also includes a dedicated architecture figure. These details are located in Section 3.4, the MLP architecture figure, and the MLP hyperparameter table. Revision highlight in manuscript: forest green in the Section 3.4 text and emerald green on the dedicated MLP architecture figure.

### R1-26
> I think, there is a mathematical oversight in the claim that yk=ln(Ck+1) "strictly enforces non-negativity in predictions". Because you subsequently apply a Z-score normalization, the target variable yk will frequently take negative values (representing data points below the mean). If the model outputs a negative prediction (ypred<0) and you apply the inverse transformation Ck, the resulting physical concentration will be strictly negative (since e^x<1 for any x<0). To truly enforce non-negativity, you must clarify if a post-processing bounds check (applying a max(0,Ck) function) was used on the final inverse-transformed predictions. Without this, the claim is mathematically incorrect.

Response: Thank you for this important correction. The revised manuscript addresses the issue at the formulation level rather than through transformed-target rhetoric. ICSOR is now trained and deployed in physical component space, and non-negativity is enforced explicitly at deployment through the affine-presolve-plus-LP correction described in Section 2.4. The physical-admissibility results in Section 4.3 then report 0.0\% non-negativity violations for ICSOR on the test set. Revision highlight in manuscript: crimson.

### R1-27
> In Equation 10, the prediction requires the inverse matrix A. While you correctly note that LASSO regularization helps prevent ill-conditioning of the converged model, during the initial epochs of AdamW optimization, the weights in and could fluctuate wildly, potentially causing determinant to approach zero and crashing the solver. Please briefly clarify if any numerical safeguards.

Response: Thank you. The revised manuscript now provides explicit numerical safeguards. Section 2.6 explains that the recursive coupled-QP estimator constrains the coupling update, fixes the diagonal of $\Gamma$, imposes box bounds, and rejects candidates that make $R = I_F - \Gamma$ poorly conditioned beyond the prescribed threshold. The deployment subsection also makes the inversion step explicit. These safeguards are described in Sections 2.4 and 2.6. Revision highlight in manuscript: brown.

### R1-28
> In Table 4 (Clarifier column), the standard deviations for XGBoost and SVR training times are reported as exactly as 0.0000. In a 10-fold cross-validation procedure, observing absolutely zero variance across 10 distinct training loops is highly improbable. Please verify the code used to log these specific times or check for rounding/precision truncation errors.

Response: Thank you. The specific runtime table that prompted this concern is no longer part of the revised manuscript. The revised manuscript now reports training-time behavior through a repeated dataset-size study based on repeated random train-test evaluations at each retained size, summarized by runtime curves and accompanying discussion. This replacement appears in Section 4.2 and the runtime figure. Revision highlight in manuscript: violet.

### R1-29
> In Tables 7 and 8, the final column is labeled: Sparsity Ratio (%), but the numbers reflect the percentage of retained (non-zero) terms. The correct mathematical term for the proportion of non-zero elements is "Density". If you want to report 'Sparsity', the value should be the proportion of zeros. Please rename this column.

Response: Thank you. This nomenclature issue has been corrected in the revised manuscript. The interpretability table no longer uses the misleading "Sparsity Ratio" label; instead, it reports retained coefficients and "\% Retained," which matches the quantity being shown. The revision is located in the coefficient-count table in Section 4.4. Revision highlight in manuscript: purple.

### R1-30
> You claim the model generalizes even when retained coefficients (p) exceed training samples (N). Statistically, LASSO only prevents overfitting in p>N scenarios if the model is highly sparse. However, Tables 7 and 8 show the model is quite dense (>90% retention in B, Γ, and Λ). A dense model with p>N lacks residual degrees of freedom and would be expected to overfit. Please clarify this contradiction: how are the effective degrees of freedom calculated here?

Response: Thank you. The revised manuscript addresses this concern by changing both the model and the benchmark scale. In the revised study, the one-shot benchmark uses 8,000 training samples, and the reported COD interpretability table shows 847 candidate coefficients with 460 retained above the export threshold, so the revised comparison no longer relies on a retained-coefficient regime exceeding the training-sample count. The generalization discussion is then based on dataset-size resampling results and explicit train-validation RMSE gaps rather than on the earlier claim. These revisions are located in Sections 3.5, 4.2, and 4.4. Revision highlight in manuscript: violet.

## Reviewer 2

### R2-1
> Complexity of "Interpretability": The model retains hundreds of active coefficients (864 for the CSTR Θ tensor alone). If a model requires complex heatmaps to be understood, its claim of being "intuitive" or "algebraically interpretable" is diminished compared to simpler linear baselines.

Response: Thank you. The revised manuscript now addresses this concern directly rather than implying uniform sparsity. Section 4.4 explicitly states that interpretability does not arise from a uniformly sparse model, but from a structured separation between a sparse quadratic driver and a comparatively dense coupling matrix. The coefficient-count table and the accompanying discussion in Section 4.4 were added specifically to make that distinction clear. Revision highlight in manuscript: purple.

### R2-2
> The most significant limitation is that the CoBRE framework is a grey-box statistical fit that does not strictly enforce mass conservation.

Response: Thank you. This concern motivated a central redesign of the manuscript. Compared with the previous version, the revised manuscript now develops ICSOR specifically to enforce exact stoichiometric mass conservation together with component non-negativity, and this requirement is built into both the formulation and the deployment rule. The key revisions are in Sections 2.2-2.4, with empirical confirmation in Section 4.3. Revision highlight in manuscript: crimson.

### R2-3
> Lack of Constraints: Unlike the Activated Sludge Models (ASM) it approximates, CoBRE does not incorporate hard physical constraints (In−Out−Reaction=0).

Response: Thank you. This point has been addressed directly. The revised manuscript introduces the invariant matrix construction, states the conservation relation $Ac_{out} = Ac_{in}$, and enforces the corresponding equality constraints during deployment together with non-negativity constraints. The relevant revisions are located in Sections 2.2, 2.4, and 3.3. Revision highlight in manuscript: crimson.

### R2-4
> Under-tuned Benchmarks: Using only 20 Optuna TPE trials for complex models like ANN, XGBoost, and LightGBM is insufficient. These algorithms typically require hundreds of iterations to reach peak performance; there is a high risk that the "strong black-box baselines" were under-tuned, artificially favoring CoBRE.

Response: Thank you. This issue has been addressed by revising the benchmark-tuning protocol. Section 3.4 now states that all retained models were tuned with a common Optuna design using 100 trials per model, no pruning, and no timeout, and the selected model-specific settings are reported in the corresponding tables. This is a direct change from the previous benchmark design. Revision highlight in manuscript: forest green.

### R2-5
> The Parameter-to-Data Paradox: In the clarifier study, the number of retained coefficients often exceeds the number of training samples. While the authors argue the "coupled structure" suppresses variance, this remains a significant risk for overfitting that requires more transparent validation.

Response: Thank you. The revised manuscript no longer uses the earlier clarifier study as the main benchmark and instead evaluates a standalone CSTR benchmark with 8,000 training samples in the fixed split and an additional repeated dataset-size analysis based on repeated random train-test evaluations at each retained size. The current generalization claims are therefore tied to explicit train-validation RMSE gaps, ranking stability, and learning-curve behavior rather than the earlier argument. These revisions are located in Sections 3.5 and 4.2. Revision highlight in manuscript: violet.

### R2-6
> Training Time Context: While CoBRE is 3x faster than an ANN, a difference of ~90 seconds in offline training is negligible in the context of wastewater processes that operate on much longer time scales.

Response: Thank you. The revised discussion now places runtime in a more appropriate offline-training context. Section 4.2 explicitly states that ICSOR is slower than the fastest retained methods but remains practical for offline surrogate fitting, and the runtime comparison is framed as a computational tradeoff rather than as a decisive operational limitation. This revision appears in the runtime discussion in Section 4.2. Revision highlight in manuscript: violet.

