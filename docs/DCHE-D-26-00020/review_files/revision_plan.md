# Revision Plan for Outstanding Reviewer Items

## Reviewer 1

### R1-31
> Figure 3: Please enlarge the legend and all axis/tick labels, as the figure is currently difficult to read. Figure 5: Same comments.

Revision plan: This item is not yet evidenced by the current manuscript text because the figure typography is controlled in the exported figure assets rather than in the surrounding prose. The revision will be implemented as follows.

1. Re-export the learning-curve, runtime, and interpretability figures with larger tick labels, axis labels, legend fonts, and annotation sizes using a single shared plotting style.
2. Verify readability at the final single-column journal width used by the current Elsevier template, not only at notebook or screen scale.
3. Update the corresponding figure files referenced in the manuscript and, if needed, slightly reduce caption length so the figures can occupy more visual width.
4. Rebuild the PDF and inspect the final rendered pages to confirm that the revised figures remain legible in print-scale viewing.

### R1-32
> Please adjust the significant figures in the text and tables. Reporting a retention percentage as "72.001% (1.4858)" or a term count standard deviation as "67.1107" implies an unrealistic level of precision.

Revision plan: This issue remains in the current draft. Several tables and narrative sentences still report more precision than the benchmark can justify, especially in the coefficient-retention table and some runtime and accuracy summaries.

1. Define a manuscript-wide precision policy before editing the tables. A practical rule is four decimals for primary predictive metrics, one decimal for percentages unless the quantity is near zero, and integer counts for retained coefficients.
2. Apply that policy consistently to the fixed-split benchmark table, per-target RMSE table, repeated-size summary table, physical-admissibility table, and the coefficient-retention table in Section 4.4.
3. Revise nearby prose so that quoted numerical comparisons match the new rounded values and do not preserve superseded precision from the current draft.
4. Rebuild the manuscript and check that the revised rounding does not create apparent ties that would require rewording rank claims.

### R1-33
> Asymmetry in the Θ interaction tensors (Figure 5). The text interprets individual cells of the Θ heatmaps (Figures 5b-5e) as specific physical synergies or inhibitions between inputs xi and xj. However, since multiplication is commutative, the true mathematical weight of the interaction is the sum of the symmetric elements. The heatmaps presented are visibly asymmetric (e.g., in Fig 5b, the Influent_COD/Influent_BOD intersection is deep blue on one side of the diagonal and light gray on the other). This indicates the optimizer randomly distributed the weights between the upper and lower triangular matrices. Interpreting a single cell in an asymmetric matrix representing commutative products is mathematically invalid. The model code must either constrain Θ to be symmetric during training, or the visualization must plot the symmetric sum: (Θ+ΘT)/2 before any physical interpretation is claimed.

Revision plan: The current manuscript still presents a $\Theta_{cc}$ heatmap and interprets pairwise curvature, but it does not yet state whether the plotted matrix is symmetrized or whether the parameterization itself enforces symmetry. This remains an open interpretability issue.

1. Inspect the exported $\Theta_{cc}$ array used for Figure 4 and determine whether ordered Kronecker features allow separate coefficients for $(i,j)$ and $(j,i)$.
2. If asymmetry is present, revise the visualization pipeline so the reported heatmap uses the symmetric interaction summary $\tfrac{1}{2}(\Theta_{cc} + \Theta_{cc}^T)$ before interpretation.
3. Add a brief methodological note in Section 2.5 or Section 4.4 explaining that commutative second-order interaction effects are interpreted from the symmetrized tensor, not from a single ordered entry.
4. Recheck the interpretability narrative so that any quoted pairwise effects refer to the symmetrized interaction magnitude and sign.

## Reviewer 2

### R2-1
> While the manuscript claims a "novel" bridge between black-box and mechanistic models, it is unclear how this provides a practical advantage over standard multivariate methods.
>
> Redundancy with PLS/PCA: In wastewater treatment, Partial Least Squares (PLS) already provides interpretable loading weights that identify key influent drivers. The authors must explicitly state what CoBRE's interaction tensor (Θ) reveals to a process engineer that a standard PLS loading plot or an ANN sensitivity analysis does not.

Revision plan: The revised manuscript improves the interpretability discussion, but it still does not make the comparison against PLS loading plots and ANN sensitivity analyses explicit enough. That practical contrast should be stated directly in the Introduction and revisited in the interpretability discussion.

1. Add one short paragraph to the end of the Introduction explaining what the ICSOR block structure provides beyond PLS. The comparison should state that PLS loadings describe linear latent directions, whereas ICSOR separates first-order operating effects, first-order influent effects, within-block curvature, cross-block interactions, and cross-component redistribution in the predicted ASM state.
2. Add one short paragraph in Section 4.4 comparing the interpretive object in ICSOR against ANN sensitivity analysis. The comparison should emphasize that ANN sensitivities are local response diagnostics, while the ICSOR coefficients are explicit global model parameters attached to named physical variable blocks.
3. Tie the comparison to one concrete example from the COD analysis so the claimed practical advantage is demonstrated rather than only asserted.

### R2-2
> Physical Inconsistency: The authors should report the Mass Balance Closure Error for Solids, COD, Nitrogen and Phosphorus across the test set. Without this, the model may predict effluent concentrations that are physically impossible (e.g., exceeding influent loads), rendering its "interpretability" technically hollow.

Revision plan: The current manuscript reports invariant-violation frequency, which is a strong physical-admissibility diagnostic, but it does not yet break the conservation outcome into named wastewater-material balances such as COD, nitrogen, phosphorus, and solids. Adding that breakdown would make the physical interpretation more transparent to process engineers.

1. Map the invariant basis used in $A$ to the wastewater-material balances that are meaningful for readers, or derive an auxiliary reporting matrix that expresses COD, nitrogen, phosphorus, and solids closure directly from the component state.
2. Compute per-sample closure errors for those reported balances on the final test predictions for ICSOR and the unconstrained baselines.
3. Add a compact results table or supplementary table reporting at least mean, median, and maximum closure error by balance, together with the violation frequency already reported.
4. Update Section 4.3 so the physical-admissibility discussion references both the invariant-level guarantee and the named balance-level closure results.

### R2-3
> Mathematical Implementation: Please clarify the relationship between l1_lambda and λ reg in Equation (11) and define the specific thresholding mechanism used to prune coefficients to zero.

Revision plan: The old CoBRE-specific $l1$ notation no longer appears in the revised manuscript, so that part of the comment is obsolete. However, the second part remains relevant because Section 4.4 still refers to an export threshold applied in the comparison notebook without defining the threshold inside the manuscript.

1. Add one sentence in Section 4.4 stating the exact numerical threshold used to classify a fitted coefficient as retained for reporting purposes.
2. Clarify whether this threshold is only a reporting/export rule or whether it also affects training, model selection, or deployment.
3. If the threshold depends on target-specific scaling, define that scaling explicitly so the retained-coefficient table can be reproduced from the manuscript alone.

### R2-4
> Coefficient Stability: Are the heatmaps in Figure 5 representative of a single fold or an average across all 10 folds?. For a model to be considered "physically interpretable," the learned coefficients must remain stable across different data partitions.

Revision plan: This point is not yet addressed in the current draft. The present interpretability figure appears to correspond to a single fitted model, and the manuscript does not yet quantify coefficient stability across repeated resamples.

1. Decide whether the main interpretability figure should represent a single authoritative fit or a stability summary across repeated fits.
2. If a single fit is retained in the main text, add a sentence stating that explicitly and provide a supplementary stability analysis over repeated resamples for the most important coefficients or coefficient blocks.
3. A practical stability diagnostic would report the median and interquartile range of the largest-magnitude $\Theta_{cc}$ and $\Gamma$ coefficients across the repeated dataset-size or repeated full-size refits.
4. Update Section 4.4 so physical interpretation is limited to coefficients that remain sign-stable and magnitude-stable across resamples.

