# DCHE-D-26-00020 Submission Guideline

This folder is prepared for Editorial Manager (EM) as a flat LaTeX submission package. Zip the contents of this folder, not a parent directory with extra files.

## Submission Rules Applied Here

- All submission files must stay at one folder level. No subfolders are allowed in the zip.
- The manuscript must not depend on files outside this folder.
- Do not include local build byproducts such as `.aux`, `.log`, `.out`, `.abs`, or an auto-built manuscript PDF with the same basename as the `.tex` file.
- Filenames should contain only one period before the extension and avoid special characters.
- Upload LaTeX source files as `Manuscript` items and graphics as `Figure` items if you upload files individually instead of as a zip.

## Files That Should Be In This Folder Before Zipping

- `manuscript.tex`
- `cas-sc.cls`
- `cas-common.sty`
- Any additional `.sty`, `.cls`, `.bst`, `.bib`, or `.bbl` files actually referenced by the manuscript
- Any figure files actually referenced by the manuscript, placed directly in this folder

## Files That Should Not Be In The Submission Zip

- `manuscript.pdf`
- `manuscript.aux`
- `manuscript.log`
- `manuscript.out`
- `manuscript.abs`
- Any cached, temporary, or editor-generated files

## Local PDF Output Rule

When generating a local PDF from the TeX source, do not leave the generated PDF or other LaTeX build artifacts inside this submission folder.

- Write all generated LaTeX artifacts to a subfolder under `docs/latex_pdfs` that matches the source folder name.
- Name the generated PDF after the submission folder.
- For this submission, the artifact folder should be `docs/latex_pdfs/DCHE-D-26-00020` and the output PDF should be `DCHE-D-26-00020.pdf` inside that folder.

This keeps the submission folder clean and avoids including local build output in the zip package.

## Manuscript-Specific Notes

- The manuscript has been set to `nologo` mode so it does not require Elsevier thumbnail assets from an external `thumbnails/` folder.
- The manuscript no longer uses an external `\graphicspath` that points outside this submission folder.
- The custom footer override that removes the Elsevier preprint line is kept inside the manuscript source.
- Author-identifying metadata has already been removed for anonymous submission.

## Complete Asset Generation Guide

This section defines how to generate every table and figure that appears in `manuscript.tex`.

The goal is not only to reproduce the final manuscript assets, but also to make the provenance of every reported value explicit. Use this section as the authoritative checklist when replacing the illustrative draft values with final benchmark exports.

### Common Data Contract For All Results Assets

All result tables and figures must be generated from one consistent benchmark contract.

- Use the same accepted 10,000-sample ASM2d-TSN dataset described in the methodology.
- Use the same 22 physical inputs for every model: `HRT`, `Aeration`, and the 20 influent ASM component concentrations.
- Use the same 20-component ASM effluent targets during model fitting.
- Convert final component predictions to reported composites only through the same external composition map `I_comp`.
- Keep the final selected Optuna hyperparameters fixed during the repeated dataset-size study.
- Treat ICSOR differently only where the manuscript already does: it is trained and deployed in physical component coordinates with its invariant and non-negativity logic preserved.

### Common Metric Definitions For All Results Tables And Figures

Unless a subsection below states otherwise, compute all reported metrics from the final reported composite outputs `COD`, `TN`, `TP`, and `TSS`.

For a target with observations `y_i` and predictions `\hat{y}_i` over `n` samples:

- Mean squared error:

	$$
	\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
	$$

- Root mean squared error:

	$$
	\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
	$$

- Mean absolute error:

	$$
	\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
	$$

- Mean absolute percentage error:

	$$
	\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
	$$

- Coefficient of determination:

	$$
	R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
	$$

For aggregate fixed-split reporting across the four composites, use the same aggregation rule across all models. The manuscript draft currently assumes one shared aggregate metric space; preserve that exact aggregation rule when exporting the final values.

### Source Categories

Every manuscript asset belongs to one of three categories.

- Static design table or figure: copied from the methodological setup and not recalculated from benchmark reruns.
- Study-configuration table: exported from final model configurations or Optuna study summaries.
- Benchmark-results asset: recalculated from the final fixed-split benchmark, repeated dataset-size sweep, admissibility diagnostics, or coefficient-analysis outputs.

When regenerating the manuscript, update the benchmark-results assets first, then update any text in the outline that references those values.

## Asset-By-Asset Instructions

This subsection covers every table and figure currently referenced in the manuscript.

### Table `tab:simulation_domain`

Purpose:
Define the 22-dimensional Latin hypercube sampling envelope used to generate the benchmark.

Source:

- The sampling-domain configuration used by the ASM2d-TSN simulation workflow.
- The final values must match the actual lower and upper bounds used to generate the accepted dataset.

How to generate:

- Export the lower and upper bounds for the two operating variables and the 20 influent ASM component variables.
- Preserve the variable order used in the manuscript: operating variables first, then soluble influent components, then particulate influent components.
- Do not infer or smooth bounds from the accepted dataset; report the configured design ranges.

What to verify:

- The table contains exactly 22 sampled inputs.
- Units match the state definitions used by the simulator and the manuscript.
- The reported bounds agree with the run that produced the final accepted dataset.

### Table `tab:initial_conditions`

Purpose:
Document the deterministic heuristics used to initialize the steady-state nonlinear solver.

Source:

- The solver initialization logic implemented in the ASM2d-TSN simulation workflow.

How to generate:

- Export the exact initialization formulas and constants used by the production simulation code.
- Keep the row structure already used in the manuscript: base state, substrate heuristics, biomass heuristics, dissolved oxygen estimate, and clipping bounds.
- Treat this as a static methodology table, not a benchmark-results table.

What to verify:

- Every formula matches the actual simulation code.
- The initialization order is the same as the execution order used in the solver.
- The dynamic-relaxation fallback is described in the surrounding text, not in the table itself.

### Figure `fig:mlp_architecture` using `figure1_mlp_architecture.pdf`

Purpose:
Provide a visual summary of the retained MLP baseline architecture.

Source:

- The final selected ANN/MLP configuration reported in `tab:optuna_mlp`.

How to generate:

- Draw one labeled input block for the 22-dimensional input.
- Draw three hidden layers with widths `128`, `64`, and `32`.
- Draw one labeled output block for the 20-component effluent prediction.
- Show dense inter-layer connectivity schematically; do not attempt to plot every single edge literally if readability collapses.
- Save the final flat filename as `figure1_mlp_architecture.pdf` directly inside this submission folder.

What to verify:

- The layer widths match the final ANN table exactly.
- The figure does not imply additional skip connections, normalization layers, or dropout layers that are not part of the retained baseline.

### Table `tab:optuna_design`

Purpose:
Summarize the shared Optuna study design used across all retained models.

Source:

- The common Optuna orchestration settings used in the final tuning run.

How to generate:

- Export the exact study objective, direction, sampler, pruning policy, seed, trials per model, tuning-subset fraction, validation fraction, and derived row counts.
- Use the final study settings actually used for the manuscript benchmark.

What to verify:

- The tuning-training and tuning-validation row counts are arithmetically consistent with the 4,000-row tuning subset and 20% validation split.

### Table `tab:optuna_icsor`

Purpose:
List the final selected ICSOR hyperparameters.

Source:

- The winning ICSOR Optuna trial used for the retained benchmark and repeated dataset-size study.

How to generate:

- Export the final selected values exactly as used in the training workflow.
- Keep the same parameter naming convention used in the manuscript and code.
- If the final run changes any selected values, update both this table and any dependent narrative or interpretability discussion.

What to verify:

- The listed hyperparameters match the run used to generate the reported benchmark outputs.
- The same selected settings were held fixed during the repeated dataset-size sweep.

### Table `tab:optuna_classical_a`

Purpose:
List the final selected hyperparameters for the tree-based regressors.

Source:

- Final Optuna study results for XGBoost, LightGBM, CatBoost, AdaBoost, and Random Forest.

How to generate:

- Export the winning hyperparameters per model in one row per model.
- Preserve the current model order unless there is a manuscript-wide reason to change it.

What to verify:

- The selected values match the benchmark rerun used to generate the fixed-split and repeated-size results.
- Numeric formatting is consistent across models.

### Table `tab:optuna_classical_b`

Purpose:
List the final selected hyperparameters for SVR, k-NN, and PLS.

Source:

- Final Optuna study results for those three baseline models.

How to generate:

- Export the winning hyperparameters exactly as used in the retained benchmark.
- Keep the same units and naming used in the actual model definitions.

What to verify:

- Kernel, metric, and algorithm names are reported exactly as used in the code.

### Table `tab:optuna_mlp`

Purpose:
List the final selected hyperparameters for the retained fully connected ANN.

Source:

- Final Optuna study result for the ANN/MLP baseline.

How to generate:

- Export the winning configuration used for the benchmark rerun.
- Keep the hidden-layer tuple, solver, regularization, batch, learning rate, stopping, and shuffle settings synchronized with the architecture figure.

What to verify:

- `hidden_layer_sizes` matches `figure1_mlp_architecture.pdf`.
- Any ANN setting mentioned in the surrounding text matches this table exactly.

### Table `tab:results_benchmark_sample`

Purpose:
Report the fixed 80--20 benchmark summary across the retained models.

Source:

- The one-shot benchmark run with random seed `42`.
- Final reported composite predictions for each retained model on the 2,000-row test split.

How to calculate:

- Train each model on the 8,000-row training split.
- Evaluate the final deployed component predictions on the 2,000-row test split.
- Convert predicted component states to `COD`, `TN`, `TP`, and `TSS` through `I_comp`.
- Compute aggregate test `RMSE`, `MAE`, `R^2`, and `MAPE` in the same shared composite-reporting space for every model.
- Order the rows by the primary ranking metric used in the manuscript narrative, which is currently aggregate test `RMSE`.

What to verify:

- Every model uses the same train-test split.
- ICSOR uses its constrained deployed prediction, not an unconstrained intermediate state.
- The classical regressors are inverse-transformed back to physical component coordinates before composite calculation.

### Table `tab:results_target_sample`

Purpose:
Show the per-target RMSE values for `COD`, `TN`, `TP`, and `TSS` at the retained benchmark size.

Source:

- Final reported composite predictions at the largest retained training size used for the benchmark summary.

How to calculate:

- For each model and each reported target, compute test RMSE on the final reported composite predictions.
- Use the same test split or final-size repeated-size summary convention used by the final manuscript narrative.
- Keep the target order exactly as `COD`, `TN`, `TP`, `TSS`.

What to verify:

- The target-specific values are consistent with the same benchmark outputs used elsewhere in the results section.
- Any claim about ranking reversals across targets is grounded in this table.

### Figure `fig:results_learning_curve_sample` using `figure2_rmse_learning_curve.pdf`

Purpose:
Show how test RMSE changes with total dataset size across the repeated dataset-size study.

Source:

- The 11 dataset sizes: `500`, `1450`, `2400`, `3350`, `4300`, `5250`, `6200`, `7150`, `8100`, `9050`, and `10000`.
- The 100 repeated train-test fits at each size for every retained model.

How to calculate:

- At each dataset size and for each model, aggregate the 100 repeated test RMSE values.
- Plot the central tendency as the repeat mean.
- Plot uncertainty as a soft band around that central tendency. Use a single consistent band definition such as mean `±` one standard deviation or empirical quantiles, and keep that choice fixed across models.
- Preserve the visual styling documented in `figure_style_guide.md`.

What to verify:

- The y-values are test RMSE, not train RMSE and not MAE.
- The x-axis is total dataset size, not training rows only.
- The Optuna-selected hyperparameters remain fixed during the sweep.

### Table `tab:results_scaling_sample`

Purpose:
Compress the repeated-size performance story into four exact numeric summaries.

Source:

- The same repeated dataset-size study used for the learning curves.

How to calculate:

- `Final RMSE`: the repeat-averaged test RMSE at the largest dataset size.
- `RMSE nAUC`: compute the normalized area under the repeat-averaged RMSE learning curve:

	$$
	\mathrm{nAUC}_{i,\mathrm{RMSE}} = \frac{1}{n_{\max} - n_{\min}} \int_{n_{\min}}^{n_{\max}} \bar{M}_{i,\mathrm{RMSE}}(n)\,dn
	$$

	Evaluate this numerically with the trapezoidal rule over the sampled sizes.

- `Sample Avg. Rank`: report the average dense rank across the retained metric-target-size combinations used in the manuscript ranking summary.

- `\Delta RMSE`: compute the train-test RMSE gap at the largest training size:

	$$
	\Delta_i^{\mathrm{RMSE}}(n_{\max}) = \overline{\mathrm{RMSE}}^{\mathrm{test}}_i(n_{\max}) - \overline{\mathrm{RMSE}}^{\mathrm{train}}_i(n_{\max})
	$$

What to verify:

- All four columns are derived from the same repeated-size benchmark outputs.
- Lower values are better for `Final RMSE`, `RMSE nAUC`, and typically for the generalization gap when overfitting is the concern.

### Figure `fig:results_runtime_sample` using `figure3_runtime_learning_curve.pdf`

Purpose:
Show how training time scales with total dataset size for the retained model set.

Source:

- The wall-clock training times recorded for every repeated fit in the repeated dataset-size study.

How to calculate:

- At each dataset size and for each model, aggregate the 100 recorded training times.
- Plot the repeat mean training time as the central curve.
- Use the same uncertainty-band convention as the RMSE learning curve unless there is a strong reason to change it.
- Plot the y-axis on a logarithmic scale.
- Keep the x-axis as total dataset size so the runtime and RMSE figures remain directly comparable.

What to verify:

- Use training time only; do not mix training and inference time.
- Ensure that any one-time setup cost is handled consistently across models.

### Table `tab:results_physical_sample`

Purpose:
Report the frequency of mass-conservation and non-negativity violations on final component-space predictions.

Source:

- Final component-space predictions on the benchmark test split.

How to calculate:

- For each final component prediction `\tilde{c}`, compute mass-conservation violation:

	$$
	v_{mc} = \|A(\tilde{c} - c_{in})\|_2
	$$

- Compute non-negativity violation:

	$$
	v_{nn} = \|\min(\tilde{c}, 0)\|_1
	$$

- Count a sample as violating a criterion if the corresponding quantity exceeds `10^{-10}`.
- Report the percentage of test samples violating each condition per model.

What to verify:

- The diagnostics are applied before composite collapse, because the constraints are defined in component space.
- ICSOR uses the final deployed prediction after its full hierarchical correction logic.

### Figure `fig:results_interpretability_sample` using `figure4_icsor_structure.pdf`

Purpose:
Show the COD-only `\Theta_{cc}` interaction heatmap retained in the main manuscript.

Source:

- The final COD `\Theta_{cc}` block extracted from the fitted ICSOR coefficient structure selected for the paper.

How to generate:

- Use only the COD `\Theta_{cc}` block for the main-text figure.
- Use the full ASM component basis on both axes.
- Draw the matrix from the lower origin so labels progress outward from the matrix origin.
- Use a symmetric zero-centered diverging color scale.
- Export the final figure as `figure4_icsor_structure.pdf`.

What to verify:

- Do not include the full COD atlas in the main manuscript figure.
- Ensure the final manuscript figure and the supplementary atlas set are consistent with one another.

### Table `tab:results_icsor_regularization_sample`

Purpose:
Summarize retained coefficients by COD block and quantify how sparse each retained block remains.

Source:

- The fitted COD coefficient blocks from the retained ICSOR model.
- The same coefficient-retention rule used in the interpretability workflow.

How to calculate:

- For each COD coefficient block included in the table, count the total number of coefficients in that block.
- Apply the chosen retention rule consistently within each block. The current workflow uses a retention threshold defined relative to the maximum absolute coefficient magnitude inside the block.
- Count the number of retained coefficients in each block after thresholding.
- Compute percentage retained as:

	$$
	\%\,\mathrm{Retained} = 100 \times \frac{\mathrm{Retained\ coeffs.}}{\mathrm{Total\ coeffs.}}
	$$

- Compute the summary row from the displayed block totals and retained counts:

	$$
	\mathrm{Overall\ summary\ \%} = 100 \times \frac{\sum_b \mathrm{Retained}_b}{\sum_b \mathrm{Total}_b}
	$$

What to verify:

- The block totals agree with the actual ICSOR feature dimensions.
- The same retention threshold is used for every reported COD block unless the manuscript explicitly says otherwise.
- The summary row is recomputed after any block value changes.

## Figure Production Rules

Use these rules for every manuscript figure.

- Keep every final figure file directly inside `docs/DCHE-D-26-00020`.
- Do not use subfolder paths inside the manuscript.
- Use flat filenames already referenced by the manuscript.
- Remove the illustrative footer from final submission figures.
- Keep vector PDF output whenever practical.
- Preserve the styling choices documented in `figure_style_guide.md` unless there is a specific manuscript reason to change them.

## Table Production Rules

Use these rules for every manuscript table.

- Every value in a benchmark-results table must be reproducible from a saved benchmark export or a deterministic post-processing step.
- If a table value is derived from another reported quantity, document the formula in the notebook, script, or export step that generates it.
- Recompute summary rows after any individual row changes.
- Keep numeric precision consistent within a column.
- Do not leave illustrative numbers in place once final reruns are available.

## Recommended Generation Order

To minimize inconsistencies, regenerate the manuscript assets in this order.

1. Regenerate or confirm the accepted benchmark dataset and the invariant/composition operators.
2. Confirm the final Optuna-selected hyperparameters for every retained model.
3. Run the fixed-split benchmark exports.
4. Run the repeated dataset-size study and export learning-curve, runtime, and ranking summaries.
5. Run the physical-admissibility diagnostics on final component predictions.
6. Export the final ICSOR coefficient blocks, retention counts, and interpretability figures.
7. Update the tables and figures in the submission folder.
8. Rebuild the manuscript PDF outside the submission folder and inspect the rendered output.

## Final Asset Verification Before Zipping

Before creating the EM zip, verify all of the following.

- Every captioned table and figure referenced in `manuscript.tex` has a final, non-illustrative payload.
- Every figure filename referenced in `manuscript.tex` exists directly in `docs/DCHE-D-26-00020`.
- Every result table value can be traced back to the benchmark run used for the manuscript.
- The MLP architecture figure matches the ANN hyperparameter table.
- The Optuna tables match the retained benchmark configuration.
- The results tables and figures all come from the same final benchmark rerun.
- The interpretability figure and retained-coefficient table use the same ICSOR model and the same retention logic.
- No local build artifacts or generated PDF with basename `manuscript` remain in the submission folder.

## Pre-Zip Checklist

1. Confirm that every file referenced by `manuscript.tex` is present in this folder.
2. Confirm that no referenced figures use a path like `images/figure1.png` or any other subfolder path.
3. Confirm that no two files in the submission share the same basename across different extensions when EM would treat that as conflicting upload content.
4. Confirm that only submission-relevant source files remain in the folder.
5. Zip the folder contents into a `.zip` archive for upload.

## Recommended Zip Procedure

1. Open the `docs/DCHE-D-26-00020` folder.
2. Select the submission files inside the folder, not the parent `docs` directory.
3. Create a `.zip` archive from those files.
4. Upload the archive to EM.
5. After EM builds the PDF, open the generated PDF and inspect the compiler log if anything fails.

## Recommended Local Build Output Location

For local validation builds, place the generated artifacts here:

- `docs/latex_pdfs/DCHE-D-26-00020/`

The generated PDF should be:

- `docs/latex_pdfs/DCHE-D-26-00020/DCHE-D-26-00020.pdf`

Do not move that generated PDF or any other generated artifact back into this submission folder before zipping.

## Exact CLI Command For Local PDF Generation

Use this PowerShell command from the repository root:

```powershell
New-Item -ItemType Directory -Force -Path "docs\latex_pdfs\DCHE-D-26-00020" | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="docs\latex_pdfs\DCHE-D-26-00020" -jobname="DCHE-D-26-00020" "docs\DCHE-D-26-00020\manuscript.tex"
```

This generates the LaTeX build artifacts in:

- `docs/latex_pdfs/DCHE-D-26-00020/`

The generated PDF will be at:

- `docs/latex_pdfs/DCHE-D-26-00020/DCHE-D-26-00020.pdf`

If cross-references need a second pass, run the same command again.

## If EM Still Fails To Compile

- Check for missing `.sty` or `.cls` files.
- Check for missing figure files.
- Check for references to files stored in subfolders.
- Check for unsupported filenames or special characters.
- Check the EM-generated TeX log PDF for the first reported error.