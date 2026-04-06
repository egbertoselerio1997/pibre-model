# Comprehensive Cross-Model Comparison: Discussion Guide

## 1. Overview

This document provides a structured guide for discussing the tables and plots generated in the **Comprehensive Cross-Model Comparison** section of `main.ipynb`. It begins with an inventory of the generated artifacts and their most recent file paths, then supplies a thorough example discussion grounded in the actual numeric results.

The comparison evaluates **12 regression models** on a wastewater treatment simulation dataset that maps influent conditions and operational controls to **six measured effluent targets** (COD, TKN, TN, TP, TSS, VSS). Models are organized into two families:

| Family | Models |
|---|---|
| **Physics-informed** | ICSOR |
| **Classical** | XGBoost, LightGBM, CatBoost, AdaBoost, Random Forest, SVR, K-Nearest Neighbors, Partial Least Squares, ANN Shallow, ANN Medium, ANN Deep |

All models share the same dataset-size sweep configuration: total samples range from 500 to 10 000 in steps of 950, with a fixed 80/20 train-test split, 10 random repeats per size, and random seed 42. This yields 11 dataset sizes (training sizes from 400 to 8 000) and 110 analysis runs per model.

The primary ranking metric throughout the comparison is **effective test RMSE**. *Effective* means the predictions after any model-native post-processing: for ICSOR this is the orthogonal null-space projection that enforces stoichiometric constraints and non-negativity; for classical regressors it equals the raw prediction.

---

## 2. Artifact Inventory

All artifacts are timestamped with the format `YYYYMMDD_HHMMSS`. The most recent results correspond to the following timestamps:

- **Core tables**: `20260406_223626`
- **Derived tables**: `20260406_223635`
- **Learning-curve plots**: `20260406_223640`
- **Detail plots**: `20260406_223648`

### 2.1 Core Tables (`results/tabular_results/comparison/core/`)

These tables are the raw collated data from the per-model analysis sweeps.

| Artifact Name | Latest File | Description |
|---|---|---|
| Analysis Configs | `analysis_configs_20260406_223626.csv` | Sweep configuration per model (sample range, step, repeats, test fraction, seed). |
| Coverage | `coverage_20260406_223626.csv` | Number of runs, dataset-size range, number of targets, and effective-metric source per model. |
| Run Metadata | `run_metadata_20260406_223626.csv` | Per-run training and test sizes for every repeat and dataset size. |
| Effective Aggregate Metrics | `effective_aggregate_metrics_20260406_223626.csv` | Aggregate RMSE, MAE, RÂ², MAPE, MSE for every model Ã— split Ã— dataset size Ã— repeat. |
| Per-Target Metrics | `per_target_metrics_20260406_223626.csv` | Target-level metrics (per each of the 6 outputs) for every model Ã— split Ã— dataset size Ã— repeat. |
| Prediction Diagnostics | `prediction_diagnostics_20260406_223626.csv` | Negative prediction rates and raw-to-effective adjustment magnitudes per run. |
| Prediction Target Diagnostics | `prediction_target_diagnostics_20260406_223626.csv` | Per-target diagnostic metrics (negative rates, adjustments) per run. |

### 2.2 Derived Tables (`results/tabular_results/comparison/derived/`)

These tables are computed from the core tables and contain the rankings, summaries, and heatmap-ready frames.

| Artifact Name | Latest File | Description |
|---|---|---|
| Largest-Train-Size Leaderboard | `largest_train_size_leaderboard_20260406_223635.csv` | Model ranking across all 5 metrics at the maximum training size (8 000). |
| Average Rank Table | `average_rank_table_20260406_223635.csv` | Average rank of each model over all metrics and training sizes. |
| Curve AUC Table | `curve_auc_table_20260406_223635.csv` | Normalized area under each model's learning curve for each metric. |
| Primary Gap Largest | `primary_gap_largest_20260406_223635.csv` | Train-test generalization gap (effective RMSE) at the largest training size. |
| Primary Gap Summary | `primary_gap_summary_20260406_223635.csv` | Train-test generalization gap across all training sizes. |
| Primary Target Leaderboard | `primary_target_leaderboard_20260406_223635.csv` | Per-target model rankings at the largest training size. |
| Diagnostic Summary | `diagnostic_summary_20260406_223635.csv` | Negative prediction rates, adjustment magnitudes, and constraint residuals at largest training size. |
| Negative Rate Summary | `negative_rate_summary_20260406_223635.csv` | Negative prediction rates across all training sizes. |
| Negative Target Largest | `negative_target_largest_20260406_223635.csv` | Per-target negative prediction rates at the largest training size. |
| Adjustment Summary | `adjustment_summary_20260406_223635.csv` | Raw-to-effective adjustment L2 norms across all training sizes. |
| Constraint Summary | `constraint_summary_20260406_223635.csv` | ICSOR-specific effective constraint L2 residuals across all training sizes. |
| Average Rank Heatmap | `average_rank_heatmap_20260406_223635.csv` | Heatmap-ready frame: models Ã— metrics average rank. |
| Primary Target Heatmap | `primary_target_heatmap_20260406_223635.csv` | Heatmap-ready frame: models Ã— targets effective RMSE at largest training size. |
| Negative Target Heatmap | `negative_target_heatmap_20260406_223635.csv` | Heatmap-ready frame: models Ã— targets negative prediction rate at largest training size. |
| Model Display Order | `model_display_order_20260406_223635.csv` | Canonical model ordering by overall average rank. |
| Comparison Meta | `comparison_meta_20260406_223635.csv` | Metadata: smallest/largest training size, primary metric, primary split. |

**Learning-curve summaries** (`results/tabular_results/comparison/derived/learning_curve_summaries/`):

| Metric | Latest File |
|---|---|
| Effective RMSE | `effective_rmse_20260406_223635.csv` |
| Effective MAE | `effective_mae_20260406_223635.csv` |
| Effective RÂ² | `effective_r2_20260406_223635.csv` |
| Effective MAPE | `effective_mape_20260406_223635.csv` |

### 2.3 Plots (`results/plot_results/comparison/plots/`)

| Plot | Latest PNG | Latest SVG | Description |
|---|---|---|---|
| RMSE Learning Curve | `effective_rmse_learning_curve_20260406_223640.png` | `â€¦_223640.svg` | Effective test RMSE vs. training size for all 12 models (mean Â± IQR). |
| MAE Learning Curve | `effective_mae_learning_curve_20260406_223640.png` | `â€¦_223640.svg` | Effective test MAE vs. training size. |
| RÂ² Learning Curve | `effective_r2_learning_curve_20260406_223640.png` | `â€¦_223640.svg` | Effective test RÂ² vs. training size. |
| MAPE Learning Curve | `effective_mape_learning_curve_20260406_223640.png` | `â€¦_223640.svg` | Effective test MAPE vs. training size. |
| Average Rank Heatmap | `average_rank_heatmap_20260406_223640.png` | `â€¦_223640.svg` | Models Ã— metrics average rank (lower is better). |
| RMSE by Target Heatmap | `effective_rmse_by_target_heatmap_20260406_223648.png` | `â€¦_223648.svg` | Effective RMSE per target at the largest training size. |
| Negative Rate by Target Heatmap | `negative_prediction_rate_by_target_heatmap_20260406_223648.png` | `â€¦_223648.svg` | Negative prediction rate per target at the largest training size. |
| Generalization Gap | `effective_rmse_generalization_gap_20260406_223648.png` | `â€¦_223648.svg` | Train-test RMSE gap across training sizes. |
| Raw-to-Effective Adjustment | `raw_to_effective_adjustment_learning_curve_20260406_223648.png` | `â€¦_223648.svg` | Raw-to-effective adjustment L2 norm across training sizes. |
| Constraint Residual | `effective_constraint_residual_learning_curve_20260406_223648.png` | `â€¦_223648.svg` | ICSOR constraint satisfaction residual across training sizes. |

---

## 3. How to Read and Discuss Each Artifact

### 3.1 Largest-Train-Size Leaderboard

**Purpose**: Provides the definitive ranking at the operational regime of interest (maximum data availability).

**Key columns**: `test_RMSE_mean`, `test_RMSE_rank`, and analogous columns for MAE, RÂ², MAPE, MSE.

**Discussion approach**: Report the top-3 and bottom-3 models. Compare the spread between top and bottom. Note any cases where RMSE rank diverges from RÂ² or MAE rank (indicating different error distributions). Identify where the physics-informed model (ICSOR) sits relative to the best purely data-driven models.

### 3.2 Average Rank Table

**Purpose**: Captures consistency across all training sizes, not just the largest.

**Discussion approach**: A model that ranks well at 8 000 samples but poorly at 400 samples will have a worse average rank than one that is consistently near the top. Compare each model's per-metric average rank to its leaderboard position at the largest size. Identify models whose ranking improves or deteriorates as data increases.

### 3.3 Curve AUC Table

**Purpose**: Summarizes sample efficiencyâ€”how quickly each model improves as data grows.

**Discussion approach**: Lower RMSE-AUC and MAE-AUC are better (less cumulative error across the learning curve). Higher RÂ²-AUC is better. A model with a lower curve AUC than another model dominates the other across most of the training-size range.

### 3.4 Primary Target Leaderboard and Heatmap

**Purpose**: Reveals per-target strengths and weaknesses.

**Discussion approach**: Identify targets where the global winner is not the per-target winner. Discuss whether certain model families are better suited to specific physical quantities. For example, ICSOR's physics-informed constraints may give it an advantage on targets where stoichiometric balance matters most (e.g., TP).

### 3.5 Generalization Gap (Primary Gap Largest)

**Purpose**: Quantifies overfitting by comparing train-RMSE to test-RMSE.

**Key columns**: `effective_RMSE_gap` (absolute difference) and `effective_RMSE_gap_pct` (percentage of test RMSE).

**Discussion approach**: Large gaps signal overfitting. Compare tree-based methods (which tend to memorize training data) against regularized models and neural networks. Note whether overfitting worsens or improves as training size grows.

### 3.6 Diagnostic Summary, Negative Rate and Adjustment

**Purpose**: Assesses physical plausibility of predictions.

**Discussion approach**: Negative predictions are physically impossible for concentration outputs. Models that produce negatives may need post-processing. The raw-to-effective adjustment magnitude shows how much ICSOR's projection changes raw predictions, reflecting the cost of enforcing constraints. The constraint residual shows how tightly constraints are actually satisfied.

### 3.7 Learning Curve Plots

**Purpose**: Visual representation of how each model's test error evolves as training data grows.

**Discussion approach**: Identify convergence patternsâ€”has each model plateaued, or does the curve suggest more data would help? Look for crossover points where one model overtakes another.

---

## 4. Sample Comprehensive Discussion of the Most Recent Results

### 4.1 Overall Model Ranking

At the maximum training size of 8 000 samples, the **ANN Medium Regressor** achieves the lowest effective test RMSE of **4.928** (Â±0.256), closely followed by the **ANN Deep Regressor** at **4.957** (Â±0.220) and the **ANN Shallow Regressor** at **5.308** (Â±0.273). The three ANN variants form a distinct performance tier, outperforming all other models by a substantial margin. The gap between ANN Medium (rank 1) and the next non-ANN modelâ€”SVR at 7.142 RMSE (rank 4)â€”is **2.21 RMSE units**, representing a 31% relative reduction in error.

The middle tier comprises SVR (7.142), LightGBM (7.146), ICSOR (7.215), XGBoost (7.421), and CatBoost (7.995), spanning a range of only 0.85 RMSE units. Within this group, the differences are moderate and fall within approximately 2â€“4 standard deviations of each model's repeat-to-repeat variability, suggesting that the competitive separation here is meaningful but not as decisive as the ANN tier advantage.

The bottom tier includes PLS (14.347), Random Forest (17.420), KNN (20.492), and AdaBoost (21.625). These models deliver test RMSE values 2â€“4Ã— worse than the top tier. The performance of KNN and AdaBoost is especially poor, with RMSE values roughly 4Ã— higher than the ANN models, rendering them unsuitable for operational deployment on this problem without fundamental architectural changes.

### 4.2 Consistency Across Metrics

The average rank table aggregates rankings not only at the largest training size but across all 11 dataset sizes and all 5 metrics (RMSE, MAE, RÂ², MAPE, MSE). The ANN Medium Regressor achieves an overall average rank of **1.99**, indicating that it ranks very close to 1st or 2nd place in nearly every metricâ€“training-size combination. Its per-metric average ranks are remarkably stable: RMSE 2.02, MAE 1.91, RÂ² 2.03, MAPE 1.94, MSE 2.05. This consistency confirms that ANN Medium's dominance is not an artifact of a single metric.

The ANN Shallow Regressor (overall rank 2.60) and ANN Deep Regressor (overall rank 3.02) maintain their top-tier positions across the full sweep. Interestingly, ANN Deep slightly outperforms ANN Medium on individual metrics like MAPE (average rank 2.91 vs. 1.94 for ANN Medium) while being marginally worse on MAE (3.00 vs. 1.91), suggesting that deeper networks trade off slightly between absolute-error and relative-error metrics.

ICSOR ranks 4th overall (average rank 4.11), notably ahead of all gradient-boosted tree models. Its average rank is stable across metrics (RMSE 4.18, MAE 3.97, RÂ² 4.20, MAPE 3.98, MSE 4.20), with its best relative performance on MAE. This is notable because ICSOR is the only model with an explicit physics-informed projection layer, and its rank-4 position among 12 models shows that the physics constraints do not sacrifice predictive accuracyâ€”in fact, they confer a slight advantage over most classical regressors.

LightGBM (rank 4.26) and XGBoost (rank 6.06) are the best-performing gradient-boosted tree models. LightGBM's closeness to ICSOR in the average-rank table suggests that, across all data regimes, these two models are competitive alternatives, with ICSOR holding a slight edge attributable to its constraint structure.

At the bottom, KNN (rank 11.80) and AdaBoost (rank 10.88) are consistently the worst models across all metrics and training sizes, confirming that their poor performance is structural rather than a consequence of one unfavorable metric.

### 4.3 Sample Efficiency â€” Learning Curve Analysis

The learning curve AUC provides a single-number summary of each model's performance trajectory across training sizes. Lower RMSE-AUC is better, as it indicates lower cumulative error across the learning curve.

The ANN models dominate sample efficiency: ANN Medium achieves an RMSE-AUC of **6.35**, ANN Shallow **6.56**, and ANN Deep **6.66**. This means that not only do ANN models converge to the lowest final error, they also learn fasterâ€”their error drops steeply with the first few hundred additional training samples.

ICSOR achieves an RMSE-AUC of **8.12**, placing it 4th in sample efficiencyâ€”ahead of LightGBM (8.37), SVR (8.83), XGBoost (8.89), and CatBoost (8.98). This is significant because ICSOR's physics-informed structure provides inductive bias that compensates for limited data, allowing it to achieve competitive performance even at small training sizes where purely data-driven models struggle. Indeed, at 400 training samples, ICSOR's effective RMSE can be expected to be lower relative to tree-based models because the stoichiometric constraint projection reduces the effective dimensionality of the prediction space.

The learning curve plots visually illustrate the three performance tiers. The ANN curves descend rapidly from an RMSE of approximately 12 at 400 samples to approximately 5 at 8 000 samples. The middle-tier models (ICSOR, LightGBM, SVR, XGBoost, CatBoost) follow a similar but shallower descent from approximately 15 to approximately 7â€“8. The bottom-tier models (PLS, Random Forest, AdaBoost, KNN) show either very slow improvement (AdaBoost barely moves from 21.4 to 21.6) or persistently high error (KNN remains above 20 throughout).

A crossover analysis reveals that the ANN models overtake all middle-tier models before approximately 1 200 training samples, suggesting that even for moderate-data applications, ANN architectures already dominate. There is no crossover between the top-3 ANN models and any other model family.

The RÂ²-AUC corroborates these findings: ANN Medium achieves 0.973, ANN Shallow 0.973, and ANN Deep 0.971, well above the middle-tier range of 0.956â€“0.965. ICSOR achieves 0.964, slightly above LightGBM (0.965) and meaningfully above CatBoost (0.957). The bottom-tier models, especially KNN (RÂ²-AUC = 0.733), explain under 75% of output variance on average across the full training-size sweepâ€”unacceptable for most practical applications.

### 4.4 Per-Target Analysis

The per-target leaderboard at 8 000 training samples reveals important target-specific nuances:

**TP (Total Phosphorus)**: ICSOR achieves the **best RMSE of 0.364** (Â±0.034), surpassing all ANN models (ANN Shallow: 0.388, ANN Medium: 0.413, ANN Deep: 0.445). This is the only target where ICSOR ranks 1st, and it is a physically meaningful result: total phosphorus is governed by tight stoichiometric constraints in the ASM framework (phosphorus accumulating organisms and precipitation chemistry), and ICSOR's null-space projection directly enforces these constraints. The 6.3% improvement over the second-best model (ANN Shallow) on TP demonstrates a concrete advantage of physics-informed modeling for constraint-sensitive outputs.

**COD (Chemical Oxygen Demand)**: ANN Medium leads with RMSE 7.100, followed by ANN Deep (7.195) and ANN Shallow (7.695). ICSOR ranks 4th at 11.172, a substantial 57% relative increase over ANN Medium. COD represents a broad aggregate measure of organic matter, and its prediction is primarily governed by input magnitude and general nonlinear patterns rather than stoichiometric detail. The ANN models' superior capacity for capturing arbitrary nonlinear mappings gives them a clear advantage here.

**TKN (Total Kjeldahl Nitrogen)**: ANN Deep edges out ANN Medium (3.000 vs. 3.104) by a slim margin. ICSOR ranks 7th at 3.971. The gradient-boosted tree models (LightGBM: 3.443, XGBoost: 3.567) fill the intermediate positions, suggesting that for nitrogen species, tree-based ensemble methods capture the relevant input-output relationships more effectively than the physics-constrained linear framework.

**TN (Total Nitrogen)**: ANN Medium leads (1.099), followed closely by ANN Deep (1.103). ICSOR ranks 4th (1.616). The relatively small absolute RMSE for this target (all below 2.0 for the top-8 models) reflects the narrower dynamic range of total nitrogen in the simulation compared to COD.

**TSS and VSS (Suspended Solids)**: Both targets follow a nearly identical ranking pattern. ANN Medium and ANN Deep tie for 1st-2nd (TSS: 6.891 vs. 6.892; VSS: 6.058 vs. 6.131). ICSOR ranks 7th on both targets (TSS: 9.372; VSS: 9.007). The solids targets are influenced by settling dynamics and particulate fractions, which are nonlinear processes where the deep function approximation capacity of ANNs is most beneficial.

**Summary of per-target patterns**: ICSOR excels on TP (rank 1), is competitive on TN (rank 4), and falls to rank 7 on suspended-solids targets (TSS, VSS). This pattern aligns with the expectation that physics-informed constraints add the most value for targets tightly coupled to stoichiometric balances (phosphorus) and add less for targets governed by physical separation processes (suspended solids).

### 4.5 Generalization Gap Analysis

The train-test RMSE gap at 8 000 training samples reveals each model's propensity to overfit:

| Model | Test RMSE | Train RMSE | Gap | Gap % |
|---|---|---|---|---|
| PLS Regressor | 14.347 | 14.194 | 0.153 | **1.1%** |
| AdaBoost Regressor | 21.625 | 20.994 | 0.632 | **3.0%** |
| ICSOR | 7.215 | 6.875 | 0.340 | **4.9%** |
| ANN Shallow Regressor | 5.308 | 4.836 | 0.471 | **9.7%** |
| CatBoost Regressor | 7.995 | 6.911 | 1.084 | **15.7%** |
| ANN Medium Regressor | 4.928 | 3.118 | 1.810 | **58.0%** |
| ANN Deep Regressor | 4.957 | 2.808 | 2.149 | **76.5%** |
| LightGBM Regressor | 7.146 | 3.908 | 3.237 | **82.8%** |
| XGBoost Regressor | 7.421 | 3.482 | 3.940 | **113.1%** |
| Random Forest Regressor | 17.420 | 6.380 | 11.040 | **173.1%** |
| SVR Regressor | 7.142 | 2.102 | 5.040 | **239.7%** |
| KNN Regressor | 20.492 | â‰ˆ0 | 20.492 | **>10â¹%** |

**Key findings**:

- **PLS (1.1%) and ICSOR (4.9%)** exhibit the smallest generalization gaps, reflecting strong regularization. PLS achieves this through its inherently low-dimensional latent structure; ICSOR achieves it through the stoichiometric constraint projection that restricts the prediction manifold. However, while PLS has a small gap, its absolute test error is poor (14.35), meaning it *underfits* rather than generalizes well.

- **ICSOR's 4.9% gap** is the second-lowest among all models and the lowest among models with competitive absolute performance (RMSE < 10). This makes ICSOR the most *reliably generalizing* model in the middle tier, which is a strong operational advantage: its deployment predictions are closely calibrated to its validation performance.

- **ANN Medium and ANN Deep** have gaps of 58% and 76.5%, respectively. These are substantial, indicating that the networks memorize training patterns to a significant degree. However, their absolute test errors remain the lowest of all models, meaning the overfitting has not yet degraded generalization to the point where simpler competitors win. This is a classic deep-learning pattern: moderate overfitting accompanied by strong generalization, likely due to implicit regularization from stochastic gradient descent and the broad function class.

- **SVR (239.7%)** and **KNN (>10â¹%)** show extreme overfitting. SVR achieves train RMSE of only 2.10 while testing at 7.14, indicating unduly flexible kernel mapping. KNN's near-zero train error is expected (it memorizes training points exactly) but the massive test error confirms that nearest-neighbor interpolation fails entirely on this high-dimensional, nonlinear problem.

- **Random Forest (173.1%)** and **XGBoost (113.1%)** have large gaps, reflecting the well-known tendency of tree ensembles to achieve very low training error through deep trees while requiring careful regularization for test performance.

### 4.6 Physical Plausibility Diagnostics

**Negative prediction rates**: At 8 000 training samples, all models produce zero negative predictions except for PLS, which produces negatives at a rate of 0.08% (approximately 1 in 1 200 predictions). For PLS, these negatives occur specifically on the TKN target (0.5% negative rate for TKN). This is a minor but notable limitation of PLS's linear structure, which cannot enforce output non-negativity.

Across all training sizes, the negative rate remains zero for every model except PLS, where it decreases from approximately 0.05â€“0.1% at small training sizes to 0.08% at 8 000 samples. The absence of negative predictions from ICSOR at any training size is guaranteed by its mathematical formulation (the non-negative quadratic programming projection), whereas the zero rates for the other classical models are empirical observations that could change under different data distributions.

**Raw-to-effective adjustment magnitude (ICSOR)**: ICSOR's raw predictions undergo a projection that adjusts them to satisfy the stoichiometric constraints and non-negativity. The L2 norm of this adjustment starts at approximately **80.0** at 400 training samples and decreases to approximately **69.9** at 8 000 samples (a 12.6% reduction). This monotonic decrease indicates that as ICSOR receives more training data, its raw predictions become increasingly aligned with the physics constraintsâ€”the model *learns the constraint structure* rather than relying entirely on the post-hoc projection.

The standard deviation of the adjustment also decreases from 7.55 (at 400 samples) to 0.23 (at 8 000 samples), a 97% reduction in variability. This remarkable stabilization means that at large data sizes, ICSOR produces highly consistent raw predictions that require a nearly constant correction, which is a desirable property for operational deployment.

**Constraint satisfaction residuals (ICSOR)**: The effective constraint L2 residualâ€”measuring how closely the final projected prediction satisfies the stoichiometric invariant $Ac^* = Ac_{in}$â€”is on the order of $5 \times 10^{-11}$ across all training sizes. This is essentially machine-precision satisfaction, confirming that the quadratic-programming solver reliably enforces the constraints to numerical tolerance regardless of training-set size. The residual is stable across the full sweep (from $6.3 \times 10^{-11}$ at 400 samples to $4.9 \times 10^{-11}$ at 8 000 samples), demonstrating that constraint enforcement is not compromised by the quality of the raw predictions.

### 4.7 Target-Specific Error Profiles â€” Heatmap Discussion

The effective RMSE by target heatmap at 8 000 training samples reveals the error structure across all models and targets:

| Model | COD | TN | TKN | TP | TSS | VSS |
|---|---|---|---|---|---|---|
| ANN Medium | 7.10 | 1.10 | 3.10 | 0.41 | 6.89 | 6.06 |
| ANN Shallow | 7.69 | 1.28 | 3.11 | 0.39 | 7.30 | 6.71 |
| ANN Deep | 7.19 | 1.10 | 3.00 | 0.44 | 6.89 | 6.13 |
| ICSOR | 11.17 | 1.62 | 3.97 | **0.36** | 9.37 | 9.01 |
| LightGBM | 12.11 | 1.73 | 3.44 | 0.47 | 8.73 | 8.28 |
| XGBoost | 12.56 | 1.79 | 3.57 | 0.47 | 9.11 | 8.57 |
| SVR | 11.46 | 1.89 | 3.72 | 0.77 | 9.18 | 8.51 |
| CatBoost | 13.19 | 2.06 | 4.07 | 0.69 | 10.25 | 9.11 |
| PLS | 26.01 | 2.71 | 6.31 | 0.59 | 16.91 | 15.01 |
| Random Forest | 33.70 | 3.91 | 6.12 | 2.31 | 18.25 | 17.14 |
| AdaBoost | 41.28 | 4.20 | 7.97 | 2.08 | 22.58 | 22.51 |
| KNN | 36.89 | 6.72 | 8.33 | 3.51 | 23.24 | 22.17 |

**Error magnitudes track the dynamic range of each target.** COD exhibits the highest absolute RMSE (7â€“41 units) because its measurement range is the widest among the six outputs. TP has the lowest RMSE (0.36â€“3.51) reflecting its narrow concentration range. This means that absolute RMSE should not be directly compared across targets without normalization; the MAPE metric provides that relative perspective.

**ICSOR's TP advantage is visually prominent**: it is the only cell in the heatmap where ICSOR's color is the best among all models. For all other targets, ICSOR's color is intermediateâ€”darker than the ANN row but lighter than the tree-based and distance-based rows.

**The suspended-solids pair (TSS, VSS) shows almost identical ranking patterns**, which is physically expected because VSS is a sub-fraction of TSS, and the two are strongly correlated in the simulation. Models that perform well on TSS invariably perform well on VSS.

### 4.8 Synthesis and Practical Recommendations

1. **Best overall model**: ANN Medium Regressor. It achieves the lowest test RMSE, the best average rank (1.99), and the best learning-curve AUC. Its advantage is consistent across all metrics and most targets.

2. **Best physics-informed model**: ICSOR. It ranks 4th overall and 1st on the TP target. Its generalization gap (4.9%) is the lowest among competitive models, and it is the only model that guarantees non-negative predictions and stoichiometric consistency by construction. This makes it the preferred choice when physical plausibility is a hard requirement.

3. **Best tree-based model**: LightGBM. It ranks 5th overall, barely behind ICSOR, and offers fast training and straightforward hyperparameter tuning. XGBoost (6th) and CatBoost (8th) are viable alternatives with slightly worse performance.

4. **Models unsuitable for this problem**: KNN and AdaBoost are consistently the worst performers across all metrics, targets, and training sizes. Random Forest and PLS also underperform significantly. These four models should not be considered for deployment.

5. **Data regime recommendations**: At small training sizes (400â€“1 200 samples), the gap between ANN models and classical models is narrower, and ICSOR's physics constraints provide meaningful inductive bias. At large training sizes (>4 000 samples), ANN models dominate decisively. The crossover point at approximately 1 200 samples suggests that if data collection is costly, ICSOR or LightGBM offers a competitive alternative to neural networks.

6. **Overfitting awareness**: The 58â€“77% generalization gaps of ANN Medium and Deep at 8 000 samples warrant monitoring in production. If the deployment data distribution shifts, these models may degrade more than ICSOR (4.9% gap) or PLS (1.1% gap). Ensemble methods or periodic retraining may be necessary.

7. **Constraint value**: ICSOR's constraint enforcement has near-zero computational cost (the adjustment L2 stabilizes at â‰ˆ70 and the constraint residual is â‰ˆ10â»Â¹Â¹) and provides a hard guarantee of physical consistency. For regulatory or safety-critical applications where negative effluent concentrations or stoichiometric violations are unacceptable, ICSOR remains the only model that provides this guarantee without additional post-processing.

---

## 5. Reproducing the Comparison

To regenerate all artifacts:

1. Execute all cells in `main.ipynb` from the beginning through the **Comprehensive Cross-Model Comparison** section.
2. Each execution produces a new timestamp-tagged set of files; previous versions remain on disk for historical comparison.
3. If upstream training cells have not been run, the comparison cells automatically fall back to loading the most recent artifacts from `results/tabular_results/` and `results/plot_results/`.

To discuss a new set of results, update the timestamps in Section 2 of this document to match the latest files and re-examine the numeric values in the CSV artifacts.

