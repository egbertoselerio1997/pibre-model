# ASM2d-TCN Simulation

## 1. Title and simulation summary

The current ASM2d-TCN implementation in this repository provides a mechanistic steady-state simulation workflow plus the canonical workbook contract for a two-step nitrification ASM2d formulation. The simulation module exposes the same high-level bundle shape used by the simulation portion of `main.ipynb`: dataset, metadata, Petersen matrix, composition matrix, matrix bundle, and artifact paths.

The main reporting difference relative to the ASM1 notebook flow is deliberate: ASM2d-TCN reports composite output variables only. The saved dataset therefore contains operational inputs, influent state variables, and effluent composite outputs, but it does not persist raw effluent state variables as dependent columns.

## 2. Background and system or process context

ASM2d extends the activated-sludge model family to include biological phosphorus removal together with nitrogen and carbon conversions. The present repository variant is being prepared as an ASM2d-TCN formulation with explicit two-step nitrification, meaning nitrite and nitrate are represented separately instead of being collapsed into a single oxidized-nitrogen state.

For this repository, the workbook still fixes the process ordering, state ordering, composite-output ordering, and parameter naming. On top of that contract, the repository now implements a mechanistic steady-state simulation routine that uses the configured ASM2d-TCN process rates and matrices to generate reproducible composite-output datasets for notebook use.

## 3. Mathematical definition and governing relations

The model contains a stoichiometric matrix whose rows are processes and whose columns are state variables. The state set contains dissolved states

$$
[S_A, S_F, S_I, S_{N2}, S_{NH4}, S_{NO2}, S_{NO3}, S_{PO4}, S_{ALK}, S_{O2}]
$$

and particulate states

$$
[X_I, X_S, X_H, X_{PAO}, X_{PP}, X_{PHA}, X_{AOB}, X_{NOB}, X_{TSS}, X_{MeOH}, X_{MeP}].
$$

Several stoichiometric coefficients are not entered directly. They are derived from continuity equations:

$$
\nu_{j,NH4} = -\sum_{i \neq NH4} \nu_{j,i} i_{N,i}
$$

$$
\nu_{j,PO4} = -\sum_{i \neq PO4} \nu_{j,i} i_{P,i}
$$

$$
\nu_{j,ALK} = \frac{\nu_{j,NH4}}{14} - \frac{\nu_{j,NO2}}{14} - \frac{\nu_{j,NO3}}{14} + \frac{\nu_{j,PO4}}{31}
$$

$$
\nu_{j,TSS} = \sum_{i \in \text{particulates}} \nu_{j,i} i_{TSS,i}
$$

The composition matrix maps internal state variables to standard composite variables `[COD, TN, TKN, TP, TSS, VSS]`.

The current runtime implementation treats the reactor as a completely mixed steady-state system and solves the nonlinear residual balances directly. For hydraulic retention time $\tau$ with dilution rate $D = 24 / \tau$, the steady-state residual is

$$
r(x) = D(x_{in} - x) + \nu^T \rho(x, u) + a(x, u)
$$

where $\nu$ is the Petersen matrix, $\rho(x, u)$ is the ASM2d-TCN process-rate vector, and $a(x, u)$ is the external oxygen-transfer contribution driven by the configured aeration setting.

For each sampled operating point, the repository solves $r(x) = 0$ with a bounded nonlinear least-squares solve. The runtime uses an ASM1-style protocol:

1. construct a physically biased initial guess from the influent state, HRT, aeration setting, and optional warm start from the previous successful solve
2. solve the bounded residual system with `scipy.optimize.least_squares`
3. repeat from a biomass-rich multistart guess and keep the solution with the smaller maximum residual
4. if the residual is still above the configured acceptance threshold, run a `solve_ivp` dynamic-relaxation trajectory and use its terminal state as a final least-squares initial guess
5. reject the operating point if the final solution is unsuccessful or remains above the configured acceptance threshold

## 4. Inputs, outputs, state variables, and assumptions

Inputs to the simulation are configuration-driven:

- hydraulic retention time and aeration sampled from configured operational ranges
- influent state variables sampled from configured state ranges
- the ordered process list
- the ordered state-variable list
- the ordered composite-variable list
- the full parameter table with values and units

Outputs of the simulation workflow are:

- a dataset containing operational inputs, influent state variables, and effluent composite outputs only
- metadata describing the simulation schema and matrix shapes
- the numeric Petersen and composition matrices used by the notebook
- optional persisted CSV and JSON artifacts under `data/asm2d-tcn/simulation`

Current assumptions:

- the workbook is the canonical reference asset, not the runtime source of truth
- `config/params.json` remains the authoritative repository parameter source
- the workbook formulas mirror that configuration for transparency and later validation
- the current runtime is a mechanistic steady-state nonlinear solve intended to support notebook reproducibility and higher-fidelity dataset generation

## 5. Implementation used in this repository

The implementation is in [src/models/simulation/asm2d_tcn_simulation.py](src/models/simulation/asm2d_tcn_simulation.py).

The repository currently implements:

1. loading the ASM2d-TCN workbook definition from `config/params.json`
2. resolving the canonical workbook path from `config/paths.json`
3. generating the parameter table sheet
4. generating the stoichiometric matrix sheet with direct and continuity-derived formulas
5. generating the composition matrix sheet with parameter-linked formulas
6. building numeric Petersen and composition matrices from the configured workbook contract
7. sampling operational conditions and influent states
8. solving mechanistic steady-state effluent states internally and reporting composite outputs only
9. writing the canonical `.xlsx` file under `data/asm2d-tcn`
10. persisting simulation artifacts under `data/asm2d-tcn/simulation`

## 6. Architecture, orchestration, or adopted approach details and standard name, when relevant

The adopted approach is a configuration-driven simulation and workbook contract. The runtime module reads only repository configuration and writes only configured artifacts. This keeps path resolution compliant with repository rules and avoids hardcoded filesystem locations.

Runtime architecture:

1. `parameter_table` is written first as the source table for model constants
2. the numeric Petersen and composition matrices are built from the same configured coefficient definitions
3. operational conditions and influent states are sampled from configured ranges
4. a bounded nonlinear steady-state solve updates the internal state, with multistart and dynamic-relaxation fallback when the first pass is not acceptable
5. the composition matrix maps the internal state to the composite output space
6. only converged operating points are admitted into the generated dataset
7. the dataset and metadata are returned in the notebook-facing contract and may also be persisted to disk

This design ensures that when a parameter value changes in configuration, both the workbook and the runtime matrices stay synchronized.

## 7. Dataset-generation or execution workflow

The present workflow is:

1. call `run_asm2d_tcn_simulation`
2. generate a dataset with composite-only output targets
3. optionally persist the dataset and metadata under `data/asm2d-tcn/simulation`
4. inspect the returned Petersen and composition matrices in the notebook
5. use the same matrices for null-space and constraint analysis

The companion helper `create_asm2d_tcn_workbook` continues to generate the canonical workbook under `data/asm2d-tcn`.

## 8. Limitations and expected failure modes

Current limitations:

- the notebook-facing dataset reports composites only, so raw effluent state trajectories are not persisted as targets
- the implementation is aimed at simulation-notebook reproducibility and matrix analysis, not yet at plant-calibrated prediction

Expected failure modes:

- formula breakage if parameter names or row ordering are changed without regenerating the workbook
- contract drift if the workbook is edited manually while `config/params.json` is not updated to match
- downstream inconsistency if future ASM2d-TCN code hardcodes state or process order instead of loading configuration
- solver convergence failure if sampled influent states or operating conditions are incompatible with a physically acceptable steady state under the configured kinetics

## 9. References

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

The ASM2d-TCN stoichiometric structure, composition mapping, continuity equations, and parameter values in this repository follow the user-provided reference article captured during implementation planning.