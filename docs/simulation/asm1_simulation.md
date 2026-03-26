# ASM1 Simulation Summary

## 1. Title and simulation summary

The asm1_simulation module is now implemented as a mechanistic steady-state activated-sludge CSTR solver for a single completely mixed aerobic reactor. The model no longer maps sampled laboratory analytes directly to effluent through surrogate removal factors. Instead, it samples mechanistic influent state variables, solves the nonlinear steady-state reactor balances, and then maps the solved reactor outlet state to a smaller panel of directly interpretable analytes.

The generated dataset is intended for simulation-driven studies where the predictors are operating conditions plus influent state variables, and the responses are the steady-state reactor outlet states together with selected analyte summaries such as COD, NH4-N, NO3-N, TP, TSS, and dissolved oxygen.

## 2. Background and system or process context

Activated-sludge process models are based on species balances over soluble and particulate state variables rather than directly over routine laboratory analytes. A realistic CSTR simulation therefore starts from internal process states, not from composite measurements such as BOD5 or pH alone.

The implementation in this repository follows that principle. It treats the reactor as a single continuously stirred tank with continuous inflow, continuous outflow, biological reaction source terms, and oxygen transfer. The present implementation is closest to an ASM-family carbon-nitrogen-phosphorus assimilation model for a single aerobic basin without secondary clarification or recycle. Because no clarifier is included, the outlet corresponds to the mixed-liquor concentration leaving the CSTR.

## 3. Mathematical definition and governing relations

Let the state vector be

$$
x = [S_S, S_I, S_{NH4}, S_{NO3}, S_{PO4}, S_O, S_{Alk}, X_I, X_S, X_H, X_{AUT}]^T
$$

where the soluble states represent readily biodegradable substrate, inert soluble COD, ammonium nitrogen, oxidized nitrogen, orthophosphate phosphorus, dissolved oxygen, and alkalinity; the particulate states represent inert particulates, slowly biodegradable substrate, heterotrophic biomass, and autotrophic biomass.

For a tank with hydraulic retention time $\tau$ and volumetric dilution rate $D = 1 / \tau$, the steady-state mass balance for each state is written as

$$
0 = D(x_{in} - x) + \nu^T r(x, u)
$$

where $x_{in}$ is the influent state vector, $\nu$ is the explicit Petersen matrix, and $r(x, u)$ is the process-rate vector evaluated at the reactor state and operating condition. In this repository the Petersen matrix has 7 rows and 11 columns. The first six rows are biochemical conversions and the seventh row is a deliberate mass-transfer extension representing aeration.

The implemented process set is:

1. hydrolysis of slowly biodegradable substrate
2. aerobic heterotrophic growth
3. anoxic heterotrophic growth
4. autotrophic growth and nitrification
5. heterotrophic decay
6. autotrophic decay
7. aeration mass transfer

Representative rate expressions are Monod-type kinetics:

$$
\rho_{H,ae} = \mu_H \frac{S_S}{K_S + S_S} \frac{S_O}{K_{OH} + S_O} X_H
$$

$$
\rho_{H,an} = \mu_H \eta_g \frac{S_S}{K_S + S_S} \frac{K_{OH}}{K_{OH} + S_O} \frac{S_{NO3}}{K_{NO} + S_{NO3}} X_H
$$

$$
\rho_A = \mu_A \frac{S_{NH4}}{K_{NH} + S_{NH4}} \frac{S_O}{K_{OA} + S_O} X_{AUT}
$$

Oxygen transfer is represented mechanistically by the seventh process rate,

$$
r_7 = K_L a \left(S_{O,sat} - S_O\right)
$$

with $K_L a$ computed from the configured aeration intensity. The associated Petersen row contains a nonzero coefficient only in the dissolved-oxygen column. This is not the standard textbook presentation, where oxygen transfer is often added outside the biochemical Petersen matrix, but it is an intentional implementation choice here so that all state-source terms are exposed in one matrix object.

The repository also constructs an explicit composition matrix $C$ that maps solved internal states to reported analytes:

$$
y = Cx
$$

where $y = [COD, TSS, VSS, TN, TP, NH4\text{-}N, NO3\text{-}N, PO4\text{-}P, DO, Alkalinity]^T$. This matrix is built from the configured solids, nitrogen, and phosphorus observation factors so that the analyte mapping is explicit and inspectable.

The nonlinear steady-state algebraic system is solved numerically for each sampled operating point using a bounded least-squares residual solve.

## 4. Inputs, outputs, state variables, and assumptions

Independent columns (inputs):

- HRT
- Aeration
- In_S_S
- In_S_I
- In_S_NH4_N
- In_S_NO3_N
- In_S_PO4_P
- In_S_O2
- In_S_Alkalinity
- In_X_I
- In_X_S
- In_X_H
- In_X_AUT

Dependent columns (outputs):

- Out_S_S
- Out_S_I
- Out_S_NH4_N
- Out_S_NO3_N
- Out_S_PO4_P
- Out_S_O2
- Out_S_Alkalinity
- Out_X_I
- Out_X_S
- Out_X_H
- Out_X_AUT
- Out_COD
- Out_TSS
- Out_VSS
- Out_TN
- Out_TP
- Out_NH4_N
- Out_NO3_N
- Out_PO4_P
- Out_DO
- Out_Alkalinity

Assumptions:

- the reactor is a single completely mixed aerobic CSTR
- the outlet is the reactor mixed liquor because no clarifier or recycle loop is included
- temperature is implicit in the configured kinetic constants rather than dynamically varied
- pH is not exported because the present mechanistic state set does not yet include the aqueous chemistry needed to solve it rigorously
- BOD5 and nitrite are not exported because they are not direct states of the implemented reactor model

## 5. Implementation used in this repository

Implementation is in `src/models/simulation/asm1_simulation.py`.

Main implementation blocks:

1. Load the parameter namespace from `config/params.json`.
2. Construct the explicit Petersen matrix and composition matrix from the configured stoichiometric and observation factors.
3. Sample mechanistic influent state variables and operating variables from configured ranges.
4. Solve the steady-state nonlinear CSTR balances for each sampled operating point using dilution plus matrix-based process contributions.
5. Map the solved outlet state to measured analyte summaries through the composition matrix.
6. Build metadata describing the state-based schema and matrix shapes.
7. Persist the dataset and metadata using the shared simulation utilities.

## 6. Architecture, orchestration, or adopted approach details and standard name, when relevant

The adopted approach is a mechanistic steady-state activated-sludge CSTR model with nonlinear residual solving and explicit matrix exposure.

Execution architecture:

1. configuration load
2. Petersen/composition matrix construction
3. influent-state and operating-point sampling
4. single-point nonlinear steady-state solve
5. analyte observation mapping
6. dataset assembly and artifact persistence

The simulation remains deterministic when the random seed is fixed.

## 7. Dataset-generation or execution workflow

The workflow is:

1. Import and call `run_asm1_simulation` from the simulation package.
2. Generate a table of sampled influent states, solved outlet states, and mapped analyte outputs.
3. Save outputs under the repository simulation artifact contract:
   - `data/asm1_simulation/data_{date_time}.csv`
   - `data/asm1_simulation/metadata_{date_time}.json`
4. Use the metadata file as the contract for downstream loading.

## 8. Limitations and expected failure modes

Key limitations:

- this is a single-reactor model and does not include secondary clarification, sludge recycle, wastage, or multi-zone phosphorus release and uptake
- the kinetics are literature-default values and are not calibrated to a specific plant
- pH, inorganic carbon chemistry, and explicit nitrite dynamics are intentionally excluded until the required chemistry states are added
- because there is no clarifier, the outlet solids represent the CSTR mixed liquor rather than final clarified effluent
- the aeration row is a non-standard matrix extension used for implementation clarity; it should be interpreted as a mass-transfer source term, not as a biological conversion

Expected failure modes:

- unrealistic operating points if the sampled influent state ranges are not physically compatible with a single aerobic reactor
- biomass washout if the sampled HRT is too short relative to the configured growth kinetics
- solver convergence failure if the parameter set or sampled influent states produce inconsistent balances

## 9. References

Henze, M., Gujer, W., Mino, T., and van Loosdrecht, M. Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific and Technical Report No. 9, 2000.

Gujer, W. Systems Analysis for Water Technology. Springer, 2008.
