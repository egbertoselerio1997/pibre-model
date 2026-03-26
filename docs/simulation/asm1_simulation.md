# ASM1 Simulation Summary

## 1. Title and simulation summary

The ASM1 simulation is an ASM1-inspired continuous stirred-tank reactor dataset generator written for rapid machine-learning experimentation. It does not solve the full activated-sludge differential equation system over time. Instead, it maps sampled influent and operating conditions into synthetic effluent compositions by combining a stoichiometric reaction representation with randomly sampled reaction extents.

The resulting dataset is intended to support regression workflows in which the predictors are operating conditions plus influent composites, and the responses are effluent composites.

## 2. Background and system or process context

Activated Sludge Model No. 1, usually abbreviated ASM1, is a standard biokinetic framework for biological wastewater treatment. It represents the interactions among soluble substrates, biomass populations, oxygen, nitrogen species, and particulate components. In full form, ASM1 is a dynamic process model with state evolution governed by coupled mass-balance and reaction-rate equations.

The implementation used in this repository adopts the stoichiometric structure of ASM1 as an organizing principle, but compresses the detailed state space into a smaller set of composite variables:

- total chemical oxygen demand
- total nitrogen
- dissolved oxygen
- nitrate
- alkalinity

This reduced representation is appropriate when the goal is to create a consistent synthetic dataset for machine-learning model development rather than to emulate a full plant simulator.

## 3. Mathematical definition and governing relations

The simulation uses a Petersen-style stoichiometric matrix $\nu \in \mathbb{R}^{8 \times 13}$ together with a composition matrix $I \in \mathbb{R}^{13 \times 5}$. Their product defines a composite-space stoichiometric map:

$$
S_{comp} = \nu I
$$

where $S_{comp} \in \mathbb{R}^{8 \times 5}$ maps eight biological-process extents into five composite balance changes.

For each synthetic sample, the influent vector is

$$
C_{in} \in \mathbb{R}^{5}
$$

and a vector of sampled process extents is generated as

$$
e \in \mathbb{R}^{8}
$$

with only the first three extents active in the present implementation:

$$
e_1 = C_{in,\mathrm{COD}} \cdot u_1, \quad u_1 \sim U(0.1, 0.4)
$$

$$
e_2 = C_{in,\mathrm{COD}} \cdot u_2, \quad u_2 \sim U(0.0, 0.1)
$$

$$
e_3 = C_{in,\mathrm{N}} \cdot u_3, \quad u_3 \sim U(0.1, 0.8)
$$

The effluent composition is then computed algebraically:

$$
C_{out} = C_{in} + e S_{comp}
$$

This means each dataset row is generated independently. The model is therefore a stochastic algebraic simulator rather than a time-marching dynamic reactor solver.

## 4. Inputs, outputs, state variables, and assumptions

Inputs sampled for each row are:

- hydraulic retention time
- aeration intensity
- influent total chemical oxygen demand
- influent total nitrogen
- influent dissolved oxygen
- influent nitrate
- influent alkalinity

Outputs produced for each row are:

- effluent total chemical oxygen demand
- effluent total nitrogen
- effluent dissolved oxygen
- effluent nitrate
- effluent alkalinity

The implementation assumes:

- each row is statistically independent from every other row
- operating variables are sampled independently from uniform distributions
- reaction extents are sampled independently from uniform fractions of selected influent composites
- only a subset of the available ASM1 reaction rows is activated during dataset generation
- the stoichiometric projection is sufficient to produce a useful reduced-order synthetic benchmark

## 5. Implementation used in this repository

The repository implementation lives in the ASM1 simulation module under src/models/simulation. All numerical settings are loaded at runtime from config/params.json, including:

- sample count
- random seed
- influent sampling ranges
- operating-variable sampling ranges
- ASM1 yield and stoichiometric constants
- active reaction-extent rules

When the simulation is executed through the system entry point, it produces a tabular dataset and a metadata JSON contract. The metadata records the predictor columns, target columns, and the CSV path corresponding to the generated dataset.

## 6. Architecture, orchestration, or adopted approach details and standard name, when relevant

The adopted approach is best described as a reduced-order, stoichiometric synthetic-data generator derived from ASM1 concepts. The architecture is:

1. Load path and parameter configuration.
2. Build the composite stoichiometric matrix from configured ASM1 constants.
3. Sample influent and operational inputs from configured uniform ranges.
4. Sample biological reaction extents from configured fractional rules.
5. Compute effluent composites using the stoichiometric map.
6. Save the dataset and metadata contract to the configured repository locations.

This architecture is deterministic once the random seed is fixed.

## 7. Dataset-generation or execution workflow

The workflow executed in the main notebook is:

1. Import the reusable simulation function from the source package.
2. Run the simulation with the configured parameters.
3. Save the dataset under data/asm1_simulation using the repository naming contract.
4. Save a companion metadata JSON file in the same folder family.
5. Inspect the generated dataframe head and the metadata contract from the notebook.

The current file naming contract is:

$$
\text{data/asm1\_simulation/data\_\{date\_time\}.csv}
$$

$$
\text{data/asm1\_simulation/metadata\_\{date\_time\}.json}
$$

## 8. Limitations and expected failure modes

This simulation should not be interpreted as a high-fidelity wastewater-process simulator. Important limitations are:

- there is no dynamic state evolution over time
- process kinetics are not solved explicitly
- the operating variables do not currently modulate the reaction extents mechanistically
- only three reaction extents are active during sample generation
- the generated outputs can be physically stylized without representing a fully calibrated treatment plant

Expected failure modes include unrealistic effluent combinations if parameter ranges are widened excessively, or poor downstream machine-learning performance if the synthetic sampling ranges do not resemble the intended application domain.

## 9. References

Henze, M., Grady, C. P. L., Gujer, W., Marais, G. v. R., and Matsuo, T. Activated Sludge Model No. 1. IAWPRC Scientific and Technical Reports, 1987.

Gujer, W. Systems Analysis for Water Technology. Springer, 2008.