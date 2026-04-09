# ASM2D-TSN Workbook Reconstruction Specification

This document describes the canonical workbook at `data/asm2d-tsn/asm2d_tsn_workbook.xlsx` for schema version `3.0`.

The active contract is a 20-state ASM2D-TSN model with five measured composites: `COD`, `TN`, `TKN`, `TP`, `TSS`.

The internal state basis no longer includes `X_TSS`. `TSS` remains a measured composite and is computed directly from particulate-state composition coefficients.

## 1. Scope and invariants

- Workbook filename: `asm2d_tsn_workbook.xlsx`
- Worksheet order: `stoichiometric_matrix`, `composition_matrix`, `parameter_table`
- Blank matrix cells are intentionally blank
- `parameter_table` values are referenced by absolute formulas from the matrix sheets

## 2. Workbook layout contract

| Sheet | Used range | Freeze panes | AutoFilter |
| --- | --- | --- | --- |
| `stoichiometric_matrix` | `A1:V29` | `C2` | `A1:V29` |
| `composition_matrix` | `A1:H21` | `D2` | `A1:H21` |
| `parameter_table` | `A1:F82` | `A2` | `A1:F82` |

## 3. State and output ordering

### 3.1 State columns (20 total)

`S_O`, `S_F`, `S_A`, `S_NH4`, `S_NO2`, `S_NO3`, `S_N2`, `S_PO4`, `S_I`, `S_ALK`, `X_I`, `X_S`, `X_H`, `X_PAO`, `X_PP`, `X_PHA`, `X_AOB`, `X_NOB`, `X_MeP`, `X_MeOH`

### 3.2 Measured composites (5 total)

`COD`, `TN`, `TKN`, `TP`, `TSS`

## 4. Matrix dimensions

- Petersen matrix shape: `(28, 20)`
- Composition matrix shape: `(5, 20)`

## 5. Composition matrix rules

### 5.1 Nitrogen and phosphorus continuity terms

- `TN` and `TKN` follow configured nitrogen content coefficients
- `TP` follows configured phosphorus content coefficients plus direct particulate-phosphorus terms (`X_PP`) and precipitate phosphorus (`X_MeP`)

### 5.2 Direct TSS composition mapping

`TSS` is computed directly from particulate states with these coefficients:

- `X_I`: `i_TSS_XI`
- `X_S`: `i_TSS_XS`
- `X_H`: `i_TSS_BM`
- `X_PAO`: `i_TSS_BM`
- `X_PP`: `i_TSS_PP`
- `X_PHA`: `i_TSS_PHA`
- `X_AOB`: `i_TSS_BM`
- `X_NOB`: `i_TSS_BM`
- `X_MeOH`: `1`
- `X_MeP`: `1`

There is no `X_TSS -> TSS` identity row in the composition matrix.

## 6. Stoichiometric matrix rules

### 6.1 Derived entries

- `S_NH4` is derived from nitrogen continuity
- `S_PO4` is derived from phosphorus continuity unless explicitly assigned by a process
- `S_ALK` is derived from `S_NH4`, `S_NO2`, `S_NO3`, and `S_PO4`

### 6.2 Precipitation and redissolution rows

Precipitation and redissolution use direct ferric-solid stoichiometry without `X_TSS`:

- Precipitation: `S_PO4 = -1`, `X_MeOH = -3.45`, `X_MeP = 4.87`
- Redissolution: `S_PO4 = 1`, `X_MeOH = 3.45`, `X_MeP = -4.87`

The net solids effect is represented through `X_MeOH` and `X_MeP`, and therefore propagates into `TSS` through the composition matrix.

## 7. Parameter table notes

- Row count remains `81` parameters plus header (`A1:F82`)
- TSS-related coefficients remain present and are now interpreted as direct composition coefficients
- The `i_TSS_PHA` and `i_TSS_PP` descriptions refer to the TSS composite definition

## 8. Minimal validation checklist

- Workbook sheets are exactly `stoichiometric_matrix`, `composition_matrix`, `parameter_table` in this order
- `stoichiometric_matrix` has range `A1:V29`
- `composition_matrix` has range `A1:H21`
- State list excludes `X_TSS`
- Composition outputs are exactly `COD`, `TN`, `TKN`, `TP`, `TSS`
- Numeric matrix shapes are `(28, 20)` and `(5, 20)`
- Precipitation/redissolution rows contain no `X_TSS` terms
- `TSS` composition includes direct terms from particulate states, including `X_MeOH` and `X_MeP`