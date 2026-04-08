# The Nature of the Carbon Source Rules the Competition Between PAO and Denitrifiers in Systems for Simultaneous Biological Nitrogen and Phosphorus Removal

Authors: Javier Guerrero, Albert Guisasola, and Juan A. Baeza

## Editorial note

This version preserves the supplied biological description, process numbering, parameter values, and references. The notation and equations have been rewritten in standard activated-sludge-model form so the stoichiometric and kinetic expressions are readable and internally consistent.

## Biological model description

The model used in this work is an extension of ASM2d that describes the biological processes occurring in a system for simultaneous biological organic matter and nutrient removal (Henze et al., 2000). The major extension was the inclusion of nitrite as a state variable and, therefore, nitrification was modeled as a two-step process including AOB and NOB as autotrophic biomass. Denitrification was also described in two steps to include the denitrifying capacity of PAO, since some of them can use either nitrate and/or nitrite as electron acceptor. This extension required seven new processes and three new state variables (AOB, NOB and nitrate). Hence, the extended model included 21 components and 28 processes.

## Notation convention used below

- Standard ASM subscripts are used throughout, for example $S_O$ for dissolved oxygen, $S_{NH_4}$ for ammonium, and $X_{PAO}$ for phosphorus-accumulating organisms.
- Concentration units are written in the conventional ASM form of g m$^{-3}$, which is numerically identical to mg L$^{-1}$.
- Repeated source labels such as $K_{O2}$, $K_{NO3}$, and $\eta_{NO3}$ are distinguished by process group where needed, without changing the underlying model content.
- In Tables S2 and S3, coefficients not shown are zero.

## Table S1. Definition of model components

### Dissolved components

| Symbol | Units | Description |
| --- | --- | --- |
| $S_O$ | g O$_2$ m$^{-3}$ | Dissolved oxygen |
| $S_F$ | g COD m$^{-3}$ | Readily biodegradable substrate |
| $S_A$ | g COD m$^{-3}$ | Fermentation products |
| $S_{NH_4}$ | g N m$^{-3}$ | Ammonium |
| $S_{NO_2}$ | g N m$^{-3}$ | Nitrite |
| $S_{NO_3}$ | g N m$^{-3}$ | Nitrate |
| $S_{N_2}$ | g N m$^{-3}$ | Nitrogen gas |
| $S_{PO_4}$ | g P m$^{-3}$ | Phosphate |
| $S_I$ | g COD m$^{-3}$ | Inert, non-biodegradable organics |
| $S_{ALK}$ | mol HCO$_3^-$ m$^{-3}$ | Bicarbonate alkalinity |

### Particulate components

| Symbol | Units | Description |
| --- | --- | --- |
| $X_I$ | g COD m$^{-3}$ | Inert, non-biodegradable organics |
| $X_S$ | g COD m$^{-3}$ | Slowly biodegradable substrate |
| $X_H$ | g COD m$^{-3}$ | Heterotrophic biomass |
| $X_{PAO}$ | g COD m$^{-3}$ | Phosphorus accumulating organisms, PAO |
| $X_{PP}$ | g P m$^{-3}$ | Stored poly-phosphate of PAO |
| $X_{PHA}$ | g COD m$^{-3}$ | Organic storage products of PAO |
| $X_{AOB}$ | g COD m$^{-3}$ | Ammonium oxidizing bacteria, AOB |
| $X_{NOB}$ | g COD m$^{-3}$ | Nitrite oxidizing bacteria, NOB |
| $X_{MeP}$ | g TSS m$^{-3}$ | Ferric-phosphate |
| $X_{MeOH}$ | g TSS m$^{-3}$ | Ferric-hydroxide |
| $X_{TSS}$ | g TSS m$^{-3}$ | Particulate material as model component |

## Shared process index for Tables S2-S4

1. Aerobic hydrolysis
2. Anoxic hydrolysis (NO$_2$)
3. Anoxic hydrolysis (NO$_3$)
4. Anaerobic hydrolysis
5. Aerobic growth on $S_F$
6. Aerobic growth on $S_A$
7. Anoxic growth on $S_F$ (denitrification NO$_3$ to NO$_2$)
8. Anoxic growth on $S_F$ (denitrification NO$_2$ to N$_2$)
9. Anoxic growth on $S_A$ (denitrification NO$_3$ to NO$_2$)
10. Anoxic growth on $S_A$ (denitrification NO$_2$ to N$_2$)
11. Fermentation
12. Lysis of $X_H$
13. Storage of $X_{PHA}$
14. Aerobic storage of $X_{PP}$
15. Anoxic storage of $X_{PP}$ (denitrification NO$_3$ to NO$_2$)
16. Anoxic storage of $X_{PP}$ (denitrification NO$_2$ to N$_2$)
17. Aerobic growth
18. Anoxic growth (denitrification NO$_3$ to NO$_2$)
19. Anoxic growth (denitrification NO$_2$ to N$_2$)
20. Lysis of $X_{PAO}$
21. Lysis of $X_{PP}$
22. Lysis of $X_{PHA}$
23. Aerobic growth of $X_{AOB}$
24. Aerobic growth of $X_{NOB}$
25. Lysis of $X_{AOB}$
26. Lysis of $X_{NOB}$
27. Precipitation
28. Redissolution

## Continuity relations used in Tables S2 and S3

For the continuity-derived dissolved coefficients:

$$
\nu_{i,S_{NH_4}} = -\sum_{j \ne S_{NH_4}} i_{N,j}\,\nu_{i,j}
$$

$$
\nu_{i,S_{PO_4}} = -\sum_{j \ne S_{PO_4}} i_{P,j}\,\nu_{i,j}
$$

$$
\nu_{i,S_{ALK}} = \frac{\nu_{i,S_{NH_4}} - \nu_{i,S_{NO_2}} - \nu_{i,S_{NO_3}}}{14} + \frac{\nu_{i,S_{PO_4}}}{31}
$$

In the supplied source, $\nu_{i,X_{TSS}}$ denotes the continuity-derived TSS coefficient for process $i$.

## Table S2. Process stoichiometry for the two-step nitrification and denitrification model (dissolved components)

Only non-zero dissolved-component coefficients are listed.

### Hydrolysis processes

| Process | Non-zero dissolved stoichiometric coefficients |
| --- | --- |
| 1. Aerobic hydrolysis | $S_F = 1 - f_{SI}$; $S_{NH_4} = \nu_{1,S_{NH_4}}$; $S_{PO_4} = \nu_{1,S_{PO_4}}$; $S_I = f_{SI}$; $S_{ALK} = \nu_{1,S_{ALK}}$ |
| 2. Anoxic hydrolysis (NO$_2$) | $S_F = 1 - f_{SI}$; $S_{NH_4} = \nu_{2,S_{NH_4}}$; $S_{PO_4} = \nu_{2,S_{PO_4}}$; $S_I = f_{SI}$; $S_{ALK} = \nu_{2,S_{ALK}}$ |
| 3. Anoxic hydrolysis (NO$_3$) | $S_F = 1 - f_{SI}$; $S_{NH_4} = \nu_{3,S_{NH_4}}$; $S_{PO_4} = \nu_{3,S_{PO_4}}$; $S_I = f_{SI}$; $S_{ALK} = \nu_{3,S_{ALK}}$ |
| 4. Anaerobic hydrolysis | $S_F = 1 - f_{SI}$; $S_{NH_4} = \nu_{4,S_{NH_4}}$; $S_{PO_4} = \nu_{4,S_{PO_4}}$; $S_I = f_{SI}$; $S_{ALK} = \nu_{4,S_{ALK}}$ |

### Heterotrophic organisms, $X_H$

| Process | Non-zero dissolved stoichiometric coefficients |
| --- | --- |
| 5. Aerobic growth on $S_F$ | $S_O = 1 - 1/Y_H$; $S_F = -1/Y_H$; $S_{NH_4} = \nu_{5,S_{NH_4}}$; $S_{PO_4} = \nu_{5,S_{PO_4}}$; $S_{ALK} = \nu_{5,S_{ALK}}$ |
| 6. Aerobic growth on $S_A$ | $S_O = 1 - 1/Y_H$; $S_A = -1/Y_H$; $S_{NH_4} = -i_{NBM}$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{6,S_{ALK}}$ |
| 7. Anoxic growth on $S_F$ (NO$_3$ to NO$_2$) | $S_F = -1/Y_H$; $S_{NH_4} = \nu_{7,S_{NH_4}}$; $S_{NO_2} = (1 - Y_H)/((8/7)Y_H)$; $S_{NO_3} = -(1 - Y_H)/((8/7)Y_H)$; $S_{PO_4} = \nu_{7,S_{PO_4}}$; $S_{ALK} = \nu_{7,S_{ALK}}$ |
| 8. Anoxic growth on $S_F$ (NO$_2$ to N$_2$) | $S_F = -1/Y_H$; $S_{NH_4} = \nu_{8,S_{NH_4}}$; $S_{NO_2} = -(1 - Y_H)/(1.72Y_H)$; $S_{N_2} = (1 - Y_H)/(1.72Y_H)$; $S_{PO_4} = \nu_{8,S_{PO_4}}$; $S_{ALK} = \nu_{8,S_{ALK}}$ |
| 9. Anoxic growth on $S_A$ (NO$_3$ to NO$_2$) | $S_A = -1/Y_H$; $S_{NH_4} = -i_{NBM}$; $S_{NO_2} = (1 - Y_H)/((8/7)Y_H)$; $S_{NO_3} = -(1 - Y_H)/((8/7)Y_H)$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{9,S_{ALK}}$ |
| 10. Anoxic growth on $S_A$ (NO$_2$ to N$_2$) | $S_A = -1/Y_H$; $S_{NH_4} = -i_{NBM}$; $S_{NO_2} = -(1 - Y_H)/(1.72Y_H)$; $S_{N_2} = (1 - Y_H)/(1.72Y_H)$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{10,S_{ALK}}$ |
| 11. Fermentation | $S_F = -1$; $S_A = 1$; $S_{NH_4} = \nu_{11,S_{NH_4}}$; $S_{PO_4} = \nu_{11,S_{PO_4}}$; $S_{ALK} = \nu_{11,S_{ALK}}$ |
| 12. Lysis of $X_H$ | $S_{NH_4} = \nu_{12,S_{NH_4}}$; $S_{PO_4} = \nu_{12,S_{PO_4}}$; $S_{ALK} = \nu_{12,S_{ALK}}$ |

### Accumulating phosphorus organisms, $X_{PAO}$

| Process | Non-zero dissolved stoichiometric coefficients |
| --- | --- |
| 13. Storage of $X_{PHA}$ | $S_A = -1$; $S_{PO_4} = Y_{PO4}$; $S_{ALK} = \nu_{13,S_{ALK}}$ |
| 14. Aerobic storage of $X_{PP}$ | $S_O = -Y_{PHA}$; $S_{PO_4} = -1$; $S_{ALK} = \nu_{14,S_{ALK}}$ |
| 15. Anoxic storage of $X_{PP}$ (NO$_3$ to NO$_2$) | $S_{NO_2} = Y_{PHA}/(8/7)$; $S_{NO_3} = -Y_{PHA}/(8/7)$; $S_{PO_4} = -1$; $S_{ALK} = \nu_{15,S_{ALK}}$ |
| 16. Anoxic storage of $X_{PP}$ (NO$_2$ to N$_2$) | $S_{NO_2} = -Y_{PHA}/1.72$; $S_{N_2} = Y_{PHA}/1.72$; $S_{PO_4} = -1$; $S_{ALK} = \nu_{16,S_{ALK}}$ |
| 17. Aerobic growth | $S_O = 1 - 1/Y_{PAO}$; $S_{NH_4} = -i_{NBM}$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{17,S_{ALK}}$ |
| 18. Anoxic growth (NO$_3$ to NO$_2$) | $S_{NH_4} = -i_{NBM}$; $S_{NO_2} = (1 - Y_{PAO})/((8/7)Y_{PAO})$; $S_{NO_3} = -(1 - Y_{PAO})/((8/7)Y_{PAO})$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{18,S_{ALK}}$ |
| 19. Anoxic growth (NO$_2$ to N$_2$) | $S_{NH_4} = -i_{NBM}$; $S_{NO_2} = -(1 - Y_{PAO})/(1.72Y_{PAO})$; $S_{N_2} = (1 - Y_{PAO})/(1.72Y_{PAO})$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{19,S_{ALK}}$ |
| 20. Lysis of $X_{PAO}$ | $S_{NH_4} = \nu_{20,S_{NH_4}}$; $S_{PO_4} = \nu_{20,S_{PO_4}}$; $S_{ALK} = \nu_{20,S_{ALK}}$ |
| 21. Lysis of $X_{PP}$ | $S_{PO_4} = 1$; $S_{ALK} = \nu_{21,S_{ALK}}$ |
| 22. Lysis of $X_{PHA}$ | $S_A = 1$; $S_{ALK} = \nu_{22,S_{ALK}}$ |

### Nitrifying organisms, $X_{AOB}$ and $X_{NOB}$

| Process | Non-zero dissolved stoichiometric coefficients |
| --- | --- |
| 23. Aerobic growth of $X_{AOB}$ | $S_O = -(3.43 - Y_{AOB})/Y_{AOB}$; $S_{NH_4} = -i_{NBM} - 1/Y_{AOB}$; $S_{NO_2} = 1/Y_{AOB}$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{23,S_{ALK}}$ |
| 24. Aerobic growth of $X_{NOB}$ | $S_O = -(1.14 - Y_{NOB})/Y_{NOB}$; $S_{NH_4} = -i_{NBM}$; $S_{NO_2} = -1/Y_{NOB}$; $S_{NO_3} = 1/Y_{NOB}$; $S_{PO_4} = -i_{PBM}$; $S_{ALK} = \nu_{24,S_{ALK}}$ |
| 25. Lysis of $X_{AOB}$ | $S_{NH_4} = \nu_{25,S_{NH_4}}$; $S_{PO_4} = \nu_{25,S_{PO_4}}$; $S_{ALK} = \nu_{25,S_{ALK}}$ |
| 26. Lysis of $X_{NOB}$ | $S_{NH_4} = \nu_{26,S_{NH_4}}$; $S_{PO_4} = \nu_{26,S_{PO_4}}$; $S_{ALK} = \nu_{26,S_{ALK}}$ |

### Phosphorus precipitation and redissolution

| Process | Non-zero dissolved stoichiometric coefficients |
| --- | --- |
| 27. Precipitation | $S_{PO_4} = -1$; $S_{ALK} = \nu_{27,S_{ALK}}$ |
| 28. Redissolution | $S_{PO_4} = 1$; $S_{ALK} = \nu_{28,S_{ALK}}$ |

## Table S3. Process stoichiometry for the two-step nitrification and denitrification model (particulate components)

Only non-zero particulate-component coefficients are listed.

### Hydrolysis processes

| Process | Non-zero particulate stoichiometric coefficients |
| --- | --- |
| 1. Aerobic hydrolysis | $X_S = -1$; $X_{TSS} = \nu_{1,X_{TSS}}$ |
| 2. Anoxic hydrolysis (NO$_2$) | $X_S = -1$; $X_{TSS} = \nu_{2,X_{TSS}}$ |
| 3. Anoxic hydrolysis (NO$_3$) | $X_S = -1$; $X_{TSS} = \nu_{3,X_{TSS}}$ |
| 4. Anaerobic hydrolysis | $X_S = -1$; $X_{TSS} = \nu_{4,X_{TSS}}$ |

### Heterotrophic organisms, $X_H$

| Process | Non-zero particulate stoichiometric coefficients |
| --- | --- |
| 5. Aerobic growth on $S_F$ | $X_H = 1$; $X_{TSS} = \nu_{5,X_{TSS}}$ |
| 6. Aerobic growth on $S_A$ | $X_H = 1$; $X_{TSS} = \nu_{6,X_{TSS}}$ |
| 7. Anoxic growth on $S_F$ (NO$_3$ to NO$_2$) | $X_H = 1$; $X_{TSS} = \nu_{7,X_{TSS}}$ |
| 8. Anoxic growth on $S_F$ (NO$_2$ to N$_2$) | $X_H = 1$; $X_{TSS} = \nu_{8,X_{TSS}}$ |
| 9. Anoxic growth on $S_A$ (NO$_3$ to NO$_2$) | $X_H = 1$; $X_{TSS} = \nu_{9,X_{TSS}}$ |
| 10. Anoxic growth on $S_A$ (NO$_2$ to N$_2$) | $X_H = 1$; $X_{TSS} = \nu_{10,X_{TSS}}$ |
| 11. Fermentation | $X_{TSS} = \nu_{11,X_{TSS}}$ |
| 12. Lysis of $X_H$ | $X_I = f_{XI}$; $X_S = 1 - f_{XI}$; $X_H = -1$; $X_{TSS} = \nu_{12,X_{TSS}}$ |

### Accumulating phosphorus organisms, $X_{PAO}$

| Process | Non-zero particulate stoichiometric coefficients |
| --- | --- |
| 13. Storage of $X_{PHA}$ | $X_{PP} = -Y_{PO4}$; $X_{PHA} = 1$; $X_{TSS} = \nu_{13,X_{TSS}}$ |
| 14. Aerobic storage of $X_{PP}$ | $X_{PP} = 1$; $X_{PHA} = -Y_{PHA}$; $X_{TSS} = \nu_{14,X_{TSS}}$ |
| 15. Anoxic storage of $X_{PP}$ (NO$_3$ to NO$_2$) | $X_{PP} = 1$; $X_{PHA} = -Y_{PHA}$; $X_{TSS} = \nu_{15,X_{TSS}}$ |
| 16. Anoxic storage of $X_{PP}$ (NO$_2$ to N$_2$) | $X_{PP} = 1$; $X_{PHA} = -Y_{PHA}$; $X_{TSS} = \nu_{16,X_{TSS}}$ |
| 17. Aerobic growth | $X_{PAO} = 1$; $X_{PHA} = -1/Y_{PAO}$; $X_{TSS} = \nu_{17,X_{TSS}}$ |
| 18. Anoxic growth (NO$_3$ to NO$_2$) | $X_{PAO} = 1$; $X_{PHA} = -1/Y_{PAO}$; $X_{TSS} = \nu_{18,X_{TSS}}$ |
| 19. Anoxic growth (NO$_2$ to N$_2$) | $X_{PAO} = 1$; $X_{PHA} = -1/Y_{PAO}$; $X_{TSS} = \nu_{19,X_{TSS}}$ |
| 20. Lysis of $X_{PAO}$ | $X_I = f_{XI}$; $X_S = 1 - f_{XI}$; $X_{PAO} = -1$; $X_{TSS} = \nu_{20,X_{TSS}}$ |
| 21. Lysis of $X_{PP}$ | $X_{PP} = -1$; $X_{TSS} = \nu_{21,X_{TSS}}$ |
| 22. Lysis of $X_{PHA}$ | $X_{PHA} = -1$; $X_{TSS} = \nu_{22,X_{TSS}}$ |

### Nitrifying organisms, $X_{AOB}$ and $X_{NOB}$

| Process | Non-zero particulate stoichiometric coefficients |
| --- | --- |
| 23. Aerobic growth of $X_{AOB}$ | $X_{AOB} = 1$; $X_{TSS} = \nu_{23,X_{TSS}}$ |
| 24. Aerobic growth of $X_{NOB}$ | $X_{NOB} = 1$; $X_{TSS} = \nu_{24,X_{TSS}}$ |
| 25. Lysis of $X_{AOB}$ | $X_I = f_{XI}$; $X_S = 1 - f_{XI}$; $X_{AOB} = -1$; $X_{TSS} = \nu_{25,X_{TSS}}$ |
| 26. Lysis of $X_{NOB}$ | $X_I = f_{XI}$; $X_S = 1 - f_{XI}$; $X_{NOB} = -1$; $X_{TSS} = \nu_{26,X_{TSS}}$ |

### Phosphorus precipitation and redissolution

| Process | Non-zero particulate stoichiometric coefficients |
| --- | --- |
| 27. Precipitation | $X_{TSS} = 1.42$; $X_{MeOH} = -3.45$; $X_{MeP} = 4.87$ |
| 28. Redissolution | $X_{TSS} = -1.42$; $X_{MeOH} = 3.45$; $X_{MeP} = -4.87$ |

## Table S4. Process kinetics for the two-step nitrification and denitrification model

For compact notation, define

$$
M(x;K) = \frac{x}{K + x}
$$

$$
F_F = \frac{S_F}{S_A + S_F}, \qquad F_A = \frac{S_A}{S_A + S_F}, \qquad F_{NO_2} = \frac{S_{NO_2}}{S_{NO_3} + S_{NO_2}}, \qquad F_{NO_3} = \frac{S_{NO_3}}{S_{NO_3} + S_{NO_2}}
$$

$$
H_X = \frac{X_S/X_H}{K_X + X_S/X_H}, \qquad R_{PP} = \frac{X_{PP}/X_{PAO}}{K_{PP} + X_{PP}/X_{PAO}}, \qquad R_{PHA} = \frac{X_{PHA}/X_{PAO}}{K_{PHA} + X_{PHA}/X_{PAO}}
$$

$$
C_{PP} = \frac{K_{MAX} - X_{PP}/X_{PAO}}{K_{IPP} + K_{MAX} - X_{PP}/X_{PAO}}
$$

### Hydrolysis processes

$$
\begin{aligned}
\rho_1 &= K_H\,M(S_O;K_{O,hyd})\,H_X\,X_H \\
\rho_2 &= \eta^{hyd}_{NO_2}\,K_H\,\frac{K_{O,hyd}}{K_{O,hyd}+S_O}\,M(S_{NO_2};K_{NO_2,hyd})\,F_{NO_2}\,H_X\,X_H \\
\rho_3 &= \eta^{hyd}_{NO_3}\,K_H\,\frac{K_{O,hyd}}{K_{O,hyd}+S_O}\,M(S_{NO_3};K_{NO_3,hyd})\,F_{NO_3}\,H_X\,X_H \\
\rho_4 &= \eta^{hyd}_{fe}\,K_H\,\frac{K_{O,hyd}}{K_{O,hyd}+S_O}\,\frac{K_{NOX,hyd}}{K_{NOX,hyd}+S_{NO_3}+S_{NO_2}}\,H_X\,X_H
\end{aligned}
$$

### Heterotrophic organisms, $X_H$

$$
\begin{aligned}
\rho_5 &= \mu_H\,M(S_O;K_{O,H})\,M(S_F;K_F)\,F_F\,M(S_{NH_4};K_{NH_4,H})\,M(S_{PO_4};K_{PO_4,H})\,M(S_{ALK};K_{ALK,H})\,X_H \\
\rho_6 &= \mu_H\,M(S_O;K_{O,H})\,M(S_A;K_A)\,F_A\,M(S_{NH_4};K_{NH_4,H})\,M(S_{PO_4};K_{PO_4,H})\,M(S_{ALK};K_{ALK,H})\,X_H \\
\rho_7 &= \mu_H\,\frac{K_{O,H}}{K_{O,H}+S_O}\,M(S_F;K_F)\,F_F\,M(S_{NH_4};K_{NH_4,H})\,M(S_{PO_4};K_{PO_4,H})\,M(S_{ALK};K_{ALK,H})\,\eta^H_{NO_3}\,M(S_{NO_3};K_{NO_3,H})\,F_{NO_3}\,X_H \\
\rho_8 &= \mu_H\,\frac{K_{O,H}}{K_{O,H}+S_O}\,M(S_F;K_F)\,F_F\,M(S_{NH_4};K_{NH_4,H})\,M(S_{PO_4};K_{PO_4,H})\,M(S_{ALK};K_{ALK,H})\,\eta^H_{NO_2}\,M(S_{NO_2};K_{NO_2,H})\,F_{NO_2}\,X_H \\
\rho_9 &= \mu_H\,\frac{K_{O,H}}{K_{O,H}+S_O}\,M(S_A;K_A)\,F_A\,M(S_{NH_4};K_{NH_4,H})\,M(S_{PO_4};K_{PO_4,H})\,M(S_{ALK};K_{ALK,H})\,\eta^H_{NO_3}\,M(S_{NO_3};K_{NO_3,H})\,F_{NO_3}\,X_H \\
\rho_{10} &= \mu_H\,\frac{K_{O,H}}{K_{O,H}+S_O}\,M(S_A;K_A)\,F_A\,M(S_{NH_4};K_{NH_4,H})\,M(S_{PO_4};K_{PO_4,H})\,M(S_{ALK};K_{ALK,H})\,\eta^H_{NO_2}\,M(S_{NO_2};K_{NO_2,H})\,F_{NO_2}\,X_H \\
\rho_{11} &= q_{Fe}\,\mu_H\,\frac{K_{O,H}}{K_{O,H}+S_O}\,\frac{K_{NOX,H}}{K_{NOX,H}+S_{NO_2}+S_{NO_3}}\,M(S_F;K_{fe})\,M(S_{ALK};K_{ALK,H})\,X_H \\
\rho_{12} &= b_H\,X_H
\end{aligned}
$$

### Accumulating phosphorus organisms, $X_{PAO}$

$$
\begin{aligned}
\rho_{13} &= q_{PHA}\,M(S_A;K_A)\,M(S_{ALK};K_{ALK,PAO})\,R_{PP}\,X_{PAO} \\
\rho_{14} &= q_{PP}\,M(S_O;K_{O,PAO})\,M(S_{PO_4};K_{PS})\,M(S_{ALK};K_{ALK,PAO})\,R_{PHA}\,C_{PP}\,X_{PAO} \\
\rho_{15} &= q_{PP}\,\frac{K_{O,PAO}}{K_{O,PAO}+S_O}\,M(S_{PO_4};K_{PS})\,M(S_{ALK};K_{ALK,PAO})\,R_{PHA}\,C_{PP}\,\eta^{PAO}_{NO_3}\,M(S_{NO_3};K_{NO_3,PAO})\,F_{NO_3}\,X_{PAO} \\
\rho_{16} &= q_{PP}\,\frac{K_{O,PAO}}{K_{O,PAO}+S_O}\,M(S_{PO_4};K_{PS})\,M(S_{ALK};K_{ALK,PAO})\,R_{PHA}\,C_{PP}\,\eta^{PAO}_{NO_2}\,M(S_{NO_2};K_{NO_2,PAO})\,F_{NO_2}\,X_{PAO} \\
\rho_{17} &= \mu_{PAO}\,M(S_O;K_{O,PAO})\,M(S_{NH_4};K_{NH_4,PAO})\,M(S_{PO_4};K_{PO_4,PAO})\,R_{PHA}\,M(S_{ALK};K_{ALK,PAO})\,X_{PAO} \\
\rho_{18} &= \mu_{PAO}\,\frac{K_{O,PAO}}{K_{O,PAO}+S_O}\,M(S_{NH_4};K_{NH_4,PAO})\,M(S_{PO_4};K_{PO_4,PAO})\,R_{PHA}\,M(S_{ALK};K_{ALK,PAO})\,\eta^{PAO}_{NO_3}\,M(S_{NO_3};K_{NO_3,PAO})\,F_{NO_3}\,X_{PAO} \\
\rho_{19} &= \mu_{PAO}\,\frac{K_{O,PAO}}{K_{O,PAO}+S_O}\,M(S_{NH_4};K_{NH_4,PAO})\,M(S_{PO_4};K_{PO_4,PAO})\,R_{PHA}\,M(S_{ALK};K_{ALK,PAO})\,\eta^{PAO}_{NO_2}\,M(S_{NO_2};K_{NO_2,PAO})\,F_{NO_2}\,X_{PAO} \\
\rho_{20} &= b_{PAO}\,X_{PAO}\,M(S_{ALK};K_{ALK,PAO}) \\
\rho_{21} &= b_{PP}\,X_{PP}\,M(S_{ALK};K_{ALK,PAO}) \\
\rho_{22} &= b_{PHA}\,X_{PHA}\,M(S_{ALK};K_{ALK,PAO})
\end{aligned}
$$

### Nitrifying organisms, $X_{AOB}$ and $X_{NOB}$

$$
\begin{aligned}
\rho_{23} &= \mu_{AOB}\,M(S_O;K_{O,AOB})\,M(S_{NH_4};K_{NH_4,AOB})\,M(S_{PO_4};K_{PO_4,nit})\,M(S_{ALK};K_{ALK,nit})\,X_{AOB} \\
\rho_{24} &= \mu_{NOB}\,M(S_O;K_{O,NOB})\,M(S_{NO_2};K_{NO_2,NOB})\,M(S_{PO_4};K_{PO_4,nit})\,M(S_{ALK};K_{ALK,nit})\,X_{NOB} \\
\rho_{25} &= b_{AOB}\,X_{AOB} \\
\rho_{26} &= b_{NOB}\,X_{NOB}
\end{aligned}
$$

### Phosphorus precipitation and redissolution

$$
\begin{aligned}
\rho_{27} &= k_{PRE}\,S_{PO_4}\,X_{MeOH} \\
\rho_{28} &= k_{RED}\,X_{MeP}\,M(S_{ALK};K_{ALK,chem})
\end{aligned}
$$

## Table S5. Kinetic parameters for the two-step nitrification and two-step denitrification model at $T = 20^{\circ}$C and pH = 7.5

### Hydrolysis processes

| Standardized symbol | Source label | ASM2d value | Extended model value | Units | Description |
| --- | --- | ---: | ---: | --- | --- |
| $K_H$ | KH | 3.00 | 3.00 | d$^{-1}$ | Hydrolysis rate constant |
| $\eta^{hyd}_{NO_3}$ | nNO3 | 0.60 | 0.60 | - | Anoxic hydrolysis reduction factor |
| $\eta^{hyd}_{NO_2}$ | nNO2 |  | 0.60 | - | Anoxic hydrolysis reduction factor |
| $\eta^{hyd}_{fe}$ | nfe | 0.40 | 0.40 | - | Anaerobic hydrolysis reduction factor |
| $K_{O,hyd}$ | KO2 | 0.20 | 0.20 | g O$_2$ m$^{-3}$ | Saturation/inhibition coefficient for oxygen |
| $K_{NO_3,hyd}$ | KNO3 | 0.50 | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrate |
| $K_{NO_2,hyd}$ | KNO2 |  | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrite |
| $K_{NOX,hyd}$ | KNOX |  | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrogen oxides |
| $K_X$ | KX | 0.10 | 0.10 | g $X_S$ (g $X_H$)$^{-1}$ | Saturation coefficient for particulate COD |

### Heterotrophic organisms, $X_H$

| Standardized symbol | Source label | ASM2d value | Extended model value | Units | Description |
| --- | --- | ---: | ---: | --- | --- |
| $\mu_H$ | $\mu$H | 6.00 | 6.00 | g $X_S$ (g $X_H$)$^{-1}$ d$^{-1}$ | Maximum growth rate on substrate |
| $q_{Fe}$ | qFe | 3.00 | 3.00 | g $X_S$ (g $X_H$)$^{-1}$ d$^{-1}$ | Maximum rate for fermentation |
| $\eta^{H}_{NO_3}$ | nNO3 | 0.80 | 0.90* | - | Reduction factor for denitrification |
| $\eta^{H}_{NO_2}$ | nNO2 |  | 0.90* | - | Reduction factor for denitrification via nitrite |
| $b_H$ | bH | 0.40 | 0.40 | d$^{-1}$ | Rate constant for lysis and decay |
| $K_{O,H}$ | KO2 | 0.20 | 0.20 | g O$_2$ m$^{-3}$ | Saturation/inhibition coefficient for oxygen |
| $K_F$ | KF | 4.00 | 4.00 | g COD m$^{-3}$ | Saturation coefficient for growth on $S_F$ |
| $K_{fe}$ | Kfe | 4.00 | 4.00 | g COD m$^{-3}$ | Saturation coefficient for fermentation of $S_F$ |
| $K_A$ | KA | 4.00 | 4.00 | g COD m$^{-3}$ | Saturation coefficient for growth on $S_A$ |
| $K_{NO_3,H}$ | KNO3 | 0.50 | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrate |
| $K_{NO_2,H}$ | KNO2 |  | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrite |
| $K_{NOX,H}$ | KNOx |  | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrogen oxides |
| $K_{NH_4,H}$ | KNH4 | 0.05 | 0.05 | g N m$^{-3}$ | Saturation coefficient for ammonium (nutrient) |
| $K_{PO_4,H}$ | KPO4 | 0.01 | 0.01 | g P m$^{-3}$ | Saturation coefficient for phosphate (nutrient) |
| $K_{ALK,H}$ | KALK | 0.10 | 0.10 | mol HCO$_3^-$ m$^{-3}$ | Saturation coefficient for alkalinity |

### Accumulating phosphorus organisms, $X_{PAO}$

| Standardized symbol | Source label | ASM2d value | Extended model value | Units | Description |
| --- | --- | ---: | ---: | --- | --- |
| $q_{PHA}$ | qPHA | 3.00 | 5.00* | g $X_{PHA}$ (g $X_{PAO}$)$^{-1}$ d$^{-1}$ | Rate constant for storage of $X_{PHA}$ |
| $q_{PP}$ | qPP | 1.50 | 0.60* | g $X_{PHA}$ (g $X_{PAO}$)$^{-1}$ d$^{-1}$ | Rate constant for storage of $X_{PP}$ |
| $\mu_{PAO}$ | $\mu$PAO | 1.00 | 0.56* | d$^{-1}$ | Maximum growth rate of PAO |
| $\eta^{PAO}_{NO_3}$ | nNO3 | 0.60 | 0.07* | - | Reduction factor for denitrification |
| $\eta^{PAO}_{NO_2}$ | nNO2 |  | 0.90* | - | Reduction factor for denitrification via nitrite |
| $b_{PAO}$ | bPAO | 0.20 | 0.20 | d$^{-1}$ | Lysis rate of $X_{PAO}$ |
| $b_{PP}$ | bPP | 0.20 | 0.20 | d$^{-1}$ | Lysis rate of $X_{PP}$ |
| $b_{PHA}$ | bPHA | 0.20 | 0.20 | d$^{-1}$ | Lysis rate of $X_{PHA}$ |
| $K_{O,PAO}$ | KO2 | 0.20 | 0.20 | g O$_2$ m$^{-3}$ | Saturation/inhibition coefficient for oxygen |
| $K_{NO_3,PAO}$ | KNO3 | 0.50 | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrate |
| $K_{NO_2,PAO}$ | KNO2 |  | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrite |
| $K_{NOX,PAO}$ | KNOx |  | 0.50 | g N m$^{-3}$ | Saturation/inhibition coefficient for nitrogen oxides |
| $K_A$ | KA | 4.00 | 4.00 | g COD m$^{-3}$ | Saturation coefficient for growth on $S_A$ |
| $K_{NH_4,PAO}$ | KNH4 | 0.05 | 0.05 | g N m$^{-3}$ | Saturation coefficient for ammonium |
| $K_{PS}$ | KPS | 0.20 | 0.20 | g P m$^{-3}$ | Saturation coefficient for phosphorus in storage of PP |
| $K_{PO_4,PAO}$ | KPO4 | 0.01 | 0.01 | g P m$^{-3}$ | Saturation coefficient for phosphate (nutrient) |
| $K_{ALK,PAO}$ | KALK | 0.10 | 0.10 | mol HCO$_3^-$ m$^{-3}$ | Saturation coefficient for alkalinity |
| $K_{PP}$ | KPP | 0.01 | 0.01 | g $X_{PP}$ (g $X_{PAO}$)$^{-1}$ | Saturation coefficient for poly-phosphate |
| $K_{MAX}$ | KMAX | 0.34 | 0.34 | g $X_{PP}$ (g $X_{PAO}$)$^{-1}$ | Maximum ratio $X_{PP}/X_{PAO}$ |
| $K_{IPP}$ | KIPP | 0.02 | 0.02 | g $X_{PP}$ (g $X_{PAO}$)$^{-1}$ | Inhibition coefficient for PP storage |
| $K_{PHA}$ | KPHA | 0.01 | 0.01 | g $X_{PHA}$ (g $X_{PAO}$)$^{-1}$ | Saturation coefficient for PHA |

### Nitrifying organisms, $X_{AOB}$ and $X_{NOB}$

| Standardized symbol | Source label | ASM2d value | Extended model value | Units | Description |
| --- | --- | ---: | ---: | --- | --- |
| $\mu_{AOB}$ | $\mu$AOB |  | 1.81 | d$^{-1}$ | Maximum growth rate of $X_{AOB}$ |
| $\mu_{NOB}$ | $\mu$NOB |  | 1.52 | d$^{-1}$ | Maximum growth rate of $X_{NOB}$ |
| $b_{AOB}$ | bAOB |  | 0.20** | d$^{-1}$ | Decay rate of $X_{AOB}$ |
| $b_{NOB}$ | bNOB |  | 0.17** | d$^{-1}$ | Decay rate of $X_{NOB}$ |
| $K_{O,AOB}$ | KO2, AOB |  | 0.74** | g O$_2$ m$^{-3}$ | Saturation/inhibition coefficient for oxygen |
| $K_{O,NOB}$ | KO2, NOB |  | 1.75** | g O$_2$ m$^{-3}$ | Saturation/inhibition coefficient for oxygen |
| $K_{NH_4,AOB}$ | KNH4, AOB |  | 0.50 | g N m$^{-3}$ | Saturation coefficient for ammonium |
| $K_{NO_2,NOB}$ | KNO2, NOB |  | 0.50 | g N m$^{-3}$ | Saturation coefficient for nitrite |
| $K_{ALK,nit}$ | KALK | 0.50 | 0.50 | mol HCO$_3^-$ m$^{-3}$ | Saturation coefficient for alkalinity |
| $K_{PO_4,nit}$ | KPO4 | 0.01 | 0.01 | g P m$^{-3}$ | Saturation coefficient for phosphate |

### Phosphorus precipitation and redissolution

| Standardized symbol | Source label | ASM2d value | Extended model value | Units | Description |
| --- | --- | ---: | ---: | --- | --- |
| $k_{PRE}$ | kPRE | 1.00 | 1.00 | m$^3$ (g Fe(OH)$_3$)$^{-1}$ d$^{-1}$ | Rate constant for P precipitation |
| $k_{RED}$ | kRED | 0.60 | 0.60 | d$^{-1}$ | Rate constant for redissolution |
| $K_{ALK,chem}$ | KALK | 0.50 | 0.50 | mol HCO$_3^-$ m$^{-3}$ | Saturation coefficient for alkalinity |

Notes: `*` calibrated parameters. `**` data from Jubany et al. (2008).

## Table S6. Stoichiometric parameters for the two-step nitrification and two-step denitrification model at $T = 20^{\circ}$C and pH = 7.5

| Standardized symbol | Source label | Value | Units | Description |
| --- | --- | ---: | --- | --- |
| $i_{N,SI}$ | iNSI | 0.01 | g N g COD$^{-1}$ | N content of inert COD $S_I$ |
| $i_{N,SF}$ | iNSF | 0.00 | g N g COD$^{-1}$ | N content of fermentable substrates $S_F$ |
| $i_{N,XI}$ | iNXI | 0.02 | g N g COD$^{-1}$ | N content of inert particulate COD $X_I$ |
| $i_{N,XS}$ | iNXS | 0.00 | g N g COD$^{-1}$ | N content of slowly biodegradable substrates $X_S$ |
| $i_{N,BM}$ | iNBM | 0.07 | g N g COD$^{-1}$ | N content of biomass: $X_H$, $X_{PAO}$, $X_{AOB}$, and $X_{NOB}$ |
| $i_{P,SI}$ | iPSI | 0.00 | g P g COD$^{-1}$ | P content of inert COD $S_I$ |
| $i_{P,SF}$ | iPSF | 0.00 | g P g COD$^{-1}$ | P content of fermentable substrates $S_F$ |
| $i_{P,XI}$ | iPXI | 0.01 | g P g COD$^{-1}$ | P content of inert particulate COD $X_I$ |
| $i_{P,XS}$ | iPXS | 0.00 | g P g COD$^{-1}$ | P content of slowly biodegradable substrates $X_S$ |
| $i_{P,BM}$ | iPBM | 0.02 | g P g COD$^{-1}$ | P content of biomass: $X_H$, $X_{PAO}$, $X_{AOB}$, and $X_{NOB}$ |
| $i_{TSS,XI}$ | iTSSXI | 0.75 | g TSS g COD$^{-1}$ | TSS to COD ratio for $X_I$ |
| $i_{TSS,XS}$ | iTSSXS | 0.75 | g TSS g COD$^{-1}$ | TSS to COD ratio for $X_S$ |
| $i_{TSS,BM}$ | iTSSBM | 0.90 | g TSS g COD$^{-1}$ | TSS to COD ratio for biomass: $X_H$, $X_{PAO}$, $X_{AOB}$, and $X_{NOB}$ |
| $f_{SI}$ | fSI | 0.00 | g COD g COD$^{-1}$ | Production of $S_I$ in hydrolysis |
| $f_{XI}$ | fXI | 0.10 | g COD g COD$^{-1}$ | Fraction of inert COD generated in biomass lysis |
| $Y_H$ | YH | 0.625 | g COD g COD$^{-1}$ | Yield coefficient for heterotrophic biomass $X_H$ |
| $Y_{PAO}$ | YPAO | 0.625 | g COD g COD$^{-1}$ | Yield coefficient for PAO biomass ($X_{PAO}$) |
| $Y_{PO4}$ | YPO4 | 0.40 | g P g COD$^{-1}$ | PP requirement (PO$_4$ release) per PHA stored |
| $Y_{PHA}$ | YPHA | 0.20 | g COD g P$^{-1}$ | PHA requirement for PP storage |
| $Y_{AOB}$ | YAOB** | 0.18 | g COD g N$^{-1}$ | Yield coefficient for autotrophic biomass ($X_{AOB}$) |
| $Y_{NOB}$ | YNOB** | 0.08 | g COD g N$^{-1}$ | Yield coefficient for autotrophic biomass ($X_{NOB}$) |

Note: `**` data from Jubany et al. (2008).

## References

- Henze, M., Gujer, W., Mino, T., van Loosdrecht, M. (2000) Activated Sludge Models ASM1, ASM2, ASM2d, ASM3, IWA Publishing, London.
- Jubany, I., Carrera, J., Lafuente, J., Baeza, J.A. (2008) Start-up of a nitrification system with automatic control to treat highly concentrated ammonium wastewater: Experimental results and modeling. Chem. Eng. J. 144 (3), 407-419.