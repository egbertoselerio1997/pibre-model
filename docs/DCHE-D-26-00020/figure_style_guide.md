# Figure Style Guide For The DCHE Results Draft

This note documents the exact visual technique used to build the retained sample figures currently stored in this folder:

- `figure2_rmse_learning_curve.pdf`
- `figure3_runtime_learning_curve.pdf`
- `figure4_icsor_structure.pdf`
- `figureS1_cod_icsor_structure.pdf`
- `figureS2_tn_icsor_structure.pdf`
- `figureS3_tp_icsor_structure.pdf`
- `figureS4_tss_icsor_structure.pdf`

These figures were created as manuscript-facing sample assets with a lightweight Python script using `matplotlib` and `numpy`.

Three earlier draft figures were intentionally removed from the outline manuscript because they duplicated information already carried more precisely by tables:

- the ranked benchmark bar chart,
- the per-target RMSE heatmap,
- the physical-admissibility bar chart.

Their construction patterns are still documented below as optional recipes because the styling technique remains reusable even though those draft PDFs are no longer kept in this folder.

## Important Submission Note

This markdown file is an internal authoring aid.

The submission guideline for this folder is stricter than a normal docs folder. Before creating the final submission zip, remove this markdown file unless you explicitly intend to upload it as supporting source material.

## Core Visual Intent

The images were designed to feel:

- clean enough for a journal manuscript,
- warm and deliberate rather than default scientific-plot blue,
- structured enough to compare many models at once,
- readable when printed or embedded in a single-column paper,
- consistent across bar charts, line charts, heatmaps, and coefficient visuals.

The goal was not maximal decoration. The goal was disciplined visual character.

Three decisions drive the look:

1. A white page with low-ink, dotted grids instead of heavy panel furniture.
2. A muted earth-and-mineral palette instead of default matplotlib colors.
3. A reserved accent color for `ICSOR` so the eye always knows where the method of interest is.

## The Palette

The exact palette used in the sample figures was:

```python
PALETTE = [
    "#264653",  # deep teal
    "#2A9D8F",  # mineral green
    "#E9C46A",  # muted amber
    "#F4A261",  # warm sand
    "#E76F51",  # coral rust
    "#6D597A",  # muted plum, reserved for ICSOR emphasis
    "#577590",  # steel blue
    "#BC4749",  # brick red
    "#8D99AE",  # slate gray
    "#ADB5BD",  # fog gray
]

ICSOR_COLOR = "#6D597A"
```

### How The Palette Was Used

- `ICSOR_COLOR` was used whenever ICSOR needed to stand apart from the baseline models.
- The other colors were cycled through the competing models.
- Warm colors were used sparingly so the page stayed balanced.
- Gray tones were kept at the end of the palette so weaker or lower-priority series could be visually softened if needed.

## Global Matplotlib Settings

These were the base plotting settings:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
```

### Why These Settings Work

- `font.size = 10` keeps annotations compact without looking cramped.
- `axes.titlesize = 13` gives enough hierarchy to the title without creating poster-style emphasis.
- Tick labels at `9` stay legible when many models appear in the same figure.
- White figure and axes backgrounds keep the manuscript page clean and printer-friendly.

## Layout Rules

These images were built with a few consistent layout rules.

### 1. Always Use Wide, Stable Aspect Ratios

Examples used in the sample script:

- ranked benchmark bar chart: `(9.0, 5.6)`
- learning curve: `(10.2, 6.0)`
- runtime learning curve: `(10.2, 6.0)`
- target heatmap: `(8.8, 5.6)`
- dual-panel admissibility chart: `(11.0, 5.8)`
- target-wise coefficient atlas: `(8.4, 12.8)`

These sizes give enough breathing room for long model names and dense annotations without looking oversized in the manuscript.

### 2. Prefer Horizontal Comparison When Model Names Are Long

For model leaderboards, horizontal bars are easier to scan than vertical bars because:

- labels stay readable,
- ranking is visually obvious,
- values can be printed directly to the right of each bar.

### 3. Use Light Grids, Never Heavy Box Framing

The grids were intentionally quiet:

```python
ax.grid(axis="x", linestyle=":", alpha=0.35)
```

This gives the reader alignment help without taking attention away from the data.

### 4. Annotate With Restraint

Values were added when they helped interpretation directly:

- bar-chart values at the bar ends,
- heatmap cell values inside the matrix,
- no annotation clutter on already dense learning curves.

### 5. Add A Small Footer To Mark Illustrative Assets

The sample figures include a small footer note:

```python
def add_footer(fig):
    fig.text(
        0.99,
        0.01,
        "Illustrative sample layout only; replace with final benchmark exports.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
```

This is useful while drafting because it prevents accidental confusion between demonstration graphics and final benchmark outputs.

For the final submission figures, remove this footer.

## Chart Recipes

The sample image set was built from seven repeatable chart recipes. Four of them correspond to the retained figures in this folder, and three are documented as optional patterns that were removed from the manuscript outline because they duplicated tables.

## Recipe 1: Ranked Benchmark Bar Chart

Originally used in the removed draft asset:

- `figure2_benchmark_ranked_rmse.pdf`

### Design Logic

- Sort models by the primary comparison metric.
- Use a horizontal bar chart.
- Invert the y-axis so the best model appears at the top.
- Highlight `ICSOR` with the reserved plum accent.
- Print metric values just beyond the bar end.

### Core Code Pattern

```python
order = np.argsort(rmse)
ordered_models = [models[i] for i in order]
ordered_rmse = rmse[order]
ordered_colors = [
    ICSOR_COLOR if models[i] == "ICSOR" else PALETTE[i % len(PALETTE)]
    for i in order
]

fig, ax = plt.subplots(figsize=(9.0, 5.6))
ax.barh(ordered_models, ordered_rmse, color=ordered_colors)

for idx, value in enumerate(ordered_rmse):
    ax.text(value + 0.2, idx, f"{value:.2f}", va="center", fontsize=9)

ax.set_xlabel("Aggregate test RMSE")
ax.set_title("Sample Benchmark Ranking by Aggregate Test RMSE")
ax.invert_yaxis()
ax.grid(axis="x", linestyle=":", alpha=0.35)
```

### Why It Looks Good

- The ranking is obvious in under one second.
- The bars carry the visual comparison.
- The value labels remove the need to cross-reference the table for every number.
- ICSOR is visible immediately without shouting.

## Recipe 2: Learning Curves With Soft Uncertainty Bands

Used in the retained figure:

- `figure2_rmse_learning_curve.pdf`

### Design Logic

- Use lines for central tendency.
- Use transparent filled bands for uncertainty.
- Make ICSOR slightly thicker and slightly darker.
- Push the legend below the plot when many models are present.

### Core Code Pattern

```python
fig, ax = plt.subplots(figsize=(10.2, 6.0))

for idx, model in enumerate(models):
    color = ICSOR_COLOR if model == "ICSOR" else PALETTE[idx % len(PALETTE)]
    lw = 2.7 if model == "ICSOR" else 1.8
    alpha = 0.24 if model == "ICSOR" else 0.12

    ax.plot(sizes, curve, label=model, color=color, linewidth=lw)
    ax.fill_between(sizes, curve - band, curve + band, color=color, alpha=alpha)

ax.set_xlabel("Total dataset size")
ax.set_ylabel("Effective test RMSE")
ax.set_title("Sample Learning-Curve Layout for the Repeated Dataset-Size Analysis")
ax.grid(True, linestyle=":", alpha=0.35)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
```

### Why It Looks Good

- The bands give uncertainty without turning the plot muddy.
- The legend outside the axes preserves the plotting area.
- The thicker ICSOR line creates focus without requiring a separate inset.

## Recipe 3: Runtime Learning Curves With A Log-Scaled Y-Axis

Used in the retained figure:

- `figure3_runtime_learning_curve.pdf`

### Design Logic

- Keep the same visual grammar as the RMSE learning curve so performance and runtime can be compared as a pair.
- Use the same palette, the same ICSOR highlight, and the same uncertainty-band treatment.
- Switch only the y-axis to logarithmic scale so large runtime differences remain readable without flattening the faster models.

### Core Code Pattern

```python
fig, ax = plt.subplots(figsize=(10.2, 6.0))

for idx, model in enumerate(models):
    color = ICSOR_COLOR if model == "ICSOR" else PALETTE[idx % len(PALETTE)]
    lw = 2.7 if model == "ICSOR" else 1.8
    alpha = 0.24 if model == "ICSOR" else 0.12

    ax.plot(sizes, curve, label=model, color=color, linewidth=lw)
    ax.fill_between(sizes, np.maximum(0.2, curve - band), curve + band, color=color, alpha=alpha)

ax.set_xlabel("Total dataset size")
ax.set_ylabel("Training runtime (s)")
ax.set_yscale("log")
ax.set_title("Illustrative Runtime Learning-Curve Layout for the Repeated Dataset-Size Analysis")
ax.grid(True, which="both", linestyle=":", alpha=0.35)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
```

### Why It Looks Good

- It visually matches the RMSE learning-curve figure, so readers can compare accuracy scaling and runtime scaling without changing visual vocabulary.
- The logarithmic y-axis prevents slower models from dominating the canvas.
- Minor models remain readable because the lower runtime curves are not crushed against zero.

### When To Use The Log Scale

Use a log y-axis when runtime spans more than roughly one order of magnitude across models or sizes. Keep a linear y-axis only if the runtime range is narrow enough that the slower models do not flatten the rest of the curves.

## Recipe 4: Annotated Heatmap With Dynamic Text Contrast

Originally used in the removed draft asset:

- `figure4_target_rmse_heatmap.pdf`

### Design Logic

- Heatmaps are efficient when many models are compared across a small set of targets.
- Cell values are worth printing because the matrix is still small enough to read.
- The text color changes depending on background intensity.

### Core Code Pattern

```python
fig, ax = plt.subplots(figsize=(8.8, 5.6))
im = ax.imshow(per_target, aspect="auto", cmap="YlOrRd")

ax.set_xticks(np.arange(len(targets)), labels=targets)
ax.set_yticks(np.arange(len(models)), labels=models)
ax.set_title("Sample Per-Target RMSE Heatmap at Maximum Training Size")

for i in range(per_target.shape[0]):
    for j in range(per_target.shape[1]):
        value = per_target[i, j]
        text_color = "black" if value < per_target.max() * 0.55 else "white"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE")
```

### Why It Looks Good

- `YlOrRd` works well for error magnitudes because darker and hotter colors intuitively read as worse.
- The small text remains readable because of the dynamic text-color rule.
- The matrix stays compact enough for single-column or one-and-a-half-column use.

## Recipe 5: Dual-Panel Constraint Comparison

Originally used in the removed draft asset:

- `figure5_physical_admissibility.pdf`

### Design Logic

- Put mass-conservation and non-negativity on separate but aligned panels.
- Share the y-axis so the same model order is preserved.
- Keep both panels visually matched.

### Core Code Pattern

```python
fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), sharey=True)

axes[0].barh(plot_order, mass_violation, color=colors)
axes[0].set_yticks(plot_order, labels=models)
axes[0].invert_yaxis()
axes[0].set_xlabel("Violation frequency (%)")
axes[0].set_title("Mass conservation")
axes[0].grid(axis="x", linestyle=":", alpha=0.35)

axes[1].barh(plot_order, nonneg_violation, color=colors)
axes[1].set_xlabel("Violation frequency (%)")
axes[1].set_title("Non-negativity")
axes[1].grid(axis="x", linestyle=":", alpha=0.35)
```

### Why It Looks Good

- The two constraints can be compared side by side without forcing them onto one overloaded axis.
- Shared y-order keeps the figure cognitively cheap to read.
- The ICSOR highlight is especially effective here because the method is expected to separate sharply on exactly these diagnostics.

## Recipe 6: Main-Text Single-Block Theta_cc Heatmap

Used in the retained figure:

- `figure4_icsor_structure.pdf`

### Design Logic

- Show only the COD `Theta_cc` block in the manuscript so the interpretability discussion stays focused.
- Keep the ASM component basis complete on both axes.
- Draw the heatmap from the lower origin so the component ordering starts at the matrix origin.
- Use a symmetric zero-centered color scale because the coefficients are signed.
- Pair the panel with a single colorbar instead of additional subplots.

### Core Code Pattern

```python
theta_cc = build_target_blocks("COD")["Theta_cc"]
max_abs = float(np.max(np.abs(theta_cc)))
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=200)
image = ax.imshow(theta_cc, cmap="coolwarm", norm=norm, origin="lower", aspect="auto")
ax.set_xticks(np.arange(len(component_labels)), labels=component_labels, rotation=62, ha="right")
ax.set_yticks(np.arange(len(component_labels)), labels=component_labels)
ax.set_title(r"COD-only $\Theta_{cc}$ interaction surface")
fig.colorbar(image, ax=ax, pad=0.02)
```

### Why It Looks Good

- A single dense block is still readable in the main text.
- The panel isolates the interaction surface that the manuscript actually discusses.
- The square matrix reads naturally as a cross-loading map rather than as a collage.

## Recipe 7: Supplementary Target-Wise Coefficient Atlases

Used in the retained figure set:

- `figureS1_cod_icsor_structure.pdf`
- `figureS2_tn_icsor_structure.pdf`
- `figureS3_tp_icsor_structure.pdf`
- `figureS4_tss_icsor_structure.pdf`

### Design Logic

- Build one complete atlas per reported target.
- Keep all seven retained blocks in the same figure: `b`, `W_u`, `W_in`, `Theta_uu`, `Theta_uc`, `Theta_cc`, and `Gamma`.
- Keep the ASM component basis complete in the `W_in`, `Theta_uc`, `Theta_cc`, and `Gamma` panels.
- Draw every heatmap from the lower origin so the axis ordering starts at the mathematical origin. For the operational blocks this means the lower-left entry is `HRT` by `HRT`, followed by `Aeration` as the second coordinate.
- Keep the color scale symmetric around zero for signed coefficients.
- Annotate only the smallest blocks (`b`, `W_u`, and `Theta_uu`) so the dense ASM-component panels stay readable.

### Core Code Pattern

```python
fig = plt.figure(figsize=(8.4, 12.8), dpi=200)
grid = fig.add_gridspec(
    nrows=5,
    ncols=4,
    height_ratios=[0.95, 1.05, 1.25, 3.2, 3.2],
    width_ratios=[1.0, 1.0, 1.0, 0.08],
)

ax_b = fig.add_subplot(grid[0, 0])
ax_w_u = fig.add_subplot(grid[0, 1])
ax_theta_uu = fig.add_subplot(grid[0, 2])
ax_w_in = fig.add_subplot(grid[1, 0:3])
ax_theta_uc = fig.add_subplot(grid[2, 0:3])
ax_theta_cc = fig.add_subplot(grid[3, 0:3])
ax_gamma = fig.add_subplot(grid[4, 0:3])

image = ax_theta_uu.imshow(theta_uu, cmap="coolwarm", norm=norm, origin="lower", aspect="auto")
ax_theta_uu.set_xticks(np.arange(2), labels=["HRT", "Aeration"])
ax_theta_uu.set_yticks(np.arange(2), labels=["HRT", "Aeration"])
ax_theta_uu.set_title(r"$\Theta_{uu}$")

ax_theta_cc.imshow(theta_cc, cmap="coolwarm", norm=norm, origin="lower", aspect="auto")
ax_theta_cc.set_xticks(np.arange(len(component_labels)), labels=component_labels, rotation=62, ha="right")
ax_theta_cc.set_yticks(np.arange(len(component_labels)), labels=component_labels)
ax_theta_cc.set_title(r"$\Theta_{cc}$")

ax_gamma.imshow(gamma, cmap="coolwarm", norm=norm, origin="lower", aspect="auto")
ax_gamma.set_xticks(np.arange(len(component_labels)), labels=component_labels, rotation=62, ha="right")
ax_gamma.set_yticks(np.arange(len(component_labels)), labels=component_labels)
ax_gamma.set_title(r"$\Gamma$")
```

### Why It Looks Good

- The atlas format lets the reader see every retained block for one target without flipping between multiple small figures.
- The lower-origin heatmaps make the operational labels read in the same direction as the mathematical coordinate system.
- Moving all four complete atlases to the supplement preserves completeness without crowding the paper.
- A zero-centered diverging colormap remains visually honest for signed coefficients across all seven blocks.

## Replication Workflow

If you want to recreate this style for future figures, use this workflow.

1. Build the data arrays or pandas tables first.
2. Decide which single visual question the figure must answer.
3. Choose one of the seven recipes above instead of improvising the chart structure.
4. Apply the shared palette and rcParams.
5. Reserve `ICSOR_COLOR` for the model or feature that must be visually tracked across figures.
6. Keep the background white and the grid faint.
7. Annotate only when the annotation reduces table lookups.
8. Export to vector PDF.
9. Use flat filenames if the figure will live in this submission folder.

## Minimal Reusable Script Skeleton

This is the shortest useful starting point for rebuilding the style.

```python
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path("docs/DCHE-D-26-00020")

PALETTE = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#6D597A",
    "#577590",
    "#BC4749",
    "#8D99AE",
    "#ADB5BD",
]
ICSOR_COLOR = "#6D597A"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def add_footer(fig):
    fig.text(
        0.99,
        0.01,
        "Illustrative sample layout only; replace with final benchmark exports.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

# Build figure here
fig, ax = plt.subplots(figsize=(9.0, 5.6))
ax.grid(axis="x", linestyle=":", alpha=0.35)
add_footer(fig)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUTDIR / "your_figure_name.pdf", bbox_inches="tight")
plt.close(fig)
```

## Export Rules

The sample figures were exported like this:

```python
fig.savefig(outdir / "figure_name.pdf", bbox_inches="tight")
```

### Why PDF Was Used

- vector output stays crisp in the manuscript,
- text and line art survive resizing better,
- it matches the surrounding LaTeX workflow naturally.

## If You Want To Recreate The Same Feel With Final Benchmark Outputs

Keep these decisions fixed even when the numbers change:

- keep the palette,
- keep the ICSOR accent color,
- keep the restrained dotted grid,
- keep the same aspect-ratio family,
- keep the same annotation logic,
- keep the final filenames if `manuscript.tex` already references them.

Change only these items:

- replace the illustrative arrays with final notebook outputs,
- replace the footer text or remove it,
- tighten axis labels and captions to the final scientific claim,
- reduce the model set if the manuscript narrows to retained baselines only.

## Recommended Next Refinement

If this visual language will be reused repeatedly, the next logical improvement is to promote it from an inline script into a dedicated plotting helper module or a single reproducible script such as:

- `docs/DCHE-D-26-00020/build_sample_figures.py`

That would make the style reproducible from one command instead of by copying code blocks out of this note.

If needed, this guide can also be converted into a stricter house style with:

- shared theme tokens,
- shared figure factory functions,
- standard title and caption wording,
- automatic ICSOR highlighting,
- manuscript-safe filename enforcement.