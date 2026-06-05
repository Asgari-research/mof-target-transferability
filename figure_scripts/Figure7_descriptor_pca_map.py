"""
Figure 7 – Descriptor-space PCA map and top descriptor loadings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from pathlib import Path

# =============================================================================
# GLOBAL FONT SETTINGS
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.title_fontsize'] = 12

# ── Paths ────────────────────────────────────────────────────────────────────
output_dir      = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures")
figure_data_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\main_figures")
output_dir.mkdir(parents=True, exist_ok=True)

proj_csv = figure_data_dir / "Figure7_descriptor_pca_projection.csv"
load_csv = figure_data_dir / "Figure7_descriptor_pca_top_loadings.csv"

# ── Load data ────────────────────────────────────────────────────────────────
df_proj = pd.read_csv(proj_csv)
df_load = pd.read_csv(load_csv)

# ── Publication rename dictionary ────────────────────────────────────────────
FEATURE_LABELS = {
    "POAVAf"       : "Void Fraction",
    "vASA"         : r"Vol. Surface Area (m$^2$/cm$^3$)",
    "UC_volume"    : r"Unit-Cell Volume (Å$^3$)",
    "AVA"          : r"Acc. Pore Volume (cm$^3$/g)",
    "POAVA"        : r"Probe-Occ. Void Vol. (cm$^3$/g)",
    "Density"      : r"Framework Density (g/cm$^3$)",
    "ASA"          : r"Grav. Surface Area (m$^2$/g)",
    "log_pld_plus1": r"log(PLD + 1) (Å)",
}
df_load["feature"] = df_load["feature"].replace(FEATURE_LABELS)

# ── Prepare loadings ─────────────────────────────────────────────────────────
df_load    = df_load.sort_values("combined_abs_loading", ascending=True).reset_index(drop=True)
features   = df_load["feature"].tolist()
pc1_vals   = df_load["pc1_loading"].tolist()
pc2_vals   = df_load["pc2_loading"].tolist()
n_features = len(features)

# ── Colorblind-safe Wong palette ─────────────────────────────────────────────
PC1_COLOR = "#0072B2"
PC2_COLOR = "#E69F00"

# ── Quantile-normalised colormap ─────────────────────────────────────────────
z          = df_proj["mean_target_z"].dropna()
boundaries = np.unique(np.quantile(z, np.linspace(0, 1, 11)))
norm       = BoundaryNorm(boundaries, ncolors=256)

# ═════════════════════════════════════════════════════════════════════════════
# Layout
# ═════════════════════════════════════════════════════════════════════════════
#   Spacing knobs :
LEFT        = 0.07
RIGHT       = 0.97
TOP         = 0.96
BOTTOM      = 0.11
WSPACE      = 0.04
HSPACE      = 0.04
PANEL_GAP   = 0.8

fig = plt.figure(figsize=(17, 8))


outer = fig.add_gridspec(
    1, 2,
    width_ratios=[1, 1],
    wspace=PANEL_GAP,
    left=LEFT, right=RIGHT,
    top=TOP, bottom=BOTTOM,
)

inner = outer[0].subgridspec(
    2, 4,
    width_ratios =[1.00, 0.03, 0.09, 0.05],
    height_ratios=[0.12, 1.00],
    wspace=WSPACE,
    hspace=HSPACE,
)

ax_top     = fig.add_subplot(inner[0, 0])   # PC1 top marginal
ax_scatter = fig.add_subplot(inner[1, 0])   # main scatter
ax_right   = fig.add_subplot(inner[1, 2])   # PC2 right marginal
ax_cbar    = fig.add_subplot(inner[1, 3])   # colorbar


ax_bars = fig.add_subplot(outer[1])

# ═════════════════════════════════════════════════════════════════════════════
# (a)  Main scatter
# ═════════════════════════════════════════════════════════════════════════════
sc = ax_scatter.scatter(
    df_proj["pc1"], df_proj["pc2"],
    c=df_proj["mean_target_z"],
    cmap="coolwarm",
    norm=norm,
    s=5, alpha=0.50,
    linewidths=0,
    rasterized=True,
)

ax_scatter.axhline(0, color="grey", lw=0.6, alpha=0.4, zorder=0)
ax_scatter.axvline(0, color="grey", lw=0.6, alpha=0.4, zorder=0)
ax_scatter.grid(alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)
ax_scatter.set_xlabel("PC1 (67.4%)", fontweight="bold")
ax_scatter.set_ylabel("PC2 (12.9%)", fontweight="bold")
ax_scatter.tick_params(labelsize=12)

# ═════════════════════════════════════════════════════════════════════════════
# (a)  Top marginal — PC1 distribution
# ═════════════════════════════════════════════════════════════════════════════

ax_top.hist(df_proj["pc1"], bins=80, color="#999999", alpha=0.75, linewidth=0)
ax_top.set_xlim(ax_scatter.get_xlim())
ax_top.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

for spine in ax_top.spines.values():
    spine.set_visible(False)

# ═════════════════════════════════════════════════════════════════════════════
# (a)  Right marginal — PC2 distribution
# ═════════════════════════════════════════════════════════════════════════════
ax_right.hist(df_proj["pc2"], bins=80, color="#999999", alpha=0.75,
              linewidth=0, orientation="horizontal")
ax_right.set_ylim(ax_scatter.get_ylim())
ax_right.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
for spine in ax_right.spines.values():
    spine.set_visible(False)

# ═════════════════════════════════════════════════════════════════════════════
# (a)  Colorbar
# ═════════════════════════════════════════════════════════════════════════════
cbar = fig.colorbar(sc, cax=ax_cbar)
cbar.set_label(
    "Mean standardised\nuptake across 4 targets",
    labelpad=8.5, fontsize=13, fontweight='bold'
)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(boundaries[::2])
cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

# ═════════════════════════════════════════════════════════════════════════════
# (a)  Biplot arrows — top-3 loading vectors
# ═════════════════════════════════════════════════════════════════════════════
x_range     = df_proj["pc1"].max() - df_proj["pc1"].min()
y_range     = df_proj["pc2"].max() - df_proj["pc2"].min()
arrow_scale = 0.25 * min(x_range, y_range)
top3_idx    = df_load.nlargest(3, "combined_abs_loading").index

for idx in top3_idx:
    dx    = df_load.loc[idx, "pc1_loading"] * arrow_scale
    dy    = df_load.loc[idx, "pc2_loading"] * arrow_scale
    short = df_load.loc[idx, "feature"].split()[0]
    ax_scatter.annotate(
        "", xy=(dx, dy), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#333333",
                        lw=1.4, mutation_scale=10),
        zorder=5,
    )
    ox = 0.8 if dx >= 0 else -0.8
    oy = 0.4 if dy >= 0 else -0.4
    ax_scatter.text(
        dx + ox, dy + oy, short,
        fontsize=12, color="#333333", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        zorder=6,
    )

# Panel label (a) on the top-marginal axes
ax_top.text(-0.10, 0.5, "a)", transform=ax_top.transAxes,
            fontsize=18, fontweight="bold", va="center")

# ═════════════════════════════════════════════════════════════════════════════
# (b)  Horizontal bar chart
# ═════════════════════════════════════════════════════════════════════════════
y_pos      = np.arange(n_features)
bar_height = 0.38

ax_bars.barh(
    y_pos - bar_height / 2, pc1_vals, height=bar_height,
    color=PC1_COLOR, alpha=0.88, edgecolor="white", linewidth=0.4,
    label="PC1",
)
ax_bars.barh(
    y_pos + bar_height / 2, pc2_vals, height=bar_height,
    color=PC2_COLOR, alpha=0.88, edgecolor="white", linewidth=0.4,
    label="PC2",
)

# Y-axis labels — uniform weight
ax_bars.set_yticks(y_pos)
ax_bars.set_yticklabels(features, fontsize=14)
ax_bars.yaxis.set_tick_params(pad=4)


x_max_abs = max(abs(v) for v in pc1_vals + pc2_vals)
annot_pos =  x_max_abs + 0.04
annot_neg = -x_max_abs - 0.04

ax_bars.axvline(0, color="#666666", linewidth=1.0, linestyle="--", zorder=3)
ax_bars.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)
ax_bars.set_xlabel("Loading", fontweight="bold")
ax_bars.tick_params(axis="x", labelsize=14)
ax_bars.set_xlim(-(x_max_abs + 0.15), x_max_abs + 0.15)

ax_bars.legend(
    loc="upper left",
    bbox_to_anchor=(0.01, 0.98),
    framealpha=0.9,
    title="Component", title_fontsize=12,
    fontsize=11,
    edgecolor="#cccccc",
)

# Panel label (b)
ax_bars.text(-0.1, 1.05, "b)", transform=ax_bars.transAxes,
             fontsize=18, fontweight="bold", va="bottom")
pos = ax_bars.get_position()
ax_bars.set_position([
    pos.x0,
    pos.y0 - 0.005,
    pos.width,
    pos.height - 0.1
])

# ── Save ──────────────────────────────────────────────────────────────────────
for ext, dpi in [("png", 300), ("pdf", None)]:
    fpath = output_dir / f"Figure7_descriptor_pca.{ext}"
    kwargs = dict(bbox_inches="tight", dpi=dpi) if dpi else dict(bbox_inches="tight")
    plt.savefig(fpath, **kwargs)
    print(f"✓ Saved {fpath}")

plt.show()