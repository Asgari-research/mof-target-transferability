"""
Figure 5 – Transfer gain / loss composite (ΔR² vs scratch target training).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
data_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\main_figures")
output_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures")
output_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(data_dir / "Figure5_transfer_gain_loss_source.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Clean and prepare data
# ═══════════════════════════════════════════════════════════════════════════

# ── Target-name mapping → publication labels ──────────────────────────────
TARGET_LABELS = {
    "ch4_58"   : r"CH$_4$ 5.8 bar",
    "ch4_65"   : r"CH$_4$ 65 bar",
    "co2_0015" : r"CO$_2$ 0.015 bar",
    "co2_015"  : r"CO$_2$ 0.15 bar",
}

# ── Descriptor family → short label ──────────────────────────────────────
FAMILY_LABELS = {
    "enriched_interpretable" : "Enriched interpretable",
    "enriched_plus_topology" : "Enriched + Topology",
}

# ── Transfer variant → panel title ───────────────────────────────────────
VARIANT_TITLES = {
    "direct_transport"  : "a)  Direct transport",
    "pretrain_finetune" : "b)  Pretrain – fine-tune",
}

# Build readable pair label:  "CH₄ 5.8 → CO₂ 0.015"
df["pair_label"] = (
    df["source_target"].map(TARGET_LABELS)
    + "  →  "
    + df["target_target"].map(TARGET_LABELS)
)
df["family_label"] = df["family_name"].map(FAMILY_LABELS)

metric = "delta_r2_vs_scratch"

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Colour palette
# ═══════════════════════════════════════════════════════════════════════════
PALETTE = {
    "Enriched interpretable": "#1F4E79",
    "Enriched + Topology": "#f79605",
}

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Figure layout
# ═══════════════════════════════════════════════════════════════════════════
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["Arial"],
    "font.size"        : 16,
    "axes.titlesize"   : 16,
    "axes.labelsize"   : 14,
    "xtick.labelsize"  : 12,
    "ytick.labelsize"  : 12,
    "legend.fontsize"  : 12,
})

variants = ["direct_transport", "pretrain_finetune"]


fig, axes = plt.subplots(
    1, 2,
    figsize=(29, 12),
    gridspec_kw={"wspace": 0.8}
)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Draw each panel
# ═══════════════════════════════════════════════════════════════════════════
for ax, variant in zip(axes, variants):

    # ── Subset & sort ────────────────────────────────────────────────────
    sub = (
        df[df["transfer_variant"] == variant]
        [["pair_label", "family_label", metric]]
        .copy()
        .sort_values(metric, ascending=True)
        .reset_index(drop=True)
    )

    # ── Colours per row ───────────────────────────────────────────────────
    bar_colors = [PALETTE[f] for f in sub["family_label"]]

    # ── Horizontal bars  ─────────
    y_pos = np.arange(len(sub))
    bars  = ax.barh(
        y_pos,
        sub[metric],
        color=bar_colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    # ── Y-axis labels (larger font) ──────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["pair_label"], fontsize=18)

    # ── Reference line at ΔR² = 0 ────────────────────────────────────────
    ax.axvline(0, color="#555555", linewidth=1.2, linestyle="--", zorder=3)

    # ── Annotate top-3 most negative bars with their value ───────────────
    worst3 = sub.nsmallest(3, metric).index
    for idx in worst3:
        val = sub.loc[idx, metric]
        ax.text(
            val - 0.04, idx,
            f"{val:.2f}",
            va="center", ha="right",
            fontsize=9, color="white", fontweight="bold"
        )

    # ── Axes formatting  ────────────────────────────────────
    ax.set_title(VARIANT_TITLES[variant], fontsize=18, fontweight="bold",
                 pad=13, loc="left")
    ax.set_xlabel(r"$\Delta R^2$ vs scratch target training",
                  fontsize=18, fontweight="bold",labelpad=3)
    ax.tick_params(axis="x", labelsize=20)
    ax.grid(axis="x", alpha=0.4, linestyle="--", linewidth=0.5)
    ax.grid(axis="y", visible=False)


    for i in range(0, len(sub), 2):
        ax.axhspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0)


    xmin = sub[metric].min()
    xmax = sub[metric].max()
    pad  = (xmax - xmin) * 0.1
    ax.set_xlim(xmin - pad, max(xmax + pad, 0.15))

# ═══════════════════════════════════════════════════════════════════════════
# 5.  Shared legend (larger font)
# ═══════════════════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(color=color, label=label, alpha=0.88)
    for label, color in PALETTE.items()
]
fig.legend(
    handles=legend_handles,
    title="Descriptor family",

    title_fontsize=18,
    fontsize=18,
    loc="lower center",
    bbox_to_anchor=(0.52, 0),
    ncol=2,
    framealpha=0.9,
    edgecolor="#cccccc",
)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Save
# ═══════════════════════════════════════════════════════════════════════════
for ext, dpi in [("png", 300), ("pdf", None)]:
    fpath  = output_dir / f"Figure5_transfer_gain_loss.{ext}"
    kwargs = dict(bbox_inches="tight", dpi=dpi) if dpi else dict(bbox_inches="tight")
    plt.savefig(fpath, **kwargs)
    print(f" Saved {fpath}")

plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 7.  Console summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Worst 5 ΔR² per transfer variant")
print("=" * 70)
for variant in variants:
    sub = df[df["transfer_variant"] == variant].nsmallest(5, metric)
    print(f"\n{variant.replace('_', ' ').title()}")
    print("-" * 70)
    for _, row in sub.iterrows():
        print(f"  {row['pair_label']:<40} {row['family_label']:<28} "
              f"ΔR²={row[metric]:>7.3f}")
print("=" * 70)