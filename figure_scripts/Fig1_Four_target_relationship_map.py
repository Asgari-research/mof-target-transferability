#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1: Target Analysis
Panel a: Spearman correlation heatmap (target similarity)
Panel b: CO2 (0.015 bar) uptake distribution histogram
Panel c: CO2 (0.15 bar) uptake distribution histogram
Panel d: CH4 (5.8 bar) vs CH4 (65 bar) scatter plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib

# Set Arial as the default font
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.linewidth'] = 0.8

# Define paths
main_figures_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\main_figures")
output_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PANEL a: Spearman Correlation Heatmap
# ============================================================================
print("Loading Panel A data...")
spearman_file = main_figures_dir / "Figure1_panelA_target_spearman_matrix.csv"

if spearman_file.exists():
    df_spearman = pd.read_csv(spearman_file, index_col=0)

    # Rename columns and index with proper formatting
    rename_dict = {
        'ch4_58': 'CH$_4$ (5.8 bar)',
        'ch4_65': 'CH$_4$ (65 bar)',
        'co2_0015': 'CO$_2$ (0.015)',
        'co2_015': 'CO$_2$ (0.15)'
    }

    df_spearman = df_spearman.rename(index=rename_dict, columns=rename_dict)
    print(f"  Loaded Spearman matrix: {df_spearman.shape}")
else:
    raise FileNotFoundError(f"Spearman correlation CSV not found: {spearman_file}. Please provide the file.")

# ============================================================================
# PANEL b: CO2 (0.015 bar) Uptake Distribution
# ============================================================================
print("Loading Panel B data...")
co2_0015_file = main_figures_dir / "Figure1_panelB_co2_0015_hist_bins.csv"

if co2_0015_file.exists():
    df_co2_0015 = pd.read_csv(co2_0015_file)
    print(f"  Loaded CO2 0.015 bar data: {len(df_co2_0015)} bins")
else:
    raise FileNotFoundError(f"CO2 0.015 bar data CSV not found: {co2_0015_file}. Please provide the file.")

# ============================================================================
# PANEL c: CO2 (0.15 bar) Uptake Distribution
# ============================================================================
print("Loading Panel C data...")
co2_015_file = main_figures_dir / "Figure1_panelC_co2_015_hist_bins.csv"

if co2_015_file.exists():
    df_co2_015 = pd.read_csv(co2_015_file)
    print(f"  Loaded CO2 0.15 bar data: {len(df_co2_015)} bins")
else:
    raise FileNotFoundError(f"CH4 scatter data CSV not found: {co2_015_file}. Please provide the file.")

# ============================================================================
# PANEL d: CH4 (5.8 bar) vs CH4 (65 bar) Scatter
# ============================================================================
print("Loading Panel D data...")
scatter_file = main_figures_dir / "Figure1_panelD_ch4_58_vs_ch4_65_scatter.csv"

if scatter_file.exists():
    df_scatter = pd.read_csv(scatter_file)
    print(f"  Loaded scatter data: {len(df_scatter)} points")
    # Assuming columns are named appropriately
    # Try to detect column names
    if 'ch4_58' in df_scatter.columns and 'ch4_65' in df_scatter.columns:
        x_col = 'ch4_58'
        y_col = 'ch4_65'
    elif len(df_scatter.columns) >= 2:
        x_col = df_scatter.columns[0]
        y_col = df_scatter.columns[1]
    else:
        x_col, y_col = 'x', 'y'
else:

    raise FileNotFoundError(f"CH4 scatter data CSV not found: {scatter_file}. Please provide the file.")

# ============================================================================
# CREATE FIGURE WITH 2x2 SUBPLOTS
# ============================================================================
fig = plt.figure(figsize=(18, 16))

# Create GridSpec for better control

gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.35,
                      left=0.08, right=0.95, bottom=0.08, top=0.92)


# Panel A: Heatmap (top-left)
ax_a = fig.add_subplot(gs[0, 0])
heatmap = sns.heatmap(
    df_spearman,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    vmin=-1.0,
    vmax=1.0,
    center=0,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={
        'shrink': 0.8,
        'label': 'Spearman Correlation',
        'aspect': 30
    },
    ax=ax_a
)

# Customize heatmap
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Spearman Correlation', fontsize=22, fontweight='bold',labelpad=15)

ax_a.set_title('a)', loc='left', fontsize=24, fontweight='bold', pad=25)
ax_a.set_title('Target Similarity Matrix', fontsize=16, fontweight='bold', pad=15)
plt.setp(ax_a.get_xticklabels(), rotation=45, ha='right', fontsize=20)
plt.setp(ax_a.get_yticklabels(), rotation=0, fontsize=20)
ax_a.set_ylabel('')

# Set font for annotation text
for text in heatmap.texts:
    text.set_fontsize(18)

# Panel B: CO2 (0.015 bar) histogram (top-right)
ax_b = fig.add_subplot(gs[0, 1])

# Prepare data for histogram
if 'bin_left' in df_co2_0015.columns and 'bin_right' in df_co2_0015.columns:
    bin_left = df_co2_0015['bin_left'].values
    bin_right = df_co2_0015['bin_right'].values
    counts = df_co2_0015['count'].values
    widths = bin_right - bin_left
    ax_b.bar(bin_left, counts, width=widths, color='#1f77b4',
             alpha=0.7, edgecolor='black', linewidth=0.5, align='edge')
elif 'bin_center' in df_co2_0015.columns:
    bin_center = df_co2_0015['bin_center'].values
    counts = df_co2_0015['count'].values
    width = 0.05 if len(bin_center) > 0 else 0.1
    ax_b.bar(bin_center, counts, width=width, color='#1f77b4',
             alpha=0.7, edgecolor='black', linewidth=0.5)

# Calculate statistics
total_count = counts.sum()
bin_center_vals = (bin_left + bin_right) / 2 if 'bin_left' in df_co2_0015.columns else bin_center
mean_uptake = (bin_center_vals * counts).sum() / total_count
cumsum = counts.cumsum()
median_idx = np.searchsorted(cumsum, total_count / 2)
median_uptake = bin_center_vals[median_idx] if median_idx < len(bin_center_vals) else bin_center_vals[-1]
p95_idx = np.searchsorted(cumsum, total_count * 0.95)
p95_uptake = bin_center_vals[p95_idx] if p95_idx < len(bin_center_vals) else bin_center_vals[-1]

# Add statistical lines

ax_b.axvline(x=mean_uptake, color='red', linestyle='-', linewidth=2, label='Mean')
ax_b.axvline(x=median_uptake, color='blue', linestyle='--', linewidth=2, label='Median')
ax_b.axvline(x=p95_uptake, color='gray', linestyle=':', linewidth=2, alpha=0.8, label='95th')


# Styling
ax_b.set_xlabel('Uptake (mmol/g)', fontsize=20, fontweight='bold')
ax_b.set_ylabel('Counts', fontsize=20, fontweight='bold')
ax_b.set_title('b)', loc='left', fontsize=24, fontweight='bold',pad=15)
ax_b.set_title('CO$_2$ (0.015 bar) Distribution', fontsize=16, fontweight='bold')
ax_b.set_xlim(0, 6)
ax_b.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax_b.set_axisbelow(True)
ax_b.legend(loc='upper right', fontsize=20, frameon=True, fancybox=True)

# Format y-axis with K
y_ticks = ax_b.get_yticks()
y_ticks = [tick for tick in y_ticks if tick >= 0]
ax_b.set_yticks(y_ticks)
y_labels = [f'{int(y / 1000)}K' if y >= 1000 else str(int(y)) if y == int(y) else f'{y:.0f}' for y in y_ticks]
ax_b.set_yticklabels(y_labels,fontsize=20, fontweight='bold')
ax_b.tick_params(axis='both', labelsize=20)

# Panel C: CO2 (0.15 bar) histogram (bottom-left)
ax_c = fig.add_subplot(gs[1, 0])

if 'bin_left' in df_co2_015.columns and 'bin_right' in df_co2_015.columns:
    bin_left = df_co2_015['bin_left'].values
    bin_right = df_co2_015['bin_right'].values
    counts = df_co2_015['count'].values
    widths = bin_right - bin_left
    ax_c.bar(bin_left, counts, width=widths, color='#ff7f0e',
             alpha=0.7, edgecolor='black', linewidth=0.5, align='edge')
elif 'bin_center' in df_co2_015.columns:
    bin_center = df_co2_015['bin_center'].values
    counts = df_co2_015['count'].values
    width = 0.1 if len(bin_center) > 0 else 0.1
    ax_c.bar(bin_center, counts, width=width, color='#ff7f0e',
             alpha=0.7, edgecolor='black', linewidth=0.5)

# Calculate statistics
total_count = counts.sum()
bin_center_vals = (bin_left + bin_right) / 2 if 'bin_left' in df_co2_015.columns else bin_center
mean_uptake = (bin_center_vals * counts).sum() / total_count
cumsum = counts.cumsum()
median_idx = np.searchsorted(cumsum, total_count / 2)
median_uptake = bin_center_vals[median_idx] if median_idx < len(bin_center_vals) else bin_center_vals[-1]
p95_idx = np.searchsorted(cumsum, total_count * 0.95)
p95_uptake = bin_center_vals[p95_idx] if p95_idx < len(bin_center_vals) else bin_center_vals[-1]

# Add statistical lines


ax_c.axvline(x=mean_uptake, color='red', linestyle='-', linewidth=2, label='Mean')
ax_c.axvline(x=median_uptake, color='blue', linestyle='--', linewidth=2, label='Median')
ax_c.axvline(x=p95_uptake, color='gray', linestyle=':', linewidth=2, alpha=0.8, label='95th')


# Styling
ax_c.set_xlabel('Uptake (mmol/g)', fontsize=20, fontweight='bold')
ax_c.set_ylabel('Counts', fontsize=20, fontweight='bold')
ax_c.set_title('c)', loc='left', fontsize=24, fontweight='bold',pad=15)
ax_c.set_title('CO$_2$ (0.15 bar) Distribution', fontsize=16, fontweight='bold')
ax_c.set_xlim(0, 12)
ax_c.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax_c.set_axisbelow(True)
ax_c.legend(loc='upper right', fontsize=20, frameon=True, fancybox=True)

# Format y-axis with K
y_ticks = ax_c.get_yticks()
y_ticks = [tick for tick in y_ticks if tick >= 0]
ax_c.set_yticks(y_ticks)
y_labels = [f'{int(y / 1000)}K' if y >= 1000 else str(int(y)) if y == int(y) else f'{y:.0f}' for y in y_ticks]
ax_c.set_yticklabels(y_labels,fontsize=14, fontweight='bold')
ax_c.tick_params(axis='both', labelsize=20)

# Panel D: Scatter plot CH4 5.8 bar vs CH4 65 bar (bottom-right)
ax_d = fig.add_subplot(gs[1, 1])

x_data = df_scatter[x_col].values
y_data = df_scatter[y_col].values

# Create 2D histogram for density visualization
hb = ax_d.hexbin(x_data, y_data, gridsize=50, cmap='hot', mincnt=1, alpha=0.7)
cbar_d = plt.colorbar(hb, ax=ax_d, shrink=0.8)
cbar_d.set_label('Density', fontsize=22, fontweight='bold',labelpad=15)

# Styling
ax_d.set_xlabel('CH$_4$ Uptake at 5.8 bar (mmol/g)', fontsize=20, fontweight='bold')
ax_d.set_ylabel('CH$_4$ Uptake at 65 bar (mmol/g)', fontsize=20, fontweight='bold')
ax_d.set_title('d)', loc='left', fontsize=24, fontweight='bold',pad=15)
ax_d.set_title('CH$_4$ (5.8 bar) vs CH$_4$ (65 bar)', fontsize=16, fontweight='bold')
ax_d.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
ax_d.set_axisbelow(True)
ax_d.tick_params(axis='both', labelsize=20)

ax_d.set_xlim(0,12)
ax_d.set_ylim(0,64)

# ============================================================================
# SAVE FIGURE
# ============================================================================
# Save as PNG
output_path_png = output_dir / 'Figure1_four_target_relationship_map.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"\n Figure saved as PNG: {output_path_png}")

# Save as PDF
output_path_pdf = output_dir / 'Figure1_four_target_relationship_map.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Figure saved as PDF: {output_path_pdf}")

# Save summary statistics
stats_summary = pd.DataFrame({
    'Target': ['CO$_2$ (0.015 bar)', 'CO$_2$ (0.15 bar)'],
    'Mean Uptake (mmol/g)': [f'{mean_uptake:.4f}' for mean_uptake in [mean_uptake for _ in range(2)]],
    'Median Uptake (mmol/g)': [f'{median_uptake:.4f}' for median_uptake in [median_uptake for _ in range(2)]],
    '95th Percentile (mmol/g)': [f'{p95_uptake:.4f}' for p95_uptake in [p95_uptake for _ in range(2)]]
})
stats_summary.to_csv(output_dir / 'Figure1_statistics.csv', index=False)
print(f"Statistics saved to: {output_dir / 'Figure1_statistics.csv'}")

print(f"\n Script completed successfully!")
print(f"\nPanels Summary:")
print(f"  • Panel A: Spearman correlation matrix ({df_spearman.shape[0]}x{df_spearman.shape[1]})")
print(f"  • Panel B: CO2 (0.015 bar) histogram with {len(df_co2_0015)} bins")
print(f"  • Panel C: CO2 (0.15 bar) histogram with {len(df_co2_015)} bins")


plt.show()