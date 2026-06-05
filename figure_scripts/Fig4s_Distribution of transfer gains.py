"""
Figure 4S: Distribution of transfer gains
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'ytick.labelsize': 12,
    'xtick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Define directories
figure_data_si_dir = r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\si_figures"
output_dir = r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# File paths
same_gas_path = os.path.join(figure_data_si_dir, "FigureS4_same_gas_delta_r2_bins.csv")
cross_gas_path = os.path.join(figure_data_si_dir, "FigureS4_cross_gas_delta_r2_bins.csv")

# Load data
same_gas_df = pd.read_csv(same_gas_path)
cross_gas_df = pd.read_csv(cross_gas_path)

# Filter non-zero bins
same_gas_plot = same_gas_df[same_gas_df['count'] > 0].copy()
cross_gas_plot = cross_gas_df[cross_gas_df['count'] > 0].copy()

# Create figure with specific size (suitable for column width)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


# Get consistent bin width
bin_width = 0.1

# Panel A: Same Gas Transfer
bars1 = ax1.bar(same_gas_plot['bin_center'], same_gas_plot['count'],
                width=bin_width, color='#2E86AB', edgecolor='black',
                linewidth=0.8, alpha=0.8)
ax1.set_xlabel('ΔR²', fontsize=14)
ax1.set_ylabel('Count', fontsize=14,labelpad=1.5)
ax1.set_title('a) Same gas', fontsize=12, fontweight='bold', loc='left')
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

# Panel B: Cross Gas Transfer
bars2 = ax2.bar(cross_gas_plot['bin_center'], cross_gas_plot['count'],
                width=bin_width, color='#A23B72', edgecolor='black',
                linewidth=0.8, alpha=0.8)
ax2.set_xlabel('ΔR²', fontsize=14)
ax2.set_ylabel('Count', fontsize=14,labelpad=1.5)
ax2.set_title('b) Cross gas', fontsize=12, fontweight='bold', loc='left')
ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

# Set consistent x-axis limits
x_min = min(-3.0, cross_gas_df['bin_center'].min())
x_max = 0.5
ax1.set_xlim(x_min, x_max)
ax2.set_xlim(x_min, x_max)

y_max = 18
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, y_max)


# Add statistics
same_gas_total = same_gas_plot['count'].sum()
same_gas_mean = np.average(same_gas_plot['bin_center'], weights=same_gas_plot['count'])

cross_gas_total = cross_gas_plot['count'].sum()
cross_gas_mean = np.average(cross_gas_plot['bin_center'], weights=cross_gas_plot['count'])

plt.tight_layout(pad=1.5, w_pad=3.0)


# Save high-resolution figure for publication
output_pdf = os.path.join(output_dir, 'FigureS4_delta_r2_distribution.pdf')
output_png = os.path.join(output_dir, 'FigureS4_delta_r2_distribution.png')

plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight', transparent=False)

print(f"Figure saved to:")
print(f"  PDF: {output_pdf}")
print(f"  PNG: {output_png}")

plt.show()