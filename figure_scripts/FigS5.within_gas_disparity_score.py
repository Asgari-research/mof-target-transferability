"""
Figure 5S: Within-gas disparity distribution in the PCA sample
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
fig_s5= os.path.join(figure_data_si_dir, "FigureS5_same_gas_gap_bins.csv")


# Load data
fig_s5_df = pd.read_csv(fig_s5)


# Filter non-zero bins
fig_s5_plot = fig_s5_df[fig_s5_df['count'] > 0].copy()


# Create figure with specific size
fig, ax1 = plt.subplots(figsize=(12, 5))


bin_width =1


bars1 = ax1.bar(fig_s5_plot['bin_center'], fig_s5_plot['count'],
                width=bin_width,
                color='#4682B4',
                linewidth=0.8,
                alpha=0.75)



ax1.set_xlabel('Within-gas disparity score', fontsize=16)
ax1.set_ylabel('Count', fontsize=16,labelpad=1.5)
ax1.tick_params(axis='both', labelsize=14)

ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')





fig_s5_total = fig_s5_plot['count'].sum()
fig_s5_mean = np.average(fig_s5_plot['bin_center'], weights=fig_s5_plot['count'])


plt.tight_layout()

# Save high-resolution figure for publication
output_pdf = os.path.join(output_dir, 'FigureS5_within_gas_disparity_score.pdf')
output_png = os.path.join(output_dir, 'FigureS5_within_gas_disparity_score.png')

plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight', transparent=False)

print(f"Figure saved to:")
print(f"  PDF: {output_pdf}")
print(f"  PNG: {output_png}")

plt.show()