#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S3: Best-Model Absolute-Error Distributions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib


matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] =18
matplotlib.rcParams['axes.linewidth'] = 0.8

# Define base paths
output_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures")
figure_data_si_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\si_figures")

output_dir.mkdir(parents=True, exist_ok=True)

print("\nChecking for files in si_figures directory:")
if figure_data_si_dir.exists():
    files = list(figure_data_si_dir.glob("*abs_error*"))
    for f in files:
        print(f"  - {f.name}")
else:
    print(f"  Directory not found: {figure_data_si_dir}")


targets = {
    "co2_0015": {
        "name": "CO$_2$(0.015 bar)",
        "color": "#1f77b4",
        "file": None,
        "x_max": 1.0
    },
    "co2_015": {
        "name": "CO$_2$ (0.15 bar)",
        "color": "#ff7f0e",
        "file": None,
        "x_max": 4.0
    },
    "ch4_58": {
        "name": "CH$_4$ (5.8 bar)",
        "color": "#2ca02c",
        "file": None,
        "x_max": 7.0
    },
    "ch4_65": {
        "name": "CH$_4$ (65 bar)",
        "color": "#d62728",
        "file": None,
        "x_max": 8.0
    }
}

# Find the correct files
for target_key in targets:
    possible_names = [
        f"FigureS3_{target_key}_abs_error_bins.csv",
        f"figS3_{target_key}_abs_error_bins.csv",
        f"{target_key}_abs_error_bins.csv",
        f"FigureS3_{target_key}_abs_error.csv",
    ]

    if targets[target_key]["file"] is None:
        for name in possible_names:
            file_path = figure_data_si_dir / name
            if file_path.exists():
                targets[target_key]["file"] = file_path
                print(f"Found {target_key} in si_figures: {name}")
                break

    if targets[target_key]["file"] is None:
        print(f"File not found for {target_key}")


if all(t["file"] is None for t in targets.values()):
    print("\nERROR: No CSV files found!")
    print("Please check the file paths and try again.")
    exit(1)


fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.subplots_adjust(hspace=0.6, wspace=0.5,top=0.9, bottom=0.2)

# Store summary statistics
summary_stats = []

for idx, (target_key, target_info) in enumerate(targets.items()):
    ax = axes[idx // 2, idx % 2]

    bins_df = pd.read_csv(target_info["file"])

    bin_left = bins_df["bin_left"].values
    bin_right = bins_df["bin_right"].values
    bin_center = bins_df["bin_center"].values
    counts = bins_df["count"].values

    widths = bin_right - bin_left

    nonzero_mask = counts > 0
    bin_center_filtered = bin_center[nonzero_mask]
    counts_filtered = counts[nonzero_mask]
    widths_filtered = widths[nonzero_mask]

    total_count = counts.sum()
    mean_abs_error = (bin_center * counts).sum() / total_count

    cumsum = counts.cumsum()
    median_idx = np.searchsorted(cumsum, total_count / 2)
    median_abs_error = bin_center[median_idx] if median_idx < len(bin_center) else bin_center[-1]

    p95_idx = np.searchsorted(cumsum, total_count * 0.95)
    p95_abs_error = bin_center[p95_idx] if p95_idx < len(bin_center) else bin_center[-1]

    summary_stats.append({
        "Target": target_info["name"],
        "Total Structures": f"{total_count:,}",
        "Mean Absolute Error": f"{mean_abs_error:.4f}",
        "Median Absolute Error": f"{median_abs_error:.4f}",
        "95th Percentile": f"{p95_abs_error:.4f}"
    })

    # Create histogram bars (without legend)
    ax.bar(bin_center_filtered, counts_filtered, width=widths_filtered,
           color=target_info["color"], alpha=0.7,
           edgecolor='black', linewidth=0.3)

    # Add statistical lines (without legend)
    ax.axvline(x=mean_abs_error, color='red', linestyle='-', linewidth=1.5)
    ax.axvline(x=median_abs_error, color='blue', linestyle='--', linewidth=1.5)
    ax.axvline(x=p95_abs_error, color='gray', linestyle=':', linewidth=1.3, alpha=0.8)

    # Set x-axis limits
    ax.set_xlim(0, target_info["x_max"])

    # Styling
    ax.set_xlabel('Absolute Error (mmol/g)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Count', fontsize=18, fontweight='bold')
    ax.set_title(target_info["name"], fontsize=18, fontweight='bold', pad=10)

    # Format y-axis ticks with K for thousands
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    y_labels = [f'{int(y / 1000)}K' if y >= 1000 else str(int(y)) for y in y_ticks]
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # Add subplot label
    ax.text(0.02, 1.02, f'{chr(97 + idx)})', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='bottom', ha='right')

# Create a single shared legend below all subplots
legend_elements = [
    plt.Line2D([0], [0], color='red', linestyle='-', linewidth=1.5, label='Mean'),
    plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5, label='Median'),
    plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='95th Percentile'),
]

# Add legend below the figure
fig.legend(handles=legend_elements,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.0),
           ncol=3,
           fontsize=18,
           frameon=True,
           fancybox=False,
           edgecolor='black',
           handlelength=2.5,
           handleheight=1.5,
           handletextpad=0.8,
           borderpad=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Save figure
plt.savefig(output_dir / 'FigureS3_best_model_abs_error_distributions.png', dpi=600, bbox_inches='tight')
plt.savefig(output_dir / 'FigureS3_best_model_abs_error_distributions.pdf', bbox_inches='tight')

print(f"\nFigure saved to: {output_dir}")

# Save summary statistics
if summary_stats:
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'figS3_abs_error_statistics.csv', index=False)
    print(f"Summary statistics saved to: {output_dir / 'figS3_abs_error_statistics.csv'}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

plt.show()