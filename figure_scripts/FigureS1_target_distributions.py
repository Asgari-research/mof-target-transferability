#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S1: Target Distributions (Uptake vs Counts)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib


matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['axes.linewidth'] = 0.8

# Define base paths
output_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures")
figure_data_si_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\si_figures")

output_dir.mkdir(parents=True, exist_ok=True)

# Define targets with display settings
targets = {
    "co2_0015": {
        "name": "CO$_2$ (0.015 bar)",
        "color": "#1f77b4"
    },
    "co2_015": {
        "name": "CO$_2$ (0.15 bar)",
        "color": "#ff7f0e"
    },
    "ch4_58": {
        "name": "CH$_4$ (5.8 bar)",
        "color": "#2ca02c"
    },
    "ch4_65": {
        "name": "CH$_4$ (65 bar)",
        "color": "#d62728"
    }
}


# Find and load data files for each target
def find_and_load_data(target_key):

    possible_files = [
        f"FigureS1_{target_key}_hist_bins.csv",
        f"FigureS1_{target_key}_hist_raw.csv",
        f"{target_key}_hist_bins.csv",
        f"{target_key}_hist_raw.csv",
        f"FigureS1_{target_key}_hist.csv",
    ]

    for filename in possible_files:
        file_path = figure_data_si_dir / filename
        if file_path.exists():
            print(f"Found {target_key}: {filename}")
            try:
                df = pd.read_csv(file_path)

                if 'bin_left' in df.columns and 'bin_right' in df.columns and 'count' in df.columns:
                    return df[['bin_left', 'bin_right', 'count']].copy()
                elif 'bin_center' in df.columns and 'count' in df.columns:

                    df['bin_left'] = df['bin_center'] - 0.05
                    df['bin_right'] = df['bin_center'] + 0.05
                    return df[['bin_left', 'bin_right', 'count']].copy()
                elif 'uptake' in df.columns and 'counts' in df.columns:
                    df = df.rename(columns={'uptake': 'bin_center', 'counts': 'count'})
                    df['bin_left'] = df['bin_center'] - 0.05
                    df['bin_right'] = df['bin_center'] + 0.05
                    return df[['bin_left', 'bin_right', 'count']].copy()
                elif len(df.columns) >= 2:
                    df.columns = ['bin_center', 'count']
                    df['bin_left'] = df['bin_center'] - 0.05
                    df['bin_right'] = df['bin_center'] + 0.05
                    return df[['bin_left', 'bin_right', 'count']].copy()
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
                continue

    print(f"  No data file found for {target_key}")
    return None


# Load data for all targets and compute x_max dynamically
for target_key in targets:
    df = find_and_load_data(target_key)
    if df is not None:
        df = df.dropna()
        df = df[df['count'] > 0]
        df = df.sort_values('bin_left')
        targets[target_key]['data'] = df

        # Compute x_max from actual data (max bin_right) + 5% padding
        max_right = df['bin_right'].max()
        padding = 0.05 * max_right
        targets[target_key]['x_max'] = max_right + padding
        print(f"  Loaded {len(df)} data points for {target_key}, x_max = {targets[target_key]['x_max']:.3f}")
    else:
        targets[target_key]['data'] = None
        targets[target_key]['x_max'] = 1.0
        print(f"  WARNING: No data for {target_key}")

# Check if we have at least some data
if all(t.get('data') is None for t in targets.values()):
    print("\nERROR: No data files found!")
    exit(1)


fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Store summary statistics
summary_stats = []

for idx, (target_key, target_info) in enumerate(targets.items()):
    ax = axes[idx // 2, idx % 2]

    if target_info.get('data') is None:
        ax.text(0.5, 0.5, f"No data available for\n{target_info['name']}",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(target_info["name"], fontsize=18, fontweight='bold', pad=10)
        ax.text(0.02, 1.02, f'{chr(97 + idx)})', transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='bottom', ha='left')
        continue

    data = target_info['data']
    bin_left = data['bin_left'].values
    bin_right = data['bin_right'].values
    counts = data['count'].values

    widths = bin_right - bin_left
    bin_center = (bin_left + bin_right) / 2
    total_count = counts.sum()
    mean_uptake = (bin_center * counts).sum() / total_count

    cumsum = counts.cumsum()
    median_idx = np.searchsorted(cumsum, total_count / 2)
    median_uptake = bin_center[median_idx] if median_idx < len(bin_center) else bin_center[-1]

    p95_idx = np.searchsorted(cumsum, total_count * 0.95)
    p95_uptake = bin_center[p95_idx] if p95_idx < len(bin_center) else bin_center[-1]

    summary_stats.append({
        "Target": target_info["name"],
        "Total Structures": f"{total_count:,}",
        "Mean Uptake (mmol/g)": f"{mean_uptake:.4f}",
        "Median Uptake (mmol/g)": f"{median_uptake:.4f}",
        "95th Percentile (mmol/g)": f"{p95_uptake:.4f}"
    })

    # Create bar plot
    ax.bar(bin_left, counts, width=widths,
           color=target_info["color"],
           alpha=0.7,
           edgecolor='black',
           linewidth=0.5,
           align='edge')

    # Add statistical lines
    ax.axvline(x=mean_uptake, color='red', linestyle='-', linewidth=1.8, label='Mean')
    ax.axvline(x=median_uptake, color='blue', linestyle='--', linewidth=1.8, label='Median')
    ax.axvline(x=p95_uptake, color='gray', linestyle=':', linewidth=1.5, alpha=0.9, label='95th Percentile')


    ax.set_xlim(0, target_info["x_max"])
    # ----------------------------------------------------------

    ax.set_xlabel('Uptake (mmol/g)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Counts', fontsize=18, fontweight='bold')
    ax.set_title(target_info["name"], fontsize=18, fontweight='bold', pad=10)

    # Format y-axis ticks with K for thousands
    y_ticks = ax.get_yticks()
    y_ticks = [tick for tick in y_ticks if tick >= 0]
    ax.set_yticks(y_ticks)
    y_labels = []
    for y in y_ticks:
        if y >= 1000:
            y_labels.append(f'{int(y / 1000)}K')
        elif y == int(y):
            y_labels.append(f'{int(y)}')
        else:
            y_labels.append(f'{y:.1f}')
    ax.set_yticklabels(y_labels)

    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add subplot label (a), b), c), d))
    ax.text(0.03, 1.02, f'{chr(97 + idx)})', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='bottom', ha='left')

# Create legend
legend_elements = [
    plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='Mean'),
    plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Median'),
    plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2, alpha=0.8, label='95th Percentile'),
]

# Add legend below the figure
fig.legend(handles=legend_elements,
           loc='lower center',
           bbox_to_anchor=(0.5, 0),
           ncol=3,
           fontsize=18,
           frameon=True,
           fancybox=False,
           edgecolor='black')

# Adjust subplots layout
plt.subplots_adjust(
    left=0.08,
    right=0.95,
    bottom=0.12,
    top=0.94,
    hspace=0.35,
    wspace=0.35
)

# Save figure
plt.savefig(output_dir / 'FigureS1_target_distributions.png', dpi=600, bbox_inches='tight')
plt.savefig(output_dir / 'FigureS1_target_distributions.pdf', bbox_inches='tight')

print(f"\n Figure saved to: {output_dir}")

# Save summary statistics
if summary_stats:
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'figS1_uptake_statistics.csv', index=False)
    print(f"Summary statistics saved to: {output_dir / 'figS1_uptake_statistics.csv'}")
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(summary_df.to_string(index=False))

# Also save the combined data
all_data = []
for target_key, target_info in targets.items():
    if target_info.get('data') is not None:
        df = target_info['data'].copy()
        df['target'] = target_info['name']
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_dir / 'figS1_all_uptake_data.csv', index=False)
    print(f"✓ Combined data saved to: {output_dir / 'figS1_all_uptake_data.csv'}")

print("\nScript completed successfully!")
plt.show()