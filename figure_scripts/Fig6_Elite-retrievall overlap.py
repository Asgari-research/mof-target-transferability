"""
Figure6: Elite-retrieval overlap
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib


matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10

# Define paths
file_path = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\main_figures\Figure6_elite_overlap_matrix.csv")
output_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Read the CSV file with target_key as index
df = pd.read_csv(file_path, index_col=0)

# Rename columns and index with proper formatting
rename_dict = {
    'ch4_58': 'CH$_4$ (5.8 bar)',
    'ch4_65': 'CH$_4$ (65 bar)',
    'co2_0015': 'CO$_2$ (0.015)',
    'co2_015': 'CO$_2$ (0.15)'
}

df_renamed = df.rename(index=rename_dict, columns=rename_dict)



ax = plt.gca()
df_renamed.index.name = None
df_renamed.columns.name = None

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 6))

heatmap = sns.heatmap(
    df_renamed,
    annot=True,
    fmt='.3f',
    cmap='Blues',
    vmin=0.0,
    vmax=1.0,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={
        'shrink': 0.8,
        'label': 'Overlap Score',
        'aspect': 30
    },
    ax=ax
)

# Set Arial font for colorbar label
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)


# Styling with Arial font
plt.xticks(rotation=45, ha='right', fontsize=12 ,fontname='Arial')
plt.yticks(rotation=0, fontsize=12, fontname='Arial')

# Remove the y-axis label
ax.set_ylabel('')

# Set font for annotation text
for text in heatmap.texts:
    text.set_fontname('Arial')
    text.set_fontsize(14)

plt.tight_layout()

# Save the figure
output_path = output_dir / 'Fig6_Elite-retrieval overlap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PDF
output_path_pdf = output_dir / 'Fig6_Elite-retrieval overlap.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"PDF version saved to: {output_path_pdf}")


plt.show()