"""
Figure S2: Benchmark heatmaps
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"


plt.rcParams["font.size"] = 12


plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

plt.rcParams["figure.titlesize"] = 20


def create_professional_heatmaps(r2_csv_path, rmse_csv_path, output_dir):

    print("Loading data...")
    r2_df = pd.read_csv(r2_csv_path)
    rmse_df = pd.read_csv(rmse_csv_path)

    if 'config' in r2_df.columns:
        r2_df = r2_df.set_index('config')
    if 'config' in rmse_df.columns:
        rmse_df = rmse_df.set_index('config')

    r2_df = r2_df.loc[:, ~r2_df.columns.str.contains('^Unnamed')]
    rmse_df = rmse_df.loc[:, ~rmse_df.columns.str.contains('^Unnamed')]

    column_renaming = {
        'co2_0015': 'CO$_2$ (0.015 bar)',
        'co2_0015_bar': 'CO$_2$(0.015 bar)',
        'co2_015': 'CO$_2$ (0.15 bar)',
        'co2_015_bar': 'CO$_2$ (0.15 bar)',
        'ch4_58': 'CH$_4$(5.8 bar)',
        'ch4_58_bar': 'CH$_4$(5.8 bar)',
        'ch4_65': 'CH$_4$(65 bar)',
        'ch4_65_bar': 'CH$_4$ (65 bar)',
        'co2_0.015': 'CO$_4$ (0.015 bar)',
        'co2_0.15': 'CO$_4$ (0.15 bar)',
        'ch4_5.8': 'CH$_4$ (5.8 bar)',
        'ch4_65.0': 'CH$_4$ (65 bar)'
    }

    r2_df.columns = [column_renaming.get(col, col) for col in r2_df.columns]
    rmse_df.columns = [column_renaming.get(col, col) for col in rmse_df.columns]

    r2_df.index = [idx.replace('_', ' ').title() for idx in r2_df.index]
    rmse_df.index = [idx.replace('_', ' ').title() for idx in rmse_df.index]

    print(f"\nR² Matrix shape: {r2_df.shape}")
    print(f"R² Columns: {r2_df.columns.tolist()}")
    print(f"R² Index: {r2_df.index.tolist()[:5]}...")


    fig, axes = plt.subplots(1, 2, figsize=(30, 22))

    # ==================== R² Heatmap ====================
    ax1 = axes[0]

    im1 = ax1.imshow(r2_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax1.set_xticks(np.arange(len(r2_df.columns)))
    ax1.set_yticks(np.arange(len(r2_df.index)))
    ax1.set_xticklabels(r2_df.columns, rotation=45, ha='right', fontsize=22)
    ax1.set_yticklabels(r2_df.index, fontsize=22)
    ax1.set_title('a)', loc='left', fontweight='bold', fontsize=26)
    ax1.set_title('Mean Test $R^2$', fontweight='bold', fontsize=24, pad=15)

    # Add value labels
    for i in range(len(r2_df.index)):
        for j in range(len(r2_df.columns)):
            value = r2_df.values[i, j]

            if value < 0.3:
                text_color = 'white'
            elif value > 0.7:
                text_color = 'white'
            else:
                text_color = 'black'

            ax1.text(j, i, f'{value:.2f}', ha="center", va="center",
                     color=text_color, fontsize=22, fontweight='bold')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.06)
    cbar1.set_label('$R^2$ Score', fontsize=26, fontweight='bold')
    cbar1.ax.tick_params(labelsize=26)

    # ==================== RMSE Heatmap ====================
    ax2 = axes[1]

    im2 = ax2.imshow(rmse_df.values, cmap='YlOrRd', aspect='auto')

    ax2.set_xticks(np.arange(len(rmse_df.columns)))
    ax2.set_yticks(np.arange(len(rmse_df.index)))
    ax2.set_xticklabels(rmse_df.columns, rotation=45, ha='right', fontsize=22)
    ax2.set_yticklabels(rmse_df.index, fontsize=22)
    ax2.set_title('b)', loc='left', fontweight='bold', fontsize=26)
    ax2.set_title('Mean Test RMSE', fontweight='bold', fontsize=24, pad=15)


    min_val = rmse_df.values.min()
    max_val = rmse_df.values.max()
    mid_val = (min_val + max_val) / 2

    for i in range(len(rmse_df.index)):
        for j in range(len(rmse_df.columns)):
            value = rmse_df.values[i, j]
            text_color = 'white' if value > mid_val else 'black'

            ax2.text(j, i, f'{value:.2f}', ha="center", va="center",
                     color=text_color, fontsize=22, fontweight='bold')

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.07)
    cbar2.set_label('RMSE (mmol/g)', fontsize=26, fontweight='bold')
    cbar2.ax.tick_params(labelsize=26)



    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.5)

    # Save figures
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    plt.savefig(output_dir / "FigureS2_benchmark_heatmaps.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "FigureS2_benchmark_heatmaps.pdf",
                bbox_inches='tight', facecolor='white')

    print(f"\n Heatmaps saved to: {output_dir / 'FigureS2_benchmark_heatmaps.png'}")
    print(f"✓ PDF version saved to: {output_dir / 'FigureS2_benchmark_heatmaps.pdf'}")

    plt.show()
    plt.close()

    return fig


# ==================== Main Execution ====================
if __name__ == "__main__":

    r2_csv_path = r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\si_figures\FigureS2_r2_heatmap_matrix.csv"
    rmse_csv_path = r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\si_figures\FigureS2_rmse_heatmap_matrix.csv"


    output_dir = r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures"

    print("=" * 60)
    print("Creating Professional Figure S2 Heatmaps")
    print("=" * 60)


    from pathlib import Path

    if not Path(r2_csv_path).exists():
        print(f"  Error: R² file not found at: {r2_csv_path}")
    if not Path(rmse_csv_path).exists():
        print(f" Error: RMSE file not found at: {rmse_csv_path}")

    if Path(r2_csv_path).exists() and Path(rmse_csv_path).exists():

        create_professional_heatmaps(r2_csv_path, rmse_csv_path, output_dir)

        print("\n" + "=" * 60)
        print("✓ Heatmaps created successfully!")
        print(f"  Output directory: {output_dir}")
        print("\nColormap used:")
        print("  - R²: RdYlGn (red=low performance, green=high performance)")
        print("  - RMSE: YlOrRd (yellow=low error, red=high error)")
        print("=" * 60)
    else:
        print("\n Cannot create heatmaps: Missing input files")