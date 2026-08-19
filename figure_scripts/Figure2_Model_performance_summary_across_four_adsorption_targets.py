"""
Figure 2: Model performance comparison across adsorption targets

Models:
- Random Forest
- Histogram Gradient Boosting
- MLP
- Ridge
- Dummy

Panel a: Mean test R2
Panel b: Mean test RMSE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# Paths
# =====================================================

input_r2 = r"C:\Me2\python\Target_trasferability\target_transferability_lighter_outputs\results\figure_data\si_figures\FigureS2_r2_heatmap_matrix.csv"

input_rmse = r"C:\Me2\python\Target_trasferability\target_transferability_lighter_outputs\results\figure_data\si_figures\FigureS2_rmse_heatmap_matrix.csv"


output_dir = Path(
    r"C:\Me2\python\Target_trasferability\target_transferability_lighter_outputs\Exclusive_Outputs\figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# Style
# =====================================================

plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"

plt.rcParams["font.size"] = 16

plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18



# =====================================================
# Extract model performance
# =====================================================

def extract_models(csv_path, metric):

    df = pd.read_csv(csv_path)


    config_column = df.columns[0]

    df = df.set_index(config_column)


    print("\nAvailable configurations:")
    print(df.index.tolist())

    model_patterns = {
        "Random Forest": "random_forest",
        "HGB": "hist_gb",
        "MLP": "mlp",
        "Ridge": "ridge",
        "Dummy": "dummy_mean"
    }


    output = []


    for model, pattern in model_patterns.items():


        subset = df[
            df.index.str.contains(
                pattern,
                case=False,
                na=False
            )
        ]


        if subset.empty:

            print(
                "Not found:",
                model
            )

            continue


        if metric == "R2":

            best = subset.max()

        else:

            best = subset.min()


        best["Model"] = model

        output.append(best)



    result = pd.DataFrame(output)

    result = result.set_index(
        "Model"
    )


    return result



# =====================================================
# Plot
# =====================================================


def create_figure(r2, rmse):
    rename = {
        "ch4_58": "CH$_4$\n5.8 bar",
        "ch4_65": "CH$_4$\n65 bar",
        "co2_0015": "CO$_2$\n0.015 bar",
        "co2_015": "CO$_2$\n0.15 bar"
    }

    r2.rename(columns=rename, inplace=True)
    rmse.rename(columns=rename, inplace=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # -------------------------
    # R2 Panel
    # -------------------------
    ax = axes[0]

    # Nature-style colormap: white -> red (for R2)
    im = ax.imshow(
        r2.values,
        cmap="Reds",
        vmin=0.3,
        vmax=1.0,
        aspect="auto"
    )

    ax.set_title("a)", loc="left", fontweight="bold", fontsize=18, pad=15)
    ax.set_title("Mean test $R^2$", loc="center", fontsize=16, pad=15)

    ax.set_xticks(np.arange(len(r2.columns)))
    ax.set_xticklabels(r2.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(r2.index)))
    ax.set_yticklabels(r2.index)


    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            value = r2.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=16,
                fontweight="bold",
                color="black"
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=14)

    # -------------------------
    # RMSE Panel
    # -------------------------
    ax = axes[1]

    # Nature-style colormap: white -> blue (for RMSE - lower is better)
    im = ax.imshow(rmse.values, cmap="Blues", vmin=0, vmax=8.0, aspect="auto")

    ax.set_title("b)", loc="left", fontweight="bold", fontsize=18, pad=15)
    ax.set_title("Mean test RMSE", loc="center", fontsize=16, pad=15)

    ax.set_xticks(np.arange(len(rmse.columns)))
    ax.set_xticklabels(rmse.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rmse.index)))
    ax.set_yticklabels(rmse.index)


    for i in range(rmse.shape[0]):
        for j in range(rmse.shape[1]):
            value = rmse.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=16,
                fontweight="bold",
                color="black"
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()


    plt.savefig(output_dir / "Figure2_model_comparison.png", dpi=600, bbox_inches="tight")
    plt.savefig(output_dir / "Figure2_model_comparison.pdf", bbox_inches="tight")
    plt.show()
# =====================================================
# Main
# =====================================================

if __name__ == "__main__":


    r2 = extract_models(
        input_r2,
        "R2"
    )


    rmse = extract_models(
        input_rmse,
        "RMSE"
    )


    print("\nR2:")
    print(r2)


    print("\nRMSE:")
    print(rmse)


    r2.to_csv(
        output_dir/"Figure2_model_comparison_r2.csv"
    )


    rmse.to_csv(
        output_dir/"Figure2_model_comparison_rmse.csv"
    )


    create_figure(
        r2,
        rmse
    )


    print("Finished!")