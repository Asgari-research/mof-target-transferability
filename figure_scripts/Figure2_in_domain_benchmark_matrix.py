"""
Figure 2: In-domain benchmark matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Set publication-ready parameters
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] =20
plt.rcParams["ytick.labelsize"] = 20


def format_model_name(model_name):

    model_mapping = {
        'mlp': 'MLP',
        'random_forest': 'Random Forest',
        'hist_gb': 'HistGB',
        'ridge': 'Ridge',
        'dummy_mean': 'Dummy Mean'
    }

    if model_name in model_mapping:
        return model_mapping[model_name]

    for key, value in model_mapping.items():
        if key.lower() == model_name.lower():
            return value

    return model_name.replace('_', ' ').title()


def format_descriptor_name(family_name):

    descriptor_mapping = {
        'compact_geom': 'Compact Geometry',
        'enriched_interpretable': 'Enriched Interpretable',
        'enriched_plus_topology': 'Enriched Plus Topology',
    }

    for key, value in descriptor_mapping.items():
        if key.lower() == family_name.lower():
            return value

    return family_name.replace('_', ' ').title()


def safe_save_figure(fig, filepath, dpi=300, bbox_inches='tight', facecolor='white', max_attempts=3):

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_attempts):
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
            print(f" Saved: {filepath.name}")
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                print(f"   File in use, retrying in 2 seconds...")
                time.sleep(2)
            else:
                alt_filename = filepath.stem + f"_temp_{int(time.time())}" + filepath.suffix
                alt_filepath = filepath.parent / alt_filename
                fig.savefig(alt_filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
                print(f"   Saved as: {alt_filename}")
                return False
        except Exception as e:
            print(f"  Error: {e}")
            return False
    return False


def save_csv_safe(df, filepath):

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(filepath, index=False)
        print(f"  Saved CSV: {filepath.name}")
        return True
    except PermissionError:
        alt_filename = filepath.stem + f"_temp_{int(time.time())}" + filepath.suffix
        alt_filepath = filepath.parent / alt_filename
        df.to_csv(alt_filepath, index=False)
        print(f"  CSV saved as: {alt_filename}")
        return True


def create_publication_figure(data_dir, output_dir):

    targets_config = [
        {'data_path': data_dir / "Figure2_co2_0015_bar_data.csv", 'title': 'CO$_2$ (0.015 bar)', 'panel': 'a'},
        {'data_path': data_dir / "Figure2_co2_015_bar_data.csv", 'title': 'CO$_2$ (0.15 bar)', 'panel': 'b'},
        {'data_path': data_dir / "Figure2_ch4_58_bar_data.csv", 'title': 'CH$_4$ (5.8 bar)', 'panel': 'c'},
        {'data_path': data_dir / "Figure2_ch4_65_bar_data.csv", 'title': 'CH$_4$ (65 bar)', 'panel': 'd'}
    ]

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], hspace=0.35, wspace=0.5)

    axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    all_configs = {}
    config_counter = 1
    all_values_data = []

    for idx, config in enumerate(targets_config):
        ax = axes[idx]

        if not config['data_path'].exists():
            print(f"  File not found: {config['data_path'].name}")
            continue

        df = pd.read_csv(config['data_path'])
        df_sorted = df.sort_values('r2_mean', ascending=True).copy()

        configs_raw = df_sorted['config'].values
        r2_values = df_sorted['r2_mean'].values
        r2_stds = df_sorted['r2_std'].values if 'r2_std' in df_sorted.columns else [0] * len(r2_values)

        y_positions = []
        y_labels = []
        for i, raw_config in enumerate(configs_raw):
            if raw_config not in all_configs:
                family = raw_config.split('\n')[0] if '\n' in raw_config else raw_config
                model = raw_config.split('\n')[1] if '\n' in raw_config else "Unknown"

                all_configs[raw_config] = {
                    'index': config_counter,
                    'family': family,
                    'model': model,
                    'family_formatted': format_descriptor_name(family),
                    'model_formatted': format_model_name(model)
                }
                config_counter += 1
            y_positions.append(all_configs[raw_config]['index'])
            y_labels.append(all_configs[raw_config]['index'])

            # Store for CSV
            all_values_data.append({
                'Panel': config['panel'].upper(),
                'Target': config['title'].replace('$_$', '').replace('$', ''),
                'Index': all_configs[raw_config]['index'],
                'Descriptor_Family': all_configs[raw_config]['family_formatted'],
                'Model': all_configs[raw_config]['model_formatted'],
                'R²_Mean': r2_values[i],
                'R²_Std': r2_stds[i]
            })

        bars = ax.barh(y_positions, r2_values, xerr=r2_stds,
                       color='#2E86AB', alpha=0.85,
                       edgecolor='black', linewidth=0.8,
                       capsize=4, error_kw={'elinewidth': 1.5, 'capsize': 4})

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=20)
        ax.invert_yaxis()

        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xlim(-0.1, 1.05)
        ax.set_xlabel('Mean Test $R^2$', fontsize=22, fontweight='bold')
        ax.set_ylabel('Model Configuration', fontsize=22, fontweight='bold')

        ax.text(0.02, 1.07, config['panel']+')', transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top', ha='left')

        ax.set_title(f"{config['title']}", fontsize=18, fontweight='bold', pad=15, loc='center')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', labelsize=18)


    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis('off')


    legend_lines = []
    for raw_config, info in sorted(all_configs.items(), key=lambda x: x[1]['index']):
        legend_lines.append(f"{info['index']}: {info['family_formatted']} | {info['model_formatted']}")

    legend_text = "\n\n".join(legend_lines)

    legend_ax.text(0.5, 0.5, legend_text,
               transform=legend_ax.transAxes,
               fontsize=20,
                color = 'darkblue',
               va='center',
               ha='center',
                multialignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                         edgecolor='black', linewidth=1.5),
                   family='Arial')

    legend_ax.text(0.5, 0.5, legend_text,
                   transform=legend_ax.transAxes,
                   fontsize=20,
                   color='black',
                   va='center',
                   ha='center',
                   multialignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                             edgecolor='black', linewidth=1.5),
                   family='Arial')

    legend_ax.text(0.5, 0.91, 'Model Configuration',
                   transform=legend_ax.transAxes,
                   fontsize=20,
                   fontweight='bold',
                   ha='center',
                   va='top')


    plt.subplots_adjust(right=0.85, left=0.06, top=0.92, bottom=0.08, hspace=0.35, wspace=0.35)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "Figure2_in_domain_benchmark_matrix.png"
    pdf_path = output_dir / "Figure2_in_domain_benchmark_matrix.pdf"

    safe_save_figure(fig, png_path, dpi=300, bbox_inches='tight', facecolor='white')
    safe_save_figure(fig, pdf_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\  Figure 2 saved")


    return fig, all_values_data


def save_r2_values_csv(all_values_data, output_dir):

    if not all_values_data:
        print("  No data available")
        return None

    df = pd.DataFrame(all_values_data)


    df['R²_Mean'] = df['R²_Mean'].apply(lambda x: f"{x:.3f}")
    df['R²_Std'] = df['R²_Std'].apply(lambda x: f"{x:.3f}")

    # CSV path
    output_dir = Path(output_dir)
    csv_path = output_dir / "R2_values_all_targets.csv"

    save_csv_safe(df, csv_path)

    print(f"\n  R² values saved as CSV: {csv_path.name}")
    return df


# ==================== Main Execution ====================
if __name__ == "__main__":
    data_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\results\figure_data\main_figures")
    output_dir = Path(r"C:\Me2\python\MCO2\target_transferability_lighter_outputs\Exclusive_Outputs\figures")

    print("=" * 70)
    print("Creating Figure 2")
    print("=" * 70)


    fig, all_values = create_publication_figure(data_dir, output_dir)


    print("\n" + "=" * 70)
    print("Saving R² Values as CSV")
    print("=" * 70)
    save_r2_values_csv(all_values, output_dir)

    print("\n" + "=" * 70)
    print("✓ All completed!")
    print(f"  Output directory: {output_dir}")
    print("\nFiles created:")
    print("  Figure2_in_domain_benchmark_matrix.png/pdf (main figure)")
    print("   R2_values_all_targets.csv (R² values)")
    print("=" * 70)

plt.show()