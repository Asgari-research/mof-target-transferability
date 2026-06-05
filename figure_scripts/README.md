# Figure scripts

This directory contains the figure-building scripts used for the manuscript and Supplementary Information.

The scripts are preserved as project scripts. Some scripts contain local Windows paths from the original analysis environment. Before rerunning them on a new machine, update the path variables near the top of each script so that they point to the generated `target_transferability_lighter_outputs/` directory.

Recommended workflow:

1. Run the main analysis script first:

```bash
python Target_Transferability_revised_version.py
```

2. Confirm that figure-source CSV files exist under:

```text
target_transferability_lighter_outputs/results/figure_data/main_figures/
target_transferability_lighter_outputs/results/figure_data/si_figures/
```

3. Run the required figure script after adjusting paths if needed.
