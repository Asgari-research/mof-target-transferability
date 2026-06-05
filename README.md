# MOF Target Transferability

Code and repository documentation for the study:

**Stable Models, Unstable Candidates: Target Transferability in MOF Machine Learning for Gas Uptake Prediction**

This repository contains the Python workflow used to audit which conclusions transfer across MOF gas-uptake targets. The project compares CO<sub>2</sub> uptake at 0.015 and 0.15 bar and CH<sub>4</sub> uptake at 5.8 and 65 bar using a strict common ARC–MOF-derived cohort, interpretable descriptor families, shared random splits, in-domain benchmarks, transfer-learning tests, residual-overlap analysis, elite-retrieval overlap, and descriptor-space PCA.

## What is included

- `Target_Transferability_revised_version.py`  
  Main end-to-end analysis pipeline. It builds the strict common cohort, creates shared train/test splits, trains the in-domain models, runs transfer analyses, exports metrics/tables/predictions/checkpoints, and writes figure-source tables.

- `figure_scripts/`  
  Standalone figure-building scripts used to regenerate manuscript and supplementary figures from exported figure-source CSV files.

- `docs/`  
  Reproducibility notes, ARC–MOF data-access notes, repository checklist, and file manifest.

- `data/README.md`  
  Explains which input files are expected and why raw ARC–MOF-derived files are not redistributed.

## What is not included

This repository does **not** redistribute the raw ARC–MOF database, raw CIF files, raw adsorption tables, or any third-party database files. Users must obtain the relevant ARC–MOF source files from the original providers and follow the original licence and citation requirements.

The underlying database publication is:

> Jake Burner, Jun Luo, Andrew White, Adam Mirmiran, Ohmin Kwon, Peter G. Boyd, Stephen Maley, Marco Gibaldi, Scott Simrod, Victoria Ogden, and Tom K. Woo. **ARC–MOF: A Diverse Database of Metal-Organic Frameworks with DFT-Derived Partial Atomic Charges and Descriptors for Machine Learning.** *Chemistry of Materials* **35**(3), 900–916 (2023). DOI: `10.1021/acs.chemmater.2c02485`.

## Expected input files

The main script expects the prepared analysis table in the repository root:

```text
clean_data.csv
```

Optional topology metadata may also be placed in the repository root:

```text
all_topology_lists.csv
```

These files are intentionally ignored by Git. They are derived from third-party data and should not be committed unless redistribution is explicitly permitted.

The required `clean_data.csv` columns include:

```text
filename
Crystalnet
uptake(mmol/g) CO2 at 0.015 bar
uptake(mmol/g) CO2 at 0.15 bar
uptake(mmol/g) methane at 5.8 bar
uptake(mmol/g) methane at 65 bar
UC_volume
Density
ASA
vASA
gASA
NASA
gNASA
vNASA
AVA
AVAf
AVAg
NAVA
NAVAf
NAVAg
POAVA
POAVAf
POAVAg
NPOAVA
NPOAVAf
NPOAVAg
Di
Df
Dif
```

## Installation

A minimal Python environment can be created with either `pip` or `conda`.

Using `pip`:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate mof-target-transferability
```

## Minimal workflow

1. Clone the repository.
2. Obtain ARC–MOF data from the original source and prepare `clean_data.csv`.
3. Place `clean_data.csv` in the repository root.
4. Run the main pipeline:

```bash
python Target_Transferability_revised_version.py
```

The main pipeline writes outputs to:

```text
target_transferability_lighter_outputs/
```

The generated output tree includes:

```text
target_transferability_lighter_outputs/
├── data_processed/
│   └── split_definitions/
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── tables/
│   ├── models/
│   ├── metadata/
│   ├── feature_importance/
│   └── figure_data/
├── manuscript_assets/
│   └── main_figures/
└── supplementary_assets/
    └── si_figures/
```

5. Run the figure scripts only after the corresponding figure-source CSV files have been generated.

## Important note about figure scripts

The uploaded figure scripts are preserved as project scripts. Some contain local Windows paths from the original analysis environment. Before rerunning those scripts on a different machine, update the input/output path variables to point to your local `target_transferability_lighter_outputs/` directory. The analysis code itself is not changed here.

## Main analysis settings

The main script uses:

- five persistent random train/test splits,
- 20% test fraction,
- compact geometry, enriched interpretable, and enriched plus topology descriptor families,
- dummy mean, ridge, random forest, histogram gradient boosting, and shallow MLP models,
- residual-overlap, benchmark-rank transportability, transfer-gain/loss, elite-overlap, PCA, and permutation-importance analyses.

## Citation

If you use this repository, please cite:

1. The associated manuscript, once available.
2. The ARC–MOF database paper:

```bibtex
@article{burner_arcmof_2023,
  title   = {ARC--MOF: A Diverse Database of Metal--Organic Frameworks with DFT-Derived Partial Atomic Charges and Descriptors for Machine Learning},
  author  = {Burner, Jake and Luo, Jun and White, Andrew and Mirmiran, Adam and Kwon, Ohmin and Boyd, Peter G. and Maley, Stephen and Gibaldi, Marco and Simrod, Scott and Ogden, Victoria and Woo, Tom K.},
  journal = {Chemistry of Materials},
  year    = {2023},
  volume  = {35},
  number  = {3},
  pages   = {900--916},
  doi     = {10.1021/acs.chemmater.2c02485}
}
```

## Licence

The code and documentation in this repository are released under the MIT License.

The ARC–MOF database and any derived third-party data remain under their original licences and attribution requirements. This repository does not grant redistribution rights for ARC–MOF source files or any other third-party data.
