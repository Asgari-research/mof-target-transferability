# Data directory

Raw ARC--MOF files and derived private input tables are **not tracked** in this repository.

## Required local input

To run the main analysis, place the prepared analysis table at the repository root:

```text
clean_data.csv
```

The main script reads this file from the same folder as `Target_Transferability_revised_version.py`.

Optional file:

```text
all_topology_lists.csv
```

## Why the input data are not included

The analysis uses an ARC--MOF-derived table. The raw database, CIF files, adsorption targets, descriptors, and any derived third-party data should be obtained from the original ARC--MOF source and used according to the original licence and citation requirements.

This repository therefore provides code, documentation, and reproducibility instructions, but does not redistribute ARC--MOF source files.

## ARC--MOF citation

Burner, J.; Luo, J.; White, A.; Mirmiran, A.; Kwon, O.; Boyd, P. G.; Maley, S.; Gibaldi, M.; Simrod, S.; Ogden, V.; Woo, T. K. ARC--MOF: A Diverse Database of Metal--Organic Frameworks with DFT-Derived Partial Atomic Charges and Descriptors for Machine Learning. *Chemistry of Materials* **2023**, 35(3), 900--916. DOI: `10.1021/acs.chemmater.2c02485`.
