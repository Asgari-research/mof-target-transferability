# Reproducibility notes

## Main command

```bash
python Target_Transferability_revised_version.py
```

## Input placement

The prepared input table must be placed beside the main script:

```text
clean_data.csv
```

The optional topology-list file can also be placed there:

```text
all_topology_lists.csv
```

These files are excluded from Git because they are derived from ARC--MOF and may contain third-party data.

## Main output folder

The main script creates:

```text
target_transferability_lighter_outputs/
```

Within this folder, the pipeline writes processed datasets, split definitions, metrics, prediction tables, fitted models, metadata, feature-importance outputs, figure-source CSV files, and figure assets.

## Determinism and splits

The analysis uses fixed random settings in the script:

```text
RANDOM_STATE = 42
OUTER_RANDOM_SPLITS = 5
TEST_SIZE = 0.20
```

The same split definitions are reused across targets, descriptor families, and model classes.

## Computational note

The script limits numerical libraries to a single thread by default and uses `N_JOBS = 1`. This makes runs more stable on desktop machines, but it may be slow for the full ARC--MOF-derived cohort.

## Figure regeneration

Figure scripts should be treated as post-processing scripts. They should be run after the main pipeline has generated the source CSV tables. Several figure scripts contain local absolute paths from the original analysis environment; update those paths before running them elsewhere.

## Recommended final archive

For journal/Zenodo archiving, include:

- source code,
- README and documentation,
- `requirements.txt` or `environment.yml`,
- figure-source CSV files if redistribution is allowed,
- final manuscript figure PDFs/PNGs if redistribution is allowed,
- manuscript and SI source files if desired,
- no raw ARC--MOF database files unless explicitly permitted.
