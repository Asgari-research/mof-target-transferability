# Repository setup checklist

Repository URL intended for this project:

```text
https://github.com/Asgari-research/mof-target-transferability
```

## One-time setup

```bash
git clone https://github.com/Asgari-research/mof-target-transferability.git
cd mof-target-transferability
```

Copy the prepared repository files into the cloned folder, then run:

```bash
git status
git add README.md LICENSE CITATION.cff references.bib requirements.txt environment.yml .gitignore .gitattributes
git add Target_Transferability_revised_version.py figure_scripts data docs results manuscript_assets supplementary_assets
git commit -m "Add target-transferability analysis code and repository documentation"
git push origin main
```

## If the repository is empty and clone fails

```bash
mkdir mof-target-transferability
cd mof-target-transferability
git init
git remote add origin https://github.com/Asgari-research/mof-target-transferability.git
```

Then copy the prepared files into this folder and run the same `git add`, `git commit`, and `git push` steps.

## Before pushing

Run:

```bash
git status
```

Make sure these are **not** staged:

```text
clean_data.csv
all_topology_lists.csv
target_transferability_lighter_outputs/
*.joblib
*.pkl
raw ARC-MOF files
raw CIF files
large model/prediction artifacts unless intentionally archived
```

## Suggested repository description

```text
Code and documentation for auditing target transferability in MOF gas-uptake machine learning.
```

## Suggested topics

```text
metal-organic-frameworks
mof
machine-learning
adsorption
transfer-learning
reproducibility
arc-mof
```
