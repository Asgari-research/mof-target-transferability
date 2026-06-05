#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target Transferability Across Methane and CO2 Tasks
===================================================

Standalone pipeline — works fine in VS Code, Spyder, or plain terminal.

What it does
---------------------
1. Reads clean_data.csv from the same folder as this script.
2. Builds a strict common cohort so every comparison uses the same structures.
3. Sets up three descriptor families: compact geometry, enriched, and topology-augmented.
4. Creates five random 80/20 splits and reuses them across every target and model.
5. Trains each regressor independently per target and logs metrics as it goes.
6. Saves metrics, predictions, fitted models, tables, figures, logs, and checkpoints
   continuously so interrupted runs can be resumed safely.
7. Runs cross-target rank transport, residual overlap, transfer experiments, and elite-list overlap.
8. Produces manuscript-style and SI-style figures/tables as CSV, PKL, PNG, and PDF,
   plus CSV figure-source exports for all main-text and SI figures.

Input
-----
Required:  clean_data.csv  (same folder)

Optional:  all_topology_lists.csv

Output
------
Everything lands in:  target_transferability_outputs/

"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import pickle
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Keep third-party numerical libraries on a single CPU thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_NAME = "target_transferability_lighter"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / f"{PROJECT_NAME}_outputs"

INPUT_CLEAN_DATA = DATA_DIR / "clean_data.csv"
INPUT_TOPOLOGY_LISTS = DATA_DIR / "all_topology_lists.csv"

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

RESUME_IF_AVAILABLE = True
RANDOM_STATE = 42
OUTER_RANDOM_SPLITS = 5
TEST_SIZE = 0.20

# N_JOBS = max(1, min(4, (os.cpu_count() or 2) - 1))
N_JOBS = 1


# Light hyperparameters for desktop‑scale ARC‑MOF analysis

TOPOLOGY_MIN_COUNT = 100
TOPOLOGY_TOP_N = 50
RF_N_ESTIMATORS = 80
RF_MAX_DEPTH = 16
RF_MIN_SAMPLES_LEAF = 5
HGB_MAX_ITER = 140
HGB_MAX_DEPTH = 6
MLP_HIDDEN_LAYERS = (32,)
MLP_MAX_ITER = 220
MLP_TRANSFER_MAX_ITER = 220
FINETUNE_PRETRAIN_EPOCHS = 30
FINETUNE_TARGET_EPOCHS = 50
PERMUTATION_IMPORTANCE_SAMPLE_N = 20000
PERMUTATION_IMPORTANCE_REPEATS = 3
MODEL_COMPRESSION = 3

plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# The four adsorption targets
# ---------------------------------------------------------------------------
TARGETS = {
    "co2_0015": "uptake(mmol/g) CO2 at 0.015 bar",
    "co2_015": "uptake(mmol/g) CO2 at 0.15 bar",
    "ch4_58": "uptake(mmol/g) methane at 5.8 bar",
    "ch4_65": "uptake(mmol/g) methane at 65 bar",
}

ID_COL = "filename"
TOPO_COL = "Crystalnet"


# All geometry columns
RAW_GEOM_COLUMNS = [
    "UC_volume", "Density", "ASA", "vASA", "gASA", "NASA", "gNASA", "vNASA",
    "AVA", "AVAf", "AVAg", "NAVA", "NAVAf", "NAVAg", "POAVA", "POAVAf", "POAVAg",
    "NPOAVA", "NPOAVAf", "NPOAVAg", "Di", "Df", "Dif",
]

# Compact family: six standard Zeo++ descriptors
COMPACT_COLUMNS = ["Density", "ASA", "AVA", "AVAf", "Di", "Df"]

# Base for the enriched family before we add engineered columns
ENRICHED_BASE_COLUMNS = ["UC_volume", "Density", "ASA", "vASA", "NASA", "AVA", "AVAf", "POAVA", "POAVAf", "Di", "Df", "Dif"]

EPS = 1e-9                      # avoid div-by-zero in ratios
TOP_K_FRACTION = 0.05           # top-5% for elite-retrieval overlap
TOP_K_MIN = 100                 # at least 100 structures for elite retrieval
PCA_ANALYSIS_SAMPLE_N = 12000   # subsample for the descriptor-space PCA
TOP_LOADING_FEATURES = 8        # for Table 6


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        ensure_dir(log_path.parent)

    def log(self, message: str) -> None:
        msg = f"[{now_str()}] {message}"
        print(msg, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


LOGGER = RunLogger(OUTPUT_DIR / "run_log.txt")


def fmt_elapsed(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    if minutes < 60:
        return f"{minutes} min {rem:.1f} s"
    hours = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours} h {minutes} min {rem:.1f} s"


def log_stage(message: str) -> float:
    LOGGER.log(f"--- {message} ---")
    return time.time()


def log_stage_done(stage_name: str, t0: float) -> None:
    LOGGER.log(f"--- Completed {stage_name} in {fmt_elapsed(time.time() - t0)} ---")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def safe_json_dump(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def save_dataframe(df: pd.DataFrame, path_no_suffix: Path, index: bool = False) -> None:
    ensure_dir(path_no_suffix.parent)
    shape_text = f"shape={df.shape[0]:,}x{df.shape[1]:,}"
    LOGGER.log(f"Saving dataframe to CSV/PKL: {path_no_suffix} ({shape_text})")
    t0 = time.time()
    df.to_csv(path_no_suffix.with_suffix(".csv"), index=index)
    LOGGER.log(f"  Wrote CSV in {fmt_elapsed(time.time() - t0)}")
    t1 = time.time()
    df.to_pickle(path_no_suffix.with_suffix(".pkl"))
    LOGGER.log(f"  Wrote PKL in {fmt_elapsed(time.time() - t1)}")


def safe_spearman(x, y) -> float:
    try:
        stat = spearmanr(x, y, nan_policy="omit").statistic
        if stat is None or (isinstance(stat, float) and (np.isnan(stat) or np.isinf(stat))):
            return np.nan
        return float(stat)
    except Exception:
        return np.nan


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    spear = safe_spearman(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spear}


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    df_old = load_existing_csv(path)
    df_new = pd.concat([df_old, pd.DataFrame(rows)], ignore_index=True)
    tmp = path.with_suffix(".tmp.csv")
    df_new.to_csv(tmp, index=False)
    tmp.replace(path)


def append_row_csv(path: Path, row: Dict[str, Any]) -> None:
    append_rows_csv(path, [row])


def maybe_save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    ensure_dir(path_no_suffix.parent)
    LOGGER.log(f"Saving figure PNG/PDF: {path_no_suffix}")
    t0 = time.time()
    fig.savefig(path_no_suffix.with_suffix(".png"), bbox_inches="tight")
    LOGGER.log(f"  Wrote PNG in {fmt_elapsed(time.time() - t0)}")
    t1 = time.time()
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    LOGGER.log(f"  Wrote PDF in {fmt_elapsed(time.time() - t1)}")
    plt.close(fig)


def save_figure_source(df: pd.DataFrame, path_no_suffix: Path, index: bool = False) -> None:
    ensure_dir(path_no_suffix.parent)
    LOGGER.log(f"Saving figure-source CSV: {path_no_suffix.with_suffix('.csv').name} (rows={len(df):,}, cols={df.shape[1]:,})")
    df.to_csv(path_no_suffix.with_suffix(".csv"), index=index)


def histogram_bin_table(values, bins: int = 40) -> pd.DataFrame:
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    counts, edges = np.histogram(arr, bins=bins)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": 0.5 * (edges[:-1] + edges[1:]),
            "count": counts,
        }
    )


# ---------------------------------------------------------------------------
# Split definition dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitDef:
    split_id: int
    seed: int
    train_index: List[int]
    test_index: List[int]

# ---------------------------------------------------------------------------
# Output directory tree
# ---------------------------------------------------------------------------

PATHS = {
    "data_processed": OUTPUT_DIR / "data_processed",
    "split_definitions": OUTPUT_DIR / "data_processed" / "split_definitions",
    "metrics": OUTPUT_DIR / "results" / "metrics",
    "predictions": OUTPUT_DIR / "results" / "predictions",
    "tables": OUTPUT_DIR / "results" / "tables",
    "tables_si": OUTPUT_DIR / "results" / "tables" / "SI",
    "models": OUTPUT_DIR / "results" / "models",
    "figures_main": OUTPUT_DIR / "manuscript_assets" / "main_figures",
    "figures_si": OUTPUT_DIR / "supplementary_assets" / "si_figures",
    "metadata": OUTPUT_DIR / "results" / "metadata",
    "feature_importance": OUTPUT_DIR / "results" / "feature_importance",
    "figure_data_main": OUTPUT_DIR / "results" / "figure_data" / "main_figures",
    "figure_data_si": OUTPUT_DIR / "results" / "figure_data" / "si_figures",
}
for p in PATHS.values():
    ensure_dir(p)


# ---------------------------------------------------------------------------
# Data loading and cohort construction
# ---------------------------------------------------------------------------


def assert_required_files() -> None:
    if not INPUT_CLEAN_DATA.exists():
        raise FileNotFoundError(f"Required input file not found: {INPUT_CLEAN_DATA}")


def load_raw_data() -> pd.DataFrame:
    LOGGER.log(f"Loading data from {INPUT_CLEAN_DATA}")
    df = pd.read_csv(INPUT_CLEAN_DATA)

    # Sometimes pandas sneaks in 'Unnamed' columns from old CSVs — drop them
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        LOGGER.log(f"Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # Make sure all the columns we care about are there
    missing = [c for c in [ID_COL, TOPO_COL] + list(TARGETS.values()) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in clean_data.csv: {missing}")

    # Clean up IDs and topology info
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[TOPO_COL] = df[TOPO_COL].fillna("missing").astype(str).str.strip()

    # Make sure target columns are numeric
    for target_col in TARGETS.values():
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Drop duplicate IDs, keep the first occurrence
    if df[ID_COL].duplicated().any():
        ndup = int(df[ID_COL].duplicated().sum())
        LOGGER.log(f"Found {ndup} duplicated '{ID_COL}' entries. Keeping first occurrence.")
        df = df.drop_duplicates(subset=[ID_COL], keep="first").copy()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Make sure geometry columns are numeric
    for col in RAW_GEOM_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pore geometry ratios
    if {"Di", "Df"}.issubset(df.columns):
        df["lcd_pld_ratio"] = df["Di"] / (df["Df"] + EPS)
        df["log_pld_plus1"] = np.log1p(np.clip(df["Df"], a_min=0, a_max=None))
        df["log_lcd_plus1"] = np.log1p(np.clip(df["Di"], a_min=0, a_max=None))
    if {"ASA", "AVA"}.issubset(df.columns):
        df["sa_pv_ratio"] = df["ASA"] / (df["AVA"] + EPS)
    if {"AVAf", "Density"}.issubset(df.columns):
        df["vf_density_ratio"] = df["AVAf"] / (df["Density"] + EPS)
    if "Dif" in df.columns:
        df["cavity_window_gap"] = df["Dif"]

     # Frequency of each topology
    topo_counts = df[TOPO_COL].fillna("missing").astype(str).value_counts(dropna=False)
    df["topology_frequency"] = df[TOPO_COL].map(topo_counts).fillna(0).astype(float)
    return df


def build_common_cohort(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows with complete + finite data in all required columns

    needed_cols = list(TARGETS.values()) + COMPACT_COLUMNS + ENRICHED_BASE_COLUMNS
    needed_cols = sorted(set([c for c in needed_cols if c in df.columns] + [ID_COL, TOPO_COL]))
    LOGGER.log(f"Building strict common cohort using {len(needed_cols)} required columns.")
    cohort = df.dropna(subset=needed_cols).copy()

    numeric_subset = [c for c in needed_cols if c not in [ID_COL, TOPO_COL]]
    finite_mask = np.isfinite(cohort[numeric_subset].select_dtypes(include=[np.number])).all(axis=1)
    cohort = cohort.loc[finite_mask].copy()

    cohort = engineer_features(cohort)
    cohort = cohort.reset_index(drop=True)

    LOGGER.log(f"Strict common cohort size: {len(cohort):,}")
    return cohort


def save_dataset_descriptions(raw_df: pd.DataFrame, common_df: pd.DataFrame) -> None:
    """Write Table 1: per‑target summary stats + cohort size."""
    rows = []
    for name, target_col in TARGETS.items():
        s = pd.to_numeric(raw_df[target_col], errors="coerce")
        rows.append({
            "target_key": name,
            "target_column": target_col,
            "available_n": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        })
    rows.append({"target_key": "strict_common_cohort", "target_column": "all targets jointly", "available_n": int(len(common_df)), "mean": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan})
    save_dataframe(pd.DataFrame(rows), PATHS["tables"] / "table_1_summary_statistics", index=False)


def get_descriptor_families(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    enriched_cols = ["UC_volume", "Density", "ASA", "vASA", "NASA", "AVA", "AVAf", "POAVA",
                     "POAVAf", "Di", "Df", "Dif", "lcd_pld_ratio", "cavity_window_gap",
                     "sa_pv_ratio", "vf_density_ratio", "log_pld_plus1", "log_lcd_plus1"]

    compact = [c for c in COMPACT_COLUMNS if c in df.columns]
    enriched = [c for c in enriched_cols if c in df.columns]

    families = {
        "compact_geom": {"numeric": compact, "categorical": []},
        "enriched_interpretable": {"numeric": enriched, "categorical": []},
    }
    if TOPO_COL in df.columns:
        families["enriched_plus_topology"] = {"numeric": enriched + (["topology_frequency"] if "topology_frequency" in df.columns else []), "categorical": [TOPO_COL]}
    meta = []
    for fam, cols in families.items():
        meta.append({"family": fam, "n_numeric": len(cols["numeric"]), "n_categorical": len(cols["categorical"]), "numeric_columns": ", ".join(cols["numeric"]), "categorical_columns": ", ".join(cols["categorical"])})
    save_dataframe(pd.DataFrame(meta), PATHS["metadata"] / "descriptor_families", index=False)
    return families


def prepare_family_frame(
    df: pd.DataFrame,
    family_def: Dict[str, List[str]],
    min_topology_count: int = TOPOLOGY_MIN_COUNT,
    topology_top_n: int = TOPOLOGY_TOP_N,
) -> pd.DataFrame:
    cols = [ID_COL] + family_def["numeric"] + family_def["categorical"]
    out = df[cols].copy()

    if TOPO_COL in family_def["categorical"]:
        # collapse rare/unseen topologies into "other"
        topo = out[TOPO_COL].fillna("missing").astype(str)
        vc = topo.value_counts()
        keep = set(vc[vc >= min_topology_count].head(topology_top_n).index)
        out[TOPO_COL] = topo.where(topo.isin(keep), other="other")
    return out


def build_or_load_splits(df: pd.DataFrame) -> List[SplitDef]:
    split_path = PATHS["split_definitions"] / "outer_random_splits.json"
    if RESUME_IF_AVAILABLE and split_path.exists():
        LOGGER.log("Loading existing split definitions.")
        data = json.loads(split_path.read_text(encoding="utf-8"))
        splits = [SplitDef(**item) for item in data]
        LOGGER.log(f"Loaded {len(splits)} existing split definitions from disk.")
        return splits
    LOGGER.log("Creating new persistent random splits.")
    splits = []
    for split_id in range(OUTER_RANDOM_SPLITS):
        seed = RANDOM_STATE + split_id
        LOGGER.log(f"  Creating split {split_id + 1}/{OUTER_RANDOM_SPLITS} with seed={seed}")
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, random_state=seed, shuffle=True)
        splits.append(SplitDef(split_id=split_id, seed=seed, train_index=train_idx.tolist(), test_index=test_idx.tolist()))
    safe_json_dump([asdict(s) for s in splits], split_path)
    LOGGER.log(f"Saved {len(splits)} split definitions to {split_path}")
    return splits


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers = [("num", numeric_pipe, numeric_cols)]
    if categorical_cols:
        cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=np.float32))])
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def get_model_catalog() -> Dict[str, Any]:

    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features="sqrt",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        ),
        "hist_gb": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_depth=HGB_MAX_DEPTH,
            max_iter=HGB_MAX_ITER,
            l2_regularization=1e-3,
            random_state=RANDOM_STATE,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation="relu",
            alpha=2e-3,
            learning_rate_init=8e-4,
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(numeric_cols: List[str], categorical_cols: List[str], model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", make_preprocessor(numeric_cols, categorical_cols)), ("model", model)])


def benchmark_cache_key(target_key: str, family_name: str, model_name: str, split_id: int) -> str:
    return f"{target_key}__{family_name}__{model_name}__split{split_id}"


def fit_and_predict(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, model_name, model_obj):
    pipe = make_pipeline(numeric_cols, categorical_cols, clone(model_obj))
    LOGGER.log(
        f"        Starting fit for {model_name}; "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"numeric={len(numeric_cols)}, categorical={len(categorical_cols)}"
    )
    t_fit = time.time()
    pipe.fit(X_train, y_train)
    LOGGER.log(f"        Finished fit for {model_name} in {fmt_elapsed(time.time() - t_fit)}")
    t_pred = time.time()
    pred = pipe.predict(X_test)
    LOGGER.log(f"        Finished prediction for {model_name} in {fmt_elapsed(time.time() - t_pred)}")
    metrics = metric_bundle(y_test, pred)
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))
    metrics["model_name"] = model_name
    return pipe, pred, metrics


def run_in_domain_benchmarks(common_df, families, splits):
    metrics_path = PATHS["metrics"] / "in_domain_metrics.csv"
    preds_dir = PATHS["predictions"] / "in_domain"
    ensure_dir(preds_dir)
    existing_metrics = load_existing_csv(metrics_path)
    done_keys = set(existing_metrics["cache_key"]) if not existing_metrics.empty and "cache_key" in existing_metrics.columns else set()
    model_catalog = get_model_catalog()

    total_jobs = len(TARGETS) * len(families) * len(splits) * len(model_catalog)
    remaining_jobs = sum(1 for target_key in TARGETS for family_name in families for split in splits for model_name in model_catalog if benchmark_cache_key(target_key, family_name, model_name, split.split_id) not in done_keys)
    completed_jobs = total_jobs - remaining_jobs
    LOGGER.log(f"In-domain benchmark plan: total_jobs={total_jobs}, already_done={completed_jobs}, remaining={remaining_jobs}")

    for target_idx, (target_key, target_col) in enumerate(TARGETS.items(), start=1):
        LOGGER.log(f"Running in-domain benchmarks for target {target_idx}/{len(TARGETS)}: {target_key} | {target_col}")
        y_all = common_df[target_col].values

        for family_idx, (family_name, family_def) in enumerate(families.items(), start=1):

            LOGGER.log(f"  Descriptor family {family_idx}/{len(families)}: {family_name}")

            fam_df = prepare_family_frame(common_df, family_def)

            X_all = fam_df.drop(columns=[ID_COL]).copy()
            numeric_cols = [c for c in family_def["numeric"] if c in X_all.columns]
            categorical_cols = [c for c in family_def["categorical"] if c in X_all.columns]

            for split_idx, split in enumerate(splits, start=1):
                LOGGER.log(f"    Split {split_idx}/{len(splits)} (split_id={split.split_id}, seed={split.seed})")
                train_idx = np.array(split.train_index)
                test_idx = np.array(split.test_index)
                X_train = X_all.iloc[train_idx].copy()
                X_test = X_all.iloc[test_idx].copy()
                y_train = y_all[train_idx]
                y_test = y_all[test_idx]
                id_test = common_df.iloc[test_idx][ID_COL].values

                for model_idx, (model_name, model_obj) in enumerate(model_catalog.items(), start=1):
                    ck = benchmark_cache_key(target_key, family_name, model_name, split.split_id)
                    if ck in done_keys:
                        LOGGER.log(f"      [{completed_jobs + 1}/{total_jobs}] Skipping completed job: {ck}")
                        completed_jobs += 1
                        continue
                    LOGGER.log(f"      [{completed_jobs + 1}/{total_jobs}] Fitting {ck} ({model_idx}/{len(model_catalog)} models in this split)")
                    t0 = time.time()
                    pipe, pred, metrics = fit_and_predict(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, model_name, model_obj)
                    elapsed = time.time() - t0

                    model_path = PATHS["models"] / "in_domain" / f"{sanitize_name(ck)}.joblib"
                    ensure_dir(model_path.parent)
                    joblib.dump(pipe, model_path, compress=MODEL_COMPRESSION)

                    pred_df = pd.DataFrame({ID_COL: id_test, "split_id": split.split_id, "target_key": target_key, "target_column": target_col, "descriptor_family": family_name, "model_name": model_name, "y_true": y_test, "y_pred": pred, "abs_error": np.abs(y_test - pred), "residual": y_test - pred})
                    pred_df.to_csv(preds_dir / f"{sanitize_name(ck)}.csv", index=False)

                    row = {"cache_key": ck, "target_key": target_key, "target_column": target_col, "descriptor_family": family_name, "model_name": model_name, "split_id": split.split_id, "seed": split.seed, "elapsed_sec": elapsed, **metrics}
                    append_row_csv(metrics_path, row)
                    done_keys.add(ck)
                    completed_jobs += 1

                    LOGGER.log(
                        "         Completed "
                        f"{ck} in {fmt_elapsed(elapsed)} | "
                        f"R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}"
                    )

    metrics_df = pd.read_csv(metrics_path)
    pred_frames = [pd.read_csv(fp) for fp in sorted(preds_dir.glob("*.csv"))]
    preds_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    if not preds_df.empty:
        save_dataframe(preds_df, PATHS["predictions"] / "all_in_domain_predictions", index=False)

    LOGGER.log(f"Completed in-domain benchmarks. Metrics rows={len(metrics_df):,}; prediction rows={len(preds_df):,}")
    return metrics_df, preds_df

def summarize_in_domain_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    agg = metrics_df.groupby(["target_key", "target_column", "descriptor_family", "model_name"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"), spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"), n_splits=("split_id", "nunique"),
    )

    agg["rank_r2"] = agg.groupby("target_key")["r2_mean"].rank(ascending=False, method="average")
    agg["rank_rmse"] = agg.groupby("target_key")["rmse_mean"].rank(ascending=True, method="average")
    agg["rank_spearman"] = agg.groupby("target_key")["spearman_mean"].rank(ascending=False, method="average")

    agg["config_name"] = agg["descriptor_family"] + " | " + agg["model_name"]

    save_dataframe(agg, PATHS["tables"] / "table_2_in_domain_model_results", index=False)
    return agg


def compute_rank_transportability(summary_df: pd.DataFrame):
    pivot = summary_df.pivot_table(index="config_name", columns="target_key", values="r2_mean", aggfunc="mean")
    target_keys = list(pivot.columns)

    rows = []
    mat = pd.DataFrame(index=target_keys, columns=target_keys, dtype=float)
    for a in target_keys:
        for b in target_keys:
            rho = safe_spearman(pivot[a], pivot[b])
            mat.loc[a, b] = rho
            rows.append({"target_a": a, "target_b": b, "rank_spearman_r2": rho})

    corr_df = pd.DataFrame(rows)
    save_dataframe(corr_df, PATHS["tables"] / "table_3a_cross_target_rank_transportability", index=False)
    save_dataframe(mat.reset_index().rename(columns={"index": "target_key"}), PATHS["tables"] / "table_3a_rank_transportability_matrix", index=False)
    return corr_df, mat


def pick_best_config_per_target(summary_df: pd.DataFrame) -> pd.DataFrame:
    best = summary_df.sort_values(["target_key", "r2_mean", "rmse_mean"], ascending=[True, False, True]).groupby("target_key", as_index=False).head(1).copy()
    save_dataframe(best, PATHS["metadata"] / "best_config_per_target", index=False)
    return best


def collect_best_predictions(preds_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    merged = preds_df.merge(best_df[["target_key", "descriptor_family", "model_name"]], on=["target_key", "descriptor_family", "model_name"], how="inner")
    save_dataframe(merged, PATHS["predictions"] / "best_config_predictions", index=False)
    return merged


def compute_residual_transportability(best_preds_df: pd.DataFrame):
    agg = best_preds_df.groupby([ID_COL, "target_key"], as_index=False).agg(mean_abs_error=("abs_error", "mean"), mean_residual=("residual", "mean"), mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean"))
    wide = agg.pivot(index=ID_COL, columns="target_key", values="mean_abs_error")
    target_keys = list(wide.columns)
    rows = []
    mat = pd.DataFrame(index=target_keys, columns=target_keys, dtype=float)
    for a in target_keys:
        for b in target_keys:
            rho = safe_spearman(wide[a], wide[b])
            mat.loc[a, b] = rho
            rows.append({"target_a": a, "target_b": b, "residual_abs_spearman": rho})
    residual_df = pd.DataFrame(rows)
    save_dataframe(residual_df, PATHS["tables"] / "table_3b_residual_transportability", index=False)
    save_dataframe(mat.reset_index().rename(columns={"index": "target_key"}), PATHS["tables"] / "table_3b_residual_transportability_matrix", index=False)
    return residual_df, mat


def standardize_target(y_train: np.ndarray, y_other: np.ndarray):
    mu = float(np.mean(y_train))
    sigma = float(np.std(y_train))
    if sigma < EPS:
        sigma = 1.0

    return (y_train - mu) / sigma, (y_other - mu) / sigma, mu, sigma


def build_preprocessed_arrays(X_train, X_test, numeric_cols, categorical_cols):
    pre = make_preprocessor(numeric_cols, categorical_cols)
    Xt_train = pre.fit_transform(X_train)
    Xt_test = pre.transform(X_test)
    return pre, Xt_train, Xt_test



def run_transfer_experiments(common_df, families, splits):
    out_path = PATHS["metrics"] / "transfer_metrics.csv"
    existing = load_existing_csv(out_path)
    done_keys = set(existing["cache_key"]) if not existing.empty and "cache_key" in existing.columns else set()
    candidate_families = ["enriched_interpretable"] + (["enriched_plus_topology"] if "enriched_plus_topology" in families else [])

    total_expected = len(candidate_families) * len(splits) * (len(TARGETS) + len(TARGETS) * (len(TARGETS) - 1) * 3)
    LOGGER.log(f"Transfer experiment plan: candidate_families={candidate_families}, approximate_total_jobs={total_expected}, already_done={len(done_keys)}")

    progress_counter = 0

    for family_idx, family_name in enumerate(candidate_families, start=1):
        LOGGER.log(f"Transfer family {family_idx}/{len(candidate_families)}: {family_name}")
        family_def = families[family_name]
        fam_df = prepare_family_frame(common_df, family_def)
        X_all = fam_df.drop(columns=[ID_COL]).copy()
        numeric_cols = [c for c in family_def["numeric"] if c in X_all.columns]
        categorical_cols = [c for c in family_def["categorical"] if c in X_all.columns]

        #--------------------------------------------------------------------------------
        for split_idx, split in enumerate(splits, start=1):
            LOGGER.log(f"  Transfer split {split_idx}/{len(splits)} (split_id={split.split_id}, seed={split.seed})")
            train_idx = np.array(split.train_index)
            test_idx = np.array(split.test_index)
            X_train = X_all.iloc[train_idx].copy()
            X_test = X_all.iloc[test_idx].copy()

            mt_ck_prefix = f"multi_target__{family_name}__split{split.split_id}"
            mt_expected_keys = {f"{mt_ck_prefix}__{target_key}" for target_key in TARGETS}

            #--------------------------------------------------------------------------------
            # Independent multi‑output baseline
            if not mt_expected_keys.issubset(done_keys):
                LOGGER.log(f"    Running supplementary multi-target model: {mt_ck_prefix}")
                t_mt = time.time()
                y_train_mt = common_df.iloc[train_idx][list(TARGETS.values())].values
                y_test_mt = common_df.iloc[test_idx][list(TARGETS.values())].values
                _, Xt_train, Xt_test = build_preprocessed_arrays(X_train, X_test, numeric_cols, categorical_cols)
                mt_model = MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        learning_rate=0.06,
                        max_depth=HGB_MAX_DEPTH,
                        max_iter=HGB_MAX_ITER,
                        l2_regularization=1e-3,
                        random_state=RANDOM_STATE,
                    ),
                    n_jobs=N_JOBS,
                )
                mt_model.fit(Xt_train, y_train_mt)
                y_pred_mt = mt_model.predict(Xt_test)
                rows_to_write = []
                for i, (target_key, _) in enumerate(TARGETS.items()):
                    ck = f"{mt_ck_prefix}__{target_key}"
                    progress_counter += 1
                    if ck in done_keys:
                        LOGGER.log(f"      Skipping completed transfer job: {ck}")
                        continue
                    m = metric_bundle(y_test_mt[:, i], y_pred_mt[:, i])
                    rows_to_write.append(
                        {
                            "cache_key": ck,
                            "family_name": family_name,
                            "split_id": split.split_id,
                            "source_target": "all_targets_joint",
                            "target_target": target_key,
                            "transfer_variant": "independent_multi_output",
                            **m,
                        }
                    )
                append_rows_csv(out_path, rows_to_write)
                done_keys.update([row["cache_key"] for row in rows_to_write])
                LOGGER.log(f"    Completed multi-target block in {fmt_elapsed(time.time() - t_mt)} | wrote {len(rows_to_write)} rows")

            # ------------------------------------------------------------------
            # Pairwise transfer experiments
            pair_total = len(TARGETS) * (len(TARGETS) - 1)
            pair_counter = 0
            for source_key, source_col in TARGETS.items():
                for target_key, target_col in TARGETS.items():
                    if source_key == target_key:
                        continue
                    pair_counter += 1
                    LOGGER.log(f"    Transfer pair {pair_counter}/{pair_total}: {source_key} -> {target_key}")
                    ck_base = f"transfer__{family_name}__{source_key}__to__{target_key}__split{split.split_id}"

                    # ----- Scratch baseline (train on target from scratch)
                    scratch_ck = ck_base + "__scratch_target"
                    if scratch_ck not in done_keys:
                        LOGGER.log(f"      Scratch baseline: {scratch_ck}")
                        t0 = time.time()
                        y_train_t = common_df.iloc[train_idx][target_col].values
                        y_test_t = common_df.iloc[test_idx][target_col].values
                        _, Xt_train, Xt_test = build_preprocessed_arrays(X_train, X_test, numeric_cols, categorical_cols)
                        y_train_t_std, _, mu_t, sigma_t = standardize_target(y_train_t, y_test_t)
                        scratch_model = MLPRegressor(
                            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=8e-4,
                            max_iter=MLP_TRANSFER_MAX_ITER,
                            early_stopping=True,
                            validation_fraction=0.10,
                            n_iter_no_change=15,
                            random_state=RANDOM_STATE,
                        )
                        scratch_model.fit(Xt_train, y_train_t_std)
                        pred = scratch_model.predict(Xt_test) * sigma_t + mu_t
                        m = metric_bundle(y_test_t, pred)
                        append_row_csv(
                            out_path,
                            {
                                "cache_key": scratch_ck,
                                "family_name": family_name,
                                "split_id": split.split_id,
                                "source_target": source_key,
                                "target_target": target_key,
                                "transfer_variant": "scratch_target",
                                **m,
                            },
                        )
                        done_keys.add(scratch_ck)
                        LOGGER.log(f"        Done in {fmt_elapsed(time.time() - t0)} | R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}")
                    else:
                        LOGGER.log(f"      Skipping completed scratch baseline: {scratch_ck}")


                    # ----- Direct transport (source model applied to target without adaptation)
                    direct_ck = ck_base + "__direct_transport"
                    if direct_ck not in done_keys:
                        LOGGER.log(f"      Direct transport: {direct_ck}")
                        t0 = time.time()
                        y_train_s = common_df.iloc[train_idx][source_col].values
                        y_test_t = common_df.iloc[test_idx][target_col].values
                        _, Xt_train, Xt_test = build_preprocessed_arrays(X_train, X_test, numeric_cols, categorical_cols)
                        y_train_s_std, _, _, _ = standardize_target(y_train_s, y_test_t)
                        direct_model = MLPRegressor(
                            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=8e-4,
                            max_iter=MLP_TRANSFER_MAX_ITER,
                            early_stopping=True,
                            validation_fraction=0.10,
                            n_iter_no_change=15,
                            random_state=RANDOM_STATE,
                        )
                        direct_model.fit(Xt_train, y_train_s_std)
                        y_train_t = common_df.iloc[train_idx][target_col].values
                        mu_t = float(np.mean(y_train_t))
                        sigma_t = float(np.std(y_train_t))
                        sigma_t = sigma_t if sigma_t > EPS else 1.0
                        pred = direct_model.predict(Xt_test) * sigma_t + mu_t
                        m = metric_bundle(y_test_t, pred)
                        append_row_csv(
                            out_path,
                            {
                                "cache_key": direct_ck,
                                "family_name": family_name,
                                "split_id": split.split_id,
                                "source_target": source_key,
                                "target_target": target_key,
                                "transfer_variant": "direct_transport",
                                **m,
                            },
                        )
                        done_keys.add(direct_ck)
                        LOGGER.log(f"        Done in {fmt_elapsed(time.time() - t0)} | R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}")
                    else:
                        LOGGER.log(f"      Skipping completed direct transport: {direct_ck}")

                    # ----- Pretrain on source, finetune on target (warm_start)
                    finetune_ck = ck_base + "__pretrain_finetune"
                    if finetune_ck not in done_keys:
                        LOGGER.log(f"      Pretrain + finetune: {finetune_ck}")
                        t0 = time.time()
                        y_train_s = common_df.iloc[train_idx][source_col].values
                        y_train_t = common_df.iloc[train_idx][target_col].values
                        y_test_t = common_df.iloc[test_idx][target_col].values
                        _, Xt_train, Xt_test = build_preprocessed_arrays(X_train, X_test, numeric_cols, categorical_cols)
                        y_train_s_std, _, _, _ = standardize_target(y_train_s, y_test_t)
                        y_train_t_std, _, mu_t, sigma_t = standardize_target(y_train_t, y_test_t)
                        model = MLPRegressor(
                            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=8e-4,
                            max_iter=1,
                            warm_start=True,
                            random_state=RANDOM_STATE,
                            shuffle=True,
                        )
                        for _ in range(FINETUNE_PRETRAIN_EPOCHS):
                            model.partial_fit(Xt_train, y_train_s_std)
                        for _ in range(FINETUNE_TARGET_EPOCHS):
                            model.partial_fit(Xt_train, y_train_t_std)
                        pred = model.predict(Xt_test) * sigma_t + mu_t
                        m = metric_bundle(y_test_t, pred)
                        append_row_csv(
                            out_path,
                            {
                                "cache_key": finetune_ck,
                                "family_name": family_name,
                                "split_id": split.split_id,
                                "source_target": source_key,
                                "target_target": target_key,
                                "transfer_variant": "pretrain_finetune",
                                **m,
                            },
                        )
                        done_keys.add(finetune_ck)
                        LOGGER.log(f"        Done in {fmt_elapsed(time.time() - t0)} | R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}")
                    else:
                        LOGGER.log(f"      Skipping completed pretrain + finetune: {finetune_ck}")

    # ------------------------------------------------------------------------------------------
    transfer_df = load_existing_csv(out_path)
    save_dataframe(transfer_df, PATHS["tables"] / "table_4_transfer_gain_loss_raw", index=False)
    LOGGER.log(f"Completed transfer experiments. Rows in transfer metrics={len(transfer_df):,}")
    return transfer_df

def summarize_transfer_results(transfer_df: pd.DataFrame) -> pd.DataFrame:
    if transfer_df.empty:
        return transfer_df

    summary = transfer_df.groupby(["family_name", "source_target", "target_target", "transfer_variant"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"), spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"), n_splits=("split_id", "nunique"),
    )

    scratch = summary[summary["transfer_variant"] == "scratch_target"].copy().rename(columns={"rmse_mean": "rmse_scratch",
                                                                                              "mae_mean": "mae_scratch",
                                                                                              "r2_mean": "r2_scratch",
                                                                                              "spearman_mean": "spearman_scratch"
                                                                                              })[["family_name", "source_target", "target_target",
                                                                                             "rmse_scratch", "mae_scratch", "r2_scratch", "spearman_scratch"]]

    merged = summary.merge(scratch, on=["family_name", "source_target", "target_target"], how="left")
    merged["delta_rmse_vs_scratch"] = merged["rmse_mean"] - merged["rmse_scratch"]
    merged["delta_mae_vs_scratch"] = merged["mae_mean"] - merged["mae_scratch"]
    merged["delta_r2_vs_scratch"] = merged["r2_mean"] - merged["r2_scratch"]
    merged["delta_spearman_vs_scratch"] = merged["spearman_mean"] - merged["spearman_scratch"]

    save_dataframe(merged, PATHS["tables"] / "table_4_transfer_gain_loss", index=False)
    return merged


def compute_elite_retrieval_overlap(best_preds_df: pd.DataFrame):
    agg = best_preds_df.groupby([ID_COL, "target_key"], as_index=False).agg(mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean"))
    pred_wide = agg.pivot(index=ID_COL, columns="target_key", values="mean_pred")
    true_wide = agg.pivot(index=ID_COL, columns="target_key", values="mean_true")
    n = len(pred_wide)
    top_k = max(TOP_K_MIN, int(math.ceil(TOP_K_FRACTION * n)))
    LOGGER.log(f"Elite retrieval overlap uses top_k={top_k} out of n={n}")

    targets = list(pred_wide.columns)
    rows = []
    matrix = pd.DataFrame(index=targets, columns=targets, dtype=float)

    for a in targets:
        pred_top_a = set(pred_wide[a].nlargest(top_k).index)
        for b in targets:
            pred_top_b = set(pred_wide[b].nlargest(top_k).index)
            inter = len(pred_top_a & pred_top_b)
            union = len(pred_top_a | pred_top_b)
            jacc = inter / union if union else np.nan
            matrix.loc[a, b] = jacc

            true_top_b = set(true_wide[b].nlargest(top_k).index)

            # precision: among top‑k predicted for A, how many are actually in top‑k true for B
            precision = len(pred_top_a & true_top_b) / max(len(pred_top_a), 1)
            rows.append({"target_a": a, "target_b": b, "predicted_topk_jaccard": jacc, "precision_topA_against_trueTopB": precision, "top_k": top_k})

    df = pd.DataFrame(rows)
    save_dataframe(df, PATHS["tables"] / "table_elite_overlap_summary", index=False)
    save_dataframe(matrix.reset_index().rename(columns={"index": "target_key"}), PATHS["tables"] / "table_elite_overlap_matrix", index=False)
    return df, matrix


def fit_best_models_on_full_data(common_df, best_df, families):
    rows = []
    for _, row in best_df.iterrows():
        target_key = row["target_key"]
        target_col = TARGETS[target_key]
        family_name = row["descriptor_family"]
        model_name = row["model_name"]
        family_def = families[family_name]

        fam_df = prepare_family_frame(common_df, family_def)
        X = fam_df.drop(columns=[ID_COL]).copy()
        y = common_df[target_col].values
        numeric_cols = [c for c in family_def["numeric"] if c in X.columns]
        categorical_cols = [c for c in family_def["categorical"] if c in X.columns]

        model = clone(get_model_catalog()[model_name])
        pipe = Pipeline(steps=[("preprocessor", make_preprocessor(numeric_cols, categorical_cols)), ("model", model)])
        LOGGER.log(f"Fitting best full-data model for {target_key}: {family_name} | {model_name}")
        t_fit = time.time()
        pipe.fit(X, y)
        LOGGER.log(f"  Model fit completed in {fmt_elapsed(time.time() - t_fit)}")

        model_path = PATHS["models"] / "best_full_data" / f"{target_key}__{family_name}__{model_name}.joblib"
        ensure_dir(model_path.parent)
        joblib.dump(pipe, model_path, compress=MODEL_COMPRESSION)


        try:
            LOGGER.log(f"  Computing permutation importance for {target_key}")
            t_imp = time.time()

            if len(X) > PERMUTATION_IMPORTANCE_SAMPLE_N:
                rng = np.random.default_rng(RANDOM_STATE)
                imp_idx = np.sort(rng.choice(len(X), size=PERMUTATION_IMPORTANCE_SAMPLE_N, replace=False))
                X_imp = X.iloc[imp_idx].copy()
                y_imp = y[imp_idx]
                LOGGER.log(f"  Permutation importance uses sampled n={len(X_imp):,} from full n={len(X):,}")
            else:
                X_imp = X
                y_imp = y
                LOGGER.log(f"  Permutation importance uses full n={len(X_imp):,}")

            result = permutation_importance(
                pipe,
                X_imp,
                y_imp,
                n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                scoring="r2",
            )
            LOGGER.log(f"  Permutation importance completed in {fmt_elapsed(time.time() - t_imp)}")
            """
            try:
                feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
            except Exception:
                feat_names = [f"feature_{i}" for i in range(len(result.importances_mean))]
            """
            feat_names = X_imp.columns.tolist()
            if len(feat_names) != len(result.importances_mean):
                raise ValueError(
                    f"Permutation importance length mismatch for {target_key}: "
                    f"{len(feat_names)} input features vs "
                    f"{len(result.importances_mean)} importance values."
                )
            imp_df = pd.DataFrame({"target_key": target_key,
                                   "family_name": family_name,
                                   "model_name": model_name,
                                   "feature": feat_names,
                                   "importance_mean": result.importances_mean,
                                   "importance_std": result.importances_std
                                   }).sort_values("importance_mean", ascending=False)

            save_dataframe(imp_df, PATHS["feature_importance"] / f"importance__{target_key}__{family_name}__{model_name}", index=False)
            rows.extend(imp_df.head(20).to_dict("records"))
        except Exception as exc:
            LOGGER.log(f"Permutation importance failed for {target_key}: {exc}")

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        save_dataframe(out_df, PATHS["tables_si"] / "table_feature_importance_top20", index=False)
    return out_df


def run_descriptor_regime_analysis(common_df: pd.DataFrame, families: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    if "enriched_interpretable" not in families:
        return {}

    family_def = families["enriched_interpretable"]
    fam_df = prepare_family_frame(common_df, family_def)
    feature_cols = [c for c in family_def["numeric"] if c in fam_df.columns]
    if len(feature_cols) < 2:
        return {}

    X_full = fam_df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_full)
    X_scaled = scaler.fit_transform(X_imp)

    # sample to keep PCA tractable
    n_total = X_scaled.shape[0]
    rng = np.random.default_rng(RANDOM_STATE)
    analysis_n = min(PCA_ANALYSIS_SAMPLE_N, n_total)
    sample_idx = np.sort(rng.choice(n_total, size=analysis_n, replace=False))
    X_sample = X_scaled[sample_idx]
    sampled_df = common_df.iloc[sample_idx].reset_index(drop=True).copy()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_sample)

    # standardize each target separately, then average across them
    target_df = sampled_df[list(TARGETS.values())].copy()
    z_df = (target_df - target_df.mean()) / target_df.std(ddof=0).replace(0, 1.0)
    z_df.columns = list(TARGETS.keys())

    projection_df = pd.DataFrame(
        {
            ID_COL: sampled_df[ID_COL].values,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "mean_target_z": z_df.mean(axis=1).values,
            "same_gas_gap": (
                np.abs(z_df["co2_015"] - z_df["co2_0015"]) +
                np.abs(z_df["ch4_65"] - z_df["ch4_58"])
            ).values,
        }
    )
    for target_key in TARGETS:
        projection_df[target_key] = sampled_df[TARGETS[target_key]].values
    save_dataframe(projection_df, PATHS["data_processed"] / "descriptor_pca_projection_sampled", index=False)

    loadings = pd.DataFrame(
        {
            "feature": feature_cols,
            "pc1_loading": pca.components_[0],
            "pc2_loading": pca.components_[1],
        }
    )
    loadings["pc1_abs_loading"] = loadings["pc1_loading"].abs()
    loadings["pc2_abs_loading"] = loadings["pc2_loading"].abs()
    loadings["combined_abs_loading"] = loadings["pc1_abs_loading"] + loadings["pc2_abs_loading"]
    loadings = loadings.sort_values("combined_abs_loading", ascending=False).reset_index(drop=True)
    save_dataframe(loadings, PATHS["tables_si"] / "table_descriptor_pca_loadings", index=False)

    top_loading_table = loadings.head(TOP_LOADING_FEATURES).copy()
    save_dataframe(top_loading_table, PATHS["tables"] / "table_descriptor_pca_top_loadings", index=False)

    quantile_rows = []
    for label, ascending in [("top_10pct_mean_target_z", False), ("bottom_10pct_mean_target_z", True)]:
        n_sel = max(50, int(np.ceil(0.10 * len(projection_df))))
        sel = projection_df.nsmallest(n_sel, "mean_target_z") if ascending else projection_df.nlargest(n_sel, "mean_target_z")
        row = {"group": label, "n": int(len(sel))}
        for col in feature_cols:
            row[col] = float(sampled_df.loc[sel.index, col].mean())
        quantile_rows.append(row)
    quantile_summary = pd.DataFrame(quantile_rows)
    save_dataframe(quantile_summary, PATHS["tables_si"] / "table_descriptor_pca_quantile_profiles", index=False)

    result = {
        "projection": projection_df,
        "loadings": loadings,
        "top_loadings": top_loading_table,
        "analysis_sample_n": int(analysis_n),
        "n_total": int(n_total),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    safe_json_dump(
        {
            "analysis_sample_n": result["analysis_sample_n"],
            "n_total": result["n_total"],
            "explained_variance_ratio": result["explained_variance_ratio"],
        },
        PATHS["metadata"] / "descriptor_pca_analysis_summary.json",
    )
    return result

def make_figure_7_descriptor_regimes(regime_result: Dict[str, Any]):
    if not regime_result:
        return
    proj = regime_result["projection"].copy()
    load = regime_result["top_loadings"].copy().iloc[::-1]
    ev = regime_result["explained_variance_ratio"]

    save_figure_source(proj, PATHS["figure_data_main"] / "Figure7_descriptor_pca_projection")
    save_figure_source(regime_result["loadings"].copy(), PATHS["figure_data_main"] / "Figure7_descriptor_pca_all_loadings")
    save_figure_source(load.copy().reset_index(drop=True), PATHS["figure_data_main"] / "Figure7_descriptor_pca_top_loadings")
    save_figure_source(
        pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": ev[:2]}),
        PATHS["figure_data_main"] / "Figure7_descriptor_pca_explained_variance",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.45, 1.0]})

    sc = axes[0].scatter(
        proj["pc1"],
        proj["pc2"],
        c=proj["mean_target_z"],
        cmap="coolwarm",
        s=16,
        alpha=0.70,
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title(
        f"Descriptor PCA map (sample n={regime_result['analysis_sample_n']:,}; var={100*ev[0]:.1f}%/{100*ev[1]:.1f}%)"
    )
    cbar = plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Mean standardized uptake across 4 targets")

    y = np.arange(len(load))
    axes[1].barh(y - 0.18, load["pc1_loading"], height=0.35, label="PC1")
    axes[1].barh(y + 0.18, load["pc2_loading"], height=0.35, label="PC2")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(load["feature"])
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Loading")
    axes[1].set_title("Top descriptor loadings")
    axes[1].legend()

    fig.suptitle("Figure 7. Descriptor-space PCA and dominant geometric drivers", y=1.02, fontsize=14)
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_main"] / "Figure7_descriptor_pca_map")


def make_si_descriptor_regime_figures(regime_result: Dict[str, Any]):
    if not regime_result:
        return

    proj = regime_result["projection"].copy()
    gap_df = pd.DataFrame({"same_gas_gap": proj["same_gas_gap"].values})
    save_figure_source(gap_df, PATHS["figure_data_si"] / "FigureS5_same_gas_gap_raw")
    save_figure_source(histogram_bin_table(proj["same_gas_gap"], bins=40), PATHS["figure_data_si"] / "FigureS5_same_gas_gap_bins")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(proj["same_gas_gap"], bins=40, alpha=0.8)
    ax.set_xlabel("Within-gas disparity score")
    ax.set_ylabel("Count")
    ax.set_title("Figure S5. Within-gas disparity distribution in PCA sample")
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_si"] / "FigureS5_within_gas_disparity_distribution")


def add_heatmap(ax, mat: pd.DataFrame, title: str, cmap: str = "viridis", vmin=None, vmax=None, fmt: str = ".2f"):
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, format(float(val), fmt), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_figure_1_relationship_map(common_df: pd.DataFrame):
    target_df = common_df[list(TARGETS.values())].copy()
    target_key_order = list(TARGETS.keys())
    corr = target_df.corr(method="spearman")

    corr_export = corr.copy()
    corr_export.index = target_key_order
    corr_export.columns = target_key_order
    corr_export.index.name = "target_key"
    save_figure_source(corr_export.reset_index(), PATHS["figure_data_main"] / "Figure1_panelA_target_spearman_matrix")

    raw_export = target_df.copy()
    raw_export.columns = target_key_order
    save_figure_source(raw_export, PATHS["figure_data_main"] / "Figure1_all_target_values")

    hist1_raw = pd.DataFrame({"uptake_mmol_per_g": target_df.iloc[:, 0]})
    hist2_raw = pd.DataFrame({"uptake_mmol_per_g": target_df.iloc[:, 1]})
    save_figure_source(hist1_raw, PATHS["figure_data_main"] / f"Figure1_panelB_{target_key_order[0]}_hist_raw")
    save_figure_source(histogram_bin_table(target_df.iloc[:, 0], bins=40), PATHS["figure_data_main"] / f"Figure1_panelB_{target_key_order[0]}_hist_bins")
    save_figure_source(hist2_raw, PATHS["figure_data_main"] / f"Figure1_panelC_{target_key_order[1]}_hist_raw")
    save_figure_source(histogram_bin_table(target_df.iloc[:, 1], bins=40), PATHS["figure_data_main"] / f"Figure1_panelC_{target_key_order[1]}_hist_bins")

    scatter_df = pd.DataFrame(
        {
            target_key_order[2]: target_df.iloc[:, 2].values,
            target_key_order[3]: target_df.iloc[:, 3].values,
        }
    )
    save_figure_source(scatter_df, PATHS["figure_data_main"] / f"Figure1_panelD_{target_key_order[2]}_vs_{target_key_order[3]}_scatter")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    add_heatmap(axes[0, 0], corr_export, "Spearman correlation across four adsorption targets", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 1].hist(target_df.iloc[:, 0], bins=40, alpha=0.8)
    axes[0, 1].set_title(target_key_order[0] + " distribution")
    axes[0, 1].set_xlabel("Uptake (mmol/g)")
    axes[1, 0].hist(target_df.iloc[:, 1], bins=40, alpha=0.8)
    axes[1, 0].set_title(target_key_order[1] + " distribution")
    axes[1, 0].set_xlabel("Uptake (mmol/g)")
    axes[1, 1].scatter(target_df.iloc[:, 2], target_df.iloc[:, 3], s=8, alpha=0.4)
    axes[1, 1].set_xlabel(target_key_order[2])
    axes[1, 1].set_ylabel(target_key_order[3])
    axes[1, 1].set_title("Example same-gas cross-pressure relationship")
    fig.suptitle("Figure 1. Four-target relationship map", y=1.02, fontsize=14)
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_main"] / "Figure1_four_target_relationship_map")

    s1_raw = target_df.copy()
    s1_raw.columns = target_key_order
    save_figure_source(s1_raw, PATHS["figure_data_si"] / "FigureS1_all_target_distributions_raw")
    for tkey, tcol in TARGETS.items():
        save_figure_source(
            pd.DataFrame({"uptake_mmol_per_g": common_df[tcol].values}),
            PATHS["figure_data_si"] / f"FigureS1_{tkey}_hist_raw",
        )
        save_figure_source(
            histogram_bin_table(common_df[tcol], bins=50),
            PATHS["figure_data_si"] / f"FigureS1_{tkey}_hist_bins",
        )

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (tkey, tcol) in zip(axes2.flatten(), TARGETS.items()):
        ax.hist(common_df[tcol], bins=50, alpha=0.85)
        ax.set_title(tkey)
        ax.set_xlabel("Uptake (mmol/g)")
        ax.set_ylabel("Count")
    fig2.tight_layout()
    maybe_save_figure(fig2, PATHS["figures_si"] / "FigureS1_target_distributions")


def make_figure_2_benchmark_matrix(summary_df: pd.DataFrame):
    summary_df = summary_df.copy()
    summary_df["config"] = summary_df["descriptor_family"] + "\n" + summary_df["model_name"]
    targets = list(summary_df["target_key"].unique())

    export_cols = [
        "target_key",
        "target_column",
        "descriptor_family",
        "model_name",
        "config",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "spearman_mean",
        "spearman_std",
        "n_splits",
        "rank_r2",
        "rank_rmse",
        "rank_spearman",
    ]
    save_figure_source(summary_df[export_cols], PATHS["figure_data_main"] / "Figure2_in_domain_benchmark_matrix_source")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, t in zip(axes, targets):
        sub = summary_df[summary_df["target_key"] == t].copy().sort_values("r2_mean", ascending=False)
        save_figure_source(sub[export_cols], PATHS["figure_data_main"] / f"Figure2_{t}_bar_data")
        ax.barh(sub["config"], sub["r2_mean"])
        ax.invert_yaxis()
        ax.set_title(t)
        ax.set_xlabel("Mean test $R^2$")
    fig.suptitle("Figure 2. In-domain benchmark matrix", y=1.02, fontsize=14)
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_main"] / "Figure2_in_domain_benchmark_matrix")

    pivot_r2 = summary_df.pivot_table(index="config", columns="target_key", values="r2_mean", aggfunc="mean")
    pivot_rmse = summary_df.pivot_table(index="config", columns="target_key", values="rmse_mean", aggfunc="mean")
    pivot_r2_export = pivot_r2.copy()
    pivot_r2_export.index.name = "config"
    pivot_rmse_export = pivot_rmse.copy()
    pivot_rmse_export.index.name = "config"
    save_figure_source(pivot_r2_export.reset_index(), PATHS["figure_data_si"] / "FigureS2_r2_heatmap_matrix")
    save_figure_source(pivot_rmse_export.reset_index(), PATHS["figure_data_si"] / "FigureS2_rmse_heatmap_matrix")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 10))
    add_heatmap(axes2[0], pivot_r2, "Mean test $R^2$", cmap="viridis")
    add_heatmap(axes2[1], pivot_rmse, "Mean test RMSE", cmap="viridis")
    fig2.tight_layout()
    maybe_save_figure(fig2, PATHS["figures_si"] / "FigureS2_benchmark_heatmaps")


def make_figure_3_rank_transportability(rank_mat: pd.DataFrame):
    rank_export = rank_mat.copy()
    rank_export.index.name = "target_key"
    save_figure_source(rank_export.reset_index(), PATHS["figure_data_main"] / "Figure3_rank_transportability_matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    add_heatmap(ax, rank_mat, "Figure 3. Cross-target ranking transportability", cmap="coolwarm", vmin=-1, vmax=1)
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_main"] / "Figure3_cross_target_ranking_transportability")


def make_figure_5_transfer_gain_loss(transfer_summary_df: pd.DataFrame):
    if transfer_summary_df.empty:
        return
    plot_df = transfer_summary_df[transfer_summary_df["transfer_variant"].isin(["direct_transport", "pretrain_finetune"])].copy()
    plot_df["pair"] = plot_df["source_target"] + "→" + plot_df["target_target"]
    plot_df["label"] = plot_df["family_name"] + " | " + plot_df["transfer_variant"]
    save_figure_source(plot_df, PATHS["figure_data_main"] / "Figure5_transfer_gain_loss_source")
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["delta_r2_vs_scratch"])
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["pair"] + "\n" + plot_df["label"], rotation=90)
    ax.set_ylabel(r"$\Delta R^2$ vs scratch target training")
    ax.set_title("Figure 5. Transfer gain/loss composite")
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_main"] / "Figure5_transfer_gain_loss_composite")


def make_figure_4_residual_overlap(residual_mat: pd.DataFrame):
    residual_export = residual_mat.copy()
    residual_export.index.name = "target_key"
    save_figure_source(residual_export.reset_index(), PATHS["figure_data_main"] / "Figure4_residual_overlap_matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    add_heatmap(ax, residual_mat, "Figure 4. Residual-structure overlap", cmap="coolwarm", vmin=-1, vmax=1)
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_main"] / "Figure4_residual_structure_overlap")


def make_figure_6_elite_overlap(elite_mat: pd.DataFrame):
    elite_export = elite_mat.copy()
    elite_export.index.name = "target_key"
    save_figure_source(elite_export.reset_index(), PATHS["figure_data_main"] / "Figure6_elite_overlap_matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    add_heatmap(ax, elite_mat, "Figure 6. Elite retrieval overlap", cmap="viridis", vmin=0, vmax=1)
    fig.tight_layout()
    maybe_save_figure(fig, PATHS["figures_main"] / "Figure6_elite_retrieval_overlap")


def make_si_figures(best_preds_df: pd.DataFrame, transfer_summary_df: pd.DataFrame, regime_result: Dict[str, Any] | None = None):
    if not best_preds_df.empty:
        abs_error_export = best_preds_df[["target_key", "split_id", "y_true", "y_pred", "abs_error", "residual"]].copy()
        save_figure_source(abs_error_export, PATHS["figure_data_si"] / "FigureS3_best_model_abs_error_raw")
        for target_key in TARGETS.keys():
            sub = best_preds_df[best_preds_df["target_key"] == target_key]
            save_figure_source(
                pd.DataFrame({"abs_error": sub["abs_error"].values}),
                PATHS["figure_data_si"] / f"FigureS3_{target_key}_abs_error_raw",
            )
            save_figure_source(
                histogram_bin_table(sub["abs_error"], bins=40),
                PATHS["figure_data_si"] / f"FigureS3_{target_key}_abs_error_bins",
            )

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, target_key in zip(axes.flatten(), TARGETS.keys()):
            sub = best_preds_df[best_preds_df["target_key"] == target_key]
            ax.hist(sub["abs_error"], bins=40, alpha=0.8)
            ax.set_title(target_key)
            ax.set_xlabel("Absolute error")
        fig.tight_layout()
        maybe_save_figure(fig, PATHS["figures_si"] / "FigureS3_best_model_abs_error_distributions")
    if not transfer_summary_df.empty:
        ts = transfer_summary_df[transfer_summary_df["transfer_variant"].isin(["direct_transport", "pretrain_finetune"])].copy()
        ts["relation"] = ts.apply(lambda row: "same_gas" if row["source_target"].split("_")[0] == row["target_target"].split("_")[0] else "cross_gas", axis=1)
        save_figure_source(ts, PATHS["figure_data_si"] / "FigureS4_transfer_gain_distribution_source")
        same = ts.loc[ts["relation"] == "same_gas", "delta_r2_vs_scratch"].dropna()
        cross = ts.loc[ts["relation"] == "cross_gas", "delta_r2_vs_scratch"].dropna()
        save_figure_source(pd.DataFrame({"delta_r2_vs_scratch": same.values}), PATHS["figure_data_si"] / "FigureS4_same_gas_delta_r2_raw")
        save_figure_source(pd.DataFrame({"delta_r2_vs_scratch": cross.values}), PATHS["figure_data_si"] / "FigureS4_cross_gas_delta_r2_raw")
        save_figure_source(histogram_bin_table(same, bins=20), PATHS["figure_data_si"] / "FigureS4_same_gas_delta_r2_bins")
        save_figure_source(histogram_bin_table(cross, bins=20), PATHS["figure_data_si"] / "FigureS4_cross_gas_delta_r2_bins")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(same, bins=20, alpha=0.7, label="same gas")
        ax.hist(cross, bins=20, alpha=0.7, label="cross gas")
        ax.set_xlabel(r"$\Delta R^2$ vs scratch")
        ax.set_ylabel("Count")
        ax.set_title("Figure S4. Distribution of transfer gains")
        ax.legend()
        fig.tight_layout()
        maybe_save_figure(fig, PATHS["figures_si"] / "FigureS4_transfer_gain_distribution")
    if regime_result:
        make_si_descriptor_regime_figures(regime_result)


def build_manuscript_helper_tables(summary_df, rank_df, residual_df, transfer_summary_df, regime_result: Dict[str, Any] | None = None):
    best_per_target = summary_df.sort_values(["target_key", "r2_mean", "rmse_mean"], ascending=[True, False, True]).groupby("target_key").head(3).copy()
    save_dataframe(best_per_target, PATHS["tables"] / "table_main_best_three_configs_per_target", index=False)
    combo = rank_df.merge(residual_df, on=["target_a", "target_b"], how="outer")
    save_dataframe(combo, PATHS["tables"] / "table_3_cross_target_ranking_and_residual_summary", index=False)
    if not transfer_summary_df.empty:
        compact = transfer_summary_df[transfer_summary_df["transfer_variant"].isin(["scratch_target", "direct_transport", "pretrain_finetune", "multi_target_joint"])].copy()
        save_dataframe(compact, PATHS["tables"] / "table_main_transfer_summary", index=False)
    if regime_result:
        regime_table = regime_result["top_loadings"].copy()
        save_dataframe(regime_table, PATHS["tables"] / "table_main_descriptor_pca_summary", index=False)


def write_run_manifest(raw_df, common_df, families, splits):
    manifest = {
        "project_name": PROJECT_NAME,
        "created_at": now_str(),
        "input_clean_data": str(INPUT_CLEAN_DATA),
        "input_topology_lists": str(INPUT_TOPOLOGY_LISTS) if INPUT_TOPOLOGY_LISTS.exists() else None,
        "n_rows_raw": int(len(raw_df)),
        "n_rows_common_cohort": int(len(common_df)),
        "targets": TARGETS,
        "descriptor_families": families,
        "outer_random_splits": OUTER_RANDOM_SPLITS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "lightweight_settings": {
            "topology_min_count": TOPOLOGY_MIN_COUNT,
            "topology_top_n": TOPOLOGY_TOP_N,
            "rf_n_estimators": RF_N_ESTIMATORS,
            "rf_max_depth": RF_MAX_DEPTH,
            "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            "hgb_max_iter": HGB_MAX_ITER,
            "hgb_max_depth": HGB_MAX_DEPTH,
            "mlp_hidden_layers": list(MLP_HIDDEN_LAYERS),
            "mlp_max_iter": MLP_MAX_ITER,
            "mlp_transfer_max_iter": MLP_TRANSFER_MAX_ITER,
            "finetune_pretrain_epochs": FINETUNE_PRETRAIN_EPOCHS,
            "finetune_target_epochs": FINETUNE_TARGET_EPOCHS,
            "permutation_importance_sample_n": PERMUTATION_IMPORTANCE_SAMPLE_N,
            "permutation_importance_repeats": PERMUTATION_IMPORTANCE_REPEATS,
            "model_compression": MODEL_COMPRESSION,
            "n_jobs": N_JOBS,
        },
        "python": sys.version,
    }
    safe_json_dump(manifest, PATHS["metadata"] / "run_manifest.json")


def main():
    start = time.time()
    LOGGER.log("=" * 88)
    LOGGER.log("Starting Target Transferability pipeline.")
    LOGGER.log(f"Lightweight scientific execution requested with N_JOBS={N_JOBS}.")
    assert_required_files()

    t_stage = log_stage("Load raw data")
    raw_df = load_raw_data()
    log_stage_done("Load raw data", t_stage)

    t_stage = log_stage("Build strict common cohort")
    common_df = build_common_cohort(raw_df)
    log_stage_done("Build strict common cohort", t_stage)

    t_stage = log_stage("Save raw and common datasets")
    save_dataframe(raw_df, PATHS["data_processed"] / "raw_loaded_data", index=False)
    save_dataframe(common_df, PATHS["data_processed"] / "strict_common_cohort", index=False)
    log_stage_done("Save raw and common datasets", t_stage)

    t_stage = log_stage("Create summary tables and descriptor families")
    save_dataset_descriptions(raw_df, common_df)
    families = get_descriptor_families(common_df)
    log_stage_done("Create summary tables and descriptor families", t_stage)

    t_stage = log_stage("Build or load persistent splits")
    splits = build_or_load_splits(common_df)
    log_stage_done("Build or load persistent splits", t_stage)

    t_stage = log_stage("Write run manifest")
    write_run_manifest(raw_df, common_df, families, splits)
    log_stage_done("Write run manifest", t_stage)

    t_stage = log_stage("Run in-domain benchmarks")
    metrics_df, preds_df = run_in_domain_benchmarks(common_df, families, splits)
    log_stage_done("Run in-domain benchmarks", t_stage)

    t_stage = log_stage("Summarize in-domain results and compute transportability")
    summary_df = summarize_in_domain_results(metrics_df)
    rank_df, rank_mat = compute_rank_transportability(summary_df)
    best_df = pick_best_config_per_target(summary_df)
    best_preds_df = collect_best_predictions(preds_df, best_df)
    residual_df, residual_mat = compute_residual_transportability(best_preds_df)
    log_stage_done("Summarize in-domain results and compute transportability", t_stage)

    t_stage = log_stage("Run transfer experiments")
    transfer_df = run_transfer_experiments(common_df, families, splits)
    transfer_summary_df = summarize_transfer_results(transfer_df)
    log_stage_done("Run transfer experiments", t_stage)

    t_stage = log_stage("Elite overlap and descriptor regime analysis")
    elite_df, elite_mat = compute_elite_retrieval_overlap(best_preds_df)
    regime_result = run_descriptor_regime_analysis(common_df, families)
    log_stage_done("Elite overlap and descriptor regime analysis", t_stage)

    t_stage = log_stage("Fit best full-data models and helper tables")
    fit_best_models_on_full_data(common_df, best_df, families)
    build_manuscript_helper_tables(summary_df, rank_df, residual_df, transfer_summary_df, regime_result)
    log_stage_done("Fit best full-data models and helper tables", t_stage)

    t_stage = log_stage("Render manuscript and SI figures plus source CSVs")
    make_figure_1_relationship_map(common_df)
    make_figure_2_benchmark_matrix(summary_df)
    make_figure_3_rank_transportability(rank_mat)
    make_figure_5_transfer_gain_loss(transfer_summary_df)
    make_figure_4_residual_overlap(residual_mat)
    make_figure_6_elite_overlap(elite_mat)
    make_figure_7_descriptor_regimes(regime_result)
    make_si_figures(best_preds_df, transfer_summary_df, regime_result)
    log_stage_done("Render manuscript and SI figures plus source CSVs", t_stage)

    t_stage = log_stage("Write final summary")
    final_summary = {
        "n_raw_rows": int(len(raw_df)),
        "n_common_rows": int(len(common_df)),
        "targets": list(TARGETS.keys()),
        "families": list(families.keys()),
        "descriptor_regime_analysis_run": bool(regime_result),
        "elapsed_sec_total": time.time() - start,
        "top_level_outputs": {
            "tables": str(PATHS["tables"]),
            "figures_main": str(PATHS["figures_main"]),
            "figures_si": str(PATHS["figures_si"]),
            "figure_data_main": str(PATHS["figure_data_main"]),
            "figure_data_si": str(PATHS["figure_data_si"]),
            "models": str(PATHS["models"]),
        }
    }
    safe_json_dump(final_summary, PATHS["metadata"] / "final_summary.json")
    log_stage_done("Write final summary", t_stage)

    LOGGER.log("Pipeline completed successfully.")
    LOGGER.log(f"Total elapsed time: {fmt_elapsed(time.time() - start)}")
    LOGGER.log("=" * 88)


if __name__ == "__main__":
    main()
