"""
data_preprocessing.py
=====================
Reproducible preprocessing pipeline for the Supply Chain Backorder dataset.

Key cleaning decisions (documented):
- Last row is a dataset summary row (SKU = "(N rows)") — dropped.
- perf_6_month_avg / perf_12_month_avg: -99 is a sentinel for
  "no supplier performance data available", NOT a valid score.
  Replaced with NaN and treated as missing; then median-imputed.
- national_inv: Negative values are REAL (products with net negative
  stock due to committed orders exceeding available inventory). These
  are kept because they have a 15% backorder rate vs 0.4% for positive
  stock — highly predictive. The three rows with national_inv == -99
  are treated as sentinel missing values.
- lead_time: ~6% of rows missing. Missing is a signal in itself
  (these products have a lower but non-zero backorder rate). Imputed
  with median; a binary flag lead_time_missing is created to preserve
  the missingness signal.
- Binary categorical columns (Yes/No): encoded to 1/0.
- Target (went_on_backorder): rows with missing target excluded from
  supervised training/evaluation (only 1 row in each file).
- SKU: identifier only — NOT used as a feature.
- No oversampling applied globally; class_weight='balanced' used during
  modeling to handle imbalance.
- All fit operations (imputers) are fitted on training data only.
"""

import pandas as pd
import numpy as np
import joblib
import os

# ────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────

# Root of the project (two levels up from this file)
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_ROOT = os.path.dirname(PROJECT_ROOT)  # workspace root

TRAIN_PATH = os.path.join(DATA_ROOT, "Training_BOP.csv")
TEST_PATH = os.path.join(DATA_ROOT, "Testing_BOP.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# All feature columns (excluding SKU and target)
FEATURE_COLS = [
    "national_inv",
    "lead_time",
    "in_transit_qty",
    "forecast_3_month",
    "forecast_6_month",
    "forecast_9_month",
    "sales_1_month",
    "sales_3_month",
    "sales_6_month",
    "sales_9_month",
    "min_bank",
    "potential_issue",
    "pieces_past_due",
    "perf_6_month_avg",
    "perf_12_month_avg",
    "local_bo_qty",
    "deck_risk",
    "oe_constraint",
    "ppap_risk",
    "stop_auto_buy",
    "rev_stop",
]

TARGET_COL = "went_on_backorder"
SKU_COL = "sku"

# Binary Yes/No columns
BINARY_COLS = [
    "potential_issue",
    "deck_risk",
    "oe_constraint",
    "ppap_risk",
    "stop_auto_buy",
    "rev_stop",
]

# Performance columns with -99 sentinel
PERF_COLS = ["perf_6_month_avg", "perf_12_month_avg"]

# Numerical columns (after encoding binaries)
NUMERIC_COLS = [
    "national_inv",
    "lead_time",
    "in_transit_qty",
    "forecast_3_month",
    "forecast_6_month",
    "forecast_9_month",
    "sales_1_month",
    "sales_3_month",
    "sales_6_month",
    "sales_9_month",
    "min_bank",
    "pieces_past_due",
    "perf_6_month_avg",
    "perf_12_month_avg",
    "local_bo_qty",
]

# Engineered features
ENGINEERED_COLS = [
    "lead_time_missing",          # Binary flag: lead_time was originally NaN
    "inv_demand_ratio",           # national_inv / (forecast_3_month + 1)
    "stock_coverage_months",      # national_inv / (sales_1_month + 1)
    "demand_gap",                 # forecast_3_month - national_inv
    "supply_vs_demand",           # in_transit_qty / (forecast_3_month + 1)
    "total_risk_flags",           # Count of Yes risk binary flags
    "sales_acceleration",         # sales_3_month / (3 * sales_1_month + 1) - trend
    "avg_supplier_perf",          # Mean of perf_6/12 (with -99 replaced)
    "nat_inv_is_negative",        # Flag: national_inv < 0
    "past_due_flag",              # Flag: pieces_past_due > 0
    "local_bo_flag",              # Flag: local_bo_qty > 0
]

# Final modeling columns (features model actually uses)
MODEL_FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + ENGINEERED_COLS


# ────────────────────────────────────────────────────
# LOADING
# ────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load CSV and drop the trailing summary row."""
    df = pd.read_csv(path, low_memory=False)
    # The last row has a non-numeric SKU like "(N rows)" — drop it
    df = df[pd.to_numeric(df[SKU_COL], errors="coerce").notna()].copy()
    df[SKU_COL] = df[SKU_COL].astype(str)
    return df


def load_training_data() -> pd.DataFrame:
    return load_raw(TRAIN_PATH)


def load_testing_data() -> pd.DataFrame:
    return load_raw(TEST_PATH)


# ────────────────────────────────────────────────────
# CLEANING
# ────────────────────────────────────────────────────

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning rules to a raw dataframe.
    Does NOT fit any statistics — all replacements are rule-based.
    """
    df = df.copy()

    # 1. perf columns: replace -99 sentinel with NaN
    for col in PERF_COLS:
        df[col] = df[col].replace(-99.0, np.nan)

    # 2. national_inv: replace the 3 rows with exactly -99 with NaN
    #    (distinguishing them from legitimate negative inventory)
    #    Legitimate negatives (e.g., -27256) are kept as-is.
    df["national_inv"] = df["national_inv"].replace(-99.0, np.nan)

    # 3. Binary columns: standardize Yes→1, No→0, NaN→NaN (imputed later)
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # 4. Target encoding: Yes→1, No→0
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return df


# ────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from cleaned data.
    All features are logically defensible and derived from pre-backorder information.
    """
    df = df.copy()

    # --- Missingness flags (created BEFORE imputation) ---

    # Flag for missing lead_time (carries its own signal)
    df["lead_time_missing"] = df["lead_time"].isna().astype(int)

    # Flag for negative national inventory
    df["nat_inv_is_negative"] = (df["national_inv"] < 0).astype(int)

    # --- Inventory & demand pressure ---

    # Inventory-demand ratio: how many units of 3-month forecast does inventory cover?
    # Cap at 1000 to avoid extreme outliers when forecast is near zero
    df["inv_demand_ratio"] = (
        df["national_inv"] / (df["forecast_3_month"].fillna(0) + 1)
    ).clip(-1000, 1000)

    # Stock coverage in months relative to monthly sales
    df["stock_coverage_months"] = (
        df["national_inv"] / (df["sales_1_month"].fillna(0) + 1)
    ).clip(-500, 500)

    # Demand gap: how much more is forecasted over the next 3 months vs. what's in stock?
    # Positive = demand exceeds inventory (pressure signal)
    df["demand_gap"] = (
        df["forecast_3_month"].fillna(0) - df["national_inv"].fillna(0)
    )

    # In-transit vs 3-month forecast
    df["supply_vs_demand"] = (
        df["in_transit_qty"].fillna(0) / (df["forecast_3_month"].fillna(0) + 1)
    ).clip(0, 500)

    # --- Risk aggregation ---

    risk_flag_cols = ["potential_issue", "deck_risk", "oe_constraint", "ppap_risk", "rev_stop"]
    df["total_risk_flags"] = df[risk_flag_cols].fillna(0).sum(axis=1)

    # --- Sales trend ---
    # Sales acceleration: is 3-month sales accelerating vs. 1-month run rate?
    df["sales_acceleration"] = (
        df["sales_3_month"].fillna(0) / (3 * df["sales_1_month"].fillna(0) + 1)
    ).clip(0, 100)

    # --- Supplier performance composite ---
    # Average of available perf scores (NaN already replaced sentinel -99)
    df["avg_supplier_perf"] = df[["perf_6_month_avg", "perf_12_month_avg"]].mean(axis=1)

    # --- Operational flags ---
    df["past_due_flag"] = (df["pieces_past_due"].fillna(0) > 0).astype(int)
    df["local_bo_flag"] = (df["local_bo_qty"].fillna(0) > 0).astype(int)

    return df


# ────────────────────────────────────────────────────
# IMPUTATION (fitted on training data only)
# ────────────────────────────────────────────────────

def fit_imputer(df: pd.DataFrame) -> dict:
    """
    Fit median imputers for all numeric+binary columns on the training data.
    Returns a dict of {column: median_value}.
    """
    imputer_values = {}
    all_cols = list(set(NUMERIC_COLS + BINARY_COLS + ["avg_supplier_perf",
                                                        "inv_demand_ratio",
                                                        "stock_coverage_months",
                                                        "demand_gap",
                                                        "supply_vs_demand",
                                                        "sales_acceleration"]))
    for col in all_cols:
        if col in df.columns:
            median_val = df[col].median()
            imputer_values[col] = median_val
    return imputer_values


def apply_imputer(df: pd.DataFrame, imputer_values: dict) -> pd.DataFrame:
    """Apply pre-fitted imputation values to a dataframe."""
    df = df.copy()
    for col, val in imputer_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ────────────────────────────────────────────────────
# FULL PIPELINE
# ────────────────────────────────────────────────────

def prepare_training_data(df_train_raw: pd.DataFrame):
    """
    Full pipeline for training data:
    1. Clean
    2. Drop rows with missing target
    3. Engineer features
    4. Fit imputer on training data only
    5. Apply imputation
    Returns: (X_train, y_train, sku_train, imputer_values)
    """
    df = clean_raw(df_train_raw)

    # Drop rows with missing target (only 1 row)
    n_before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    n_dropped = n_before - len(df)
    print(f"  Dropped {n_dropped} rows with missing target (training)")

    y = df[TARGET_COL].astype(int)
    sku = df[SKU_COL].copy()

    df = engineer_features(df)

    # Fit imputer on training
    imputer_values = fit_imputer(df)

    # Apply imputation
    df = apply_imputer(df, imputer_values)

    # Select modeling features
    feat_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    X = df[feat_cols].copy()

    print(f"  Training: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  Backorder rate: {y.mean():.4f} ({y.sum():,} positives / {len(y):,} total)")

    return X, y, sku, imputer_values, feat_cols


def prepare_test_data(df_test_raw: pd.DataFrame, imputer_values: dict, feat_cols: list):
    """
    Full pipeline for test/holdout data:
    Uses pre-fitted imputer from training — NO re-fitting.
    Returns: (X_test, y_test, sku_test)
    """
    df = clean_raw(df_test_raw)

    # Drop rows with missing target for evaluation, but keep for prediction
    has_target = df[TARGET_COL].notna()
    n_dropped = (~has_target).sum()
    print(f"  Dropped {n_dropped} rows with missing target (test)")
    df = df[has_target].copy()

    y = df[TARGET_COL].astype(int)
    sku = df[SKU_COL].copy()

    df = engineer_features(df)
    df = apply_imputer(df, imputer_values)

    X = df[feat_cols].copy()

    print(f"  Test: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  Backorder rate: {y.mean():.4f} ({y.sum():,} positives / {len(y):,} total)")

    return X, y, sku


def prepare_inference_data(df_raw: pd.DataFrame, imputer_values: dict, feat_cols: list) -> pd.DataFrame:
    """
    Prepare data for inference (no target column required).
    Used in the Streamlit batch prediction page.
    """
    df = clean_raw(df_raw)
    df = engineer_features(df)
    df = apply_imputer(df, imputer_values)
    
    # Add any missing columns with 0
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df[feat_cols].copy()


# ────────────────────────────────────────────────────
# PERSISTENCE
# ────────────────────────────────────────────────────

def save_preprocessing_artifacts(imputer_values: dict, feat_cols: list):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {"imputer_values": imputer_values, "feat_cols": feat_cols},
        os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"),
    )
    print(f"  Saved preprocessing artifacts to {MODELS_DIR}/preprocessing_pipeline.joblib")


def load_preprocessing_artifacts() -> tuple:
    path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    artifacts = joblib.load(path)
    return artifacts["imputer_values"], artifacts["feat_cols"]
