"""
Full dataset inspection script — Phase A.
Run from the workspace root (where the CSV files live).
"""
import pandas as pd
import numpy as np
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(DATA_ROOT, "Training_BOP.csv")
TEST_PATH  = os.path.join(DATA_ROOT, "Testing_BOP.csv")

# ── Load ─────────────────────────────────────────────
print("=== LOADING DATA ===")
train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
test_raw  = pd.read_csv(TEST_PATH,  low_memory=False)
print(f"Training raw: {train_raw.shape}")
print(f"Testing raw:  {test_raw.shape}")

# Drop summary row
def drop_summary(df):
    return df[pd.to_numeric(df["sku"], errors="coerce").notna()].copy()

train = drop_summary(train_raw)
test  = drop_summary(test_raw)
print(f"Training after drop summary: {train.shape}")
print(f"Testing after drop summary:  {test.shape}")

# ── Columns / dtypes ─────────────────────────────────
print("\n=== COLUMNS ===")
print(list(train.columns))

print("\n=== DATA TYPES ===")
print(train.dtypes.to_string())

# ── Missing values ───────────────────────────────────
print("\n=== MISSING VALUES (TRAINING) ===")
miss_train = train.isnull().sum()
miss_train_pct = 100 * miss_train / len(train)
miss_summary = pd.DataFrame({"missing": miss_train, "pct": miss_train_pct.round(2)})
print(miss_summary[miss_summary["missing"] > 0].to_string())

print("\n=== MISSING VALUES (TESTING) ===")
miss_test = test.isnull().sum()
miss_test_pct = 100 * miss_test / len(test)
miss_summary_t = pd.DataFrame({"missing": miss_test, "pct": miss_test_pct.round(2)})
print(miss_summary_t[miss_summary_t["missing"] > 0].to_string())

# ── Duplicates ───────────────────────────────────────
print(f"\n=== DUPLICATES ===")
print(f"Training duplicates: {train.duplicated().sum()}")
print(f"Testing duplicates:  {test.duplicated().sum()}")

# ── Target distribution ──────────────────────────────
print("\n=== TARGET DISTRIBUTION (TRAINING) ===")
print(train["went_on_backorder"].value_counts(dropna=False).to_string())
target_missing_train = train["went_on_backorder"].isna().sum()
print(f"Missing target: {target_missing_train}")

print("\n=== TARGET DISTRIBUTION (TESTING) ===")
print(test["went_on_backorder"].value_counts(dropna=False).to_string())
target_missing_test = test["went_on_backorder"].isna().sum()
print(f"Missing target: {target_missing_test}")

# Backorder rates
train_valid = train[train["went_on_backorder"].isin(["Yes","No"])].copy()
test_valid  = test[test["went_on_backorder"].isin(["Yes","No"])].copy()
train_rate = (train_valid["went_on_backorder"] == "Yes").mean()
test_rate  = (test_valid["went_on_backorder"] == "Yes").mean()
print(f"\nBackorder rate (train): {train_rate:.4%}")
print(f"Backorder rate (test):  {test_rate:.4%}")

# ── Numerical distributions ──────────────────────────
num_cols = ["national_inv","lead_time","in_transit_qty","forecast_3_month","forecast_6_month",
            "forecast_9_month","sales_1_month","sales_3_month","sales_6_month","sales_9_month",
            "min_bank","pieces_past_due","perf_6_month_avg","perf_12_month_avg","local_bo_qty"]

print("\n=== NUMERICAL DISTRIBUTIONS (TRAINING) ===")
print(train[num_cols].describe().round(2).to_string())

# ── Sentinel / suspicious values ─────────────────────
print("\n=== SENTINEL & SUSPICIOUS VALUES ===")

# -99 counts
for col in num_cols:
    n99 = (train[col] == -99).sum()
    if n99 > 0:
        print(f"  {col}: -99 count = {n99:,}")

# Negative values
neg_inv = (train["national_inv"] < 0) & (train["national_inv"] != -99)
print(f"\n  national_inv negative (excl -99): {neg_inv.sum():,}")
print(f"  national_inv min: {train['national_inv'].min():.0f}")
print(f"  national_inv max: {train['national_inv'].max():.0f}")
print(f"  national_inv == -99: {(train['national_inv'] == -99).sum()}")

# Zero-heavy
print("\n=== ZERO-HEAVY VARIABLES ===")
for col in num_cols:
    pct_zero = 100 * (train[col] == 0).sum() / len(train)
    if pct_zero > 50:
        print(f"  {col}: {pct_zero:.1f}% zeros")

# Skewness (excl -99 sentinel)
print("\n=== SKEWNESS ===")
for col in num_cols:
    series = train[col].replace(-99, np.nan).dropna()
    skew = series.skew()
    print(f"  {col}: skew={skew:.2f}")

# ── Categorical/binary distributions ─────────────────
print("\n=== BINARY COLUMN DISTRIBUTIONS ===")
binary_cols = ["potential_issue","deck_risk","oe_constraint","ppap_risk","stop_auto_buy","rev_stop","went_on_backorder"]
for col in binary_cols:
    print(f"  {col}: {train[col].value_counts(dropna=False).to_dict()}")

# ── Backorder rate by negative inventory ─────────────
print("\n=== BACKORDER RATE BY INVENTORY SIGN ===")
tv = train_valid.copy()
tv["target"] = (tv["went_on_backorder"] == "Yes").astype(int)
print(f"  negative inv: {tv[tv['national_inv'] < 0]['target'].mean():.4f}")
print(f"  zero inv:     {tv[tv['national_inv'] == 0]['target'].mean():.4f}")
print(f"  positive inv: {tv[tv['national_inv'] > 0]['target'].mean():.4f}")

# ── Potential leakage check ──────────────────────────
print("\n=== LEAKAGE CHECK ===")
print("  local_bo_qty (local backorder qty): could reflect current backorder status")
print(f"  local_bo_qty > 0 backorder rate: {tv[tv['local_bo_qty'] > 0]['target'].mean():.4f}")
print(f"  local_bo_qty = 0 backorder rate: {tv[tv['local_bo_qty'] == 0]['target'].mean():.4f}")
print("  Decision: Keep — local_bo_qty reflects EXISTING local backorder state which")
print("            is a real pre-event signal available to supply chain managers.")

print("\n=== INSPECTION COMPLETE ===")
