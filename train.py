"""
train.py
========
Main training script. Run this to:
1. Load and preprocess data
2. Run all descriptive/diagnostic analytics
3. Train models
4. Evaluate models
5. Generate all output artifacts

Usage:
    cd Supply_Chain_Backorder_Project
    python train.py
"""

import os
import sys
import time

# Add src to path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from data_preprocessing import (
    load_training_data, load_testing_data,
    clean_raw, prepare_training_data, prepare_test_data,
    save_preprocessing_artifacts
)
from analysis import run_all_analysis
from modeling import run_modeling_pipeline, generate_risk_prioritization
from utils import save_metrics

def main():
    t_start = time.time()
    print("=" * 60)
    print("SUPPLY CHAIN BACKORDER PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/6] Loading raw data...")
    df_train_raw = load_training_data()
    df_test_raw = load_testing_data()
    print(f"  Training raw: {df_train_raw.shape}")
    print(f"  Testing raw:  {df_test_raw.shape}")

    # ── 2. Clean (for analytics — before imputer fitting) ─────────
    print("\n[2/6] Cleaning data for analytics...")
    from data_preprocessing import clean_raw
    df_train_clean = clean_raw(df_train_raw)
    df_test_clean = clean_raw(df_test_raw)
    print(f"  Training cleaned: {df_train_clean.shape}")
    print(f"  Testing cleaned:  {df_test_clean.shape}")

    # ── 3. Run analytics ─────────────────────────────────────────
    print("\n[3/6] Running descriptive & diagnostic analytics...")
    kpis = run_all_analysis(df_train_clean, df_test_clean)
    print(f"  Backorder rate (train): {kpis['backorder_rate_train_pct']:.4f}%")
    print(f"  Backorder rate (test):  {kpis['backorder_rate_test_pct']:.4f}%")

    # ── 4. Prepare modeling data ──────────────────────────────────
    print("\n[4/6] Preparing modeling data...")
    X_train, y_train, sku_train, imputer_values, feat_cols = prepare_training_data(df_train_raw)
    X_test, y_test, sku_test = prepare_test_data(df_test_raw, imputer_values, feat_cols)
    save_preprocessing_artifacts(imputer_values, feat_cols)

    # ── 5. Train & evaluate models ────────────────────────────────
    print("\n[5/6] Training and evaluating models...")
    model_results = run_modeling_pipeline(X_train, y_train, X_test, y_test, feat_cols)
    save_metrics(model_results, "model_results_summary.json")

    # ── 6. Prescriptive: risk prioritization ─────────────────────
    print("\n[6/6] Generating prescriptive risk prioritization...")
    import joblib
    from utils import MODELS_DIR
    best_model = joblib.load(os.path.join(MODELS_DIR, "backorder_model.joblib"))
    risk_df = generate_risk_prioritization(
        best_model, X_test, y_test, sku_test, df_test_raw, imputer_values, feat_cols
    )

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best model: {model_results['best_model']}")
    print(f"  LR  ROC-AUC: {model_results['lr_test']['roc_auc']:.4f}")
    print(f"  RF  ROC-AUC: {model_results['rf_test']['roc_auc']:.4f}")
    print(f"  LR  PR-AUC:  {model_results['lr_test']['pr_auc']:.4f}")
    print(f"  RF  PR-AUC:  {model_results['rf_test']['pr_auc']:.4f}")
    print(f"\n  Risk summary (test set):")
    risk_summary = risk_df["risk_category"].value_counts()
    for cat, cnt in risk_summary.items():
        print(f"    {cat}: {cnt:,}")
    print("\n  Outputs saved to:")
    print(f"    models/      — trained model artifacts")
    print(f"    outputs/figures/   — all visualizations")
    print(f"    outputs/metrics/   — all evaluation metrics")
    print(f"    outputs/predictions/ — risk prioritization CSV")


if __name__ == "__main__":
    main()
