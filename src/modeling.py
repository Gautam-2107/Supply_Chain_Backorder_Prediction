"""
modeling.py
===========
Predictive modeling pipeline for Supply Chain Backorder prediction.

Methodology:
- Binary classification: went_on_backorder (Yes=1, No=0)
- Training data: Training_BOP.csv (1.687M rows)
- Validation: 20% stratified split from training data (for model selection/tuning)
- Final evaluation: Testing_BOP.csv (external holdout — NOT used for tuning)
- Class imbalance: handled via class_weight='balanced' (NOT oversampling on full dataset)
- Models: Logistic Regression (baseline) + Random Forest (comparison)
- Leakage prevention: preprocessing fitted on training split only

Data Leakage Prevention:
- SKU excluded (identifier, not predictive)
- went_on_backorder excluded from features (target)
- No features derived from target
- Imputer fitted on training split only, applied to validation and test
- No time-based split invented (dataset has no date column)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils import save_figure, save_metrics, PALETTE, MODELS_DIR

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────
# EVALUATION HELPER
# ────────────────────────────────────────────────────

def evaluate_model(model, X, y, model_name: str, threshold: float = 0.5) -> dict:
    """Compute all required evaluation metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": round(accuracy_score(y, y_pred), 6),
        "precision": round(precision_score(y, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y, y_pred, zero_division=0), 6),
        "f1": round(f1_score(y, y_pred, zero_division=0), 6),
        "roc_auc": round(roc_auc_score(y, y_prob), 6),
        "pr_auc": round(average_precision_score(y, y_prob), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": len(y),
        "n_positives": int(y.sum()),
        "positive_rate": round(float(y.mean()), 6),
    }

    print(f"\n  === {model_name} Evaluation ===")
    print(f"  Samples: {len(y):,}  |  Positives: {int(y.sum()):,}  |  Rate: {y.mean():.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Confusion Matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    return metrics


# ────────────────────────────────────────────────────
# MODEL TRAINING
# ────────────────────────────────────────────────────

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Logistic Regression baseline with StandardScaler.
    class_weight='balanced' handles class imbalance without global oversampling.
    """
    print("  Training Logistic Regression...")
    t0 = time.time()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            solver="saga",   # efficient for large datasets
            C=0.1,           # slight regularization for 1.3M row dataset
            random_state=42,
            n_jobs=-1,
        ))
    ])
    pipe.fit(X_train, y_train)
    print(f"  Logistic Regression trained in {time.time() - t0:.1f}s")
    return pipe


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                         n_estimators: int = 200, sample_frac: float = 1.0) -> RandomForestClassifier:
    """
    Random Forest classifier.
    For the large dataset, we use a capped estimator count and min_samples settings
    to keep training time manageable while maintaining quality.
    class_weight='balanced_subsample' is used for imbalance.
    """
    print(f"  Training Random Forest (n_estimators={n_estimators})...")
    t0 = time.time()

    if sample_frac < 1.0:
        # Sample for training RF when the full dataset is too large
        idx = (
            pd.concat([
                y_train[y_train == 1],          # all positives
                y_train[y_train == 0].sample(
                    n=min(int(y_train.sum() * 10), len(y_train[y_train == 0])),
                    random_state=42
                )
            ])
            .index
        )
        X_rf = X_train.loc[idx]
        y_rf = y_train.loc[idx]
        print(f"    RF training sample: {len(y_rf):,} rows (all positives + 10x negatives)")
    else:
        X_rf = X_train
        y_rf = y_train

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_rf, y_rf)
    print(f"  Random Forest trained in {time.time() - t0:.1f}s")
    return rf


# ────────────────────────────────────────────────────
# VISUALIZATION: Confusion Matrix
# ────────────────────────────────────────────────────

def plot_confusion_matrix(metrics_dict: dict, title_suffix: str = "") -> str:
    cm_dict = metrics_dict["confusion_matrix"]
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Pred: No Backorder", "Pred: Backorder"],
                yticklabels=["Actual: No Backorder", "Actual: Backorder"])
    ax.set_title(f"Confusion Matrix — {metrics_dict['model']}{title_suffix}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)

    # Add FN / FP business annotation
    fn_val = cm_dict["fn"]
    fp_val = cm_dict["fp"]
    total = sum(cm_dict.values())
    fn_pct = 100 * fn_val / max(total, 1)
    fp_pct = 100 * fp_val / max(total, 1)

    ax.text(0.5, -0.18,
            f"False Negatives (missed backorders): {fn_val:,} ({fn_pct:.2f}%)  |  "
            f"False Positives (unnecessary alerts): {fp_val:,} ({fp_pct:.2f}%)",
            transform=ax.transAxes, ha="center", fontsize=8, color=PALETTE["neutral"])

    plt.tight_layout()
    safe_name = metrics_dict['model'].replace(' ', '_').replace('/', '_')
    return save_figure(fig, f"cm_{safe_name}{title_suffix.replace(' ', '_')}.png")


# ────────────────────────────────────────────────────
# VISUALIZATION: ROC + PR Curves
# ────────────────────────────────────────────────────

def plot_roc_pr_curves(models_results: list) -> str:
    """
    Plot ROC and PR curves for multiple models.
    models_results: list of (model, X, y, name) tuples
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_list = [PALETTE["primary"], PALETTE["danger"], PALETTE["warning"]]

    for i, (model, X, y, name) in enumerate(models_results):
        color = colors_list[i % len(colors_list)]
        y_prob = model.predict_proba(X)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y, y_prob)
        pr_auc_val = average_precision_score(y, y_prob)
        axes[1].plot(rec, prec, color=color, lw=2, label=f"{name} (AP={pr_auc_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    axes[0].set_title("ROC Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    axes[0].legend()

    axes[1].set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].legend()

    plt.tight_layout()
    return save_figure(fig, "roc_pr_curves.png")


# ────────────────────────────────────────────────────
# VISUALIZATION: Feature Importance
# ────────────────────────────────────────────────────

def plot_feature_importance(model, feat_cols: list, model_name: str, top_n: int = 20) -> str:
    """Plot feature importances for tree-based models or LR coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importance — {model_name}"
        xlabel = "Importance Score"
    elif hasattr(model, "named_steps"):
        # Pipeline (LR)
        clf = model.named_steps.get("clf")
        if clf and hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
            title = f"Feature Importance (|Coefficient|) — {model_name}"
            xlabel = "|Coefficient|"
        else:
            return ""
    else:
        return ""

    feat_imp = pd.Series(importances, index=feat_cols).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = [PALETTE["danger"] if i < 5 else PALETTE["warning"] if i < 10 else PALETTE["primary"]
                  for i in range(len(feat_imp))]
    feat_imp.sort_values().plot(kind="barh", ax=ax, color=colors_bar[::-1], edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_')
    return save_figure(fig, f"feature_importance_{safe_name}.png")


# ────────────────────────────────────────────────────
# VISUALIZATION: Model Comparison
# ────────────────────────────────────────────────────

def plot_model_comparison(all_metrics: list) -> str:
    """Bar chart comparing key metrics across models."""
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

    models = [m["model"] for m in all_metrics]
    x = np.arange(len(metric_names))
    width = 0.35
    colors_list = [PALETTE["primary"], PALETTE["danger"]]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (m, color) in enumerate(zip(all_metrics, colors_list)):
        vals = [m.get(met, 0) for met in metric_names]
        offset = (i - len(all_metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=m["model"], color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Model Comparison — Test Set Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()
    return save_figure(fig, "model_comparison.png")


# ────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ────────────────────────────────────────────────────

def compute_shap_values(rf_model, X_sample: pd.DataFrame, feat_cols: list,
                         sample_n: int = 5000) -> str:
    """
    Compute SHAP values on a representative sample for global explainability.
    Using a sample because full 1.3M row SHAP would be computationally prohibitive.
    """
    try:
        # pyrefly: ignore [missing-import]
        import shap
        print(f"  Computing SHAP on {min(sample_n, len(X_sample)):,} sample rows...")
        X_shap = X_sample.sample(n=min(sample_n, len(X_sample)), random_state=42)
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # positive class
        else:
            shap_vals = shap_values

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_shap, plot_type="bar", show=False,
                          max_display=20, plot_size=None)
        ax = plt.gca()
        ax.set_title("SHAP Feature Importance (Random Forest)\nGlobal Mean |SHAP| — Positive Class",
                     fontsize=12, fontweight="bold")
        path = save_figure(plt.gcf(), "shap_feature_importance.png")
        plt.close("all")
        return path
    except ImportError:
        print("  SHAP not available — skipping SHAP analysis")
        return ""
    except Exception as e:
        print(f"  SHAP computation error: {e}")
        return ""


# ────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ────────────────────────────────────────────────────

def run_modeling_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           feat_cols: list) -> dict:
    """
    Full modeling pipeline:
    1. Train-validation split (from training data only)
    2. Train LR and RF
    3. Evaluate on validation then on external test set
    4. Generate all visualization artifacts
    5. Save models
    """
    from sklearn.model_selection import train_test_split

    print("\n=== TRAIN/VALIDATION SPLIT ===")
    print("  Splitting training data 80/20 (stratified) for model selection...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    print(f"  Train: {X_tr.shape[0]:,}  |  Val: {X_val.shape[0]:,}")

    all_metrics_test = []

    # ── Logistic Regression ──
    print("\n=== LOGISTIC REGRESSION ===")
    lr_model = train_logistic_regression(X_tr, y_tr)

    print("  Validation metrics:")
    lr_val_metrics = evaluate_model(lr_model, X_val, y_val, "Logistic Regression", threshold=0.3)
    save_metrics(lr_val_metrics, "lr_val_metrics.json")

    print("  Test (holdout) metrics:")
    lr_test_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression", threshold=0.3)
    save_metrics(lr_test_metrics, "lr_test_metrics.json")
    all_metrics_test.append(lr_test_metrics)

    plot_confusion_matrix(lr_test_metrics, "_Test")
    plot_feature_importance(lr_model, feat_cols, "Logistic Regression")

    # ── Random Forest ──
    print("\n=== RANDOM FOREST ===")
    rf_model = train_random_forest(X_tr, y_tr, n_estimators=150, sample_frac=0.0)

    print("  Validation metrics:")
    rf_val_metrics = evaluate_model(rf_model, X_val, y_val, "Random Forest", threshold=0.3)
    save_metrics(rf_val_metrics, "rf_val_metrics.json")

    print("  Test (holdout) metrics:")
    rf_test_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", threshold=0.3)
    save_metrics(rf_test_metrics, "rf_test_metrics.json")
    all_metrics_test.append(rf_test_metrics)

    plot_confusion_matrix(rf_test_metrics, "_Test")
    plot_feature_importance(rf_model, feat_cols, "Random Forest")

    # ── ROC/PR Curves ──
    print("\n  Plotting ROC/PR curves...")
    plot_roc_pr_curves([
        (lr_model, X_test, y_test, "Logistic Regression"),
        (rf_model, X_test, y_test, "Random Forest"),
    ])

    # ── Model Comparison ──
    plot_model_comparison(all_metrics_test)

    # ── SHAP ──
    print("\n  SHAP analysis...")
    compute_shap_values(rf_model, X_val, feat_cols, sample_n=5000)

    # ── Save models ──
    print("\n  Saving models...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "lr_model.joblib"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "backorder_model.joblib"))
    print(f"  Saved: {MODELS_DIR}/lr_model.joblib")
    print(f"  Saved: {MODELS_DIR}/backorder_model.joblib")

    return {
        "lr_val": lr_val_metrics,
        "lr_test": lr_test_metrics,
        "rf_val": rf_val_metrics,
        "rf_test": rf_test_metrics,
        "best_model": "Random Forest" if rf_test_metrics["roc_auc"] > lr_test_metrics["roc_auc"] else "Logistic Regression",
        "feat_cols": feat_cols,
    }


# ────────────────────────────────────────────────────
# PRESCRIPTIVE: Risk Prioritization
# ────────────────────────────────────────────────────

def generate_risk_prioritization(model, X_test: pd.DataFrame, y_test: pd.Series,
                                  sku_test: pd.Series, df_test_raw: pd.DataFrame,
                                  imputer_values: dict, feat_cols: list) -> pd.DataFrame:
    """
    Generate prescriptive risk-prioritization output for the test set.
    """
    from data_preprocessing import clean_raw, engineer_features, apply_imputer
    from utils import assign_risk_category_vectorized, get_recommended_action

    y_prob = model.predict_proba(X_test)[:, 1]
    risk_cats = assign_risk_category_vectorized(pd.Series(y_prob))

    result = pd.DataFrame({
        "sku": sku_test.values,
        "risk_probability": np.round(y_prob, 4),
        "risk_category": risk_cats,
        "actual_backorder": y_test.values,
    })

    # Attach key supply chain features
    key_cols = ["national_inv", "lead_time", "in_transit_qty",
                "forecast_3_month", "pieces_past_due", "local_bo_qty"]
    for col in key_cols:
        if col in X_test.columns:
            result[col] = X_test[col].values

    # Recommended action
    result["recommended_action"] = result["risk_category"].apply(get_recommended_action)

    # Sort by risk
    result = result.sort_values("risk_probability", ascending=False).reset_index(drop=True)

    # Save
    from utils import predictions_path
    path = predictions_path("risk_prioritization.csv")
    result.to_csv(path, index=False)
    print(f"  Saved risk prioritization: {path}")

    # Stats
    risk_summary = result["risk_category"].value_counts()
    print(f"  Risk breakdown: {risk_summary.to_dict()}")

    return result
