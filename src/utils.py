"""
utils.py
========
Shared utility functions for the Supply Chain Backorder project.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ────────────────────────────────────────────────────
# PATH HELPERS
# ────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")


def fig_path(name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)


def metrics_path(name: str) -> str:
    os.makedirs(METRICS_DIR, exist_ok=True)
    return os.path.join(METRICS_DIR, name)


def predictions_path(name: str) -> str:
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    return os.path.join(PREDICTIONS_DIR, name)


# ────────────────────────────────────────────────────
# PLOTTING STYLE
# ────────────────────────────────────────────────────

PALETTE = {
    "primary": "#2563EB",
    "danger": "#DC2626",
    "warning": "#D97706",
    "success": "#16A34A",
    "neutral": "#6B7280",
    "bg": "#F9FAFB",
    "yes": "#DC2626",
    "no": "#2563EB",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.grid": True,
    "grid.color": "#F3F4F6",
    "grid.alpha": 0.8,
    "font.family": "DejaVu Sans",
})


def save_figure(fig, name: str, dpi: int = 150) -> str:
    path = fig_path(name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def pct_formatter(x, pos):
    return f"{x:.1f}%"


# ────────────────────────────────────────────────────
# METRICS PERSISTENCE
# ────────────────────────────────────────────────────

def save_metrics(metrics_dict: dict, name: str):
    path = metrics_path(name)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"  Saved metrics: {path}")


def load_metrics(name: str) -> dict:
    path = metrics_path(name)
    with open(path) as f:
        return json.load(f)


# ────────────────────────────────────────────────────
# RISK CATEGORIZATION
# ────────────────────────────────────────────────────

def assign_risk_category(prob: float) -> str:
    """
    Assign risk category based on predicted probability.
    Thresholds are derived from the operational context:
    - High Risk (>=0.30): Products with >30% predicted backorder probability warrant
      immediate intervention given the large cost of a stockout.
    - Medium Risk (0.10-0.30): Products that need monitoring and may require
      proactive procurement review.
    - Low Risk (<0.10): Standard monitoring only.
    """
    if prob >= 0.30:
        return "High Risk"
    elif prob >= 0.10:
        return "Medium Risk"
    else:
        return "Low Risk"


def assign_risk_category_vectorized(probs: pd.Series) -> pd.Series:
    conditions = [probs >= 0.30, probs >= 0.10]
    choices = ["High Risk", "Medium Risk"]
    return np.select(conditions, choices, default="Low Risk")


def get_recommended_action(risk_cat: str, row: dict = None) -> str:
    """Return a prioritized recommended action based on risk category and context."""
    if risk_cat == "High Risk":
        actions = ["Expedite replenishment immediately"]
        if row:
            if row.get("lead_time", 0) and float(row.get("lead_time", 0)) > 14:
                actions.append("review supplier lead time (high)")
            if row.get("pieces_past_due", 0) and float(row.get("pieces_past_due", 0)) > 0:
                actions.append("resolve past-due shipments")
            if row.get("local_bo_qty", 0) and float(row.get("local_bo_qty", 0)) > 0:
                actions.append("address existing local backorders")
        return "; ".join(actions)
    elif risk_cat == "Medium Risk":
        actions = ["Flag for procurement review"]
        if row:
            if row.get("national_inv", 0) is not None and float(row.get("national_inv", 999)) < 10:
                actions.append("increase safety stock")
            if row.get("forecast_3_month", 0) and float(row.get("forecast_3_month", 0)) > 0:
                actions.append("verify forecast accuracy")
        return "; ".join(actions)
    else:
        return "Monitor per standard cycle"


RISK_COLORS = {
    "High Risk": "#DC2626",
    "Medium Risk": "#D97706",
    "Low Risk": "#16A34A",
}

RISK_EMOJI = {
    "High Risk": "🔴",
    "Medium Risk": "🟡",
    "Low Risk": "🟢",
}
