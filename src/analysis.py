"""
analysis.py
===========
Descriptive and diagnostic analytics for the Supply Chain Backorder project.

Generates all visualizations and computes KPIs.
All analysis is performed on the cleaned, pre-imputation dataset so that
the charts reflect real data characteristics (including missingness patterns).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils import save_figure, save_metrics, PALETTE, FIGURES_DIR, METRICS_DIR

# ────────────────────────────────────────────────────
# KPI CALCULATION
# ────────────────────────────────────────────────────

def compute_kpis(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """
    Compute business KPIs from the cleaned training dataset.
    """
    # Use valid-target rows
    train_valid = df_train[df_train["went_on_backorder"].isin([1, 0])].copy()
    test_valid = df_test[df_test["went_on_backorder"].isin([1, 0])].copy()

    total_products_train = len(train_valid)
    total_backorders_train = int(train_valid["went_on_backorder"].sum())
    backorder_rate_train = float(train_valid["went_on_backorder"].mean())

    total_products_test = len(test_valid)
    total_backorders_test = int(test_valid["went_on_backorder"].sum())
    backorder_rate_test = float(test_valid["went_on_backorder"].mean())

    # Inventory statistics
    inv = train_valid["national_inv"]
    neg_inv_count = int((inv < 0).sum())

    # Lead time
    lt = train_valid["lead_time"].dropna()

    # Operational risk
    deck_risk_pct = float(train_valid["deck_risk"].mean()) if "deck_risk" in train_valid.columns else None
    ppap_risk_pct = float(train_valid["ppap_risk"].mean()) if "ppap_risk" in train_valid.columns else None

    kpis = {
        "total_products_train": total_products_train,
        "total_backorders_train": total_backorders_train,
        "backorder_rate_train": round(backorder_rate_train, 6),
        "backorder_rate_train_pct": round(backorder_rate_train * 100, 4),
        "total_products_test": total_products_test,
        "total_backorders_test": total_backorders_test,
        "backorder_rate_test": round(backorder_rate_test, 6),
        "backorder_rate_test_pct": round(backorder_rate_test * 100, 4),
        "products_with_negative_inventory": neg_inv_count,
        "pct_negative_inventory": round(100 * neg_inv_count / total_products_train, 2),
        "median_lead_time": round(float(lt.median()), 1),
        "mean_lead_time": round(float(lt.mean()), 2),
        "pct_missing_lead_time": round(100 * train_valid["lead_time"].isna().mean(), 2),
        "deck_risk_pct": round(deck_risk_pct * 100, 2) if deck_risk_pct else None,
        "ppap_risk_pct": round(ppap_risk_pct * 100, 2) if ppap_risk_pct else None,
        "median_national_inv": round(float(inv.median()), 1),
        "mean_national_inv": round(float(inv.mean()), 1),
        "pct_zero_transit": round(100 * (train_valid["in_transit_qty"] == 0).mean(), 1)
            if "in_transit_qty" in train_valid.columns else None,
    }
    return kpis


# ────────────────────────────────────────────────────
# VISUALIZATION 1: Backorder Distribution (Target Overview)
# ────────────────────────────────────────────────────

def plot_backorder_distribution(df: pd.DataFrame) -> str:
    """Bar chart showing backorder vs non-backorder counts with rate annotation."""
    valid = df[df["went_on_backorder"].isin([1, 0])].copy()
    counts = valid["went_on_backorder"].value_counts().sort_index()
    labels = ["No Backorder", "Backorder"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [PALETTE["no"], PALETTE["yes"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: count
    bars = axes[0].bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Backorder Count Distribution", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Products", fontsize=11)
    axes[0].set_xlabel("")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # Right: percentage
    total = sum(values)
    pcts = [100 * v / total for v in values]
    bars2 = axes[1].bar(labels, pcts, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Backorder Rate (%)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    for bar, pct in zip(bars2, pcts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f"{pct:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    rate = 100 * values[1] / total
    fig.suptitle(
        f"Supply Chain Backorder Overview  |  Overall Rate: {rate:.2f}%  |  "
        f"Total Products: {total:,}",
        fontsize=12, color=PALETTE["neutral"], y=1.01
    )
    plt.tight_layout()
    return save_figure(fig, "01_backorder_distribution.png")


# ────────────────────────────────────────────────────
# VISUALIZATION 2: National Inventory by Backorder Status
# ────────────────────────────────────────────────────

def plot_inventory_by_backorder(df: pd.DataFrame) -> str:
    """Box plots of national inventory split by backorder status (log-transformed)."""
    valid = df[df["went_on_backorder"].isin([1, 0])].copy()
    valid["status"] = valid["went_on_backorder"].map({1: "Backorder", 0: "No Backorder"})

    # Clip to reasonable range for visualization (keep negatives visible)
    clipped = valid["national_inv"].clip(-500, 5000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: box plot
    groups = [
        clipped[valid["status"] == "No Backorder"].values,
        clipped[valid["status"] == "Backorder"].values,
    ]
    bp = axes[0].boxplot(groups, tick_labels=["No Backorder", "Backorder"],
                          patch_artist=True, notch=True, showfliers=False,
                          medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(PALETTE["no"] + "88")
    bp["boxes"][1].set_facecolor(PALETTE["yes"] + "88")
    axes[0].set_title("National Inventory by Backorder Status\n(clipped: -500 to 5000)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("National Inventory (units)", fontsize=10)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5, label="Zero inventory")
    axes[0].legend(fontsize=9)

    # Right: backorder rate by inventory bracket
    bins = [-np.inf, 0, 5, 20, 50, 200, np.inf]
    labels = ["<0 (negative)", "0–5", "6–20", "21–50", "51–200", ">200"]
    valid["inv_bin"] = pd.cut(valid["national_inv"], bins=bins, labels=labels)
    rates = valid.groupby("inv_bin", observed=True)["went_on_backorder"].mean() * 100
    counts = valid.groupby("inv_bin", observed=True)["went_on_backorder"].count()

    colors_bars = [PALETTE["danger"] if r > 5 else PALETTE["warning"] if r > 1 else PALETTE["success"]
                   for r in rates.values]
    bars = axes[1].bar(rates.index.astype(str), rates.values, color=colors_bars, edgecolor="white")
    axes[1].set_title("Backorder Rate by Inventory Level", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Backorder Rate (%)", fontsize=10)
    axes[1].set_xlabel("National Inventory Bracket", fontsize=10)
    for bar, rate_val, cnt in zip(bars, rates.values, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{rate_val:.1f}%\n(n={cnt/1000:.0f}K)", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return save_figure(fig, "02_inventory_by_backorder.png")


# ────────────────────────────────────────────────────
# VISUALIZATION 3: Lead Time vs Backorder Rate
# ────────────────────────────────────────────────────

def plot_leadtime_analysis(df: pd.DataFrame) -> str:
    """Analysis of lead time vs backorder rate."""
    valid = df[df["went_on_backorder"].isin([1, 0])].copy()
    valid_lt = valid[valid["lead_time"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Backorder rate by lead time bucket
    bins = [0, 4, 8, 12, 20, 53]
    labels = ["≤4 wks", "5–8 wks", "9–12 wks", "13–20 wks", ">20 wks"]
    valid_lt["lt_bin"] = pd.cut(valid_lt["lead_time"], bins=bins, labels=labels, include_lowest=True)
    lt_rates = valid_lt.groupby("lt_bin", observed=True)["went_on_backorder"].mean() * 100
    lt_counts = valid_lt.groupby("lt_bin", observed=True)["went_on_backorder"].count()

    colors_bars = [PALETTE["danger"] if r > 2 else PALETTE["warning"] if r > 0.5 else PALETTE["success"]
                   for r in lt_rates.values]
    bars = axes[0].bar(lt_rates.index.astype(str), lt_rates.values, color=colors_bars, edgecolor="white")
    axes[0].set_title("Backorder Rate by Lead Time Bucket", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Backorder Rate (%)", fontsize=10)
    axes[0].set_xlabel("Lead Time (weeks)", fontsize=10)
    for bar, r, n in zip(bars, lt_rates.values, lt_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{r:.2f}%\n(n={n/1000:.0f}K)", ha="center", va="bottom", fontsize=8)

    # Right: Missing vs present lead time
    missing_rate = valid[valid["lead_time"].isna()]["went_on_backorder"].mean() * 100
    present_rate = valid[valid["lead_time"].notna()]["went_on_backorder"].mean() * 100
    missing_n = valid["lead_time"].isna().sum()
    present_n = valid["lead_time"].notna().sum()

    axes[1].bar(["Lead Time Present", "Lead Time Missing"],
                [present_rate, missing_rate],
                color=[PALETTE["primary"], PALETTE["neutral"]],
                width=0.5, edgecolor="white")
    axes[1].set_title("Backorder Rate: Missing vs. Present Lead Time", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Backorder Rate (%)", fontsize=10)
    for i, (r, n) in enumerate([(present_rate, present_n), (missing_rate, missing_n)]):
        axes[1].text(i, r + 0.05, f"{r:.2f}%\n(n={n/1000:.0f}K)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return save_figure(fig, "03_leadtime_analysis.png")


# ────────────────────────────────────────────────────
# VISUALIZATION 4: Supplier Performance vs Backorder
# ────────────────────────────────────────────────────

def plot_supplier_performance(df: pd.DataFrame) -> str:
    """Supplier performance (perf_6_month_avg) vs backorder status."""
    valid = df[df["went_on_backorder"].isin([1, 0])].copy()

    # Replace -99 sentinel
    valid["perf_6"] = valid["perf_6_month_avg"].replace(-99, np.nan)
    valid["perf_12"] = valid["perf_12_month_avg"].replace(-99, np.nan)
    valid["status"] = valid["went_on_backorder"].map({1: "Backorder", 0: "No Backorder"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Distribution of perf_6 by backorder status
    for status, color in [("No Backorder", PALETTE["no"]), ("Backorder", PALETTE["yes"])]:
        subset = valid[valid["status"] == status]["perf_6"].dropna()
        axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=status,
                     density=True, range=(0, 1))
    axes[0].set_title("6-Month Supplier Performance\nDistribution by Backorder Status", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Supplier Performance Score (0–1)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend()
    axes[0].axvline(valid["perf_6"].median(), color="gray", linestyle="--",
                    alpha=0.7, label=f"Median: {valid['perf_6'].median():.2f}")

    # Right: Backorder rate by perf quintile
    valid_perf = valid[valid["perf_6"].notna()].copy()
    valid_perf["perf_bin"] = pd.qcut(valid_perf["perf_6"], q=5, duplicates="drop")
    perf_rates = valid_perf.groupby("perf_bin", observed=True)["went_on_backorder"].mean() * 100
    labels_perf = [str(b) for b in perf_rates.index]

    axes[1].bar(range(len(perf_rates)), perf_rates.values,
                color=[PALETTE["danger"] if r > 2 else PALETTE["warning"] if r > 0.5 else PALETTE["success"]
                       for r in perf_rates.values],
                edgecolor="white")
    axes[1].set_xticks(range(len(perf_rates)))
    axes[1].set_xticklabels([f"Q{i+1}" for i in range(len(perf_rates))], fontsize=9)
    axes[1].set_title("Backorder Rate by Supplier\nPerformance Quintile (Q1=lowest)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Backorder Rate (%)", fontsize=10)
    axes[1].set_xlabel("Supplier Performance Quintile", fontsize=10)
    for i, (r, lbl) in enumerate(zip(perf_rates.values, labels_perf)):
        axes[1].text(i, r + 0.05, f"{r:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return save_figure(fig, "04_supplier_performance.png")


# ────────────────────────────────────────────────────
# VISUALIZATION 5: Risk Flags vs Backorder Rate
# ────────────────────────────────────────────────────

def plot_risk_flags(df: pd.DataFrame) -> str:
    """Compare backorder rates across operational risk flags."""
    valid = df[df["went_on_backorder"].isin([1, 0])].copy()

    risk_cols = ["potential_issue", "deck_risk", "oe_constraint", "ppap_risk", "rev_stop", "stop_auto_buy"]
    risk_labels = ["Potential\nIssue", "Deck Risk", "OE\nConstraint", "PPAP Risk", "Rev Stop", "Stop\nAuto-Buy"]

    rates_yes = []
    rates_no = []
    for col in risk_cols:
        if col in valid.columns:
            rate_yes = valid[valid[col] == 1]["went_on_backorder"].mean() * 100
            rate_no = valid[valid[col] == 0]["went_on_backorder"].mean() * 100
            rates_yes.append(rate_yes)
            rates_no.append(rate_no)
        else:
            rates_yes.append(0)
            rates_no.append(0)

    x = np.arange(len(risk_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    bars_yes = ax.bar(x - width / 2, rates_yes, width, label="Flag = Yes", color=PALETTE["danger"], alpha=0.85)
    bars_no = ax.bar(x + width / 2, rates_no, width, label="Flag = No", color=PALETTE["success"], alpha=0.85)

    ax.set_title("Backorder Rate by Operational Risk Flag Status", fontsize=13, fontweight="bold")
    ax.set_ylabel("Backorder Rate (%)", fontsize=11)
    ax.set_xlabel("Risk Flag", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(risk_labels, fontsize=9)
    ax.legend()

    for bar in bars_yes:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars_no:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return save_figure(fig, "05_risk_flags.png")


# ────────────────────────────────────────────────────
# VISUALIZATION 6: Forecast vs Inventory Pressure
# ────────────────────────────────────────────────────

def plot_demand_inventory_pressure(df: pd.DataFrame) -> str:
    """Demand-inventory pressure analysis."""
    valid = df[df["went_on_backorder"].isin([1, 0])].copy()
    valid["status"] = valid["went_on_backorder"].map({1: "Backorder", 0: "No Backorder"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: In-transit qty vs backorder (rate by bracket)
    bins_it = [-0.1, 0, 10, 50, 200, np.inf]
    labels_it = ["0 (none)", "1–10", "11–50", "51–200", ">200"]
    valid["it_bin"] = pd.cut(valid["in_transit_qty"], bins=bins_it, labels=labels_it)
    it_rates = valid.groupby("it_bin", observed=True)["went_on_backorder"].mean() * 100
    it_counts = valid.groupby("it_bin", observed=True)["went_on_backorder"].count()

    colors_it = [PALETTE["danger"] if r > 2 else PALETTE["warning"] if r > 0.5 else PALETTE["success"]
                 for r in it_rates.values]
    bars = axes[0].bar(it_rates.index.astype(str), it_rates.values, color=colors_it, edgecolor="white")
    axes[0].set_title("Backorder Rate by In-Transit Quantity", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Backorder Rate (%)", fontsize=10)
    axes[0].set_xlabel("In-Transit Quantity (units)", fontsize=10)
    for bar, r, n in zip(bars, it_rates.values, it_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{r:.2f}%\n(n={n/1000:.0f}K)", ha="center", va="bottom", fontsize=8)

    # Right: pieces_past_due presence
    valid["ppd_flag"] = (valid["pieces_past_due"] > 0).astype(int)
    ppd_rates = valid.groupby("ppd_flag")["went_on_backorder"].mean() * 100
    ppd_labels = ["No Past-Due Pieces", "Has Past-Due Pieces"]
    colors_ppd = [PALETTE["success"], PALETTE["danger"]]
    axes[1].bar(ppd_labels, [ppd_rates.get(0, 0), ppd_rates.get(1, 0)],
                color=colors_ppd, width=0.5, edgecolor="white")
    axes[1].set_title("Backorder Rate:\nProducts With vs Without Past-Due Pieces", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Backorder Rate (%)", fontsize=10)
    for i, (lbl, r) in enumerate(zip(ppd_labels, [ppd_rates.get(0, 0), ppd_rates.get(1, 0)])):
        axes[1].text(i, r + 0.1, f"{r:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return save_figure(fig, "06_demand_inventory_pressure.png")


# ────────────────────────────────────────────────────
# VISUALIZATION 7: Correlation / Feature-Backorder Heatmap
# ────────────────────────────────────────────────────

def plot_feature_correlations(df: pd.DataFrame) -> str:
    """Point-biserial correlations of numeric features with the backorder target."""
    valid = df[df["went_on_backorder"].isin([1, 0])].copy()

    # Replace -99 sentinel in perf columns before correlation
    valid["perf_6_month_avg"] = valid["perf_6_month_avg"].replace(-99, np.nan)
    valid["perf_12_month_avg"] = valid["perf_12_month_avg"].replace(-99, np.nan)

    num_cols = [
        "national_inv", "lead_time", "in_transit_qty",
        "forecast_3_month", "forecast_6_month", "forecast_9_month",
        "sales_1_month", "sales_3_month", "sales_6_month", "sales_9_month",
        "min_bank", "pieces_past_due", "perf_6_month_avg", "perf_12_month_avg",
        "local_bo_qty"
    ]
    binary_cols = ["potential_issue", "deck_risk", "oe_constraint", "ppap_risk", "stop_auto_buy", "rev_stop"]

    corr_data = {}
    target = valid["went_on_backorder"]

    for col in num_cols + binary_cols:
        if col in valid.columns:
            series = valid[col].dropna()
            idx = series.index.intersection(target.index)
            r, _ = stats.pointbiserialr(target.loc[idx], series.loc[idx])
            corr_data[col] = r

    corr_series = pd.Series(corr_data).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    colors_bar = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in corr_series.values]
    bars = ax.barh(corr_series.index, corr_series.values, color=colors_bar, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Point-Biserial Correlation with Backorder\n(positive = higher value → more backorders)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Correlation Coefficient", fontsize=10)

    red_patch = mpatches.Patch(color=PALETTE["danger"], alpha=0.85, label="Positive correlation (risk factor)")
    green_patch = mpatches.Patch(color=PALETTE["success"], alpha=0.85, label="Negative correlation (protective)")
    ax.legend(handles=[red_patch, green_patch], fontsize=9)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001 if w >= 0 else w - 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", ha="left" if w >= 0 else "right", fontsize=8)

    plt.tight_layout()
    return save_figure(fig, "07_feature_correlations.png")


# ────────────────────────────────────────────────────
# RUN ALL ANALYSIS
# ────────────────────────────────────────────────────

def run_all_analysis(df_train_clean: pd.DataFrame, df_test_clean: pd.DataFrame) -> dict:
    """
    Run all descriptive and diagnostic analytics.
    df_train_clean / df_test_clean should already have:
    - target encoded as 1/0
    - binary cols encoded as 1/0
    - perf -99 NOT yet replaced (we handle it internally)
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Computing KPIs...")
    kpis = compute_kpis(df_train_clean, df_test_clean)
    save_metrics(kpis, "kpis.json")

    print("Generating Figure 1: Backorder Distribution...")
    plot_backorder_distribution(df_train_clean)

    print("Generating Figure 2: Inventory by Backorder...")
    plot_inventory_by_backorder(df_train_clean)

    print("Generating Figure 3: Lead Time Analysis...")
    plot_leadtime_analysis(df_train_clean)

    print("Generating Figure 4: Supplier Performance...")
    plot_supplier_performance(df_train_clean)

    print("Generating Figure 5: Risk Flags...")
    plot_risk_flags(df_train_clean)

    print("Generating Figure 6: Demand-Inventory Pressure...")
    plot_demand_inventory_pressure(df_train_clean)

    print("Generating Figure 7: Feature Correlations...")
    plot_feature_correlations(df_train_clean)

    print(f"Analysis complete. KPIs saved.")
    return kpis
