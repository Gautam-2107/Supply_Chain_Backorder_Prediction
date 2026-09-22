"""
app.py
======
Supply Chain Risk & Backorder Prediction — Professional Streamlit Application

Five pages:
  1. Executive Overview     — KPIs, backorder summary, key risk drivers
  2. Supply Chain Analysis  — Interactive charts with filters
  3. Individual Prediction  — Form-based prediction per product
  4. Batch Risk Analysis    — CSV upload → predictions → download
  5. Model Performance      — Evaluation metrics, confusion matrix, feature importance

Usage:
    streamlit run app.py
    OR (if streamlit not on PATH):
    python -m streamlit run app.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(_HERE, "src")
MODELS_DIR = os.path.join(_HERE, "models")
FIGURES_DIR = os.path.join(_HERE, "outputs", "figures")
METRICS_DIR = os.path.join(_HERE, "outputs", "metrics")
PREDS_DIR  = os.path.join(_HERE, "outputs", "predictions")
DATA_ROOT = _HERE

TRAIN_PATH = os.path.join(DATA_ROOT, "Training_BOP_sample.csv")
TEST_PATH  = os.path.join(DATA_ROOT, "Testing_BOP.csv")

sys.path.insert(0, SRC_DIR)

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Supply Chain Risk Dashboard",
    page_icon="📦",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────────────
# DESIGN — CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* GLOBAL TYPOGRAPHY FOR DARK THEME */
html, body, [class*="css"], .stMarkdown, .stMarkdown p, .stMarkdown li {
    font-family: 'Inter', sans-serif;
    color: #f1f5f9; /* Primary text: near-white */
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #e2e8f0 !important;
    font-weight: 600;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* KPI CARDS */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #334155;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    min-height: 110px;
}
.kpi-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #cbd5e1; /* Light gray */
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 2.3rem;
    font-weight: 700;
    color: #ffffff; /* Bright white */
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #94a3b8; /* Medium-light gray */
    margin-top: 0.3rem;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }

/* SECTIONS */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff; /* Bright white */
    border-left: 4px solid #3b82f6;
    padding-left: 0.8rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* INSIGHT BOXES */
.insight-box {
    background: linear-gradient(135deg, #1e3a8a, #172554);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    color: #f8fafc;
}
.warning-box {
    background: linear-gradient(135deg, #451a03, #78350f);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    color: #fdf6e3;
}
.danger-box {
    background: linear-gradient(135deg, #7f1d1d, #450a0a);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    color: #fff1f2;
}
.success-box {
    background: linear-gradient(135deg, #14532d, #052e16);
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    color: #f0fdf4;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────

PLOTLY_COLORS = {
    "primary": "#2563EB",
    "danger":  "#DC2626",
    "warning": "#D97706",
    "success": "#16A34A",
    "neutral": "#9CA3AF",
}
PLOTLY_TEMPLATE = "plotly_dark"


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def kpi_card_html(icon, label, value, sub=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """

def fmt_number(n, decimals=0):
    if n is None:
        return "N/A"
    if isinstance(n, float):
        if decimals == 0:
            return f"{int(n):,}"
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"

def assign_risk(prob):
    if prob >= 0.30:
        return "High Risk"
    elif prob >= 0.10:
        return "Medium Risk"
    return "Low Risk"

def get_action(risk_cat, row=None):
    if risk_cat == "High Risk":
        parts = ["🚨 **Expedite replenishment immediately**"]
        if row:
            if row.get("lead_time") and float(row.get("lead_time") or 0) > 14:
                parts.append("📞 Review supplier lead time (>14 weeks)")
            if row.get("pieces_past_due") and float(row.get("pieces_past_due") or 0) > 0:
                parts.append("⚠️ Resolve past-due shipments")
            if row.get("local_bo_qty") and float(row.get("local_bo_qty") or 0) > 0:
                parts.append("📦 Address existing local backorders")
        return "  \n".join(parts)
    elif risk_cat == "Medium Risk":
        parts = ["🔔 **Flag for procurement review**"]
        if row:
            if row.get("national_inv") is not None and float(row.get("national_inv") or 999) < 10:
                parts.append("📈 Increase safety stock")
            if float(row.get("forecast_3_month") or 0) > 0:
                parts.append("🔍 Verify forecast accuracy")
        return "  \n".join(parts)
    return "✅ **Monitor per standard cycle**"

def artifacts_available():
    return (
        os.path.exists(os.path.join(MODELS_DIR, "backorder_model.joblib")) and
        os.path.exists(os.path.join(METRICS_DIR, "kpis.json"))
    )


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_kpis():
    path = os.path.join(METRICS_DIR, "kpis.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_model_metrics():
    files = {
        "lr_test": "lr_test_metrics.json",
        "lr_val":  "lr_val_metrics.json",
        "rf_test": "rf_test_metrics.json",
        "rf_val":  "rf_val_metrics.json",
    }
    result = {}
    for key, fname in files.items():
        path = os.path.join(METRICS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                result[key] = json.load(f)
    return result if result else None

@st.cache_data(show_spinner=False)
def load_predictions():
    path = os.path.join(PREDS_DIR, "risk_prioritization.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    pp_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    rf_path = os.path.join(MODELS_DIR, "backorder_model.joblib")
    lr_path = os.path.join(MODELS_DIR, "lr_model.joblib")
    if not (os.path.exists(pp_path) and os.path.exists(rf_path)):
        return None, None, None, None
    pp = joblib.load(pp_path)
    rf = joblib.load(rf_path)
    lr = joblib.load(lr_path) if os.path.exists(lr_path) else None
    return pp["imputer_values"], pp["feat_cols"], rf, lr

@st.cache_data(show_spinner=False)
def load_sample_train(n=100_000):
    if not os.path.exists(TRAIN_PATH):
        return None
    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    df = df[pd.to_numeric(df["sku"], errors="coerce").notna()].copy()
    df["went_on_backorder"] = df["went_on_backorder"].map({"Yes": 1, "No": 0})
    df = df[df["went_on_backorder"].notna()]
    for col in ["potential_issue","deck_risk","oe_constraint","ppap_risk","stop_auto_buy","rev_stop"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["perf_6_month_avg"] = df["perf_6_month_avg"].replace(-99.0, np.nan)
    df["perf_12_month_avg"] = df["perf_12_month_avg"].replace(-99.0, np.nan)
    df["national_inv"] = df["national_inv"].replace(-99.0, np.nan)
    # Use random sampling to preserve true class balance for accurate EDA rates
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# SINGLE PREDICTION
# ──────────────────────────────────────────────────────────────────────

def predict_single(feature_dict, imputer_values, feat_cols, model):
    df_raw = pd.DataFrame([feature_dict])
    for col in ["potential_issue","deck_risk","oe_constraint","ppap_risk","stop_auto_buy","rev_stop"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0)
    for col in ["perf_6_month_avg","perf_12_month_avg"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].replace(-99.0, np.nan)

    ni = df_raw.get("national_inv", pd.Series([0]))[0]
    fc3 = df_raw.get("forecast_3_month", pd.Series([0]))[0] or 0
    s1  = df_raw.get("sales_1_month",    pd.Series([0]))[0] or 0
    s3  = df_raw.get("sales_3_month",    pd.Series([0]))[0] or 0
    it  = df_raw.get("in_transit_qty",   pd.Series([0]))[0] or 0

    df_raw["lead_time_missing"] = int(df_raw["lead_time"].isna().values[0])
    df_raw["nat_inv_is_negative"] = int((ni or 0) < 0)
    df_raw["inv_demand_ratio"]     = min(max((ni or 0) / (fc3 + 1), -1000), 1000)
    df_raw["stock_coverage_months"] = min(max((ni or 0) / (s1 + 1), -500), 500)
    df_raw["demand_gap"]           = fc3 - (ni or 0)
    df_raw["supply_vs_demand"]     = min(it / (fc3 + 1), 500)
    risk_flag_cols = ["potential_issue","deck_risk","oe_constraint","ppap_risk","rev_stop"]
    df_raw["total_risk_flags"] = sum(df_raw.get(c, pd.Series([0]))[0] or 0 for c in risk_flag_cols)
    df_raw["sales_acceleration"] = min(s3 / (3 * s1 + 1), 100)
    p6 = df_raw.get("perf_6_month_avg",  pd.Series([np.nan]))[0]
    p12 = df_raw.get("perf_12_month_avg", pd.Series([np.nan]))[0]
    vals_perf = [v for v in [p6, p12] if v is not None and not np.isnan(float(v))]
    df_raw["avg_supplier_perf"] = float(np.mean(vals_perf)) if vals_perf else np.nan
    ppd = df_raw.get("pieces_past_due", pd.Series([0]))[0] or 0
    lbo = df_raw.get("local_bo_qty",    pd.Series([0]))[0] or 0
    df_raw["past_due_flag"] = int(ppd > 0)
    df_raw["local_bo_flag"] = int(lbo > 0)

    for col, val in imputer_values.items():
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(val)
    for col in feat_cols:
        if col not in df_raw.columns:
            df_raw[col] = 0

    X = df_raw[feat_cols].copy()
    return float(model.predict_proba(X)[0, 1])


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align: center;
    padding: 0.8rem 0 1.5rem 0;
">
    <div style="
        font-size: 2.2rem;
        font-weight: 800;
        color: #f1f5f9;
    ">
        🚚 AI-Powered Supply Chain Risk & Backorder Prediction
    </div>
    <div style="
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    ">
        Supply Chain & Logistics • IBM SkillsBuild Capstone
    </div>
</div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 📌 Dashboard Navigation")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Executive Overview",
         "🔍 Supply Chain Analysis",
         "🎯 Individual Prediction",
         "📁 Batch Risk Analysis",
         "📈 Model Performance"],
        label_visibility="hidden",
    )
    st.markdown("---")
    if artifacts_available():
        st.success("✅ Model artifacts loaded")
    else:
        st.error("⚠️ Run `python train.py` first")
    st.markdown("---")
    st.markdown("""<div style='font-size:0.8rem; color:#cbd5e1;'>
    IBM SkillsBuild Capstone<br>Supply Chain & Logistics<br>AI-Powered Backorder Prediction
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# PAGE 1: EXECUTIVE OVERVIEW
# ──────────────────────────────────────────────────────────────────────

if page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    st.markdown("*Supply Chain Backorder Risk Dashboard — AI-Powered Insights*")

    if not artifacts_available():
        st.error("Model artifacts not found. Please run `python train.py` first.")
        st.code("cd Supply_Chain_Backorder_Project\npython train.py")
        st.stop()

    kpis = load_kpis()
    model_metrics = load_model_metrics()
    preds = load_predictions()

    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card_html("📦", "Total Products (Train)",
            fmt_number(kpis["total_products_train"]),
            f"Test: {fmt_number(kpis['total_products_test'])}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card_html("⚠️", "Backorder Rate",
            f"{kpis['backorder_rate_train_pct']:.2f}%",
            f"Test: {kpis['backorder_rate_test_pct']:.2f}%"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card_html("🔴", "Backorders (Train)",
            fmt_number(kpis["total_backorders_train"]),
            f"Test: {fmt_number(kpis['total_backorders_test'])}"), unsafe_allow_html=True)
    with c4:
        neg_inv = kpis.get("products_with_negative_inventory", 0)
        st.markdown(kpi_card_html("📉", "Negative Inventory",
            fmt_number(neg_inv),
            f"{kpis.get('pct_negative_inventory', 0):.2f}% of products"), unsafe_allow_html=True)
    with c5:
        if preds is not None:
            high_risk = len(preds[preds["risk_category"] == "High Risk"])
        elif model_metrics and "rf_test" in model_metrics:
            m = model_metrics["rf_test"]["confusion_matrix"]
            high_risk = m["tp"] + m["fp"]
        else:
            high_risk = "—"
        st.markdown(kpi_card_html("🚨", "High-Risk Products",
            fmt_number(high_risk) if isinstance(high_risk, int) else high_risk,
            "Model-predicted on test set"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown('<div class="section-header">Backorder Rate Overview</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["No Backorder", "Backorder"],
            values=[kpis["total_products_train"] - kpis["total_backorders_train"],
                    kpis["total_backorders_train"]],
            hole=0.6,
            marker=dict(colors=[PLOTLY_COLORS["primary"], PLOTLY_COLORS["danger"]],
                        line=dict(color="white", width=3)),
            textinfo="label+percent",
        ))
        bo_rate = kpis["backorder_rate_train_pct"]
        fig.add_annotation(text=f"<b>{bo_rate:.2f}%</b><br>Backorder Rate",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18))
        fig.update_layout(title="Training Dataset — Backorder Distribution",
                          height=350, template=PLOTLY_TEMPLATE,
                          margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Supply Chain KPIs</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-box">
        📊 <b>Dataset Size</b><br>
        Training: {fmt_number(kpis['total_products_train'])} products<br>
        Test (holdout): {fmt_number(kpis['total_products_test'])} products
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="warning-box">
        ⏱️ <b>Lead Time</b><br>
        Median: {kpis.get('median_lead_time', 'N/A')} weeks<br>
        Missing: {kpis.get('pct_missing_lead_time', 0):.1f}% of products
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="danger-box">
        🏭 <b>Operational Risk</b><br>
        Deck Risk Flag: {kpis.get('deck_risk_pct', 0):.1f}% of products<br>
        PPAP Risk Flag: {kpis.get('ppap_risk_pct', 0):.1f}% of products
        </div>""", unsafe_allow_html=True)
        median_inv = kpis.get('median_national_inv', 0)
        st.markdown(f"""<div class="success-box">
        📦 <b>Inventory Summary</b><br>
        Median inventory: {fmt_number(median_inv)} units<br>
        Zero in-transit: {kpis.get('pct_zero_transit', 0):.1f}% of products
        </div>""", unsafe_allow_html=True)

    if model_metrics:
        st.markdown('<div class="section-header">Model Performance Snapshot (Test Set)</div>', unsafe_allow_html=True)
        st.caption("Random Forest evaluated on external holdout — not used for training or tuning")
        rf_m = model_metrics.get("rf_test", {})
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        for col, (label, val, sub) in zip([m1,m2,m3,m4,m5,m6], [
            ("ROC-AUC", rf_m.get("roc_auc", 0), "Discriminative power"),
            ("PR-AUC",  rf_m.get("pr_auc", 0),  "Precision-recall balance"),
            ("Recall",  rf_m.get("recall", 0),   "% backorders caught"),
            ("Precision",rf_m.get("precision",0),"% alerts that are real"),
            ("F1-Score", rf_m.get("f1", 0),      "Balanced metric"),
            ("Accuracy", rf_m.get("accuracy",0), "Overall classification"),
        ]):
            with col:
                color = "#16a34a" if val >= 0.7 else "#d97706" if val >= 0.5 else "#dc2626"
                st.markdown(f"""
                <div style='background:{color}15; border:1px solid {color}44;
                            border-radius:10px; padding:0.8rem; text-align:center;'>
                    <div style='font-size:0.75rem; color:#cbd5e1; text-transform:uppercase; font-weight:600;'>{label}</div>
                    <div style='font-size:1.7rem; font-weight:700; color:{color};'>{val:.3f}</div>
                    <div style='font-size:0.7rem; color:#e2e8f0;'>{sub}</div>
                </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="danger-box">
        <b>🔴 High Risk Signals</b><br>
        • Negative national inventory (15.1% backorder rate)<br>
        • Past-due pieces present<br>
        • Multiple operational risk flags active<br>
        • High local backorder quantity
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="insight-box">
        <b>💡 Key Findings</b><br>
        • Negative inventory products are <b>24× more likely</b> to backorder (than positive inventory)<br>
        • local_bo_qty > 0 → 5.99% backorder rate (vs 0.59% baseline)<br>
        • 79.7% of products have zero in-transit quantity<br>
        • Class imbalance: only 0.67% of products backorder
        </div>""", unsafe_allow_html=True)

    if preds is not None:
        st.markdown('<div class="section-header">Risk Category Distribution (Test Set)</div>', unsafe_allow_html=True)
        risk_counts = preds["risk_category"].value_counts()
        cats = ["High Risk", "Medium Risk", "Low Risk"]
        vals_risk = [risk_counts.get(c, 0) for c in cats]
        fig2 = go.Figure(go.Bar(
            x=cats, y=vals_risk,
            marker_color=[PLOTLY_COLORS["danger"], PLOTLY_COLORS["warning"], PLOTLY_COLORS["success"]],
            text=[f"{v:,}" for v in vals_risk], textposition="outside",
        ))
        fig2.update_layout(title="Products by Risk Category — External Test Set",
                           xaxis_title="Risk Category", yaxis_title="Products",
                           template=PLOTLY_TEMPLATE, height=300,
                           margin=dict(t=50, b=30, l=30, r=30))
        st.plotly_chart(fig2, width="stretch")


# ──────────────────────────────────────────────────────────────────────
# PAGE 2: SUPPLY CHAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────

elif page == "🔍 Supply Chain Analysis":
    st.markdown("# 🔍 Supply Chain Analysis")
    st.markdown("*Interactive charts — 100K representative random sample (preserves true backorder rates)*")

    df = load_sample_train(n=100_000)
    if df is None:
        st.error(f"Training data not found at: {TRAIN_PATH}")
        st.stop()

    chart_type = st.selectbox("Select Analysis View", [
        "📦 Inventory Distribution by Backorder Status",
        "⏱️ Lead Time vs Backorder Rate",
        "📈 Demand & In-Transit Patterns",
        "🏭 Supplier Performance vs Backorder",
        "🚩 Operational Risk Flags",
        "📊 Feature Correlation with Backorder",
    ])

    if chart_type == "📦 Inventory Distribution by Backorder Status":
        st.markdown('<div class="section-header">National Inventory vs Backorder Status</div>', unsafe_allow_html=True)
        bins = [-float("inf"), 0, 5, 20, 50, 200, float("inf")]
        labels = ["<0 (negative)", "0–5", "6–20", "21–50", "51–200", ">200"]
        df["inv_bin"] = pd.cut(df["national_inv"], bins=bins, labels=labels)
        rates = df.groupby("inv_bin", observed=True)["went_on_backorder"].mean() * 100
        counts = df.groupby("inv_bin", observed=True)["went_on_backorder"].count()
        colors_bar = [PLOTLY_COLORS["danger"] if r > 5 else PLOTLY_COLORS["warning"] if r > 1 else PLOTLY_COLORS["success"]
                      for r in rates.values]
        fig = go.Figure(go.Bar(
            x=rates.index.astype(str).tolist(), y=rates.values.tolist(),
            marker_color=colors_bar,
            text=[f"{r:.1f}%\n(n={n:,})" for r, n in zip(rates.values, counts.values)],
            textposition="outside",
        ))
        fig.update_layout(title="Backorder Rate by Inventory Level",
                          xaxis_title="National Inventory (units)", yaxis_title="Backorder Rate (%)",
                          template=PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig, width="stretch")
        st.markdown("""<div class="danger-box">
        <b>Key Insight:</b> Products with <b>negative inventory</b> have dramatically higher backorder rates.
        Negative inventory means committed orders exceed available stock — the strongest inventory-level risk signal.
        </div>""", unsafe_allow_html=True)

    elif chart_type == "⏱️ Lead Time vs Backorder Rate":
        st.markdown('<div class="section-header">Lead Time Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            valid_lt = df[df["lead_time"].notna()].copy()
            valid_lt["lt_bin"] = pd.cut(valid_lt["lead_time"], bins=[0,4,8,12,20,53],
                                         labels=["≤4 wks","5–8 wks","9–12 wks","13–20 wks",">20 wks"],
                                         include_lowest=True)
            lt_rates = valid_lt.groupby("lt_bin", observed=True)["went_on_backorder"].mean() * 100
            lt_counts = valid_lt.groupby("lt_bin", observed=True)["went_on_backorder"].count()
            colors_lt = [PLOTLY_COLORS["danger"] if r > 2 else PLOTLY_COLORS["warning"] if r > 0.5 else PLOTLY_COLORS["success"]
                         for r in lt_rates.values]
            fig1 = go.Figure(go.Bar(x=lt_rates.index.astype(str).tolist(), y=lt_rates.values.tolist(),
                                    marker_color=colors_lt,
                                    text=[f"{r:.2f}%\n(n={n:,})" for r, n in zip(lt_rates.values, lt_counts.values)],
                                    textposition="outside"))
            fig1.update_layout(title="Backorder Rate by Lead Time", xaxis_title="Lead Time",
                               yaxis_title="Backorder Rate (%)", template=PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig1, width="stretch")
        with col2:
            mr = df[df["lead_time"].isna()]["went_on_backorder"].mean() * 100
            pr = df[df["lead_time"].notna()]["went_on_backorder"].mean() * 100
            mn = df["lead_time"].isna().sum()
            pn = df["lead_time"].notna().sum()
            fig2 = go.Figure(go.Bar(x=["Lead Time Present","Lead Time Missing"], y=[pr, mr],
                                    marker_color=[PLOTLY_COLORS["primary"], PLOTLY_COLORS["neutral"]],
                                    text=[f"{r:.2f}%\n(n={n:,})" for r, n in [(pr,pn),(mr,mn)]],
                                    textposition="outside"))
            fig2.update_layout(title="Backorder: Missing vs Present Lead Time",
                               yaxis_title="Backorder Rate (%)", template=PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig2, width="stretch")

    elif chart_type == "📈 Demand & In-Transit Patterns":
        st.markdown('<div class="section-header">Demand & In-Transit Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            df["it_bin"] = pd.cut(df["in_transit_qty"], bins=[-0.1,0,10,50,200,float("inf")],
                                   labels=["0","1–10","11–50","51–200",">200"])
            it_rates = df.groupby("it_bin", observed=True)["went_on_backorder"].mean() * 100
            it_counts = df.groupby("it_bin", observed=True)["went_on_backorder"].count()
            colors_it = [PLOTLY_COLORS["danger"] if r > 2 else PLOTLY_COLORS["warning"] if r > 0.5 else PLOTLY_COLORS["success"]
                         for r in it_rates.values]
            fig3 = go.Figure(go.Bar(x=it_rates.index.astype(str).tolist(), y=it_rates.values.tolist(),
                                    marker_color=colors_it,
                                    text=[f"{r:.2f}%\nn={n:,}" for r, n in zip(it_rates.values, it_counts.values)],
                                    textposition="outside"))
            fig3.update_layout(title="Backorder Rate by In-Transit Qty",
                               xaxis_title="In-Transit Quantity", yaxis_title="Backorder Rate (%)",
                               template=PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig3, width="stretch")
        with col2:
            df["ppd_flag"] = (df["pieces_past_due"] > 0).astype(int)
            ppd_rates = df.groupby("ppd_flag")["went_on_backorder"].mean() * 100
            ppd_ns = df.groupby("ppd_flag")["went_on_backorder"].count()
            fig4 = go.Figure(go.Bar(x=["No Past-Due Pieces","Has Past-Due Pieces"],
                                    y=[ppd_rates.get(0,0), ppd_rates.get(1,0)],
                                    marker_color=[PLOTLY_COLORS["success"], PLOTLY_COLORS["danger"]],
                                    text=[f"{r:.2f}%\nn={n:,}" for r, n in [(ppd_rates.get(0,0), ppd_ns.get(0,0)),
                                                                              (ppd_rates.get(1,0), ppd_ns.get(1,0))]],
                                    textposition="outside"))
            fig4.update_layout(title="Backorder Rate: Past-Due Pieces",
                               yaxis_title="Backorder Rate (%)", template=PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig4, width="stretch")

    elif chart_type == "🏭 Supplier Performance vs Backorder":
        st.markdown('<div class="section-header">Supplier Performance Analysis</div>', unsafe_allow_html=True)
        valid_perf = df[df["perf_6_month_avg"].notna()].copy()
        col1, col2 = st.columns(2)
        with col1:
            bo_perf = valid_perf[valid_perf["went_on_backorder"] == 1]["perf_6_month_avg"]
            no_bo_perf = valid_perf[valid_perf["went_on_backorder"] == 0]["perf_6_month_avg"]
            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(x=no_bo_perf.tolist(), name="No Backorder", nbinsx=40,
                                        histnorm="probability density", opacity=0.7,
                                        marker_color=PLOTLY_COLORS["primary"]))
            fig5.add_trace(go.Histogram(x=bo_perf.tolist(), name="Backorder", nbinsx=40,
                                        histnorm="probability density", opacity=0.7,
                                        marker_color=PLOTLY_COLORS["danger"]))
            fig5.update_layout(title="Supplier Performance Distribution (6-Month Avg)",
                               xaxis_title="Score (0–1)", yaxis_title="Density",
                               barmode="overlay", template=PLOTLY_TEMPLATE, height=380,
                               legend=dict(orientation="h"))
            st.plotly_chart(fig5, width="stretch")
        with col2:
            vp2 = valid_perf[valid_perf["perf_6_month_avg"].between(0, 1)].copy()
            vp2["perf_bin"] = pd.qcut(vp2["perf_6_month_avg"], q=5, duplicates="drop",
                                       labels=["Q1 (Lowest)","Q2","Q3","Q4","Q5 (Highest)"])
            perf_rates2 = vp2.groupby("perf_bin", observed=True)["went_on_backorder"].mean() * 100
            colors_p = [PLOTLY_COLORS["danger"] if r > 2 else PLOTLY_COLORS["warning"] if r > 0.5 else PLOTLY_COLORS["success"]
                        for r in perf_rates2.values]
            fig6 = go.Figure(go.Bar(x=perf_rates2.index.astype(str).tolist(), y=perf_rates2.values.tolist(),
                                    marker_color=colors_p,
                                    text=[f"{r:.2f}%" for r in perf_rates2.values],
                                    textposition="outside"))
            fig6.update_layout(title="Backorder Rate by Supplier Performance Quintile",
                               xaxis_title="Performance Quintile", yaxis_title="Backorder Rate (%)",
                               template=PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig6, width="stretch")

    elif chart_type == "🚩 Operational Risk Flags":
        st.markdown('<div class="section-header">Operational Risk Flag Analysis</div>', unsafe_allow_html=True)
        risk_cols = ["potential_issue","deck_risk","oe_constraint","ppap_risk","rev_stop","stop_auto_buy"]
        risk_labels = ["Potential Issue","Deck Risk","OE Constraint","PPAP Risk","Rev Stop","Stop Auto-Buy"]
        rates_yes, rates_no = [], []
        for col in risk_cols:
            if col in df.columns:
                rates_yes.append(df[df[col] == 1]["went_on_backorder"].mean() * 100)
                rates_no.append(df[df[col] == 0]["went_on_backorder"].mean() * 100)
            else:
                rates_yes.append(0); rates_no.append(0)
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(name="Flag = Yes", x=risk_labels, y=rates_yes,
                              marker_color=PLOTLY_COLORS["danger"], opacity=0.85,
                              text=[f"{r:.1f}%" for r in rates_yes], textposition="outside"))
        fig7.add_trace(go.Bar(name="Flag = No", x=risk_labels, y=rates_no,
                              marker_color=PLOTLY_COLORS["success"], opacity=0.85,
                              text=[f"{r:.1f}%" for r in rates_no], textposition="outside"))
        fig7.update_layout(title="Backorder Rate by Operational Risk Flag",
                           xaxis_title="Risk Flag", yaxis_title="Backorder Rate (%)",
                           barmode="group", template=PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig7, width="stretch")

    elif chart_type == "📊 Feature Correlation with Backorder":
        st.markdown('<div class="section-header">Point-Biserial Correlation with Backorder Target</div>', unsafe_allow_html=True)
        from scipy import stats as sp_stats
        num_cols = ["national_inv","lead_time","in_transit_qty","forecast_3_month",
                    "sales_1_month","sales_3_month","min_bank","pieces_past_due",
                    "perf_6_month_avg","perf_12_month_avg","local_bo_qty"]
        bin_cols = ["potential_issue","deck_risk","oe_constraint","ppap_risk","stop_auto_buy","rev_stop"]
        corr_data = {}
        target_s = df["went_on_backorder"]
        for col in num_cols + bin_cols:
            if col in df.columns:
                series = df[col].dropna()
                idx = series.index.intersection(target_s.index)
                if len(idx) > 10:
                    r, _ = sp_stats.pointbiserialr(target_s.loc[idx], series.loc[idx])
                    corr_data[col] = r
        corr_series = pd.Series(corr_data).sort_values()
        colors_c = [PLOTLY_COLORS["danger"] if v > 0 else PLOTLY_COLORS["success"] for v in corr_series.values]
        fig8 = go.Figure(go.Bar(
            x=corr_series.values.tolist(), y=corr_series.index.tolist(),
            orientation="h", marker_color=colors_c,
            text=[f"{v:.3f}" for v in corr_series.values], textposition="outside",
        ))
        fig8.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig8.update_layout(title="Point-Biserial Correlation with Backorder",
                           xaxis_title="Correlation Coefficient", yaxis_title="",
                           template=PLOTLY_TEMPLATE, height=520, margin=dict(l=160))
        st.plotly_chart(fig8, width="stretch")
        st.markdown("""<div class="insight-box">
        <b>Note:</b> Correlation indicates statistical association only — not causality.
        Red bars = positive correlation (higher value → more backorders). Green = protective factor.
        </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# PAGE 3: INDIVIDUAL PREDICTION
# ──────────────────────────────────────────────────────────────────────

elif page == "🎯 Individual Prediction":
    st.markdown("# 🎯 Individual Risk Prediction")
    st.markdown("*Enter product parameters to get a backorder risk assessment*")

    imputer_values, feat_cols, rf_model, lr_model = load_model_artifacts()
    if rf_model is None:
        st.error("Model artifacts not found. Run `python train.py` first.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">Product Supply Chain Parameters</div>', unsafe_allow_html=True)
        col_sup, col_dem, col_risk = st.columns(3)
        with col_sup:
            st.markdown("##### 📦 Inventory & Supply")
            sku_input = st.text_input("SKU (reference only)", value="PROD-001")
            national_inv = st.number_input("National Inventory", value=50)
            in_transit_qty = st.number_input("In-Transit Qty", value=0)
            lead_time_val = st.number_input("Lead Time (weeks)", value=8.0)
            min_bank = st.number_input("Min Bank", value=0)
            local_bo_qty = st.number_input("Local Backorder Qty", value=0)
            pieces_past_due = st.number_input("Pieces Past Due", value=0)
        with col_dem:
            st.markdown("##### 📈 Demand & Forecast")
            forecast_3_month = st.number_input("Forecast 3 Month", value=0)
            forecast_6_month = st.number_input("Forecast 6 Month", value=0)
            forecast_9_month = st.number_input("Forecast 9 Month", value=0)
            sales_1_month = st.number_input("Sales 1 Month", value=0)
            sales_3_month = st.number_input("Sales 3 Month", value=0)
            sales_6_month = st.number_input("Sales 6 Month", value=0)
            sales_9_month = st.number_input("Sales 9 Month", value=0)
        with col_risk:
            st.markdown("##### 🏭 Supplier & Operations")
            perf_6_month_avg = st.slider("Supplier Perf 6-Month", 0.0, 1.0, 0.82)
            perf_12_month_avg = st.slider("Supplier Perf 12-Month", 0.0, 1.0, 0.81)
            st.markdown("**Operational Risk Flags**")
            potential_issue = st.selectbox("Potential Issue", ["No", "Yes"])
            deck_risk = st.selectbox("Deck Risk", ["No", "Yes"])
            oe_constraint = st.selectbox("OE Constraint", ["No", "Yes"])
            ppap_risk = st.selectbox("PPAP Risk", ["No", "Yes"])
            stop_auto_buy = st.selectbox("Stop Auto-Buy", ["Yes", "No"])
            rev_stop = st.selectbox("Rev Stop", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict Backorder Risk", width="stretch")

    if submitted:
        feature_dict = {
            "national_inv": national_inv, "lead_time": lead_time_val,
            "in_transit_qty": in_transit_qty, "forecast_3_month": forecast_3_month,
            "forecast_6_month": forecast_6_month, "forecast_9_month": forecast_9_month,
            "sales_1_month": sales_1_month, "sales_3_month": sales_3_month,
            "sales_6_month": sales_6_month, "sales_9_month": sales_9_month,
            "min_bank": min_bank, "pieces_past_due": pieces_past_due,
            "perf_6_month_avg": perf_6_month_avg, "perf_12_month_avg": perf_12_month_avg,
            "local_bo_qty": local_bo_qty, "potential_issue": potential_issue,
            "deck_risk": deck_risk, "oe_constraint": oe_constraint,
            "ppap_risk": ppap_risk, "stop_auto_buy": stop_auto_buy, "rev_stop": rev_stop,
        }
        try:
            prob = predict_single(feature_dict, imputer_values, feat_cols, rf_model)
            risk_cat = assign_risk(prob)
            action = get_action(risk_cat, feature_dict)
            pct = prob * 100
            risk_color = {"High Risk": "#dc2626", "Medium Risk": "#d97706", "Low Risk": "#16a34a"}[risk_cat]
            risk_emoji = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}[risk_cat]

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""<div style='background:{risk_color}15; border:2px solid {risk_color};
                            border-radius:14px; padding:1.5rem; text-align:center;'>
                    <div style='font-size:0.9rem; color:#cbd5e1; text-transform:uppercase; font-weight:600;'>Risk Category</div>
                    <div style='font-size:3rem; margin:0.5rem 0;'>{risk_emoji}</div>
                    <div style='font-size:1.6rem; font-weight:700; color:{risk_color};'>{risk_cat}</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div style='background:linear-gradient(135deg,#1e293b,#0f172a);
                            border-radius:14px; padding:1.5rem; text-align:center;'>
                    <div style='font-size:0.9rem; color:#e2e8f0; text-transform:uppercase; font-weight:600;'>Backorder Probability</div>
                    <div style='font-size:3rem; font-weight:700; color:{risk_color}; margin:0.5rem 0;'>{pct:.1f}%</div>
                    <div style='font-size:0.8rem; color:#cbd5e1;'>SKU: {sku_input}</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=round(pct, 1),
                    number={"suffix": "%", "font": {"size": 24}},
                    title={"text": "Risk %", "font": {"size": 14}},
                    gauge={"axis": {"range": [0,100]}, "bar": {"color": risk_color},
                           "steps": [{"range":[0,10],"color":"#dcfce7"},
                                     {"range":[10,30],"color":"#fef3c7"},
                                     {"range":[30,100],"color":"#fee2e2"}]}
                ))
                gauge.update_layout(height=200, margin=dict(t=30,b=0,l=20,r=20))
                st.plotly_chart(gauge, width="stretch")

            st.markdown(f"**📋 Recommended Action:**\n\n{action}")

            st.markdown("**📊 Key Risk Factors for This Product:**")
            factors = []
            if national_inv < 0:
                factors.append(("🔴", "Negative inventory", "Overcommitted — highest backorder signal"))
            elif national_inv == 0:
                factors.append(("🟡", "Zero inventory", "No stock on hand"))
            if pieces_past_due > 0:
                factors.append(("🔴", "Past-due pieces present", f"{pieces_past_due:,} units delayed"))
            if local_bo_qty > 0:
                factors.append(("🟡", "Local backorder quantity", f"{local_bo_qty:,} units"))
            if lead_time_val > 14:
                factors.append(("🟡", "High lead time", f"{lead_time_val:.0f} weeks"))
            if forecast_3_month > national_inv > 0:
                factors.append(("🟡", "Demand exceeds inventory", f"Forecast {forecast_3_month:,} > Inv {national_inv:,}"))
            if perf_6_month_avg < 0.7:
                factors.append(("🟡", "Below-average supplier performance", f"Score: {perf_6_month_avg:.2f}"))
            active_flags = sum([potential_issue=="Yes", deck_risk=="Yes", ppap_risk=="Yes", rev_stop=="Yes"])
            if active_flags >= 2:
                factors.append(("🔴", f"{active_flags} risk flags active", "High operational risk profile"))
            if not factors:
                st.markdown("✅ No critical risk factors detected.")
            else:
                for icon, title, detail in factors:
                    st.markdown(f"{icon} **{title}** — {detail}")

            st.markdown("""<div class="insight-box" style='margin-top:1rem;'>
            <b>Model Context (FACT / PREDICTION / RECOMMENDATION):</b><br>
            🤖 <b>PREDICTION</b>: Random Forest probability score based on historical patterns.<br>
            💡 <b>RECOMMENDATION</b>: Derived from model output — not a guarantee of backorder occurrence.
            Use alongside other supply chain intelligence.
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ──────────────────────────────────────────────────────────────────────
# PAGE 4: BATCH RISK ANALYSIS
# ──────────────────────────────────────────────────────────────────────

elif page == "📁 Batch Risk Analysis":
    st.markdown("# 📁 Batch Risk Analysis")
    st.markdown("*Upload a CSV to score multiple products for backorder risk*")

    imputer_values, feat_cols, rf_model, lr_model = load_model_artifacts()
    if rf_model is None:
        st.error("Model artifacts not found. Run `python train.py` first.")
        st.stop()

    with st.expander("📄 Expected CSV Format"):
        st.markdown("""
        Include any subset of these columns (missing ones filled with training medians):
        `sku`, `national_inv`, `lead_time`, `in_transit_qty`, `forecast_3_month`, `forecast_6_month`,
        `forecast_9_month`, `sales_1_month`, `sales_3_month`, `sales_6_month`, `sales_9_month`,
        `min_bank`, `pieces_past_due`, `perf_6_month_avg`, `perf_12_month_avg`, `local_bo_qty`,
        `potential_issue` (Yes/No), `deck_risk` (Yes/No), `oe_constraint` (Yes/No),
        `ppap_risk` (Yes/No), `stop_auto_buy` (Yes/No), `rev_stop` (Yes/No)
        """)

    preds = load_predictions()
    if preds is not None:
        st.markdown('<div class="section-header">Pre-Generated: Test Set Risk Assessment</div>', unsafe_allow_html=True)
        st.caption(f"{len(preds):,} products from the external holdout test set")

        risk_counts = preds["risk_category"].value_counts()
        s1, s2, s3 = st.columns(3)
        for col, cat, color in zip([s1,s2,s3],
                                    ["High Risk","Medium Risk","Low Risk"],
                                    [PLOTLY_COLORS["danger"],PLOTLY_COLORS["warning"],PLOTLY_COLORS["success"]]):
            with col:
                n = risk_counts.get(cat, 0)
                pct = 100 * n / len(preds)
                st.markdown(f"""<div style='background:{color}15; border:1px solid {color}44;
                            border-radius:10px; padding:0.8rem; text-align:center;'>
                    <div style='font-size:0.9rem; color:#cbd5e1; font-weight:600;'>{cat}</div>
                    <div style='font-size:1.8rem; font-weight:700; color:{color};'>{n:,}</div>
                    <div style='font-size:0.85rem; color:#e2e8f0;'>{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        filter_risk = st.multiselect("Filter by Risk Category",
                                      ["High Risk","Medium Risk","Low Risk"],
                                      default=["High Risk","Medium Risk"])
        filtered = preds[preds["risk_category"].isin(filter_risk)] if filter_risk else preds
        st.markdown(f"Showing {len(filtered):,} products")

        display_cols = [c for c in ["sku","risk_probability","risk_category","actual_backorder",
                                     "national_inv","lead_time","forecast_3_month",
                                     "pieces_past_due","local_bo_qty","recommended_action"]
                        if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(500).style.format(
            {c: "{:,.0f}" for c in ["national_inv","forecast_3_month","pieces_past_due","local_bo_qty"]
             if c in filtered.columns} | ({"risk_probability": "{:.1%}"} if "risk_probability" in filtered.columns else {}),
            na_rep="N/A"), width="stretch", height=400)

        st.download_button("⬇️ Download Risk Assessment (CSV)",
                           data=filtered.to_csv(index=False),
                           file_name="backorder_risk_assessment.csv", mime="text/csv",
                           width="stretch")

    st.markdown("---")
    st.markdown('<div class="section-header">Upload New Products for Scoring</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_up):,} rows, {len(df_up.columns)} columns")

            for col in ["potential_issue","deck_risk","oe_constraint","ppap_risk","stop_auto_buy","rev_stop"]:
                if col in df_up.columns:
                    df_up[col] = df_up[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
            for col in ["perf_6_month_avg","perf_12_month_avg"]:
                if col in df_up.columns:
                    df_up[col] = df_up[col].replace(-99.0, np.nan)
            if "national_inv" in df_up.columns:
                df_up["national_inv"] = df_up["national_inv"].replace(-99.0, np.nan)

            def safe_col(df, col, default=0):
                return df[col].fillna(default) if col in df.columns else pd.Series([default]*len(df), index=df.index)

            df_up["lead_time_missing"] = safe_col(df_up, "lead_time", np.nan).isna().astype(int)
            df_up["nat_inv_is_negative"] = (safe_col(df_up, "national_inv") < 0).astype(int)
            df_up["inv_demand_ratio"] = (safe_col(df_up, "national_inv") / (safe_col(df_up, "forecast_3_month") + 1)).clip(-1000, 1000)
            df_up["stock_coverage_months"] = (safe_col(df_up, "national_inv") / (safe_col(df_up, "sales_1_month") + 1)).clip(-500, 500)
            df_up["demand_gap"] = safe_col(df_up, "forecast_3_month") - safe_col(df_up, "national_inv")
            df_up["supply_vs_demand"] = (safe_col(df_up, "in_transit_qty") / (safe_col(df_up, "forecast_3_month") + 1)).clip(0, 500)
            df_up["total_risk_flags"] = sum(safe_col(df_up, c) for c in ["potential_issue","deck_risk","oe_constraint","ppap_risk","rev_stop"])
            df_up["sales_acceleration"] = (safe_col(df_up, "sales_3_month") / (3 * safe_col(df_up, "sales_1_month") + 1)).clip(0, 100)
            perf_cols_avail = [c for c in ["perf_6_month_avg","perf_12_month_avg"] if c in df_up.columns]
            df_up["avg_supplier_perf"] = df_up[perf_cols_avail].mean(axis=1) if perf_cols_avail else 0.8
            df_up["past_due_flag"] = (safe_col(df_up, "pieces_past_due") > 0).astype(int)
            df_up["local_bo_flag"] = (safe_col(df_up, "local_bo_qty") > 0).astype(int)

            for col, val in imputer_values.items():
                if col in df_up.columns:
                    df_up[col] = df_up[col].fillna(val)
                else:
                    df_up[col] = val
            for col in feat_cols:
                if col not in df_up.columns:
                    df_up[col] = 0

            X_up = df_up[feat_cols].copy()
            probs = rf_model.predict_proba(X_up)[:, 1]
            df_up["risk_probability"] = probs
            df_up["risk_category"] = pd.Series(probs).apply(assign_risk)
            df_up["recommended_action"] = df_up["risk_category"].apply(get_action)

            out_cols = (["sku"] if "sku" in df_up.columns else []) + \
                       ["risk_probability","risk_category","recommended_action"] + \
                       [c for c in ["national_inv","lead_time","forecast_3_month","pieces_past_due"] if c in df_up.columns]
            output_df = df_up[out_cols].sort_values("risk_probability", ascending=False)

            rc = output_df["risk_category"].value_counts()
            st.info(f"High Risk: {rc.get('High Risk',0):,} | Medium: {rc.get('Medium Risk',0):,} | Low: {rc.get('Low Risk',0):,}")
            st.dataframe(output_df.head(500).style.format({"risk_probability": "{:.1%}"}, na_rep="N/A"),
                         width="stretch", height=400)
            st.download_button("⬇️ Download Results",
                               data=output_df.to_csv(index=False),
                               file_name="uploaded_risk_scores.csv", mime="text/csv",
                               width="stretch")
        except Exception as e:
            st.error(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────
# PAGE 5: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance")
    st.markdown("*All metrics on the external holdout test set — never used for training or tuning*")

    model_metrics = load_model_metrics()
    if not model_metrics:
        st.error("Metrics not found. Run `python train.py` first.")
        st.stop()

    rf_test = model_metrics.get("rf_test", {})
    lr_test = model_metrics.get("lr_test", {})

    st.markdown("""<div class="insight-box">
    <b>Methodology:</b> Training_BOP.csv split 80/20 (stratified) for train/validation.
    Testing_BOP.csv held out — used <em>only</em> for final evaluation.
    Class imbalance (0.67% positive rate) handled via <code>class_weight='balanced'</code>.
    Decision threshold: 0.30 (adjusted from 0.5 to improve recall for high-cost false negatives).
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Comparison — Test Set Metrics</div>', unsafe_allow_html=True)
    metric_names = ["accuracy","precision","recall","f1","roc_auc","pr_auc"]
    metric_labels = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC"]
    comp_data = []
    for name, m in [("Logistic Regression", lr_test), ("Random Forest", rf_test)]:
        row = {"Model": name}
        for mn, ml in zip(metric_names, metric_labels):
            row[ml] = f"{m.get(mn, 0):.4f}"
        comp_data.append(row)
    st.dataframe(pd.DataFrame(comp_data).set_index("Model"), width="stretch")

    fig_comp = go.Figure()
    for (name, m), color in zip([("Logistic Regression", lr_test), ("Random Forest", rf_test)],
                                 [PLOTLY_COLORS["primary"], PLOTLY_COLORS["danger"]]):
        vals = [m.get(mn, 0) for mn in metric_names]
        fig_comp.add_trace(go.Bar(name=name, x=metric_labels, y=vals, marker_color=color, opacity=0.85,
                                   text=[f"{v:.3f}" for v in vals], textposition="outside"))
    fig_comp.update_layout(title="Model Comparison — Test Set",
                           yaxis=dict(range=[0, 1.15], title="Score"),
                           barmode="group", template=PLOTLY_TEMPLATE, height=380)
    st.plotly_chart(fig_comp, width="stretch")

    st.markdown('<div class="section-header">Confusion Matrices — Test Set</div>', unsafe_allow_html=True)
    st.caption("False Negative = missed backorder (costly stockout). False Positive = unnecessary alert.")
    cm_col1, cm_col2 = st.columns(2)
    for col, (name, metrics) in zip([cm_col1, cm_col2],
                                     [("Logistic Regression", lr_test), ("Random Forest", rf_test)]):
        with col:
            cm_d = metrics.get("confusion_matrix", {})
            if cm_d:
                tn, fp, fn, tp = cm_d["tn"], cm_d["fp"], cm_d["fn"], cm_d["tp"]
                total = tn + fp + fn + tp
                fig_cm = go.Figure(go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=["Pred: No Backorder","Pred: Backorder"],
                    y=["Actual: No Backorder","Actual: Backorder"],
                    colorscale="Blues", showscale=False,
                    text=[[f"{tn:,}\n({100*tn/total:.1f}%)", f"{fp:,}\n({100*fp/total:.2f}%)"],
                          [f"{fn:,}\n({100*fn/total:.2f}%)", f"{tp:,}\n({100*tp/total:.2f}%)"]],
                    texttemplate="%{text}",
                ))
                fig_cm.update_layout(title=f"Confusion Matrix — {name}", height=320,
                                     template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_cm, width="stretch")

    col1, col2 = st.columns(2)
    fn = rf_test.get("confusion_matrix", {}).get("fn", "N/A")
    fp = rf_test.get("confusion_matrix", {}).get("fp", "N/A")
    with col1:
        st.markdown(f"""<div class="danger-box">
        <b>🔴 False Negatives (Missed Backorders): {fn:,}</b><br>
        Actual backorders incorrectly predicted as "No Backorder".
        Each is a potential stockout and customer disruption.
        Our model prioritizes minimizing these via high <b>Recall</b>.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="warning-box">
        <b>🟡 False Positives (Unnecessary Alerts): {fp:,}</b><br>
        Actual no-backorders incorrectly predicted as "High Risk".
        Generates unnecessary procurement review effort.
        We accept this trade-off (lower precision) to catch more true backorders.
        </div>""", unsafe_allow_html=True)

    imputer_values, feat_cols, rf_model, lr_model = load_model_artifacts()
    if rf_model is not None and feat_cols:
        st.markdown('<div class="section-header">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
        st.caption("Mean Decrease in Impurity — top contributing features. Importance ≠ causality.")
        importances = rf_model.feature_importances_
        feat_imp = pd.Series(importances, index=feat_cols).sort_values(ascending=False).head(20)
        colors_imp = [PLOTLY_COLORS["danger"] if i < 5 else PLOTLY_COLORS["warning"] if i < 10 else PLOTLY_COLORS["primary"]
                      for i in range(len(feat_imp))]
        fig_imp = go.Figure(go.Bar(
            x=feat_imp.values[::-1].tolist(), y=feat_imp.index[::-1].tolist(),
            orientation="h", marker_color=colors_imp[::-1],
            text=[f"{v:.4f}" for v in feat_imp.values[::-1]], textposition="outside",
        ))
        fig_imp.update_layout(title="Top 20 Feature Importances — Random Forest",
                              xaxis_title="Importance", template=PLOTLY_TEMPLATE,
                              height=540, margin=dict(l=200))
        st.plotly_chart(fig_imp, width="stretch")

        if lr_model is not None and hasattr(lr_model, "named_steps"):
            clf = lr_model.named_steps.get("clf")
            if clf and hasattr(clf, "coef_"):
                coefs = np.abs(clf.coef_[0])
                lr_imp = pd.Series(coefs, index=feat_cols).sort_values(ascending=False).head(15)
                fig_lr = go.Figure(go.Bar(
                    x=lr_imp.values[::-1].tolist(), y=lr_imp.index[::-1].tolist(),
                    orientation="h", marker_color=PLOTLY_COLORS["primary"],
                    text=[f"{v:.4f}" for v in lr_imp.values[::-1]], textposition="outside",
                ))
                fig_lr.update_layout(title="Top 15 |Coefficients| — Logistic Regression",
                                     xaxis_title="|Coefficient|", template=PLOTLY_TEMPLATE,
                                     height=420, margin=dict(l=200))
                st.plotly_chart(fig_lr, width="stretch")

    roc_img = os.path.join(FIGURES_DIR, "roc_pr_curves.png")
    if os.path.exists(roc_img):
        st.markdown('<div class="section-header">ROC & Precision-Recall Curves</div>', unsafe_allow_html=True)
        st.image(roc_img, width="stretch")

    st.markdown('<div class="section-header">Prescriptive Recommendations</div>', unsafe_allow_html=True)
    st.markdown("""
    **1. 🚨 Resource-Constrained Prioritization**
    - **FACT:** The positive class is highly imbalanced; checking all products is impossible.
    - **PREDICTION:** The Random Forest identifies High Risk products (probability ≥ 30%).
    - **RECOMMENDATION:** Filter to **High Risk** and sort by `risk_probability` descending to concentrate procurement resources.

    **2. 📦 Immediate Negative Inventory Review**
    - **FACT:** Products with `national_inv < 0` have a 15.1% historical backorder rate (24× higher than positive inventory).
    - **PREDICTION:** The model heavily weights negative inventory as a backorder risk.
    - **RECOMMENDATION:** Expedite replenishment for these products immediately.

    **3. 🔔 Past-Due Pieces Alert**
    - **FACT:** Products with past-due pieces show a substantially higher observed backorder rate.
    - **PREDICTION:** The model predicts elevated risk for products with `pieces_past_due > 0`.
    - **RECOMMENDATION:** Automate procurement alerts for any product developing past-due quantities.

    **4. 🏭 Dual-Sourcing for Critical SKUs**
    - **FACT:** Lower supplier performance (e.g., `< 0.70`) correlates with higher backorders.
    - **PREDICTION:** The model identifies these products as Medium to High risk when demand is present.
    - **RECOMMENDATION:** Review supplier diversification for critical SKUs with low performance scores.
    """)

