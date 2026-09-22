# AI-Powered Supply Chain Risk & Backorder Prediction

## IBM SkillsBuild Data Analytics with AI Academic Internship

An end-to-end Machine Learning and Data Analytics project designed to help supply-chain managers identify products at risk of going on backorder and take preventive action before inventory shortages disrupt operations.

---

## Project Overview

Supply-chain managers need to identify products that are at risk of going on backorder so that preventive action can be taken before inventory shortages disrupt operations.

This project analyzes inventory levels, demand forecasts, supply quantities, lead times, supplier performance, and operational risk factors to:

- Understand current supply-chain conditions
- Identify factors associated with backorders
- Predict backorder risk
- Categorize products by risk level
- Recommend preventive supply-chain actions

**Domain:** Supply Chain & Logistics  
**Stakeholder:** Supply Chain / Inventory Manager  
**Target Variable:** `went_on_backorder`

---

## Business Questions

1. What does the overall inventory and supply-chain situation look like?
2. Which inventory, demand, lead-time, supplier-performance, and operational-risk factors are associated with backorders?
3. Which product conditions are associated with particularly high backorder risk?
4. Can backorder occurrence be predicted using information available before the backorder occurs?
5. Which predicted high-risk products should be prioritized for intervention?

---

## Analytical Workflow

The project follows a four-tier analytics framework:

### 1. Descriptive Analysis
- Dataset structure and distributions
- Backorder frequency
- Inventory conditions
- Supply and demand patterns
- Key supply-chain KPIs

### 2. Diagnostic Analysis
- Inventory vs. backorder relationships
- Supplier performance
- Lead-time patterns
- Operational risk indicators
- Statistical associations with backorders

### 3. Predictive Analysis
Two classification models are evaluated:

- Logistic Regression — baseline model
- Random Forest — primary model

The project addresses severe class imbalance because only approximately **0.67%** of training products went on backorder.

### 4. Prescriptive Analysis
Products are categorized into:

- High Risk
- Medium Risk
- Low Risk

Risk categories are mapped to practical supply-chain actions such as replenishment review, supplier monitoring, and resolution of past-due shipments.

---

## Dataset

The project uses the Backorder Prediction dataset containing inventory, demand, supply, supplier-performance, and operational-risk attributes.

### Main datasets

- `Training_BOP.csv` — original training dataset
- `Testing_BOP.csv` — external holdout test dataset
- `Training_BOP_sample.csv` — 100,000-row representative sample included with the deployed dashboard

The original datasets are used for the complete modeling workflow. The included sample is used by the deployed Streamlit dashboard to keep the repository lightweight.

### Important columns

```text
sku
national_inv
lead_time
in_transit_qty
forecast_3_month
forecast_6_month
forecast_9_month
sales_1_month
sales_3_month
sales_6_month
sales_9_month
min_bank
potential_issue
pieces_past_due
perf_6_month_avg
perf_12_month_avg
local_bo_qty
deck_risk
oe_constraint
ppap_risk
stop_auto_buy
rev_stop
went_on_backorder
```

---

## Data Preprocessing & Feature Engineering

The preprocessing pipeline handles:

- Missing values
- Sentinel values such as `-99`
- Binary Yes/No operational flags
- Data type conversion
- Feature engineering
- Class imbalance

Engineered features include:

- `inv_demand_ratio`
- `stock_coverage_months`
- `demand_gap`
- `supply_vs_demand`
- `total_risk_flags`
- `sales_acceleration`
- `avg_supplier_perf`
- `past_due_flag`
- `local_bo_flag`
- `lead_time_missing`
- `nat_inv_is_negative`

---

## Machine Learning Models

### Logistic Regression

Used as an interpretable baseline classification model.

### Random Forest

Used as the primary tree-based classification model to capture non-linear relationships and interactions between supply-chain variables.

### Class Imbalance

The target variable is highly imbalanced, with approximately 0.67% positive backorder cases.

Class-weighting techniques are used during model development to give greater importance to the minority backorder class.

### Decision Threshold

A probability threshold of **0.30** is used for the deployed risk classification workflow to prioritize recall and reduce missed high-risk backorder cases.

---

## Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

The final evaluation uses an external holdout test dataset that is kept separate from model training.

---

## Explainability & Risk Analysis

The project analyzes important supply-chain signals associated with backorders, including:

- National inventory
- Lead time
- Past-due quantities
- Supplier performance
- Local backorder quantity
- Operational risk flags
- Demand and forecast variables

Feature-importance analysis is included to provide additional interpretability for the Random Forest model.

> Statistical association is interpreted as association rather than proof of causation.

---

## Risk Prioritization

The prediction system converts model probabilities into risk categories.

### High Risk
Products requiring immediate review, particularly when inventory is depleted or other critical risk signals are present.

### Medium Risk
Products requiring closer monitoring and preventive planning.

### Low Risk
Products with comparatively lower predicted backorder risk under the model.

The dashboard also provides recommended actions based on the identified risk factors.

---

## Business Recommendations

1. **Expedite High-Risk Replenishments**  
   Prioritize products flagged as high risk for procurement and inventory review.

2. **Monitor Supplier Lead Times**  
   Review suppliers and products associated with extended lead times.

3. **Address Past-Due Shipments**  
   Investigate products with outstanding past-due quantities.

4. **Monitor Operational Risk Flags**  
   Use multiple operational risk indicators to support preventive intervention.

5. **Use Risk-Based Prioritization**  
   Focus limited supply-chain management resources on products with higher predicted risk.

---

## Interactive Streamlit Dashboard

The project includes a five-page Streamlit application:

### 1. Executive Overview
Provides supply-chain KPIs, backorder overview, risk distribution, key findings, and model performance.

### 2. Supply Chain Analysis
Provides interactive analysis of inventory, lead time, demand, in-transit quantities, supplier performance, operational risk flags, and feature associations.

### 3. Individual Risk Prediction
Allows users to enter product supply-chain parameters and receive:

- Backorder probability
- Risk category
- Key risk factors
- Recommended action

### 4. Batch Risk Analysis
Allows users to upload a CSV and generate risk predictions for multiple products.

### 5. Model Performance
Displays:

- Model comparison
- Test-set metrics
- Confusion matrices
- Feature importance
- ROC and Precision-Recall curves

---

## Project Structure

```text
Supply_Chain_Backorder_Project/
│
├── app.py
├── project.ipynb
├── train.py
├── requirements.txt
├── README.md
├── Training_BOP_sample.csv
│
├── .streamlit/
│   └── config.toml
│
├── src/
│   ├── analysis.py
│   ├── data_preprocessing.py
│   ├── modeling.py
│   └── utils.py
│
├── models/
│   └── Saved trained model artifacts
│
└── outputs/
    ├── figures/
    ├── metrics/
    └── predictions/
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- SciPy
- Joblib
- Jupyter Notebook
- Streamlit

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Supply_Chain_Backorder_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## Running the Complete Training Pipeline

If the original training datasets are available in the expected data location, the complete pipeline can be executed using:

```bash
python train.py
```

The pipeline generates:

- Trained models
- Evaluation metrics
- Visualizations
- Test-set predictions
- Risk prioritization outputs

> The complete training pipeline operates on the original datasets. The deployed dashboard uses the included 100,000-row representative training sample for interactive supply-chain analysis.

---

## Key Project Outcomes

The project provides an end-to-end workflow connecting:

```text
Raw Supply Chain Data
        ↓
Data Cleaning
        ↓
Exploratory & Diagnostic Analysis
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Risk Prediction
        ↓
Risk Categorization
        ↓
Prescriptive Actions
        ↓
Interactive Dashboard
```

This allows supply-chain stakeholders to move from understanding what happened to identifying risk and deciding where preventive action should be focused.

---

## Academic Context

Developed as part of the:

**IBM SkillsBuild Data Analytics with AI Academic Internship**

**Project Domain:** Supply Chain & Logistics

**Project Focus:** AI-Powered Supply Chain Risk & Backorder Prediction

---

## Author

**Gautam Raju Saripalli**

B.Tech — Artificial Intelligence & Data Science

Chaitanya Bharathi Institute of Technology (CBIT)
