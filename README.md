# Loan Default Prediction
**Home Credit Default Risk — End-to-End ML Pipeline**

Predicting whether a loan applicant will default using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset. Built as a portfolio project covering the full data science lifecycle from raw data to a tuned, explainable model.

---

## Results

| Stage | AUC |
|---|---|
| Baseline — `application_train.csv` only | 0.7706 |
| + Bureau feature aggregation | 0.7740 |
| + Optuna hyperparameter tuning (50 trials) | 0.7766 |

---

## Pipeline Overview

### Phase 1 — Data Understanding
- 307,511 loan applications, 122 raw features
- Target: binary (1 = defaulted, 0 = repaid) — 8% positive class (heavily imbalanced)
- Auto-install cell ensures `shap`, `optuna`, `lightgbm` and core libraries are present on any machine

### Phase 2 — Data Cleaning
- Audited 67 columns with missing values
- Retained `EXT_SOURCE_1/2/3` despite >40% missingness — strongest predictors in credit risk literature
- Flagged `DAYS_EMPLOYED = 365243` as a data anomaly (encoded "unemployed"), created `FLAG_EMPLOYED_ANOMALY` binary flag before imputing
- Created `FLAG_OWN_CAR_AGE_MISSING` indicator feature

### Phase 3 — Exploratory Data Analysis
- `EXT_SOURCE_2` shows clearest separation between defaulters and non-defaulters
- Younger applicants (20–30) default at ~12% vs ~5% for 50–60 age group
- Raw income and loan amount show minimal class separation — ratio features are more predictive
- All correlations with TARGET < 0.20 — confirms non-linear model needed
- Identified multicollinearity: `DAYS_BIRTH` (duplicate of `AGE_YEARS`) and `REGION_RATING_CLIENT` (0.95 corr with `REGION_RATING_CLIENT_W_CITY`) dropped

### Phase 4 — Feature Engineering
8 new features engineered from domain knowledge:

| Feature | Rationale |
|---|---|
| `CREDIT_INCOME_RATIO` | Debt burden relative to income |
| `ANNUITY_INCOME_RATIO` | Monthly repayment affordability |
| `CREDIT_TERM` | Implied loan duration |
| `GOODS_CREDIT_RATIO` | Collateral coverage |
| `EMPLOYED_TO_AGE_RATIO` | Employment stability proxy |
| `DOCUMENT_SUBMISSION_RATE` | Application completeness signal |
| `SOCIAL_CIRCLE_DEFAULT_RATE` | Peer default risk |
| `TOTAL_CREDIT_ENQUIRIES` | Recent credit-seeking behaviour |

### Phase 5 — Preprocessing
- Label encoded all categorical features (including `CODE_GENDER` XNA → mode imputed)
- `AGE_GROUP` and `DAYS_EMPLOYED` dropped post-EDA (replaced by engineered features)
- Stratified 80/20 train/test split
- `scale_pos_weight = 11.39` to handle class imbalance in LightGBM

### Phase 6 — Baseline Modelling
- LightGBM with early stopping (50 rounds patience), up to 1000 estimators
- Threshold tuning via F1-score vs threshold curve — optimal threshold < 0.5 for imbalanced data
- Baseline AUC: **0.7706**

### Phase 7 — Enrichment, Explainability & Tuning

**7A — Bureau Feature Aggregation**
- Joined `bureau.csv` + `bureau_balance.csv`
- `bureau_balance` aggregated to bureau level first (DPD rate, status counts), then merged up to application level
- 16 new bureau features including `BUREAU_DEBT_CREDIT_RATIO`, `BUREAU_DPD_RATE_MEAN`, `BUREAU_OVERDUE_SUM`, `BUREAU_ACTIVE_RATIO`
- Total enriched feature set: 98 features
- AUC lift: +0.0034 → **0.7740**

**7B — SHAP Explainability**
- `TreeExplainer` on 2,000-sample subset for speed
- `EXT_SOURCE_2` and `EXT_SOURCE_3` confirmed as dominant predictors globally
- `BUREAU_DEBT_CREDIT_RATIO` appears in top 3 for high-risk applicants — validates bureau join added real signal
- Waterfall plot on a single defaulter: low EXT_SOURCE scores drive +1.14 and +0.7 log-odds push toward default from base rate of -0.59

**7C — Optuna Hyperparameter Tuning**
- 50 trials, TPE sampler (~21 min on M1 base MacBook)
- Best config: 936 trees, `learning_rate=0.017`, `num_leaves=74`, `max_depth=9`, `reg_lambda=3.33`
- AUC lift: +0.0026 → **0.7766**

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, metrics |
| `lightgbm` | Gradient boosted trees |
| `shap` | Model explainability |
| `optuna` | Hyperparameter optimisation |

---

## Dataset

Download from [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data). Place the following files in the project root:

```
application_train.csv
bureau.csv
bureau_balance.csv
```

Files not included in this repo due to size.

---

## Known Limitations & Next Steps

- Only `application_train.csv`, `bureau.csv`, and `bureau_balance.csv` used — 4 additional tables available (`installments_payments`, `previous_application`, `credit_card_balance`, `POS_CASH_balance`)
- Single train/test split — no cross-validation yet
- Median imputation on `EXT_SOURCE_1/3` introduces distribution spike artifact
- No probability calibration — predicted scores not yet validated against true default rates
- Planned: Phase 8 (cross-validation + KS statistic), Phase 9 (scorecard / risk tier banding)

---

## Author

**Danial** — MSc Data Science candidate, Universiti Teknologi PETRONAS
