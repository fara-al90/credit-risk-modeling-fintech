# Credit Risk Modeling (FinTech Case Study)

## Problem

Predict whether a borrower will experience serious delinquency within 2 years  
(Target variable: `SeriousDlqin2yrs`)

This project compares:

- Logistic Regression (baseline model)
- HistGradientBoostingClassifier (tree-based model)

Models are evaluated using:
- ROC-AUC
- PR-AUC
- Cost-sensitive threshold optimization


## Dataset

Dataset: **Give Me Some Credit** (Kaggle)

- 150,000 observations
- 11 features
- Default rate: ~6.7% (imbalanced classification problem)

Dataset file is not included in this repository.  
Place `cs-training.csv` inside a local `data/` folder.

## Exploratory Data Analysis

Key findings:

- Default rate: 6.684%
- Strongest predictors:
  - NumberOfTime30-59DaysPastDueNotWorse
  - NumberOfTimes90DaysLate
  - NumberOfTime60-89DaysPastDueNotWorse
- Age negatively correlated with default
- Missing values:
  - MonthlyIncome (~29k missing)
  - NumberOfDependents (~3.9k missing)

A missing indicator feature (`MonthlyIncome_missing`) was created.


## Data Preprocessing

- Median imputation (via pipeline)
- Outlier capping:
  - Late payments capped at 10
  - Revolving utilization capped at 1.5
  - Debt ratio capped at 5
- Train/test split (80/20, stratified)


# Models


## 1. Logistic Regression (Baseline)

Pipeline:
- SimpleImputer (median)
- StandardScaler
- LogisticRegression

### Performance

- ROC-AUC: **0.859**
- PR-AUC: **0.381**

### Cost-Sensitive Optimization
Assumed business costs:
- False Negative (approve defaulter): $10,000
- False Positive (reject good borrower): $1,000

Optimal threshold: **0.53**  
Minimum expected cost: **$10,363,000**

## 2.HistGradientBoostingClassifier

Tree-based gradient boosting model.

Advantages:
- Captures non-linear relationships
- Handles interactions
- No scaling required
- Native handling of missing values

### Performance

- ROC-AUC: **0.867**
- PR-AUC: **0.408**

### Cost-Sensitive Optimization

Optimal threshold: **0.08**  
Minimum expected cost: **$9,881,000**

# Business Comparison

| Model        | ROC-AUC | PR-AUC | Min Expected Cost |
|-------------|---------|--------|-------------------|
| Logistic    | 0.859   | 0.381  | 10,363,000        |
| HGB         | 0.867   | 0.408  | 9,881,000         |

 HistGradientBoosting reduces expected cost by **~$482,000** under the defined business cost structure.

# Key Insight

Optimising for ROC-AUC alone is insufficient in credit risk.

Threshold tuning based on **financial cost** significantly changes the decision boundary.

At the optimized threshold (0.08), HGB:

- Catches ~74% of defaulters
- Minimizes financial loss under the defined cost assumptions

# Recommendation

Under the assumed cost structure:

**HistGradientBoostingClassifier + threshold 0.08**  
is the preferred model.

However:
- Logistic regression offers better interpretability.
- Regulatory environments may favor simpler models.

---

# Next Steps

- Cross-validation for stability
- Calibration analysis
- Feature importance (permutation)
- Regulatory interpretability considerations


# Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
