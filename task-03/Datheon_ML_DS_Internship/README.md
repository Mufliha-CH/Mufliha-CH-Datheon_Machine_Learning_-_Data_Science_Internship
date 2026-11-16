# Datheon Machine Learning & Data Science Internship

This repository contains my completed internship tasks.

## Repository structure
```
.
├── README.md
├── requirements.txt
└── Task3_Churn_Classifier/
    ├── Telco_Churn_Classifier.ipynb
    └── Telco-Customer-Churn.csv
```

## Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)

**Goal:** Predict which telecom customers will churn, handle severe class imbalance, and reach ROC-AUC > 80%.

**Approach**
1. Loaded the Telco Customer Churn dataset (7,043 customers, ~26.5% churn).
2. Converted `TotalCharges` to numeric and filled 11 nulls with 0 (customers with tenure 0).
3. Encoded categorical features with `OneHotEncoder` and scaled numeric features.
4. Balanced classes with **SMOTE**, applied inside the cross-validation pipeline to avoid data leakage.
5. Trained `RandomForestClassifier` and `XGBClassifier` with 5-fold stratified cross-validation.
6. Evaluated on a held-out 20% test set; plotted ROC curves, confusion matrices, and top-10 feature importances.

**Results**

| Model | 5-fold CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Random Forest | 0.839 | 0.835 |
| XGBoost | 0.843 | 0.840 |

Both models exceed the 0.80 target.

**Tools:** Python, Scikit-Learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib, Pandas

## How to run
```bash
pip install -r requirements.txt
cd Task3_Churn_Classifier
jupyter notebook Telco_Churn_Classifier.ipynb
```
Run all cells from top to bottom.

## Other tasks
Additional tasks will be added in their own folders (e.g. `Task1_.../`).
