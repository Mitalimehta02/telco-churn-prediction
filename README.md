# Customer Churn Prediction

End-to-end ML pipeline to predict telecom customer churn using the [IBM Telco dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Compares classical ML models against PyTorch neural networks.

---

## Project Structure

```
churn_prediction/
├── data/                          ← place dataset CSV here
├── src/
│   ├── 01_data_understanding.py
│   ├── 02_preprocessing.py
│   ├── 03_eda.py
│   ├── 04_class_imbalance.py
│   ├── 05_ml_models.py
│   ├── 06_deep_learning.py        ← PyTorch
│   ├── 07_model_comparison.py
│   ├── 08_explainability.py       ← SHAP + permutation importance
│   └── 09_business_insights.py
├── models/                        ← saved models + preprocessors
├── plots/                         ← all output visualizations
├── reports/                       ← CSVs with metrics
├── app.py                         ← Streamlit dashboard
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in `data/`.

Run scripts in order:

```bash
python src/01_data_understanding.py
python src/02_preprocessing.py
python src/03_eda.py
python src/04_class_imbalance.py
python src/05_ml_models.py
python src/06_deep_learning.py
python src/07_model_comparison.py
python src/08_explainability.py
python src/09_business_insights.py
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

---

## Models

**Classical ML** (scikit-learn + XGBoost): Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM, KNN

**Deep Learning** (PyTorch): Simple MLP, Deep MLP with Dropout, Batch Norm + Dropout MLP

Class imbalance handled via SMOTE, class weights, and random undersampling — compared in `04_class_imbalance.py`.

---

## Evaluation

Models evaluated on: Accuracy, Precision, Recall, F1, ROC-AUC.

Recall is prioritised — missing a churner (false negative) is more costly than a false alarm.

Explainability via SHAP values and permutation importance. Top churn drivers: **contract type**, **tenure**, and **monthly charges**.

---

## Results Summary

| Model | AUC | Recall |
|---|---|---|
| XGBoost | ~0.84 | ~0.76 |
| Random Forest | ~0.83 | ~0.72 |
| Gradient Boosting | ~0.82 | ~0.70 |
| Deep MLP + Dropout | ~0.81 | ~0.71 |
| Simple MLP | ~0.80 | ~0.68 |

> Results on real dataset. Synthetic data used for pipeline testing produces lower scores.

---

## Key Findings

- Month-to-month customers churn at ~42% vs ~3% for 2-year contract customers
- New customers (tenure < 12 months) are the highest risk group
- Fiber optic users churn significantly more than DSL users
- Customers with no add-on services (security, tech support) are more likely to leave
- Contacting the top 20% by risk score catches ~2.5x more churners than random outreach
