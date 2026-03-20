import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

X_train = np.load('models/X_train.npy', allow_pickle=True)
X_test  = np.load('models/X_test.npy',  allow_pickle=True)
y_train = np.load('models/y_train.npy', allow_pickle=True)
y_test  = np.load('models/y_test.npy',  allow_pickle=True)


def eval_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    rpt = classification_report(y_test, y_pred, output_dict=True)
    return {
        'accuracy':  rpt['accuracy'],
        'precision': rpt['1']['precision'],
        'recall':    rpt['1']['recall'],
        'f1':        rpt['1']['f1-score'],
        'auc':       roc_auc_score(y_test, y_proba)
    }


results = {}

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
results['Baseline'] = eval_model(lr, X_test, y_test)

lr_w = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_w.fit(X_train, y_train)
results['Class Weights'] = eval_model(lr_w, X_test, y_test)

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
np.save('models/X_train_smote.npy', X_sm)
np.save('models/y_train_smote.npy', y_sm)
lr_sm = LogisticRegression(max_iter=1000, random_state=42)
lr_sm.fit(X_sm, y_sm)
results['SMOTE'] = eval_model(lr_sm, X_test, y_test)

rus = RandomUnderSampler(random_state=42)
X_us, y_us = rus.fit_resample(X_train, y_train)
lr_us = LogisticRegression(max_iter=1000, random_state=42)
lr_us.fit(X_us, y_us)
results['Undersample'] = eval_model(lr_us, X_test, y_test)

print(pd.DataFrame(results).T.round(4))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
for i, metric in enumerate(['accuracy', 'recall', 'f1']):
    vals = [results[k][metric] for k in results]
    bars = axes[i].bar(list(results.keys()), vals, color=colors, edgecolor='white')
    axes[i].set_title(metric.capitalize())
    axes[i].set_ylim(0, 1.0)
    axes[i].set_xticklabels(list(results.keys()), rotation=10, ha='right')
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{v:.3f}', ha='center', fontsize=9)
plt.suptitle('Imbalance Handling — Logistic Regression', fontsize=13)
plt.tight_layout()
plt.savefig('plots/06_imbalance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
