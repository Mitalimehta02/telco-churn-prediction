import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

X_train = np.load('models/X_train_smote.npy', allow_pickle=True)
y_train = np.load('models/y_train_smote.npy', allow_pickle=True)
X_test  = np.load('models/X_test.npy',        allow_pickle=True)
y_test  = np.load('models/y_test.npy',         allow_pickle=True)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    'XGBoost':              XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                          subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                                          random_state=42, n_jobs=-1),
    'SVM':                  SVC(kernel='rbf', probability=True, random_state=42),
    'KNN':                  KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
}

results  = []
roc_data = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    rpt = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    results.append({
        'Model':     name,
        'Accuracy':  round(rpt['accuracy'], 4),
        'Precision': round(rpt['1']['precision'], 4),
        'Recall':    round(rpt['1']['recall'], 4),
        'F1':        round(rpt['1']['f1-score'], 4),
        'AUC':       round(auc, 4),
    })
    roc_data[name] = y_proba
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv('reports/ml_results.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
for (name, y_proba), color in zip(roc_data.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} ({roc_auc_score(y_test, y_proba):.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Classical ML Models')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/07_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
hm = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]
sns.heatmap(hm, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5,
            vmin=0.5, vmax=1.0, annot_kws={'weight': 'bold'}, ax=ax)
ax.tick_params(axis='y', rotation=0)
ax.set_title('Model Performance Heatmap')
plt.tight_layout()
plt.savefig('plots/08_model_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

top3 = results_df.head(3)['Model'].tolist()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, name in enumerate(top3):
    m  = joblib.load(f"models/{name.replace(' ', '_').lower()}.pkl")
    cm = confusion_matrix(y_test, m.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    axes[i].set_title(f'{name}\nAUC={results_df[results_df.Model==name].AUC.values[0]:.3f}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/09_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
