import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

X_test    = np.load('models/X_test.npy', allow_pickle=True).astype(np.float32)
y_test    = np.load('models/y_test.npy', allow_pickle=True)
n_features = X_test.shape[1]


class SimpleMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(1)

class DeepMLPDropout(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n,128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64,32),  nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(32,1),   nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(1)

class BNDropoutMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128,64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(64,32),  nn.ReLU(),
                                  nn.Linear(32,1),   nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(1)


def sklearn_metrics(name):
    m       = joblib.load(f"models/{name.replace(' ', '_').lower()}.pkl")
    y_proba = m.predict_proba(X_test)[:, 1]
    y_pred  = m.predict(X_test)
    rpt = classification_report(y_test, y_pred, output_dict=True)
    return {'Model': name, 'Type': 'Classical ML',
            'Accuracy': round(rpt['accuracy'], 4), 'Precision': round(rpt['1']['precision'], 4),
            'Recall': round(rpt['1']['recall'], 4), 'F1': round(rpt['1']['f1-score'], 4),
            'AUC': round(roc_auc_score(y_test, y_proba), 4)}, y_proba

def dl_metrics(model, weights_path, name):
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        y_proba = model(torch.from_numpy(X_test)).numpy()
    y_pred = (y_proba >= 0.5).astype(int)
    rpt = classification_report(y_test, y_pred, output_dict=True)
    return {'Model': name, 'Type': 'Deep Learning',
            'Accuracy': round(rpt['accuracy'], 4), 'Precision': round(rpt['1']['precision'], 4),
            'Recall': round(rpt['1']['recall'], 4), 'F1': round(rpt['1']['f1-score'], 4),
            'AUC': round(roc_auc_score(y_test, y_proba), 4)}, y_proba


all_results, all_proba = [], {}

for name in ['Logistic Regression', 'Decision Tree', 'Random Forest',
             'Gradient Boosting', 'XGBoost', 'SVM', 'KNN']:
    try:
        row, proba = sklearn_metrics(name)
        all_results.append(row)
        all_proba[name] = proba
    except Exception as e:
        print(f"Skipping {name}: {e}")

dl_configs = [
    (SimpleMLP(n_features),       'models/simple_mlp.pt',              'Simple MLP'),
    (DeepMLPDropout(n_features),  'models/deep_mlp__dropout.pt',       'Deep MLP + Dropout'),
    (BNDropoutMLP(n_features),    'models/bn__dropout_mlp.pt',         'BN + Dropout MLP'),
]
for model, path, name in dl_configs:
    try:
        row, proba = dl_metrics(model, path, name)
        all_results.append(row)
        all_proba[name] = proba
    except Exception as e:
        print(f"Skipping {name}: {e}")

results_df = pd.DataFrame(all_results).sort_values('AUC', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv('reports/final_comparison.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ml_colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22']
dl_colors = ['#2c3e50','#7f8c8d','#bdc3c7']
ml_idx, dl_idx = 0, 0

for row in results_df.itertuples():
    if row.Model not in all_proba:
        continue
    fpr, tpr, _ = roc_curve(y_test, all_proba[row.Model])
    if row.Type == 'Classical ML':
        axes[0].plot(fpr, tpr, color=ml_colors[ml_idx % len(ml_colors)], lw=2,
                     label=f"{row.Model} ({row.AUC:.3f})")
        ml_idx += 1
    else:
        axes[1].plot(fpr, tpr, color=dl_colors[dl_idx % len(dl_colors)], lw=2,
                     label=f"{row.Model} ({row.AUC:.3f})")
        dl_idx += 1

for ax, title in zip(axes, ['Classical ML', 'Deep Learning (PyTorch)']):
    ax.plot([0,1],[0,1],'k--',lw=1.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {title}')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('ROC Curves: ML vs Deep Learning', fontsize=14)
plt.tight_layout()
plt.savefig('plots/13_final_roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
hm = results_df.set_index('Model')[['Accuracy','Precision','Recall','F1','AUC']]
sns.heatmap(hm, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5,
            vmin=0.5, vmax=1.0, annot_kws={'weight':'bold'}, ax=ax)
ax.tick_params(axis='y', rotation=0)
ax.set_title('All Models — Final Performance Comparison')
plt.tight_layout()
plt.savefig('plots/14_final_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nBest model:", results_df.iloc[0]['Model'], "| AUC:", results_df.iloc[0]['AUC'])
