import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

X_train = np.load('models/X_train_smote.npy', allow_pickle=True).astype(np.float32)
y_train = np.load('models/y_train_smote.npy', allow_pickle=True).astype(np.float32)
X_test  = np.load('models/X_test.npy',        allow_pickle=True).astype(np.float32)
y_test  = np.load('models/y_test.npy',         allow_pickle=True).astype(np.float32)

n_features  = X_train.shape[1]
train_ds    = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


class SimpleMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),          nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(1)


class DeepMLPDropout(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),           nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(1)


class BNDropoutMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),           nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(1)


def train_model(model, train_loader, epochs=100, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    all_X = train_loader.dataset.tensors[0]
    all_y = train_loader.dataset.tensors[1]

    # Stratified val split — shuffle first so val set has both classes
    idx   = torch.randperm(len(all_X))
    all_X, all_y = all_X[idx], all_y[idx]
    n_val = int(0.15 * len(all_X))
    X_val, y_val = all_X[:n_val], all_y[:n_val]
    X_tr,  y_tr  = all_X[n_val:], all_y[n_val:]
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    train_losses, val_losses = [], []
    best_loss, patience_counter, best_weights = float('inf'), 0, None

    for _ in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        train_losses.append(epoch_loss / len(tr_loader))
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    return model, train_losses, val_losses


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_proba = model(torch.from_numpy(X_test)).numpy()
    y_pred = (y_proba >= 0.5).astype(int)
    rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # '1' key may be missing if model predicts only one class — fall back to 0
    cls = rpt.get('1', {'precision': 0, 'recall': 0, 'f1-score': 0})
    return {
        'Accuracy':  round(rpt['accuracy'], 4),
        'Precision': round(cls['precision'], 4),
        'Recall':    round(cls['recall'], 4),
        'F1':        round(cls['f1-score'], 4),
        'AUC':       round(roc_auc_score(y_test, y_proba), 4),
    }, y_proba


dl_models = {
    'Simple MLP':        SimpleMLP(n_features),
    'Deep MLP + Dropout': DeepMLPDropout(n_features),
    'BN + Dropout MLP':  BNDropoutMLP(n_features),
}

results, histories, probas = {}, {}, {}

for name, model in dl_models.items():
    print(f"Training {name}...")
    model, tl, vl   = train_model(model, train_loader)
    metrics, y_proba = evaluate(model, X_test, y_test)
    results[name]   = metrics
    histories[name] = (tl, vl)
    probas[name]    = y_proba
    fname = name.replace(' ', '_').replace('+', '').replace('__', '_').lower()
    torch.save(model.state_dict(), f"models/{fname}.pt")

results_df = pd.DataFrame(results).T
print(results_df)
results_df.to_csv('reports/dl_results.csv')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, (tl, vl)) in zip(axes, histories.items()):
    ax.plot(tl, label='Train', color='#3498db')
    ax.plot(vl, label='Val',   color='#e74c3c')
    ax.set_title(name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
plt.suptitle('Training History — PyTorch Models', fontsize=13)
plt.tight_layout()
plt.savefig('plots/10_dl_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e74c3c', '#3498db', '#2ecc71']
for (name, y_proba), color in zip(probas.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} ({roc_auc_score(y_test, y_proba):.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — PyTorch Models')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/11_dl_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
results_df[['Accuracy', 'Recall', 'F1', 'AUC']].plot(kind='bar', ax=ax, edgecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha='right')
ax.set_ylim(0.5, 1.0)
ax.set_title('PyTorch DL Models — Metric Comparison')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/12_dl_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Deep learning plots saved.")
