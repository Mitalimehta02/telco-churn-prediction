import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.inspection import permutation_importance
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("plots", exist_ok=True)

X_test       = np.load('models/X_test.npy', allow_pickle=True)
y_test       = np.load('models/y_test.npy', allow_pickle=True)
feature_names = joblib.load('models/feature_names.pkl')

rf  = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')

rf_imp  = pd.DataFrame({'Feature': feature_names, 'Importance': rf.feature_importances_})
rf_imp  = rf_imp.sort_values('Importance', ascending=False).head(15)
xgb_imp = pd.DataFrame({'Feature': feature_names, 'Importance': xgb.feature_importances_})
xgb_imp = xgb_imp.sort_values('Importance', ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(rf_imp['Feature'][::-1],  rf_imp['Importance'][::-1],
             color=sns.color_palette('Blues_d', 15),   edgecolor='white')
axes[0].set_title('Random Forest — Feature Importance')
axes[0].set_xlabel('Gini Importance')
axes[1].barh(xgb_imp['Feature'][::-1], xgb_imp['Importance'][::-1],
             color=sns.color_palette('Oranges_d', 15), edgecolor='white')
axes[1].set_title('XGBoost — Feature Importance')
axes[1].set_xlabel('Gain')
plt.tight_layout()
plt.savefig('plots/15_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

perm = permutation_importance(rf, X_test, y_test, n_repeats=10,
                               random_state=42, scoring='roc_auc', n_jobs=-1)
perm_df = pd.DataFrame({'Feature': feature_names,
                         'Mean': perm.importances_mean,
                         'Std':  perm.importances_std}) \
            .sort_values('Mean', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(perm_df['Feature'][::-1], perm_df['Mean'][::-1],
        xerr=perm_df['Std'][::-1], color=sns.color_palette('Greens_d', 15),
        edgecolor='white', capsize=3)
ax.set_title('Permutation Importance (Random Forest, ROC-AUC drop)')
ax.set_xlabel('Mean AUC Decrease')
plt.tight_layout()
plt.savefig('plots/16_permutation_importance.png', dpi=150, bbox_inches='tight')
plt.close()

explainer   = shap.TreeExplainer(xgb)
X_sample    = X_test[:500]
shap_values = explainer.shap_values(X_sample)

fig = plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
plt.title('SHAP Summary — XGBoost', fontsize=13)
plt.tight_layout()
plt.savefig('plots/17_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                  plot_type='bar', show=False, max_display=15)
plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=13)
plt.tight_layout()
plt.savefig('plots/18_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

print("Explainability plots saved.")
