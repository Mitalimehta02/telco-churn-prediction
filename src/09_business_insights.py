import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

X_test = np.load('models/X_test.npy', allow_pickle=True)
y_test = np.load('models/y_test.npy', allow_pickle=True)

model   = joblib.load('models/xgboost.pkl')
y_proba = model.predict_proba(X_test)[:, 1]

risk_df = pd.DataFrame({'churn_probability': y_proba, 'actual_churn': y_test})
risk_df['risk_tier'] = pd.cut(y_proba, bins=[0, 0.4, 0.7, 1.0],
                               labels=['Low Risk', 'Medium Risk', 'High Risk'])

print("Risk tier distribution:")
print(risk_df['risk_tier'].value_counts())
print("\nActual churn rate per tier:")
print(risk_df.groupby('risk_tier', observed=True)['actual_churn'].mean().round(3))

risk_df.sort_values('churn_probability', ascending=False).to_csv(
    'reports/customer_risk_scores.csv', index=False
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y_proba[y_test == 0], bins=30, alpha=0.6, color='#2ecc71', label='No Churn', density=True)
axes[0].hist(y_proba[y_test == 1], bins=30, alpha=0.6, color='#e74c3c', label='Churned',  density=True)
axes[0].axvline(0.4, color='#f39c12', ls='--', label='Medium threshold')
axes[0].axvline(0.7, color='#c0392b', ls='--', label='High threshold')
axes[0].set_xlabel('Predicted Churn Probability')
axes[0].set_ylabel('Density')
axes[0].set_title('Churn Probability Distribution')
axes[0].legend(fontsize=9)

tier_counts = risk_df['risk_tier'].value_counts()
tier_colors = {'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#2ecc71'}
bars = axes[1].bar(tier_counts.index, tier_counts.values,
                   color=[tier_colors[t] for t in tier_counts.index], edgecolor='white', width=0.5)
axes[1].set_title('Customers by Risk Tier')
axes[1].set_xlabel('Risk Tier')
axes[1].set_ylabel('Number of Customers')
for bar, val in zip(bars, tier_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 str(val), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/19_risk_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()

sorted_idx    = np.argsort(y_proba)[::-1]
sorted_actual = y_test[sorted_idx]
baseline_rate = y_test.mean()
lift     = np.cumsum(sorted_actual) / (np.arange(1, len(sorted_actual) + 1) * baseline_rate)
coverage = np.arange(1, len(sorted_actual) + 1) / len(sorted_actual) * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(coverage, lift, color='#e74c3c', lw=2.5, label='XGBoost')
ax.axhline(y=1, color='gray', ls='--', label='Random baseline')
ax.fill_between(coverage, 1, lift, where=lift > 1, alpha=0.15, color='#e74c3c')
ax.set_xlabel('% Customers Contacted (by Risk Score)')
ax.set_ylabel('Cumulative Lift')
ax.set_title('Lift Curve — How Much Better Than Random Targeting?')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/20_lift_curve.png', dpi=150, bbox_inches='tight')
plt.close()

lift_20 = lift[int(len(lift) * 0.20)]
print(f"\nLift at 20% coverage: {lift_20:.2f}x")
print("Plots saved to plots/")
