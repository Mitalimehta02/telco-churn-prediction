import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/telco_churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cc = df['Churn'].value_counts()
axes[0].bar(cc.index, cc.values, color=['#2ecc71', '#e74c3c'], width=0.5)
axes[0].set_title('Churn Count')
axes[1].pie(cc.values, labels=cc.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
axes[1].set_title('Churn Split')
plt.tight_layout()
plt.savefig('plots/01_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title('Churn Rate by Contract Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(['No Churn', 'Churned'])
plt.tight_layout()
plt.savefig('plots/02_churn_by_contract.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, color in [('No', '#2ecc71'), ('Yes', '#e74c3c')]:
    axes[0].hist(df[df['Churn'] == label]['tenure'], bins=30, alpha=0.6, color=color, label=label)
axes[0].set_title('Tenure by Churn')
axes[0].set_xlabel('Tenure (months)')
axes[0].legend(title='Churn')
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, alpha=0.4,
            palette={'No': '#2ecc71', 'Yes': '#e74c3c'}, ax=axes[1])
axes[1].set_title('Monthly Charges Distribution by Churn')
plt.tight_layout()
plt.savefig('plots/03_tenure_charges.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
internet_churn.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title('Churn Rate by Internet Service')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(['No Churn', 'Churned'])
plt.tight_layout()
plt.savefig('plots/04_churn_by_internet.png', dpi=150, bbox_inches='tight')
plt.close()

df_temp = df.copy()
df_temp['Churn'] = (df_temp['Churn'] == 'Yes').astype(int)
corr = df_temp[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0, linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("EDA plots saved to plots/")
