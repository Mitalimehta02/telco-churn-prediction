import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Make all paths relative to the project root, not the src/ folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/telco_churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

def tenure_group(t):
    if t <= 12: return 'New'
    elif t <= 24: return 'Growing'
    elif t <= 48: return 'Established'
    else: return 'Loyal'

df['tenure_group'] = df['tenure'].apply(tenure_group)

binary_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod', 'tenure_group'])

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

np.save('models/X_train.npy', X_train.values)
np.save('models/X_test.npy', X_test.values)
np.save('models/y_train.npy', y_train.values)
np.save('models/y_test.npy', y_test.values)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Features: {list(X.columns)}")
