import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

df = pd.read_csv("data/telco_churn.csv")

print(df.shape)
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())

# TotalCharges is stored as object - has spaces where values should be null
print((df['TotalCharges'] == ' ').sum())

print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(3))
