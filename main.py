import pandas as pd
import numpy as np
df = pd.read_csv('/Users/pragya/Development/credit_risk_project/data/application_train.csv')
print(df.shape)
print(df.head())
print(df['TARGET'].value_counts())
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0].head(20))
print(df.dtypes.value_counts())
categorical_cols = df.select_dtypes(include='str').columns
print(categorical_cols)
print(df[categorical_cols].nunique())
print(df.groupby('TARGET')['AMT_INCOME_TOTAL'].median())
print(df.groupby('TARGET')['AMT_CREDIT'].median())
print(df.groupby('TARGET')['DAYS_BIRTH'].median())
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"Overall missing: {total_missing / total_cells:.2%}")
print(df['DAYS_EMPLOYED'].describe())
print((df['DAYS_EMPLOYED'] == 365243).sum())
df['DAYS_EMPLOYED_ANOMALY'] = df['DAYS_EMPLOYED'] == 365243
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df = df.copy()
print(df['DAYS_EMPLOYED'].isnull().sum())
print(df['DAYS_EMPLOYED_ANOMALY'].sum())
df = df[df['CODE_GENDER'] != 'XNA']
print(df['CODE_GENDER'].value_counts())
threshold = 0.5
cols_before = df.shape[1]
df = df.loc[:, df.isnull().mean() < threshold]
cols_after = df.shape[1]
print(f"Dropped {cols_before - cols_after} columns with >50% missing")
remaining_missing = df.isnull().sum().sort_values(ascending=False)
print(remaining_missing[remaining_missing > 0].head(10))
cols_with_missing = df.columns[df.isnull().any()].tolist()
for col in cols_with_missing:
    if df[col].dtype != 'object':  # only flag numeric columns for now
        df[col + '_MISSING'] = df[col].isnull().astype(int)
from sklearn.impute import SimpleImputer

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print(df[numeric_cols].isnull().sum().sum())
df = df.copy()