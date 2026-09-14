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
