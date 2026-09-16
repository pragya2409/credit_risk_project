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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# One-hot encode categorical columns (needed for LR; harmless for LightGBM/XGBoost)
categorical_cols = X.select_dtypes(include='str').columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Shape after encoding: {X.shape}")

import re
X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in X.columns]
print(f"Duplicate columns after cleaning: {X.columns.duplicated().sum()}")

# Train/test split — stratify to preserve the 92/8 class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Scale for LR specifically — LightGBM/XGBoost use X_train/X_test directly, unscaled
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

print("Scaling complete.")
df.to_csv('data/application_train_clean.csv', index=False)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_preds)

print(f"Logistic Regression baseline AUC: {lr_auc:.4f}")

from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(n_estimators=500, random_state=42)
lgbm_model.fit(X_train, y_train)  # note: raw X_train, not scaled — LightGBM doesn't need scaling

lgbm_preds = lgbm_model.predict_proba(X_test)[:, 1]
lgbm_auc = roc_auc_score(y_test, lgbm_preds)

print(f"LightGBM baseline AUC: {lgbm_auc:.4f}")