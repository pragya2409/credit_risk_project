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

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_preds)

print(f"Logistic Regression baseline AUC: {lr_auc:.4f}")

#LightGBM
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(n_estimators=500, random_state=42)
lgbm_model.fit(X_train, y_train)  # note: raw X_train, not scaled — LightGBM doesn't need scaling

lgbm_preds = lgbm_model.predict_proba(X_test)[:, 1]
lgbm_auc = roc_auc_score(y_test, lgbm_preds)

print(f"LightGBM baseline AUC: {lgbm_auc:.4f}")

#XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)  # same raw, unscaled X_train as LightGBM

xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_preds)

print(f"XGBoost baseline AUC: {xgb_auc:.4f}")

from sklearn.model_selection import StratifiedKFold
import numpy as np

def generate_oof_predictions(model_fn, X, y, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        print(f"Fold {fold+1}/{n_splits} done")

    return oof_preds, test_preds


# Test it on just LightGBM for now
lgbm_oof, lgbm_test = generate_oof_predictions(
    lambda: LGBMClassifier(n_estimators=500, random_state=42),
    X_train, y_train, X_test
)

lgbm_oof_auc = roc_auc_score(y_train, lgbm_oof)
print(f"LightGBM OOF AUC (train set, via CV): {lgbm_oof_auc:.4f}")

# XGBoost OOF — same function, unscaled data, tuned learning rate from earlier
xgb_oof, xgb_test = generate_oof_predictions(
    lambda: XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, eval_metric='logloss'),
    X_train, y_train, X_test
)
xgb_oof_auc = roc_auc_score(y_train, xgb_oof)
print(f"XGBoost OOF AUC: {xgb_oof_auc:.4f}")


# LR OOF — note: uses the SCALED data, not X_train/X_test
lr_oof, lr_test = generate_oof_predictions(
    lambda: LogisticRegression(max_iter=1000),
    X_train_scaled, y_train, X_test_scaled
)
lr_oof_auc = roc_auc_score(y_train, lr_oof)
print(f"Logistic Regression OOF AUC: {lr_oof_auc:.4f}")

import pandas as pd

# Assemble meta-features from OOF predictions
X_meta_train = pd.DataFrame({
    'lgbm': lgbm_oof,
    'xgb': xgb_oof,
    'lr': lr_oof
})
X_meta_test = pd.DataFrame({
    'lgbm': lgbm_test,
    'xgb': xgb_test,
    'lr': lr_test
})

# Train the meta-learner
meta_model = LogisticRegression()
meta_model.fit(X_meta_train, y_train)

final_test_preds = meta_model.predict_proba(X_meta_test)[:, 1]
final_auc = roc_auc_score(y_test, final_test_preds)

print(f"Stacked ensemble AUC (on true test set): {final_auc:.4f}")
print(f"Meta-model weights: LightGBM={meta_model.coef_[0][0]:.3f}, XGBoost={meta_model.coef_[0][1]:.3f}, LR={meta_model.coef_[0][2]:.3f}")

import pickle
pickle.dump({
    "idx_train": X_train.index.to_numpy(), "idx_test": X_test.index.to_numpy(),
    "y_train": y_train.to_numpy(), "y_test": y_test.to_numpy(),
    "oof":  {"lgbm": lgbm_oof,  "xgb": xgb_oof,  "lr": lr_oof},
    "test": {"lgbm": lgbm_test, "xgb": xgb_test, "lr": lr_test},
}, open("stack_cache.pkl", "wb"))

from sklearn.metrics import average_precision_score
# Week 5: Comparison summary
results_summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'LightGBM', 'XGBoost', 'Stacked Ensemble'],
    'Standalone/Final AUC': [0.7469, 0.7520, 0.7562, final_auc]
})

# PR-AUC (average precision) — more informative than AUC on imbalanced data
lr_pr_auc = average_precision_score(y_test, lr_preds)
lgbm_pr_auc = average_precision_score(y_test, lgbm_preds)
xgb_pr_auc = average_precision_score(y_test, xgb_preds)
stacked_pr_auc = average_precision_score(y_test, final_test_preds)

results_summary['PR-AUC'] = [lr_pr_auc, lgbm_pr_auc, xgb_pr_auc, stacked_pr_auc]

print(results_summary)
results_summary.to_csv('results_summary.csv', index=False)

#ROC Curve Overlay
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# ROC curve overlay — all 4 models
plt.figure(figsize=(8, 6))
for name, preds in [('Logistic Regression', lr_preds), ('LightGBM', lgbm_preds),
                     ('XGBoost', xgb_preds), ('Stacked Ensemble', final_test_preds)]:
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Baselines vs. Stacked Ensemble')
plt.legend()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# AUC / PR-AUC bar chart
results_summary.set_index('Model')[['Standalone/Final AUC', 'PR-AUC']].plot(kind='bar', figsize=(8, 5))
plt.title('Model Comparison: AUC vs PR-AUC')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()