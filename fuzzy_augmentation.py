import pandas as pd
from sklearn.model_selection import train_test_split

# Load labeled data
train = pd.read_csv("data/application_train.csv")

# Split: 80% train_sub, 20% holdout_eval, stratified on TARGET
train_sub, holdout_eval = train_test_split(
    train,
    test_size=0.2,
    stratify=train["TARGET"],
    random_state=42,
)

# Sanity checks
print("train_sub:", train_sub.shape)
print("holdout_eval:", holdout_eval.shape)
print("train_sub default rate:", train_sub["TARGET"].mean().round(4))
print("holdout_eval default rate:", holdout_eval["TARGET"].mean().round(4))

# Save so every later step uses the exact same split
train_sub.to_csv("data/train_sub.csv", index=False)
holdout_eval.to_csv("data/holdout_eval.csv", index=False)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# --- Load the saved split + the unlabeled pool ---
train_sub = pd.read_csv("data/train_sub.csv")
holdout_eval = pd.read_csv("data/holdout_eval.csv")
app_test = pd.read_csv("data/application_test.csv")

features = [c for c in train_sub.columns if c not in ("TARGET", "SK_ID_CURR")]

# --- Make categorical columns consistent across all three frames ---
cat_cols = train_sub[features].select_dtypes(include=["object", "str"]).columns.tolist()
for col in cat_cols:
    levels = pd.concat([train_sub[col], holdout_eval[col], app_test[col]]).dropna().unique()
    cat_type = pd.CategoricalDtype(categories=sorted(levels))
    for df in (train_sub, holdout_eval, app_test):
        df[col] = df[col].astype(cat_type)

X_train, y_train = train_sub[features], train_sub["TARGET"]
X_hold, y_hold = holdout_eval[features], holdout_eval["TARGET"]

# --- Baseline model (these params get reused for the augmented model in step 5) ---
params = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
baseline = lgb.LGBMClassifier(**params)
baseline.fit(X_train, y_train)

# --- Evaluate on the untouched holdout ---
hold_pred = baseline.predict_proba(X_hold)[:, 1]
baseline_auc = roc_auc_score(y_hold, hold_pred)
baseline_prauc = average_precision_score(y_hold, hold_pred)
print(f"Baseline AUC:    {baseline_auc:.4f}")
print(f"Baseline PR-AUC: {baseline_prauc:.4f}")

# --- Step 3: score the unlabeled pool ---
X_pool = app_test[features]
p_bad = baseline.predict_proba(X_pool)[:, 1]   # P(TARGET=1) = P(default)

print("Pool size:", len(p_bad))
print("Mean P(default):", p_bad.mean().round(4))
print(pd.Series(p_bad).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4))

# --- Step 4: fuzzy duplication ---
# TARGET=1 means default (bad), TARGET=0 means repaid (good)
pool_good = app_test[features].copy()
pool_good["TARGET"] = 0
pool_good["weight"] = 1 - p_bad

pool_bad = app_test[features].copy()
pool_bad["TARGET"] = 1
pool_bad["weight"] = p_bad

pool_aug = pd.concat([pool_good, pool_bad], ignore_index=True)

# Real labeled rows keep full weight
real = train_sub[features + ["TARGET"]].copy()
real["weight"] = 1.0

augmented = pd.concat([real, pool_aug], ignore_index=True)

# --- Sanity checks ---
print("augmented shape:", augmented.shape)
print("weighted default rate:",
      (augmented["TARGET"] * augmented["weight"]).sum() / augmented["weight"].sum())
print("pool share of total weight:",
      pool_aug["weight"].sum() / augmented["weight"].sum())


# --- Step 5: retrain with fuzzy weights, same params as baseline ---
X_aug = augmented[features]
y_aug = augmented["TARGET"]
w_aug = augmented["weight"]

augmented_model = lgb.LGBMClassifier(**params)
augmented_model.fit(X_aug, y_aug, sample_weight=w_aug)

# --- Step 6 preview: evaluate on the same untouched holdout ---
aug_pred = augmented_model.predict_proba(X_hold)[:, 1]
aug_auc = roc_auc_score(y_hold, aug_pred)
aug_prauc = average_precision_score(y_hold, aug_pred)

print(f"Baseline   AUC: {baseline_auc:.4f}   PR-AUC: {baseline_prauc:.4f}")
print(f"Augmented  AUC: {aug_auc:.4f}   PR-AUC: {aug_prauc:.4f}")
print(f"Delta      AUC: {aug_auc - baseline_auc:+.4f}   PR-AUC: {aug_prauc - baseline_prauc:+.4f}")

# --- Multi-seed robustness check ---
import numpy as np
from sklearn.model_selection import train_test_split

full = pd.read_csv("data/application_train.csv")
pool_raw = pd.read_csv("data/application_test.csv")

for col in cat_cols:
    levels = pd.concat([full[col], pool_raw[col]]).dropna().unique()
    ct = pd.CategoricalDtype(categories=sorted(levels))
    full[col] = full[col].astype(ct)
    pool_raw[col] = pool_raw[col].astype(ct)

results = []
for seed in [1, 2, 3, 4, 5]:
    tr, ho = train_test_split(full, test_size=0.2, stratify=full["TARGET"], random_state=seed)
    p = {**params, "random_state": seed}

    base = lgb.LGBMClassifier(**p).fit(tr[features], tr["TARGET"])
    b_pred = base.predict_proba(ho[features])[:, 1]

    pb = base.predict_proba(pool_raw[features])[:, 1]
    good = pool_raw[features].assign(TARGET=0, weight=1 - pb)
    bad = pool_raw[features].assign(TARGET=1, weight=pb)
    real_df = tr[features + ["TARGET"]].assign(weight=1.0)
    aug = pd.concat([real_df, good, bad], ignore_index=True)

    aug_m = lgb.LGBMClassifier(**p).fit(aug[features], aug["TARGET"], sample_weight=aug["weight"])
    a_pred = aug_m.predict_proba(ho[features])[:, 1]

    row = dict(
        seed=seed,
        base_auc=roc_auc_score(ho["TARGET"], b_pred),
        aug_auc=roc_auc_score(ho["TARGET"], a_pred),
        base_pr=average_precision_score(ho["TARGET"], b_pred),
        aug_pr=average_precision_score(ho["TARGET"], a_pred),
    )
    results.append(row)
    print(row)

res = pd.DataFrame(results)
res["d_auc"] = res["aug_auc"] - res["base_auc"]
res["d_pr"] = res["aug_pr"] - res["base_pr"]
print(res[["seed", "d_auc", "d_pr"]].round(4))
print(f"Mean d_auc: {res['d_auc'].mean():+.4f}  std: {res['d_auc'].std():.4f}")
print(f"Mean d_pr:  {res['d_pr'].mean():+.4f}  std: {res['d_pr'].std():.4f}")
res.to_csv("fuzzy_augmentation_results.csv", index=False)