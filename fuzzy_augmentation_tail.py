import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

train_sub = pd.read_csv("data/train_sub.csv")
holdout_eval = pd.read_csv("data/holdout_eval.csv")

features = [c for c in train_sub.columns if c not in ("TARGET", "SK_ID_CURR")]
cat_cols = train_sub[features].select_dtypes(include=["object", "str"]).columns.tolist()

for col in cat_cols:
    levels = pd.concat([train_sub[col], holdout_eval[col]]).dropna().unique()
    ct = pd.CategoricalDtype(categories=sorted(levels))
    train_sub[col] = train_sub[col].astype(ct)
    holdout_eval[col] = holdout_eval[col].astype(ct)

# --- Proxy approval score: average of EXT_SOURCE_1/2/3, ignoring NaNs ---
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
ext_score = train_sub[ext_cols].mean(axis=1, skipna=True)

# Bottom 20% = proxy rejects (hidden). Top 80% = proxy accepts.
cutoff = ext_score.quantile(0.20)
is_reject = ext_score <= cutoff

accepts = train_sub[~is_reject].reset_index(drop=True)
rejects = train_sub[is_reject].reset_index(drop=True)
rejects_true_target = rejects["TARGET"].copy()   # keep the ground truth aside

print("accepts:", accepts.shape, "default rate:", accepts["TARGET"].mean().round(4))
print("rejects:", rejects.shape, "default rate:", rejects["TARGET"].mean().round(4))

# --- Baseline: train on accepts only, labels of rejects hidden ---
params = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
              random_state=42, n_jobs=-1, verbose=-1)
baseline_tail = lgb.LGBMClassifier(**params)
baseline_tail.fit(accepts[features], accepts["TARGET"])

# Eval 1: on the hidden rejects' TRUE labels — the real test
pred_on_rejects = baseline_tail.predict_proba(rejects[features])[:, 1]
auc_rejects = roc_auc_score(rejects_true_target, pred_on_rejects)
pr_rejects = average_precision_score(rejects_true_target, pred_on_rejects)

# Eval 2: on the untouched overall holdout, for context
pred_on_holdout = baseline_tail.predict_proba(holdout_eval[features])[:, 1]
auc_holdout = roc_auc_score(holdout_eval["TARGET"], pred_on_holdout)
pr_holdout = average_precision_score(holdout_eval["TARGET"], pred_on_holdout)

print(f"Baseline (accepts-only) on hidden rejects:  AUC {auc_rejects:.4f}  PR-AUC {pr_rejects:.4f}")
print(f"Baseline (accepts-only) on holdout_eval:     AUC {auc_holdout:.4f}  PR-AUC {pr_holdout:.4f}")

# --- Step B2: fuzzy-augment using the hidden rejects ---
p_bad_tail = baseline_tail.predict_proba(rejects[features])[:, 1]  # labels still hidden here

good_tail = rejects[features].copy()
good_tail["TARGET"] = 0
good_tail["weight"] = 1 - p_bad_tail

bad_tail = rejects[features].copy()
bad_tail["TARGET"] = 1
bad_tail["weight"] = p_bad_tail

pool_aug_tail = pd.concat([good_tail, bad_tail], ignore_index=True)
real_tail = accepts[features + ["TARGET"]].copy()
real_tail["weight"] = 1.0
augmented_tail = pd.concat([real_tail, pool_aug_tail], ignore_index=True)

print("augmented_tail shape:", augmented_tail.shape)
print("weighted default rate:",
      (augmented_tail["TARGET"] * augmented_tail["weight"]).sum() / augmented_tail["weight"].sum())

# --- Retrain on augmented data, same params ---
aug_model_tail = lgb.LGBMClassifier(**params)
aug_model_tail.fit(augmented_tail[features], augmented_tail["TARGET"],
                    sample_weight=augmented_tail["weight"])

# --- The real test: evaluate against the rejects' TRUE labels ---
pred_aug_on_rejects = aug_model_tail.predict_proba(rejects[features])[:, 1]
auc_aug_rejects = roc_auc_score(rejects_true_target, pred_aug_on_rejects)
pr_aug_rejects = average_precision_score(rejects_true_target, pred_aug_on_rejects)

# --- Also check the overall holdout didn't get worse ---
pred_aug_on_holdout = aug_model_tail.predict_proba(holdout_eval[features])[:, 1]
auc_aug_holdout = roc_auc_score(holdout_eval["TARGET"], pred_aug_on_holdout)
pr_aug_holdout = average_precision_score(holdout_eval["TARGET"], pred_aug_on_holdout)

print(f"Baseline  on rejects:  AUC {auc_rejects:.4f}  PR-AUC {pr_rejects:.4f}")
print(f"Augmented on rejects:  AUC {auc_aug_rejects:.4f}  PR-AUC {pr_aug_rejects:.4f}")
print(f"Delta     on rejects:  AUC {auc_aug_rejects - auc_rejects:+.4f}  PR-AUC {pr_aug_rejects - pr_rejects:+.4f}")
print()
print(f"Baseline  on holdout:  AUC {auc_holdout:.4f}  PR-AUC {pr_holdout:.4f}")
print(f"Augmented on holdout:  AUC {auc_aug_holdout:.4f}  PR-AUC {pr_aug_holdout:.4f}")
print(f"Delta     on holdout:  AUC {auc_aug_holdout - auc_holdout:+.4f}  PR-AUC {pr_aug_holdout - pr_holdout:+.4f}")
# --- Multi-seed check for the tail-based experiment ---
results_tail = []
for seed in [1, 2, 3, 4, 5]:
    p = {**params, "random_state": seed}

    # Re-split by EXT_SOURCE tail is deterministic (doesn't depend on seed),
    # so what varies here is only the model's own randomness.
    base_s = lgb.LGBMClassifier(**p)
    base_s.fit(accepts[features], accepts["TARGET"])

    b_rej_pred = base_s.predict_proba(rejects[features])[:, 1]
    b_auc_rej = roc_auc_score(rejects_true_target, b_rej_pred)
    b_pr_rej = average_precision_score(rejects_true_target, b_rej_pred)

    p_bad_s = base_s.predict_proba(rejects[features])[:, 1]
    good_s = rejects[features].assign(TARGET=0, weight=1 - p_bad_s)
    bad_s = rejects[features].assign(TARGET=1, weight=p_bad_s)
    real_s = accepts[features + ["TARGET"]].assign(weight=1.0)
    aug_s = pd.concat([real_s, good_s, bad_s], ignore_index=True)

    aug_model_s = lgb.LGBMClassifier(**p)
    aug_model_s.fit(aug_s[features], aug_s["TARGET"], sample_weight=aug_s["weight"])

    a_rej_pred = aug_model_s.predict_proba(rejects[features])[:, 1]
    a_auc_rej = roc_auc_score(rejects_true_target, a_rej_pred)
    a_pr_rej = average_precision_score(rejects_true_target, a_rej_pred)

    row = dict(seed=seed, base_auc=b_auc_rej, aug_auc=a_auc_rej,
               base_pr=b_pr_rej, aug_pr=a_pr_rej)
    results_tail.append(row)
    print(row)

res_tail = pd.DataFrame(results_tail)
res_tail["d_auc"] = res_tail["aug_auc"] - res_tail["base_auc"]
res_tail["d_pr"] = res_tail["aug_pr"] - res_tail["base_pr"]
print(res_tail[["seed", "d_auc", "d_pr"]].round(4))
print(f"Mean d_auc: {res_tail['d_auc'].mean():+.4f}  std: {res_tail['d_auc'].std():.4f}")
print(f"Mean d_pr:  {res_tail['d_pr'].mean():+.4f}  std: {res_tail['d_pr'].std():.4f}")
res_tail.to_csv("fuzzy_augmentation_tail_results.csv", index=False)
# --- XGBoost version of Experiment B ---
import xgboost as xgb

xgb_params = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=True,   # uses the same pandas 'category' dtype columns
    eval_metric="auc",
)

# --- Baseline: accepts only ---
xgb_baseline = xgb.XGBClassifier(**xgb_params)
xgb_baseline.fit(accepts[features], accepts["TARGET"])

xgb_pred_rejects = xgb_baseline.predict_proba(rejects[features])[:, 1]
xgb_auc_rejects = roc_auc_score(rejects_true_target, xgb_pred_rejects)
xgb_pr_rejects = average_precision_score(rejects_true_target, xgb_pred_rejects)

xgb_pred_holdout = xgb_baseline.predict_proba(holdout_eval[features])[:, 1]
xgb_auc_holdout = roc_auc_score(holdout_eval["TARGET"], xgb_pred_holdout)
xgb_pr_holdout = average_precision_score(holdout_eval["TARGET"], xgb_pred_holdout)

print(f"XGB Baseline on rejects: AUC {xgb_auc_rejects:.4f}  PR-AUC {xgb_pr_rejects:.4f}")
print(f"XGB Baseline on holdout: AUC {xgb_auc_holdout:.4f}  PR-AUC {xgb_pr_holdout:.4f}")

# --- Fuzzy-augment using XGBoost's own scores on the hidden rejects ---
xgb_p_bad = xgb_baseline.predict_proba(rejects[features])[:, 1]

xgb_good = rejects[features].copy()
xgb_good["TARGET"] = 0
xgb_good["weight"] = 1 - xgb_p_bad

xgb_bad = rejects[features].copy()
xgb_bad["TARGET"] = 1
xgb_bad["weight"] = xgb_p_bad

xgb_pool = pd.concat([xgb_good, xgb_bad], ignore_index=True)
xgb_real = accepts[features + ["TARGET"]].copy()
xgb_real["weight"] = 1.0
xgb_augmented = pd.concat([xgb_real, xgb_pool], ignore_index=True)

# --- Retrain on augmented data ---
xgb_aug_model = xgb.XGBClassifier(**xgb_params)
xgb_aug_model.fit(xgb_augmented[features], xgb_augmented["TARGET"],
                   sample_weight=xgb_augmented["weight"])

xgb_aug_pred_rejects = xgb_aug_model.predict_proba(rejects[features])[:, 1]
xgb_aug_auc_rejects = roc_auc_score(rejects_true_target, xgb_aug_pred_rejects)
xgb_aug_pr_rejects = average_precision_score(rejects_true_target, xgb_aug_pred_rejects)

xgb_aug_pred_holdout = xgb_aug_model.predict_proba(holdout_eval[features])[:, 1]
xgb_aug_auc_holdout = roc_auc_score(holdout_eval["TARGET"], xgb_aug_pred_holdout)
xgb_aug_pr_holdout = average_precision_score(holdout_eval["TARGET"], xgb_aug_pred_holdout)

print(f"XGB Baseline  on rejects: AUC {xgb_auc_rejects:.4f}  PR-AUC {xgb_pr_rejects:.4f}")
print(f"XGB Augmented on rejects: AUC {xgb_aug_auc_rejects:.4f}  PR-AUC {xgb_aug_pr_rejects:.4f}")
print(f"XGB Delta     on rejects: AUC {xgb_aug_auc_rejects - xgb_auc_rejects:+.4f}  PR-AUC {xgb_aug_pr_rejects - xgb_pr_rejects:+.4f}")
print()
print(f"XGB Baseline  on holdout: AUC {xgb_auc_holdout:.4f}  PR-AUC {xgb_pr_holdout:.4f}")
print(f"XGB Augmented on holdout: AUC {xgb_aug_auc_holdout:.4f}  PR-AUC {xgb_aug_pr_holdout:.4f}")
print(f"XGB Delta     on holdout: AUC {xgb_aug_auc_holdout - xgb_auc_holdout:+.4f}  PR-AUC {xgb_aug_pr_holdout - xgb_pr_holdout:+.4f}")
# --- Logistic Regression version of Experiment B ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# One-hot encode categoricals (fit on accepts, align columns across all three sets)
accepts_lr = pd.get_dummies(accepts[features], columns=cat_cols, dummy_na=True)
rejects_lr = pd.get_dummies(rejects[features], columns=cat_cols, dummy_na=True)
holdout_lr = pd.get_dummies(holdout_eval[features], columns=cat_cols, dummy_na=True)

accepts_lr, rejects_lr = accepts_lr.align(rejects_lr, join="left", axis=1, fill_value=0)
accepts_lr, holdout_lr = accepts_lr.align(holdout_lr, join="left", axis=1, fill_value=0)

# Median-impute remaining numeric NaNs, using accepts' medians only (no leakage)
medians = accepts_lr.median(numeric_only=True)
accepts_lr = accepts_lr.fillna(medians)
rejects_lr = rejects_lr.fillna(medians)
holdout_lr = holdout_lr.fillna(medians)

# Scale, fit on accepts only
scaler_lr = StandardScaler()
accepts_lr_scaled = pd.DataFrame(scaler_lr.fit_transform(accepts_lr), columns=accepts_lr.columns)
rejects_lr_scaled = pd.DataFrame(scaler_lr.transform(rejects_lr), columns=rejects_lr.columns)
holdout_lr_scaled = pd.DataFrame(scaler_lr.transform(holdout_lr), columns=holdout_lr.columns)

# --- Baseline LR: accepts only ---
lr_baseline = LogisticRegression(max_iter=1000)
lr_baseline.fit(accepts_lr_scaled, accepts["TARGET"])

lr_pred_rejects = lr_baseline.predict_proba(rejects_lr_scaled)[:, 1]
lr_auc_rejects = roc_auc_score(rejects_true_target, lr_pred_rejects)
lr_pr_rejects = average_precision_score(rejects_true_target, lr_pred_rejects)

lr_pred_holdout = lr_baseline.predict_proba(holdout_lr_scaled)[:, 1]
lr_auc_holdout = roc_auc_score(holdout_eval["TARGET"], lr_pred_holdout)
lr_pr_holdout = average_precision_score(holdout_eval["TARGET"], lr_pred_holdout)

print(f"LR Baseline on rejects: AUC {lr_auc_rejects:.4f}  PR-AUC {lr_pr_rejects:.4f}")
print(f"LR Baseline on holdout: AUC {lr_auc_holdout:.4f}  PR-AUC {lr_pr_holdout:.4f}")

# --- Fuzzy-augment using LR's own scores on the hidden rejects ---
lr_p_bad = lr_baseline.predict_proba(rejects_lr_scaled)[:, 1]

lr_good = rejects_lr_scaled.copy()
lr_good["TARGET"] = 0
lr_good["weight"] = 1 - lr_p_bad

lr_bad = rejects_lr_scaled.copy()
lr_bad["TARGET"] = 1
lr_bad["weight"] = lr_p_bad

lr_pool = pd.concat([lr_good, lr_bad], ignore_index=True)
lr_real = accepts_lr_scaled.copy()
lr_real["TARGET"] = accepts["TARGET"].values
lr_real["weight"] = 1.0
lr_augmented = pd.concat([lr_real, lr_pool], ignore_index=True)

lr_feat_cols = accepts_lr_scaled.columns.tolist()
lr_aug_model = LogisticRegression(max_iter=1000)
lr_aug_model.fit(lr_augmented[lr_feat_cols], lr_augmented["TARGET"],
                  sample_weight=lr_augmented["weight"])

lr_aug_pred_rejects = lr_aug_model.predict_proba(rejects_lr_scaled)[:, 1]
lr_aug_auc_rejects = roc_auc_score(rejects_true_target, lr_aug_pred_rejects)
lr_aug_pr_rejects = average_precision_score(rejects_true_target, lr_aug_pred_rejects)

lr_aug_pred_holdout = lr_aug_model.predict_proba(holdout_lr_scaled)[:, 1]
lr_aug_auc_holdout = roc_auc_score(holdout_eval["TARGET"], lr_aug_pred_holdout)
lr_aug_pr_holdout = average_precision_score(holdout_eval["TARGET"], lr_aug_pred_holdout)

print(f"LR Baseline  on rejects: AUC {lr_auc_rejects:.4f}  PR-AUC {lr_pr_rejects:.4f}")
print(f"LR Augmented on rejects: AUC {lr_aug_auc_rejects:.4f}  PR-AUC {lr_aug_pr_rejects:.4f}")
print(f"LR Delta     on rejects: AUC {lr_aug_auc_rejects - lr_auc_rejects:+.4f}  PR-AUC {lr_aug_pr_rejects - lr_pr_rejects:+.4f}")
print()
print(f"LR Baseline  on holdout: AUC {lr_auc_holdout:.4f}  PR-AUC {lr_pr_holdout:.4f}")
print(f"LR Augmented on holdout: AUC {lr_aug_auc_holdout:.4f}  PR-AUC {lr_aug_pr_holdout:.4f}")
print(f"LR Delta     on holdout: AUC {lr_aug_auc_holdout - lr_auc_holdout:+.4f}  PR-AUC {lr_aug_pr_holdout - lr_pr_holdout:+.4f}")
import numpy as np
from sklearn.model_selection import StratifiedKFold

# --- Non-augmented OOF, but supporting multiple eval sets at once ---
def generate_oof_predictions_v2(model_fn, X, y, eval_sets, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = {name: np.zeros(len(ev)) for name, ev in eval_sets.items()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        for name, ev in eval_sets.items():
            test_preds[name] += model.predict_proba(ev)[:, 1] / n_splits
        print(f"Fold {fold+1}/{n_splits} done (baseline)")

    return oof_preds, test_preds


# --- Augmented OOF: each fold trains a scorer, fuzzy-augments with its own
#     reject scores, then retrains on fold-train + fuzzy rejects ---
def generate_oof_predictions_augmented(model_fn, X, y, reject_X, eval_sets, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = {name: np.zeros(len(ev)) for name, ev in eval_sets.items()}
    feat_cols = X.columns.tolist()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fold-scorer: trained on this fold's accepts only, no reject exposure yet
        scorer = model_fn()
        scorer.fit(X_tr, y_tr)
        p_bad = scorer.predict_proba(reject_X)[:, 1]

        # Fuzzy-duplicate rejects using this fold's own scores
        good = reject_X.copy(); good["TARGET"] = 0; good["weight"] = 1 - p_bad
        bad = reject_X.copy(); bad["TARGET"] = 1; bad["weight"] = p_bad
        pool = pd.concat([good, bad], ignore_index=True)

        real = X_tr.copy()
        real["TARGET"] = y_tr.values
        real["weight"] = 1.0

        fold_aug = pd.concat([real, pool], ignore_index=True)

        model = model_fn()
        model.fit(fold_aug[feat_cols], fold_aug["TARGET"], sample_weight=fold_aug["weight"])

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        for name, ev in eval_sets.items():
            test_preds[name] += model.predict_proba(ev)[:, 1] / n_splits
        print(f"Fold {fold+1}/{n_splits} done (augmented)")

    return oof_preds, test_preds


# --- Shared eval sets ---
eval_sets_tree = {"rejects": rejects[features], "holdout": holdout_eval[features]}
eval_sets_lr = {"rejects": rejects_lr_scaled, "holdout": holdout_lr_scaled}

lgbm_fn = lambda: lgb.LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
xgb_fn = lambda: xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42,
                                    eval_metric="logloss", tree_method="hist", enable_categorical=True)
lr_fn = lambda: LogisticRegression(max_iter=1000)

# --- BASELINE stack ---
lgbm_oof_b, lgbm_t_b = generate_oof_predictions_v2(lgbm_fn, accepts[features], accepts["TARGET"], eval_sets_tree)
xgb_oof_b, xgb_t_b = generate_oof_predictions_v2(xgb_fn, accepts[features], accepts["TARGET"], eval_sets_tree)
lr_oof_b, lr_t_b = generate_oof_predictions_v2(lr_fn, accepts_lr_scaled, accepts["TARGET"], eval_sets_lr)

X_meta_b = pd.DataFrame({"lgbm": lgbm_oof_b, "xgb": xgb_oof_b, "lr": lr_oof_b})
meta_b = LogisticRegression().fit(X_meta_b, accepts["TARGET"])

meta_rejects_b = pd.DataFrame({"lgbm": lgbm_t_b["rejects"], "xgb": xgb_t_b["rejects"], "lr": lr_t_b["rejects"]})
meta_holdout_b = pd.DataFrame({"lgbm": lgbm_t_b["holdout"], "xgb": xgb_t_b["holdout"], "lr": lr_t_b["holdout"]})
pred_rejects_b = meta_b.predict_proba(meta_rejects_b)[:, 1]
pred_holdout_b = meta_b.predict_proba(meta_holdout_b)[:, 1]

auc_stack_rejects_b = roc_auc_score(rejects_true_target, pred_rejects_b)
pr_stack_rejects_b = average_precision_score(rejects_true_target, pred_rejects_b)
auc_stack_holdout_b = roc_auc_score(holdout_eval["TARGET"], pred_holdout_b)
pr_stack_holdout_b = average_precision_score(holdout_eval["TARGET"], pred_holdout_b)

print(f"BASELINE stack  on rejects: AUC {auc_stack_rejects_b:.4f}  PR-AUC {pr_stack_rejects_b:.4f}")
print(f"BASELINE stack  on holdout: AUC {auc_stack_holdout_b:.4f}  PR-AUC {pr_stack_holdout_b:.4f}")

# --- AUGMENTED stack ---
lgbm_oof_a, lgbm_t_a = generate_oof_predictions_augmented(lgbm_fn, accepts[features], accepts["TARGET"], rejects[features], eval_sets_tree)
xgb_oof_a, xgb_t_a = generate_oof_predictions_augmented(xgb_fn, accepts[features], accepts["TARGET"], rejects[features], eval_sets_tree)
lr_oof_a, lr_t_a = generate_oof_predictions_augmented(lr_fn, accepts_lr_scaled, accepts["TARGET"], rejects_lr_scaled, eval_sets_lr)

X_meta_a = pd.DataFrame({"lgbm": lgbm_oof_a, "xgb": xgb_oof_a, "lr": lr_oof_a})
meta_a = LogisticRegression().fit(X_meta_a, accepts["TARGET"])

meta_rejects_a = pd.DataFrame({"lgbm": lgbm_t_a["rejects"], "xgb": xgb_t_a["rejects"], "lr": lr_t_a["rejects"]})
meta_holdout_a = pd.DataFrame({"lgbm": lgbm_t_a["holdout"], "xgb": xgb_t_a["holdout"], "lr": lr_t_a["holdout"]})
pred_rejects_a = meta_a.predict_proba(meta_rejects_a)[:, 1]
pred_holdout_a = meta_a.predict_proba(meta_holdout_a)[:, 1]

auc_stack_rejects_a = roc_auc_score(rejects_true_target, pred_rejects_a)
pr_stack_rejects_a = average_precision_score(rejects_true_target, pred_rejects_a)
auc_stack_holdout_a = roc_auc_score(holdout_eval["TARGET"], pred_holdout_a)
pr_stack_holdout_a = average_precision_score(holdout_eval["TARGET"], pred_holdout_a)

print(f"AUGMENTED stack on rejects: AUC {auc_stack_rejects_a:.4f}  PR-AUC {pr_stack_rejects_a:.4f}")
print(f"AUGMENTED stack on holdout: AUC {auc_stack_holdout_a:.4f}  PR-AUC {pr_stack_holdout_a:.4f}")

print()
print(f"STACK DELTA on rejects: AUC {auc_stack_rejects_a - auc_stack_rejects_b:+.4f}  PR-AUC {pr_stack_rejects_a - pr_stack_rejects_b:+.4f}")
print(f"STACK DELTA on holdout: AUC {auc_stack_holdout_a - auc_stack_holdout_b:+.4f}  PR-AUC {pr_stack_holdout_a - pr_stack_holdout_b:+.4f}")
print(f"Meta weights (baseline):  LGBM={meta_b.coef_[0][0]:.3f} XGB={meta_b.coef_[0][1]:.3f} LR={meta_b.coef_[0][2]:.3f}")
print(f"Meta weights (augmented): LGBM={meta_a.coef_[0][0]:.3f} XGB={meta_a.coef_[0][1]:.3f} LR={meta_a.coef_[0][2]:.3f}")

import pickle, numpy as np
keep = [c for c in holdout_eval.columns
        if c in ("SK_ID_CURR", "TARGET", "DAYS_BIRTH") or "GENDER" in c]
print("holdout_eval kept cols:", keep, "| rows:", len(holdout_eval))
print("index sample:", list(holdout_eval.index[:3]))
pickle.dump(
    holdout_eval[keep].assign(pred_base=np.asarray(pred_holdout_b),
                              pred_aug=np.asarray(pred_holdout_a)),
    open("fuzzy_tail_cache.pkl", "wb"))