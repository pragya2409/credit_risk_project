import pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

c = pickle.load(open("stack_cache.pkl", "rb"))
app = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
lc = pd.read_csv("lc_scores_hc.csv")

# Alignment checks (fail loudly rather than silently mismatch rows)
assert (lc["SK_ID_CURR"].to_numpy() == app["SK_ID_CURR"].to_numpy()).all(), "LC scores not in HC row order"
assert (app.loc[c["idx_train"], "TARGET"].to_numpy() == c["y_train"]).all(), "train index misaligned"
assert (app.loc[c["idx_test"], "TARGET"].to_numpy() == c["y_test"]).all(), "test index misaligned"

lc_tr = lc.loc[c["idx_train"], "lc_score"].to_numpy()
lc_te = lc.loc[c["idx_test"], "lc_score"].to_numpy()
y_tr, y_te = c["y_train"], c["y_test"]

names = ["lgbm", "xgb", "lr"]
base_tr = np.column_stack([c["oof"][k] for k in names])
base_te = np.column_stack([c["test"][k] for k in names])

def fit_pred(Xtr, Xte):
    m = LogisticRegression().fit(Xtr, y_tr)        # same meta-learner as main.py
    return m, m.predict_proba(Xte)[:, 1]

m3, p3 = fit_pred(base_tr, base_te)
m4, p4 = fit_pred(np.column_stack([base_tr, lc_tr]), np.column_stack([base_te, lc_te]))

auc3, auc4 = roc_auc_score(y_te, p3), roc_auc_score(y_te, p4)
pr3, pr4 = average_precision_score(y_te, p3), average_precision_score(y_te, p4)
print(f"3-model stack: AUC {auc3:.4f} | PR-AUC {pr3:.4f}   (should reproduce ~0.7579)")
print(f"4-model stack: AUC {auc4:.4f} | PR-AUC {pr4:.4f}")
print(f"Delta AUC: {auc4 - auc3:+.5f}")

# Paired bootstrap on the test set for the AUC delta
rng = np.random.default_rng(42)
n, deltas = len(y_te), []
for _ in range(1000):
    i = rng.integers(0, n, n)
    if y_te[i].min() == y_te[i].max():
        continue
    deltas.append(roc_auc_score(y_te[i], p4[i]) - roc_auc_score(y_te[i], p3[i]))
lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"Bootstrap 95% CI for delta AUC: [{lo:+.5f}, {hi:+.5f}]")

# Diagnostics
print("\nMeta weights (4-model):", dict(zip(names + ["lc"], m4.coef_[0].round(3))))
print("Corr of LC score with OOF preds:",
      {k: round(float(np.corrcoef(lc_tr, c["oof"][k])[0, 1]), 3) for k in names})
print(f"LC score alone on HC test: AUC {roc_auc_score(y_te, lc_te):.4f}")

pd.DataFrame([{"auc_3": auc3, "auc_4": auc4, "pr_3": pr3, "pr_4": pr4,
               "delta": auc4 - auc3, "ci_lo": lo, "ci_hi": hi}]
             ).to_csv("lc_ablation_results.csv", index=False)