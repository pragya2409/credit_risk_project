import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

APPROVAL, N_BOOT, SEED = 0.80, 1000, 42
BINS, LABELS = [0, 30, 40, 50, 60, 200], ["<30", "30-39", "40-49", "50-59", "60+"]
METRICS = ["DI_ratio", "tpr_gap", "fpr_gap", "auc_gap"]

def add_groups(df):
    df = df[df["CODE_GENDER"].isin(["F", "M"])].copy()
    df["age_band"] = pd.cut(-df["DAYS_BIRTH"] / 365.25, bins=BINS, labels=LABELS, right=False)
    return df

# ---- data: identical to fairness.py ----
c = pickle.load(open("stack_cache.pkl", "rb"))
names = ["lgbm", "xgb", "lr"]
meta = LogisticRegression().fit(np.column_stack([c["oof"][k] for k in names]), c["y_train"])
stack_te = meta.predict_proba(np.column_stack([c["test"][k] for k in names]))[:, 1]
app = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"])
A = app.loc[c["idx_test"], ["CODE_GENDER", "DAYS_BIRTH"]].reset_index(drop=True)
A["TARGET"], A["xgb"], A["stack3"] = c["y_test"], c["test"]["xgb"], stack_te
A = add_groups(A)
Bt = add_groups(pickle.load(open("fuzzy_tail_cache.pkl", "rb")))

TABLES = [("A_test", A, ["xgb", "stack3"]), ("B_tail_holdout", Bt, ["pred_base", "pred_aug"])]
PAIRS = {"A_test": ("xgb", "stack3"), "B_tail_holdout": ("pred_base", "pred_aug")}  # (reference, new)

def group_stats(score, y, g, k):
    """Same definitions as fairness.py, vectorised."""
    appr = (score <= np.quantile(score, APPROVAL)).astype(float)
    dec = 1.0 - appr
    n = np.bincount(g, minlength=k).astype(float)
    bad = np.bincount(g, weights=y.astype(float), minlength=k)
    good = n - bad
    with np.errstate(divide="ignore", invalid="ignore"):
        approval = np.bincount(g, weights=appr, minlength=k) / n
        tpr = np.bincount(g, weights=y * dec, minlength=k) / bad
        fpr = np.bincount(g, weights=(1 - y) * dec, minlength=k) / good
    auc = np.full(k, np.nan)
    for j in range(k):
        if bad[j] > 0 and good[j] > 0:
            m = g == j
            r = rankdata(score[m])
            auc[j] = (r[y[m] == 1].sum() - bad[j] * (bad[j] + 1) / 2) / (bad[j] * good[j])
    return np.array([np.nanmin(approval) / np.nanmax(approval),
                     np.nanmax(tpr) - np.nanmin(tpr),
                     np.nanmax(fpr) - np.nanmin(fpr),
                     np.nanmax(auc) - np.nanmin(auc)])

est_rows, ci_rows, diff_rows = [], [], []
rng = np.random.default_rng(SEED)

for table, df, scores in TABLES:
    y = df["TARGET"].to_numpy().astype(int)
    assert df["age_band"].notna().all()
    G = {"CODE_GENDER": (pd.Categorical(df["CODE_GENDER"]).codes.astype(int), 2),
         "age_band": (df["age_band"].cat.codes.to_numpy().astype(int), len(LABELS))}
    S = {s: df[s].to_numpy().astype(float) for s in scores}
    n = len(df)
    print(f"{table}: {n} rows, bootstrapping {N_BOOT} resamples ...", flush=True)

    point = {(s, a): group_stats(S[s], y, *G[a]) for s in scores for a in G}
    boot = {(s, a): np.empty((N_BOOT, 4)) for s in scores for a in G}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)              # same rows for every model: paired
        for a, (g, k) in G.items():
            for s in scores:
                boot[(s, a)][b] = group_stats(S[s][idx], y[idx], g[idx], k)

    for (s, a), pt in point.items():
        lo, hi = np.nanpercentile(boot[(s, a)], [2.5, 97.5], axis=0)
        for i, m in enumerate(METRICS):
            est_rows.append({"table": table, "model": s, "attr": a, "metric": m,
                             "estimate": pt[i], "ci_low": lo[i], "ci_high": hi[i]})
    ref, new = PAIRS[table]
    for a in G:
        d = boot[(new, a)] - boot[(ref, a)]
        lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
        pt = point[(new, a)] - point[(ref, a)]
        for i, m in enumerate(METRICS):
            diff_rows.append({"table": table, "comparison": f"{new} - {ref}", "attr": a,
                              "metric": m, "diff": pt[i], "ci_low": lo[i], "ci_high": hi[i],
                              "excludes_0": bool(lo[i] > 0 or hi[i] < 0)})

est, diff = pd.DataFrame(est_rows), pd.DataFrame(diff_rows)

# ---- check: point estimates must reproduce fairness_summary.csv ----
old = pd.read_csv("fairness_summary.csv")
chk = est.pivot_table(index=["table", "model", "attr"], columns="metric", values="estimate").reset_index()
chk = chk.merge(old, on=["table", "model", "attr"], suffixes=("_new", "_old"))
maxdiff = max(float((chk[m + "_new"] - chk[m + "_old"]).abs().max()) for m in METRICS)
print(f"\nCHECK vs fairness_summary.csv: max |difference| = {maxdiff:.2e} over {len(chk)} rows")
assert maxdiff < 1e-6, "point estimates do not match fairness.py -- stop and check"

est.to_csv("fairness_bootstrap.csv", index=False)
diff.to_csv("fairness_bootstrap_diff.csv", index=False)
pd.set_option("display.width", 220)
print("\nPoint estimates with 95% CI:\n", est.round(4).to_string(index=False))
print("\nDifferences (new minus reference), paired:\n", diff.round(4).to_string(index=False))