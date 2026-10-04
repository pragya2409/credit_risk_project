import pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

APPROVAL = 0.80   # each model approves its lowest-risk 80%, same threshold for every group
BINS, LABELS = [0, 30, 40, 50, 60, 200], ["<30", "30-39", "40-49", "50-59", "60+"]

def add_groups(df):
    df = df[df["CODE_GENDER"].isin(["F", "M"])].copy()
    df["age_band"] = pd.cut(-df["DAYS_BIRTH"] / 365.25, bins=BINS, labels=LABELS, right=False)
    return df

# ---- Table A: main.py test set (XGBoost vs 3-model stack), rebuilt from the cache ----
c = pickle.load(open("stack_cache.pkl", "rb"))
names = ["lgbm", "xgb", "lr"]
meta = LogisticRegression().fit(np.column_stack([c["oof"][k] for k in names]), c["y_train"])
stack_te = meta.predict_proba(np.column_stack([c["test"][k] for k in names]))[:, 1]

app = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"])
A = app.loc[c["idx_test"], ["CODE_GENDER", "DAYS_BIRTH"]].reset_index(drop=True)
A["TARGET"], A["xgb"], A["stack3"] = c["y_test"], c["test"]["xgb"], stack_te
A = add_groups(A)

# ---- Table B: tail-experiment holdout (baseline vs augmented stack) ----
B = add_groups(pickle.load(open("fuzzy_tail_cache.pkl", "rb")))

def group_metrics(df, score, attr):
    thr = df[score].quantile(APPROVAL)
    df = df.assign(approved=(df[score] <= thr))
    rows = []
    for g, d in df.groupby(attr, observed=True):
        bad, good = d[d["TARGET"] == 1], d[d["TARGET"] == 0]
        rows.append({"attr": attr, "group": g, "n": len(d),
                     "default_rate": d["TARGET"].mean(), "mean_pred": d[score].mean(),
                     "auc": roc_auc_score(d["TARGET"], d[score]),
                     "approval_rate": d["approved"].mean(),
                     "tpr": (~bad["approved"]).mean(),    # defaulters correctly declined
                     "fpr": (~good["approved"]).mean()})  # good applicants wrongly declined
    return pd.DataFrame(rows)

def summary(m):
    return {"DI_ratio": m["approval_rate"].min() / m["approval_rate"].max(),
            "tpr_gap": m["tpr"].max() - m["tpr"].min(),
            "fpr_gap": m["fpr"].max() - m["fpr"].min(),
            "auc_gap": m["auc"].max() - m["auc"].min()}

out, summ = [], []
for table, df, scores in [("A_test", A, ["xgb", "stack3"]),
                          ("B_tail_holdout", B, ["pred_base", "pred_aug"])]:
    print(f"{table}: {len(df)} rows")
    for s in scores:
        print(f"  {s} overall AUC: {roc_auc_score(df['TARGET'], df[s]):.4f}")
        for attr in ["CODE_GENDER", "age_band"]:
            m = group_metrics(df, s, attr)
            out.append(m.assign(table=table, model=s))
            summ.append({"table": table, "model": s, "attr": attr, **summary(m)})

res, sm = pd.concat(out), pd.DataFrame(summ)
res.to_csv("fairness_results.csv", index=False)
sm.to_csv("fairness_summary.csv", index=False)

print("\n=== SUMMARY (DI_ratio: 1.0 is parity, 0.8 is the usual reference line) ===")
print(sm.round(4).to_string(index=False))
print("\n=== GENDER, per group ===")
print(res[res["attr"] == "CODE_GENDER"].round(4).to_string(index=False))