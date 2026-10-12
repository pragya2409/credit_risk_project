import os, sys, time, pickle, re
from glob import glob
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEEDS_ALL = [42, 7, 123, 2024, 31]
SEEDS = [int(a) for a in sys.argv[1:]] or SEEDS_ALL
OUT = "repeated_runs"
os.makedirs(OUT, exist_ok=True)

# ---- preprocessing: identical to main.py, done once (it does not depend on the split) ----
df = pd.read_csv("data/application_train.csv")
df['DAYS_EMPLOYED_ANOMALY'] = df['DAYS_EMPLOYED'] == 365243
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df = df[df['CODE_GENDER'] != 'XNA']
df = df.loc[:, df.isnull().mean() < 0.5]
for col in df.columns[df.isnull().any()].tolist():
    if df[col].dtype != 'object':
        df[col + '_MISSING'] = df[col].isnull().astype(int)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])
df = df.copy()
X = df.drop(columns=['TARGET'])
y = df['TARGET']
cat = X.select_dtypes(include='str').columns.tolist()
X = pd.get_dummies(X, columns=cat, drop_first=True)
X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in X.columns]
print("Feature table:", X.shape, flush=True)


def oof_preds(model_fn, Xtr, ytr, Xte, seed, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof, te = np.zeros(len(Xtr)), np.zeros(len(Xte))
    for tr, va in skf.split(Xtr, ytr):
        m = model_fn()
        m.fit(Xtr.iloc[tr], ytr.iloc[tr])
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
        te += m.predict_proba(Xte)[:, 1] / n_splits
    return oof, te


def run_seed(seed):
    path = f"{OUT}/seed{seed}.pkl"
    if os.path.exists(path):
        print(f"seed {seed}: already done, skipping", flush=True)
        return
    t0 = time.time()
    def lap(msg):
        print(f"seed {seed} | {msg} | {(time.time()-t0)/60:.1f} min", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler()
    Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=Xte.columns, index=Xte.index)

    mk_lgbm = lambda: LGBMClassifier(n_estimators=500, random_state=seed, verbose=-1)
    mk_xgb = lambda: XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=seed,
                                   eval_metric='logloss')
    mk_lr = lambda: LogisticRegression(max_iter=1000)

    # standalone: single fit on the full training set
    stand = {}
    stand["lr"] = mk_lr().fit(Xtr_s, ytr).predict_proba(Xte_s)[:, 1];   lap("standalone LR")
    stand["lgbm"] = mk_lgbm().fit(Xtr, ytr).predict_proba(Xte)[:, 1];   lap("standalone LightGBM")
    stand["xgb"] = mk_xgb().fit(Xtr, ytr).predict_proba(Xte)[:, 1];     lap("standalone XGBoost")

    # 5-fold OOF; test predictions = average of the 5 fold models ("bagged")
    oof, bag = {}, {}
    oof["lgbm"], bag["lgbm"] = oof_preds(mk_lgbm, Xtr, ytr, Xte, seed);       lap("OOF LightGBM")
    oof["xgb"], bag["xgb"] = oof_preds(mk_xgb, Xtr, ytr, Xte, seed);          lap("OOF XGBoost")
    oof["lr"], bag["lr"] = oof_preds(mk_lr, Xtr_s, ytr, Xte_s, seed);         lap("OOF LR")

    cols = ["lgbm", "xgb", "lr"]
    meta = LogisticRegression().fit(pd.DataFrame({k: oof[k] for k in cols}), ytr)
    stack = meta.predict_proba(pd.DataFrame({k: bag[k] for k in cols}))[:, 1]
    print(f"seed {seed} | stack AUC {roc_auc_score(yte, stack):.4f} | "
          f"meta weights {np.round(meta.coef_[0], 3)}", flush=True)

    if seed == 42 and os.path.exists("standalone_cache.pkl") and os.path.exists("stack_cache.pkl"):
        sa = pickle.load(open("standalone_cache.pkl", "rb"))
        sk = pickle.load(open("stack_cache.pkl", "rb"))
        print("CHECK idx_test identical to stack_cache:",
              np.array_equal(sk["idx_test"], Xte.index.to_numpy()))
        for k in cols:
            print(f"CHECK {k}: standalone max|diff| {np.abs(sa[k]-stand[k]).max():.2e}, "
                  f"bagged max|diff| {np.abs(sk['test'][k]-bag[k]).max():.2e}")

    pickle.dump({"seed": seed, "idx_test": Xte.index.to_numpy(), "y_test": yte.to_numpy(),
                 "standalone": stand, "bagged": bag, "stack": stack,
                 "meta_coef": meta.coef_[0], "minutes": (time.time() - t0) / 60},
                open(path, "wb"))
    lap("saved")


def summarize():
    rows = []
    for f in sorted(glob(f"{OUT}/seed*.pkl")):
        d = pickle.load(open(f, "rb"))
        sets = dict(d["standalone"])
        sets.update({k + "_bag": v for k, v in d["bagged"].items()})
        sets["stack"] = d["stack"]
        row = {"seed": d["seed"], "minutes": round(d["minutes"], 1)}
        for k, p in sets.items():
            row[f"{k}_auc"] = roc_auc_score(d["y_test"], p)
            row[f"{k}_pr"] = average_precision_score(d["y_test"], p)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv("repeated_runs_results.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nAUC by seed:\n", res[["seed", "minutes"] + [c for c in res.columns if c.endswith("_auc")]]
          .round(4).to_string(index=False))


for s in SEEDS:
    run_seed(s)
summarize()