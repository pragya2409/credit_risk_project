import os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATS = ["log_income", "loan_to_income", "payment_to_income", "emp_years", "owns_property"]
PCT = ["log_income", "loan_to_income", "payment_to_income", "emp_years"]  # owns_property stays binary

def to_pct(df):
    for c in PCT:
        df[c] = df[c].rank(pct=True)   # NaNs stay NaN
    return df

# ---------- LendingClub: preprocess ----------
lc = pd.read_parquet("data/lc/lc_sub.parquet")
lc = lc[~lc["home_ownership"].isin(["ANY", "OTHER", "NONE"])]
lc = lc[lc["annual_inc"] > 0].copy()

emp = lc["emp_length"].str.extract(r"(\d+)")[0].astype(float)
emp[lc["emp_length"].str.startswith("<", na=False)] = 0
lc["emp_years"] = emp.clip(upper=10)

lc["log_income"] = np.log(lc["annual_inc"])
lc["loan_to_income"] = lc["loan_amnt"] / lc["annual_inc"]
lc["payment_to_income"] = lc["installment"] * 12 / lc["annual_inc"]
lc["owns_property"] = lc["home_ownership"].isin(["OWN", "MORTGAGE"]).astype(int)
lc = to_pct(lc)                                   # <-- BEFORE the split

# ---------- LendingClub: own split + model ----------
X, y = lc[FEATS], lc["TARGET"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

model = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False)])

lc_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
print(f"LC standalone AUC (LC test): {lc_auc:.4f}  | best_iter: {model.best_iteration_}")

# ---------- Home Credit: same 5 features ----------
cols = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_EMPLOYED", "FLAG_OWN_REALTY"]
hc = pd.read_csv("data/application_train.csv", usecols=cols)

inc = hc["AMT_INCOME_TOTAL"].where(hc["AMT_INCOME_TOTAL"] > 0)
hc["log_income"] = np.log(inc)
hc["loan_to_income"] = hc["AMT_CREDIT"] / inc
hc["payment_to_income"] = hc["AMT_ANNUITY"] * 12 / inc
days = hc["DAYS_EMPLOYED"].replace(365243, np.nan)
hc["emp_years"] = (-days / 365).clip(0, 10)
hc["owns_property"] = (hc["FLAG_OWN_REALTY"] == "Y").astype(int)
hc = to_pct(hc)                                   # <-- BEFORE the medians print and scoring

print("\nMedians (rows: LC, HC):")
print(pd.DataFrame({"LC": lc[FEATS].median(), "HC": hc[FEATS].median()}).T.round(3))

# ---------- Score HC with the LC model ----------
hc["lc_score"] = model.predict_proba(hc[FEATS])[:, 1]
hc_auc = roc_auc_score(hc["TARGET"], hc["lc_score"])
print(f"\nLC model AUC on Home Credit (cross-domain): {hc_auc:.4f}")

os.makedirs("data/lc", exist_ok=True)
hc[["SK_ID_CURR", "lc_score"]].to_csv("lc_scores_hc.csv", index=False)
model.booster_.save_model("data/lc/lc_lgbm.txt")
pd.DataFrame([{"lc_standalone_auc": lc_auc, "lc_on_hc_auc": hc_auc,
               "best_iter": model.best_iteration_, "n_lc_rows": len(lc)}]
             ).to_csv("lc_transfer_summary.csv", index=False)
# Univariate AUC per feature in each domain (0.5 = no signal; <0.5 = reversed direction)
rows = []
for f in FEATS:
    a_lc = roc_auc_score(lc["TARGET"], lc[f].fillna(lc[f].median()))
    a_hc = roc_auc_score(hc["TARGET"], hc[f].fillna(hc[f].median()))
    rows.append([f, round(a_lc, 3), round(a_hc, 3)])
print("\nUnivariate AUC:")
print(pd.DataFrame(rows, columns=["feature", "LC", "HC"]))

# In-domain ceiling: same 5 features, HC model, HC split
Xh, yh = hc[FEATS], hc["TARGET"]
Xa, Xb, ya, yb = train_test_split(Xh, yh, test_size=0.2, stratify=yh, random_state=42)
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                       min_child_samples=200, random_state=42, verbose=-1).fit(Xa, ya)
print(f"\nHC in-domain AUC, same 5 features: {roc_auc_score(yb, m.predict_proba(Xb)[:, 1]):.4f}")