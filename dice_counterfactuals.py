import warnings
import numpy as np
import pandas as pd
import dice_ml
from common import load_model_and_data

warnings.filterwarnings("ignore")

N_RANDOM = 30
FLOOR = 0.5            # max 50% reduction
SHOWCASE = [134750, 187299, 45367]
VARY = ["AMT_CREDIT", "AMT_ANNUITY"]

model, X_train, X_test, y_train, y_test = load_model_and_data()
p_test = model.predict_proba(X_test)[:, 1]
cutoff = np.quantile(p_test, 0.80)

class CutoffModel:
    def __init__(self, m, cutoff):
        self.m, self.shift = m, np.log(cutoff / (1 - cutoff))
        self.nan_cols = []                       # set per applicant
    def _restore(self, X):
        X = X.copy()
        if self.nan_cols:
            X[self.nan_cols] = np.nan
        return X
    def predict_proba(self, X):
        p = np.clip(self.m.predict_proba(self._restore(X))[:, 1], 1e-6, 1 - 1e-6)
        q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) - self.shift)))
        return np.column_stack([1 - q, q])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

wrapped = CutoffModel(model, cutoff)
med = X_train.median()
sample = X_train.sample(5000, random_state=42).fillna(med).astype(float)
data_df = sample.copy()
data_df["outcome"] = wrapped.predict(sample)   # nan_cols empty here: background data as filled

d = dice_ml.Data(dataframe=data_df, continuous_features=list(sample.columns),
                 outcome_name="outcome")
m = dice_ml.Model(model=wrapped, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")

rejected_idx = X_test.index[p_test > cutoff]
rng = np.random.default_rng(42)
pool = [i for i in rejected_idx if i not in SHOWCASE]
chosen = SHOWCASE + list(rng.choice(pool, size=N_RANDOM, replace=False))

rows, summary = [], []
for n, idx in enumerate(chosen, 1):
    raw = X_test.loc[[idx]]
    wrapped.nan_cols = list(raw.columns[raw.isna().iloc[0]])
    query = raw.fillna(med).astype(float)
    r = query.iloc[0]
    p0 = model.predict_proba(raw)[0, 1]
    assert p0 > cutoff, f"row {idx} is not actually rejected (p={p0:.3f})"

    permitted = {f: [r[f] * FLOOR, r[f]] for f in VARY}
    cfs = None
    try:
        cf = exp.generate_counterfactuals(
            query, total_CFs=3, desired_class=0,
            features_to_vary=VARY, permitted_range=permitted, random_seed=42)
        cfs = cf.cf_examples_list[0].final_cfs_df
    except Exception as e:
        print(f"[{n}/{len(chosen)}] row {idx}: no CF ({type(e).__name__})")

    found = 0
    if cfs is not None and len(cfs):
        cfs = cfs.drop(columns=["outcome"], errors="ignore")
        chk = cfs.copy()
        chk[wrapped.nan_cols] = np.nan
        p_new = model.predict_proba(chk)[:, 1]
        for k, (_, c) in enumerate(cfs.iterrows()):
            if p_new[k] >= cutoff:
                continue
            found += 1
            rows.append({
                "row_id": idx, "p_original": p0, "cf_id": k, "p_new": p_new[k],
                "credit_orig": r["AMT_CREDIT"], "credit_cf": c["AMT_CREDIT"],
                "credit_change_pct": 100 * (c["AMT_CREDIT"] / r["AMT_CREDIT"] - 1),
                "annuity_orig": r["AMT_ANNUITY"], "annuity_cf": c["AMT_ANNUITY"],
                "annuity_change_pct": 100 * (c["AMT_ANNUITY"] / r["AMT_ANNUITY"] - 1),
            })
    summary.append({"row_id": idx, "p_original": p0, "n_valid_cfs": found})
    print(f"[{n}/{len(chosen)}] row {idx}: p={p0:.3f}, valid CFs={found}")

pd.DataFrame(rows).to_csv("counterfactuals.csv", index=False)
s = pd.DataFrame(summary)
s.to_csv("counterfactuals_summary.csv", index=False)
print("\nApplicants with >=1 valid CF:", (s.n_valid_cfs > 0).sum(), "of", len(s))
print(s.sort_values("p_original").to_string(index=False))