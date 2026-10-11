import numpy as np
import pandas as pd
import dice_ml
from common import load_model_and_data

model, X_train, X_test, y_train, y_test = load_model_and_data()

p_test = model.predict_proba(X_test)[:, 1]
cutoff = np.quantile(p_test, 0.80)

class CutoffModel:
    def __init__(self, m, cutoff):
        self.m, self.shift = m, np.log(cutoff / (1 - cutoff))
    def predict_proba(self, X):
        p = np.clip(self.m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) - self.shift)))
        return np.column_stack([1 - q, q])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

wrapped = CutoffModel(model, cutoff)

med = X_train.median()
sample = X_train.sample(5000, random_state=42).fillna(med).astype(float)
data_df = sample.copy()
data_df["outcome"] = wrapped.predict(sample)

d = dice_ml.Data(dataframe=data_df,
                 continuous_features=list(sample.columns),
                 outcome_name="outcome")
m = dice_ml.Model(model=wrapped, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")

query = X_test.loc[[187299]].fillna(med).astype(float)
row = query.iloc[0]
print("p(default):", model.predict_proba(query)[0, 1], "cutoff:", cutoff)

floor = 0.5   # max 50% reduction
permitted = {
    "AMT_CREDIT":  [row["AMT_CREDIT"]  * floor, row["AMT_CREDIT"]],
    "AMT_ANNUITY": [row["AMT_ANNUITY"] * floor, row["AMT_ANNUITY"]],
}

cf = exp.generate_counterfactuals(
    query, total_CFs=3, desired_class=0,
    features_to_vary=["AMT_CREDIT", "AMT_ANNUITY"],
    permitted_range=permitted,
    random_seed=42,
)

cfs = cf.cf_examples_list[0].final_cfs_df.drop(columns=["outcome"], errors="ignore")
print("Counterfactuals found:", len(cfs))
if len(cfs):
    print("New p(default):", model.predict_proba(cfs)[:, 1], "cutoff:", cutoff)
    changed = [c for c in cfs.columns if (cfs[c].values != row[c]).any()]
    pd.set_option("display.width", 200)
    print("Original:\n", query[changed].to_string(index=False))
    print("Counterfactuals:\n", cfs[changed].to_string(index=False))