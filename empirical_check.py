import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

W = 24
ORDER = ["0", "1", "2", "3", "4", "5", "C", "X"]
app = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
_, idx_test = train_test_split(app.index, test_size=0.2, stratify=app.TARGET, random_state=42)
assert {134750, 187299, 45367} <= set(idx_test)
hold = app.loc[idx_test]

bur = pd.read_csv("data/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
bur = bur.merge(hold, on="SK_ID_CURR").drop_duplicates("SK_ID_BUREAU")
tmap = bur.set_index("SK_ID_BUREAU")["TARGET"]

bb = pd.read_csv("data/bureau_balance.csv",
                 dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"})
bb = bb[bb.SK_ID_BUREAU.isin(tmap.index)]
bb["s"] = bb["STATUS"].cat.set_categories(ORDER).cat.codes.astype("int8")
bb = bb.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"], ignore_index=True)
cid = bb.SK_ID_BUREAU.values
mo = bb.MONTHS_BALANCE.values.astype("int32")
s = bb.s.values
n = len(bb)

def prefix(flag):
    return np.concatenate([[0], np.cumsum(flag, dtype="int32")])
PX, PL, P5 = prefix(s == 7), prefix((s >= 1) & (s <= 5)), prefix(s == 5)

# Start: a Current month followed by W consecutive observed months with no X
i = np.arange(n - W)
ok = (cid[i + W] == cid[i]) & (mo[i + W] - mo[i] == W) & (s[i] == 0) \
     & ((PX[i + W + 1] - PX[i + 1]) == 0)
elig = i[ok]
_, first = np.unique(cid[elig], return_index=True)     # first eligible start per credit
start = elig[first]

late_n = PL[start + W + 1] - PL[start + 1]
d = pd.DataFrame({
    "ever_late": late_n > 0,
    "reach_default": (P5[start + W + 1] - P5[start + 1]) > 0,
    "closed_at_end": s[start + W] == 6,
    "late_months": late_n,
    "TARGET": tmap.reindex(cid[start]).values,
})
cols = ["ever_late", "reach_default", "closed_at_end", "late_months"]
print(f"Holdout applicants' credits with a clean {W}-month window: {len(d):,}")
print("\nEmpirical (all):\n", d[cols].mean().round(4).to_string())
print("\nEmpirical by TARGET:\n", d.groupby("TARGET")[cols].mean().round(4).to_string())

print("\nSimulated (overall means):")
for f in ["simulation_results_fit_seed42.csv", "simulation_results_fit_seed7.csv"]:
    if os.path.exists(f):
        r = pd.read_csv(f)
        print(f, r[["p_ever_late", "p_reach_default", "p_closed_at_end", "mean_late_months"]]
              .mean().round(4).to_dict())