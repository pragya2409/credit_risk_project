import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Current", "Late 1-30", "Late 31-60", "Late 61-90",
          "Late 91-120", "Default(5)", "Closed"]
ORDER = ["0", "1", "2", "3", "4", "5", "C", "X"]
K = 7

# Same split as common.py (same y, test_size, random_state)
app = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
_, idx_test = train_test_split(app.index, test_size=0.2, stratify=app.TARGET, random_state=42)
assert {134750, 187299, 45367} <= set(idx_test)      # showcase rows live in the holdout
hold_app = app.loc[idx_test, "SK_ID_CURR"]

bb = pd.read_csv("data/bureau_balance.csv",
                 dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"})
bb["s"] = bb["STATUS"].cat.set_categories(ORDER).cat.codes.astype("int8")
bb = bb.drop(columns="STATUS").sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"], ignore_index=True)
cid, mo, s = bb["SK_ID_BUREAU"].values, bb["MONTHS_BALANCE"].values, bb["s"].values
cur, nxt = s[:-1], s[1:]
ok = (cid[:-1] == cid[1:]) & ((mo[1:] - mo[:-1]) == 1) & (cur < K) & (nxt < K)
cur, nxt, credit = cur[ok].astype("int64"), nxt[ok].astype("int64"), cid[:-1][ok]
del bb

bur = pd.read_csv("data/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
hold_credits = bur.loc[bur.SK_ID_CURR.isin(hold_app), "SK_ID_BUREAU"].values
fit = ~np.isin(credit, hold_credits)
print(f"Dropping {(~fit).sum():,} of {len(fit):,} transitions (holdout applicants' credits)")
cur, nxt, credit = cur[fit], nxt[fit], credit[fit]

tmap = (bur.merge(app, on="SK_ID_CURR").drop_duplicates("SK_ID_BUREAU")
           .set_index("SK_ID_BUREAU")["TARGET"])
tgt = pd.Series(credit).map(tmap).values

def count(sel):
    c = np.bincount(cur[sel] * K + nxt[sel], minlength=K * K).reshape(K, K)
    return pd.DataFrame(c, index=LABELS, columns=LABELS)

pd.set_option("display.width", 220)
res = {}
for name, sel in {"all": np.ones(len(cur), bool), "target0": tgt == 0, "target1": tgt == 1}.items():
    c = count(sel); p = c.div(c.sum(axis=1), axis=0)
    c.to_csv(f"transition_counts_{name}_fit.csv"); p.to_csv(f"transition_matrix_{name}_fit.csv")
    res[name] = (c, p)
    print(f"{name}: {int(c.values.sum()):,} transitions; thin rows (<1000):",
          list(c.index[c.sum(axis=1) < 1000]))

cmp = pd.DataFrame({"non_defaulters": res["target0"][1].loc["Current", LABELS[1:6]].sum(),
                    "defaulters": res["target1"][1].loc["Current", LABELS[1:6]].sum()}, index=["Current -> any late"])
cmp["ratio"] = cmp.defaulters / cmp.non_defaulters
cmp.to_csv("transition_comparison_fit.csv")
print("\n", cmp.round(4).to_string())