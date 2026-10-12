import numpy as np
import pandas as pd

STATES = ["0", "1", "2", "3", "4", "5", "C"]
LABELS = ["Current", "Late 1-30", "Late 31-60", "Late 61-90",
          "Late 91-120", "Default(5)", "Closed"]
ORDER = STATES + ["X"]            # X gets code 7 and is dropped below
K = len(STATES)

# ---- 1. State sequences (compact dtypes, 27M rows) ----
bb = pd.read_csv("data/bureau_balance.csv",
                 dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16",
                        "STATUS": "category"})
bb["s"] = bb["STATUS"].cat.set_categories(ORDER).cat.codes.astype("int8")
bb = bb.drop(columns="STATUS").sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"],
                                           ignore_index=True)

cid = bb["SK_ID_BUREAU"].values
mo = bb["MONTHS_BALANCE"].values
s = bb["s"].values
cur, nxt = s[:-1], s[1:]
# same credit, consecutive months (MONTHS_BALANCE rises by 1), neither end is X
ok = (cid[:-1] == cid[1:]) & ((mo[1:] - mo[:-1]) == 1) & (cur < K) & (nxt < K)
print(f"Rows: {len(bb):,}; usable transitions: {ok.sum():,}")
cur = cur[ok].astype("int64")
nxt = nxt[ok].astype("int64")
credit = cid[:-1][ok]
del bb

# ---- 2. Link credit -> applicant -> TARGET ----
bur = pd.read_csv("data/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
tr = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
tmap = (bur.merge(tr, on="SK_ID_CURR", how="inner")
           .drop_duplicates("SK_ID_BUREAU")
           .set_index("SK_ID_BUREAU")["TARGET"])
tgt = pd.Series(credit).map(tmap).values       # NaN if applicant is not in train
print(f"Transitions with a labeled applicant: {np.isfinite(tgt).sum():,} "
      f"({100 * np.isfinite(tgt).mean():.0f}%)")

# ---- 3. Count and normalise ----
def count(sel):
    c = np.bincount(cur[sel] * K + nxt[sel], minlength=K * K).reshape(K, K)
    return pd.DataFrame(c, index=LABELS, columns=LABELS)

groups = {"all": np.ones(len(cur), bool), "target0": tgt == 0, "target1": tgt == 1}
res = {}
pd.set_option("display.width", 220)
for name, sel in groups.items():
    c = count(sel)
    p = c.div(c.sum(axis=1), axis=0)
    c.to_csv(f"transition_counts_{name}.csv")
    p.to_csv(f"transition_matrix_{name}.csv")
    res[name] = (c, p)
    n = c.sum(axis=1).astype(int)
    print(f"\n=== {name}: {int(c.values.sum()):,} transitions, "
          f"{np.unique(credit[sel]).size:,} credits ===")
    print(pd.DataFrame({"n_from": n, "thin(<1000)": n < 1000}).to_string())
    print(p.round(4).to_string())

# ---- 4. Defaulters vs non-defaulters ----
pairs = [("Current", "Late 1-30"), ("Late 1-30", "Current"),
         ("Late 1-30", "Late 31-60"), ("Late 31-60", "Late 61-90"),
         ("Late 61-90", "Late 91-120"), ("Late 91-120", "Default(5)"),
         ("Default(5)", "Default(5)")]
late = LABELS[1:6]
rows = []
for a, b in pairs:
    p0, p1 = res["target0"][1].loc[a, b], res["target1"][1].loc[a, b]
    rows.append({"transition": f"{a} -> {b}", "non_defaulters": p0,
                 "defaulters": p1, "ratio": p1 / p0,
                 "n_from_defaulters": int(res["target1"][0].loc[a].sum())})
p0 = res["target0"][1].loc["Current", late].sum()
p1 = res["target1"][1].loc["Current", late].sum()
rows.append({"transition": "Current -> any late", "non_defaulters": p0,
             "defaulters": p1, "ratio": p1 / p0,
             "n_from_defaulters": int(res["target1"][0].loc["Current"].sum())})
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv("transition_comparison.csv", index=False)
print("\nDefaulters vs non-defaulters:\n", cmp_df.round(4).to_string(index=False))

