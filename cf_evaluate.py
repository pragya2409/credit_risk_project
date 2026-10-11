import numpy as np
import pandas as pd

c = pd.read_csv("counterfactuals.csv")
man = pd.read_csv("manual_perturbation.csv")
tr = pd.read_csv("data/application_train.csv", usecols=["AMT_CREDIT", "AMT_ANNUITY"])

# ---- Plausibility bound: implied loan term (months) = credit / annuity ----
term_tr = (tr.AMT_CREDIT / tr.AMT_ANNUITY).dropna()
lo, hi = term_tr.quantile([0.01, 0.99])
print(f"Implied term in training data: 1st pct {lo:.1f} mo, 99th pct {hi:.1f} mo\n")

# ---- DiCE: sparsity and plausibility per counterfactual ----
c["n_changed"] = ((c.credit_change_pct.abs() > 0.01).astype(int)
                  + (c.annuity_change_pct.abs() > 0.01).astype(int))
c["term_orig"] = c.credit_orig / c.annuity_orig
c["term_cf"] = c.credit_cf / c.annuity_cf
c["plausible"] = c.term_cf <= hi

print("DiCE sparsity (features changed per counterfactual):")
print(c.n_changed.value_counts().sort_index().to_string(), "\n")
print(f"DiCE counterfactuals with plausible implied term: "
      f"{c.plausible.sum()} of {len(c)} ({100*c.plausible.mean():.0f}%)")
bad = c[~c.plausible]
if len(bad):
    print(bad[["row_id", "credit_change_pct", "annuity_change_pct",
               "term_orig", "term_cf"]].round(1).to_string(index=False))

# ---- Manual: annuity-only is the one strategy that changes the implied term ----
ao = man.dropna(subset=["annuity_only"]).copy()
# need original term per applicant: reuse from the CF file
orig_term = c.drop_duplicates("row_id").set_index("row_id").term_orig
ao["term_cf"] = ao.row_id.map(orig_term) / (1 + ao.annuity_only / 100)
ao["plausible"] = ao.term_cf.between(lo, hi)
print(f"\nManual annuity-only flips with plausible implied term: "
      f"{ao.plausible.sum()} of {len(ao)}")

# ---- Proximity: smallest change per applicant, both methods ----
man["manual_best"] = man[["credit_only", "annuity_only", "both"]].max(axis=1)  # closest to 0
both = man.dropna(subset=["manual_best", "dice_best"])
print(f"\nProximity on the {len(both)} applicants both methods flip:")
print(f"  manual median smallest cut: {both.manual_best.median():.1f}%")
print(f"  DiCE   median smallest cut: {both.dice_best.median():.1f}%")
print(f"  DiCE larger than manual by >5 pts: {((both.manual_best - both.dice_best) > 5).sum()} of {len(both)}")

# ---- Eyeball table: 10 applicants ----
print("\nExamples (flippable applicants, sorted by p_original):")
ex = man.dropna(subset=["manual_best"]).sort_values("p_original").iloc[::2].head(10)
print(ex[["row_id", "p_original", "credit_only", "annuity_only", "both", "dice_best"]]
      .round(3).to_string(index=False))


print(ao[~ao.plausible][["row_id", "annuity_only", "term_cf"]].round(1).to_string(index=False))