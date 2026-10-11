import pandas as pd

s = pd.read_csv("counterfactuals_summary.csv")
c = pd.read_csv("counterfactuals.csv")

s["has_cf"] = s.n_valid_cfs > 0
print(f"Validity: {s.has_cf.sum()} of {len(s)} ({100*s.has_cf.mean():.0f}%)\n")

s["band"] = pd.cut(s.p_original, [0.118, 0.15, 0.20, 1.0])
print(s.groupby("band", observed=True).has_cf.agg(["sum", "count", "mean"]), "\n")

best = c.loc[c.groupby("row_id").credit_change_pct.idxmax()]   # smallest cut per applicant
print("Smallest DiCE cut per applicant (credit change %):")
print(best.credit_change_pct.describe().round(1), "\n")
print("Annuity ever changed:", (c.annuity_change_pct.abs() > 0.01).any())
print("Highest p_original with a valid CF:", s[s.has_cf].p_original.max().round(3))
print("Lowest p_original with no CF:", s[~s.has_cf].p_original.min().round(3))

per = c.groupby("row_id").credit_change_pct
annuity_only_cf = (per.max().abs() < 0.01)    # at least one CF leaves credit unchanged
all_annuity_only = (per.min().abs() < 0.01)   # every CF leaves credit unchanged
print("\nApplicants with >=1 annuity-only counterfactual:", annuity_only_cf.sum(), "of", c.row_id.nunique())
print("Applicants whose counterfactuals are ALL annuity-only:", all_annuity_only.sum())
