import numpy as np
import pandas as pd
from common import load_model_and_data

model, X_train, X_test, y_train, y_test = load_model_and_data()
cutoff = np.quantile(model.predict_proba(X_test)[:, 1], 0.80)

s = pd.read_csv("counterfactuals_summary.csv")
c = pd.read_csv("counterfactuals.csv")
scales = np.arange(1.0, 0.4999, -0.005)        # 0% to -50% in 0.5% steps

def first_flip(raw, cols):
    base = raw.iloc[0]
    batch = pd.concat([raw] * len(scales), ignore_index=True)
    for col in cols:
        batch[col] = base[col] * scales
    p = model.predict_proba(batch)[:, 1]
    ok = np.where(p < cutoff)[0]
    return 100 * (scales[ok[0]] - 1) if len(ok) else np.nan

rows = []
for idx in s.row_id:
    raw = X_test.loc[[idx]]                     # real row, NaNs intact
    rows.append({
        "row_id": idx,
        "p_original": model.predict_proba(raw)[0, 1],
        "credit_only": first_flip(raw, ["AMT_CREDIT"]),
        "annuity_only": first_flip(raw, ["AMT_ANNUITY"]),
        "both": first_flip(raw, ["AMT_CREDIT", "AMT_ANNUITY"]),
    })
man = pd.DataFrame(rows)

# DiCE: smallest "largest single change" across its valid counterfactuals
c["dice_max_change"] = -c[["credit_change_pct", "annuity_change_pct"]].abs().max(axis=1)
dice = c.groupby("row_id").dice_max_change.max().rename("dice_best")  # closest to 0
out = (man.merge(s[["row_id", "n_valid_cfs"]], on="row_id")
          .merge(dice, on="row_id", how="left")
          .sort_values("p_original"))
out.to_csv("manual_perturbation.csv", index=False)

pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))

man_any = out[["credit_only", "annuity_only", "both"]].notna().any(axis=1)
print("\nFlippable by manual search (any strategy):", man_any.sum(), "of", len(out))
for k in ["credit_only", "annuity_only", "both"]:
    print(f"  {k}: {out[k].notna().sum()} flippable, median cut {out[k].median():.1f}%")
print("DiCE found none but manual flips:",
      ((out.n_valid_cfs == 0) & man_any).sum())
print("Highest p_original manual can flip:",
      out[man_any].p_original.max().round(3))