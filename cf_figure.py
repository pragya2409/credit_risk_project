import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import load_model_and_data

model, X_train, X_test, y_train, y_test = load_model_and_data()
cutoff = np.quantile(model.predict_proba(X_test)[:, 1], 0.80)
man = pd.read_csv("manual_perturbation.csv").set_index("row_id")

ids = [187299, 48060, 207380]
rows = []
for i in ids:
    raw = X_test.loc[[i]]                      # real row, NaNs intact
    pct = man.loc[i, "both"]
    new = raw.copy()
    for col in ["AMT_CREDIT", "AMT_ANNUITY"]:
        new[col] = raw[col].values * (1 + pct / 100)
    rows.append({
        "row_id": i, "cut_pct": pct,
        "p_before": model.predict_proba(raw)[0, 1],
        "p_after": model.predict_proba(new)[0, 1],
        "credit_before": raw["AMT_CREDIT"].iloc[0],
        "credit_after": new["AMT_CREDIT"].iloc[0],
    })
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
r.to_csv("cf_figure_data.csv", index=False)

labels = [f"Applicant {i}\n({c:+.1f}%)" for i, c in zip(r.row_id, r.cut_pct)]
x = np.arange(len(r)); w = 0.35
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].bar(x - w/2, r.p_before, w, label="Before")
ax[0].bar(x + w/2, r.p_after, w, label="After")
ax[0].axhline(cutoff, color="k", ls="--", lw=1, label=f"Rejection cutoff ({cutoff:.3f})")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("Predicted default probability")
ax[0].set_ylim(0.10, 0.15)
ax[0].legend(loc="upper center", ncol=3, fontsize=8)
ax[0].set_title("Decision flips below the cutoff (y-axis starts at 0.10)")

ax[1].bar(x - w/2, r.credit_before / 1000, w, label="Before")
ax[1].bar(x + w/2, r.credit_after / 1000, w, label="After")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("Loan amount (thousands)"); ax[1].legend()
ax[1].set_title("Loan reduction needed")

plt.tight_layout()
plt.savefig("counterfactual_before_after.png", dpi=200)