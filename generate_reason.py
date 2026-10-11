# generate_reasons.py
from collections import Counter
import numpy as np
import pandas as pd
import shap
from common import load_model_and_data
from reasons_config import EXCLUDED, REASONS
from collections import defaultdict
import matplotlib.pyplot as plt
from reasons_config import is_excluded, REASONS, PROTECTED

model, X_train, X_test, y_train, y_test = load_model_and_data()

# Reject the riskiest 20% (80% approval rate)
p = model.predict_proba(X_test)[:, 1]
cutoff = np.quantile(p, 0.80)
mask = p > cutoff
rejected = X_test[mask]
print(f"Rejected: {len(rejected)} applicants (cutoff p={cutoff:.3f})")

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(rejected)
if isinstance(sv, list):
    sv = sv[1]
sv = pd.DataFrame(sv, index=rejected.index, columns=rejected.columns)

unmapped = Counter()

MIN_SHAP = 0.02

def top_reasons(row, k=4):
    groups = defaultdict(float)
    for feat, val in row.items():
        if is_excluded(feat) :
            continue
        if feat not in REASONS:
            if val > MIN_SHAP:
                unmapped[feat] += 1
            continue
        groups[REASONS[feat]] += val  # group-level SHAP
    ranked = sorted(groups.items(), key=lambda kv: -kv[1])
    return [r for r, v in ranked if v > MIN_SHAP][:k]

reasons = sv.apply(top_reasons, axis=1)

res = pd.DataFrame({
    "p_default": p[mask],
    "reasons": reasons.apply(" | ".join),
    "n_reasons": reasons.apply(len),
}, index=rejected.index)
res.to_csv("reasons.csv")

print(res["n_reasons"].value_counts().sort_index())
print("\nUnmapped features seen as positive drivers:")
print(unmapped.most_common(10))
print(res.head(10).to_string())

from itertools import chain
freq = pd.Series(list(chain.from_iterable(reasons))).value_counts() / len(res)
print("\nReason frequency (share of rejected applicants):")
print(freq.round(3).to_string())

#waterfall plot
base = explainer.expected_value
base = base[1] if np.ndim(base) > 0 else base

picks = {
    "clear_cut": res["p_default"].idxmax(),
    "borderline": (res["p_default"] - cutoff).abs().idxmin(),
    "typical": (res["p_default"] - res["p_default"].median()).abs().idxmin(),
}
for name, idx in picks.items():
    exp = shap.Explanation(values=sv.loc[idx].values, base_values=base,
                           data=rejected.loc[idx].values,
                           feature_names=list(rejected.columns))
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(f"waterfall_{name}.png", dpi=200)
    plt.close()
    print(f"\n{name} (row {idx}, p={res.loc[idx, 'p_default']:.3f}):\n  {res.loc[idx, 'reasons']}")

from reasons_config import PROTECTED

top3 = sv.apply(lambda r: set(r[r > 0].nlargest(3).index), axis=1)
print("\nProtected attribute in top-3 positive drivers (unfiltered):")
for f in sorted(PROTECTED):
    print(f"  {f}: {top3.apply(lambda s: f in s).mean():.3f}")
print("  any protected:", round(top3.apply(lambda s: bool(s & PROTECTED)).mean(), 3))
