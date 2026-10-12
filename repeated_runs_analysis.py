import numpy as np
import pandas as pd
from scipy import stats

r = pd.read_csv("repeated_runs_results.csv")
n = len(r)
models = ["lr", "lgbm", "xgb", "lgbm_bag", "xgb_bag", "stack"]
pd.set_option("display.width", 200)

for metric, tag in [("AUC", "_auc"), ("PR-AUC", "_pr")]:
    print(f"\n=== {metric} across {n} seeds ===")
    summ = pd.DataFrame({m: r[m + tag].agg(["mean", "std", "min", "max"]) for m in models}).T
    print(summ.round(4).to_string())

    print(f"\nPaired differences, stack minus model ({metric}):")
    rows = []
    for m in ["xgb", "xgb_bag", "lgbm", "lgbm_bag", "lr"]:
        d = r["stack" + tag] - r[m + tag]
        t = stats.ttest_1samp(d, 0.0) if n > 1 else None
        rows.append({"vs": m, "mean_diff": d.mean(), "std_diff": d.std(),
                     "stack_wins": f"{(d > 0).sum()}/{n}",
                     "t_pvalue": t.pvalue if t is not None else np.nan})
    print(pd.DataFrame(rows).round(4).to_string(index=False))