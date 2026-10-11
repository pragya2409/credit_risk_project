import json
import numpy as np
import pandas as pd

df = pd.read_csv("data/upi/upi_transactions_2024.csv", parse_dates=["timestamp"])
amt = df["amount (INR)"]
out = {}

# 1. Amount distribution (spending-shock size)
la = np.log(amt)
out["log_amount_mean"], out["log_amount_std"] = float(la.mean()), float(la.std())
print("Amount quantiles:", amt.quantile([.1, .5, .9, .99]).to_dict())
print(f"log-amount mean {la.mean():.3f}, std {la.std():.3f}, skew {la.skew():.2f}")

# 2. Category mix
mix = df.groupby("merchant_category")["amount (INR)"].agg(["size", "mean"])
mix["share"] = mix["size"] / len(df)
print("\nCategory mix:\n", mix.sort_values("share", ascending=False).round(3).to_string())
out["category_share"] = mix["share"].round(4).to_dict()

# 3. Monthly volatility of total spend
m = df.set_index("timestamp").resample("MS")["amount (INR)"].agg(["size", "sum"])
m["sum_change_pct"] = 100 * m["sum"].pct_change()
print("\nMonthly totals:\n", m.round(1).to_string())
out["monthly_spend_change_std_pct"] = float(m["sum_change_pct"].std())
out["monthly_spend_cv"] = float(m["sum"].std() / m["sum"].mean())

# 4. Failure rate
fail = (df.transaction_status != "SUCCESS").mean()
print(f"\nNon-success share: {fail:.4f}")
out["failure_rate"] = float(fail)

# 5. Quick check on how 'real' it looks: is the data flat across categories/hours?
print("\nShare by hour (min/max):", df.hour_of_day.value_counts(normalize=True).agg(["min", "max"]).round(4).to_dict())
print("Mean amount by category (min/max):", mix["mean"].agg(["min", "max"]).round(1).to_dict())

json.dump(out, open("upi_params.json", "w"), indent=2)