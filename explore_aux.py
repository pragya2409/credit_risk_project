import numpy as np
import pandas as pd

# ---- bureau_balance: coverage + raw transitions ----
bb = pd.read_csv("data/bureau_balance.csv",
                 dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"})
bb = bb.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"])

months = bb.groupby("SK_ID_BUREAU", sort=False).size()
print("Credits with history:", len(months))
print("Months per credit:\n", months.describe().round(1).to_string(), "\n")

bur = pd.read_csv("data/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
has = bur.SK_ID_BUREAU.isin(months.index)
print(f"Bureau credits with monthly history: {has.sum()} of {len(bur)} ({100*has.mean():.0f}%)")
train_ids = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR"]).SK_ID_CURR
app_has = train_ids.isin(bur.loc[has, "SK_ID_CURR"])
print(f"Train applicants with >=1 credit with history: {app_has.sum()} of {len(train_ids)} ({100*app_has.mean():.0f}%)\n")

g = bb.groupby("SK_ID_BUREAU", sort=False)
bb["next_status"] = g["STATUS"].shift(-1)
bb["next_month"] = g["MONTHS_BALANCE"].shift(-1)
t = bb[(bb.next_month - bb.MONTHS_BALANCE) == 1]          # consecutive months only
counts = pd.crosstab(t.STATUS, t.next_status)
counts.to_csv("bb_transition_counts_raw.csv")
pd.set_option("display.width", 200)
print("Raw transition counts (rows = this month, cols = next month):\n", counts.to_string(), "\n")
print("Row-normalised:\n", counts.div(counts.sum(axis=1), axis=0).round(4).to_string(), "\n")
del bb, t

# ---- installments_payments: lateness and shortfall ----
ip = pd.read_csv("data/installments_payments.csv",
                 usecols=["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
                 dtype="float32")
late = ip.DAYS_ENTRY_PAYMENT - ip.DAYS_INSTALMENT
short = ip.AMT_INSTALMENT - ip.AMT_PAYMENT
print("Installments:", len(ip), "rows; missing payment date:", ip.DAYS_ENTRY_PAYMENT.isna().mean().round(4))
print(f"  paid late (>0 days):  {(late > 0).mean():.3f}")
print(f"  paid >30 days late:   {(late > 30).mean():.3f}")
print(f"  underpaid (>1 unit):  {(short > 1).mean():.3f}")
print("  days late quantiles:", late.quantile([.5, .9, .99]).to_dict())