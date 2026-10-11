import pandas as pd

for name in ["bureau_balance", "bureau", "installments_payments"]:
    df = pd.read_csv(f"data/{name}.csv", nrows=200_000)
    print(f"\n=== {name} ===")
    print(df.shape)
    print(df.head(3).to_string())
    print(df.isna().mean().round(3)[lambda s: s > 0].to_string())

bb = pd.read_csv("data/bureau_balance.csv", usecols=["STATUS"])
print("\nbureau_balance STATUS counts:")
print(bb.STATUS.value_counts().to_string())