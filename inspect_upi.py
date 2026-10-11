import pandas as pd

df = pd.read_csv("data/upi/upi_transactions_2024.csv")
pd.set_option("display.width", 220)

print(df.shape)
print(df.dtypes.to_string(), "\n")
print(df.head(5).to_string(), "\n")
print("Missing share (columns with any):")
print(df.isna().mean().round(3)[lambda s: s > 0].to_string(), "\n")

print("Unique values per column:")
print(df.nunique().sort_values().to_string(), "\n")

# date range, if there is a date-like column
for c in df.columns:
    if any(k in c.lower() for k in ["date", "time", "timestamp"]):
        t = pd.to_datetime(df[c], errors="coerce")
        print(f"{c}: {t.min()} to {t.max()} ({t.notna().mean():.2f} parsed)")

num = df.select_dtypes("number")
print("\nNumeric summary:")
print(num.describe().round(2).T.to_string())