import pandas as pd

lc = pd.read_parquet("data/lc/lc_sub.parquet")
lc["TARGET"] = (lc["loan_status"] == "Charged Off").astype(int)
lc.to_parquet("data/lc/lc_sub.parquet")
print(lc.shape, lc["TARGET"].mean())
pd.set_option("display.max_rows", None)

inv = pd.DataFrame({
    "dtype": lc.dtypes.astype(str),
    "null_pct": lc.isna().mean().round(3),
    "n_unique": lc.nunique(),
})
print(inv)
print(lc["emp_length"].unique())
print(lc["term"].unique())
print(lc["home_ownership"].value_counts())
