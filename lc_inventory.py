import pandas as pd
import numpy as np

# Candidate columns only (avoids loading all ~150 columns of a 2M-row file)
use = ["loan_amnt", "term", "installment", "emp_length", "home_ownership",
       "annual_inc", "verification_status", "purpose", "dti",
       "earliest_cr_line", "open_acc", "total_acc", "revol_bal", "revol_util",
       "delinq_2yrs", "pub_rec", "addr_state", "issue_d", "loan_status"]

lc = pd.read_csv("data/lc/accepted_2007_to_2018Q4.csv.gz", usecols=use, engine="pyarrow")

# Resolved loans only, binary target (1 = default)
lc = lc[lc["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
lc["TARGET"] = (lc["loan_status"] == "Charged Off").astype(int)

# Stratified subsample ~250K
lc = lc.sample(n=min(250_000, len(lc)), random_state=42)

lc.to_parquet("data/lc/lc_sub.parquet")
print(lc.shape, lc["TARGET"].mean())

inv = pd.DataFrame({
    "dtype": lc.dtypes,
    "null_pct": lc.isna().mean().round(3),
    "n_unique": lc.nunique(),
})
print(inv)
print(lc["emp_length"].unique())   # check the encoding you'll have to harmonize
