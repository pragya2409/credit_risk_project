import pandas as pd

# Accepted loans (the file Week 7 used): read only 5 rows
acc = pd.read_csv("data/lc/accepted_2007_to_2018Q4.csv.gz", nrows=5, low_memory=False)
print("ACCEPTED:", acc.shape[1], "columns")
print(list(acc.columns), "\n")

# Rejected applications
rej = pd.read_csv("data/lc/rejected_2007_to_2018Q4.csv.gz", nrows=5, low_memory=False)
print("REJECTED:", rej.shape[1], "columns")
print(list(rej.columns), "\n")

# The subset you already saved in Week 7
sub = pd.read_parquet("data/lc/lc_sub.parquet")
print("lc_sub.parquet:", sub.shape)
print(list(sub.columns), "\n")

# Your own column mapping between the two datasets
pd.set_option("display.width", 200)
print(pd.read_csv("feature_mapping.csv").to_string(index=False))