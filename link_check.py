import pandas as pd
ids = pd.read_csv("data/bureau_balance.csv", usecols=["SK_ID_BUREAU"]).SK_ID_BUREAU.unique()
bur = pd.read_csv("data/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
tr = pd.read_csv("data/application_train.csv", usecols=["SK_ID_CURR"]).SK_ID_CURR
te = pd.read_csv("data/application_test.csv", usecols=["SK_ID_CURR"]).SK_ID_CURR
b = pd.DataFrame({"SK_ID_BUREAU": ids}).merge(bur, on="SK_ID_BUREAU", how="left")
print("credits with history:", len(b))
print("not in bureau.csv:", b.SK_ID_CURR.isna().mean().round(3))
print("applicant in train:", b.SK_ID_CURR.isin(tr).mean().round(3))
print("applicant in test: ", b.SK_ID_CURR.isin(te).mean().round(3))