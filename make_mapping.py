import pandas as pd

mapping = pd.DataFrame([
    ["log_income", "log(AMT_INCOME_TOTAL)", "log(annual_inc)", "Different currencies; use log, never raw. Drop income <= 0"],
    ["loan_to_income", "AMT_CREDIT / AMT_INCOME_TOTAL", "loan_amnt / annual_inc", "Scale-free"],
    ["payment_to_income", "AMT_ANNUITY*12 / AMT_INCOME_TOTAL", "installment*12 / annual_inc", "Scale-free"],
    ["emp_years", "-DAYS_EMPLOYED/365", "emp_length parsed to years", "HC: 365243 -> NaN; cap both at 10"],
    ["owns_property", "FLAG_OWN_REALTY", "home_ownership in {OWN, MORTGAGE}", "Coarse binary; drop ANY/OTHER/NONE"],
], columns=["concept", "home_credit_column", "lendingclub_column", "notes"])
mapping.to_csv("feature_mapping.csv", index=False)
mapping = pd.DataFrame([
    ["log_income", "log(AMT_INCOME_TOTAL)", "log(annual_inc)", "Within-domain percentile rank. Drop income <= 0"],
    ["loan_to_income", "AMT_CREDIT / AMT_INCOME_TOTAL", "loan_amnt / annual_inc", "Within-domain percentile rank"],
    ["payment_to_income", "AMT_ANNUITY*12 / AMT_INCOME_TOTAL", "installment*12 / annual_inc", "Within-domain percentile rank (x12 factor irrelevant under ranking)"],
    ["emp_years", "-DAYS_EMPLOYED/365", "emp_length parsed to years", "HC: 365243 -> NaN; capped at 10; percentile rank"],
    ["owns_property", "FLAG_OWN_REALTY", "home_ownership in {OWN, MORTGAGE}", "Binary, not ranked"],
], columns=["concept", "home_credit_column", "lendingclub_column", "notes"])
mapping.to_csv("feature_mapping.csv", index=False)