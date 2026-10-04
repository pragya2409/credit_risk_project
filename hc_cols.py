import pandas as pd
print(list(pd.read_csv("data/application_train.csv", nrows=0).columns))