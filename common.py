# common.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def load_model_and_data():
    df = pd.read_csv("data/application_train.csv")
    y = df["TARGET"]
    X = pd.get_dummies(df.drop(columns=["TARGET", "SK_ID_CURR"]), drop_first=True)
    X.columns = X.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)  # your params here
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
