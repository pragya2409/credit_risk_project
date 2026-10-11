import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---- setup: replace with your existing pipeline if you have it ----
df = pd.read_csv("data/application_train.csv")
y = df["TARGET"]
X = pd.get_dummies(df.drop(columns=["TARGET", "SK_ID_CURR"]), drop_first=True)
X.columns = X.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)  # LightGBM-safe names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = lgb.LGBMClassifier(random_state=42, verbose=-1)  # use your tuned params here
model.fit(X_train, y_train)
print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
# --------------------------------------------------------------------

# Sample for speed
X_s = X_test.sample(8000, random_state=42)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_s)
if isinstance(sv, list):          # older SHAP returns [class0, class1]
    sv = sv[1]
base = explainer.expected_value
base = base[1] if np.ndim(base) > 0 else base

# Sanity check: SHAP + base value should equal the model's raw log-odds
raw = model.predict(X_s, raw_score=True)
print("max reconstruction error:", np.abs(sv.sum(axis=1) + base - raw).max())

# Global plots
shap.summary_plot(sv, X_s, show=False, max_display=20)
plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=200); plt.close()

shap.summary_plot(sv, X_s, plot_type="bar", show=False, max_display=20)
plt.tight_layout(); plt.savefig("shap_bar.png", dpi=200); plt.close()

# Top features table
imp = pd.Series(np.abs(sv).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
imp.head(20).to_csv("shap_top_features.csv", header=["mean_abs_shap"])
print(imp.head(30).to_string())

for feat in ["DAYS_BIRTH", "OWN_CAR_AGE"]:
    shap.dependence_plot(feat, sv, X_s, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_dep_{feat}.png", dpi=200)
    plt.close()

print("OWN_CAR_AGE missing share:", X_s["OWN_CAR_AGE"].isna().mean())