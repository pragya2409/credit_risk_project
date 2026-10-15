# Figure 3.3: Missingness by column (Home Credit application_train)
# Run: python missingness_plot.py   (or pass the CSV path as an argument)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

def find_file(name="application_train.csv"):
    if len(sys.argv) > 1:                      # path given on the command line
        return Path(sys.argv[1])
    here = Path(__file__).resolve().parent
    for base in (Path.cwd(), here, here.parent):
        for sub in ("", "data", "Data", "dataset", "datasets", "home-credit-default-risk"):
            p = base / sub / name
            if p.exists():
                return p
    for base in (here, here.parent):           # last resort: search the project tree
        hits = list(base.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {name}. Run: python missingness_plot.py /path/to/{name}")

path = find_file()
print("Using:", path)
df = pd.read_csv(path)
miss = (df.isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]
top = miss.head(30)

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top.index[::-1], top.values[::-1], color="#2B6CB0")
ax.set_xlabel("Missing values (%)")
ax.set_title(f"Top 30 columns by missingness ({len(miss)} of {df.shape[1]} columns have any missing values)")
ax.set_xlim(0, 100)
for y, v in enumerate(top.values[::-1]):
    ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("missingness_plot.png", dpi=200, facecolor="white")

miss.rename("missing_pct").round(2).to_csv("missingness_by_column.csv")
print(miss.head(10))