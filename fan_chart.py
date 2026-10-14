
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
BETAS = [0.0, 0.5, 1.0, 2.0]          # use [0.0, 2.0] for a two-panel version
QUANTS = {"p5": .05, "p25": .25, "median": .50, "p75": .75, "p95": .95}

summaries = {}
rows = []
for beta in BETAS:
    paths = pd.read_csv(f"fan_chart_paths_beta{beta:g}_seed{SEED}.csv").values * 100  # percent
    months = np.arange(paths.shape[1])
    s = {"mean": paths.mean(0)}
    for name, q in QUANTS.items():
        s[name] = np.quantile(paths, q, axis=0)
    summaries[beta] = (months, s)
    for m in months:
        rows.append({"beta": beta, "month": int(m), **{k: v[m] for k, v in s.items()}})

pd.DataFrame(rows).round(4).to_csv("fan_chart_summary.csv", index=False)

fig, axes = plt.subplots(1, len(BETAS), figsize=(4.2 * len(BETAS), 4.2),
                         sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes)
for ax, beta in zip(axes, BETAS):
    months, s = summaries[beta]
    ax.fill_between(months, s["p5"], s["p95"], color="#4C78A8", alpha=0.20, label="5-95% of paths")
    ax.fill_between(months, s["p25"], s["p75"], color="#4C78A8", alpha=0.45, label="25-75% of paths")
    ax.plot(months, s["median"], color="#1F3A5F", lw=2, label="median path")
    ax.set_title("baseline (β = 0)" if beta == 0 else f"UPI stress (β = {beta:g})")
    ax.set_xlabel("month")
    ax.set_xlim(0, months.max())
    ax.grid(alpha=0.3)
axes[0].set_ylabel("share of applicants in a late state (%)")
axes[0].legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle(f"Monte Carlo fan chart: late-payment share over 24 months "
             f"(500 paths x 5,000 applicants, seed {SEED})")
fig.savefig("fan_chart.png", dpi=200)
print("saved fan_chart.png and fan_chart_summary.csv")