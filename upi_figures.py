import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["Current", "Late 1-30", "Late 31-60", "Late 61-90",
          "Late 91-120", "Default(5)", "Closed"]

# ---------- Figure 1: pooled transition matrix (train-only fit) ----------
P = pd.read_csv("transition_matrix_all_fit.csv", index_col=0).loc[LABELS, LABELS]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(P.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(7)); ax.set_xticklabels(LABELS, rotation=35, ha="right")
ax.set_yticks(range(7)); ax.set_yticklabels(LABELS)
for i in range(7):
    for j in range(7):
        v = P.values[i, j]
        ax.text(j, i, f"{100*v:.1f}", ha="center", va="center",
                color="white" if v > 0.5 else "black", fontsize=9)
ax.set_xlabel("Next month"); ax.set_ylabel("This month")
ax.set_title("Monthly transition probabilities (%), pooled credits")
fig.colorbar(im, ax=ax, label="Probability")
plt.tight_layout(); plt.savefig("transition_matrix_heatmap.png", dpi=200); plt.close()

# ---------- Figure 2: stress sweep ----------
s = pd.read_csv("stress_sweep_seed42.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(s.beta, 100 * s.ever_late_mean, marker="o", label="Mean")
ax[0].plot(s.beta, 100 * s.ever_late_p95, marker="o", label="95th percentile across paths")
ax[0].set_title("Applicants ever late within 24 months")
ax[0].set_ylabel("Share of applicants (%)")
ax[1].plot(s.beta, 100 * s.worst_month_late_mean, marker="o", label="Mean")
ax[1].plot(s.beta, 100 * s.worst_month_late_p95, marker="o", label="95th percentile across paths")
ax[1].set_title("Share late in the worst month of a path")
for a in ax:
    a.set_xlabel("β (stress sensitivity)"); a.set_xticks(s.beta); a.legend()
plt.tight_layout(); plt.savefig("stress_sweep.png", dpi=200); plt.close()

# ---------- Figure 3: empirical vs simulated ----------
# Copied from empirical_check.py output and the seed-42 train-only simulator run
metrics = ["Ever late (%)", "Reach Default(5) (%)", "Closed at month 24 (%)", "Mean late months"]
emp = [17.31, 1.01, 82.45, 0.5995]
sim = [25.46, 0.89, 74.24, 0.6361]
fig, ax = plt.subplots(1, 4, figsize=(13, 3.8))
for k, a in enumerate(ax):
    bars = a.bar(["Empirical", "Simulated"], [emp[k], sim[k]],
                 color=["#1f77b4", "#ff7f0e"])
    a.bar_label(bars, fmt="%.2f" if k in (1, 3) else "%.1f")
    a.set_title(metrics[k], fontsize=10)
    a.set_ylim(0, max(emp[k], sim[k]) * 1.2)
plt.suptitle("24 months from a Current start: held-out credits vs simulator", y=1.02)
plt.tight_layout(); plt.savefig("empirical_vs_simulated.png", dpi=200, bbox_inches="tight"); plt.close()
print("Saved: transition_matrix_heatmap.png, stress_sweep.png, empirical_vs_simulated.png")