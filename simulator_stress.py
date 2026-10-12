import sys
import numpy as np
import pandas as pd
from common import load_model_and_data

H, N_PATHS, N_APPS = 24, 500, 5000
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SUFFIX = "_fit"                       # train-only matrices from CP5
BETAS = [0.0, 0.5, 1.0, 2.0]          # beta = 0 is the plain chain
LABELS = ["Current", "Late 1-30", "Late 31-60", "Late 61-90",
          "Late 91-120", "Default(5)", "Closed"]
K, CUR, DEF, CLO = 7, 0, 5, 6

def load(name):
    return pd.read_csv(f"transition_matrix_{name}{SUFFIX}.csv", index_col=0).loc[LABELS, LABELS].values

P_all, P0, P1 = load("all"), load("target0"), load("target1")
P = P_all.copy()
P[CLO] = 0; P[CLO, CLO] = 1
P[DEF] = 0; P[DEF, DEF] = 1
P = P / P.sum(axis=1, keepdims=True)

# ---- UPI shocks: standardized month-over-month change in total spend (11 values) ----
upi = pd.read_csv("data/upi/upi_transactions_2024.csv",
                  usecols=["timestamp", "amount (INR)"], parse_dates=["timestamp"])
chg = upi.set_index("timestamp").resample("MS")["amount (INR)"].sum().pct_change().dropna()
Z = ((chg - chg.mean()) / chg.std()).values
print(f"{len(Z)} monthly spend changes, std {100*chg.std():.2f}% -> standardized shocks")

# ---- applicants (same draw as simulator.py for the same SEED) ----
model, X_train, X_test, y_train, y_test = load_model_and_data()
p_all = model.predict_proba(X_test)[:, 1]
rng0 = np.random.default_rng(SEED)
idx = rng0.choice(len(X_test), size=N_APPS, replace=False)
p = p_all[idx]
cur_rows = (1 - p)[:, None] * P0[CUR] + p[:, None] * P1[CUR]
cum = np.broadcast_to(np.cumsum(P, axis=1), (N_APPS, K, K)).copy()
cum[:, CUR, :] = np.cumsum(cur_rows, axis=1)
cum[:, :, -1] = 1.0
p1 = cur_rows[:, 1]                    # each applicant's Current -> Late 1-30 probability
a = np.arange(N_APPS)[:, None]

def run(beta):
    rng = np.random.default_rng(SEED + 1000)      # identical draws for every beta
    norm = np.mean(np.exp(beta * Z))              # keeps the average multiplier at 1
    state = np.zeros((N_APPS, N_PATHS), dtype=np.int8)
    ever_late = np.zeros(state.shape, bool)
    ever_def = np.zeros(state.shape, bool)
    worst = np.zeros(N_PATHS)
    for t in range(H):
        s = Z[rng.integers(0, len(Z), N_PATHS)]   # one shock per path-month, shared by all applicants
        mult = np.exp(beta * s) / norm
        th = cum[a, state]                        # (applicants, paths, K) copy
        # Only Current -> Late 1-30 changes; the mass moves between those two cells
        th[..., 0] -= (state == CUR) * p1[:, None] * (mult[None, :] - 1)
        th[..., 0] = np.maximum(th[..., 0], 0)
        u = rng.random(state.shape)
        state = (u[..., None] > th).sum(-1).astype(np.int8)
        is_late = (state >= 1) & (state <= 5)
        ever_late |= is_late
        ever_def |= (state == DEF)
        worst = np.maximum(worst, is_late.mean(0))
    return ever_late.mean(0), ever_def.mean(0), worst   # each: one value per path

rows = []
for beta in BETAS:
    el, ed, wl = run(beta)
    rows.append({"beta": beta,
                 "ever_late_mean": el.mean(), "ever_late_std": el.std(), "ever_late_p95": np.quantile(el, .95),
                 "default_mean": ed.mean(), "default_p95": np.quantile(ed, .95),
                 "worst_month_late_mean": wl.mean(), "worst_month_late_p95": np.quantile(wl, .95)})
    print(f"beta={beta} done")

out = pd.DataFrame(rows)
out.to_csv(f"stress_sweep_seed{SEED}.csv", index=False)
pd.set_option("display.width", 220)
print("\nAcross 500 simulated paths (each path = one sequence of shocks over 24 months):")
print(out.round(4).to_string(index=False))