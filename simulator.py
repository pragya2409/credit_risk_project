
import sys
import numpy as np
import pandas as pd
from common import load_model_and_data

# ---- settings ----
H, N_PATHS, N_APPS = 24, 500, 5000,   # months, paths per applicant, applicants
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
ABSORB_DEFAULT = True                          # treat Default(5) as absorbing
LABELS = ["Current", "Late 1-30", "Late 31-60", "Late 61-90",
          "Late 91-120", "Default(5)", "Closed"]
K, CUR, DEF, CLO = 7, 0, 5, 6

def load(name):
    return pd.read_csv(f"transition_matrix_{name}{SUFFIX}.csv", index_col=0).loc[LABELS, LABELS].values

P_all, P0, P1 = load("all"), load("target0"), load("target1")

# Pooled matrix for every row; Closed (and optionally Default) made exactly absorbing
P = P_all.copy()
P[CLO] = 0; P[CLO, CLO] = 1
if ABSORB_DEFAULT:
    P[DEF] = 0; P[DEF, DEF] = 1
P = P / P.sum(axis=1, keepdims=True)

# ---- applicants and their risk ----
model, X_train, X_test, y_train, y_test = load_model_and_data()
p_all = model.predict_proba(X_test)[:, 1]
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_test), size=N_APPS, replace=False)
p, ids = p_all[idx], X_test.index[idx]

# Applicant-specific Current row: the model's p(default) is read as the probability
# of belonging to the defaulter group, so the row is a mixture of the TARGET=0 and
# TARGET=1 Current rows. All other rows are pooled.
cur_rows = (1 - p)[:, None] * P0[CUR] + p[:, None] * P1[CUR]
cum = np.broadcast_to(np.cumsum(P, axis=1), (N_APPS, K, K)).copy()
cum[:, CUR, :] = np.cumsum(cur_rows, axis=1)
cum[:, :, -1] = 1.0

# ---- simulate: everyone starts Current ----
state = np.zeros((N_APPS, N_PATHS), dtype=np.int8)
ever_def = np.zeros(state.shape, dtype=bool)
ever_late = np.zeros(state.shape, dtype=bool)
late_months = np.zeros(state.shape, dtype=np.int16)
a = np.arange(N_APPS)[:, None]

for t in range(H):
    u = rng.random(state.shape)
    state = (u[..., None] > cum[a, state]).sum(-1).astype(np.int8)
    is_late = (state >= 1) & (state <= 5)
    ever_late |= is_late
    ever_def |= (state == DEF)
    late_months += is_late

res = pd.DataFrame({
    "row_id": ids, "p_default": p,
    "p_ever_late": ever_late.mean(1),
    "p_reach_default": ever_def.mean(1),
    "mean_late_months": late_months.mean(1),
    "p_closed_at_end": (state == CLO).mean(1),
})
res.to_csv(f"simulation_results{SUFFIX}_{SEED}.csv", index=False)

res["risk_quintile"] = pd.qcut(res.p_default, 5, labels=False) + 1
cols = ["p_default", "p_ever_late", "p_reach_default", "mean_late_months", "p_closed_at_end"]
print(f"H={H} months, {N_PATHS} paths x {N_APPS} applicants, absorbing default={ABSORB_DEFAULT}")
print("\nBy risk quintile (1 = safest):")
print(res.groupby("risk_quintile")[cols].mean().round(4).to_string())
print("\nOverall:\n", res[cols].mean().round(4).to_string())