import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


def _components(preds, y):
    """preds: (k, n) scores, y: (n,) 0/1. Returns AUCs (k,) and covariance (k, k)."""
    y = np.asarray(y)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    aucs = np.empty(k)
    v01 = np.empty((k, m))   # structural components, positives
    v10 = np.empty((k, n))   # structural components, negatives
    for r in range(k):
        tx = rankdata(pos[r])
        ty = rankdata(neg[r])
        tz = rankdata(np.concatenate([pos[r], neg[r]]))
        aucs[r] = tz[:m].sum() / (m * n) - (m + 1) / (2 * n)
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1 - (tz[m:] - ty) / m
    S = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, S


def delong_test(y, p1, p2):
    """Paired DeLong test, model 1 minus model 2."""
    aucs, S = _components(np.vstack([p1, p2]), y)
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = np.sqrt(var)
    z = diff / se
    return dict(auc1=aucs[0], auc2=aucs[1], diff=diff, se=se,
                ci_lo=diff - 1.96 * se, ci_hi=diff + 1.96 * se,
                z=z, p_raw=2 * norm.sf(abs(z)))


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_diff_se(y, p1, p2, B=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i])
    return d.std(ddof=1)


if __name__ == "__main__":
    c = pickle.load(open("stack_cache.pkl", "rb"))
    s = pickle.load(open("standalone_cache.pkl", "rb"))
    y = c["y_test"]

    X_oof = np.column_stack([c["oof"][k] for k in ("lgbm", "xgb", "lr")])
    X_te = np.column_stack([c["test"][k] for k in ("lgbm", "xgb", "lr")])
    meta = LogisticRegression(max_iter=1000).fit(X_oof, c["y_train"])

    preds = {
        "stack": meta.predict_proba(X_te)[:, 1],
        "xgb": s["xgb"], "lgbm": s["lgbm"], "lr": s["lr"],          # standalone
        "xgb_bag": c["test"]["xgb"], "lgbm_bag": c["test"]["lgbm"],  # fold-averaged
    }
    # convert to float64 BEFORE anything else uses them
    preds = {k: np.asarray(v, dtype=np.float64) for k, v in preds.items()}

    for k, v in preds.items():
        print(f"{k:9s} AUC {roc_auc_score(y, v):.4f}  PR-AUC {average_precision_score(y, v):.4f}")

    # Validation: DeLong AUC must equal sklearn's
    for k, v in preds.items():
        a, _ = _components(np.vstack([v, v]), y)
        d = abs(a[0] - roc_auc_score(y, v))
        print(f"{k:9s} unique scores {len(np.unique(v)):6d}  |DeLong - sklearn| = {d:.2e}")
        assert d < 1e-9, k
    print("AUC match: OK")

    def run(pairs, label):
        rows = [dict(family=label, model_a=a, model_b=b, **delong_test(y, preds[a], preds[b]))
                for a, b in pairs]
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p_raw"].values)  # Holm within this family only
        return df

    primary = run([("stack", "xgb"), ("stack", "lgbm"),
                   ("stack", "lr"), ("xgb", "lgbm")], "primary")
    exploratory = run([("stack", "xgb_bag"), ("xgb_bag", "xgb")], "exploratory_posthoc")
    df = pd.concat([primary, exploratory])
    df.to_csv("delong_results.csv", index=False)
    cols = ["family", "model_a", "model_b", "diff", "ci_lo", "ci_hi", "p_raw", "p_holm"]
    print(df[cols].round(5).to_string(index=False))