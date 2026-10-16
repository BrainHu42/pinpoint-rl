"""Quick gate: listwise logit area chooser on dev3k (5-fold CV x 3 seeds), area features + MegaLoc, with vs without the 27B names columns."""
import numpy as np, torch
f = np.load("artifacts/agent/hypothesis_features_dev3k_setft_own_e4_set.npz")
ml = np.load("artifacts/agent/megaloc_dev3k_setft_own.npz")["x"]
nm = np.load("artifacts/agent/names_dev3k_setft_own.npz")["x"]
photo = f["photo"]
cols = ["rank", "support", "density", "n_cand", "top1", "top3", "mass_0.1", "mass_0.2", "mass_0.5", "photographers", "read_n", "read_p", "read_share",
        "r_support", "r_areas", "d_top1", "set_mass", "set_top_in", "d_settop"]
base = np.stack([np.nan_to_num(f[c].astype(np.float64)) for c in cols], 1)
base[:, cols.index("d_top1")] = np.log1p(base[:, cols.index("d_top1")])
base[:, cols.index("d_settop")] = np.log1p(base[:, cols.index("d_settop")])
y = (f["km"] < 25).astype(np.float64)
P = np.unique(photo)
def fit_eval(X, seed):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(seed); folds = rng.permutation(len(P)) % 5
    hits = []
    for k in range(5):
        tr, te = np.isin(photo, P[folds != k]), np.isin(photo, P[folds == k])
        Xt = torch.tensor(X[tr], dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.float32); pt = torch.tensor(np.unique(photo[tr], return_inverse=True)[1])
        w = torch.zeros(X.shape[1], requires_grad=True); opt = torch.optim.LBFGS([w], max_iter=200)
        npht = int(pt.max()) + 1
        def closure():
            opt.zero_grad()
            s = Xt @ w
            m = torch.full((npht,), -1e9).scatter_reduce(0, pt, s, "amax")
            lse = torch.log(torch.zeros(npht).scatter_add(0, pt, torch.exp(s - m[pt]))) + m
            good = torch.zeros(npht).scatter_add(0, pt, yt * torch.exp(s - m[pt]))
            has = torch.zeros(npht).scatter_add(0, pt, yt) > 0
            loss = -(torch.log(good[has] + 1e-12) + m[has] - lse[has]).mean() + 1e-3 * (w * w).sum()
            loss.backward(); return loss
        opt.step(closure)
        s = X[te] @ w.detach().numpy()
        for q in np.unique(photo[te]):
            J = np.flatnonzero(te)[photo[te] == q]
            hits.append(y[J[np.argmax(s[photo[te] == q])]])
    return np.mean(hits)
for name, X in (("area feats", base), ("+ megaloc", np.hstack([base, ml])), ("+ megaloc + names", np.hstack([base, ml, nm])), ("+ names", np.hstack([base, nm]))):
    print(f"{name:22s}", " ".join(f"{100 * fit_eval(X, s):.2f}" for s in range(3)), flush=True)
