# Is the answer head under-trained on near-misses, or are its features the limit? (setft_own, MegaLoc on, clues read, every area zoomed)
#   pairs (right area, wrong area) in a photo; "near" = the wrong area's answer 25-100 km from the truth, "far" = > 100 km
#   head: trained on train as in the pipeline (3 seeds, mean logits); pairwise accuracy on dev3k near / far pairs
#   pairwise models on area-token features (pre-zoom incl. MegaLoc + post-zoom + pooled candidate stats), trained on train pairs (near only /
#   all), scored on dev3k near pairs: if a near-specialist beats the head clearly, near-miss training signal is missing.
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from geo_search_env.experiment.action_agent import _released_own_dev3k_km  # noqa: E402,F401
from geo_search_env.experiment.hypothesis_agent import fit_answer, load  # noqa: E402

CKPT = "setft_own"
tr = load("train", CKPT, extra=("megaloc",))
dv = load("dev3k", CKPT, stats=tr["stats"], extra=("megaloc",))
for d in (tr, dv):
    d.pop("pre_noclue", None)  # clues read: the full state


def tokens(d):
    """Per area-token feature vector: pre + post + mean / max of the candidate features over valid slots."""
    v = d["cvalid"].float().unsqueeze(-1)
    cmean = (d["cand"] * v).sum(2) / v.sum(2).clamp(min=1)
    cmax = d["cand"].masked_fill(~d["cvalid"].unsqueeze(-1), -1e4).max(2).values.clamp(min=-10)
    return torch.cat([d["pre"], d["post"], cmean, cmax], -1).cpu().numpy()


def pairs(d):
    km = d["km_zoom"].cpu().numpy()
    mask = d["mask"].cpu().numpy()
    out = {"near": [], "far": []}
    for p in range(d["n"]):
        idx = np.flatnonzero(mask[p])
        right = [i for i in idx if km[p, i] < 25]
        for i in right:
            for j in idx:
                if 25 <= km[p, j] < 100:
                    out["near"].append((p, i, j))
                elif km[p, j] >= 100:
                    out["far"].append((p, i, j))
    return {k: np.asarray(v, int).reshape(-1, 3) for k, v in out.items()}


ptr, pdv = pairs(tr), pairs(dv)
for name, P, d in (("train", ptr, tr), ("dev3k", pdv, dv)):
    n_near = len(np.unique(P["near"][:, 0])) if len(P["near"]) else 0
    n_any = len(np.unique(np.concatenate([P["near"][:, 0], P["far"][:, 0]])))
    print(f"{name}: {d['n']} photos; photos with a right area and a near-miss (25-100 km) wrong area {n_near} ({100 * n_near / d['n']:.1f}%), "
          f"with a right and any wrong area {n_any} ({100 * n_any / d['n']:.1f}%); pairs near {len(P['near'])}, far {len(P['far'])}", flush=True)

# the pipeline's answer head
logits = []
for s in (0, 1, 2):
    net = fit_answer(tr, np.arange(tr["n"]), s)
    with torch.no_grad():
        logits.append(torch.log_softmax(net.answer_logits(dv, dv["mask"]), 1))
L = torch.stack(logits).mean(0).cpu().numpy()


def pair_acc(score, P):
    return np.mean(score[P[:, 0], P[:, 1]] > score[P[:, 0], P[:, 2]]) if len(P) else np.nan


print(f"\nhead pairwise accuracy on dev3k: near {100 * pair_acc(L, pdv['near']):.1f}% ({len(pdv['near'])} pairs), far {100 * pair_acc(L, pdv['far']):.1f}%")
# photo level: photos with a right and a near-miss wrong area
km = dv["km_zoom"].cpu().numpy()
pick = np.where(dv["mask"].cpu().numpy(), L, -np.inf).argmax(1)
ph = np.unique(pdv["near"][:, 0])
pk = km[ph, pick[ph]]
print(f"  on the {len(ph)} dev3k photos with a right and a near-miss area: head right {100 * np.mean(pk < 25):.1f}%, picks a near-miss "
      f"{100 * np.mean((pk >= 25) & (pk < 100)):.1f}%, a far one {100 * np.mean(pk >= 100):.1f}%")

# pairwise specialists on token features (symmetric: x_i - x_j -> 1, x_j - x_i -> 0)
Xtr, Xdv = tokens(tr), tokens(dv)


def design(X, P):
    a, b = X[P[:, 0], P[:, 1]], X[P[:, 0], P[:, 2]]
    return np.concatenate([a - b, b - a]), np.concatenate([np.ones(len(P)), np.zeros(len(P))])


def fit_mlp(X, y, seed=0, epochs=30, linear=False):
    torch.manual_seed(seed)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32, device="cuda"), torch.as_tensor(y, dtype=torch.float32, device="cuda")
    net = (torch.nn.Linear(X.shape[1], 1, bias=False) if linear else
           torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 1))).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        for bi in torch.randperm(len(Xt), device="cuda").split(512):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[bi]).squeeze(-1), yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def spec_acc(net, X, P):
    a, b = X[P[:, 0], P[:, 1]], X[P[:, 0], P[:, 2]]
    with torch.no_grad():
        z = net(torch.as_tensor(a - b, dtype=torch.float32, device="cuda")).squeeze(-1).cpu().numpy()
    return np.mean(z > 0)


for label, P in (("near pairs only", ptr["near"]), ("all pairs", np.concatenate([ptr["near"], ptr["far"]]))):
    X, y = design(Xtr, P)
    lin = np.mean([spec_acc(fit_mlp(X, y, s, linear=True), Xdv, pdv["near"]) for s in (0, 1, 2)])
    mlp = np.mean([spec_acc(fit_mlp(X, y, s), Xdv, pdv["near"]) for s in (0, 1, 2)])
    print(f"specialist trained on train {label:16s} ({len(P)} pairs): dev3k near-pair accuracy linear {100 * lin:.1f}%, MLP {100 * mlp:.1f}%")
