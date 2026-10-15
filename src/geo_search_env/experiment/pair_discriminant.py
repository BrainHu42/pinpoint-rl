# "What differs between these candidates" without an LLM (the user, after lesson 102). Every signal so far scores query vs area A and query
# vs area B separately; retrieval proposed both because the query resembles both, so those similarities are dominated by what A and B share.
# Here, per pair of candidate areas (the head's top two; the decisive pairs of lessons 99-101), a discriminant is fitted on the fly between
# the two areas' index photos (within 25 km of each area's centre, photographer excluded, raw SigLIP2) and applied to the query:
#   rand: N random photos per area (what the areas look like in general);  near: the N photos per area nearest the query (the local look);
#   kernel ridge (linear kernel, +1 / -1 targets, three ridge strengths), the centroid difference (a similarity control) and, for rand,
#   the share of A among the query's 16 nearest of the union (a density-normalised vote).
# Evaluated as lesson 100: AUC alone, then logistic regression on [head margin, features] (train 5-fold CV; fit on all train, applied to dev3k).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pair_discriminant features --tag dev3k   (then --tag train)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pair_discriminant report

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .contrast_gate import _pairs_path
from .contrast_model import TRAIN_PAIRS

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
RADIUS_KM, N, MIN_N = 25.0, 256, 8
LAMBDAS = (0.1, 1.0, 10.0)
NAMES = [f"rand_ridge{l}" for l in LAMBDAS] + [f"near_ridge{l}" for l in LAMBDAS] + ["rand_centroid", "near_centroid", "rand_knn16", "n_small"]


def _pairs(tag: str) -> list[dict]:
    return json.loads((TRAIN_PAIRS if tag == "train" else _pairs_path()).read_text(encoding="utf-8"))


def features(tag: str) -> None:
    import torch

    from ..tools.pinpoint import Pinpoint

    pr = _pairs(tag)
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    center = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")["center"]
    tool = Pinpoint()
    rr = tool.reranker
    A = tool.index_author()
    q_author = tool.mp16_author(np.asarray([e["row"] for e in photos]))
    views = np.load(OUT / f"crops_{tag}_{CKPT_TAG}.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    F = np.zeros((len(pr), len(NAMES)), np.float32)

    def ridge(Xa, Xb, q):
        """Kernel ridge (linear kernel, dual form) A = +1 vs B = -1, balanced; scores of q for each lambda (scaled by the mean kernel diagonal)."""
        X = torch.cat([Xa, Xb]).float()
        y = torch.cat([torch.ones(len(Xa)), -torch.ones(len(Xb))]).to(X.device)
        mu = X.mean(0)
        Xc, qc = X - mu, q.float() - mu
        K = Xc @ Xc.T
        scale = K.diagonal().mean()
        kq = Xc @ qc
        out = []
        for lam in LAMBDAS:
            alpha = torch.linalg.solve(K + lam * scale * torch.eye(len(K), device=K.device), y)
            out.append(float(kq @ alpha))
        return out

    with torch.inference_mode():
        for i, o in enumerate(pr):
            p = o["photo"]
            q, _ = rr.encode_queries(np.asarray(views[p, 0], np.float32)[None])
            q = q[0].half()
            pools = []
            for h in o["rows"]:
                near = tool.rows_near(center[h][None], RADIUS_KM)
                pools.append(near[A[near] != q_author[p]])
            n = min(N, len(pools[0]), len(pools[1]))
            if n < MIN_N:
                F[i, -1] = 1
                continue
            rand, close = [], []
            for rows in pools:
                r = rng.choice(rows, n, replace=False) if len(rows) > n else rows
                rand.append(rr._dequantize(0, rows=np.sort(r)))
                if len(rows) > 50_000:  # very dense areas: the nearest among a random 50k
                    rows = rng.choice(rows, 50_000, replace=False)
                E = rr._dequantize(0, rows=np.sort(rows))
                close.append(E[torch.topk(E @ q, n).indices])
            f_rand, f_near = ridge(rand[0], rand[1], q), ridge(close[0], close[1], q)
            c_rand = float((rand[0].float().mean(0) - rand[1].float().mean(0)) @ q.float())
            c_near = float((close[0].float().mean(0) - close[1].float().mean(0)) @ q.float())
            s = torch.cat([rand[0] @ q, rand[1] @ q]).float()
            top = torch.topk(s, 16).indices
            knn = float((top < n).float().mean() - 0.5)
            F[i] = f_rand + f_near + [c_rand, c_near, knn, 0.0]
            if i % 500 == 0:
                print(f"  {i} / {len(pr)} (n {n})", flush=True)
    np.save(OUT / f"pair_disc_{tag}.npy", F)
    print(f"-> {OUT / f'pair_disc_{tag}.npy'}: {len(F)} pairs, {int(F[:, -1].sum())} with < {MIN_N} photos in an area")


def _logreg(X, y, Xt):
    import torch

    mu, sd = X.mean(0), X.std(0) + 1e-6
    A, B = torch.tensor((X - mu) / sd, dtype=torch.float64), torch.tensor((Xt - mu) / sd, dtype=torch.float64)
    w = torch.zeros(A.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    yy = torch.tensor(y, dtype=torch.float64)
    opt = torch.optim.LBFGS([w], max_iter=300)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w[:-1] + w[-1], yy) + 1e-3 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.sigmoid(B @ w[:-1] + w[-1]).detach().numpy()


def _auc(s, y):
    a, b = s[y == 1], s[y == 0]
    return (a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()


def report() -> None:
    data = {}
    for tag in ("train", "dev3k"):
        pr = _pairs(tag)
        km = np.asarray([o["km"] for o in pr])
        m = np.asarray([o["logit"][0] - o["logit"][1] for o in pr])
        data[tag] = (np.load(OUT / f"pair_disc_{tag}.npy"), m, km)
    Ft, Mt, Kt = data["train"]
    yt = (Kt[:, 0] < 25).astype(int)
    ok = Ft[:, -1] == 0
    print(f"train: {len(yt)} decisive pairs, head right {yt.mean():.1%}; {ok.mean():.1%} with >= {MIN_N} photos in both areas")
    print(f"  head margin AUC {_auc(Mt, yt):.3f} (all) / {_auc(Mt[ok], yt[ok]):.3f} (with features)")
    for j, name in enumerate(NAMES[:-1]):
        print(f"  {name:16s} AUC alone {_auc(Ft[ok, j], yt[ok]):.3f}")
    folds = np.random.default_rng(1).permutation(len(yt)) % 5
    Fd, Md, Kd = data["dev3k"]
    for label, cols in (("margin", []), ("margin + rand ridge", [0, 1, 2]), ("margin + near ridge", [3, 4, 5]),
                        ("margin + centroids + knn", [6, 7, 8]), ("margin + all", list(range(len(NAMES))))):
        X = np.column_stack([Mt] + [Ft[:, c] for c in cols])
        Xd = np.column_stack([Md] + [Fd[:, c] for c in cols])
        p = np.zeros(len(yt))
        for f in range(5):
            p[folds == f] = _logreg(X[folds != f], yt[folds != f], X[folds == f])
        pd = _logreg(X, yt, Xd)
        line = f"{label:26s}: train CV AUC {_auc(p, yt):.3f}, acc {np.mean((p >= .5) == yt):.1%}"
        for thr in (25, 1):
            right = Kd < thr
            dec = right[:, 0] != right[:, 1]
            sw = pd < 0.5
            fx, br = int((sw & right[:, 1] & ~right[:, 0]).sum()), int((sw & right[:, 0] & ~right[:, 1]).sum())
            line += (f" | dev3k <{thr} km acc {np.mean(np.where(pd[dec] >= .5, right[dec, 0], right[dec, 1])):.1%} "
                     f"fix {fx} brk {br} ({100 * (fx - br) / len(pd):+.1f})")
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("features", "report"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    features(args.tag) if args.cmd == "features" else report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
