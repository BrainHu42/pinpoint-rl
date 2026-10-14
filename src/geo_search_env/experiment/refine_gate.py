# Gate for the refine agent (lesson 91, the user's idea: move answers into the next tighter bucket). Per dev3k photo, Pinpoint (retrained
# reranker) restricted to R km around its own top-1 (R = 25: target 1 km; R = 200: target 25 km), photographer excluded. A local chooser over
# that list, with evidence that failed globally but may work between neighbours: reranker score / rank / retrieval features, distance from
# the current answer, agreement of nearby candidates (soft kernels), forward cosine and reverse rank (round trip, lesson 90) in the contrastive
# space, and the best crop's raw SigLIP2 cosine (lesson 89 views). Chooser: listwise logit / small MLP, 2-fold x 3 seeds on dev3k; measured as
# the net change at every threshold vs keeping the top-1 (moves can also break photos already in a bucket).
# Usage: PYTHONPATH=src PINPOINT_RERANKER_CKPT=... .venv/bin/python -m geo_search_env.experiment.refine_gate search --radius 25
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.refine_gate evidence --radius 25
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.refine_gate gate --radius 25

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .refine_headroom import _hav

OUT = Path("artifacts/agent")
TAG, CKPT_TAG = "dev3k", "setft_a"
THRESHOLDS = (1.0, 25.0, 200.0, 750.0, 2500.0)
TARGET = {25: 1.0, 200: 25.0}


def _photos() -> list[dict]:
    return json.loads((Path("artifacts/photo_sets") / f"{TAG}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))


def _path(radius: int, kind: str) -> Path:
    return OUT / f"refine_{kind}_{TAG}_{CKPT_TAG}_r{radius}.npz"


def search(radius: int, batch: int = 250) -> None:
    from ..tools.pinpoint import Constraint, Pinpoint

    photos = _photos()
    top1 = np.asarray([e["candidates"][0][1:] for e in photos], np.float64)
    tool = Pinpoint()
    views = np.load(OUT / f"crops_{TAG}_{CKPT_TAG}.npy", mmap_mode="r")
    authors = tool.mp16_author(np.asarray([e["row"] for e in photos]))
    parts = []
    for s in range(0, len(photos), batch):
        idx = np.arange(s, min(s + batch, len(photos)))
        parts.append(tool.search(np.asarray(views[idx, 0], np.float32), [Constraint(near=(top1[i][None], float(radius))) for i in idx], authors[idx]))
        print(f"  {idx[-1] + 1} / {len(photos)}", flush=True)
    c = max(p.rows.shape[1] for p in parts)
    pad = lambda arrs, fill: np.concatenate([np.pad(a, [(0, 0), (0, c - a.shape[1])] + [(0, 0)] * (a.ndim - 2), constant_values=fill) for a in arrs])
    np.savez(_path(radius, "search"), rows=pad([p.rows for p in parts], -1), cache_rows=pad([p.cache_rows for p in parts], -1),
             latlon=pad([p.latlon for p in parts], np.nan), scores=pad([p.scores for p in parts], -np.inf), features=pad([p.features for p in parts], 0.0))
    print(f"-> {_path(radius, 'search')}")


def evidence(radius: int, chunk: int = 512_000) -> None:
    """Forward cosine / reverse rank (contrastive space) and best-crop raw cosine for every candidate."""

    import torch

    from ..tools.pinpoint import Pinpoint
    from .strategy_search import MP16_EMBED, _memmap_gallery

    photos = _photos()
    z = np.load(_path(radius, "search"))
    rows, cache_rows = z["rows"], z["cache_rows"]
    valid = rows >= 0
    tool = Pinpoint()
    rr = tool.reranker
    q_author = tool.mp16_author(np.asarray([e["row"] for e in photos]))
    A = torch.as_tensor(tool.index_author(), device=rr.device)
    views = np.load(OUT / f"crops_{TAG}_{CKPT_TAG}.npy", mmap_mode="r")
    _, q_emb = rr.encode_queries(np.asarray(views[:, 0], np.float32))
    E = torch.as_tensor(np.asarray(rr.image), device=rr.device)
    q, k = np.nonzero(valid)
    r = rows[q, k]
    fwd = np.zeros(len(r), np.float32)
    rev = np.zeros(len(r), np.int64)
    with torch.inference_mode():
        for s in range(0, len(r), 2048):
            rb = torch.as_tensor(r[s:s + 2048], device=rr.device)
            qb = torch.as_tensor(q[s:s + 2048], device=rr.device)
            er = E[rb]
            thr = (er.float() * q_emb[qb]).sum(1)
            qa = torch.as_tensor(q_author[q[s:s + 2048]], device=rr.device)
            cnt = torch.zeros(len(rb), dtype=torch.long, device=rr.device)
            for c in range(0, len(E), chunk):
                closer = (er @ E[c:c + chunk].T).float() > thr[:, None]
                closer &= A[c:c + chunk][None, :] != qa[:, None]
                cnt += closer.sum(1)
            fwd[s:s + 2048], rev[s:s + 2048] = thr.cpu().numpy(), (cnt - 1).clamp(min=0).cpu().numpy()
            if (s // 2048) % 20 == 0:
                print(f"  reverse ranks {s + len(rb)} / {len(r)}", flush=True)
    del E
    g = _memmap_gallery(MP16_EMBED)["embeddings"]
    cr = cache_rows[q, k]
    order = np.argsort(cr)
    emb = np.empty((len(cr), g.shape[1]), np.float32)
    emb[order] = g[cr[order]]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    crop = np.zeros(len(cr), np.float32)
    full = np.zeros(len(cr), np.float32)
    for s in range(0, len(cr), 50_000):
        v = np.asarray(views[q[s:s + 50_000]], np.float32)  # [b, 10, 1536]
        sims = np.einsum("bvd,bd->bv", v, emb[s:s + 50_000])
        full[s:s + 50_000], crop[s:s + 50_000] = sims[:, 0], sims[:, 1:].max(1)
    shape = rows.shape
    out = {name: np.zeros(shape, np.float32) for name in ("fwd", "rev", "crop", "full")}
    for name, v in (("fwd", fwd), ("rev", np.log1p(rev)), ("crop", crop), ("full", full)):
        out[name][q, k] = v
    np.savez(_path(radius, "evidence"), **out)
    print(f"-> {_path(radius, 'evidence')}; reverse rank median {np.median(rev):.0f}")


def _features(radius: int):
    photos = _photos()
    z, ev = np.load(_path(radius, "search")), np.load(_path(radius, "evidence"))
    valid = z["rows"] >= 0
    ll = z["latlon"]
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    top1 = np.asarray([e["candidates"][0][1:] for e in photos], np.float64)
    km = np.where(valid, _hav(truth[:, 0, None], truth[:, 1, None], ll[..., 0], ll[..., 1]), np.inf)
    base_km = _hav(truth[:, 0], truth[:, 1], top1[:, 0], top1[:, 1])
    s = np.where(valid, z["scores"], -np.inf)
    best = s.max(1, keepdims=True)
    soft = np.where(valid, np.exp((s - best) / 0.2), 0.0)
    soft /= soft.sum(1, keepdims=True)
    move = np.where(valid, _hav(top1[:, 0, None], top1[:, 1, None], ll[..., 0], ll[..., 1]), 0.0)
    pair = _hav(ll[:, :, None, 0], ll[:, :, None, 1], ll[:, None, :, 0], ll[:, None, :, 1])  # [Q, C, C]
    pair = np.where(valid[:, :, None] & valid[:, None, :], pair, np.inf)
    dens = [np.einsum("qcd,qd->qc", (pair < r_km).astype(np.float64), soft) for r_km in (0.5, 1.0, 5.0)]
    count1 = (pair < 1.0).sum(2).astype(np.float64)
    rank = np.broadcast_to(np.arange(valid.shape[1])[None], valid.shape).astype(np.float64)
    rel = lambda v: np.where(valid, v - np.where(valid, v, -np.inf).max(1, keepdims=True), 0.0)
    cols = [s - best, soft, np.minimum(rank, 20), rank == 0, np.log1p(move), *dens, np.log1p(count1),
            ev["fwd"], rel(ev["fwd"]), -ev["rev"], rel(-ev["rev"]), ev["crop"], rel(ev["crop"]), ev["full"], rel(ev["full"])]
    names = ["score_gap", "soft", "rank", "is_top", "log_move", "dens0.5", "dens1", "dens5", "count1", "fwd", "fwd_rel", "rev", "rev_rel",
             "crop", "crop_rel", "full", "full_rel"]
    x = np.stack([np.where(valid, np.nan_to_num(c, posinf=0, neginf=0), 0.0) for c in cols] + [z["features"][..., j] for j in range(9)], -1)
    names += [f"retr{j}" for j in range(9)]
    return x.astype(np.float32), valid, km, base_km, names


def gate(radius: int) -> None:
    import torch

    x, valid, km, base_km, names = _features(radius)
    target = TARGET[radius]
    n = len(km)
    m = x[valid].mean(0), x[valid].std(0) + 1e-6
    x = np.where(valid[..., None], (x - m[0]) / m[1], 0.0).astype(np.float32)
    y = (km < target) & valid
    reward = sum((km < t).astype(np.float32) for t in THRESHOLDS) * valid  # thresholds met by each candidate (the agent's reward)
    print(f"r = {radius} km around the retrained top-1, target {target:g} km; {n} photos, {valid.sum(1).mean():.0f} candidates; "
          f"top-1 < {target:g} km {np.mean(base_km < target):.1%}, list oracle {y.any(1).mean():.1%}")

    def fit(cols, hidden, seed, tr, objective="target"):
        torch.manual_seed(seed)
        d = len(cols)
        net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1)) if hidden else torch.nn.Linear(d, 1)
        X, M, Y = torch.tensor(x[tr][..., cols]), torch.tensor(valid[tr]), torch.tensor(y[tr])
        R = torch.tensor(reward[tr])
        has = Y.any(1)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(400):
            opt.zero_grad()
            lp = torch.log_softmax(net(X).squeeze(-1).masked_fill(~M, -1e9), 1)
            if objective == "reward":  # expected thresholds met under the softmax policy (exact policy gradient: every outcome is known)
                loss = -(lp.exp() * R).sum(1).mean()
            else:
                loss = -torch.logsumexp(lp.masked_fill(~Y, -1e9), 1)[has].mean()
            loss.backward()
            opt.step()
        return net

    def evaluate(cols, hidden, label, objective="target"):
        picks = np.zeros((3, n), int)
        margin = np.zeros((3, n))
        for seed in range(3):
            folds = np.random.default_rng(seed).permutation(n) % 2
            for f in range(2):
                tr, te = folds != f, folds == f
                net = fit(cols, hidden, 10 * seed + f, tr, objective)
                with torch.no_grad():
                    lp = torch.log_softmax(net(torch.tensor(x[te][..., cols])).squeeze(-1).masked_fill(~torch.tensor(valid[te]), -1e9), 1).numpy()
                picks[seed, te] = lp.argmax(1)
                margin[seed, te] = lp.max(1) - lp[:, 0]  # confidence of moving away from the list's top (the current answer)
        out = []
        for gap in (0.0, 0.5, 1.0, 2.0):
            res = []
            for seed in range(3):
                p = np.where(margin[seed] > gap, picks[seed], 0)
                k = km[np.arange(n), p]
                k = np.where(p == 0, base_km, k)
                res.append([np.mean(k < t) - np.mean(base_km < t) for t in THRESHOLDS] + [np.mean(p != 0)])
            res = np.mean(res, 0)
            out.append(f"gap>{gap:g}: moves {res[-1]:.0%}, " + " ".join(f"<{t:g} {100 * d:+.1f}" for t, d in zip(THRESHOLDS, res[:-1])))
        print(f"  {label:34s} " + " | ".join(out))

    idx = {nm: i for i, nm in enumerate(names)}
    base = [idx[c] for c in ("score_gap", "soft", "rank", "is_top", "log_move")] + [idx[f"retr{j}"] for j in range(9)]
    dens = [idx[c] for c in ("dens0.5", "dens1", "dens5", "count1")]
    sim = [idx[c] for c in ("fwd", "fwd_rel", "full", "full_rel")]
    rev = [idx[c] for c in ("rev", "rev_rel")]
    crop = [idx[c] for c in ("crop", "crop_rel")]
    print("  net change vs keeping the top-1, % of photos at each threshold (2-fold x 3 seeds); 'gap': move only if log P(pick) - log P(top) > gap")
    evaluate(base, 0, "reranker features (linear)")
    evaluate(base + dens, 0, "+ agreement (linear)")
    evaluate(base + dens + sim + rev, 0, "+ similarity + reverse rank (linear)")
    evaluate(base + dens + sim + rev + crop, 0, "+ crops (linear)")
    evaluate(base + dens + sim + rev + crop, 32, "all (MLP 32)")
    evaluate(base, 0, "reranker features, reward (linear)", "reward")
    evaluate(base + dens + sim + rev + crop, 0, "all, reward (linear)", "reward")
    evaluate(base + dens + sim + rev + crop, 32, "all, reward (MLP 32)", "reward")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("search", "evidence", "gate"))
    parser.add_argument("--radius", type=int, default=25, choices=(25, 200))
    args = parser.parse_args(argv)
    {"search": search, "evidence": evidence, "gate": gate}[args.cmd](args.radius)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
