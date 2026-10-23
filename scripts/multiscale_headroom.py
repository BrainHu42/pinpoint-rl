"""VALUE_PLAN step 3a: is there room at 200 / 750 / 2500 km for multi-scale answers? (cached outputs, no training of the pipeline)

On dev3k and commons26 dev, the value network's lambda 0 answer (`value_agent --dump`) against three coarse answers per photo:
  - pool mass: the pool candidate with the most reranker-score mass (softmax over the pool) within R km of it;
  - probe top: the centre of the PE-G region probe's top cell (track R's probe, R16; 2,048 cells);
  - probe mass: the cell centre with the most probe probability within R km of it.
Gate: keep the answer where its P(< 25 km) >= t, switch to the coarse answer below; per switched fraction, % < 25 / 200 / 750 / 2500 km and the
change vs the answer. Bar (pre-declared): the best switch adds >= 1 pt at 750 km net of the 25 km hits it loses (d750 + d25 >= 1), else stop.
Also, on unsure photos (P(< 25 km) < 0.3): any pool candidate / probe top-5 cell < 750 km vs the answer < 750 km (selection or candidates).

  probe:  PYTHONPATH=src .venv/bin/python scripts/multiscale_headroom.py probe   (refits the probe as pe_probe.crossfit's final fit)
  report: PYTHONPATH=src .venv/bin/python scripts/multiscale_headroom.py report [--seed 7] [--radius 500]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/value")
PROBE = Path("artifacts/retrieval/probe")
PE_PROBE = Path(__file__).resolve().parents[1].parent / "retrieval-track" / "scripts" / "pe_probe.py"  # track R's code, read only
SETS = {"dev3k": ("dev3k_setft_own", "mp16_setft_own"), "commons26dev": ("commons26dev_setft_own", "commons26_setft_own")}
THRESHOLDS = (25, 200, 750, 2500)


def _xyz(ll: np.ndarray) -> np.ndarray:
    la, lo = np.radians(ll[..., 0]), np.radians(ll[..., 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 6371.0 * np.arccos(np.clip((_xyz(a) * _xyz(b)).sum(-1), -1, 1))


def probe(seed: int = 0, folds: int = 5) -> None:
    """Full 2,048-cell log-probs, as pe_probe.crossfit computes them (same seeds, rows and photographer folds; the columns of lesson V5 came
    from it): train photos from the probe trained without their fold's photographers, dev3k and commons26 dev from the probe trained on all
    probe photos -> artifacts/value/probe_logp_{train,dev3k,commons26dev}.npy (photo-set order)."""

    import csv

    import torch

    spec = importlib.util.spec_from_file_location("pe_probe", PE_PROBE)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    p = np.load(PROBE / "photos.npz")
    ok, cent = np.load(PROBE / "ok.npy"), np.load(PROBE / "cells.npy")
    n_p = len(p["ids"])
    base = np.load(PROBE / "pooled.npy", mmap_mode="r")
    extra = np.load(PROBE / "pooled_extra.npz")
    ekeys = list(extra["keys"])
    X = torch.tensor(np.concatenate([np.asarray(base), extra["embeddings"]]), device="cuda")
    y = torch.tensor((pp._xyz(p["latlon"]) @ pp._xyz(cent).T).argmax(1), device="cuda")
    row = {k: i for i, k in enumerate(p["ids"])} | {k: len(base) + i for i, k in enumerate(ekeys)}
    authors = {}
    with pp.CSV.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for rec in r:
            authors[rec[0]] = rec[1]
    probe_auth = np.asarray([authors[i] for i in p["ids"]])
    train = [e["image_id"] for e in json.loads(Path("artifacts/photo_sets/train_setft_own/photos.json").read_text(encoding="utf-8"))]
    t_auth = np.asarray([authors[i] for i in train])
    uniq = np.unique(t_auth)
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % folds))
    t_fold = np.asarray([fold_of[a] for a in t_auth])
    usable = np.flatnonzero(ok[:n_p])
    lp_train = np.zeros((len(train), pp.CELLS), np.float32)
    for k in range(folds):
        held = set(uniq[[fold_of[a] == k for a in uniq]])
        logp = pp._fit_mlp(X, y, usable[~np.isin(probe_auth[usable], list(held))], seed + k)
        J = np.flatnonzero(t_fold == k)
        lp_train[J] = logp(np.asarray([row[train[j]] for j in J]))
        print(f"  fold {k}: {len(J)} train photos", flush=True)
    logp = pp._fit_mlp(X, y, usable, seed)
    lp = {"train": lp_train, "dev3k": logp(n_p + np.arange(len(base) - n_p)),
          "commons26dev": logp(np.asarray([row[k] for k in ekeys if k.startswith("commons26dev:")]))}
    for tag, x in lp.items():
        np.save(OUT / f"probe_logp_{tag}.npy", x.astype(np.float16))
        print(f"-> {OUT / f'probe_logp_{tag}.npy'} {x.shape}", flush=True)


def _mass_point(points: np.ndarray, w: np.ndarray, radius: float) -> np.ndarray:
    """Per photo, the point among `points` [n, k, 2] with the most weight w [n, k] within radius km."""

    d = 6371.0 * np.arccos(np.clip(np.einsum("nid,njd->nij", _xyz(points), _xyz(points)), -1, 1))
    return points[np.arange(len(points)), ((d < radius) * w[:, None, :]).sum(-1).argmax(1)]


def report(seed: int, radius: float) -> None:
    cent = np.load(PROBE / "cells.npy").astype(np.float64)
    cc = 6371.0 * np.arccos(np.clip(_xyz(cent) @ _xyz(cent).T, -1, 1)) < radius
    for tag, (photo_set, pools_name) in SETS.items():
        photos = json.loads((Path("artifacts/photo_sets") / photo_set / "photos.json").read_text(encoding="utf-8"))
        truth = np.asarray([e["truth"] for e in photos], np.float64)
        pools = np.load(Path("artifacts/pinpoint_reference") / f"{pools_name}.npz")
        idx = np.asarray([e["index"] for e in photos])
        ll, sc, valid = pools["latlon"][idx].astype(np.float64), pools["scores"][idx], pools["rows"][idx] >= 0
        assert np.allclose(ll[0, 0], photos[0]["pool"][0], atol=1e-4)
        w = np.exp(np.where(valid, sc - np.where(valid, sc, -np.inf).max(1, keepdims=True), -np.inf))
        w /= w.sum(1, keepdims=True)
        lp = np.load(OUT / f"probe_logp_{tag}.npy").astype(np.float32)
        pr = np.exp(lp)
        ans = np.load(OUT / f"answers_{tag}_seed{seed}.npz")
        km0, p25 = ans["km"], ans["p25"]
        coarse = {
            "pool mass": _km(truth, _mass_point(ll, np.where(valid, w, 0.0), radius)),
            "probe top": _km(truth, cent[lp.argmax(1)]),
            "probe mass": _km(truth, cent[(pr @ cc.T).argmax(1)]),
        }
        acc = lambda k: np.asarray([100 * np.mean(k < r) for r in THRESHOLDS])
        a0 = acc(km0)
        print(f"\n {tag} ({len(photos)} photos), value network lambda 0 (seed {seed}), % < {' / '.join(map(str, THRESHOLDS))} km: "
              + " / ".join(f"{x:.2f}" for x in a0) + f"; coarse answers alone (R {radius:g} km):")
        for name, k in coarse.items():
            print(f"   {name:11s} " + " / ".join(f"{x:.2f}" for x in acc(k)) + f"   oracle best-of(answer, {name}) at 750 km: "
                  f"{100 * np.mean(np.minimum(km0, k) < 750):.2f}")
        print(f"   gated switch (P(< 25 km) of the answer below t -> coarse answer): change in pts at {' / '.join(map(str, THRESHOLDS))} km; net = d750 + d25")
        best = (-np.inf, None)
        for name, k in coarse.items():
            for frac in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                t = np.quantile(p25, frac)
                sw = p25 < t
                d = acc(np.where(sw, k, km0)) - a0
                net = d[2] + d[0]
                best = max(best, (net, f"{name}, {frac:.0%} switched"))
                print(f"     {name:11s} switch {frac:4.0%} (t {t:.3f}): " + " / ".join(f"{x:+.2f}" for x in d) + f"   net {net:+.2f}")
        print(f"   best: {best[1]}, net {best[0]:+.2f} pts (bar +1.00: {'passed' if best[0] >= 1 else 'not met'})")
        unsure = p25 < 0.3
        near = lambda pts: (_km(truth[:, None, :], pts) < 750)
        top5 = cent[np.argsort(-lp, 1)[:, :5]]
        print(f"   unsure photos (P(< 25 km) < 0.3): {unsure.mean():.1%}; among them < 750 km: the answer {np.mean(km0[unsure] < 750):.1%}, "
              f"any pool candidate {np.mean((near(ll) & valid).any(1)[unsure]):.1%}, any probe top-5 cell {np.mean(near(top5).any(1)[unsure]):.1%}, "
              f"probe top cell {np.mean(coarse['probe top'][unsure] < 750):.1%}, pool mass {np.mean(coarse['pool mass'][unsure] < 750):.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", choices=("probe", "report"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--radius", type=float, default=500.0)
    args = ap.parse_args()
    probe() if args.node == "probe" else report(args.seed, args.radius)


if __name__ == "__main__":
    main()
