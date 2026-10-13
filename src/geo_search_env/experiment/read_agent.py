"""Reading clues as a costly action (lesson 76; merged-pipeline step 3): the agent first sees only Pinpoint's areas (clue features zeroed, no
clue-place tokens) plus a free glance at the photo (its SigLIP2 embedding, which Pinpoint computes anyway), and decides whether to pay for
the clue reader (cached 4B observe + text search + clue judge). Reading reveals the clue places as area tokens and the clue evidence on every
area; then 2 zooms (Pinpoint's top areas) and the answer head (hypothesis_agent, trained on random read / no-read states, gated pick).

Read policy: a contextual bandit with both outcomes known for every training photo (answer with and without reading, out of fold), so the
value of reading g in {-1, 0, +1} is learned directly (3-way classifier over glance + Pinpoint-only features) and the policy reads when
E[g] exceeds the cost per read; one model traces the accuracy-vs-reads curve. Compared with never / always reading, random reading, a
Pinpoint-confidence heuristic and the oracle (read exactly when it helps). Train: 5-fold CV (outcomes and policy out of fold); dev3k: models
trained on all of train.

  python -m geo_search_env.experiment.read_agent [--ckpt-tag setft_a] [--budget 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .hypothesis_agent import _answer, _rollout, _sub, _view, fit_answer, load

RATES = (0.05, 0.1, 0.2, 0.3, 0.5)


def glance(tag: str, ckpt_tag: str) -> np.ndarray:
    """Each photo's raw SigLIP2 embedding (MP16 embedding cache row), L2-normalised: the free look before deciding to read."""

    from .strategy_search import MP16_EMBED

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{ckpt_tag}" / "photos.json").read_text(encoding="utf-8"))
    manifest = json.loads((MP16_EMBED / "manifest.json").read_text(encoding="utf-8"))
    emb = np.memmap(MP16_EMBED / manifest["files"]["embeddings"], dtype=np.dtype(manifest["dtypes"]["embeddings"]), mode="r",
                    shape=(manifest["num_samples"], manifest["embedding_dim"]))
    rows = np.asarray([e["row"] for e in photos])
    order = np.argsort(rows)
    x = np.empty((len(rows), emb.shape[1]), np.float32)
    x[order] = emb[rows[order]]
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def outcomes(net, d, photos: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """km of the answer without (read = False) and with reading, `budget` zooms of Pinpoint's top areas, gated pick."""

    import torch

    out = {False: [], True: []}
    with torch.no_grad():
        for bi in np.array_split(photos, max(1, len(photos) // 1024)):
            b = _sub(d, torch.as_tensor(bi, device="cuda"))
            for read in (False, True):
                v = _view(b, torch.full((len(bi),), read, device="cuda"))
                z, _ = _rollout(net, v, budget, "top")
                out[read].append(_answer(net, v, z).cpu().numpy())
    return np.concatenate(out[False]), np.concatenate(out[True])


def summary(d) -> np.ndarray:
    """Pinpoint-only state for the read decision: the unread features of the 3 best-ranked Pinpoint areas and the number of areas."""

    pre = d["pre_noclue"].float().cpu().numpy()
    area = (d["mask"] & d["is_area"]).cpu().numpy()
    rank = np.where(area, d["rank"].cpu().numpy(), 99.0)
    order = np.argsort(rank, 1)[:, :3]
    top = np.take_along_axis(pre, order[..., None], 1) * np.take_along_axis(area, order, 1)[..., None]
    return np.concatenate([top.reshape(len(pre), -1), area.sum(1, keepdims=True).astype(np.float32)], 1)


def fit_value(x: np.ndarray, gain: np.ndarray, seed: int, epochs: int = 20):
    """3-way classifier of the gain from reading (-1 / 0 / +1); returns a function x -> E[gain]."""

    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    xt = torch.as_tensor((x - mu) / sd, device="cuda")
    yt = torch.as_tensor(gain + 1, device="cuda", dtype=torch.long)
    weight = torch.as_tensor(len(yt) / (3 * np.bincount(gain + 1, minlength=3).clip(1)), device="cuda", dtype=torch.float32) ** 0.5
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(len(xt)), max(1, len(xt) // 512)):
            loss = nn.functional.cross_entropy(net(xt[bi]), yt[bi], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def value(xx):
        with torch.no_grad():
            p = torch.softmax(net(torch.as_tensor((xx - mu) / sd, device="cuda")), 1).cpu().numpy()
        return p[:, 2] - p[:, 0]

    return value


def curve(name: str, km0: np.ndarray, km1: np.ndarray, base: np.ndarray, value: np.ndarray, confidence: np.ndarray) -> None:
    """Accuracy at matched read rates for the learned policy, random reading, the confidence heuristic and the oracle."""

    from .strategy_search import _bootstrap

    r0, r1 = km0 < 25, km1 < 25
    n = len(r0)
    print(f"\n {name} ({n} photos), % within 25 km (1 km) vs released Pinpoint {100 * np.mean(base < 25):.1f}%:")
    print(f"   never read  {100 * r0.mean():5.1f} ({100 * np.mean(km0 < 1):.1f});  always read {100 * r1.mean():5.1f} ({100 * np.mean(km1 < 1):.1f})"
          f";  reading helps {np.mean(r1 & ~r0):.1%}, hurts {np.mean(r0 & ~r1):.1%} of photos;  oracle {100 * np.mean(r0 | r1):.1f} at {np.mean(r1 & ~r0):.1%} reads")
    for rate in RATES:
        k = int(round(rate * n))
        rows = []
        for label, score in (("learned", value), ("Pinpoint unsure", -confidence)):
            read = np.zeros(n, bool)
            read[np.argsort(-score, kind="stable")[:k]] = True
            km = np.where(read, km1, km0)
            c = _bootstrap((km < 25).astype(float) - r0)
            rows.append(f"{label} {100 * np.mean(km < 25):5.1f} ({100 * c[0]:+.2f} [{100 * c[1]:+.2f}, {100 * c[2]:+.2f}])")
        rand = r0.mean() + rate * (r1.mean() - r0.mean())
        print(f"   read {rate:4.0%}: " + ";  ".join(rows) + f";  random {100 * rand:5.1f}")


def run(ckpt_tag: str, budget: int) -> None:
    tr = load("train", ckpt_tag)
    dv = load("dev3k", ckpt_tag, stats=tr["stats"])
    gl_tr, gl_dv = glance("train", ckpt_tag), glance("dev3k", ckpt_tag)
    x_tr = np.concatenate([gl_tr, summary(tr)], 1)
    x_dv = np.concatenate([gl_dv, summary(dv)], 1)
    conf_tr, conf_dv = summary(tr)[:, 0], summary(dv)[:, 0]  # log support of Pinpoint's best area (normalised)
    n = tr["n"]
    folds = np.random.default_rng(0).permutation(n) % 5
    km0, km1 = np.zeros(n), np.zeros(n)
    print(f"read agent ({ckpt_tag}): {n} train photos, {budget} zooms; answer heads out of fold", flush=True)
    for k in range(5):
        fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        net = fit_answer(tr, fit, seed=10 * k, read_mode="random")
        km0[held], km1[held] = outcomes(net, tr, held, budget)
    gain = (km1 < 25).astype(int) - (km0 < 25).astype(int)
    net = fit_answer(tr, np.arange(n), seed=100, read_mode="random")
    d0, d1 = outcomes(net, dv, np.arange(dv["n"]), budget)
    g = gl_tr.shape[1]
    np.savez(Path("artifacts/agent") / f"read_agent_{ckpt_tag}.npz", km0=km0, km1=km1, d0=d0, d1=d1, x_tr=x_tr, x_dv=x_dv, folds=folds)
    for label, cols in (("glance + Pinpoint", slice(None)), ("glance only", slice(0, g)), ("Pinpoint only", slice(g, None))):
        value_tr = np.zeros(n)
        for k in range(5):
            fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            value_tr[held] = fit_value(x_tr[fit][:, cols], gain[fit], seed=k)(x_tr[held][:, cols])
        print(f"\n=== read policy on {label} ===")
        curve("train CV", km0, km1, tr["base_km"], value_tr, conf_tr)
        curve("dev3k", d0, d1, dv["base_km"], fit_value(x_tr[:, cols], gain, seed=100)(x_dv[:, cols]), conf_dv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-tag", default="setft_a")
    parser.add_argument("--budget", type=int, default=2)
    args = parser.parse_args(argv)
    run(args.ckpt_tag, args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
