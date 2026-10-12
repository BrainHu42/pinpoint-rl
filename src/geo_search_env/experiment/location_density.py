"""Learned location distribution over the retrained reranker's candidates (replaces the hard-coded agreement rule; lesson 72).

For each photo, a small set model over its ~46 candidates (CLS states of the retrained reranker + pool scalars, distance-biased attention;
models/set_head.py layers) outputs a weight w_i and a spread sigma_i per candidate:

    p(x) = (1 - eps) * sum_i w_i N(x; c_i, sigma_i^2 I) + eps / area(Earth)        (local flat Gaussians on great-circle km)

trained on train photos by the negative log-likelihood of the true location (eps = 0.1 keeps photos whose truth is far from every candidate
from dominating). Readouts, all from p:
  - peak: the candidate where p is highest (sharp kernels win: good at 1 km);
  - mass: the candidate whose 25 km disc holds the most probability (numerical integration on a polar grid; good at 25 km);
  - proposals: candidates by 25 km mass with non-maximum suppression at 25 km (the areas to zoom into); coverage = a proposal < 25 km.
With --coverage, proposals are also trained for coverage: a second head scores candidates with a soft best-of-K loss (option 2).
With --mcl K, multiple-choice learning instead: K pick heads, each trained with the expected-hit loss, but each photo's loss goes (mostly) to
the head that does best on it (winner takes all), so the heads specialise on different hypotheses; proposals = the K heads' picks.

  python -m geo_search_env.experiment.location_density [--ckpt-tag setft_a] [--epochs 30] [--coverage]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/pinpoint_set")
EPS = 0.1
EARTH_KM2 = 4 * math.pi * 6371.0088 ** 2
SIGMA_KM = (0.2, 1000.0)


def load(ckpt_tag: str) -> dict:
    z = np.load(OUT / f"mp16_32_8_8_{ckpt_tag}.npz" if ckpt_tag else OUT / "mp16_32_8_8.npz")
    idx = {t: np.asarray([p["index"] for p in json.loads((Path("artifacts/photo_sets") / t / "photos.json").read_text(encoding="utf-8"))])
           for t in ("train", "dev3k")}
    keep = np.concatenate([idx["train"], idx["dev3k"]])
    d = {k: z[k][keep] for k in ("rows", "latlon", "scores", "hidden", "features", "truth")}
    d["valid"] = d["rows"] >= 0
    d["latlon"] = np.nan_to_num(d["latlon"])
    d["scores"] = np.where(d["valid"], d["scores"], 0.0).astype(np.float32)
    d["n_train"] = len(idx["train"])
    return d


def _km(a, b):
    """Great-circle km between [..., 2] degree tensors (broadcast)."""

    import torch

    la1, lo1, la2, lo2 = (torch.deg2rad(t) for t in (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    h = torch.sin((la2 - la1) / 2) ** 2 + torch.cos(la1) * torch.cos(la2) * torch.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * torch.asin(torch.sqrt(h.clamp(0, 1)))


def _net(dim: int = 128, layers: int = 2, coverage: bool = False, mcl: int = 0):
    import torch
    import torch.nn as nn

    from ..models.set_head import DIST_EDGES_KM, SetHead, pairwise_km

    class DensityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = SetHead(512, dim, 4, layers)  # hidden_in, scalar_in, layers, norm (its readout is unused)
            self.weight = nn.Linear(dim, 1)
            self.log_sigma = nn.Linear(dim, 1)
            self.cover = nn.Linear(dim, 1) if coverage else None
            self.heads = nn.Linear(dim, mcl) if mcl else None
            for lin in (self.weight, self.log_sigma) + ((self.cover,) if coverage else ()):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            nn.init.constant_(self.log_sigma.bias, math.log(10.0))
            if mcl:  # small random init breaks the symmetry between heads
                nn.init.normal_(self.heads.weight, std=0.05)
                nn.init.zeros_(self.heads.bias)

        def forward(self, hidden, scores, latlon, features, mask):
            t = self.trunk
            km = pairwise_km(latlon)
            buckets = torch.bucketize(km, t.edges)
            x = t.hidden_in(hidden.float()) + t.scalar_in(t.scalars(scores, km, features, mask))
            for layer in t.layers:
                x = layer(x, buckets, mask)
            x = t.norm(x)
            logit = (scores / 0.2 + self.weight(x).squeeze(-1)).masked_fill(~mask, float("-inf"))
            sigma = torch.exp(self.log_sigma(x).squeeze(-1)).clamp(*SIGMA_KM)
            cover = None if self.cover is None else (scores / 0.2 + self.cover(x).squeeze(-1)).masked_fill(~mask, float("-inf"))
            if self.heads is not None:  # [B, C, K] pick logits of the K heads, returned in place of the coverage head
                cover = (scores.unsqueeze(-1) / 0.2 + self.heads(x)).masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return logit, sigma, cover

    return DensityNet()


def log_density(logit, sigma, centres, points):
    """log p at points [B, P, 2] for mixtures (logit [B, C], sigma [B, C], centres [B, C, 2]) with the uniform floor."""

    import torch

    d = _km(points.unsqueeze(2), centres.unsqueeze(1))  # [B, P, C]
    logw = torch.log_softmax(logit, 1).unsqueeze(1)
    comp = logw - 0.5 * (d / sigma.unsqueeze(1)) ** 2 - torch.log(2 * math.pi * sigma.unsqueeze(1) ** 2)
    mix = torch.logsumexp(comp, 2) + math.log(1 - EPS)
    return torch.logaddexp(mix, torch.full_like(mix, math.log(EPS / EARTH_KM2)))


def _batch(d, b, device="cuda"):
    import torch

    t = lambda x, dt=torch.float32: torch.as_tensor(x[b], device=device, dtype=dt)
    return (t(d["hidden"]), t(d["scores"]), t(d["latlon"]), t(d["features"]), t(d["valid"], torch.bool)), t(d["truth"])


def _coverage_loss(cover, truth_km, k: int = 5):
    """Soft best-of-K: -log P(at least one of K draws from softmax(cover) lands < 25 km) with draws treated as independent:
    1 - (1 - m)^K where m is the softmax mass on candidates < 25 km."""

    import torch

    good = truth_km < 25
    m = (torch.softmax(cover, 1) * good).sum(1).clamp(1e-6, 1 - 1e-6)
    use = good.any(1)
    return -torch.log(1 - (1 - m) ** k)[use].mean() if use.any() else cover.sum() * 0


def _mcl_loss(heads, truth_km, relax: float = 0.05):
    """Winner-takes-all expected-hit loss over K pick heads [B, C, K]."""

    import torch

    reward = ((truth_km < 25).float() + (truth_km < 1).float()) / 2
    use = reward.sum(1) > 0
    if not use.any():
        return heads.sum() * 0
    logp = torch.log_softmax(heads[use], 1)
    per = -torch.logsumexp(logp + torch.log(reward[use].clamp(min=1e-12)).unsqueeze(-1), 1)  # [B, K]
    return (per.min(1).values + relax * per.mean(1)).mean()


def fit(d, photos, seed, epochs=30, coverage=False, device="cuda", mcl=0):
    import torch

    torch.manual_seed(seed)
    net = _net(coverage=coverage, mcl=mcl).to(device)
    batches = max(1, len(photos) // 256)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(photos), batches):
            inputs, truth = _batch(d, b, device)
            logit, sigma, cover = net(*inputs)
            loss = -log_density(logit, sigma, inputs[2], truth[:, None]).mean()
            if mcl:
                loss = loss + _mcl_loss(cover, _km(truth[:, None], inputs[2]))
            elif coverage:
                loss = loss + _coverage_loss(cover, _km(truth[:, None], inputs[2]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net


def _disc_grid(r: float):
    """Polar quadrature of a disc of radius r km: (offsets [P, 2] in km east / north, weights [P] summing to 1)."""

    pts, wts = [(0.0, 0.0)], [(0.1) ** 2]
    edges = [0.1, 0.4, 0.7, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        rad, n = (lo + hi) / 2 * r, 12
        for a in range(n):
            th = 2 * math.pi * (a + 0.5 * (len(pts) % 2)) / n
            pts.append((rad * math.cos(th), rad * math.sin(th)))
            wts.append((hi ** 2 - lo ** 2) / n)
    return np.asarray(pts, np.float32), np.asarray(wts, np.float32) / np.sum(wts)


def readouts(nets, d, photos, k=5, device="cuda") -> dict:
    """Per photo: peak / mass / weight picks (candidate index), proposals (K by 25 km mass with NMS; and by the coverage head if trained)."""

    import torch

    off, wts = (torch.as_tensor(x, device=device) for x in _disc_grid(25.0))
    out = {"peak": [], "mass": [], "weight": [], "prop": [], "cprop": []}
    with torch.no_grad():
        for b in np.array_split(np.asarray(photos), max(1, len(photos) // 256)):
            inputs, _ = _batch(d, b, device)
            mask, cen = inputs[4], inputs[2]
            res = [n(*inputs) for n in nets]
            logit = torch.logsumexp(torch.stack([torch.log_softmax(r[0], 1) for r in res]), 0)  # ensemble: mixture of mixtures
            sigma = torch.stack([r[1] for r in res]).mean(0)
            peak = log_density(logit, sigma, cen, cen).masked_fill(~mask, float("-inf"))
            lat = cen[..., 0:1]
            pts = torch.stack([cen[..., 0:1] + off[None, None, :, 1] / 111.2,
                               cen[..., 1:2] + off[None, None, :, 0] / (111.2 * torch.cos(torch.deg2rad(lat)).clamp(min=0.05))], -1)  # [B, C, P, 2]
            B, C, P, _ = pts.shape
            dens = log_density(logit, sigma, cen, pts.reshape(B, C * P, 2)).reshape(B, C, P).exp()
            mass = (dens * wts).sum(-1) * math.pi * 25.0 ** 2
            mass = mass.masked_fill(~mask, -1.0)
            pair = _km(cen.unsqueeze(2), cen.unsqueeze(1))

            def nms(score):
                alive, picks = mask.clone(), []
                for _ in range(k):
                    j = score.masked_fill(~alive, float("-inf")).argmax(1)
                    has = alive.any(1)
                    picks.append(torch.where(has, j, -1))
                    alive &= ~(pair[torch.arange(B), j] < 25) & has[:, None]
                return torch.stack(picks, 1)

            out["peak"].append(peak.argmax(1).cpu().numpy())
            out["mass"].append(mass.argmax(1).cpu().numpy())
            out["weight"].append(logit.argmax(1).cpu().numpy())
            out["prop"].append(nms(mass).cpu().numpy())
            if res[0][2] is not None and res[0][2].dim() == 3:  # MCL heads: each head's pick, heads ordered by confidence, duplicates (< 25 km) skipped
                probs = torch.softmax(res[0][2], 1)  # [B, C, K]
                conf, pick = probs.max(1)  # [B, K]
                order = conf.argsort(1, descending=True)
                picks = torch.gather(pick, 1, order)
                keep = torch.full_like(picks, -1)
                for j in range(picks.shape[1]):
                    cand = picks[:, j]
                    dup = torch.zeros_like(cand, dtype=torch.bool)
                    for i in range(j):
                        prev = keep[:, i]
                        dup |= (prev >= 0) & (pair[torch.arange(B), cand, prev.clamp(min=0)] < 25)
                    keep[:, j] = torch.where(dup, -1, cand)
                out["cprop"].append(keep.cpu().numpy())
            elif res[0][2] is not None:
                out["cprop"].append(nms(torch.stack([torch.log_softmax(r[2], 1) for r in res]).mean(0)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items() if v}


def report(name, d, photos, picks):
    from .strategy_search import _bootstrap

    tl = d["truth"][photos].astype(np.float64)
    ll = d["latlon"][photos].astype(np.float64)
    lat1, lon1, lat2, lon2 = np.radians(tl[:, None, 0]), np.radians(tl[:, None, 1]), np.radians(ll[..., 0]), np.radians(ll[..., 1])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    km = np.where(d["valid"][photos], 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), np.inf)
    r = np.arange(len(photos))
    top1 = km[r, np.where(d["valid"][photos], d["scores"][photos], -np.inf).argmax(1)]
    print(f" {name}: retrained top-1 {np.mean(top1 < 25):.1%} / {np.mean(top1 < 1):.1%}; candidate-list oracle {np.mean((km < 25).any(1)):.1%}")
    for key in ("weight", "peak", "mass"):
        k_ = km[r, picks[key]]
        c25, c1 = _bootstrap((k_ < 25).astype(float) - (top1 < 25)), _bootstrap((k_ < 1).astype(float) - (top1 < 1))
        print(f"   {key + ' pick':12s} < 25 km {np.mean(k_ < 25):.1%} ({100 * c25[0]:+.1f} [{100 * c25[1]:+.1f}, {100 * c25[2]:+.1f}]); "
              f"< 1 km {np.mean(k_ < 1):.1%} ({100 * c1[0]:+.1f} [{100 * c1[1]:+.1f}, {100 * c1[2]:+.1f}])")
    for key in ("prop", "cprop"):
        if key not in picks:
            continue
        p = picks[key]
        cov = [np.mean(np.any(np.stack([(p[:, j] >= 0) & (km[r, np.maximum(p[:, j], 0)] < 25) for j in range(kk)], 1), 1)) for kk in (1, 2, 3, 5)]
        print(f"   {('proposals' if key == 'prop' else 'coverage head'):12s} right area among top 1 / 2 / 3 / 5: " + " / ".join(f"{c:.1%}" for c in cov))


def agreement_baseline(d, photos):
    """The hard-coded rule (box 25 km, T 0.1 and 0.2) proposals, for the same coverage numbers."""

    from .pinpoint_tool import areas

    for T in (0.1, 0.2):
        picks = np.full((len(photos), 5), -1)
        for j, p in enumerate(photos):
            v = np.flatnonzero(d["valid"][p])
            a = areas(d["scores"][p][v], d["latlon"][p][v], temp=T)[:5]
            picks[j, :len(a)] = v[[x.founder for x in a]]
        yield T, picks


def run(ckpt_tag: str, epochs: int, coverage: bool, ensemble: int = 1, mcl: int = 0) -> None:
    d = load(ckpt_tag)
    n = d["n_train"]
    tr, dv = np.arange(n), np.arange(n, len(d["truth"]))
    print(f"location density on {ckpt_tag or 'released'} candidates: {n} train, {len(dv)} dev3k photos, epochs {epochs}, coverage head {coverage}", flush=True)
    for T, picks in agreement_baseline(d, dv):
        report(f"dev3k, agreement rule T {T}", d, dv, {"weight": picks[:, 0], "peak": picks[:, 0], "mass": picks[:, 0], "prop": picks})
    folds = np.random.default_rng(0).permutation(n) % 5
    cv = {}
    for k in range(5):
        nets = [fit(d, tr[folds != k], seed=10 * k + j, epochs=epochs, coverage=coverage, mcl=mcl) for j in range(ensemble)]
        held = tr[folds == k]
        for key, v in readouts(nets, d, held).items():
            cv.setdefault(key, np.zeros((n,) + v.shape[1:], int))[held] = v
    report("train, 5-fold CV", d, tr, cv)
    nets = [fit(d, tr, seed=100 + j, epochs=epochs, coverage=coverage, mcl=mcl) for j in range(ensemble)]
    report("dev3k (trained on all of train)", d, dv, readouts(nets, d, dv))
    import torch

    torch.save({"models": [m.state_dict() for m in nets], "coverage": coverage, "ckpt_tag": ckpt_tag},
               OUT / f"location_density_{ckpt_tag or 'released'}{'_cov' if coverage else ''}{f'_mcl{mcl}' if mcl else ''}.pt")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-tag", default="setft_a")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--mcl", type=int, default=0, help="multiple-choice learning with this many pick heads")
    args = parser.parse_args(argv)
    run(args.ckpt_tag, args.epochs, args.coverage, args.ensemble, args.mcl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
