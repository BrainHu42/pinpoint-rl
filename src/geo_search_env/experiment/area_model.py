"""Area-then-spot model (PINPOINT_SET_PLAN.md step 2): choose among a photo's area hypotheses from Pinpoint's own states inside each area.

Each hypothesis (hypothesis_compare's merged hypotheses; aligned row by row with pinpoint_set's zoom_{tag}.npz) is encoded from
  - the restricted search's ~12 candidates: CLS state, score, score - area max, softmax within the area, 9 retrieval features,
    attention-pooled with a learned query;
  - optionally the comparator's hand features (hypothesis_learn._matrix), all of them or a minimal Pinpoint-only subset;
then one transformer layer over the photo's hypotheses and a score. Loss: -log P(a hypothesis < 25 km). The spot inside the chosen area is
the agreement pick or the set-layer pick (hypothesis_learn._evaluate), as the comparator's.

  python -m geo_search_env.experiment.area_model --features none|minimal|all [--sources 0 1 2] [--ensemble 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from . import hypothesis_learn as hl

FEATURES = {"train": hl.OUT / "hypothesis_features_train_e4_vlmhyps_set.npz", "dev3k": hl.OUT / "hypothesis_features_dev3k_e4_vlmhyps_set.npz"}
ZOOM = Path("artifacts/pinpoint_set")


def _minimal(f: dict) -> np.ndarray:
    """Pinpoint-only area features: score mass, pool support, photographers, restricted top-1 (raw and minus the photo's best), area rank,
    hypothesis source, distance to Pinpoint's top-1."""

    lg = lambda x: np.log(np.maximum(x, 1e-12))
    photo = f["photo"]
    cols = []
    for k in ("mass_0.2", "support", "photographers", "top1"):
        x = lg(f[k]) if k != "top1" else np.where(np.isfinite(f[k]), f[k], -1.0)
        mx = np.full(photo.max() + 1, -np.inf)
        np.maximum.at(mx, photo, x)
        cols += [x, x - mx[photo]]
    cols += [(f["rank"] == r).astype(float) for r in range(5)]
    cols += [(f["source"] == s).astype(float) for s in (1, 2)]
    cols.append(np.log1p(f["d_top1"]))
    return np.stack(cols, 1)


def _load(tag: str, features: str, sources: Sequence[int]) -> dict:
    f = dict(np.load(FEATURES[tag]))
    zm = np.load(ZOOM / f"zoom_{tag}.npz")
    assert np.array_equal(zm["photo"], f["photo"]) and np.array_equal(zm["source"], f["source"]), "zoom file not aligned with the features"
    keep = np.isin(f["source"], sources)
    valid = zm["rows"] >= 0
    s = np.where(valid, zm["scores"], -np.inf)
    best = s.max(1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    soft = np.exp(np.where(valid, (s - best) / 0.2, -np.inf))
    soft /= np.maximum(soft.sum(1, keepdims=True), 1e-12)
    cand = np.concatenate([np.stack([np.where(valid, zm["scores"], 0), np.where(valid, s - best, 0), soft], -1), zm["features"].astype(np.float32)], -1)
    if features == "all":
        X, _ = hl._matrix(f, ("vlm",), True)
    elif features == "minimal":
        X = _minimal(f)
    else:
        X = np.zeros((len(f["photo"]), 1))
    f = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) and k != "base_km" else v) for k, v in f.items()}
    photo = f["photo"]
    groups = [np.flatnonzero(photo == p) for p in range(len(f["base_km"]))]
    return {"f": f, "groups": groups, "X": X[keep].astype(np.float32), "hidden": zm["hidden"][keep], "cand": cand[keep].astype(np.float32),
            "valid": valid[keep], "ok": f["km"] < 25}


def _net(n_x: int, n_c: int, dim: int = 64):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, dim))
            self.c = nn.Sequential(nn.Linear(n_c, dim), nn.GELU(), nn.Linear(dim, dim))
            self.query = nn.Parameter(torch.zeros(dim))
            self.x = nn.Sequential(nn.Linear(n_x, dim), nn.GELU(), nn.Linear(dim, dim))
            self.mix = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU())
            self.layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, 0.1, "gelu", batch_first=True, norm_first=True)
            self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        def forward(self, hidden, cand, valid, x, hyp_mask):
            # hidden [B, H, C, 512], cand [B, H, C, n_c], valid [B, H, C], x [B, H, n_x], hyp_mask [B, H]
            t = self.h(hidden) + self.c(cand)
            att = (t @ self.query / t.shape[-1] ** 0.5).masked_fill(~valid, -1e4)
            pooled = (torch.softmax(att, -1).unsqueeze(-1) * t).sum(-2)
            e = self.mix(torch.cat([pooled, self.x(x)], -1))
            e = self.layer(e, src_key_padding_mask=~hyp_mask)
            return self.out(e).squeeze(-1).masked_fill(~hyp_mask, float("-inf"))

    torch.backends.mha.set_fastpath_enabled(False)
    return Net()


def _batch(d, photos, mu, sd, cmu, csd, device):
    import torch

    n_h = max(len(d["groups"][p]) for p in photos)
    c = d["hidden"].shape[1]
    idx = np.full((len(photos), n_h), -1)
    for j, p in enumerate(photos):
        idx[j, :len(d["groups"][p])] = d["groups"][p]
    m = idx >= 0
    safe = np.where(m, idx, 0)
    t = lambda a, dt=torch.float32: torch.as_tensor(a, device=device, dtype=dt)
    valid = d["valid"][safe] & m[..., None]
    valid[..., 0] |= m  # an empty restricted search keeps one (zero) slot so its softmax is defined
    cand = np.where(valid[..., None], (d["cand"][safe] - cmu) / csd, 0.0)
    return (t(d["hidden"][safe]), t(cand), t(valid, torch.bool), t((d["X"][safe] - mu) / sd), t(m, torch.bool)), idx


def _fit(d, photos, seed, epochs=25, device="cuda"):
    import torch

    torch.manual_seed(seed)
    usable = np.asarray([p for p in photos if len(d["groups"][p]) and d["ok"][d["groups"][p]].any() and not d["ok"][d["groups"][p]].all()])
    rows = np.concatenate([d["groups"][p] for p in photos if len(d["groups"][p])])
    mu, sd = d["X"][rows].mean(0), d["X"][rows].std(0) + 1e-6
    cells = d["cand"][rows][d["valid"][rows]]
    cmu, csd = cells.mean(0), cells.std(0) + 1e-6
    net = _net(d["X"].shape[1], d["cand"].shape[-1]).to(device)
    batches = max(1, len(usable) // 128)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * batches, pct_start=0.1)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(usable), batches):
            inputs, idx = _batch(d, b, mu, sd, cmu, csd, device)
            logp = torch.log_softmax(net(*inputs), 1)
            ok = torch.as_tensor(np.where(idx >= 0, d["ok"][np.where(idx >= 0, idx, 0)], False), device=device)
            loss = -torch.logsumexp(logp.masked_fill(~ok, float("-inf")), 1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net, (mu, sd, cmu, csd)


def _predict(nets, norm, d, photos, device="cuda") -> dict[int, int]:
    import torch

    photos = [p for p in photos if len(d["groups"][p])]
    out = {}
    with torch.no_grad():
        for b in np.array_split(np.asarray(photos), max(1, len(photos) // 512)):
            inputs, idx = _batch(d, b, *norm, device)
            prob = sum(torch.softmax(n(*inputs), 1) for n in nets).cpu().numpy()
            for j, p in enumerate(b):
                out[int(p)] = int(idx[j, prob[j].argmax()])
    return out


def run(features: str, sources: Sequence[int], ensemble: int) -> None:
    tr, dv = _load("train", features, sources), _load("dev3k", features, sources)
    n = len(tr["groups"])
    print(f"area model: features {features}, sources {tuple(sources)}, {len(tr['X'])} train hypotheses, ensemble {ensemble}", flush=True)
    folds = np.random.default_rng(0).permutation(n) % 5
    cv = {}
    for k in range(5):
        fits = [_fit(tr, np.flatnonzero(folds != k), seed=10 * k + j) for j in range(ensemble)]
        cv |= _predict([m for m, _ in fits], fits[0][1], tr, np.flatnonzero(folds == k))
    fits = [_fit(tr, np.arange(n), seed=100 + j) for j in range(ensemble)]
    picks = _predict([m for m, _ in fits], fits[0][1], dv, np.arange(len(dv["groups"])))
    # _evaluate indexes hypotheses within the kept rows (e4: all rows of f), so f here is the filtered dict
    for agree in (False, True, "set"):
        hl._evaluate(f"train CV ({agree})", tr["f"], tr["groups"], cv, True, agree=agree)
        hl._evaluate(f"dev3k ({agree})", dv["f"], dv["groups"], picks, True, agree=agree)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", choices=("none", "minimal", "all"), default="minimal")
    parser.add_argument("--sources", type=int, nargs="*", default=(0, 1, 2))
    parser.add_argument("--ensemble", type=int, default=3)
    args = parser.parse_args(argv)
    run(args.features, args.sources, args.ensemble)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
