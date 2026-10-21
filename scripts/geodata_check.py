"""Geodata go / no-go on dev3k: do environmental map attributes (elevation, relief, climate, land cover, coast distance; grid of
scripts/geodata_grid.py) separate the right area from the chosen one on close-call errors (tiers 1, 2, 4 of viz/errors.py)?

Each area is a 25 km disc around the point it would answer at; its attributes are the disc's mean and variance per channel. A photo's
attributes are either the map values at its true location (ceiling) or predictions of a probe on its SigLIP2 embedding (realistic), trained on
MP16 photos (map values at their GPS), dev3k photos and their photographers left out. Area score = Gaussian log-likelihood of the photo's
attributes under the disc (variance = probe residual + disc variance). Reports pairwise AUC (right area over chosen) on close calls, the same
on correct photos (right answer over the runner-up), and the answer head re-ranked with the score (weight cross-fitted over two halves).

  PYTHONPATH=src .venv/bin/python scripts/geodata_check.py   (GPU for the probe; ~10 min)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from geo_search_env.experiment.strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery

GRID = Path("data/geodata/grid")
TRACES = Path("artifacts/viz/traces/dev3k")
OUT = Path("artifacts/geodata")
RES, H, W = 1 / 24, 4320, 8640
GROUPS = {
    "terrain": ["elev", "relief"],
    "climate": ["bio1", "bio4", "bio5", "bio6", "bio12", "bio15"],
    "land cover": ["lc_needleleaf", "lc_evergreen_broadleaf", "lc_deciduous_broadleaf", "lc_mixed_trees", "lc_shrubs", "lc_herbaceous",
                   "lc_cultivated", "lc_flooded", "lc_urban", "lc_snow_ice", "lc_barren", "lc_water"],
    "coast": ["coast_km"],
}
CH = [c for g in GROUPS.values() for c in g]
LOG = {"relief", "bio12", "coast_km"}
N_TRAIN, N_HOLD = 400_000, 20_000
DISC_KM, RIGHT_KM = 25.0, 25.0
CEIL_SD = 0.2  # map error assumed for the ceiling (z units)


def load_grid() -> np.ndarray:
    g = np.empty((len(CH), H, W), np.float32)
    for k, c in enumerate(CH):
        a = np.load(GRID / f"{c}.npy").astype(np.float32)
        g[k] = np.log1p(np.maximum(a, 0)) if c in LOG else a
    return g


def cells(lat, lon):
    r = np.clip(((90 - np.asarray(lat)) / RES).astype(int), 0, H - 1)
    c = (((np.asarray(lon) + 180) / RES).astype(int)) % W
    return r, c


def point(g, lat, lon):
    r, c = cells(lat, lon)
    return g[:, r, c].T  # (n, C), NaN at sea


def disc(g, lat, lon, km=DISC_KM):
    """Mean and variance per channel over the land cells within `km` (NaN if none)."""

    r0, c0 = cells(lat, lon)
    dr = int(np.ceil(km / (6371 * np.pi / 180 * RES))) + 1
    rows = np.arange(r0 - dr, r0 + dr + 1)
    rows = rows[(rows >= 0) & (rows < H)]
    lat_r = np.deg2rad(90 - (rows + 0.5) * RES)
    dc = int(min(W // 2, np.ceil(km / (6371 * np.pi / 180 * RES * max(np.cos(np.deg2rad(lat)), 0.05))) + 1))
    cols = np.arange(c0 - dc, c0 + dc + 1)
    lon_c = np.deg2rad(-180 + ((cols % W) + 0.5) * RES)
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    h = np.sin((lat_r[:, None] - la) / 2) ** 2 + np.cos(la) * np.cos(lat_r[:, None]) * np.sin((lon_c[None, :] - lo) / 2) ** 2
    rr, cc = np.nonzero(2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1))) <= km)
    v = g[:, rows[rr], cols[cc] % W]
    v = v[:, np.isfinite(v[0])]
    if v.shape[1] == 0:
        return np.full(len(CH), np.nan), np.full(len(CH), np.nan)
    return v.mean(1), v.var(1)


def dev_photos() -> list[dict]:
    photos = {p["image_id"]: p for p in json.loads(Path("artifacts/photo_sets/dev3k/photos.json").read_text())}
    out = []
    for r in json.loads((TRACES / "index.json").read_text()):
        t = json.loads((TRACES / f"{r['i']}.json").read_text())
        t["row"] = photos[t["image_id"]]["row"]
        out.append(t)
    return out


def authors_by_row(mp16) -> np.ndarray:
    vocab, authors = {}, []
    with MP16_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        a = next(reader).index("AUTHOR")
        for row in reader:
            authors.append(vocab.setdefault(row[a], len(vocab)))
    return np.asarray(authors, np.int64)[mp16["row_index"]]


def train_probe(g, mp16, dev_rows):
    """MLP from SigLIP2 embedding to the z-scored map attributes at the photo's GPS; returns (predict fn, z mean, z sd, residual sd)."""

    auth = authors_by_row(mp16)
    banned = np.isin(auth, auth[dev_rows])
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(~banned)
    rows = np.sort(rng.choice(cand, int((N_TRAIN + N_HOLD) * 1.4), replace=False))
    y = point(g, mp16["latlon"][rows, 0], mp16["latlon"][rows, 1])
    ok = np.isfinite(y).all(1)
    rows, y = rows[ok][: N_TRAIN + N_HOLD], y[ok][: N_TRAIN + N_HOLD]
    print(f"probe: {len(rows)} MP16 photos ({banned.sum()} rows of dev3k photographers left out)", flush=True)
    x = np.asarray(mp16["embeddings"][rows], np.float32)
    mu, sd = y.mean(0), y.std(0) + 1e-6
    z = (y - mu) / sd
    perm = rng.permutation(len(rows))
    tr, ho = perm[N_HOLD:], perm[:N_HOLD]
    dev = torch.device("cuda")
    X, Z = torch.tensor(x, device=dev), torch.tensor(z, device=dev, dtype=torch.float32)
    X = torch.nn.functional.normalize(X, dim=1)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1024), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(1024, 1024),
                              torch.nn.GELU(), torch.nn.Linear(1024, len(CH))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs, bs = 8, 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(tr) // bs + 1))
    trt = torch.tensor(tr, device=dev)
    for ep in range(epochs):
        net.train()
        for b in trt[torch.randperm(len(trt), device=dev)].split(bs):
            loss = torch.nn.functional.mse_loss(net(X[b]), Z[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            res = (net(X[ho]) - Z[ho]).pow(2).mean(0).sqrt().cpu().numpy()
        print(f"  epoch {ep + 1}: held-out residual sd (z) mean {res.mean():.3f}", flush=True)

    def predict(e):
        with torch.no_grad():
            return net(torch.nn.functional.normalize(torch.tensor(np.asarray(e, np.float32), device=dev), dim=1)).cpu().numpy()

    return predict, mu, sd, res


def loglik(x, m, v, sd):
    """Gaussian log-likelihood of attributes x (z units) under a disc (mean m, variance v, z units), noise sd, over channel subset."""

    var = sd ** 2 + v
    return -0.5 * np.sum((x - m) ** 2 / var + np.log(var), axis=-1)


def area_points(t):
    """Per area slot: (lat, lon, km, on_table, p) at the final step, at the zoom the final state uses."""

    last = t["agent"]["0"]["steps"][-1]
    out = []
    for s, a in enumerate(t["areas"]):
        loc = a["zoomed"] if last["zoomed"][s] else a["unzoomed"]
        p = last["answer"][s]
        on = p is not None and (last["read"] or a["source"] != "clue") and loc.get("lat") is not None
        out.append((loc.get("lat"), loc.get("lon"), loc.get("km"), on, p or 0.0))
    return out


def auc(d):
    d = np.asarray(d)
    return float(np.mean((d > 0) + 0.5 * (d == 0))) if len(d) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_grid()
    photos = dev_photos()
    errs = {e["i"]: e for e in json.loads((TRACES / "errors.json").read_text())["errors"]}
    mp16 = _memmap_gallery(MP16_EMBED)
    dev_rows = np.array([t["row"] for t in photos])
    predict, mu, sd, res = train_probe(g, mp16, dev_rows)
    groups = {"all": np.arange(len(CH))} | {k: np.array([CH.index(c) for c in v]) for k, v in GROUPS.items()}
    print("held-out R^2 per channel:", " ".join(f"{c} {1 - r * r:.2f}" for c, r in zip(CH, res)))

    pred = predict(mp16["embeddings"][np.sort(dev_rows)])[np.argsort(np.argsort(dev_rows))]
    truth = (point(g, [t["truth"]["lat"] for t in photos], [t["truth"]["lon"] for t in photos]) - mu) / sd

    # per photo: area scores under both photo attribute sources
    rec = []
    for k, t in enumerate(photos):
        pts = area_points(t)
        d = [disc(g, la, lo) if la is not None else (np.full(len(CH), np.nan),) * 2 for la, lo, *_ in pts]
        m = np.array([(x[0] - mu) / sd for x in d])
        v = np.array([x[1] / sd ** 2 for x in d])
        sc = {}
        for name, idx in groups.items():
            sc[("real", name)] = loglik(pred[k][idx], m[:, idx], v[:, idx], res[idx])
            tr = truth[k][idx]
            sc[("ceil", name)] = loglik(tr, m[:, idx], v[:, idx], CEIL_SD) if np.isfinite(tr).all() else np.full(len(pts), np.nan)
        rec.append((t, pts, sc, k))

    lines = []
    # close-call pairs: right area vs chosen (tiers 1, 2, 4)
    for tiers, label in (((1, 2, 4), "close calls (tiers 1, 2, 4)"), ((1,), "tier 1"), ((2,), "tier 2"), ((4,), "tier 4")):
        diffs = {key: [] for key in rec[0][2]}
        n = 0
        for t, pts, sc, k in rec:
            e = errs.get(t["i"])
            if not e or e["tier"] not in tiers or e.get("right_slot") is None:
                continue
            r, c = e["right_slot"], t["agent"]["0"]["final"]["slot"]
            if r == c:
                continue
            # the right slot's point: the zoom at which it is right (tier 1 rights can be at another zoom than the final state's)
            a = t["areas"][r]
            loc = next((x for x in (a["zoomed"], a["unzoomed"]) if x.get("km") is not None and x["km"] < RIGHT_KM), None)
            if loc is None:
                continue
            mr, vr = disc(g, loc["lat"], loc["lon"])
            n += 1
            for (src, name), s in sc.items():
                idx = groups[name]
                x = pred[k][idx] if src == "real" else truth[k][idx]
                noise = res[idx] if src == "real" else CEIL_SD
                sr = loglik(x, ((mr - mu) / sd)[idx], (vr / sd ** 2)[idx], noise)
                if np.isfinite(sr) and np.isfinite(s[c]):
                    diffs[(src, name)].append(sr - s[c])
        lines.append(f"\n{label}: {n} photos, AUC right area over chosen (ceiling = map at truth | realistic = SigLIP2 probe)")
        for name in groups:
            lines.append(f"  {name:11s} ceiling {auc(diffs[('ceil', name)]):.3f}   realistic {auc(diffs[('real', name)]):.3f}")

    # correct photos: the right answer vs the runner-up on the table
    diffs = {key: [] for key in rec[0][2]}
    for t, pts, sc, k in rec:
        fin = t["agent"]["0"]["final"]
        if fin["km"] is None or fin["km"] >= RIGHT_KM:
            continue
        c = fin["slot"]
        others = [s for s, p in enumerate(pts) if p[3] and s != c and p[2] is not None and p[2] >= RIGHT_KM]
        if not others:
            continue
        o = max(others, key=lambda s: pts[s][4])
        for key, s in sc.items():
            if np.isfinite(s[c]) and np.isfinite(s[o]):
                diffs[key].append(s[c] - s[o])
    lines.append(f"\ncorrect photos: {len(diffs[('real', 'all')])} with a wrong area on the table; AUC right answer over the most likely wrong area")
    for name in groups:
        lines.append(f"  {name:11s} ceiling {auc(diffs[('ceil', name)]):.3f}   realistic {auc(diffs[('real', name)]):.3f}")

    # re-rank: log p + w * score, w cross-fitted over two halves
    def acc(subset, w, src="real", name="all"):
        ok = []
        for t, pts, sc, _ in (rec[i] for i in subset):
            on = [s for s, p in enumerate(pts) if p[3]]
            if not on:
                ok.append(False)
                continue
            s_geo = np.nan_to_num(sc[(src, name)][on], nan=np.nanmin(sc[(src, name)][on]) if np.isfinite(sc[(src, name)][on]).any() else 0.0)
            tot = np.log(np.array([pts[s][4] for s in on]) + 1e-6) + w * s_geo
            km = pts[on[int(np.argmax(tot))]][2]
            ok.append(km is not None and km < RIGHT_KM)
        return np.array(ok)

    ws = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    halves = [np.arange(len(rec))[i::2] for i in range(2)]
    lines.append("\nanswer head re-ranked with the score (dev3k, all photos; w chosen on the other half)")
    for src in ("real", "ceil"):
        base_all, new_all = [], []
        for h in range(2):
            fit, ev = halves[1 - h], halves[h]
            w = max(ws, key=lambda w: acc(fit, w, src).mean())
            base_all.append(acc(ev, 0, src))
            new_all.append(acc(ev, w, src))
            lines.append(f"  {src}: half {h} w={w}")
        b, nw = np.concatenate(base_all), np.concatenate(new_all)
        lines.append(f"  {src}: {100 * b.mean():.1f}% -> {100 * nw.mean():.1f}% ({100 * (nw.mean() - b.mean()):+.2f} pts; fixes {(nw & ~b).sum()}, "
                     f"breaks {(b & ~nw).sum()})")
        for w in ws[1:]:
            a = acc(np.arange(len(rec)), w, src)
            b0 = acc(np.arange(len(rec)), 0, src)
            lines.append(f"    w={w}: {100 * (a.mean() - b0.mean()):+.2f} pts (fixes {(a & ~b0).sum()}, breaks {(b0 & ~a).sum()})")
    text = "\n".join(lines)
    print(text)
    (OUT / "geodata_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
