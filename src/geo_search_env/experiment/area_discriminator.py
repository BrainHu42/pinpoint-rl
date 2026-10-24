# Lesson 143 follow-up: use dense photos discriminatively in the answer head. Per dev3k photo, a class-balanced multinomial logistic
# regression over its candidate areas, fitted on the gallery photos near each area's centre (SigLIP2), classifies the query; its log
# probability joins the frozen answer head's (lesson 109 pipeline, setft_own, traces in artifacts/viz/traces/dev3k) with two weights chosen by
# cross-validation. A nearest-neighbour score from the same photos is the comparison (lesson 142: neighbours can't use density).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.area_discriminator {features,evaluate} [--n 1000] [--tag index]

"""features: per photo and area (<= 5), the nearest --n index photos within 25 km of the area centre (never the query photo or its
          photographer); multinomial logistic over areas with >= MIN_PHOTOS photos (balanced classes, L2), the query's log probability per
          area; also its best and mean-of-top-10 cosine per area -> artifacts/area_discriminator/features_<tag>_n<N>.npz
evaluate: 5-fold CV (x seeds) of score = a * log p_head + w * feature (+ c * log photos) per area, fitted by the likelihood of the right
          areas; % of dev3k within 25 km (answer at the chosen area's centre), fixes / breaks vs the head, paired bootstrap SE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

OUT = Path("artifacts/area_discriminator")
TRACES = Path("artifacts/viz/traces/dev3k")
PHOTO_SET = Path("artifacts/photo_sets/dev3k_setft_own")
QUERIES = Path("artifacts/density_probe/queries.npz")
AREAS = 7  # up to 5 Pinpoint areas + clue areas
RIGHT_KM = 25.0
AREA_KM = 25.0
MIN_PHOTOS = 5


def _hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _traces() -> dict[str, np.ndarray]:
    n = len(json.loads((PHOTO_SET / "photos.json").read_text(encoding="utf-8")))
    out = {"p": np.zeros((n, AREAS)), "lat": np.full((n, AREAS), np.nan), "lon": np.full((n, AREAS), np.nan), "km": np.full((n, AREAS), np.inf),
           "final_km": np.zeros(n), "final_slot": np.zeros(n, np.int64)}
    for i in range(n):
        t = json.loads((TRACES / f"{i}.json").read_text(encoding="utf-8"))
        final = t["agent"]["0"]["final"]
        answer = t["agent"]["0"]["steps"][-1]["answer"]
        for a in t["areas"]:
            k = a["slot"]
            out["lat"][i, k], out["lon"][i, k] = a["center"]
            out["km"][i, k] = _hav(t["truth"]["lat"], t["truth"]["lon"], *a["center"])
            out["p"][i, k] = answer[k]
        out["final_km"][i], out["final_slot"][i] = final["km"], final["slot"]
    return out


def _logistic(x, y, w8, classes: int, query, l2: float, device):
    import torch

    x, y, w8 = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device), torch.as_tensor(w8, device=device)
    w = torch.zeros((x.shape[1], classes), device=device, requires_grad=True)
    b = torch.zeros(classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (w8 * torch.nn.functional.cross_entropy(x @ w + b, y, reduction="none")).sum() + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.log_softmax(torch.as_tensor(query, device=device) @ w + b, -1).cpu().numpy()


def features(n_near: int, tag: str, l2: float) -> None:
    import torch
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import _discover_index, _unit_xyz
    from .strategy_search import MP16_CSV, MP16_EMBED
    import csv

    tr = _traces()
    photos = json.loads((PHOTO_SET / "photos.json").read_text(encoding="utf-8"))
    q = dict(np.load(QUERIES))
    dev = np.flatnonzero(q["set"] == "dev3k")
    q_emb = q["emb"][dev].astype(np.float32)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    count, width = manifest["shapes"]["raw_image_embeddings"]
    latlon = np.memmap(index_dir / manifest["files"]["latlon_deg"], dtype=np.float32, mode="r", shape=(count, 2)).astype(np.float64)
    cache_rows = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    t0 = time.time()
    raw = np.fromfile(index_dir / manifest["files"]["raw_image_embeddings"], dtype=np.int8).reshape(count, width)  # ~6 GB, read once
    scales = np.fromfile(index_dir / manifest["files"]["raw_image_scales"], dtype=np.float16).astype(np.float32)
    image_ids = np.asarray((MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines())
    author = {}
    with MP16_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            author[r["IMG_ID"]] = r["AUTHOR"]
    row_author = np.asarray([author.get(i, "") for i in image_ids[cache_rows]])
    row_pid = np.asarray([int(i.rsplit("_", 1)[-1][:-4]) for i in image_ids[cache_rows]])
    tree = cKDTree(_unit_xyz(latlon))
    print(f"loaded index, authors, tree ({time.time() - t0:.0f}s)", flush=True)
    chord = 2 * np.sin(AREA_KM / 6371.0 / 2)
    device = "cuda"
    n = len(photos)
    out = {"logq": np.full((n, AREAS), np.nan), "best": np.full((n, AREAS), np.nan), "top10": np.full((n, AREAS), np.nan),
           "photos": np.zeros((n, AREAS))}
    t0 = time.time()
    for i, e in enumerate(photos):
        qa, qp = author.get(e["image_id"], "?"), int(e["image_id"].rsplit("_", 1)[-1][:-4])
        xs, ys, ks = [], [], []
        for k in range(AREAS):
            if not np.isfinite(tr["lat"][i, k]):
                continue
            d, r = tree.query(_unit_xyz(np.asarray([[tr["lat"][i, k], tr["lon"][i, k]]]))[0], k=n_near + 50, distance_upper_bound=chord)
            r = r[np.isfinite(d)]
            r = r[(row_author[r] != qa) & (row_pid[r] != qp)][:n_near]
            out["photos"][i, k] = len(r)
            if len(r) == 0:
                continue
            x = raw[r].astype(np.float32) * scales[r, None]
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
            sim = x @ q_emb[i]
            out["best"][i, k], out["top10"][i, k] = sim.max(), np.sort(sim)[-10:].mean()
            if len(r) >= MIN_PHOTOS:
                xs.append(x); ys.append(np.full(len(r), len(ks))); ks.append(k)
        if len(ks) >= 2:
            x, y = np.concatenate(xs), np.concatenate(ys)
            w8 = np.concatenate([np.full(len(v), 1.0 / (len(ks) * len(v)), np.float32) for v in ys])  # balanced: equal priors per area
            logq = _logistic(x, y, w8, len(ks), q_emb[i], l2, device)
            out["logq"][i, ks] = logq
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n} photos, {(time.time() - t0) / (i + 1):.2f} s each", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"features_{tag}_n{n_near}.npz", **out, n_near=n_near, l2=l2)
    print(f"saved; photos with a fitted discriminator {100 * np.isfinite(out['logq']).any(1).mean():.1f}%")


YFCC_KM = (1.0, 5.0, 25.0)


def density() -> None:
    """Flickr photo density around each area from YFCC100M's 48M geotags (no images; ~10x the gallery's density): geotagged photos within
    YFCC_KM of the area centre, never counting the query photo or any photo by its photographer -> artifacts/area_discriminator/yfcc_density.npz"""

    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import _unit_xyz
    from .density_probe import GEO

    tr = _traces()
    q = dict(np.load(QUERIES))
    dev = np.flatnonzero(q["set"] == "dev3k")
    q_uid = q["uid"][dev]
    g = np.load(GEO)
    lat, lon, uid = g["lat"].astype(np.float64), g["lon"].astype(np.float64), g["uid"]
    t0 = time.time()
    tree = cKDTree(_unit_xyz(np.stack([lat, lon], 1)), leafsize=64, balanced_tree=False, compact_nodes=False)
    print(f"YFCC tree over {len(lat):,} photos ({time.time() - t0:.0f}s)", flush=True)
    order = np.argsort(uid, kind="stable")
    uid_sorted = uid[order]
    n = len(tr["p"])
    counts = np.zeros((n, AREAS, len(YFCC_KM)))
    for i in range(n):
        ks = np.flatnonzero(np.isfinite(tr["lat"][i]))
        if not len(ks):
            continue
        centres = _unit_xyz(np.stack([tr["lat"][i, ks], tr["lon"][i, ks]], 1))
        own = order[np.searchsorted(uid_sorted, q_uid[i]): np.searchsorted(uid_sorted, q_uid[i], "right")] if q_uid[i] >= 0 else np.zeros(0, np.int64)
        for j, km in enumerate(YFCC_KM):
            chord = 2 * np.sin(km / 6371.0 / 2)
            c = np.asarray(tree.query_ball_point(centres, chord, return_length=True), np.float64)
            if len(own):  # the photographer's photos (the query photo among them)
                d = _hav(lat[own][None], lon[own][None], tr["lat"][i, ks][:, None], tr["lon"][i, ks][:, None])
                c -= (d < km).sum(1)
            counts[i, ks, j] = c
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT / "yfcc_density.npz", counts=counts, km=np.asarray(YFCC_KM))
    print("saved")


def _fit(logp, f, right, mask):
    """Weights (a, w) maximising sum log P(right areas) with P ∝ exp(a log p + w f) over valid areas."""

    import torch

    lp, ff = torch.as_tensor(logp[mask]), torch.as_tensor(f[mask])
    r, v = torch.as_tensor(right[mask]), torch.as_tensor(np.isfinite(logp[mask]))
    theta = torch.tensor([1.0] + [0.0] * ff.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=200, line_search_fn="strong_wolfe")
    has = r.any(1)

    def closure():
        opt.zero_grad()
        s = theta[0] * torch.nan_to_num(lp, nan=0.0) + (ff * theta[1:]).sum(-1)
        s = s.masked_fill(~v, -1e9)
        ls = torch.log_softmax(s, 1)
        loss = -torch.logsumexp(ls.masked_fill(~r, -1e9), 1)[has].mean() + 1e-3 * (theta[1:] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach().numpy()


def evaluate(n_near: int, tag: str, seeds: int = 5, spatial: bool = False) -> None:
    tr = _traces()
    z = np.load(OUT / f"features_{tag}_n{n_near}.npz")
    truth = np.asarray([e["truth"] for e in json.loads((PHOTO_SET / "photos.json").read_text(encoding="utf-8"))], np.float64)
    cell = (np.floor(truth[:, 0] / 10).astype(int) + 9) * 36 + (np.floor(truth[:, 1] / 10).astype(int) + 18)  # spatial folds: 10-degree cells
    yfcc = np.load(OUT / "yfcc_density.npz")["counts"] if (OUT / "yfcc_density.npz").exists() else None
    p = tr["p"]
    valid = np.isfinite(tr["lat"]) & (p > 0)
    logp = np.where(valid, np.log(np.clip(p, 1e-9, 1)), np.nan)
    right = valid & (tr["km"] < RIGHT_KM)
    base_right = tr["final_km"] < RIGHT_KM
    head_pick = np.where(valid, p, -1).argmax(1)
    agree = (head_pick == tr["final_slot"]).mean()

    def col(x, fill):
        x = np.where(valid & np.isfinite(x), x, fill)
        return x

    variants = {
        "nearest neighbour (best cosine)": [col(z["best"], 0.0)],
        "neighbours (mean top-10 cosine)": [col(z["top10"], 0.0)],
        "best cosine + log photos + interaction": [col(z["best"], 0.0), col(np.log1p(z["photos"]), 0.0),
                                                   col(z["best"] * np.log1p(z["photos"]), 0.0)],
        **({} if yfcc is None else {
            "best cosine + YFCC density (1 / 5 / 25 km) + interactions": [col(z["best"], 0.0)]
            + [col(np.log1p(yfcc[..., j]), 0.0) for j in range(yfcc.shape[-1])]
            + [col(z["best"] * np.log1p(yfcc[..., j]), 0.0) for j in range(yfcc.shape[-1])],
            "best cosine + gallery and YFCC density + interactions": [col(z["best"], 0.0), col(np.log1p(z["photos"]), 0.0),
                                                                      col(z["best"] * np.log1p(z["photos"]), 0.0)]
            + [col(np.log1p(yfcc[..., j]), 0.0) for j in range(yfcc.shape[-1])]
            + [col(z["best"] * np.log1p(yfcc[..., j]), 0.0) for j in range(yfcc.shape[-1])],
        }),
        "best cosine relative to the photo's best area": [col(z["best"] - np.nanmax(np.where(valid, z["best"], np.nan), 1, keepdims=True), 0.0)],
        "discriminator log q": [col(z["logq"], 0.0), col(np.isfinite(z["logq"]).astype(float), 0.0)],
        "discriminator + top-10 + log photos": [col(z["logq"], 0.0), col(np.isfinite(z["logq"]).astype(float), 0.0), col(z["top10"], 0.0),
                                                col(np.log1p(z["photos"]), 0.0)],
    }
    n = len(p)
    lines = [f"Area discriminator on dev3k ({tag}, nearest {n_near} index photos per area within {AREA_KM:g} km): head {100 * base_right.mean():.1f}% "
             f"< {RIGHT_KM:g} km (head argmax = final slot on {100 * agree:.1f}%); fitted discriminator on "
             f"{100 * np.isfinite(z['logq']).any(1).mean():.1f}% of photos; area oracle {100 * right.any(1).mean():.1f}%",
             f"  5-fold CV x {seeds} seeds; switching answers at the new area's centre"]
    rng = np.random.default_rng(0)
    for name, cols in variants.items():
        f = np.stack(cols, -1)
        acc, fixes, breaks, thetas = [], [], [], []
        for seed in range(seeds):
            fold = np.random.default_rng(seed).integers(0, 5, n) if not spatial else (np.random.default_rng(seed).permutation(100_000)[cell] % 5)
            pick = np.zeros(n, np.int64)
            for k in range(5):
                th = _fit(logp, f, right, fold != k)
                thetas.append(th)
                m = fold == k
                s = th[0] * np.nan_to_num(logp[m], nan=0.0) + (f[m] * th[1:]).sum(-1)
                pick[m] = np.where(valid[m], s, -np.inf).argmax(1)
            ok = np.where(pick == tr["final_slot"], base_right, right[np.arange(n), pick])
            acc.append(ok.mean()); fixes.append((ok & ~base_right).sum()); breaks.append((~ok & base_right).sum())
            last = ok
        diff = last.astype(float) - base_right
        boot = [diff[rng.integers(0, n, n)].mean() for _ in range(1000)]
        lines.append(f"    {name:<38} {100 * np.mean(acc):5.2f}%  ({100 * (np.mean(acc) - base_right.mean()):+.2f}, SE {100 * np.std(boot):.2f}; "
                     f"fixes {np.mean(fixes):.0f} / breaks {np.mean(breaks):.0f})  weights {np.round(np.mean(thetas, 0), 3).tolist()}")
    text = "\n".join(lines)
    print(text)
    (OUT / f"evaluate_{tag}_n{n_near}{'_spatial' if spatial else ''}.txt").write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("features", "evaluate", "density"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--tag", default="index")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--spatial", action="store_true", help="evaluate: folds by 10-degree cells of the truth")
    args = parser.parse_args(argv)
    if args.cmd == "density":
        density()
    elif args.cmd == "features":
        features(args.n, args.tag, args.l2)
    else:
        evaluate(args.n, args.tag, spatial=args.spatial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
