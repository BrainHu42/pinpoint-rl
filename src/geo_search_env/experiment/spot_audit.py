# FLICKR_PLAN.md step 1: can dense same-era Flickr photos at a candidate place tell the right candidate from the look-alike the pipeline
# chose? Worked back from the truth on dev3k's close calls, with YFCC100M (a Creative Commons subset of Flickr, data/yfcc100m/geo.npz) as an
# offline lower bound for what the live Flickr API returns at a point. No API calls: photos come from Flickr's static image CDN.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.spot_audit {shortlist,fetch,report}

"""Close calls: dev3k errors (viz/errors.py tiers, artifacts/viz/traces/dev3k) where a right place is on the table: 1 policy, 2 coin flip,
4 ranked low (a right area; its centre is a pool photo) and 5 in pool (the nearest retrieval candidate within 25 km). Per photo, three points:
  truth  the true GPS (ceiling: the agent never knows it)
  right  the right area's centre / nearest right candidate (what the agent could check)
  wrong  the pipeline's answer
Reference photos per point: the --k nearest YFCC100M photos within --radius km (Flickr accuracy >= 13, no videos), never the query photo or
any photo by its photographer; each flagged as in Pinpoint's index (MP16) or new. Score of a point: best MegaLoc cosine between the query
and its references, after dropping near-duplicates of the query (cosine >= DUP; counted separately).

shortlist: points and their references -> artifacts/spot_audit/shortlist.npz
fetch:     downloads each unique reference once (resumable), MegaLoc-embeds it -> artifacts/spot_audit/megaloc.npz
report:    right vs wrong (and truth vs wrong): paired win rate (AUC), clean wins / losses at a threshold, by number of references, source
           (all / index only / new only) and tier -> artifacts/spot_audit/report.txt"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from .density_probe import GEO, MIN_ACCURACY, _haversine, _index_rows, _mp16_photo_ids

OUT = Path("artifacts/spot_audit")
TRACES = Path("artifacts/viz/traces/dev3k")
QUERIES = Path("artifacts/density_probe/queries.npz")  # dev3k photo ids and photographer uids (YFCC vocabulary)
MEGALOC_Q = Path("artifacts/agent/megaloc_queries_dev3k_setft_own.npy")
PHOTO_SET = Path("artifacts/photo_sets/dev3k_setft_own")
SETFT_CKPT = Path("artifacts/pinpoint_train/setft_own/checkpoints/ckpt_final.pt")  # the pipeline's retrained reranker
CLOSE_TIERS = (1, 2, 4, 5)
KINDS = ("truth", "right", "wrong")
DUP = 0.9  # MegaLoc cosine at which a reference counts as a copy of the query
KS = (5, 10, 25, 50)
CDN_PAUSE = 120.0  # seconds every thread waits after a 403 / 429 from the image CDN, doubling on repeats
MEGALOC_DIM = 8448
THRESH = 0.3  # a "clean" match (lesson 138: covered views matched at 0.47-0.58; random pairs ~0.03)


def _points() -> tuple[list[dict], np.ndarray]:
    errors = json.loads((TRACES / "errors.json").read_text(encoding="utf-8"))["errors"]
    pts, rows = [], []
    for e in errors:
        if e["tier"] not in CLOSE_TIERS:
            continue
        t = json.loads((TRACES / f"{e['i']}.json").read_text(encoding="utf-8"))
        final = t["agent"]["0"]["final"]
        if e.get("right_slot") is not None and e["tier"] != 5:
            right = tuple(t["areas"][e["right_slot"]]["center"])
        else:  # in pool (or a policy fix without a right area slot): the nearest right retrieval candidate
            near = [c for c in t["retrained"]["candidates"] if c["km"] < 25]
            if not near:
                continue
            c = min(near, key=lambda c: c["km"])
            right = (c["lat"], c["lon"])
        for kind, (la, lo) in zip(KINDS, ((t["truth"]["lat"], t["truth"]["lon"]), right, (final["lat"], final["lon"]))):
            pts.append({"i": e["i"], "tier": e["tier"], "kind": kind, "lat": float(la), "lon": float(lo)})
            rows.append(e["i"])
    return pts, np.asarray(rows)


def shortlist(k: int, radius: float) -> None:
    pts, _ = _points()
    q = dict(np.load(QUERIES))
    dev = np.flatnonzero(q["set"] == "dev3k")
    q_pid, q_uid = q["photo_id"][dev], q["uid"][dev]  # photo-set order
    g = np.load(GEO)
    g = {key: g[key] for key in ("photo_id", "uid", "lat", "lon", "video", "accuracy", "server", "secret")}
    sel = np.flatnonzero((g["video"] == 0) & (g["accuracy"] >= MIN_ACCURACY))
    lat, lon = g["lat"][sel], g["lon"][sel]
    size = 0.02
    cell = (np.floor(lat / size).astype(np.int64) + 10000) * 100000 + (np.floor(lon / size).astype(np.int64) + 20000)
    o = np.argsort(cell, kind="stable")
    cell_s = cell[o]
    index_pid = np.sort(_mp16_photo_ids()[_index_rows()])
    refs = {"point": [], "photo_id": [], "server": [], "secret": [], "dist": [], "in_index": []}
    for p_idx, p in enumerate(pts):
        ci, cj = int(np.floor(p["lat"] / size)), int(np.floor(p["lon"] / size))
        si = int(np.ceil(radius / (111 * size))) + 1
        sj = int(np.ceil(radius / (111 * size * max(np.cos(np.radians(p["lat"])), 0.05)))) + 1
        parts = []
        for di in range(-si, si + 1):
            for dj in range(-sj, sj + 1):
                key = (ci + di + 10000) * 100000 + (cj + dj + 20000)
                parts.append(o[np.searchsorted(cell_s, key): np.searchsorted(cell_s, key, "right")])
        idx = sel[np.concatenate(parts)]
        uid = q_uid[p["i"]]
        idx = idx[(g["photo_id"][idx] != q_pid[p["i"]]) & ((g["uid"][idx] != uid) | (uid < 0))]
        d = _haversine(p["lat"], p["lon"], g["lat"][idx], g["lon"][idx])
        keep = d < radius
        idx, d = idx[keep], d[keep]
        top = np.argsort(d, kind="stable")[:k]
        idx, d = idx[top], d[top]
        pid = g["photo_id"][idx]
        j = np.clip(np.searchsorted(index_pid, pid), 0, len(index_pid) - 1)
        refs["point"] += [p_idx] * len(idx)
        refs["photo_id"] += pid.tolist()
        refs["server"] += g["server"][idx].tolist()
        refs["secret"] += g["secret"][idx].tolist()
        refs["dist"] += d.tolist()
        refs["in_index"] += (index_pid[j] == pid).tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "shortlist.npz", **{f"p_{key}": np.asarray([p[key] for p in pts]) for key in ("i", "tier", "kind", "lat", "lon")},
             **{key: np.asarray(v) for key, v in refs.items()}, k=k, radius=radius)
    n = np.bincount(np.asarray(refs["point"], np.int64), minlength=len(pts))
    print(f"{len(pts) // 3} photos, {len(pts)} points; {len(refs['point']):,} references ({len(set(refs['photo_id'])):,} unique), "
          f"in the index {100 * np.mean(refs['in_index']):.0f}%")
    for kind in KINDS:
        m = np.asarray([p["kind"] == kind for p in pts])
        print(f"  {kind}: median {np.median(n[m]):.0f} references, none {100 * (n[m] == 0).mean():.1f}%, full ({k}) {100 * (n[m] >= k).mean():.1f}%")


class Throttled(Exception):
    pass


_pause_until = [0.0]


def _get(args) -> bytes | None:
    """One CDN image: bytes, None if gone (404 / 410 / Flickr's 'unavailable' placeholder). HTTP 403 / 429 is throttling, not absence: all
    threads pause and retry (the CDN answered every request with 403 / 429 for a while after a burst on 2026-10-24)."""

    import urllib.error
    import urllib.request

    pid, server, secret = args
    url = f"https://live.staticflickr.com/{server}/{pid}_{secret.decode() if isinstance(secret, bytes) else secret}.jpg"
    for attempt in range(8):
        wait = _pause_until[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pinpoint-rl research (brainhu42@gmail.com)"}),
                                        timeout=30) as r:
                return None if "photo_unavailable" in r.geturl() else r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            if e.code in (403, 429):
                _pause_until[0] = max(_pause_until[0], time.time() + CDN_PAUSE * 2 ** min(attempt, 4))
                continue
            time.sleep(2 ** attempt)
        except OSError:
            time.sleep(2 ** attempt)
    raise Throttled(url)


def fetch(workers: int, batch: int) -> None:
    import torch

    from .clue_verify import _megaloc_tf
    from .megaloc_gate import _model

    s = np.load(OUT / "shortlist.npz")
    ids, first = np.unique(s["photo_id"], return_index=True)
    server, secret = s["server"][first], s["secret"][first]
    part = OUT / "megaloc_part.npz"
    emb, done = None, 0
    if part.exists():
        p = np.load(part)
        emb, done = p["emb"], int(p["done"])
        print(f"resuming at {done:,}/{len(ids):,}")
    model, tf = _model(), _megaloc_tf()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for a in range(done, len(ids), batch):
            chunk = range(a, min(a + batch, len(ids)))
            blobs = list(pool.map(_get, [(ids[i], server[i], secret[i]) for i in chunk]))  # Throttled ends the run; rerun resumes
            xs = [tf(b) for b in blobs]
            ok = [n for n, x in enumerate(xs) if x is not None]
            if emb is None:
                emb = np.full((len(ids), MEGALOC_DIM), np.nan, np.float16)
            if ok:
                with torch.inference_mode():
                    emb[[chunk[n] for n in ok]] = model(torch.stack([xs[n] for n in ok]).cuda()).float().cpu().numpy()
            if (a - done) // batch % 20 == 0 or chunk[-1] == len(ids) - 1:
                np.savez(part, emb=emb, done=chunk[-1] + 1)
                rate = (chunk[-1] + 1 - done) / (time.time() - t0)
                print(f"  {chunk[-1] + 1:,}/{len(ids):,}, {rate:.0f}/s, gone {100 * np.isnan(emb[:chunk[-1] + 1, 0]).mean():.1f}%", flush=True)
    np.savez(OUT / "megaloc.npz", ids=ids, emb=emb)
    part.unlink(missing_ok=True)
    print(f"embedded {np.isfinite(emb[:, 0]).sum():,}/{len(ids):,} ({time.time() - t0:.0f}s)")


def siglip(workers: int, batch: int) -> None:
    """SigLIP2 embeddings (Pinpoint's raw image space) of the new references (not in the index, fetched by `fetch`), for `rerank`;
    downloaded again since only MegaLoc embeddings were kept -> artifacts/spot_audit/siglip.npz."""

    import torch

    from .embed_cache import _embed, _load_model, _pixels

    s, m = np.load(OUT / "shortlist.npz"), np.load(OUT / "megaloc.npz")
    new = ~s["in_index"].astype(bool)
    ids, first = np.unique(s["photo_id"][new], return_index=True)
    server, secret = s["server"][new][first], s["secret"][new][first]
    alive = np.isfinite(m["emb"][np.searchsorted(m["ids"], ids), 0])
    ids, server, secret = ids[alive], server[alive], secret[alive]
    part = OUT / "siglip_part.npz"
    emb, done = np.full((len(ids), 1536), np.nan, np.float16), 0
    if part.exists():
        p = np.load(part)
        emb, done = p["emb"], int(p["done"])
        print(f"resuming at {done:,}/{len(ids):,}")
    model, processor = _load_model()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for a in range(done, len(ids), batch):
            chunk = range(a, min(a + batch, len(ids)))
            blobs = list(pool.map(_get, [(ids[i], server[i], secret[i]) for i in chunk]))
            pix = [(n, _pixels(processor, b)) for n, b in enumerate(blobs)]
            pix = [(n, x) for n, x in pix if x is not None]
            if pix:
                emb[[chunk[n] for n, _ in pix]] = _embed(model, torch.stack([x for _, x in pix]))
            if (a - done) // batch % 20 == 0 or chunk[-1] == len(ids) - 1:
                np.savez(part, emb=emb, done=chunk[-1] + 1)
                print(f"  {chunk[-1] + 1:,}/{len(ids):,}, {(chunk[-1] + 1 - done) / (time.time() - t0):.0f}/s", flush=True)
    np.savez(OUT / "siglip.npz", ids=ids, emb=emb)
    part.unlink(missing_ok=True)
    print(f"embedded {np.isfinite(emb[:, 0]).sum():,}/{len(ids):,} ({time.time() - t0:.0f}s)")


def rerank(radius: float = 1.0, cap: int = 3000) -> None:
    """The new photos used the way Pinpoint uses photos: as gallery candidates scored by the retrained reranker (setft_own, pools 32 8 8).
    Per close call one pool over the index photos within `radius` km of the right point and of the wrong pick (the nearest `cap` each; never
    the query photo or its photographer): A = index only, B = index + the new YFCC references at those points, A' = A with every index
    photo's GPS embedding replaced by its nearest other index photo's (the new photos borrow their nearest index photo's GPS embedding, as
    the location tower isn't ported; A' vs A measures what that costs). Where does the reranker's top-1 land, and how much softmax mass
    falls nearer the right point? -> artifacts/spot_audit/rerank.txt"""

    import os

    os.environ.setdefault("PINPOINT_RERANKER_CKPT", str(SETFT_CKPT))
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import PinpointReranker, _unit_xyz
    from ..tools.pinpoint import NVME_INDEX
    from .strategy_search import MP16_EMBED
    from .yfcc_names import _mp16_authors

    s = dict(np.load(OUT / "shortlist.npz"))
    if (OUT / "siglip.npz").exists():
        sg = dict(np.load(OUT / "siglip.npz"))
    else:  # plumbing on a fetch in progress (same id order as siglip())
        m, new = np.load(OUT / "megaloc.npz"), ~s["in_index"].astype(bool)
        ids = np.unique(s["photo_id"][new])
        sg = {"ids": ids[np.isfinite(m["emb"][np.searchsorted(m["ids"], ids), 0])], "emb": np.load(OUT / "siglip_part.npz")["emb"]}
        print("PARTIAL: siglip.npz not written yet")
    photos = json.loads((PHOTO_SET / "photos.json").read_text(encoding="utf-8"))
    q = dict(np.load(QUERIES))
    dev = np.flatnonzero(q["set"] == "dev3k")
    q_emb, q_pid = q["emb"][dev].astype(np.float32), q["photo_id"][dev]
    model = PinpointReranker(index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    model.widen((32, 8, 8))
    latlon = np.asarray(model.latlon, np.float64)
    tree = cKDTree(_unit_xyz(latlon))
    chord = 2 * np.sin(radius / 6371.0 / 2)
    kind, pts = s["p_kind"], np.flatnonzero(s["p_kind"] != "truth")
    near = {}
    for p in pts:
        xyz = _unit_xyz(np.asarray([[s["p_lat"][p], s["p_lon"][p]]]))[0]
        d, r = tree.query(xyz, k=cap, distance_upper_bound=chord)
        near[p] = r[np.isfinite(d)]
    idx_rows = np.unique(np.concatenate(list(near.values())))
    image_ids = np.asarray((MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines())
    row_ids = image_ids[model.cache_indices[idx_rows]]
    authors = _mp16_authors(set(row_ids) | {p["image_id"] for p in photos})
    row_author = np.asarray([authors.get(i, "") for i in row_ids])
    row_pid = np.asarray([int(i.rsplit("_", 1)[-1][:-4]) for i in row_ids])

    # new rows: references not in the index with a SigLIP2 embedding, located from the YFCC metadata
    new_pid = sg["ids"][np.isfinite(sg["emb"][:, 0])]
    g = np.load(GEO)
    order = np.argsort(g["photo_id"])
    at = order[np.searchsorted(g["photo_id"], new_pid, sorter=order)]
    new_ll = np.stack([g["lat"][at], g["lon"][at]], 1).astype(np.float64)
    new_raw = sg["emb"][np.searchsorted(sg["ids"], new_pid)].astype(np.float32)
    new_raw /= np.linalg.norm(new_raw, axis=1, keepdims=True)
    new_proj = np.concatenate([model.encode_queries(new_raw[a:a + 4096])[1].cpu().numpy() for a in range(0, len(new_raw), 4096)])
    _, nearest = tree.query(_unit_xyz(new_ll), k=1)
    scale = np.abs(new_raw).max(1) / 127.0
    _, second = tree.query(_unit_xyz(latlon[idx_rows]), k=2)

    n_idx = len(idx_rows)
    table = {"image": np.concatenate([np.asarray(model.image[idx_rows]), new_proj.astype(np.float16)]),
             "gps": np.concatenate([np.asarray(model.gps[idx_rows]), np.asarray(model.gps[nearest])]),
             "raw": np.concatenate([np.asarray(model.raw[idx_rows]), np.round(new_raw / scale[:, None]).astype(np.int8)]),
             "raw_scales": np.concatenate([np.asarray(model.raw_scales[idx_rows]), scale.astype(np.float16)]),
             "latlon": np.concatenate([latlon[idx_rows], new_ll]).astype(np.float32)}
    gps_alt = np.concatenate([np.asarray(model.gps[second[:, 1]]), table["gps"][n_idx:]])
    cache = np.concatenate([model.cache_indices[idx_rows], np.full(len(new_pid), -1)])
    pos = {r: n for n, r in enumerate(idx_rows)}
    new_pos = {pid: n_idx + n for n, pid in enumerate(new_pid)}

    def install(gps: np.ndarray) -> None:
        model.image, model.gps, model.raw, model.raw_scales, model.latlon = table["image"], gps, table["raw"], table["raw_scales"], table["latlon"]
        model.cache_indices = cache
        model.load_resident()

    photo_i = s["p_i"][kind == "right"]
    p_right, p_wrong = np.flatnonzero(kind == "right"), np.flatnonzero(kind == "wrong")
    pools = {"A": [], "B": []}
    n_new = []
    for i, pr, pw in zip(photo_i, p_right, p_wrong):
        qa, qp = authors.get(photos[i]["image_id"], "?"), q_pid[i]
        base = [pos[r] for p in (pr, pw) for r in near[p] if row_author[pos[r]] != qa and row_pid[pos[r]] != qp]
        extra = [new_pos[pid] for p in (pr, pw) for pid in s["photo_id"][(s["point"] == p) & ~s["in_index"].astype(bool)] if pid in new_pos]
        pools["A"].append(np.unique(np.asarray(base, np.int64)))
        pools["B"].append(np.unique(np.asarray(base + extra, np.int64)))
        n_new.append(len(set(extra)))
    q_rows = q_emb[photo_i]
    ll_r = np.stack([s["p_lat"][p_right], s["p_lon"][p_right]], 1)
    ll_w = np.stack([s["p_lat"][p_wrong], s["p_lon"][p_wrong]], 1)

    def run(pool: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        top_right, mass, top_new, has = (np.zeros(len(pool)) for _ in range(4))
        for a in range(0, len(pool), 64):
            out = model.rank_subset(q_rows[a:a + 64], pool[a:a + 64])
            for j in range(len(out.rows)):
                v = out.rows[j] >= 0
                if not v.any():
                    continue
                ll = out.latlon[j][v]
                d_r = _haversine(ll_r[a + j, 0], ll_r[a + j, 1], ll[:, 0], ll[:, 1])
                d_w = _haversine(ll_w[a + j, 0], ll_w[a + j, 1], ll[:, 0], ll[:, 1])
                sc = out.scores[j][v]
                w = np.exp(sc - sc.max())
                w /= w.sum()
                has[a + j], top_right[a + j] = 1, d_r[0] < d_w[0]
                mass[a + j] = w[d_r < d_w].sum()
                top_new[a + j] = out.rows[j][0] >= n_idx
        return top_right, mass, top_new, has

    install(table["gps"])
    res = {name: run(pools[name]) for name in ("A", "B")}
    install(gps_alt)
    res["A'"] = run(pools["A"])
    ok = (res["A"][3] > 0) & (res["B"][3] > 0)
    n_new = np.asarray(n_new)
    lines = [f"Reranker over the right / wrong neighbourhoods (index photos within {radius:g} km, nearest {cap} each), dev3k close calls: "
             f"{int(ok.sum())} of {len(photo_i)} photos; new photos per pool median {np.median(n_new):.0f}"]
    for name in ("A", "A'", "B"):
        t, m_, nw, _ = res[name]
        lines.append(f"  {name:<2} top-1 nearer the right point {100 * t[ok].mean():5.1f}%  mean mass right {m_[ok].mean():.3f}"
                     + (f"  top-1 a new photo {100 * nw[ok].mean():.1f}%" if name == "B" else ""))
    ta, tb = res["A"][0][ok], res["B"][0][ok]
    lines.append(f"  A -> B: top-1 moves wrong -> right {int(((ta == 0) & (tb == 1)).sum())}, right -> wrong {int(((ta == 1) & (tb == 0)).sum())}; "
                 f"mass right +{(res['B'][1][ok] - res['A'][1][ok]).mean():.4f} (photos with >= 20 new: "
                 f"{(res['B'][1] - res['A'][1])[ok & (n_new >= 20)].mean():+.4f}, n {int((ok & (n_new >= 20)).sum())})")
    ta2 = res["A'"][0][ok]
    lines.append(f"  A -> A' (borrowed GPS embeddings): top-1 changes on {int((ta != ta2).sum())} photos, mass right "
                 f"{(res[chr(65) + chr(39)][1][ok] - res['A'][1][ok]).mean():+.4f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "rerank.txt").write_text(text + "\n", encoding="utf-8")


def discriminate(radius: float = 1.0, cap: int = 3000, l2: float = 1.0) -> None:
    """The photos used discriminatively instead of as neighbours: per close call, fit a class-balanced logistic regression on SigLIP2
    embeddings that separates the photos near the right point from those near the wrong pick (index photos within `radius` km, nearest
    `cap`; never the query photo or its photographer), with and without the new YFCC photos, and classify the query. Compared with
    nearest neighbour (best cosine per side) and prototypes (cosine to each side's mean). A side needs >= 5 photos. Also: how often is the
    best MegaLoc match at the wrong pick a near-copy-level match (>= 0.5), i.e. possibly the query's real location (label noise)?
    -> artifacts/spot_audit/discriminate.txt"""

    import torch
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import _discover_index, _unit_xyz
    from .strategy_search import MP16_EMBED
    from .yfcc_names import _mp16_authors

    s, sg = dict(np.load(OUT / "shortlist.npz")), dict(np.load(OUT / "siglip.npz"))
    photos = json.loads((PHOTO_SET / "photos.json").read_text(encoding="utf-8"))
    q = dict(np.load(QUERIES))
    dev = np.flatnonzero(q["set"] == "dev3k")
    q_emb, q_pid = q["emb"][dev].astype(np.float32), q["photo_id"][dev]
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    count = manifest["shapes"]["raw_image_embeddings"][0]
    latlon = np.memmap(index_dir / manifest["files"]["latlon_deg"], dtype=np.float32, mode="r", shape=(count, 2)).astype(np.float64)
    cache_rows = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    mp16 = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    image_ids = np.asarray((MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines())
    tree = cKDTree(_unit_xyz(latlon))
    chord = 2 * np.sin(radius / 6371.0 / 2)
    kind = s["p_kind"]
    p_right, p_wrong = np.flatnonzero(kind == "right"), np.flatnonzero(kind == "wrong")
    near = {}
    for p in np.concatenate([p_right, p_wrong]):
        d, r = tree.query(_unit_xyz(np.asarray([[s["p_lat"][p], s["p_lon"][p]]]))[0], k=cap, distance_upper_bound=chord)
        near[p] = r[np.isfinite(d)]
    rows_all = np.unique(np.concatenate(list(near.values())))
    ids_all = image_ids[cache_rows[rows_all]]
    authors = _mp16_authors(set(ids_all) | {p["image_id"] for p in photos})
    author_of = dict(zip(rows_all, (authors.get(i, "") for i in ids_all)))
    pid_of = dict(zip(rows_all, (int(i.rsplit("_", 1)[-1][:-4]) for i in ids_all)))
    new_emb = {int(pid): e for pid, e in zip(sg["ids"], sg["emb"]) if np.isfinite(e[0])}
    in_index = s["in_index"].astype(bool)

    def unit(x):
        x = np.asarray(x, np.float32)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def logistic(xr, xw, query):
        x = torch.as_tensor(np.concatenate([xr, xw]))
        y = torch.cat([torch.ones(len(xr)), torch.zeros(len(xw))])
        w8 = torch.cat([torch.full((len(xr),), 0.5 / len(xr)), torch.full((len(xw),), 0.5 / len(xw))])  # balanced: equal priors
        w, b = torch.zeros(x.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (w8 * torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y, reduction="none")).sum() + l2 * 1e-3 * (w @ w)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            return float(torch.as_tensor(query) @ w + b)

    rows = []
    for i, pr, pw in zip(s["p_i"][p_right], p_right, p_wrong):
        qa, qp, qx = authors.get(photos[i]["image_id"], "?"), q_pid[i], unit(q_emb[i])
        sides = []
        for p in (pr, pw):
            idx = [r for r in near[p] if author_of[r] != qa and pid_of[r] != qp]
            xi = unit(mp16[cache_rows[np.asarray(idx, np.int64)]]) if idx else np.zeros((0, 1536), np.float32)
            new = [new_emb[int(pid)] for pid in s["photo_id"][(s["point"] == p) & ~in_index] if int(pid) in new_emb]
            xn = unit(new) if new else np.zeros((0, 1536), np.float32)
            sides.append((xi, xn))
        (ri, rn), (wi, wn) = sides
        if min(len(ri), len(wi)) < 5:
            continue
        out = [len(ri), len(wi), len(rn), len(wn)]
        for xr, xw in ((ri, wi), (np.concatenate([ri, rn]), np.concatenate([wi, wn]))):
            out += [float((xr @ qx).max() - (xw @ qx).max()), float(unit(xr.mean(0)) @ qx - unit(xw.mean(0)) @ qx), logistic(xr, xw, qx)]
        rows.append(out)
    r = np.asarray(rows)
    lines = [f"Discriminative use of the photos, dev3k close calls with >= 5 index photos on both sides: {len(r)} of {len(p_right)}; "
             f"index photos per side median right {np.median(r[:, 0]):.0f} / wrong {np.median(r[:, 1]):.0f}; new per side median "
             f"{np.median(r[:, 2]):.0f} / {np.median(r[:, 3]):.0f}",
             "  % of close calls the query is put on the right side (> 0); 50 = coin flip; the reranker over the same pools: 19.1 -> 20.4"]
    for c, name in ((4, "nearest neighbour"), (5, "prototype (side mean)"), (6, "logistic, balanced")):
        lines.append(f"    {name:<22} index only {100 * (r[:, c] > 0).mean():5.1f}   + new photos {100 * (r[:, c + 3] > 0).mean():5.1f}"
                     f"   (moves to right {int(((r[:, c] <= 0) & (r[:, c + 3] > 0)).sum())}, to wrong {int(((r[:, c] > 0) & (r[:, c + 3] <= 0)).sum())})")
    many = (r[:, 2] >= 20) & (r[:, 3] >= 20)
    lines.append(f"    logistic where both sides got >= 20 new photos ({int(many.sum())}): index only {100 * (r[many, 6] > 0).mean():.1f}, "
                 f"+ new {100 * (r[many, 9] > 0).mean():.1f}")

    # label noise: near-copy-level MegaLoc matches at the wrong pick
    m = np.load(OUT / "megaloc.npz")
    e = m["emb"].astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    mq = np.load(MEGALOC_Q).astype(np.float32)
    mq /= np.linalg.norm(mq, axis=1, keepdims=True)
    sim = np.einsum("nd,nd->n", e[np.searchsorted(m["ids"], s["photo_id"])], mq[s["p_i"][s["point"]]])
    best = np.full(len(kind), -1.0)
    good = np.isfinite(sim)
    np.maximum.at(best, s["point"][good], sim[good])
    for t in (0.5, 0.6, 0.7):
        lines.append(f"  best MegaLoc >= {t}: at the wrong pick {int((best[p_wrong] >= t).sum())} photos, at the right point "
                     f"{int((best[p_right] >= t).sum())}, at the truth {int((best[kind == 'truth'] >= t).sum())} (of {len(p_right)})")
    text = "\n".join(lines)
    print(text)
    (OUT / "discriminate.txt").write_text(text + "\n", encoding="utf-8")


def _paired(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) over pairs (ties half); points without references score -1."""

    return float(np.mean((a > b) + 0.5 * (a == b)))


def report() -> None:
    s = dict(np.load(OUT / "shortlist.npz"))
    if (OUT / "megaloc.npz").exists():
        m = np.load(OUT / "megaloc.npz")
        ids, emb, partial = m["ids"], m["emb"], ""
    else:  # a fetch in progress: references not fetched yet count as missing (fewer per point, alike for every kind of point)
        m = np.load(OUT / "megaloc_part.npz")
        ids, emb = np.unique(s["photo_id"]), m["emb"]
        partial = f" PARTIAL: {int(m['done']):,} of {len(ids):,} references fetched;"
    q = np.load(MEGALOC_Q).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    e = emb.astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    pos = np.searchsorted(ids, s["photo_id"])
    ref_emb = e[pos]
    sim = np.einsum("nd,nd->n", ref_emb, q[s["p_i"][s["point"]]])  # NaN where the photo is gone
    rank = np.zeros(len(s["point"]), np.int64)  # order within its point (nearest first)
    for p in np.unique(s["point"]):
        r = np.flatnonzero(s["point"] == p)
        rank[r] = np.arange(len(r))
    ok = np.isfinite(sim)
    dup = ok & (sim >= DUP)
    n_pts = len(s["p_i"])
    lines = [f"Spot audit, dev3k close calls (tiers {CLOSE_TIERS}): {n_pts // 3} photos;{partial} references k <= {int(s['k'])} within "
             f"{float(s['radius']):g} km; fetched {100 * ok.mean():.1f}%; near-duplicates of the query (>= {DUP}) dropped: "
             f"{int(dup.sum())} ({len(np.unique(s['point'][dup]))} points)"]

    def score(k: int, source: str) -> np.ndarray:
        use = ok & ~dup & (rank < k)
        if source == "index":
            use &= s["in_index"].astype(bool)
        elif source == "new":
            use &= ~s["in_index"].astype(bool)
        best = np.full(n_pts, -1.0)
        np.maximum.at(best, s["point"][use], sim[use])
        return best

    kind = s["p_kind"]
    tiers = s["p_tier"][kind == "truth"]
    lines.append(f"  P(score at A > score at wrong pick), paired; clean = best >= {THRESH} and above the other point's best")
    lines.append(f"  {'k':>3} {'source':<6} {'right>wrong':>11} {'truth>wrong':>11} {'right clean win':>15} {'wrong clean win':>15} "
                 f"{'right >= thr':>12} {'wrong >= thr':>12}")
    for k in KS:
        for source in ("all", "index", "new"):
            b = score(k, source)
            t, r, w = b[kind == "truth"], b[kind == "right"], b[kind == "wrong"]
            lines.append(f"  {k:>3} {source:<6} {_paired(r, w):11.3f} {_paired(t, w):11.3f} {100 * ((r >= THRESH) & (r > w)).mean():14.1f}% "
                         f"{100 * ((w >= THRESH) & (w > r)).mean():14.1f}% {100 * (r >= THRESH).mean():11.1f}% {100 * (w >= THRESH).mean():11.1f}%")
    b = score(max(KS), "all")
    r, w = b[kind == "right"], b[kind == "wrong"]
    lines.append(f"  by tier (k {max(KS)}, all): tier n right>wrong right-clean-win wrong-clean-win")
    for tier in CLOSE_TIERS:
        mt = tiers == tier
        lines.append(f"    {tier} {int(mt.sum()):4d} {_paired(r[mt], w[mt]):.3f} {100 * ((r[mt] >= THRESH) & (r[mt] > w[mt])).mean():5.1f}% "
                     f"{100 * ((w[mt] >= THRESH) & (w[mt] > r[mt])).mean():5.1f}%")
    n_refs = np.bincount(s["point"][ok & ~dup], minlength=n_pts)
    lines.append(f"  references kept per point, median: truth {np.median(n_refs[kind == 'truth']):.0f}, right "
                 f"{np.median(n_refs[kind == 'right']):.0f}, wrong {np.median(n_refs[kind == 'wrong']):.0f}")
    for lo_, hi_ in ((0, 10), (10, 30), (30, 10**9)):
        mr = (n_refs[kind == "right"] >= lo_) & (n_refs[kind == "right"] < hi_)
        lines.append(f"    right point with {lo_}-{hi_ if hi_ < 10**9 else '':} refs: {int(mr.sum())} photos, right>wrong {_paired(r[mr], w[mr]):.3f}")
    # The wrong pick sits in denser places, which tilts best-of-k towards it: compare with the same number of kept references at both points
    # (the nearest k of each), and within a tighter radius.
    keep = ok & ~dup
    per_point = [[] for _ in range(n_pts)]
    for p, v, d in zip(s["point"][keep], sim[keep], s["dist"][keep]):
        per_point[p].append((d, v))
    per_point = [sorted(x) for x in per_point]
    lines.append("  equal reference counts (both points >= k kept; best of each point's nearest k), and within RADIUS km:")
    for radius in (float(s["radius"]), 0.25):
        for k in (5, 10, 25):
            rr, ww, tt = [], [], []
            for j in np.flatnonzero(kind == "truth"):  # points come in truth, right, wrong order per photo
                t_, r_, w_ = ([v for d, v in per_point[j + x] if d < radius][:k] for x in range(3))
                if len(r_) >= k and len(w_) >= k:
                    rr.append(max(r_)); ww.append(max(w_)); tt.append(max(t_) if len(t_) >= k else np.nan)
            if not rr:
                continue
            rr, ww, tt = map(np.asarray, (rr, ww, tt))
            mt = np.isfinite(tt)
            lines.append(f"    radius {radius:g} km, k {k:>2}: n {len(rr):3d}  right>wrong {_paired(rr, ww):.3f}  truth>wrong "
                         f"{_paired(tt[mt], ww[mt]) if mt.any() else float('nan'):.3f} (n {int(mt.sum())})  >= {THRESH}: right "
                         f"{100 * (rr >= THRESH).mean():.1f}% wrong {100 * (ww >= THRESH).mean():.1f}%  median best right {np.median(rr):.3f} "
                         f"wrong {np.median(ww):.3f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("shortlist", "fetch", "report", "siglip", "rerank", "discriminate"))
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)
    if args.cmd == "shortlist":
        shortlist(args.k, args.radius)
    elif args.cmd == "fetch":
        fetch(args.workers, args.batch)
    elif args.cmd == "siglip":
        siglip(args.workers, args.batch)
    elif args.cmd == "discriminate":
        discriminate()
    elif args.cmd == "rerank":
        rerank()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
