# How useful would a denser photo database be? Work backwards from each query's true GPS: which photos exist near it, and
# would any of them look like the query?
# Usage: .venv/bin/python -m geo_search_env.experiment.density_probe {queries,index,shortlist,fetch,report}

"""Density probe for dev3k, im2gps3k and yfcc4k (all photos; the user asked for the test benchmarks here).

Leak guard everywhere (data/yfcc100m/README.md): the query's own photo and every photo by its photographer are excluded.

queries:   query table (set, image id, Flickr photo id, photographer uid, truth, SigLIP2 embedding).
index:     one streaming pass over Pinpoint's retrieval index (4.08M MP16 photos): global SigLIP2 top-1 (similarity and its
           distance to the truth), and per radius (0.1 / 1 / 25 km of the truth) the photo count and best similarity.
           Also counts the YFCC100M photos within each radius that are not in the index ("new").
shortlist: per query, the nearest --k new YFCC photos within --radius km of the truth (Flickr accuracy >= 13).
fetch:     downloads the shortlist from Flickr and embeds it with SigLIP2 (photos already in the MP16 cache, i.e. bucket 99,
           reuse their cached embedding); stores only the embeddings.
report:    does a new nearby photo beat the current global top-1 (so nearest-neighbour retrieval would land near the truth)?

Results (2026-10-16, artifacts/density_probe/; 395k shortlisted, 21% gone from Flickr):
- The index already has a photo within 1 km of the truth for ~85% of queries (25 km: >99%); when the reranker misses at 1 km,
  ~17% are coverage gaps and the rest ranking failures (near-truth photo at SigLIP rank 12-999 or >= 1000).
- + up to 50 nearest new YFCC photos within 1 km: a new photo beats the SigLIP top-1 for 3.7 / 2.7 / 4.3 % of dev3k / im2gps3k /
  yfcc4k (near-duplicates >= 0.95 only 0.1-0.4 %); still growing with k (nearest 10 / 25 / 50: 1.3 / 2.5 / 3.7 on dev3k).
  Where the retrained reranker is wrong at 1 km, a new photo wins for 1.8 / 1.2 / 2.8 % (the truth-centred ceiling for 1 km fixes).
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.request

import numpy as np

from ..paths import PROJECT_ROOT
from ..data.benchmarks import BENCHMARKS, DATA_ROOT, EMBEDDING_KEY
from .strategy_search import MP16_EMBED

OUT = PROJECT_ROOT / "artifacts/density_probe"
GEO = PROJECT_ROOT / "data/yfcc100m/geo.npz"
INDEX = PROJECT_ROOT / "data/pinpoint/pinpoint-submission/exp/contrastive_retrieval/checkpoints/attention_retrieval_index"
RADII = (0.1, 1.0, 25.0)
SETS = ("dev3k", "im2gps3k", "yfcc4k")
MIN_ACCURACY = 13  # Flickr accuracy (16 = street); lower ones are placed at city or region level


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _mp16_photo_ids() -> np.ndarray:
    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    return np.asarray([int(i.rsplit("_", 1)[-1][:-4]) for i in ids], dtype=np.int64)


def _index_rows() -> np.ndarray:
    (root,) = [p for p in INDEX.iterdir() if (p / "manifest.json").exists()]
    return np.fromfile(root / "cache_indices.bin", dtype=np.int64)


def _geo(keys):
    g = np.load(GEO)
    return {k: g[k] for k in keys}


def _uid_of(photo_ids: np.ndarray, geo_pid_sorted: np.ndarray, order: np.ndarray, uid: np.ndarray) -> np.ndarray:
    j = np.clip(np.searchsorted(geo_pid_sorted, photo_ids), 0, len(geo_pid_sorted) - 1)
    found = geo_pid_sorted[j] == photo_ids
    return np.where(found, uid[order[j]], -1).astype(np.int32)


def queries() -> None:
    g = _geo(["photo_id", "uid"])
    order = np.argsort(g["photo_id"])
    pid_sorted = g["photo_id"][order]
    vocab = {u: i for i, u in enumerate(json.loads((GEO.parent / "geo_uid_vocab.json").read_text(encoding="utf-8")))}
    mp16 = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    rows = {"set": [], "image_id": [], "photo_id": [], "uid": [], "lat": [], "lon": []}
    embs = []
    photos = json.loads((PROJECT_ROOT / "artifacts/photo_sets/dev3k/photos.json").read_text(encoding="utf-8"))
    for p in photos:
        rows["set"].append("dev3k"); rows["image_id"].append(p["image_id"])
        rows["photo_id"].append(int(p["image_id"].rsplit("_", 1)[-1][:-4])); rows["uid"].append(None)
        rows["lat"].append(p["truth"][0]); rows["lon"].append(p["truth"][1])
        embs.append(np.asarray(mp16[p["row"]]))
    for name in ("im2gps3k", "yfcc4k"):
        spec = BENCHMARKS[name]
        root = DATA_ROOT / Path(spec.image_root_relpath).parent / "image_embeddings" / EMBEDDING_KEY
        ids = (root / "image_ids.txt").read_text(encoding="utf-8").splitlines()
        emb = np.fromfile(root / "embeddings.f16.bin", dtype=np.float16).reshape(len(ids), -1)
        with (DATA_ROOT / spec.csv_relpath).open(encoding="utf-8", newline="") as f:
            meta = {r["IMG_ID"].strip(): r for r in csv.DictReader(f)}
        for i, image_id in enumerate(ids):
            r = meta[image_id]
            author = r.get("AUTHOR") or r.get("OwnerNSID")
            rows["set"].append(name); rows["image_id"].append(image_id)
            rows["photo_id"].append(int(image_id.split("_")[0].split(".")[0])); rows["uid"].append(vocab.get(author, -1))
            rows["lat"].append(float(r["LAT"])); rows["lon"].append(float(r["LON"]))
            embs.append(emb[i])
    pids = np.asarray(rows["photo_id"], dtype=np.int64)
    looked_up = _uid_of(pids, pid_sorted, order, g["uid"])
    uid = np.asarray([looked_up[i] if u is None or u < 0 else u for i, u in enumerate(rows["uid"])], dtype=np.int32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "queries.npz", set=np.asarray(rows["set"]), image_id=np.asarray(rows["image_id"]), photo_id=pids, uid=uid,
             lat=np.asarray(rows["lat"], dtype=np.float64), lon=np.asarray(rows["lon"], dtype=np.float64), emb=np.stack(embs))
    for s in SETS:
        m = np.asarray(rows["set"]) == s
        print(f"{s}: {m.sum():,} queries, photographer found in YFCC {100 * (uid[m] >= 0).mean():.1f}%")


def _load_queries():
    q = dict(np.load(OUT / "queries.npz"))
    return q


def index(chunk: int) -> None:
    import torch

    q = _load_queries()
    rows = _index_rows()
    mp16_pid = _mp16_photo_ids()
    g = _geo(["photo_id", "uid", "lat", "lon", "video"])
    order = np.argsort(g["photo_id"])
    row_uid = _uid_of(mp16_pid[rows], g["photo_id"][order], order, g["uid"])
    print(f"index photos {len(rows):,}, photographer found {100 * (row_uid >= 0).mean():.1f}%", flush=True)
    mp16 = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2)

    dev = torch.device("cuda")
    Q = len(q["lat"])
    qe = torch.nn.functional.normalize(torch.as_tensor(q["emb"].astype(np.float32), device=dev), dim=1).half()
    qlat = torch.as_tensor(np.radians(q["lat"]), device=dev, dtype=torch.float32)[:, None]
    qlon = torch.as_tensor(np.radians(q["lon"]), device=dev, dtype=torch.float32)[:, None]
    quid = torch.as_tensor(q["uid"], device=dev)[:, None]
    qpid = torch.as_tensor(q["photo_id"], device=dev)[:, None]
    top_sim = torch.full((Q,), -2.0, device=dev); top_dist = torch.zeros(Q, device=dev)
    rad_sim = {r: torch.full((Q,), -2.0, device=dev) for r in RADII}
    rad_n = {r: torch.zeros(Q, dtype=torch.int64, device=dev) for r in RADII}
    t0 = time.time()
    for start in range(0, len(rows), chunk):
        r = rows[start : start + chunk]
        e = torch.nn.functional.normalize(torch.as_tensor(np.asarray(mp16[r[0] : r[-1] + 1][r - r[0]], dtype=np.float32), device=dev), dim=1).half()
        ll = torch.as_tensor(np.radians(latlon[r]), device=dev)
        sim = (qe @ e.T).float()
        bad = (quid == torch.as_tensor(row_uid[start : start + len(r)], device=dev)[None]) & (quid >= 0)
        bad |= qpid == torch.as_tensor(mp16_pid[r], device=dev)[None]
        sim.masked_fill_(bad, -2.0)
        a = torch.sin((ll[None, :, 0] - qlat) / 2) ** 2 + torch.cos(qlat) * torch.cos(ll[None, :, 0]) * torch.sin((ll[None, :, 1] - qlon) / 2) ** 2
        dist = 2 * 6371.0 * torch.asin(torch.sqrt(a.clamp(0, 1)))
        s, j = sim.max(dim=1)
        better = s > top_sim
        top_sim = torch.where(better, s, top_sim)
        top_dist = torch.where(better, dist.gather(1, j[:, None])[:, 0], top_dist)
        for rad in RADII:
            inside = (dist < rad) & ~bad
            rad_n[rad] += inside.sum(dim=1)
            rad_sim[rad] = torch.maximum(rad_sim[rad], sim.masked_fill(~inside, -2.0).max(dim=1).values)
        del sim, dist, a, bad
        if start // chunk % 20 == 0:
            print(f"  {start + len(r):,}/{len(rows):,} rows, {time.time() - t0:.0f}s", flush=True)

    # New photos: YFCC100M (non-video) not in the index, near each truth, photographer excluded (CPU, 0.25-degree grid).
    in_index = np.isin(g["photo_id"], mp16_pid[rows])
    keep = (g["video"] == 0) & ~in_index
    lat, lon, uid, pid = g["lat"][keep], g["lon"][keep], g["uid"][keep], g["photo_id"][keep]
    cell_size = 0.25
    cell = (np.floor(lat / cell_size).astype(np.int64) + 1000) * 10000 + (np.floor(lon / cell_size).astype(np.int64) + 1000)
    o = np.argsort(cell, kind="stable"); cell_s = cell[o]
    new_n = {r: np.zeros(Q, dtype=np.int64) for r in RADII}
    for i in range(Q):
        idx = _near(q["lat"][i], q["lon"][i], o, cell_s, cell_size)
        idx = idx[(pid[idx] != q["photo_id"][i]) & ((uid[idx] != q["uid"][i]) | (q["uid"][i] < 0))]
        d = _haversine(q["lat"][i], q["lon"][i], lat[idx], lon[idx])
        for rad in RADII:
            new_n[rad][i] = int((d < rad).sum())
    out = {"top_sim": top_sim.cpu().numpy(), "top_dist": top_dist.cpu().numpy()}
    for rad in RADII:
        out[f"sim_{rad}"] = rad_sim[rad].cpu().numpy(); out[f"n_{rad}"] = rad_n[rad].cpu().numpy(); out[f"new_{rad}"] = new_n[rad]
    np.savez(OUT / "index.npz", **out)
    print(f"wrote {OUT / 'index.npz'} ({time.time() - t0:.0f}s)")


def ranks(chunk: int) -> None:
    """Second pass: the global rank (0 = top-1) of the best index photo within each radius, photographer excluded."""
    import torch

    q = _load_queries()
    ix = dict(np.load(OUT / "index.npz"))
    rows = _index_rows()
    mp16_pid = _mp16_photo_ids()
    g = _geo(["photo_id", "uid"])
    order = np.argsort(g["photo_id"])
    row_uid = _uid_of(mp16_pid[rows], g["photo_id"][order], order, g["uid"])
    mp16 = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    dev = torch.device("cuda")
    qe = torch.nn.functional.normalize(torch.as_tensor(q["emb"].astype(np.float32), device=dev), dim=1).half()
    quid = torch.as_tensor(q["uid"], device=dev)[:, None]
    qpid = torch.as_tensor(q["photo_id"], device=dev)[:, None]
    thr = {r: torch.as_tensor(ix[f"sim_{r}"], device=dev, dtype=torch.float32)[:, None] for r in RADII}
    above = {r: torch.zeros(len(q["lat"]), dtype=torch.int64, device=dev) for r in RADII}
    for start in range(0, len(rows), chunk):
        r = rows[start : start + chunk]
        e = torch.nn.functional.normalize(torch.as_tensor(np.asarray(mp16[r[0] : r[-1] + 1][r - r[0]], dtype=np.float32), device=dev), dim=1).half()
        sim = (qe @ e.T).float()
        bad = (quid == torch.as_tensor(row_uid[start : start + len(r)], device=dev)[None]) & (quid >= 0)
        bad |= qpid == torch.as_tensor(mp16_pid[r], device=dev)[None]
        sim.masked_fill_(bad, -2.0)
        for rad in RADII:
            above[rad] += (sim > thr[rad]).sum(dim=1)
        del sim, bad
    out = {f"rank_{r}": np.where(ix[f"n_{r}"] > 0, above[r].cpu().numpy(), -1) for r in RADII}
    np.savez(OUT / "ranks.npz", **out)
    print(f"wrote {OUT / 'ranks.npz'}")


def _near(la, lo, o, cell_s, size):
    ci, cj = int(np.floor(la / size)), int(np.floor(lo / size))
    parts = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            k = (ci + di + 1000) * 10000 + (cj + dj + 1000)
            parts.append(o[np.searchsorted(cell_s, k) : np.searchsorted(cell_s, k, "right")])
    return np.concatenate(parts)


def shortlist(k: int, radius: float, limit: int) -> None:
    q = _load_queries()
    rows = _index_rows()
    mp16_pid = _mp16_photo_ids()
    g = _geo(["photo_id", "uid", "lat", "lon", "video", "accuracy", "server", "secret"])
    keep = (g["video"] == 0) & ~np.isin(g["photo_id"], mp16_pid[rows]) & (g["accuracy"] >= MIN_ACCURACY)
    sel = np.flatnonzero(keep)
    lat, lon = g["lat"][sel], g["lon"][sel]
    cell_size = 0.05
    cell = (np.floor(lat / cell_size).astype(np.int64) + 10000) * 100000 + (np.floor(lon / cell_size).astype(np.int64) + 10000)
    o = np.argsort(cell, kind="stable"); cell_s = cell[o]

    def near(la, lo):
        ci, cj = int(np.floor(la / cell_size)), int(np.floor(lo / cell_size))
        span = int(np.ceil(radius / (111 * cell_size * max(np.cos(np.radians(la)), 0.05)))) + 1
        parts = []
        for di in range(-1, 2):
            for dj in range(-span, span + 1):
                key = (ci + di + 10000) * 100000 + (cj + dj + 10000)
                parts.append(o[np.searchsorted(cell_s, key) : np.searchsorted(cell_s, key, "right")])
        return np.concatenate(parts)

    picks = []  # (query index, geo row, distance km)
    per_set = {s: np.flatnonzero(q["set"] == s)[:limit] if limit else np.flatnonzero(q["set"] == s) for s in SETS}
    for s, qs in per_set.items():
        for i in qs:
            idx = sel[near(q["lat"][i], q["lon"][i])]
            idx = idx[(g["photo_id"][idx] != q["photo_id"][i]) & ((g["uid"][idx] != q["uid"][i]) | (q["uid"][i] < 0))]
            d = _haversine(q["lat"][i], q["lon"][i], g["lat"][idx], g["lon"][idx])
            m = d < radius
            idx, d = idx[m], d[m]
            top = np.argsort(d, kind="stable")[:k]
            picks += [(i, j, dd) for j, dd in zip(idx[top], d[top])]
    picks = np.asarray(picks, dtype=np.float64)
    geo_row = picks[:, 1].astype(np.int64)
    mp16_sorted = np.argsort(mp16_pid)
    j = np.clip(np.searchsorted(mp16_pid[mp16_sorted], g["photo_id"][geo_row]), 0, len(mp16_pid) - 1)
    mp16_row = np.where(mp16_pid[mp16_sorted][j] == g["photo_id"][geo_row], mp16_sorted[j], -1)
    np.savez(OUT / "shortlist.npz", query=picks[:, 0].astype(np.int64), geo_row=geo_row, dist=picks[:, 2],
             photo_id=g["photo_id"][geo_row], server=g["server"][geo_row], secret=g["secret"][geo_row], mp16_row=mp16_row)
    print(f"shortlist: {len(picks):,} photos for {sum(len(v) for v in per_set.values()):,} queries "
          f"({(mp16_row >= 0).sum():,} already embedded in the MP16 cache)")


def _download(args):
    pid, server, secret = args
    url = f"https://live.staticflickr.com/{server}/{pid}_{secret.decode() if isinstance(secret, bytes) else secret}.jpg"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as r:
                if "photo_unavailable" in r.geturl():
                    return None
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def fetch(workers: int, batch: int) -> None:
    import torch
    from .embed_cache import _embed, _load_model, _pixels

    s = dict(np.load(OUT / "shortlist.npz"))
    n = len(s["query"])
    emb = np.zeros((n, 1536), dtype=np.float16)
    status = np.zeros(n, dtype=np.int8)  # 1 embedded, 2 from the MP16 cache, -1 gone or undecodable
    part = OUT / "fetch_part.npz"
    done = 0
    if part.exists():
        p = np.load(part); emb, status, done = p["emb"], p["status"], int(p["done"])
        print(f"resuming at {done:,}/{n:,}")
    mp16 = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    cached = np.flatnonzero(s["mp16_row"] >= 0)
    emb[cached] = mp16[s["mp16_row"][cached]]; status[cached] = 2
    model, processor = _load_model()
    todo = np.flatnonzero(status == 0)
    todo = todo[todo >= done] if done else todo
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for b0 in range(0, len(todo), batch):
            ids = todo[b0 : b0 + batch]
            blobs = list(pool.map(_download, [(s["photo_id"][i], s["server"][i], s["secret"][i]) for i in ids]))
            pix, ok = [], []
            for i, data in zip(ids, blobs):
                p = _pixels(processor, data)
                if p is None:
                    status[i] = -1
                else:
                    pix.append(p); ok.append(i)
            if pix:
                emb[ok] = _embed(model, torch.stack(pix)); status[ok] = 1
            if b0 // batch % 20 == 0:
                rate = (b0 + len(ids)) / (time.time() - t0)
                print(f"  {b0 + len(ids):,}/{len(todo):,} downloads, {rate:.0f}/s, gone {100 * (status[todo[: b0 + len(ids)]] == -1).mean():.1f}%", flush=True)
                np.savez(part, emb=emb, status=status, done=ids[-1] + 1)
    np.savez(OUT / "fetched.npz", emb=emb, status=status)
    part.unlink(missing_ok=True)
    print(f"fetched: embedded {(status == 1).sum():,}, cached {(status == 2).sum():,}, gone {(status == -1).sum():,} ({time.time() - t0:.0f}s)")


def report(fetch_dir: Path = OUT) -> None:
    q = _load_queries()
    ix = dict(np.load(OUT / "index.npz"))
    pct = lambda m: f"{100 * np.mean(m):5.1f}"
    print("== Current index (Pinpoint's 4.08M MP16 photos), photographer excluded; SigLIP2 cosine ==")
    print(f"{'set':9s} {'N':>5s} | top-1 <1km <25km | any<0.1 any<1 any<25 | med n<0.1 n<1 n<25 | new med n<0.1 n<1 | "
          f"best<1km sim (med) | top-1 sim (med)")
    for s in SETS:
        m = q["set"] == s
        print(f"{s:9s} {m.sum():5d} | {pct(ix['top_dist'][m] < 1)} {pct(ix['top_dist'][m] < 25)} | "
              f"{pct(ix['n_0.1'][m] > 0)} {pct(ix['n_1.0'][m] > 0)} {pct(ix['n_25.0'][m] > 0)} | "
              f"{np.median(ix['n_0.1'][m]):5.0f} {np.median(ix['n_1.0'][m]):5.0f} {np.median(ix['n_25.0'][m]):6.0f} | "
              f"{np.median(ix['new_0.1'][m]):5.0f} {np.median(ix['new_1.0'][m]):6.0f} | "
              f"{np.median(ix['sim_1.0'][m][ix['n_1.0'][m] > 0]):.3f} | {np.median(ix['top_sim'][m]):.3f}")
    print("\n== Would a nearby index photo already win? (best similarity within r > global top-1 means top-1 is within r) ==")
    for s in SETS:
        m = q["set"] == s
        gap = ix["top_sim"][m] - ix["sim_1.0"][m]
        print(f"{s:9s} top-1 within 1 km {pct(ix['top_dist'][m] < 1)}%; nearby photo exists but loses: {pct((ix['n_1.0'][m] > 0) & (ix['top_dist'][m] >= 1))}%"
              f" (median sim gap {np.median(gap[(ix['n_1.0'][m] > 0) & (ix['top_dist'][m] >= 1)]):.3f})")
    if not (fetch_dir / "fetched.npz").exists():
        return
    s = dict(np.load(fetch_dir / "shortlist.npz")); f = dict(np.load(fetch_dir / "fetched.npz"))
    ok = f["status"] > 0
    qe = q["emb"].astype(np.float32); qe /= np.linalg.norm(qe, axis=1, keepdims=True)
    e = f["emb"].astype(np.float32); e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    sim = np.einsum("ij,ij->i", qe[s["query"]], e)
    sim[~ok] = -2
    Q = len(q["lat"])
    best = {r: np.full(Q, -2.0) for r in (0.1, 1.0)}
    for r in best:
        m = s["dist"] < r
        np.maximum.at(best[r], s["query"][m], sim[m])
    probed = np.zeros(Q, bool); probed[np.unique(s["query"])] = True
    has_list = np.zeros(Q, bool); has_list[s["query"][ok]] = True
    print(f"\n== Denser database: + up to {np.bincount(s['query']).max()} nearest new YFCC photos within 1 km (accuracy >= {MIN_ACCURACY}) ==")
    print(f"downloaded {ok.sum():,}/{len(ok):,} ({pct(f['status'] == -1)}% gone)")
    print(f"{'set':9s} {'probed':>6s} | top-1<1km now | new photo exists | new beats top-1: <0.1km <1km | either (now or new) <1km | "
          f"new beats, top-1 >25km off")
    for st in SETS:
        m = (q["set"] == st) & probed
        if not m.any():
            continue
        now = ix["top_dist"][m] < 1
        win1 = best[1.0][m] > ix["top_sim"][m]
        win01 = best[0.1][m] > ix["top_sim"][m]
        far = ix["top_dist"][m] > 25
        print(f"{st:9s} {m.sum():6d} | {pct(now)} | {pct(has_list[m])} | {pct(win01)} {pct(win1)} | {pct(now | win1)} (+{100 * np.mean(win1 & ~now):.1f}) | "
              f"{pct(win1 & far)}")
    print("\nall queries (denominator = whole set): new beats top-1 within 1 km, split by near-duplicates (sim >= 0.95), "
          "and by how many nearest new photos are used")
    rank = np.zeros(len(sim), dtype=np.int64)  # position of each shortlisted photo in its query's distance order
    starts = np.r_[0, np.flatnonzero(np.diff(s["query"])) + 1]
    for a, b in zip(starts, np.r_[starts[1:], len(sim)]):
        rank[a:b] = np.arange(b - a)
    win1 = best[1.0] > ix["top_sim"]
    dup = np.zeros(Q, bool)
    dup[s["query"][(sim >= 0.95) & (s["dist"] < 1)]] = True
    for st in SETS:
        m = q["set"] == st
        by_k = []
        for k in (10, 25, 50):
            bk = np.full(Q, -2.0); sel = rank < k
            np.maximum.at(bk, s["query"][sel], sim[sel])
            by_k.append(pct((bk > ix["top_sim"])[m]))
        print(f"  {st:9s} {pct(win1[m])}% (near-duplicate {pct((win1 & dup)[m])}%, other {pct((win1 & ~dup)[m])}%); "
              f"nearest 10 / 25 / 50: {' / '.join(by_k)}")
    top1 = {}
    for tag in ("dev3k_setft_a", "full_setft_a"):
        for p in json.loads((PROJECT_ROOT / f"artifacts/photo_sets/{tag}/photos.json").read_text(encoding="utf-8")):
            top1[p["image_id"]] = tuple(p["candidates"][0][1:3])
    have = np.asarray([i in top1 for i in q["image_id"]])
    pin = np.full(Q, np.inf)
    pin[have] = [_haversine(q["lat"][i], q["lon"][i], *top1[q["image_id"][i]]) for i in np.flatnonzero(have)]
    print("\nvs the retrained Pinpoint reranker top-1 (dev3k; benchmarks: eval halves only):")
    for st in SETS:
        m = (q["set"] == st) & have
        print(f"  {st:9s} n={m.sum():5d} Pinpoint <1km {pct((pin < 1)[m])}; new photo wins where Pinpoint is wrong at 1 km "
              f"{pct((win1 & (pin >= 1))[m])}% (near-duplicate {pct((win1 & dup & (pin >= 1))[m])}%; Pinpoint >25 km off {pct((win1 & (pin > 25))[m])}%)")
    print("\nsimilarity of the best new photo within 1 km, among queries with one (median / 90th pct), vs current top-1 median:")
    for st in SETS:
        m = (q["set"] == st) & probed & has_list
        if m.any():
            print(f"  {st:9s} {np.median(best[1.0][m]):.3f} / {np.quantile(best[1.0][m], 0.9):.3f} vs {np.median(ix['top_sim'][m]):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("queries")
    p = sub.add_parser("index"); p.add_argument("--chunk", type=int, default=40_000)
    p = sub.add_parser("ranks"); p.add_argument("--chunk", type=int, default=40_000)
    p = sub.add_parser("shortlist"); p.add_argument("--k", type=int, default=50); p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="first N queries per set (pilot)")
    p = sub.add_parser("fetch"); p.add_argument("--workers", type=int, default=32); p.add_argument("--batch", type=int, default=128)
    p = sub.add_parser("report"); p.add_argument("--fetch-dir", type=Path, default=OUT, help="shortlist + fetched (e.g. the pilot)")
    a = ap.parse_args(argv)
    if a.cmd == "queries":
        queries()
    elif a.cmd == "index":
        index(a.chunk)
    elif a.cmd == "ranks":
        ranks(a.chunk)
    elif a.cmd == "shortlist":
        shortlist(a.k, a.radius, a.limit)
    elif a.cmd == "fetch":
        fetch(a.workers, a.batch)
    else:
        report(a.fetch_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
