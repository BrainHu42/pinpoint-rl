# Visual verification of text-search places: fetch photos of each candidate place and ask whether one looks like the query (MegaLoc).
# Usage: .venv/bin/python -m geo_search_env.experiment.clue_verify {shortlist,fetch,report} [--tag commons26 --split dev]

"""Does a fetched photo of a candidate place match the query, and does that fix the pipeline where text evidence is thin (lesson 120)?

Places per photo: the text-search clusters of its 4B phrases (artifacts/agent/clusters_<tag>_4b.json; merged within 25 km, top --max-text
by weight) plus the pipeline's answer area and Pinpoint's first area (action_agent apply npz, zoom npz) as the comparison.
Photos per place (YFCC100M, data/yfcc100m/geo.npz for the URLs):
  targeted: up to --k-text Flickr photos whose title / tags match one of the photo's phrases within 25 km of the place
            (text_search.matches; the pipeline's places use all of the photo's phrases), closest to the place first;
  nearby:   the --k-near geotagged photos closest to the place point (Flickr accuracy >= 13), whatever they show.
Score: best MegaLoc cosine between the query and a place's photos. commons26 photos were taken in 2026 and YFCC ends in 2014, so no
leak guard beyond that; dev3k (MP16 photos) would need the photo's own id and photographer excluded (`exclude`).

shortlist: places + photo ids -> artifacts/clue_verify/<tag>_<split>_shortlist.json
fetch:     downloads each unique photo once, MegaLoc-embeds it (and the queries) -> <tag>_<split>_megaloc.npz
report:    separation of right / wrong places, the pipeline's answer vs the best text place, and a switch rule chosen by grouped CV.

Result (commons26 dev, 2026-10-16; 6,693 places, 51k photos, 42k fetched): negative. Right vs wrong text places, best MegaLoc cosine
AUC 0.60 (targeted 0.60, nearby 0.51) vs text-search weight 0.68; within a photo the visual top is right on 27% vs weight order 59%
(weight + MegaLoc at most 63%, in-sample). Scores are low throughout (right places median 0.07, p90 0.16; >= 0.3 on 2.2%; random pairs
0.03, fetched photos of the same place and phrase match each other at median 0.20): 2004-14 Flickr photos rarely show the same view as
the 2026 query. Switch rule (grouped CV): 24.8 -> 24.2-24.7% (thresholds from 0.05), ceiling 31.0%.
By tier: on landmarks MegaLoc >= 0.2 marks 11.1% of right places vs 1.0% of wrong (precise, low recall); city / region / none ~nothing.
Keypoints (`keypoints`, DISK + LightGlue + MAGSAC, landmark tier, 7 targeted + 3 nearby per place): AUC 0.535 (MegaLoc 0.640, weight
0.698); >= 40 inliers right 4.6% vs wrong 0.7%; switch 42.0 -> 41.3% (0 fixes). The limit is the photos, not the matcher: on the 32
fixable landmark photos (pipeline wrong, a right text place exists) the right place's photos match on 1 (keypoints) / 4 (MegaLoc), and
37.5% have no targeted photo; contact sheets show the same city, other subjects.
OSV-5M street level (`osv`; only SigLIP2 embeddings on this machine, no images): an OSV image within 1 / 5 km of 72 / 88% of text
places; AUC right vs wrong text places 0.68 (= text weight), areas 0.61; within-photo top pick text 46% (weight 59%), areas 38% (rank
56%); switches -0.6 to +0.2. Grouped-CV logistic over all candidates: AUC 0.760 -> 0.785 with OSV, but top candidate 25.5 -> 24.1% and
switch 24.8 -> 24.2%: SigLIP similarity to street images near a place is regional look, which Pinpoint's SigLIP retrieval already has.
MegaLoc on the OSV images themselves (`osv-megaloc`: 5 nearest within 1 km per place, 25.6k images read from the HF zips by range
request, `osv_remote.py`): AUC text 0.57, areas 0.55; within-photo top pick text 22%, areas 26%; switches -0.3 to 0.0. OSV-5M is too
sparse for same-place evidence: an OSV image within 50 / 100 / 250 m of the truth for 5.7 / 13.1 / 31.0% of commons26 dev photos;
best-of-5 MegaLoc p99 0.12. Same-place street-level verification needs dense coverage at the candidate point (full Mapillary).
All OSV images within 1 km (`--k-osv 40`, 86k images; places have median 9 / mean 15 within 1 km): AUC text 0.577, areas 0.590;
within-photo top pick 29 / 30%; switches -0.2 to +0.2 (logistic with all features 24.8 -> 25.0%, 2 fixes). Scores stay near noise
(p90 at places with 21+ images: right 0.12, wrong 0.09): images hundreds of metres away show other streets.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np

from .density_probe import GEO, MIN_ACCURACY, _download, _haversine

OUT = Path("artifacts/clue_verify")
AGENT = Path("artifacts/agent")
ANSWER = "RL policy, lambda 0"
CKPT_TAG = "setft_own"
RIGHT_KM = 25.0


def _photos(tag: str, split: str):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}_{CKPT_TAG}/photos.json").read_text(encoding="utf-8"))
    keep = [i for i, p in enumerate(photos) if p.get("split", split) == split]
    return photos, keep


def _answers(tag: str):
    d = np.load(AGENT / f"action_agent_apply_km_{CKPT_TAG}_1km0.5_{tag}.npz")
    key = f"{tag}:{CKPT_TAG}|{ANSWER}"
    return d[key], d[key + "|center"], d[f"{tag}:{CKPT_TAG}|retrained"]


def _merge(clusters):
    """Clusters [(lat, lon, weight, phrase)] -> places within 25 km merged: [(lat, lon, weight sum, phrases)], heaviest first."""
    places = []
    for lat, lon, w, q in sorted(clusters, key=lambda c: -c[2]):
        for p in places:
            if _haversine(lat, lon, p[0], p[1]) < RIGHT_KM:
                p[2] += w
                if q not in p[3]:
                    p[3].append(q)
                break
        else:
            places.append([lat, lon, w, [q]])
    return places


def shortlist(tag: str, split: str, max_text: int, k_text: int, k_near: int) -> None:
    from .text_search import TextSearch

    photos, keep = _photos(tag, split)
    _, center, _ = _answers(tag)
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    first_area = {int(p): c for p, s, r, c in zip(zm["photo"], zm["source"], zm["rank"], zm["center"]) if s == 0 and r == 0}
    by_photo: dict[str, list] = {}
    for rec in json.loads((AGENT / f"clusters_{tag}_4b.json").read_text(encoding="utf-8")):
        for c in rec["clusters"]:
            by_photo.setdefault(rec["image_id"], []).append((c["lat"], c["lon"], c["weight"], rec["query"]))
    queries = json.loads((AGENT / f"queries_{tag}_4b.json").read_text(encoding="utf-8"))

    g = np.load(GEO)
    pid, lat, lon, acc, vid = g["photo_id"], g["lat"], g["lon"], g["accuracy"], g["video"]
    sel = np.flatnonzero((vid == 0) & (acc >= MIN_ACCURACY))
    size = 0.05
    cell = (np.floor(lat[sel] / size).astype(np.int64) + 10000) * 100000 + (np.floor(lon[sel] / size).astype(np.int64) + 10000)
    o = np.argsort(cell, kind="stable"); cell_s = cell[o]
    pid_order = np.argsort(pid); pid_sorted = pid[pid_order]

    def nearest(la, lo, k):
        ci, cj = int(math.floor(la / size)), int(math.floor(lo / size))
        idx = np.concatenate([o[np.searchsorted(cell_s, (ci + di + 10000) * 100000 + cj + dj + 10000):
                                np.searchsorted(cell_s, (ci + di + 10000) * 100000 + cj + dj + 10000, "right")]
                              for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        idx = sel[idx]
        d = _haversine(la, lo, lat[idx], lon[idx])
        return [int(pid[i]) for i in idx[np.argsort(d)[:k]]]

    ts = TextSearch()
    out, t0 = [], time.time()
    for n, i in enumerate(keep):
        image_id = photos[i]["image_id"]
        phrases = queries.get(image_id, [])
        places = [dict(lat=p[0], lon=p[1], weight=p[2], phrases=p[3], kind="text") for p in _merge(by_photo.get(image_id, []))[:max_text]]
        for kind, c in (("answer", center[i]), ("area1", first_area.get(i))):
            if c is None or not np.all(np.isfinite(c)):
                continue
            places.append(dict(lat=float(c[0]), lon=float(c[1]), weight=0.0, phrases=list(phrases), kind=kind))
        for p in places:
            found: dict[int, float] = {}
            for q in p["phrases"]:
                try:
                    _, _, fl, _, _, _ = ts.matches(q, near=(p["lat"], p["lon"], RIGHT_KM))
                except Exception:
                    fl = []
                for f in fl:
                    d = float(_haversine(p["lat"], p["lon"], f[2], f[3]))
                    if d < RIGHT_KM:
                        found[int(f[0])] = min(d, found.get(int(f[0]), 1e9))
            p["text_ids"] = [k for k, _ in sorted(found.items(), key=lambda kv: kv[1])][:k_text]
            p["near_ids"] = nearest(p["lat"], p["lon"], k_near)
            p["right"] = bool(_haversine(p["lat"], p["lon"], *photos[i]["truth"]) < RIGHT_KM)
        out.append(dict(photo=i, image_id=image_id, places=places))
        if n % 100 == 0:
            print(f"  {n}/{len(keep)} photos, {time.time() - t0:.0f}s", flush=True)
    ids = sorted({f for r in out for p in r["places"] for f in p["text_ids"] + p["near_ids"]})
    j = np.searchsorted(pid_sorted, ids)
    rows = pid_order[j]
    server, secret = g["server"][rows], g["secret"][rows]  # npz arrays load on every access: index once
    urls = {int(a): [int(sv), sc.decode()] for a, sv, sc in zip(ids, server, secret)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}_{split}_shortlist.json").write_text(json.dumps(dict(photos=out, urls=urls)), encoding="utf-8")
    n_pl = sum(len(r["places"]) for r in out)
    print(f"{len(out)} photos, {n_pl} places ({sum(p['kind'] == 'text' for r in out for p in r['places'])} text), "
          f"{len(ids):,} unique YFCC photos to fetch; places with a right one: "
          f"{np.mean([any(p['right'] for p in r['places']) for r in out]):.1%}")


def _megaloc_tf():
    import torch
    from PIL import Image

    from .megaloc_gate import SIZE

    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    def tf(data: bytes | None):
        if data is None:
            return None
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        except Exception:
            return None
        return torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy())

    return tf


def fetch(tag: str, split: str, workers: int, batch: int) -> None:
    import torch

    from .megaloc_gate import _model

    s = json.loads((OUT / f"{tag}_{split}_shortlist.json").read_text(encoding="utf-8"))
    model, tf = _model(), _megaloc_tf()

    def run(blobs):
        xs = [tf(b) for b in blobs]
        ok = [k for k, x in enumerate(xs) if x is not None]
        emb = np.full((len(blobs), 0), np.nan, np.float16)
        if ok:
            with torch.inference_mode():
                e = model(torch.stack([xs[k] for k in ok]).cuda()).float().cpu().numpy()
            emb = np.full((len(blobs), e.shape[1]), np.nan, np.float16)
            emb[ok] = e
        return emb

    photos, _ = _photos(tag, split)
    root = Path(f"data/pinpoint/{tag}/release/images")
    q_ids = [r["photo"] for r in s["photos"]]
    q_emb = np.concatenate([run([(root / photos[i]["image_id"]).read_bytes() for i in q_ids[a:a + batch]]) for a in range(0, len(q_ids), batch)])
    ids = sorted(int(k) for k in s["urls"])
    emb, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as pool:
        for a in range(0, len(ids), batch):
            chunk = ids[a:a + batch]
            blobs = list(pool.map(_download, [(k, *s["urls"][str(k)]) for k in chunk]))
            emb.append(run(blobs))
            if a // batch % 20 == 0:
                done = a + len(chunk)
                gone = np.mean([np.isnan(e[:, 0]).mean() if e.shape[1] else 1.0 for e in emb])
                print(f"  {done:,}/{len(ids):,} photos, {done / (time.time() - t0):.0f}/s, gone ~{100 * gone:.0f}%", flush=True)
    dim = q_emb.shape[1]
    emb = np.concatenate([e if e.shape[1] else np.full((len(e), dim), np.nan, np.float16) for e in emb])
    np.savez(OUT / f"{tag}_{split}_megaloc.npz", q_photo=np.asarray(q_ids), q_emb=q_emb, ids=np.asarray(ids, np.int64), emb=emb)
    print(f"embedded {np.isfinite(emb[:, 0]).sum():,}/{len(ids):,} ({time.time() - t0:.0f}s)")


def keypoints(tag: str, split: str, tier: str, k_text: int, k_near: int, workers: int) -> None:
    """Geometric verification instead of MegaLoc: DISK keypoints + LightGlue + MAGSAC fundamental-matrix inliers between the query and each
    place's photos (first k_text targeted, k_near nearby; re-downloaded). Needs kornia on PYTHONPATH. -> <tag>_<split>_kp_<tier>.json"""
    import cv2
    import kornia.feature as KF
    import torch
    from PIL import Image

    s = json.loads((OUT / f"{tag}_{split}_shortlist.json").read_text(encoding="utf-8"))
    tiers = {}
    with Path(f"data/pinpoint/{tag}/release/benchmark.csv").open(encoding="utf-8", newline="") as f:
        tiers = {r["IMG_ID"]: r.get("tier", "") for r in csv.DictReader(f)}
    rows = [r for r in s["photos"] if tier == "all" or tiers.get(r["image_id"]) == tier]
    dev = torch.device("cuda")
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    lg = KF.LightGlueMatcher("disk").to(dev).eval()
    side = 640

    def load(data):
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None
        im.thumbnail((side, side))
        return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)

    @torch.inference_mode()
    def feats(x):
        f = disk(x, 2048, pad_if_not_divisible=True)[0]
        laf = KF.laf_from_center_scale_ori(f.keypoints[None], torch.ones(1, len(f.keypoints), 1, 1, device=dev))
        return f.keypoints, f.descriptors, laf, torch.tensor(x.shape[-2:], device=dev)

    @torch.inference_mode()
    def inliers(a, b):
        if len(a[0]) < 8 or len(b[0]) < 8:
            return 0
        _, idx = lg(a[1], b[1], a[2], b[2], hw1=a[3], hw2=b[3])
        if len(idx) < 8:
            return 0
        p1, p2 = a[0][idx[:, 0]].cpu().numpy(), b[0][idx[:, 1]].cpu().numpy()
        try:
            _, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
        except cv2.error:  # degenerate point sets
            return 0
        return int(mask.sum()) if mask is not None else 0

    pool, t0 = ThreadPoolExecutor(workers), time.time()
    for n, r in enumerate(rows):
        q = load(Path(f"data/pinpoint/{tag}/release/images/{r['image_id']}").read_bytes())
        qf = feats(q)
        for p in r["places"]:
            p["kp_ids"] = p["text_ids"][:k_text] + [i for i in p["near_ids"][:k_near] if i not in p["text_ids"][:k_text]]
        need = sorted({i for p in r["places"] for i in p["kp_ids"]})
        blob = dict(zip(need, pool.map(_download, [(i, *s["urls"][str(i)]) for i in need])))
        score = {}
        for i, b in blob.items():
            x = load(b) if b else None
            score[i] = inliers(qf, feats(x)) if x is not None else -1
        for p in r["places"]:
            p["kp"] = [score[i] for i in p["kp_ids"]]
        if n % 20 == 0:
            print(f"  {n}/{len(rows)} photos, {time.time() - t0:.0f}s", flush=True)
    (OUT / f"{tag}_{split}_kp_{tier}.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"wrote {OUT / f'{tag}_{split}_kp_{tier}.json'} ({time.time() - t0:.0f}s)")


OSV_EMBED = Path("data/pinpoint/osv5m-embed/siglip2-giant-opt-patch16-384")


def osv(tag: str, split: str) -> None:
    """Street-level source: OSV-5M (4.89M Mapillary images; only SigLIP2 embeddings + lat/lon are on this machine). Per place (text places,
    every Pinpoint / clue area, the pipeline's answer): OSV images within 1 / 5 km and the best SigLIP2 cosine to the query.
    -> <tag>_<split>_osv.json"""
    from ..data.benchmarks import DATA_ROOT, EMBEDDING_KEY

    s = json.loads((OUT / f"{tag}_{split}_shortlist.json").read_text(encoding="utf-8"))
    photos, _ = _photos(tag, split)
    _, center, _ = _answers(tag)
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    areas: dict[int, list] = {}
    for p, src, rk, c in zip(zm["photo"], zm["source"], zm["rank"], zm["center"]):
        areas.setdefault(int(p), []).append((int(src), int(rk), float(c[0]), float(c[1])))
    qroot = DATA_ROOT / tag / "release" / "image_embeddings" / EMBEDDING_KEY
    qids = (qroot / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    qemb = np.fromfile(qroot / "embeddings.f16.bin", dtype=np.float16).reshape(len(qids), -1)
    qrow = {i: k for k, i in enumerate(qids)}
    n = int(json.loads((OSV_EMBED / "manifest.json").read_text())["num_samples"])
    ll = np.fromfile(OSV_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(n, 2)
    emb = np.memmap(OSV_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(n, -1)
    size = 0.1
    cell = (np.floor(ll[:, 0] / size).astype(np.int64) + 10000) * 100000 + (np.floor(ll[:, 1] / size).astype(np.int64) + 10000)
    o = np.argsort(cell, kind="stable"); cell_s = cell[o]

    def near(la, lo, km):
        ci, cj = int(math.floor(la / size)), int(math.floor(lo / size))
        span = int(math.ceil(km / (111 * size * max(math.cos(math.radians(la)), 0.05)))) + 1
        idx = np.concatenate([o[np.searchsorted(cell_s, (ci + di + 10000) * 100000 + cj + dj + 10000):
                                np.searchsorted(cell_s, (ci + di + 10000) * 100000 + cj + dj + 10000, "right")]
                              for di in (-1, 0, 1) for dj in range(-span, span + 1)])
        d = _haversine(la, lo, ll[idx, 0], ll[idx, 1])
        return idx[d < km], d[d < km]

    out, t0 = [], time.time()
    for n_done, r in enumerate(s["photos"]):
        i = r["photo"]
        q = qemb[qrow[r["image_id"]]].astype(np.float32); q /= np.linalg.norm(q)
        places = [dict(kind="text", lat=p["lat"], lon=p["lon"], weight=p["weight"]) for p in r["places"] if p["kind"] == "text"]
        places += [dict(kind=f"area{src}_{rk}", lat=la, lon=lo, weight=0.0) for src, rk, la, lo in sorted(areas.get(i, []))]
        if np.all(np.isfinite(center[i])):
            places.append(dict(kind="answer", lat=float(center[i][0]), lon=float(center[i][1]), weight=0.0))
        for p in places:
            idx, d = near(p["lat"], p["lon"], 5.0)
            if len(idx) > 2000:  # dense cities: the 2000 nearest
                keep = np.argsort(d)[:2000]; idx, d = idx[keep], d[keep]
            sims = (np.asarray(emb[np.sort(idx)], np.float32) @ q) / np.maximum(np.linalg.norm(np.asarray(emb[np.sort(idx)], np.float32), axis=1), 1e-8) if len(idx) else np.zeros(0)
            d = d[np.argsort(idx)]
            for km in (1.0, 5.0):
                m = d < km
                p[f"n_{km:g}"] = int(m.sum())
                p[f"osv_{km:g}"] = float(sims[m].max()) if m.any() else -1.0
            p["right"] = bool(_haversine(p["lat"], p["lon"], *photos[i]["truth"]) < RIGHT_KM)
        out.append(dict(photo=i, image_id=r["image_id"], places=places))
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(s['photos'])} photos, {time.time() - t0:.0f}s", flush=True)
    (OUT / f"{tag}_{split}_osv.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {OUT / f'{tag}_{split}_osv.json'} ({time.time() - t0:.0f}s)")


def osv_megaloc(tag: str, split: str, k: int, km: float, workers: int, batch: int) -> None:
    """MegaLoc on the actual OSV-5M images: per place in <tag>_<split>_osv.json, the k nearest OSV images within km, read from the Hugging
    Face zips (osv_remote), MegaLoc-embedded; adds 'ml' (best cosine to the query, -1 when none) and 'ml_n' per place."""
    import torch

    from .megaloc_gate import _model
    from .osv_remote import INDEX, read

    rows = json.loads((OUT / f"{tag}_{split}_osv.json").read_text(encoding="utf-8"))
    n = int(json.loads((OSV_EMBED / "manifest.json").read_text())["num_samples"])
    ll = np.fromfile(OSV_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(n, 2)
    ids = (OSV_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    zindex = json.loads(INDEX.read_text())
    size = 0.05
    cell = (np.floor(ll[:, 0] / size).astype(np.int64) + 10000) * 100000 + (np.floor(ll[:, 1] / size).astype(np.int64) + 10000)
    o = np.argsort(cell, kind="stable"); cell_s = cell[o]

    def nearest(la, lo):
        ci, cj = int(math.floor(la / size)), int(math.floor(lo / size))
        span = int(math.ceil(km / (111 * size * max(math.cos(math.radians(la)), 0.05)))) + 1
        idx = np.concatenate([o[np.searchsorted(cell_s, (ci + di + 10000) * 100000 + cj + dj + 10000):
                                np.searchsorted(cell_s, (ci + di + 10000) * 100000 + cj + dj + 10000, "right")]
                              for di in (-1, 0, 1) for dj in range(-span, span + 1)])
        d = _haversine(la, lo, ll[idx, 0], ll[idx, 1])
        keep = np.argsort(d)[:k]
        return [ids[i] for i in idx[keep][d[keep] < km] if ids[i] in zindex]

    for r in rows:
        for p in r["places"]:
            p["ml_ids"] = nearest(p["lat"], p["lon"])
    need = sorted({i for r in rows for p in r["places"] for i in p["ml_ids"]})
    print(f"{len(need):,} OSV images to read for {sum(len(r['places']) for r in rows):,} places", flush=True)
    model, tf = _model(), _megaloc_tf()
    photos, _ = _photos(tag, split)
    root = Path(f"data/pinpoint/{tag}/release/images")
    with torch.inference_mode():
        q = {r["photo"]: model(tf((root / r["image_id"]).read_bytes())[None].cuda())[0].float().cpu().numpy() for r in rows}
    emb, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as pool:
        for a in range(0, len(need), batch):
            chunk = need[a:a + batch]
            xs = [(i, tf(b)) for i, b in zip(chunk, pool.map(lambda i: read(zindex[i]), chunk))]
            xs = [(i, x) for i, x in xs if x is not None]
            if xs:
                with torch.inference_mode():
                    e = model(torch.stack([x for _, x in xs]).cuda()).float().cpu().numpy()
                emb.update({i: v for (i, _), v in zip(xs, e)})
            if a // batch % 10 == 0:
                print(f"  {a + len(chunk):,}/{len(need):,} images, {(a + len(chunk)) / (time.time() - t0):.0f}/s", flush=True)
    for r in rows:
        qe = q[r["photo"]] / np.linalg.norm(q[r["photo"]])
        for p in r["places"]:
            v = [emb[i] for i in p["ml_ids"] if i in emb]
            p["ml_n"] = len(v)
            p["ml"] = float(max(float(x @ qe / np.linalg.norm(x)) for x in v)) if v else -1.0
    (OUT / f"{tag}_{split}_osv.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"embedded {len(emb):,}/{len(need):,}; updated {OUT / f'{tag}_{split}_osv.json'} ({time.time() - t0:.0f}s)")


def _scores(tag: str, split: str):
    """Per place: best MegaLoc cosine over its targeted photos, over its nearby photos, and over both (-1 when none)."""
    s = json.loads((OUT / f"{tag}_{split}_shortlist.json").read_text(encoding="utf-8"))
    m = np.load(OUT / f"{tag}_{split}_megaloc.npz")
    q = {int(p): e.astype(np.float32) / np.linalg.norm(e.astype(np.float32)) for p, e in zip(m["q_photo"], m["q_emb"])}
    e = m["emb"].astype(np.float32)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    row = {int(k): r for r, k in enumerate(m["ids"])}
    for r in s["photos"]:
        qe = q[r["photo"]]
        for p in r["places"]:
            for key in ("text_ids", "near_ids"):
                rr = [row[k] for k in p[key] if k in row and np.isfinite(e[row[k], 0])]
                p[key[:-4] + "_sim"] = float(np.max(e[rr] @ qe)) if rr else -1.0
                p[key[:-4] + "_n"] = len(rr)
            p["sim"] = max(p["text_sim"], p["near_sim"])
    return s["photos"]


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((np.mean(pos[:, None] > neg[None]) + 0.5 * np.mean(pos[:, None] == neg[None])))


def report(tag: str, split: str) -> None:
    rows = _scores(tag, split)
    photos, _ = _photos(tag, split)
    km, center, _ = _answers(tag)
    groups = {}
    meta = Path(f"data/pinpoint/{tag}/release/benchmark.csv")
    if meta.exists():
        with meta.open(encoding="utf-8", newline="") as f:
            groups = {r["IMG_ID"]: r.get("group", r["IMG_ID"]) for r in csv.DictReader(f)}
    text = [p for r in rows for p in r["places"] if p["kind"] == "text"]
    n_photos = len(rows)
    base = np.asarray([km[r["photo"]] < RIGHT_KM for r in rows])
    print(f"{tag} {split}: {n_photos} photos, pipeline (lesson 113 answer) < 25 km {100 * base.mean():.1f}%")
    print(f"text places {len(text)} (right {100 * np.mean([p['right'] for p in text]):.1f}%); with >= 1 targeted photo "
          f"{100 * np.mean([p['text_n'] > 0 for p in text]):.1f}%, >= 1 nearby {100 * np.mean([p['near_n'] > 0 for p in text]):.1f}%")
    print("\nAUC, right vs wrong text places (pooled):")
    for key in ("text_sim", "near_sim", "sim", "weight"):
        print(f"  {key:9s} {_auc([p[key] for p in text if p['right']], [p[key] for p in text if not p['right']]):.3f}")
    print("  within-photo: share of photos (with a right and a wrong text place) where the right one has the top score:")
    for key in ("text_sim", "near_sim", "sim", "weight"):
        hits = []
        for r in rows:
            t = [p for p in r["places"] if p["kind"] == "text"]
            if any(p["right"] for p in t) and not all(p["right"] for p in t):
                hits.append(max(t, key=lambda p: p[key])["right"])
        print(f"    {key:9s} {100 * np.mean(hits):.1f}% (n={len(hits)}; heaviest-first order is the 'weight' line)")
    print("\nscore distribution (sim = max of targeted / nearby), right vs wrong text places, and the pipeline's answer when right / wrong:")
    ans = [(p, km[r["photo"]] < RIGHT_KM) for r in rows for p in r["places"] if p["kind"] == "answer"]
    for name, v in (("text right", [p["sim"] for p in text if p["right"]]), ("text wrong", [p["sim"] for p in text if not p["right"]]),
                    ("answer right", [p["sim"] for p, ok in ans if ok]), ("answer wrong", [p["sim"] for p, ok in ans if not ok])):
        v = np.asarray(v); v = v[v > -1]
        print(f"  {name:12s} n={len(v):5d} median {np.median(v):.3f}  p90 {np.quantile(v, .9):.3f}  >=0.5 {100 * np.mean(v >= .5):4.1f}%  >=0.6 {100 * np.mean(v >= .6):4.1f}%")

    # Switch rule: go to the best-scoring text place when its score >= t and beats the answer's score by >= m; grouped CV for (t, m).
    def apply(r, t, mgn, key="sim"):
        t_pl = [p for p in r["places"] if p["kind"] == "text" and p[key] > -1]
        a = next((p for p in r["places"] if p["kind"] == "answer"), None)
        a_sim = a[key] if a else -1.0
        if t_pl:
            b = max(t_pl, key=lambda p: p[key])
            if b[key] >= t and b[key] - a_sim >= mgn and (a is None or _haversine(b["lat"], b["lon"], a["lat"], a["lon"]) >= RIGHT_KM):
                return b["right"], True
        return km[r["photo"]] < RIGHT_KM, False

    grid = [(t, mg) for t in np.arange(0.05, 0.6, 0.0125) for mg in (0.0, 0.025, 0.05, 0.1, 0.15, 0.2)]
    g_of = np.asarray([groups.get(r["image_id"], r["image_id"]) for r in rows])
    ug = np.unique(g_of)
    fold = {gg: k % 5 for k, gg in enumerate(np.random.default_rng(0).permutation(ug))}
    f_of = np.asarray([fold[gg] for gg in g_of])
    print("\nswitch to the best text place (score >= t, beats the answer's score by >= m, > 25 km from the answer); 5-fold grouped CV:")
    for key in ("sim", "text_sim", "near_sim"):
        res, sw = np.zeros(n_photos, bool), np.zeros(n_photos, bool)
        for k in range(5):
            tr, te = np.flatnonzero(f_of != k), np.flatnonzero(f_of == k)
            best = max(grid, key=lambda tm: sum(apply(rows[i], *tm, key=key)[0] for i in tr))
            for i in te:
                res[i], sw[i] = apply(rows[i], *best, key=key)
        fixes, breaks = int((res & ~base).sum()), int((~res & base).sum())
        print(f"  {key:9s} {100 * base.mean():.1f} -> {100 * res.mean():.1f}% ({100 * (res.mean() - base.mean()):+.1f}; switched {sw.sum()}, "
              f"fixes {fixes}, breaks {breaks})")
    oracle = np.asarray([b or any(p["right"] for p in r["places"] if p["kind"] == "text") for r, b in zip(rows, base)])
    print(f"  ceiling (switch exactly when a text place is right and the answer wrong): {100 * oracle.mean():.1f}%")
    for t in (0.15, 0.2, 0.25, 0.3):
        out = [apply(r, t, 0.0) for r in rows]
        res = np.asarray([o[0] for o in out]); sw = np.asarray([o[1] for o in out])
        print(f"  fixed t={t}, m=0: {100 * res.mean():.1f}% (switched {sw.sum()}, fixes {int((res & ~base).sum())}, breaks {int((~res & base).sum())})")
    tiers = {}
    if meta.exists():
        with meta.open(encoding="utf-8", newline="") as f:
            tiers = {r["IMG_ID"]: r.get("tier", "") for r in csv.DictReader(f)}
    if tiers:
        print("\nby tier (commons26): text places right / wrong with sim >= 0.15 and >= 0.2, AUC of sim vs weight, fixed t=0.2 switch")
        for tier in ("landmark", "city", "region", "none"):
            idx = [k for k, r in enumerate(rows) if tiers.get(r["image_id"]) == tier]
            tp = [p for k in idx for p in rows[k]["places"] if p["kind"] == "text"]
            pos, neg = [p for p in tp if p["right"]], [p for p in tp if not p["right"]]
            out = [apply(rows[k], 0.2, 0.0) for k in idx]
            res = np.asarray([o[0] for o in out]); b = base[idx]
            print(f"  {tier:8s} photos {len(idx):4d}; right places {len(pos):3d}: >=0.15 {100 * np.mean([p['sim'] >= .15 for p in pos]):4.1f}% "
                  f">=0.2 {100 * np.mean([p['sim'] >= .2 for p in pos]):4.1f}% | wrong {len(neg):4d}: >=0.15 {100 * np.mean([p['sim'] >= .15 for p in neg]):4.1f}% "
                  f">=0.2 {100 * np.mean([p['sim'] >= .2 for p in neg]):4.1f}% | AUC sim {_auc([p['sim'] for p in pos], [p['sim'] for p in neg]):.2f} "
                  f"weight {_auc([p['weight'] for p in pos], [p['weight'] for p in neg]):.2f} | t=0.2: {100 * b.mean():.1f} -> {100 * res.mean():.1f}% "
                  f"(fixes {int((res & ~b).sum())}, breaks {int((~res & b).sum())})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("shortlist", "fetch", "keypoints", "osv", "osv-megaloc", "report"))
    ap.add_argument("--tier", default="landmark", help="keypoints: commons26 tier to run (landmark / city / region / none / all)")
    ap.add_argument("--tag", default="commons26")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-text", type=int, default=8)
    ap.add_argument("--k-text", type=int, default=10)
    ap.add_argument("--k-near", type=int, default=5)
    ap.add_argument("--k-osv", type=int, default=5, help="osv-megaloc: OSV images per place (nearest within 1 km)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args(argv)
    if a.cmd == "shortlist":
        shortlist(a.tag, a.split, a.max_text, a.k_text, a.k_near)
    elif a.cmd == "fetch":
        fetch(a.tag, a.split, a.workers, a.batch)
    elif a.cmd == "osv":
        osv(a.tag, a.split)
    elif a.cmd == "osv-megaloc":
        osv_megaloc(a.tag, a.split, a.k_osv, 1.0, a.workers, a.batch)
    elif a.cmd == "keypoints":
        keypoints(a.tag, a.split, a.tier, a.k_text, a.k_near, a.workers)
    else:
        report(a.tag, a.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
