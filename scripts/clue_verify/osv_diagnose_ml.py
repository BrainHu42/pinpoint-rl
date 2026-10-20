"""Why OSV-5M does not help, part 2: does same-place similarity single out images near the truth, for OSV vs MP16 at matched distances?
Per sampled photo: the 10 nearest OSV images and the 10 nearest MP16 photos (other photographers, not the query) within 200 m of the truth,
and 10 random of each at 5-25 km; raw SigLIP2 and MegaLoc cosine to the query. Reported per source and distance bin: photos with any,
mean cosine, share > 0.2 / 0.3, and the per-photo AUC near (< 200 m) vs far (5-25 km).

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_diagnose_ml.py
"""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from geo_search_env.experiment.clue_verify import OSV_EMBED, _haversine, _megaloc_tf
from geo_search_env.experiment.density_probe import _index_rows
from geo_search_env.experiment.hypotheses import photo_world
from geo_search_env.experiment.megaloc_gate import _model
from geo_search_env.experiment.osv_remote import INDEX, read
from geo_search_env.experiment.sft_data import MP16Images
from geo_search_env.experiment.strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery
from geo_search_env.models.pinpoint_reranker import _unit_xyz

SAMPLE = {"dev3k": 500, "commons26dev": 300}
OUT = Path("artifacts/osv_refine/diagnose_ml.npz")
n_osv = int(json.loads((OSV_EMBED / "manifest.json").read_text())["num_samples"])
osv_ll = np.fromfile(OSV_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(n_osv, 2).astype(np.float64)
osv_emb = np.memmap(OSV_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(n_osv, -1)
osv_ids = (OSV_EMBED / "image_ids.txt").read_text().splitlines()
rows = _index_rows()
mp_ll = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2)[rows].astype(np.float64)
mp_emb = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
authors = np.load("artifacts/agent/mp16_author_by_cache_row.npy")
row_index = _memmap_gallery(MP16_EMBED)["row_index"]
with MP16_CSV.open(encoding="utf-8", newline="") as stream:
    reader = csv.reader(stream)
    next(reader)
    csv_ids = [r[0] for r in reader]
trees = {"OSV": cKDTree(_unit_xyz(osv_ll)), "MP16": cKDTree(_unit_xyz(mp_ll))}
chord = lambda km: 2 * np.sin(km / 6371.0 / 2)
rng = np.random.default_rng(0)
zindex = json.loads(INDEX.read_text())
model, tf = _model(), _megaloc_tf()
mp16 = MP16Images()

items = []  # (tag, photo, source, bin, image key, siglip cosine, km)
for tag, n in SAMPLE.items():
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}_setft_own/photos.json").read_text())
    world, qrows = photo_world(photos)
    q = np.asarray(world.query_embeddings[qrows], np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    truth = np.asarray([p["truth"] for p in photos], np.float64)
    qrow = np.asarray([int(p["row"]) if "row" in p else -1 for p in photos])
    for i in np.sort(rng.choice(len(photos), n, replace=False)):
        for src, tree in trees.items():
            idx = np.asarray(tree.query_ball_point(_unit_xyz(truth[i:i + 1])[0], chord(25.0)), np.int64)
            if src == "MP16" and qrow[i] >= 0:
                c = rows[idx]
                idx = idx[(c != qrow[i]) & (authors[c] != authors[qrow[i]])]
            if src == "OSV":
                idx = idx[[osv_ids[j] in zindex for j in idx]]
            ll = osv_ll[idx] if src == "OSV" else mp_ll[idx]
            dist = _haversine(truth[i, 0], truth[i, 1], ll[:, 0], ll[:, 1])
            near = idx[dist < 0.2][np.argsort(dist[dist < 0.2])][:10]
            far = idx[(dist >= 5) & (dist < 25)]
            far = rng.choice(far, min(10, len(far)), replace=False) if len(far) else far
            for j in np.concatenate([near, far]):
                e = np.asarray(osv_emb[j] if src == "OSV" else mp_emb[rows[j]], np.float32)
                km = _haversine(truth[i, 0], truth[i, 1], *(osv_ll[j] if src == "OSV" else mp_ll[j]))
                key = osv_ids[j] if src == "OSV" else csv_ids[row_index[rows[j]]]
                items.append((tag, int(i), src, key, float(e @ q[i] / np.linalg.norm(e)), float(km)))
print(f"{len(items):,} (photo, image) pairs; reading and MegaLoc-embedding", flush=True)


def load(it):
    try:
        return tf(read(zindex[it[3]]) if it[2] == "OSV" else mp16.read(it[3]))
    except Exception:  # noqa: BLE001
        return None


qml = {tag: torch.nn.functional.normalize(torch.as_tensor(np.load(f"artifacts/agent/megaloc_queries_{tag}_setft_own.npy"), device="cuda").float(), dim=-1)
       for tag in SAMPLE}
ml = np.full(len(items), np.nan, np.float32)
with ThreadPoolExecutor(32) as pool:
    for a in range(0, len(items), 128):
        chunk = items[a:a + 128]
        xs = [(k, x) for k, x in zip(range(a, a + len(chunk)), pool.map(load, chunk)) if x is not None]
        if xs:
            with torch.inference_mode():
                e = torch.nn.functional.normalize(model(torch.stack([x for _, x in xs]).cuda()).float(), dim=-1)
            for (k, _), v in zip(xs, e):
                ml[k] = float(v @ qml[items[k][0]][items[k][1]])
        if a // 128 % 40 == 0:
            print(f"  {a + len(chunk):,}/{len(items):,}", flush=True)
np.savez(OUT, tag=np.asarray([x[0] for x in items]), photo=np.asarray([x[1] for x in items]), src=np.asarray([x[2] for x in items]),
         siglip=np.asarray([x[4] for x in items]), km=np.asarray([x[5] for x in items]), megaloc=ml)


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.mean((a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None]))


tags, ph, src, sg, km = (np.asarray([x[k] for x in items]) for k in (0, 1, 2, 4, 5))
for tag in SAMPLE:
    print(f"\n== {tag}: {SAMPLE[tag]} sampled photos")
    for s in ("OSV", "MP16"):
        for name, lo, hi in (("0-50 m", 0, 0.05), ("50-200 m", 0.05, 0.2), ("5-25 km", 5, 25)):
            m = (tags == tag) & (src == s) & (km >= lo) & (km < hi) & np.isfinite(ml)
            print(f"   {s:4s} {name:9s}: photos with any {100 * len(set(ph[m])) / SAMPLE[tag]:5.1f}%, images {m.sum():5d}; SigLIP mean {sg[m].mean():.3f}; "
                  f"MegaLoc mean {ml[m].mean():.3f}, > 0.2 {100 * np.mean(ml[m] > 0.2):4.1f}%, > 0.3 {100 * np.mean(ml[m] > 0.3):4.1f}%")
        for name, x in (("SigLIP", sg), ("MegaLoc", ml)):
            aucs, wins = [], []
            for p in set(ph[(tags == tag) & (src == s)]):
                m = (tags == tag) & (src == s) & (ph == p) & np.isfinite(ml)
                near, far = x[m & (km < 0.2)], x[m & (km >= 5)]
                if len(near) and len(far):
                    aucs.append(auc(near, far))
                    wins.append(near.max() > far.max())
            print(f"   {s:4s} {name:7s} near (< 200 m) vs far (5-25 km): per-photo AUC {np.mean(aucs):.2f}, best near beats best far {100 * np.mean(wins):.0f}% "
                  f"({len(aucs)} photos)")
