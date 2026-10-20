"""Why OSV-5M does not help: (1) scale: OSV vs MP16 retrieval top-1 accuracy at 1 / 25 / 200 / 750 / 2500 km; (2) content: raw SigLIP2
cosine to the query of OSV images vs MP16 photos at the same distance from the truth (MP16 without the query photo and its photographer),
per distance bin, with the share of photos having such an image at all. dev3k and commons26 dev.

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_diagnose.py

Diagnosis (2026-10-19; with osv_diagnose_ml.py and osv_examples.py -> artifacts/osv_refine/{diagnose12,diagnose_ml}.log, examples.jpg):
1. Scale: OSV retrieval carries regional appearance, not place identity. commons26 dev raw top-1 % < 1 / 25 / 200 / 750 / 2500 km: MP16
   3.5 / 12.5 / 29.1 / 55.0 / 76.7, OSV 1.1 / 7.9 / 22.0 / 53.3 / 75.5 (dev3k MP16 21.4 / 33.0 / 42.7 / 58.7 / 74.8, OSV 3.1 / 11.1 / 20.4
   / 39.6 / 60.3): close at 750-2500 km, which the pipeline already has, and 3-7x worse at 1 km.
2. Sampling: OSV is spread along roads, Flickr photos cluster where people photograph. An image within 50 m of the truth: OSV 5.9% of
   dev3k photos vs MP16 (other photographers) 42.1%; commons26 dev 5.7 vs 15.8%. Near OSV images are mostly 50-200 m away.
3. Content: no same-place matches. MegaLoc cosine > 0.2: OSV within 50 m 0 of 54 images, 50-200 m 4 of 633, max over 8,065 OSV pairs 0.26;
   MP16 within 50 m 4.8% (> 0.3: 1.5%), max 0.73. Near vs far (5-25 km) per-photo MegaLoc AUC: OSV 0.55 / 0.62, MP16 0.62 / 0.61
   (dev3k / commons26 dev). The query photos show people, interiors, events, landmark details, views; OSV shows the road ahead.
4. Needle in a haystack: per image, SigLIP2 likes near OSV images about as much as near MP16 photos (near vs far AUC 0.69 / 0.72 vs
   0.69 / 0.71: neighbourhood look), but an area holds ~2 near OSV images against ~2,600 far ones, so ranking never surfaces them; the best
   near OSV image beats the best far one on 41% / 43% of photos (MP16 67% / 69%, with more and closer near photos).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from geo_search_env.experiment.clue_verify import OSV_EMBED, _haversine
from geo_search_env.experiment.density_probe import _index_rows
from geo_search_env.experiment.hypotheses import photo_world
from geo_search_env.experiment.strategy_search import MP16_EMBED
from geo_search_env.models.pinpoint_reranker import _unit_xyz

BINS = [(0, 0.05), (0.05, 0.2), (0.2, 1), (1, 5), (5, 25)]
CAP = 200  # images per (photo, bin), random
SCALES = (1, 25, 200, 750, 2500)

# (1) scale
for space in ("raw", "contrastive"):
    d = np.load(f"artifacts/osv_propose/search{'_contrastive' if space == 'contrastive' else ''}.npz")
    for st in ("dev3k", "commons26"):
        m = d["set"] == st
        t = d["truth"][m]
        row = []
        for src in ("mp16", "osv"):
            ll = d[f"{src}_ll"][m][:, 0]
            km = _haversine(t[:, 0], t[:, 1], ll[:, 0], ll[:, 1])
            row.append(" / ".join(f"{100 * np.mean(km < s):4.1f}" for s in SCALES))
        print(f"(1) {st:9s} {space:11s} top-1 % < {' / '.join(map(str, SCALES))} km: MP16 {row[0]}   OSV {row[1]}")

# (2) content
n_osv = int(json.loads((OSV_EMBED / "manifest.json").read_text())["num_samples"])
osv_ll = np.fromfile(OSV_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(n_osv, 2).astype(np.float64)
osv_emb = np.memmap(OSV_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(n_osv, -1)
rows = _index_rows()
mp_ll_all = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2)
mp_ll = mp_ll_all[rows].astype(np.float64)
mp_emb = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
authors = np.load("artifacts/agent/mp16_author_by_cache_row.npy")
trees = {"OSV": cKDTree(_unit_xyz(osv_ll)), "MP16": cKDTree(_unit_xyz(mp_ll))}
chord = lambda km: 2 * np.sin(km / 6371.0 / 2)
rng = np.random.default_rng(0)
for tag in ("dev3k", "commons26dev"):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}_setft_own/photos.json").read_text())
    world, qrows = photo_world(photos)
    q = np.asarray(world.query_embeddings[qrows], np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    truth = np.asarray([p["truth"] for p in photos], np.float64)
    qrow = np.asarray([int(p["row"]) if "row" in p else -1 for p in photos])
    stats = {(src, b): ([], [], 0) for src in trees for b in range(len(BINS))}
    for i in range(len(photos)):
        for src, tree in trees.items():
            idx = np.asarray(tree.query_ball_point(_unit_xyz(truth[i:i + 1])[0], chord(25.0)), np.int64)
            if src == "MP16" and qrow[i] >= 0:
                cache = rows[idx]
                idx = idx[(cache != qrow[i]) & (authors[cache] != authors[qrow[i]])]
            ll = osv_ll[idx] if src == "OSV" else mp_ll[idx]
            dist = _haversine(truth[i, 0], truth[i, 1], ll[:, 0], ll[:, 1])
            for b, (lo, hi) in enumerate(BINS):
                sel = idx[(dist >= lo) & (dist < hi)]
                if not len(sel):
                    continue
                if len(sel) > CAP:
                    sel = rng.choice(sel, CAP, replace=False)
                sel = np.sort(sel)
                e = np.asarray(osv_emb[sel] if src == "OSV" else mp_emb[rows[sel]], np.float32)
                c = (e / np.linalg.norm(e, axis=1, keepdims=True)) @ q[i]
                means, bests, n = stats[(src, b)]
                means.append(c.mean())
                bests.append(c.max() if len(c) >= 10 else np.nan)
                stats[(src, b)] = (means, bests, n + 1)
    print(f"\n(2) {tag} ({len(photos)} photos): raw SigLIP2 cosine to the query of images at a distance from the truth "
          f"(MP16 without the query photo / photographer); per bin: photos with any, mean cosine, best of up to {CAP} (photos with >= 10)")
    for b, (lo, hi) in enumerate(BINS):
        cells = []
        for src in trees:
            means, bests, n = stats[(src, b)]
            cells.append(f"{src:4s} any {100 * n / len(photos):5.1f}%  mean {np.mean(means):.3f}  best {np.nanmean(bests):.3f}")
        print(f"   {lo:>5g}-{hi:<4g} km: " + "   |   ".join(cells))
