"""OSV-5M for spot refinement: inside the area the baseline agent chose (setft_own, --one-km 0.5, RL lambda 0; its centre), are there OSV
images within 1 km of the truth, and does Pinpoint's contrastive space rank one first? On photos the agent already has within 25 km.

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_refine_ceiling.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from geo_search_env.experiment.clue_verify import _haversine
from geo_search_env.experiment.hypotheses import photo_world
from geo_search_env.models.pinpoint_reranker import CONTRASTIVE_CHECKPOINT, _discover_index, _load_image_tower, _unit_xyz

R_KM = 25.0
base = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
root = _discover_index() / "osv_support_cache"
man = json.loads((root / "manifest.json").read_text())
emb = torch.as_tensor(np.fromfile(root / man["files"]["embeddings"], dtype=np.float16).reshape(man["shapes"]["embeddings"]), device="cuda")
ll = np.fromfile(root / man["files"]["latlon_deg"], dtype=np.float32).reshape(man["shapes"]["latlon_deg"]).astype(np.float64)
tree = cKDTree(_unit_xyz(ll))
tower = _load_image_tower(CONTRASTIVE_CHECKPOINT)[0].cuda().eval()
chord = 2 * np.sin(R_KM / 6371.0 / 2)

for tag, key in (("dev3k", "dev3k"), ("commons26dev", "commons26dev:setft_own")):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}_setft_own/photos.json").read_text())
    world, qrows = photo_world(photos)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        q = tower(torch.nn.functional.normalize(torch.as_tensor(np.asarray(world.query_embeddings[qrows], np.float32), device="cuda"), dim=-1), "mp16")
    q = torch.nn.functional.normalize(q.float(), dim=-1).half()
    km = base[f"{key}|RL policy, lambda 0"]
    centre = base[f"{key}|RL policy, lambda 0|center"].astype(np.float64)
    truth = np.asarray([p["truth"] for p in photos])
    sel = np.flatnonzero(km < R_KM)
    n_in, cover, top1, top5, top20, top50, rand50, kern = [], [], [], [], [], [], [], []
    for p in sel:
        idx = np.asarray(tree.query_ball_point(_unit_xyz(centre[p:p + 1])[0], chord), np.int64)
        n_in.append(len(idx))
        if not len(idx):
            cover.append(False), top1.append(False), top5.append(False), top20.append(False), top50.append(False), rand50.append(False), kern.append(False)
            continue
        d = _haversine(truth[p, 0], truth[p, 1], ll[idx, 0], ll[idx, 1])
        s = (emb[torch.as_tensor(idx, device="cuda")] @ q[p]).float().cpu().numpy()
        o = np.argsort(-s)
        cover.append((d < 1).any())
        top1.append(d[o[0]] < 1)
        top5.append((d[o[:5]] < 1).any())
        top20.append((d[o[:20]] < 1).any())
        top50.append((d[o[:50]] < 1).any())
        rand50.append((d[np.random.default_rng(int(p)).permutation(len(d))[:50]] < 1).any())
        k = o[:50]  # similarity-weighted kernel peak over the top 50 (1 km Laplace kernel, T 0.02)
        w = np.exp((s[k] - s[k[0]]) / 0.02)
        pair = _haversine(ll[idx[k], 0][:, None], ll[idx[k], 1][:, None], ll[idx[k], 0][None], ll[idx[k], 1][None])
        kern.append(d[k[np.argmax((w[None] * np.exp(-pair / 1.0)).sum(1))]] < 1)
    n_in, cover, top1, top5, top20, top50, rand50, kern = map(np.asarray, (n_in, cover, top1, top5, top20, top50, rand50, kern))
    b1 = km[sel] < 1
    print(f"== {tag}: {len(sel)} of {len(km)} photos within {R_KM:g} km ({len(sel) / len(km):.1%}); agent < 1 km {b1.mean():.1%} of them; "
          f"OSV images in the area: median {np.median(n_in):.0f}, none {np.mean(n_in == 0):.1%}")
    print(f"   an OSV image < 1 km of the truth exists {cover.mean():.1%} (where the agent is not < 1 km: {np.mean(cover & ~b1):.1%} of these photos)")
    for name, x in (("OSV top-1", top1), ("OSV top-5 any", top5), ("OSV top-20 any", top20), ("OSV top-50 any", top50), ("random 50 any", rand50), ("OSV kernel peak", kern)):
        print(f"   {name:16s} < 1 km {x.mean():5.1%}; right where the agent is not {np.mean(x & ~b1):5.1%}, agent right where it is not "
              f"{np.mean(~x & b1):5.1%}  (whole set: {len(sel) / len(km) * np.mean(x & ~b1):.1%} / {len(sel) / len(km) * np.mean(~x & b1):.1%})")
