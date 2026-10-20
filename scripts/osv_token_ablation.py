"""How much does Pinpoint's reranker lean on its OSV support token? dev3k, the same candidate pools scored with the token as trained and
with every candidate's OSV neighbours removed (as for a candidate with no OSV photo within 1 km); also the token's coverage.

Usage: PYTHONPATH=src [PINPOINT_RERANKER_CKPT=...] .venv/bin/python scripts/osv_token_ablation.py
"""

import json
import os
from pathlib import Path

import numpy as np

from geo_search_env.experiment.clue_verify import _haversine
from geo_search_env.experiment.strategy_search import MP16_EMBED
from geo_search_env.models.pinpoint_reranker import PinpointReranker

photos = json.loads(Path("artifacts/photo_sets/dev3k/photos.json").read_text())
mp16 = np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
emb = np.stack([np.asarray(mp16[int(p["row"])]) for p in photos]).astype(np.float32)
truth = np.asarray([p["truth"] for p in photos])

r = PinpointReranker()
raw, projected = r.encode_queries(emb)
rows, ranks, flags = r._union(r._search(raw, projected, None, None))
valid = rows >= 0
neighbors, _ = r._osv_support(np.where(valid, rows, 0).reshape(-1))
n_osv = ((neighbors >= 0).sum(1)).reshape(rows.shape)
print(f"checkpoint: {os.environ.get('PINPOINT_RERANKER_CKPT', 'released')}")
print(f"pool: {valid.sum(1).mean():.1f} candidates per photo; OSV neighbours within {r.osv_radius_km:g} km per candidate: "
      f"none {100 * np.mean(n_osv[valid] == 0):.1f}%, mean {n_osv[valid].mean():.2f} of {r.osv_k}")


def top1(scores):
    best = np.argmax(np.where(valid, scores, -np.inf), axis=1)
    ll = np.asarray(r.latlon[rows[np.arange(len(rows)), best]])
    return _haversine(truth[:, 0], truth[:, 1], ll[:, 0], ll[:, 1]), best


s_on, _ = r._score(raw, projected, rows, ranks, flags)
d_on, b_on = top1(s_on)
orig = r._osv_support
r._osv_support = lambda flat: (np.full((len(flat), r.osv_k), -1), np.full((len(flat), r.osv_k), r.osv_radius_km, np.float32))
s_off, _ = r._score(raw, projected, rows, ranks, flags)
d_off, b_off = top1(s_off)
r._osv_support = orig
for name, d in (("token on ", d_on), ("token off", d_off)):
    print(f"  {name}: top-1 < 1 km {100 * np.mean(d < 1):.1f}%, < 25 km {100 * np.mean(d < 25):.1f}%, < 200 km {100 * np.mean(d < 200):.1f}%")
changed = b_on != b_off
print(f"  top-1 changed on {100 * changed.mean():.1f}% of photos; < 25 km fixes {int(((d_off < 25) & ~(d_on < 25)).sum())} (off right, on wrong) "
      f"vs {int(((d_on < 25) & ~(d_off < 25)).sum())} (on right, off wrong)")
oracle = np.asarray([np.any(_haversine(truth[q, 0], truth[q, 1], *np.asarray(r.latlon[rows[q][valid[q]]]).T) < 25) for q in range(len(rows))])
print(f"  pool oracle < 25 km {100 * oracle.mean():.1f}%")
