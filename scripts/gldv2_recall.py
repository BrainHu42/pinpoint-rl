# GLDv2 pilot (lesson R10): recall of a right landmark among the top-K MegaLoc landmarks of the mini-gallery (all distractors), commons26 dev.
# Usage: [GLD_EMB=megaloc|siglip|both] PYTHONPATH=src .venv/bin/python scripts/gldv2_recall.py
import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_pilot as p  # noqa: E402

keepj = json.loads((p.OUT / "keep.json").read_text())
z = np.load(p.OUT / "megaloc.npz")
ids, ll = p._landmarks()
loc = dict(zip(ids.tolist(), ll))
photos, truth, centre, km = p._photos()
nq = int(z["n_queries"])
sims_t, ok = p._sims(nq)
sims_t = sims_t[torch.as_tensor(ok, device="cuda")]
lm_of = np.asarray([keepj["images"][i] for i in z["gallery_ids"]])[ok]
lms, inv = np.unique(lm_of, return_inverse=True)
print(f"embedding: {p._emb()}")
idx = torch.as_tensor(inv, device="cuda")[:, None].expand_as(sims_t)
best = torch.full((len(lms), nq), -1.0, device="cuda").scatter_reduce(0, idx, sims_t, "amax").T.cpu().numpy()
lm_ll = np.asarray([loc[int(x)] for x in lms])
order = np.argsort(-best, 1)
bm = {r["IMG_ID"]: r for r in csv.DictReader(open(p.RELEASE / "benchmark.csv"))}
tier = np.asarray([bm[e["image_id"]]["tier"] for e in photos])
d = np.stack([p._hav(truth[:, 0], truth[:, 1], lm_ll[order[:, k], 0], lm_ll[order[:, k], 1]) for k in range(100)], 1)
has1 = np.asarray([(p._hav(truth[q, 0], truth[q, 1], lm_ll[:, 0], lm_ll[:, 1]) < 1).any() for q in range(nq)])
print(f"photos with a gallery landmark within 1 km of the truth: {100 * has1.mean():.1f}%")
for r in (1, 25):
    print(f"right landmark (< {r} km) among the top K landmarks, % of all 977 / of the pipeline's misses (< 25 km):")
    for k in (1, 3, 5, 10, 20, 50, 100):
        hit = (d[:, :k] < r).any(1)
        print(f"  K {k:3d}: {100 * hit.mean():5.1f} / {100 * hit[km >= 25].mean():5.1f}   landmark tier {100 * hit[tier == 'landmark'].mean():5.1f}")
