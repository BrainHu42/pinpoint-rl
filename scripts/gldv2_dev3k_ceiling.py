# GLDv2 1 km refinement ceiling on dev3k (lesson R10): photos the pipeline places within 25 km but not 1 km, with a GLDv2 landmark near the truth.
# Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_dev3k_ceiling.py   (answers from the viewer's traces, artifacts/viz/traces/dev3k)
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_ceiling as g  # noqa: E402
import gldv2_pilot as p  # noqa: E402

ids, ll = p._landmarks()
tree = cKDTree(g._xyz(ll))
rows = []
for f in sorted(Path("artifacts/viz/traces/dev3k").glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else -1):
    if not f.stem.isdigit():
        continue
    t = json.loads(f.read_text())
    fin = t["agent"]["0"]["final"]
    rows.append((t["truth"]["lat"], t["truth"]["lon"], fin["lat"], fin["lon"], fin["km"]))
a = np.asarray(rows, np.float64)
truth, ans, km = a[:, :2], a[:, 2:4], a[:, 4]
print(f"dev3k {len(a)} photos: pipeline < 1 km {100 * np.mean(km < 1):.1f}%, < 25 km {100 * np.mean(km < 25):.1f}%")
mid = (km >= 1) & (km < 25)
for r in (0.3, 1.0):
    near_t = np.asarray([len(x) > 0 for x in tree.query_ball_point(g._xyz(truth), g._chord(r))])
    print(f"  landmark within {r} km of the truth: all {100 * near_t.mean():.1f}%; pipeline 1-25 km ({mid.sum()}) {100 * near_t[mid].mean():.1f}% "
          f"= {100 * (near_t & mid).sum() / len(a):.1f} pts of dev3k; pipeline < 1 km {100 * near_t[km < 1].mean():.1f}%")
n_within = np.asarray([len(x) for x in tree.query_ball_point(g._xyz(ans), g._chord(25))])
print(f"  landmarks within 25 km of the answer (the refinement's choices), 1-25 km photos: median {np.median(n_within[mid]):.0f}")
