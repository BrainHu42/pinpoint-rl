# GLDv2 ceiling control (lesson R9): landmarks near the truth vs near the pipeline's answer, images per nearby landmark. Needs gldv2_ceiling.py coords + ceiling.
# Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_ceiling_control.py
import csv
import json
import sys
import urllib.parse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scripts")
import gldv2_ceiling as g  # noqa: E402

O = g.OUT
cats = {int(r["landmark_id"]): urllib.parse.unquote(r["category"].rsplit("Category:", 1)[-1]).replace("_", " ")
        for r in csv.DictReader(open(O / "train_label_to_category.csv"))}
found = json.loads((O / "coords.json").read_text())
n_img = {}
for r in csv.DictReader(open(O / "train.csv")):
    n_img[int(r["landmark_id"])] = n_img.get(int(r["landmark_id"]), 0) + 1
ids = np.asarray([k for k in cats if cats[k] in found])
ll = np.asarray([found[cats[k]] for k in ids])
imgs = np.asarray([n_img.get(k, 0) for k in ids])
tree = cKDTree(g._xyz(ll))
z = np.load(O / "ceiling_commons26dev.npz")
tier, km = z["tier"], z["km"]
key = "commons26dev:setft_own|RL policy, lambda 0"
centre = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")[f"{key}|center"]
photos = json.loads(Path("artifacts/photo_sets/commons26dev_setft_own/photos.json").read_text())
truth = np.asarray([e["truth"] for e in photos], np.float64)
geo, wrong = tier != "none", km >= 25
print("landmarks with coordinates: images per landmark median", np.median(imgs), "p25/p75", np.percentile(imgs, [25, 75]))
for r in (0.3, 1):
    at_t = tree.query_ball_point(g._xyz(truth), g._chord(r))
    at_a = tree.query_ball_point(g._xyz(centre.astype(np.float64)), g._chord(r))
    has_t = np.asarray([len(x) > 0 for x in at_t])
    has_a = np.asarray([len(x) > 0 for x in at_a])
    for lab, m in (("landmark tier, wrong", (tier == "landmark") & wrong), ("geolocatable, wrong", geo & wrong), ("geolocatable, right", geo & ~wrong)):
        both = m & has_t
        nimg = [imgs[x].sum() for x, b in zip(at_t, both) if b]
        print(f"  {r} km, {lab:22s} {m.sum():3d}: landmark near truth {100 * has_t[m].mean():4.1f}%, near the pipeline's answer "
              f"{100 * has_a[m].mean():4.1f}%; near truth but not the answer {100 * (has_t & ~has_a)[m].mean():4.1f}%; images of the nearby "
              f"landmarks median {np.median(nimg) if nimg else 0:.0f}")
