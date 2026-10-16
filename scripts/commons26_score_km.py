"""Score km arrays with commons26's scorer (continent-balanced, uploader-clustered SEs): the scorer's distance replaced by our per-photo km."""
import csv, importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("c26score", "data/pinpoint/commons26/release/score.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26.npz")
photos = json.loads(open("artifacts/photo_sets/commons26_setft_own/photos.json").read())
ids = [e["image_id"] for e in photos]
tmp = "/home/brianhu/.claude/jobs/a1da8dab/tmp/c26_pred.csv"
with open(tmp, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["IMG_ID", "LAT", "LON"])
    for i, k in enumerate(ids):
        w.writerow([k, i, 0])  # LAT carries the photo's position
for key in ("released", "retrained", "RL policy, lambda 0", "RL policy, lambda 0.005"):
    km = z[f"commons26:setft_own|{key}"]
    m.geodesic_km = lambda p, l, km=km: km[p[:, 0].astype(int)]
    for split in ("test", "dev"):
        print(f"\n=== {key} / {split}")
        m.score(__import__("pathlib").Path(tmp), split)
