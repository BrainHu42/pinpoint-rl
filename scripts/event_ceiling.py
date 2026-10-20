# Usage: .venv/bin/python scripts/event_ceiling.py  (lesson R5; needs scripts/scene_zeroshot.py output)
# Ceiling for "same-event" retrieval: index photos near the truth in place AND upload time (Flickr photo id ~ upload order).
import csv, json
import numpy as np
from scipy.spatial import cKDTree

E = "data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384/"
IDX = "data/pinpoint/pinpoint-submission/exp/contrastive_retrieval/checkpoints/attention_retrieval_index/cabeb0568af3b2f1/"
ids = open(E + "image_ids.txt").read().split()
pid_all = np.asarray([int(i.split("_")[-1].split(".")[0]) for i in ids], dtype=np.int64)
cache = np.fromfile(IDX + "cache_indices.bin", dtype=np.int64)
ll = np.fromfile(IDX + "latlon_deg.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
ipid = pid_all[cache]
author = {}
with open("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv", newline="") as fh:
    rd = csv.reader(fh); next(rd)
    for r in rd:
        author[r[0]] = r[1]
iauth = np.asarray([author.get(ids[c], "") for c in cache], dtype=object)
photos = json.load(open("artifacts/photo_sets/dev3k/photos.json"))
truth = np.asarray([e["truth"] for e in photos], dtype=np.float64)
qpid = pid_all[[e["row"] for e in photos]]
qauth = np.asarray([author[ids[e["row"]]] for e in photos], dtype=object)
xyz = lambda a: np.stack([np.cos(np.radians(a[:, 0])) * np.cos(np.radians(a[:, 1])), np.cos(np.radians(a[:, 0])) * np.sin(np.radians(a[:, 1])), np.sin(np.radians(a[:, 0]))], 1)
tree = cKDTree(xyz(ll))
near = tree.query_ball_point(xyz(truth), 2 * np.sin(25 / 6371.0088 / 2))
km = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")["dev3k|RL policy, lambda 0"]
p = np.load("artifacts/retrieval/skyline_zeroshot_dev3k.npy")
labels = np.asarray(["mountain", "hills", "flat", "street", "indoor", "people", "object"])[p.argmax(1)]
low = np.isin(labels, ["indoor", "people", "object"])
wrong = km >= 25
for window, name in ((5_000_000, "~1 day"), (50_000_000, "~10 days")):
    any_m, other_m = np.zeros(len(photos), bool), np.zeros(len(photos), bool)
    for q, rows in enumerate(near):
        rows = np.asarray(rows, dtype=np.int64)
        m = np.abs(ipid[rows] - qpid[q]) < window
        any_m[q] = m.any()
        other_m[q] = (m & (iauth[rows] != qauth[q])).any()
    print(f"index photo < 25 km and |photo id diff| < {window:.0e} ({name}): all {100 * any_m.mean():.1f}%, by another photographer "
          f"{100 * other_m.mean():.1f}%")
    for lab, m in (("pipeline wrong", wrong), ("wrong & indoor/people/object", wrong & low)):
        print(f"   {lab:30s} {100 * m.mean():5.1f}% of photos: event-mate exists {100 * any_m[m].mean():.1f}% (another photographer "
              f"{100 * other_m[m].mean():.1f}%) -> ceiling {100 * (m & any_m).mean():.1f} pts ({100 * (m & other_m).mean():.1f} via others)")

print("\nluck control: same places, query photo id shifted (wrong time)")
for window in (500_000, 5_000_000):
    for shift in (0, 300_000_000, -300_000_000, 1_000_000_000):
        any_m, other_m = np.zeros(len(photos), bool), np.zeros(len(photos), bool)
        for q, rows in enumerate(near):
            rows = np.asarray(rows, dtype=np.int64)
            m = np.abs(ipid[rows] - (qpid[q] + shift)) < window
            any_m[q] = m.any()
            other_m[q] = (m & (iauth[rows] != qauth[q])).any()
        m = wrong & low
        print(f"  window {window:.0e} shift {shift:+.0e}: all {100 * any_m.mean():.1f}% (others {100 * other_m.mean():.1f}%); wrong&low "
              f"{100 * any_m[m].mean():.1f}% (others {100 * other_m[m].mean():.1f}%)")
