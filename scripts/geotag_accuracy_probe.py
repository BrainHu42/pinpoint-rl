# Are unreliable index geotags behind near-misses? Flickr geotag accuracy (YFCC100M, 1 = world .. 16 = street) of each dev3k pool candidate
# (setft_own pools), against whether the candidate is right / a near-miss, and the reranker's top-1 error by its accuracy level.
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from geo_search_env.experiment.pinpoint_reference import load_pools  # noqa: E402

emb = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
ids = np.asarray([int(Path(s.strip()).stem.split("_")[-1]) for s in open(emb / "image_ids.txt")], np.int64)  # cache row -> Flickr id
z = np.load("data/yfcc100m/geo.npz", mmap_mode="r")
pid, acc = np.asarray(z["photo_id"]), np.asarray(z["accuracy"])
o = np.argsort(pid)
pid, acc = pid[o], acc[o]
pos = np.clip(np.searchsorted(pid, ids), 0, len(pid) - 1)
row_acc = np.where(pid[pos] == ids, acc[pos], -1)
print(f"MP16 cache rows {len(ids)}; found in YFCC {np.mean(row_acc >= 0):.1%}; accuracy level distribution (found):")
v, c = np.unique(row_acc[row_acc >= 0], return_counts=True)
print("   " + ", ".join(f"{a}: {100 * n / c.sum():.1f}%" for a, n in zip(v, c)))

pools = load_pools("mp16_setft_own")
for tag in ("dev3k_setft_own",):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text())
    accs, kms, top_acc, top_km = [], [], [], []
    for e in photos:
        i = e["index"]
        val = pools["valid"][i]
        cr = pools["cache_rows"][i][val]
        ll = pools["latlon"][i][val].astype(np.float64)
        t = np.radians(np.asarray(e["truth"]))
        la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
        h = np.sin((la - t[0]) / 2) ** 2 + np.cos(t[0]) * np.cos(la) * np.sin((lo - t[1]) / 2) ** 2
        km = 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        a = row_acc[cr]
        accs.append(a)
        kms.append(km)
        j = int(np.argmax(pools["scores"][i][val]))
        top_acc.append(a[j])
        top_km.append(km[j])
    A, K = np.concatenate(accs), np.concatenate(kms)
    top_acc, top_km = np.asarray(top_acc), np.asarray(top_km)
    print(f"\n{tag}: {len(photos)} photos, {len(A)} candidates")
    print("  candidate accuracy band: share of candidates | < 25 km | 25-100 km (near-miss) | > 100 km")
    for lo_, hi_ in ((-1, 0), (1, 10), (11, 13), (14, 15), (16, 16)):
        m = (A >= lo_) & (A <= hi_)
        if m.any():
            print(f"    level {lo_:>2}-{hi_:<2} {100 * m.mean():5.1f}% | {100 * np.mean(K[m] < 25):5.1f}% | {100 * np.mean((K[m] >= 25) & (K[m] < 100)):5.1f}% | "
                  f"{100 * np.mean(K[m] >= 100):5.1f}%")
    print("  reranker top-1 by its accuracy band: share of photos | top-1 < 25 km | 25-100 km")
    for lo_, hi_ in ((-1, 0), (1, 10), (11, 13), (14, 15), (16, 16)):
        m = (top_acc >= lo_) & (top_acc <= hi_)
        if m.any():
            print(f"    level {lo_:>2}-{hi_:<2} {100 * m.mean():5.1f}% | {100 * np.mean(top_km[m] < 25):5.1f}% | {100 * np.mean((top_km[m] >= 25) & (top_km[m] < 100)):5.1f}%")
