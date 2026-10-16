"""Near-area OSM check as an area verifier on commons26 dev (areas = Pinpoint's top areas, rank order; right = centre < 25 km)."""
import csv
import json

import numpy as np

bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
rows = [json.loads(l) for l in open("artifacts/commons26/osm_near_dev.jsonl")]


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


print(f"{len(rows)} dev photos")
right = []
for r in rows:
    t = (float(bench[r["image_id"]]["LAT"]), float(bench[r["image_id"]]["LON"]))
    right.append([bool(hav(t[0], t[1], a[0], a[1]) < 25) for a in r["areas"]])
first = np.mean([bool(x) and x[0] for x in right])
anyr = np.mean([any(x) for x in right])
print(f"area 1 right {first:.1%}; any area right {anyr:.1%}")
for T in (5, 20, 100, 500, 4000):
    # per area: count of distinct queries with a match near it whose global match count <= T
    sel_photos, pick_right, first_right, flagged_right, flagged_n, cover = 0, [], [], 0, 0, 0
    for r, x in zip(rows, right):
        score = [len({q for q, _, n in h if n <= T}) for h in r["hits"]]
        for k, s in enumerate(score):
            if s:
                flagged_n += 1
                flagged_right += x[k]
        if any(score):
            cover += 1
            k = int(np.argmax(score))  # most supported area, ties -> better Pinpoint rank
            pick_right.append(x[k] if x else False)
            first_right.append(x[0] if x else False)
    n = len(rows)
    gain = (sum(pick_right) - sum(first_right)) / n
    print(f"  matches <= {T:5d}: photos with a flagged area {cover / n:5.1%}; flagged areas right {flagged_right / max(flagged_n, 1):5.1%} of {flagged_n}; "
          f"on those photos pick most-supported area {np.mean(pick_right) if pick_right else 0:5.1%} vs area 1 {np.mean(first_right) if first_right else 0:5.1%} "
          f"-> {100 * gain:+.1f} pts of dev")
# areas ranked 2-5 only: does a hit there predict right?
for T in (20, 100):
    hit_r = [x[k] for r, x in zip(rows, right) for k, h in enumerate(r["hits"]) if k > 0 and any(n <= T for _, _, n in h)]
    nohit_r = [x[k] for r, x in zip(rows, right) for k, h in enumerate(r["hits"]) if k > 0 and not any(n <= T for _, _, n in h)]
    print(f"  areas ranked 2+: with a match <= {T}: right {np.mean(hit_r):.1%} (n {len(hit_r)}), without {np.mean(nohit_r):.1%} (n {len(nohit_r)})")
