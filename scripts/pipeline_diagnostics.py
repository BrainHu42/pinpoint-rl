# Where does the current pipeline lose? (action_agent --one-km 0.5 on setft_own, RL lambda 0; dev3k and commons26 dev; cached outputs only)
#   stage buckets per photo: retrieval miss (no pool candidate < 25 km and no right area), right candidate never an area, wrong area chosen,
#   right; plus breaks / fixes vs the retrained reranker's top-1
#   slices: scene (MP16-Pro indoor / natural / urban), country, gallery density near the truth, the query's own geotag accuracy, read text,
#   the photographer's own photos in the pool (dev3k); tier / place text / continent (commons26 dev); the head's confidence
#   per slice: share, pipeline / top-1 / area-oracle / pool-oracle accuracy and each stage's loss in points of the whole set
import csv
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from geo_search_env.experiment.area_model import _paths  # noqa: E402
from geo_search_env.experiment.hypothesis_agent import SOURCES  # noqa: E402
from geo_search_env.experiment.pinpoint_reference import load_pools  # noqa: E402

APPLY = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
EMB = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
lines = []


def show(m=""):
    print(m, flush=True)
    lines.append(m)


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def stages(tag, key, pools_name):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text())
    n = len(photos)
    km = APPLY[f"{key}|RL policy, lambda 0"]
    conf = APPLY[f"{key}|RL policy, lambda 0|conf"]
    top1 = APPLY[f"{key}|retrained"]
    released = APPLY[f"{key}|released"]
    pools = load_pools(pools_name)
    truth = np.asarray([e["truth"] for e in photos])
    pool_ok = np.zeros(n, bool)
    for i, e in enumerate(photos):
        v = pools["valid"][e["index"]]
        ll = pools["latlon"][e["index"]][v]
        pool_ok[i] = (hav(truth[i, 0], truth[i, 1], ll[:, 0], ll[:, 1]) < 25).any()
    base_tag, ck = tag.rsplit("_", 2)[0], "_".join(tag.rsplit("_", 2)[1:])
    f = np.load(_paths(base_tag, ck)[0])
    keep = np.isin(f["source"], SOURCES)
    area_ok = np.zeros(n, bool)
    np.logical_or.at(area_ok, f["photo"][keep], f["km_agree"][keep] < 25)
    ok = km < 25
    bucket = np.where(ok, "right", np.where(area_ok, "wrong area chosen", np.where(pool_ok, "right candidate never an area", "retrieval miss")))
    return dict(photos=photos, n=n, km=km, conf=conf, top1=top1, released=released, pool_ok=pool_ok, area_ok=area_ok | ok, ok=ok, bucket=bucket)


def report_stages(name, s):
    n = s["n"]
    show(f"\n==== {name}: {n} photos; pipeline < 25 km {100 * s['ok'].mean():.1f} (< 1 km {100 * np.mean(s['km'] < 1):.1f}); retrained top-1 "
         f"{100 * np.mean(s['top1'] < 25):.1f}; released {100 * np.mean(s['released'] < 25):.1f}; area oracle {100 * s['area_ok'].mean():.1f}; "
         f"pool oracle {100 * s['pool_ok'].mean():.1f}")
    for b in ("right", "wrong area chosen", "right candidate never an area", "retrieval miss"):
        show(f"  {b:32s} {100 * np.mean(s['bucket'] == b):5.1f}%")
    t_ok = s["top1"] < 25
    show(f"  vs retrained top-1: fixes {np.sum(s['ok'] & ~t_ok)} ({100 * np.mean(s['ok'] & ~t_ok):.1f}%), breaks {np.sum(~s['ok'] & t_ok)} "
         f"({100 * np.mean(~s['ok'] & t_ok):.1f}%)")
    wrong = ~s["ok"]
    d = s["km"][wrong]
    show("  wrong answers by distance: " + ", ".join(f"{lo}-{hi} km {100 * np.mean((d >= lo) & (d < hi)) * wrong.mean():.1f}%"
                                                  for lo, hi in ((25, 100), (100, 750), (750, 2500), (2500, 1e9))) + " (of all photos)")
    right = s["ok"]
    show(f"  right at 25 km but > 1 km off: {100 * np.mean(right & (s['km'] >= 1)):.1f}% of photos")
    q = np.quantile(s["conf"], [0.2, 0.4, 0.6, 0.8])
    band = np.digitize(s["conf"], q)
    show("  confidence quintile (low -> high): pipeline right " + " ".join(f"{100 * s['ok'][band == b].mean():.1f}" for b in range(5)) +
         "; mean conf " + " ".join(f"{s['conf'][band == b].mean():.2f}" for b in range(5)) +
         "; area oracle " + " ".join(f"{100 * s['area_ok'][band == b].mean():.1f}" for b in range(5)))


def slice_table(name, s, slices):
    n = s["n"]
    show(f"\n  {name}: slice | share | pipeline | top-1 | area oracle | pool oracle | loss in pts of the set: wrong area, never an area, retrieval miss")
    for lab, m in slices:
        if m.sum() < 30:
            continue
        b = s["bucket"][m]
        show(f"    {lab:34s} {100 * m.mean():5.1f}% | {100 * s['ok'][m].mean():5.1f} | {100 * np.mean(s['top1'][m] < 25):5.1f} | "
             f"{100 * s['area_ok'][m].mean():5.1f} | {100 * s['pool_ok'][m].mean():5.1f} | "
             f"{100 * np.sum(b == 'wrong area chosen') / n:4.1f} {100 * np.sum(b == 'right candidate never an area') / n:4.1f} "
             f"{100 * np.sum(b == 'retrieval miss') / n:4.1f}")


# ---------- dev3k ----------
s = stages("dev3k_setft_own", "dev3k", "mp16_setft_own")
report_stages("dev3k", s)
photos = s["photos"]
row_index = np.fromfile(EMB / "row_index.i64.bin", dtype=np.int64)
csv_row = row_index[np.asarray([e["row"] for e in photos])]
need = set(csv_row.tolist())
meta, author_of = {}, {}
with open("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix = {k: hdr.index(k) for k in ("AUTHOR", "S3_Label", "Prob_indoor", "Prob_natural", "Prob_urban", "country", "continent")}
    auth_all = []
    for r, row in enumerate(rd):
        auth_all.append(row[ix["AUTHOR"]])
        if r in need:
            meta[r] = {k: row[i] for k, i in ix.items()}
author_cache = np.asarray(auth_all, dtype=object)[row_index]
M = [meta[r] for r in csv_row]
pi, pn, pu = (np.asarray([float(m[k] or 0) for m in M]) for k in ("Prob_indoor", "Prob_natural", "Prob_urban"))
scene = np.asarray(["indoor", "natural", "urban"])[np.argmax(np.stack([pi, pn, pu], 1), 1)]
country = np.asarray([m["country"] or "?" for m in M])  # MP16-Pro continent is empty
# gallery density near the truth (photos within 25 km)
from scipy.spatial import cKDTree  # noqa: E402

gl = np.fromfile(EMB / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)


def xyz(ll):
    la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


tree = cKDTree(xyz(gl))
truth = np.asarray([e["truth"] for e in photos])
dens = np.asarray(tree.query_ball_point(xyz(truth), 2 * np.sin(25 / 6371 / 2), return_length=True))
# the query's own geotag accuracy (YFCC)
z = np.load("data/yfcc100m/geo.npz", mmap_mode="r")
pid, acc = np.asarray(z["photo_id"]), np.asarray(z["accuracy"])
qid = np.asarray([int(Path(e["image_id"]).stem.split("_")[-1]) for e in photos])
o = np.argsort(pid)
p = np.searchsorted(pid[o], qid)
qacc = np.where(pid[o][np.clip(p, 0, len(o) - 1)] == qid, acc[o][np.clip(p, 0, len(o) - 1)], -1)
# read text (any read phrase) and the photographer's own photos in the pool
ev = collections.defaultdict(list)
for r in json.loads(Path("artifacts/agent/region_evidence_dev3k_setft_own_4b.json").read_text()):
    ev[r["image_id"]].append(r)
read = np.asarray([any(r["source"] in ("text 4b", "place name 4b") for r in ev[e["image_id"]]) for e in photos])
pools = load_pools("mp16_setft_own")
own = np.zeros(len(photos), bool)
for i, e in enumerate(photos):
    v = pools["valid"][e["index"]]
    own[i] = (author_cache[pools["cache_rows"][e["index"]][v]] == author_cache[e["row"]]).any()
slices = [("scene " + k, scene == k) for k in ("indoor", "natural", "urban")]
top_c = [k for k, _ in collections.Counter(country).most_common(8)]
slices += [("country " + k, country == k) for k in top_c] + [("country other", ~np.isin(country, top_c))]
slices += [(f"gallery photos within 25 km {lo}-{hi}", (dens >= lo) & (dens < hi)) for lo, hi in ((0, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9))]
slices += [("query geotag accuracy <= 10", (qacc >= 0) & (qacc <= 10)), ("query geotag accuracy 11-13", (qacc >= 11) & (qacc <= 13)),
           ("query geotag accuracy 14-16", qacc >= 14)]
slices += [("read text / names", read), ("no read text", ~read), ("photographer's own photo in pool", own), ("no own photo in pool", ~own)]
slice_table("dev3k", s, slices)

# ---------- commons26 dev ----------
s2 = stages("commons26dev_setft_own", "commons26dev:setft_own", "commons26_setft_own")
report_stages("commons26 dev", s2)
bm = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
B = [bm[e["image_id"]] for e in s2["photos"]]
tier = np.asarray([b["tier"] for b in B])
pt = np.asarray([b["place_text"] == "1" for b in B])
c2 = np.asarray([b["continent"] for b in B])
truth2 = np.asarray([e["truth"] for e in s2["photos"]])
dens2 = np.asarray(tree.query_ball_point(xyz(truth2), 2 * np.sin(25 / 6371 / 2), return_length=True))
sl2 = [("tier " + k, tier == k) for k in ("landmark", "city", "region", "none")]
sl2 += [("place text", pt), ("no place text", ~pt)]
sl2 += [("continent " + k, c2 == k) for k in sorted(set(c2))]
sl2 += [(f"gallery photos within 25 km {lo}-{hi}", (dens2 >= lo) & (dens2 < hi)) for lo, hi in ((0, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9))]
slice_table("commons26 dev", s2, sl2)
(OUT / "pipeline_diagnostics.txt").write_text("\n".join(lines) + "\n")
