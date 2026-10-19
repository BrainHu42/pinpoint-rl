# Lesson 129 drill-down: the photos the pipeline loses although retrieval found the place (dev3k, commons26 dev; cached outputs only).
#   "never an area" (a pool candidate < 25 km, no area < 25 km): best right candidate's rank by reranker score, the right candidates' share
#     of softmax(score / 0.2), and why: the photo already had 5 agreement areas (cut by the top 5) or fewer (folded into a cluster centred > 25 km away)
#   "wrong area chosen" (an area < 25 km offered, answer > 25 km): the right area's source and support rank, its support vs the top area's
#   both split by whether the photographer's own photo is in the pool (dev3k) and whether that own photo is itself right
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from geo_search_env.experiment.area_model import _paths  # noqa: E402
from geo_search_env.experiment.pinpoint_reference import load_pools  # noqa: E402

APPLY = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
EMB = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
T = 0.2  # agreement softmax temperature (lesson 68)
lines = []


def show(m=""):
    print(m, flush=True)
    lines.append(m)


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def q(x, ps=(0.25, 0.5, 0.75)):
    return " / ".join(f"{v:.3g}" for v in np.quantile(x, ps)) if len(x) else "-"


def drill(name, tag, base, key, pools_name, author=None, qcsv=None):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text())
    n = len(photos)
    km = APPLY[f"{key}|RL policy, lambda 0"]
    ok = km < 25
    pools = load_pools(pools_name)
    f = np.load(_paths(base, "setft_own")[0])
    # per photo: pool candidates sorted by score, right flags, softmax shares
    best_rank, right_share, top_share, own_any, own_right, pool_ok = (np.zeros(n) for _ in range(6))
    for i, e in enumerate(photos):
        v = pools["valid"][e["index"]]
        sc = pools["scores"][e["index"]][v].astype(np.float64)
        ll = pools["latlon"][e["index"]][v]
        o = np.argsort(-sc)
        sc, ll = sc[o], ll[o]
        w = np.exp((sc - sc.max()) / T)
        w /= w.sum()
        r = hav(e["truth"][0], e["truth"][1], ll[:, 0], ll[:, 1]) < 25
        pool_ok[i] = r.any()
        best_rank[i] = np.argmax(r) if r.any() else -1
        right_share[i] = w[r].sum()
        top_share[i] = w[0]
        if author is not None:
            mine = author[pools["csv_rows"][e["index"]][v][o]] == author[qcsv[i]]
            own_any[i] = mine.any()
            own_right[i] = (mine & r).any()
    pool_ok = pool_ok.astype(bool)
    own_any = own_any.astype(bool)
    a0 = f["source"] == 0
    n_agree = np.bincount(f["photo"][a0], minlength=n)
    srcs = np.isin(f["source"], (0, 1))
    area_ok = np.zeros(n, bool)
    np.logical_or.at(area_ok, f["photo"][srcs], f["km_agree"][srcs] < 25)
    never = pool_ok & ~area_ok & ~ok
    wrong = area_ok & ~ok
    groups = [("all", np.ones(n, bool))]
    if author is not None:
        groups += [("own photo in pool", own_any), ("no own photo in pool", ~own_any)]
    show(f"\n==== {name}: {n} photos, pipeline {100 * ok.mean():.1f}")
    if author is not None:
        show(f"  own photo in pool {100 * own_any.mean():.1f}%; own photo right (< 25 km) {100 * own_right.mean():.1f}% of photos; "
             f"pipeline right {100 * ok[own_any].mean():.1f} (own photo right {100 * ok[own_right.astype(bool)].mean():.1f}, "
             f"own photo wrong {100 * ok[own_any & ~own_right.astype(bool)].mean():.1f}), without {100 * ok[~own_any].mean():.1f}")
    show("  -- never an area: group | pts | best right candidate rank (quartiles) | right share of softmax (quartiles) | "
         "% with 5 agreement areas (cut by top 5) | % rank >= 10")
    for g, m in groups:
        mm = never & m
        show(f"    {g:22s} {100 * mm.sum() / n:4.1f} | {q(best_rank[mm])} | {q(right_share[mm])} | {100 * np.mean(n_agree[mm] >= 5):.0f}% | "
             f"{100 * np.mean(best_rank[mm] >= 10):.0f}%")
    show(f"    (reference, photos with a right area: right share {q(right_share[area_ok])}; top candidate share {q(top_share[area_ok])})")
    # wrong area: where is the right area
    rows = []
    for i in np.flatnonzero(wrong):
        s = srcs & (f["photo"] == i)
        sup, src, rk, ka = f["support"][s], f["source"][s], f["rank"][s], f["km_agree"][s]
        right = ka < 25
        j = np.flatnonzero(right)[np.argmax(sup[right])]
        rows.append((i, src[j], rk[j], sup[j], sup[src == 0].max(), right.sum(), s.sum()))
    rows = np.asarray(rows)
    show("  -- wrong area chosen: group | pts | right area from clue | right area support rank 0 / 1 / 2 / 3+ (agreement areas) | "
         "right support (quartiles) | top support (quartiles) | answer error")
    for g, m in groups:
        mm = m[rows[:, 0].astype(int)]
        r = rows[mm]
        ag = r[r[:, 1] == 0]
        show(f"    {g:22s} {100 * mm.sum() / n:4.1f} | {100 * np.mean(r[:, 1] == 1):.0f}% | "
             + " / ".join(f"{100 * np.mean(ag[:, 2] == k):.0f}" for k in (0, 1, 2)) + f" / {100 * np.mean(ag[:, 2] >= 3):.0f}% | "
             f"{q(ag[:, 3])} | {q(ag[:, 4])} | answer km {q(km[r[:, 0].astype(int)])}, < 100 km {100 * np.mean(km[r[:, 0].astype(int)] < 100):.0f}%")
    # how the pick goes per support rank of the right area, all photos with a right agreement area
    show("  -- photos with a right agreement area: right area's support rank | share of photos | pipeline right | top-support area right")
    best_rk = np.full(n, -1)
    for i in np.flatnonzero(area_ok):
        s = a0 & (f["photo"] == i) & (f["km_agree"] < 25)
        if s.any():
            best_rk[i] = f["rank"][s].min()
    for g, m in groups:
        for k in (0, 1, 2, 3):
            mm = m & ((best_rk == k) if k < 3 else (best_rk >= 3))
            show(f"    {g:22s} rank {k}{'+' if k == 3 else ' '} | {100 * mm.mean():4.1f}% | {100 * ok[mm].mean():5.1f} | {100 * (k == 0):.0f}")
    return dict(never=never, wrong=wrong, best_rank=best_rank, right_share=right_share)


photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_own/photos.json").read_text())
with open("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    ia = next(rd).index("AUTHOR")
    author = np.asarray([row[ia] for row in rd], dtype=object)
qcsv = np.fromfile(EMB / "row_index.i64.bin", dtype=np.int64)[np.asarray([e["row"] for e in photos])]
drill("dev3k", "dev3k_setft_own", "dev3k", "dev3k", "mp16_setft_own", author, qcsv)
drill("commons26 dev", "commons26dev_setft_own", "commons26dev", "commons26dev:setft_own", "commons26_setft_own")
if len(sys.argv) > 1:
    (Path(sys.argv[1]) / "candidate_drill.txt").write_text("\n".join(lines) + "\n")
