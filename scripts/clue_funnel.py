# Clue pathway funnel on dev sets (dev3k_setft_own, commons26_setft_own dev split only): where do right clue places get lost?
import collections
import json
import sys
from pathlib import Path

import numpy as np

A = Path("artifacts/agent")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
lines = []


def show(m=""):
    print(m, flush=True)
    lines.append(m)


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


for tag, split in (("dev3k_setft_own", None), ("commons26_setft_own", "dev")):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text())
    keep = {e["image_id"]: i for i, e in enumerate(photos) if split is None or e.get("split") == split}
    n = len(keep)
    ev = [r for r in json.loads((A / f"region_evidence_{tag}_4b.json").read_text()) if r["image_id"] in keep]
    judge = {(r["image_id"], r["query"]): r["p_yes"] for r in json.loads((A / f"clue_judge_{tag}_4b_4b.json").read_text()) if r["image_id"] in keep}
    # Pinpoint areas and their restricted top-1 (pipeline hypotheses)
    h = np.load(A / f"hypotheses_{tag}.npz")
    cov = np.zeros(len(photos), bool)
    centres = collections.defaultdict(list)
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    for j in np.flatnonzero(h["source"] == 0):
        p = int(h["photo"][j])
        centres[p].append(h["center"][j])
        sc = np.where(h["rows"][j] >= 0, h["scores"][j], -np.inf)
        if np.isfinite(sc).any():
            ll = h["latlon"][j][int(np.argmax(sc))]
            cov[p] |= hav(truth[p, 0], truth[p, 1], ll[0], ll[1]) < 25
    ch = np.load(A / f"hypotheses_{tag}_cluehyps.npz")
    prop_ok = collections.defaultdict(bool)
    prop_any = set()
    for j in range(len(ch["photo"])):
        p = int(ch["photo"][j])
        prop_any.add(p)
        sc = np.where(ch["rows"][j] >= 0, ch["scores"][j], -np.inf)
        c = ch["center"][j]
        prop_ok[p] |= hav(truth[p, 0], truth[p, 1], c[0], c[1]) < 25
    ids = list(keep)
    pos = np.asarray([keep[i] for i in ids])
    show(f"== {tag}{' ' + split if split else ''}: {n} photos; Pinpoint-area restricted oracle {100 * cov[pos].mean():.1f}%")
    by = collections.defaultdict(list)
    for r in ev:
        by[r["image_id"]].append(r)
    src = collections.Counter(r["source"] for r in ev)
    show(f"  evidence rows {len(ev)}; by source {dict(src)}")
    show(f"  rows with a search result {sum('region' in r for r in ev)}; judged {len(judge)}")

    def frac(pred):
        return sum(pred(i) for i in ids)

    def right(r):
        return "region" in r and r.get("km", 1e9) < 25

    read = lambda r: not r["source"].startswith("query")
    stages = [
        ("any phrase produced", lambda i: len(by[i]) > 0),
        ("  some phrase has a search result", lambda i: any("region" in r for r in by[i])),
        ("  some result < 25 km (any source)", lambda i: any(right(r) for r in by[i])),
        ("  some result < 25 km, read text (not 4B queries)", lambda i: any(right(r) and read(r) for r in by[i])),
        ("  ... and share >= 0.3", lambda i: any(right(r) and read(r) and r["share"] >= 0.3 for r in by[i])),
        ("  ... and judged", lambda i: any(right(r) and read(r) and r["share"] >= 0.3 and (i, r["query"]) in judge for r in by[i])),
        ("  ... and judge >= 0.5", lambda i: any(right(r) and read(r) and r["share"] >= 0.3 and judge.get((i, r["query"]), 0) >= 0.5 for r in by[i])),
        ("  4B queries: some result < 25 km, share >= 0.3", lambda i: any(right(r) and not read(r) and r["share"] >= 0.3 for r in by[i])),
    ]
    show("  photos (% of set) | of them, not covered by any Pinpoint area")
    for name, pred in stages:
        a = [i for i in ids if pred(i)]
        unc = [i for i in a if not cov[keep[i]]]
        show(f"  {name:55s} {100 * len(a) / n:5.1f}% ({len(a):4d}) | uncovered {100 * len(unc) / n:5.1f}% ({len(unc)})")
    show(f"  clue-place areas proposed: {100 * len([i for i in ids if keep[i] in prop_any]) / n:.1f}% of photos; with a centre < 25 km: "
         f"{100 * sum(prop_ok[keep[i]] for i in ids) / n:.1f}%; of those, uncovered by Pinpoint areas {100 * sum(prop_ok[keep[i]] and not cov[keep[i]] for i in ids) / n:.1f}%")
    # precision of read-text results by judge band, among those > 25 km from all Pinpoint areas (proposal candidates)
    show("  read-text results > 25 km from every Pinpoint area (would-be proposals), by judge P(yes): n, % < 25 km of truth")
    rows = []
    for i in ids:
        cs = np.asarray(centres[keep[i]], np.float64).reshape(-1, 2)
        for r in by[i]:
            if "region" not in r or not read(r) or r["share"] < 0.3:
                continue
            far = (not len(cs)) or (hav(r["lat"], r["lon"], cs[:, 0], cs[:, 1]) >= 25).all()
            rows.append((judge.get((i, r["query"])), far, r["km"] < 25, r["source"], r["share"], r["matches"], i))
    for lo, hi in ((None, None), (0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)):
        sel = [x for x in rows if x[1] and (x[0] is None if lo is None else (x[0] is not None and lo <= x[0] < hi))]
        if sel:
            show(f"    judge {'unjudged' if lo is None else f'[{lo}, {hi})':12s} n {len(sel):5d}  right {100 * np.mean([x[2] for x in sel]):5.1f}%  "
                 f"photos with a right one {len({x[6] for x in sel if x[2]})}")
    show("  same, by source (all judge values):")
    for s in sorted({x[3] for x in rows}):
        sel = [x for x in rows if x[1] and x[3] == s]
        if sel:
            show(f"    {s:20s} n {len(sel):5d} right {100 * np.mean([x[2] for x in sel]):5.1f}%")
    show("  4B-query results > 25 km from every Pinpoint area, share >= 0.3, by judge P(yes):")
    qrows = []
    for i in ids:
        cs = np.asarray(centres[keep[i]], np.float64).reshape(-1, 2)
        for r in by[i]:
            if "region" not in r or read(r) or r["share"] < 0.3:
                continue
            far = (not len(cs)) or (hav(r["lat"], r["lon"], cs[:, 0], cs[:, 1]) >= 25).all()
            if far:
                qrows.append((judge.get((i, r["query"])), r["km"] < 25, i))
    for lo, hi in ((None, None), (0, 0.5), (0.5, 1.01)):
        sel = [x for x in qrows if (x[0] is None if lo is None else (x[0] is not None and lo <= x[0] < hi))]
        if sel:
            show(f"    judge {'unjudged' if lo is None else f'[{lo}, {hi})':12s} n {len(sel):5d}  right {100 * np.mean([x[1] for x in sel]):5.1f}%  "
                 f"photos with a right one {len({x[2] for x in sel if x[1]})}")
    show()
(OUT / "clue_funnel.txt").write_text("\n".join(lines) + "\n")
