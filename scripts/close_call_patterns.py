"""Patterns in the pipeline's close-call errors (viz.errors tiers), dev3k first and the test traces (`full`) as a check: where they happen,
how the right area differs from the chosen one, how confident the answer head was, which tool choice would have fixed them, and what the
clue reading saw. One row per photo from the viewer's traces; report -> artifacts/viz/close_call_patterns.txt.

  PYTHONPATH=src .venv/bin/python scripts/close_call_patterns.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

TRACES = Path("artifacts/viz/traces")
OUT = Path("artifacts/viz/close_call_patterns.txt")
SETS = {"dev3k": "dev3k", "full": "test"}
GROUPS = ("right", "close", "ranked low", "in pool", "miss")  # close = tiers 1-3 (policy, coin flip, near miss)
GROUP_OF = {0: "right", 1: "close", 2: "close", 3: "close", 4: "ranked low", 5: "in pool", 6: "miss"}

continent = {}
for line in Path("data/pinpoint/geonames/countryInfo.txt").read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#"):
        p = line.split("\t")
        continent[p[4]] = p[8]
CONT = {"EU": "Europe", "AS": "Asia", "NA": "N. America", "SA": "S. America", "AF": "Africa", "OC": "Oceania", "AN": "Antarctica"}


NATURE = {"forest", "mountain", "hill", "river", "lake", "cliff", "beach", "bay", "valley", "cave", "desert", "island", "waterfall", "garden", "park"}


def scene(features: list[str]) -> str:
    if not features:
        return "none"
    return "nature" if sum(f in NATURE for f in features) * 2 >= len(features) else "landmark"


def country(place: str | None) -> str:
    return (place or "").rsplit(", ", 1)[-1].strip()


def km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, h)))


def rows(tag: str) -> list[dict]:
    errs = {e["i"]: e for e in json.loads((TRACES / tag / "errors.json").read_text())["errors"]}
    out = []
    for r in json.loads((TRACES / tag / "index.json").read_text()):
        t = json.loads((TRACES / tag / f"{r['i']}.json").read_text())
        e = errs.get(r["i"])
        tier = e["tier"] if e else 0
        fin = t["agent"]["0"]["final"]
        last = t["agent"]["0"]["steps"][-1]
        areas = t["areas"]
        ans = [x if x is not None else 0.0 for x in last["answer"]]
        top = sorted(ans, reverse=True) + [0.0, 0.0]
        chosen = areas[fin["slot"]] if areas else None
        rs = e["right_slot"] if e else None
        right = areas[rs] if rs is not None else None
        o = (t.get("clues") or {}).get("observe") or {}
        tc = country(t["truth"]["place"])
        row = {"set": tag, "i": r["i"], "bench": t["benchmark"], "tier": tier, "group": GROUP_OF[tier], "country": tc,
               "continent": CONT.get(continent.get(tc, ""), "?"), "final_km": fin["km"], "p_chosen": ans[fin["slot"]] if areas else None,
               "margin": top[0] - top[1], "n_areas": len(areas), "chosen_country": country(fin.get("place")),
               "n_zoom": sum(1 for s in t["agent"]["0"]["steps"][:-1] if s["action"] == "zoom"), "read": last["read"], "recognise": last["recognise"],
               "has_text": bool(o.get("text")), "n_names": len(o.get("place_names") or []),
               "guess_ok": bool(o.get("country")) and tc.lower() in str(o.get("country")).lower() if tc else False,
               "guess_any": bool(o.get("country")), "features": [str(f).lower() for f in (o.get("features") or [])],
               "regression": bool(e and e["regression"]), "why": e["why"] if e else "", "chosen_rank": chosen["rank"] if chosen else None,
               "chosen_source": chosen["source"] if chosen else None}
        g = (t.get("fixed") or {}).get("glance")  # the answer head with no tools: its pick among the Pinpoint areas, unzoomed
        ok = [s for s in range(len(areas)) if g and g[s] is not None and areas[s]["source"] != "clue"]
        gkm = areas[max(ok, key=lambda s: g[s])]["unzoomed"]["km"] if ok else None
        row["glance_ok"] = gkm is not None and gkm < 25 if ok else None
        row["p_glance"] = max(g[s] for s in ok) if ok else None
        for name, a in (("c", chosen), ("r", right)):
            if a is None:
                continue
            row[f"{name}_density"] = a["pre"].get("density")
            row[f"{name}_support"] = a["pre"].get("support")
            row[f"{name}_mass"] = a["pre"].get("set_mass")
            row[f"{name}_ml"] = a["megaloc"].get("best")
            row[f"{name}_rank"] = a["rank"]
            row[f"{name}_source"] = a["source"]
            row[f"{name}_photographers"] = a["post"].get("photographers")
        if right is not None and chosen is not None:
            row["right_chosen_km"] = km(right["center"], chosen["center"])
            row["p_right"] = ans[rs]
        out.append(row)
    return out


def q(xs, ps=(25, 50, 75)) -> str:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return " / ".join(f"{np.percentile(xs, p):.3g}" for p in ps) if xs else "-"


def share(xs) -> str:
    xs = list(xs)
    return f"{100 * np.mean(xs):.0f}%" if xs else "-"


def main() -> None:
    data = {tag: rows(tag) for tag in SETS}
    L = []
    p = L.append
    p("Close-call patterns (viz.errors tiers; close = tiers 1-3: policy close call, coin flip, near miss). dev3k | test (im2gps3k + yfcc4k eval halves)\n")

    p("1. Group sizes")
    for tag, rs in data.items():
        c = Counter(r["group"] for r in rs)
        p(f"  {SETS[tag]:5s} n={len(rs)}: " + ", ".join(f"{g} {100 * c[g] / len(rs):.1f}%" for g in GROUPS))
    for b in ("im2gps3k", "yfcc4k"):
        rs = [r for r in data["full"] if r["bench"] == b]
        c = Counter(r["group"] for r in rs)
        p(f"  test {b:8s} n={len(rs)}: " + ", ".join(f"{g} {100 * c[g] / len(rs):.1f}%" for g in GROUPS))

    p("\n2. By continent: share of photos in each group (continents with >= 60 photos)")
    for tag, rs in data.items():
        p(f"  {SETS[tag]}")
        by = Counter(r["continent"] for r in rs)
        for cont, n in by.most_common():
            if n < 60:
                continue
            c = Counter(r["group"] for r in rs if r["continent"] == cont)
            p(f"    {cont:11s} n={n:5d}: " + "  ".join(f"{g} {100 * c[g] / n:4.1f}%" for g in GROUPS))

    p("\n3. Countries with the most close calls (close-call rate vs the set's)")
    for tag, rs in data.items():
        base = np.mean([r["group"] == "close" for r in rs])
        by = Counter(r["country"] for r in rs if r["group"] == "close")
        tot = Counter(r["country"] for r in rs)
        p(f"  {SETS[tag]} (set rate {100 * base:.1f}%): " + ", ".join(f"{c} {n}/{tot[c]} ({100 * n / tot[c]:.0f}%)" for c, n in by.most_common(8)))

    p("\n4. Geometry of the wrong answer")
    for tag, rs in data.items():
        for g in GROUPS[1:]:
            sub = [r for r in rs if r["group"] == g]
            same = [r["chosen_country"] == r["country"] for r in sub if r["chosen_country"]]
            p(f"  {SETS[tag]:5s} {g:10s}: answer km p25/50/75 {q([r['final_km'] for r in sub])}; same country as the truth {share(same)}")
        sub = [r for r in rs if "right_chosen_km" in r and r["tier"] in (1, 2, 4)]
        p(f"  {SETS[tag]:5s} right area <-> chosen area (tiers 1, 2, 4 with a right area), km p25/50/75: {q([r['right_chosen_km'] for r in sub])}")

    p("\n5. Right area vs the chosen area (errors with a right area on the table: tiers 1, 2, 4)")
    for tag, rs in data.items():
        sub = [r for r in rs if "r_density" in r and r["tier"] in (1, 2, 4)]
        p(f"  {SETS[tag]} (n={len(sub)})")
        for k, label, better in (("density", "gallery photos in its box", "fewer"), ("support", "pool support", "less"),
                                 ("mass", "set-layer mass", "less"), ("ml", "MegaLoc best match", "lower"), ("photographers", "photographers in zoomed search", "fewer")):
            pairs = [(r[f"r_{k}"], r[f"c_{k}"]) for r in sub if r.get(f"r_{k}") is not None and r.get(f"c_{k}") is not None]
            if not pairs:
                continue
            lo = np.mean([a < b for a, b in pairs])
            med_r, med_c = np.median([a for a, _ in pairs]), np.median([b for _, b in pairs])
            p(f"    {label:30s}: right area has {better} on {100 * lo:.0f}% (median right {med_r:.3g} vs chosen {med_c:.3g})")
        p(f"    right area's rank (A1 = most support): " + ", ".join(f"A{k + 1} {v}" for k, v in sorted(Counter(r['r_rank'] for r in sub if r['r_source'] == 'pinpoint').items())[:6])
          + f"; clue areas {sum(r['r_source'] == 'clue' for r in sub)}; chosen is A1 on {share(r['c_rank'] == 0 and r['c_source'] == 'pinpoint' for r in sub)}")

    p("\n6. Answer-head confidence at the final step: chosen-area probability and top-2 margin, p25/50/75")
    for tag, rs in data.items():
        for g in GROUPS:
            sub = [r for r in rs if r["group"] == g and r["p_chosen"] is not None]
            p(f"  {SETS[tag]:5s} {g:10s}: p(chosen) {q([r['p_chosen'] for r in sub])}; margin {q([r['margin'] for r in sub])}; "
              f"margin < 0.05 on {share(r['margin'] < 0.05 for r in sub)}; areas {q([r['n_areas'] for r in sub])}")

    p("\n7. Policy close calls (tier 1): which other tool choice answers right (a photo can have several)")
    for tag, rs in data.items():
        sub = [r for r in rs if r["tier"] == 1]
        c = Counter()
        for r in sub:
            for alt in r["why"].split(": ", 1)[-1].split("; "):
                c[re.sub(r" \(.*", "", alt)] += 1
        p(f"  {SETS[tag]} (n={len(sub)}): " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        for g in ("right", "close"):
            s2 = [r for r in rs if (r["tier"] == 1 if g == "close" else r["group"] == "right")]
            p(f"    {'tier 1' if g == 'close' else 'right ':6s}: zooms {q([r['n_zoom'] for r in s2])}, read {share(r['read'] for r in s2)}, "
              f"recognise {share(r['recognise'] for r in s2)}, retrained or released top-1 right {share(r['regression'] for r in s2)}")

    p("\n8. Clue reading by group: visible text, place names read, the 4B's country guess right")
    for tag, rs in data.items():
        for g in GROUPS:
            sub = [r for r in rs if r["group"] == g]
            p(f"  {SETS[tag]:5s} {g:10s}: text {share(r['has_text'] for r in sub)}, names {share(r['n_names'] > 0 for r in sub)}, "
              f"country guess right {share(r['guess_ok'] for r in sub)} (of guesses {share(r['guess_ok'] for r in sub if r['guess_any'])})")

    p("\n9. Scene kind from the 4B's feature words (nature: forest, mountain, hill, river, lake, ...; landmark: church, tower, monument, ...)")
    for tag, rs in data.items():
        for g in GROUPS:
            sub = [r for r in rs if r["group"] == g]
            kinds = Counter(scene(r["features"]) for r in sub)
            p(f"  {SETS[tag]:5s} {g:10s}: " + ", ".join(f"{k} {100 * kinds[k] / len(sub):.0f}%" for k in ("nature", "landmark", "none")))
        for k in ("nature", "landmark", "none"):
            sub = [r for r in rs if scene(r["features"]) == k]
            c = Counter(r["group"] for r in sub)
            p(f"    {k:8s} photos n={len(sub):4d}: " + "  ".join(f"{g} {100 * c[g] / len(sub):4.1f}%" for g in GROUPS))

    p("\n10. Answer-head confidence bins: what the photos in each bin turn out to be (all photos)")
    for tag, rs in data.items():
        p(f"  {SETS[tag]}")
        for lo, hi in ((0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)):
            sub = [r for r in rs if r["p_chosen"] is not None and lo <= r["p_chosen"] < hi]
            c = Counter(r["group"] for r in sub)
            p(f"    p(chosen) {lo:.2f}-{min(hi, 1):.2f} n={len(sub):5d}: " + "  ".join(f"{g} {100 * c[g] / max(1, len(sub)):4.1f}%" for g in GROUPS))

    p("\n11. Tools vs a glance (the answer head with no tools, fixed 'glance'), all photos")
    for tag, rs in data.items():
        fixed = sum(1 for r in rs if r["glance_ok"] is False and r["group"] == "right")
        broke = sum(1 for r in rs if r["glance_ok"] and r["group"] != "right")
        p(f"  {SETS[tag]}: the agent fixes {fixed} photos a glance gets wrong and breaks {broke} a glance gets right (net {fixed - broke:+d}, "
          f"{100 * (fixed - broke) / len(rs):+.1f} pts); broken photos by tier: "
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['tier'] for r in rs if r['glance_ok'] and r['group'] != 'right').items())))
        broke = [r["p_glance"] for r in rs if r["glance_ok"] and r["group"] != "right"]
        fixed = [r["p_glance"] for r in rs if r["glance_ok"] is False and r["group"] == "right"]
        kept = [r["p_glance"] for r in rs if r["glance_ok"] and r["group"] == "right"]
        p(f"    glance confidence p25/50/75: broken {q(broke)}; fixed {q(fixed)}; right both ways {q(kept)}")
        for th in (0.5, 0.6, 0.7):
            p(f"    'keep the glance when its p >= {th}': recovers {sum(x >= th for x in broke)} broken, loses {sum(x >= th for x in fixed)} fixed")

    text = "\n".join(L)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
