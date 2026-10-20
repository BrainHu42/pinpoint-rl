"""Error triage for the pipeline viewer: every photo the agent gets wrong (> 25 km), classified by how close the pipeline came to a right
answer, most fixable first. Read from the traces of viz.traces; one JSON per set (artifacts/viz/traces/{set}/errors.json).

Tiers (a photo gets the first that applies; "right" = within 25 km of the truth):
  1 policy     - another tool choice already answers right: the other λ policy, or a fixed tool set (glance / read / recognise / zoom all /
                 all tools) whose answer head picks a right area. The fix is in the policy, not the evidence.
  2 coin flip  - a right area is on the table at the final step and the answer head gave it >= half the chosen area's probability.
  3 near miss  - the answer is 25-50 km off: right region, wrong point (refinement).
  4 ranked low - a right area is on the table, with < half the chosen area's probability.
  5 in pool    - a retrieval candidate (retrained pool, released 12, or a zoomed search) lies within 25 km, but no area answers there.
  6 miss       - no candidate within 25 km anywhere: retrieval has to find new candidates.
Within a tier: closer first (probability ratio, km, or candidate rank, per tier). `score` orders all errors: tier, then closeness.

  PYTHONPATH=src .venv/bin/python -m geo_search_env.viz.errors [--sets full dev3k]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .traces import TRACES

RIGHT_KM = 25.0
NEAR_KM = 50.0
TIERS = {1: "policy", 2: "coin flip", 3: "near miss", 4: "ranked low", 5: "in pool", 6: "miss"}
WITH_READ = {"glance": False, "read": True, "recognise": False, "zoom all": False, "all tools": True}  # fixed tool sets that reveal clue areas
ZOOMED = {"glance": False, "read": False, "recognise": False, "zoom all": True, "all tools": True}


def _names(areas: list[dict]) -> list[str]:
    """Area labels as the viewer shows them: A1.. for Pinpoint areas (by rank), C1.. for clue places."""

    c, out = 0, []
    for a in areas:
        if a["source"] == "pinpoint":
            out.append(f"A{a['rank'] + 1}")
        else:
            c += 1
            out.append(f"C{c}")
    return out


def _fmt(km: float) -> str:
    return f"{km:.0f} km" if km >= 10 else f"{km:.1f} km"


def classify(t: dict) -> dict | None:
    """The error row of one trace, or None if the agent's answer (λ 0) is right or there is no truth."""

    fin = t["agent"]["0"]["final"]
    if not t.get("truth") or fin["km"] is None or fin["km"] < RIGHT_KM:
        return None
    areas, names = t["areas"], _names(t["areas"])
    last = t["agent"]["0"]["steps"][-1]
    zoomed, ans = last["zoomed"], last["answer"]
    km_at = lambda s, z: areas[s]["zoomed"]["km"] if z else areas[s]["unzoomed"]["km"]
    chosen = fin["slot"]
    p_chosen = ans[chosen] or 1e-9
    # right areas on the table at the final step (clue areas only once read)
    on_table = [s for s in range(len(areas)) if ans[s] is not None and (last["read"] or areas[s]["source"] != "clue")]
    right = [s for s in on_table if km_at(s, zoomed[s]) is not None and km_at(s, zoomed[s]) < RIGHT_KM]
    best_right = max(right, key=lambda s: ans[s]) if right else None
    ratio = (ans[best_right] / p_chosen) if best_right is not None else 0.0
    rank_right = (sorted(on_table, key=lambda s: -ans[s]).index(best_right) + 1) if best_right is not None else None

    # other tool choices that answer right
    alt = []
    for lam, ag in t["agent"].items():
        if lam != "0" and ag["final"]["km"] is not None and ag["final"]["km"] < RIGHT_KM:
            alt.append(f"λ {lam} policy ({_fmt(ag['final']['km'])})")
    for name, ps in t.get("fixed", {}).items():
        ok = [s for s in range(len(areas)) if ps[s] is not None and (WITH_READ.get(name, True) or areas[s]["source"] != "clue")]
        if ok:
            s = max(ok, key=lambda s: ps[s])
            km = km_at(s, ZOOMED.get(name, False))
            if km is not None and km < RIGHT_KM:
                alt.append(f"{name} ({names[s]}, {_fmt(km)})")
    other_point = km_at(chosen, not zoomed[chosen])  # the chosen area's other answer point (zoomed vs not)
    flags = []
    rel_km = (t["released"]["candidates"] or [{}])[0].get("km")
    ret_km = (t["retrained"]["candidates"] or [{}])[0].get("km")
    if rel_km is not None and rel_km < RIGHT_KM:
        flags.append(f"regression: the released Pinpoint's top-1 was right ({_fmt(rel_km)})")
    if ret_km is not None and ret_km < RIGHT_KM:
        flags.append(f"regression: the retrained reranker's top-1 was right ({_fmt(ret_km)})")
    if other_point is not None and other_point < RIGHT_KM:
        flags.append(f"{names[chosen]} {'unzoomed' if zoomed[chosen] else 'zoomed'} point is right ({_fmt(other_point)})")
    hidden = [s for s in range(len(areas)) if s not in on_table and min(x for x in (areas[s]["unzoomed"]["km"], areas[s]["zoomed"]["km"], 1e9)
                                                                       if x is not None) < RIGHT_KM]
    if hidden:
        flags.append(f"right clue area {', '.join(names[s] for s in hidden)} not revealed (no read)")

    # candidates within 25 km anywhere
    pool = t["retrained"]["candidates"]
    pool_right = [k for k, c in enumerate(pool) if c.get("km") is not None and c["km"] < RIGHT_KM]
    rel_right = [k for k, c in enumerate(t["released"]["candidates"]) if c.get("km") is not None and c["km"] < RIGHT_KM]
    zoom_right = [(s, k) for s, a in enumerate(areas) for k, c in enumerate(a["candidates"]) if c.get("km") is not None and c["km"] < RIGHT_KM]
    all_km = [c["km"] for c in pool + t["released"]["candidates"] + [c for a in areas for c in a["candidates"]] if c.get("km") is not None]
    nearest = min(all_km) if all_km else None

    if alt:
        tier, close, why = 1, -ratio, "Another tool choice answers right: " + "; ".join(alt[:3])
    elif best_right is not None and ratio >= 0.5:
        tier, close = 2, -ratio
        why = (f"{names[best_right]} ({areas[best_right].get('place', '')}) is right, answer head {ans[best_right]:.0%} vs {ans[chosen]:.0%} "
               f"for {names[chosen]}")
    elif fin["km"] < NEAR_KM:
        tier, close, why = 3, fin["km"], f"Answer {_fmt(fin['km'])} off: right region, wrong point"
    elif best_right is not None:
        tier, close = 4, -ratio
        why = (f"{names[best_right]} ({areas[best_right].get('place', '')}) is right but ranked #{rank_right}: {ans[best_right]:.0%} vs "
               f"{ans[chosen]:.0%} for {names[chosen]}")
    elif pool_right or rel_right or zoom_right:
        tier = 5
        close = pool_right[0] if pool_right else 100 + (rel_right[0] if rel_right else 50)
        bits = []
        if pool_right:
            c = pool[pool_right[0]]
            bits.append(f"pool #{pool_right[0] + 1} ({c.get('place', '')}, {_fmt(c['km'])}, set prob. {c.get('p', 0):.0%})")
        if rel_right:
            bits.append(f"released #{rel_right[0] + 1}")
        if zoom_right:
            bits.append(f"zoomed search of {names[zoom_right[0][0]]}")
        why = "A right candidate exists but no area answers there: " + ", ".join(bits)
    else:
        tier, close = 6, nearest if nearest is not None else 1e9
        why = f"No candidate within 25 km; nearest {_fmt(nearest)}" if nearest is not None else "No candidates"
    return {"i": t["i"], "image_id": t["image_id"], "benchmark": t["benchmark"], "truth_place": t["truth"].get("place", ""),
            "final_km": fin["km"], "final_place": fin.get("place", ""), "released_km": rel_km, "retrained_km": ret_km,
            "regression": bool((rel_km is not None and rel_km < RIGHT_KM) or (ret_km is not None and ret_km < RIGHT_KM)),
            "tier": tier, "tier_name": TIERS[tier], "close": round(float(close), 4), "why": why, "flags": flags,
            "right_area": names[best_right] if best_right is not None else None, "right_slot": best_right, "chosen": names[chosen],
            "p_right": ans[best_right] if best_right is not None else None, "p_chosen": ans[chosen], "rank_right": rank_right,
            "nearest_km": nearest, "pool_right_rank": pool_right[0] + 1 if pool_right else None}


def build(tag: str) -> list[dict]:
    root = TRACES / tag
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    rows = []
    for r in index:
        if r["final_km"] is not None and r["final_km"] < RIGHT_KM:
            continue
        row = classify(json.loads((root / f"{r['i']}.json").read_text(encoding="utf-8")))
        if row:
            rows.append(row)
    rows.sort(key=lambda x: (x["tier"], x["close"]))
    for k, x in enumerate(rows):
        x["score"] = k  # triage order
    out = {"set": tag, "n_photos": len(index), "n_errors": len(rows), "tiers": TIERS, "errors": rows}
    (root / "errors.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    counts = {TIERS[k]: sum(x["tier"] == k for x in rows) for k in TIERS}
    print(f"{tag}: {len(rows)} errors of {len(index)} photos -> {root / 'errors.json'}; {counts}; "
          f"{sum(x['regression'] for x in rows)} regressions (a reranker's top-1 was right)", flush=True)
    return rows


def load(tag: str) -> dict:
    """errors.json of a set, built on first use (or when its traces are newer)."""

    path, index = TRACES / tag / "errors.json", TRACES / tag / "index.json"
    if not path.exists() or path.stat().st_mtime < index.stat().st_mtime:
        build(tag)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["full", "dev3k"])
    args = parser.parse_args(argv)
    for tag in args.sets:
        if (TRACES / tag / "index.json").exists():
            build(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
