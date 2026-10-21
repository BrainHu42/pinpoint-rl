"""Why place names the 4B reads don't put the pipeline near the truth (VALUE_PLAN.md scope check; lesson 110: they geocode < 25 km on 0.5% of
dev3k misses). For every photo the pipeline gets wrong (> 25 km) with a read place name, each name is classed by the offline maps search
(tools.maps: OSM + GeoNames + Overture names):
  right    the name (phrase match) exists within 25 km of the truth: read right, lost to ambiguity / ranking;
  misread  not within 25 km, but an OSM name within 5 km of the truth is a close spelling (rapidfuzz ratio >= 80 on normalized text);
  elsewhere the name exists only away from the truth: a brand, chain, event, a sign naming another place, or a misread that hits a real name;
  nowhere  no match at all (invented, garbled, or missing from the gazetteers).
A photo takes its best name's class (right > misread > elsewhere > nowhere).

  PYTHONPATH=src:<dir with rapidfuzz> .venv/bin/python scripts/name_miss_probe.py [--tags dev3k_setft_own commons26dev_setft_own]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

from geo_search_env.experiment.text_search import haversine_km, norm
from geo_search_env.tools.maps import MapsSearch

KM = {"dev3k_setft_own": "dev3k|RL policy, lambda 0", "commons26dev_setft_own": "commons26dev:setft_own|RL policy, lambda 0"}
OBS = {"dev3k_setft_own": "observe_dev3k_setft_own_4b.json", "commons26dev_setft_own": "observe_commons26dev_setft_own_4b.json"}
ORDER = ("right", "misread", "elsewhere", "nowhere")
OUT = Path("artifacts/value")


def near_names(ms: MapsSearch, lat: float, lon: float, km: float = 5.0) -> list[str]:
    dlat, dlon = km / 111.0, km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
    rows = ms.osm.execute("SELECT p.variants FROM boxes b JOIN places p ON p.rowid = b.id "
                          "WHERE b.lat_max >= ? AND b.lat_min <= ? AND b.lon_max >= ? AND b.lon_min <= ?",
                          (lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchall()
    return sorted({norm(v) for (vs,) in rows if vs for v in vs.split(" | ") if len(norm(v)) >= 3})


def classify(ms: MapsSearch, name: str, truth, names_near: list[str]) -> dict:
    from rapidfuzz import fuzz, process

    lat, lon = truth
    out = {"name": name}
    glob = ms.query(name, top=200)
    out["matches"] = len(glob)
    out["countries"] = len({p.label.rsplit(", ", 1)[-1] for p in glob if p.label})
    if glob:
        d = haversine_km(lat, lon, np.asarray([p.lat for p in glob]), np.asarray([p.lon for p in glob]))
        hit = np.flatnonzero(d < 25)
        out["rank_near"] = int(hit[0]) if len(hit) else None  # rank of the first result < 25 km in the global list (None: not in the top 200)
        out["top_km"] = float(d[0])
    near = ms.query(name, near=(lat, lon, 25.0), top=3)
    if near:
        out["cls"] = "right"
        out["near_exact"] = any(p.exact for p in near)
        return out
    key = norm(name)
    best = process.extractOne(key, names_near, scorer=fuzz.ratio) if names_near and len(key) >= 4 else None
    out["fuzzy"] = (best[0], round(best[1], 1)) if best else None
    if best and best[1] >= 80:
        out["cls"] = "misread"
    elif glob:
        out["cls"] = "elsewhere"
    else:
        out["cls"] = "nowhere"
    return out


def run(tag: str, ms: MapsSearch) -> None:
    z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
    km = z[KM[tag]]
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text(encoding="utf-8"))
    obs = json.loads(Path("artifacts/agent", OBS[tag]).read_text(encoding="utf-8"))
    rows = []
    for i, p in enumerate(photos):
        if km[i] < 25:
            continue
        parsed = (obs.get(p["image_id"]) or {}).get("parsed") or {}
        names = list(dict.fromkeys(n.strip() for n in parsed.get("place_names") or [] if isinstance(n, str) and len(n.strip()) >= 3))
        if not names:
            continue
        truth = tuple(json.loads(p["truth"]) if isinstance(p["truth"], str) else p["truth"])
        nn = near_names(ms, *truth)
        res = [classify(ms, n, truth, nn) for n in names]
        cls = min((r["cls"] for r in res), key=ORDER.index)
        rows.append({"image_id": p["image_id"], "truth": truth, "km": float(km[i]), "cls": cls, "names": res, "country": parsed.get("country")})
        if len(rows) % 25 == 0:
            print(f"  {tag}: {len(rows)} photos", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"name_miss_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_set, n_wrong = len(km), int((km >= 25).sum())
    c = collections.Counter(r["cls"] for r in rows)
    print(f"\n{tag}: {n_set} photos, pipeline wrong {n_wrong}, wrong with a read place name {len(rows)} ({100 * len(rows) / n_set:.1f} pts)")
    for k in ORDER:
        print(f"  {k:9s} {c[k]:4d} photos ({100 * c[k] / max(1, len(rows)):4.1f}% of them, {100 * c[k] / n_set:4.1f} pts of the set)")
    names = [x for r in rows for x in r["names"]]
    cn = collections.Counter(x["cls"] for x in names)
    print(f"  names: {len(names)}; " + ", ".join(f"{k} {cn[k]}" for k in ORDER))
    right = [x for r in rows if r["cls"] == "right" for x in r["names"] if x["cls"] == "right"]
    if right:
        ranks = [x.get("rank_near") for x in right]
        print(f"  right names: global matches median {np.median([x['matches'] for x in right]):.0f}, countries median "
              f"{np.median([x['countries'] for x in right]):.0f}; first result < 25 km of the truth at rank 0: {sum(r == 0 for r in ranks)}, "
              f"1-9: {sum(r is not None and 0 < r < 10 for r in ranks)}, 10-199: {sum(r is not None and r >= 10 for r in ranks)}, "
              f"beyond the top 200: {sum(r is None for r in ranks)}")
    els = [x for x in names if x["cls"] == "elsewhere"]
    if els:
        print(f"  elsewhere names: matches median {np.median([x['matches'] for x in els]):.0f}, in >= 5 countries "
              f"{np.mean([x['countries'] >= 5 for x in els]):.0%}, top result median {np.median([x.get('top_km', 0) for x in els]):.0f} km off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=list(KM))
    args = ap.parse_args()
    ms = MapsSearch()
    for tag in args.tags:
        run(tag, ms)


if __name__ == "__main__":
    main()
