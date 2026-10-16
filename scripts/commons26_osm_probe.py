"""Probe: the OSM Maps index (tools.maps, OSM + GeoNames + Overture names) on commons26 dev phrases and spans, global and near Pinpoint's areas.
Per photo: does any result land < 25 km? Compared with text_search on the same phrases / spans."""
import collections
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "global"
bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
dev = [i for i, r in bench.items() if r["split"] == "dev"]


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def work(job):
    from geo_search_env.tools.maps import MapsSearch

    global _M
    try:
        _M
    except NameError:
        _M = MapsSearch()
    i, queries, areas = job
    lat, lon = float(bench[i]["LAT"]), float(bench[i]["LON"])
    out = {"phrase": 1e9, "span": 1e9, "phrase_top": 1e9}
    for q, kind in queries:
        try:
            if mode == "global":
                res = _M.query(q, top=10)
            else:
                res = [p for a in areas for p in _M.query(q, near=(a[0], a[1], 25.0), top=5)]
        except Exception:  # noqa: BLE001
            continue
        if not res:
            continue
        d = hav(lat, lon, np.asarray([p.lat for p in res]), np.asarray([p.lon for p in res]))
        out[kind] = min(out[kind], float(d.min()))
        if kind == "phrase":
            out["phrase_top"] = min(out["phrase_top"], float(d[0]))
    return i, out


if __name__ == "__main__":
    from geo_search_env.experiment.read_constrain import _spans
    from geo_search_env.experiment.region_evidence import _raw_phrases

    phrases = _raw_phrases("commons26", "4b")
    photos = json.load(open("artifacts/photo_sets/commons26_setft_own/photos.json"))
    pidx = {e["image_id"]: k for k, e in enumerate(photos)}
    zm = np.load("artifacts/pinpoint_set/zoom_commons26_setft_own.npz")
    areas = collections.defaultdict(list)
    for j, p in enumerate(zm["photo"]):
        if zm["source"][j] == 0:
            areas[int(p)].append(zm["center"][j].astype(float).tolist())
    jobs = []
    for i in dev:
        ph = list(phrases.get(i, {}))
        qs = [(q, "phrase") for q in ph] + [(s, "span") for s in sorted({s for q in ph for s in _spans(q)} - set(ph))]
        jobs.append((i, qs, areas[pidx[i]]))
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(8) as pool:
        for n, (i, out) in enumerate(pool.map(work, jobs, chunksize=4)):
            res[i] = out
            if n % 100 == 0:
                print(n, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(f"artifacts/commons26/osm_probe_{mode}_dev.json", "w"))
    ts_phr = collections.defaultdict(lambda: 1e9)
    for r in json.load(open("artifacts/agent/clusters_commons26_4b.json")):
        if r["image_id"] in res and r["clusters"]:
            ts_phr[r["image_id"]] = min(ts_phr[r["image_id"]], min(float(hav(float(bench[r["image_id"]]["LAT"]), float(bench[r["image_id"]]["LON"]), c["lat"], c["lon"])) for c in r["clusters"]))
    ts_spn = collections.defaultdict(lambda: 1e9)
    for r in json.load(open("artifacts/agent/spans_commons26_4b_dev.json")):
        if r["clusters"]:
            ts_spn[r["image_id"]] = min(ts_spn[r["image_id"]], min(float(hav(float(bench[r["image_id"]]["LAT"]), float(bench[r["image_id"]]["LON"]), c["lat"], c["lon"])) for c in r["clusters"]))
    for g, ids in (("all dev", dev), ("place_text=1", [i for i in dev if bench[i]["place_text"] == "1"])):
        ts_any = {i: min(ts_phr[i], ts_spn[i]) < 25 for i in ids}
        osm_ph = {i: res[i]["phrase"] < 25 for i in ids}
        osm_any = {i: min(res[i]["phrase"], res[i]["span"]) < 25 for i in ids}
        print(f"{g} ({len(ids)}), {mode}: OSM phrases {np.mean(list(osm_ph.values())):.1%} (top result {np.mean([res[i]['phrase_top'] < 25 for i in ids]):.1%}), "
              f"OSM phrases+spans {np.mean(list(osm_any.values())):.1%}; text_search phrases {np.mean([ts_phr[i] < 25 for i in ids]):.1%}, phrases+spans "
              f"{np.mean(list(ts_any.values())):.1%}; OSM finds where text_search (phrases+spans) doesn't: {np.mean([osm_any[i] and not ts_any[i] for i in ids]):.1%}; "
              f"either {np.mean([osm_any[i] or ts_any[i] for i in ids]):.1%}")
