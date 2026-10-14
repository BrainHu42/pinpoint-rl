"""Are my blind-test queries findable? Each name / text item from fixtures/reader_test_dev3k.json through text_search (photo and photographer
excluded, as the pipeline) and tools.maps (OSM); top result's distance to the truth and whether any result is within 25 km."""
import json
import re
from pathlib import Path

import numpy as np

from geo_search_env.tools.maps import MapsSearch
from geo_search_env.experiment.region_evidence import _owns
from geo_search_env.experiment.text_search import TextSearch, haversine_km

key = {k["p"]: k for k in json.loads(Path("artifacts/agent/reader_test/reader_key/key.json").read_text())}
notes = json.loads(Path("fixtures/reader_test_dev3k.json").read_text())
photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_a/photos.json").read_text())
owns = _owns([photos[key[n["p"]]["index"]] for n in notes])
ts, ms = TextSearch(), MapsSearch()


def clean(x):
    x = re.sub(r"\(.*?\)", "", x).replace("?", "").replace("...", " ").strip()
    return x if len(x) >= 3 and any(c.isalpha() for c in x) else None


for n in notes:
    k = key[n["p"]]
    items = [(q, kind) for kind in ("names", "text") for q in (clean(x) for x in n[kind]) if q]
    if not items:
        continue
    truth = np.asarray(k["truth"], np.float64)
    own = owns[k["image_id"]]
    print(f"\n{n['p']} ({n['tier']}): truth near {truth.round(3).tolist()}")
    for q, kind in dict.fromkeys(items):
        r = ts.search(q, exclude=own)
        if r.clusters:
            d = [float(haversine_km(*truth, np.asarray([c.lat]), np.asarray([c.lon]))[0]) for c in r.clusters]
            t_txt = f"top {d[0]:.0f} km (weight {r.clusters[0].weight:.0f}, share {r.clusters[0].share:.2f}), any < 25 km: {min(d) < 25}, {sum(r.matches.values())} matches"
        else:
            t_txt = "no match"
        res = ms.query(q)
        if res:
            dm = [float(haversine_km(*truth, np.asarray([p.lat]), np.asarray([p.lon]))[0]) for p in res]
            m_txt = f"top {dm[0]:.0f} km, any < 25 km: {min(dm) < 25} ({len(res)} shown)"
        else:
            m_txt = "no match"
        print(f"  {kind:5s} '{q}': text_search {t_txt} | OSM {m_txt}")
