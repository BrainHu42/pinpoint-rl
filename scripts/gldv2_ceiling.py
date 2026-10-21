# GLDv2 as a landmark gallery, step 1: the ceiling without images. How many commons26 dev photos have a GLDv2 landmark near the truth?
# Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_ceiling.py {coords,ceiling}
#   coords: GLDv2 landmark -> Commons category -> Wikidata item (P373 Commons category, or the category's sitelink) -> P625 coordinates
#   ceiling: per commons26 dev photo, GLDv2 landmarks / images and MP16 index photos within 1 / 25 km of the truth, by tier and pipeline outcome
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

OUT = Path("artifacts/landmarks/gldv2")
SPARQL = "https://query.wikidata.org/sparql"
UA = "pinpoint-rl research (geolocation benchmark; contact brainhu42@gmail.com)"


def _query(q: str) -> list[dict]:
    for attempt in range(6):
        try:
            req = urllib.request.Request(SPARQL, data=urllib.parse.urlencode({"query": q, "format": "json"}).encode(),
                                         headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["results"]["bindings"]
        except Exception as error:  # noqa: BLE001
            print(f"  retry {attempt}: {error}", flush=True)
            time.sleep(10 * 2 ** attempt)
    raise RuntimeError("SPARQL failed")


def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def coords(chunk: int = 300) -> None:
    cats = {}
    for r in csv.DictReader(open(OUT / "train_label_to_category.csv")):
        cats[int(r["landmark_id"])] = urllib.parse.unquote(r["category"].rsplit("Category:", 1)[-1]).replace("_", " ")
    path = OUT / "coords.json"
    found = json.loads(path.read_text()) if path.exists() else {}
    done_path = OUT / "coords_done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    names = sorted(set(cats.values()) - done)
    t0 = time.time()
    for s in range(0, len(names), chunk):
        part = names[s : s + chunk]
        vals = " ".join(_lit(n) for n in part)
        cvals = " ".join(_lit("Category:" + n) + "@en" for n in part)
        # P373 on the item, or the category's Commons sitelink (names are tagged @en) on the item itself or on a category item whose
        # main topic (P301) has P625
        q = f"""SELECT ?name ?coord WHERE {{
          {{ VALUES ?name {{ {vals} }} ?item wdt:P373 ?name; wdt:P625 ?coord. }}
          UNION {{ VALUES ?cl {{ {cvals} }} ?sl schema:name ?cl; schema:isPartOf <https://commons.wikimedia.org/>; schema:about ?ci.
                   {{ ?ci wdt:P625 ?coord. }} UNION {{ ?ci wdt:P301 ?topic. ?topic wdt:P625 ?coord. }}
                   BIND(STRAFTER(STR(?cl), "Category:") AS ?name) }}
        }}"""
        for b in _query(q):
            c = b["coord"]["value"]
            if c.startswith("Point("):
                lon, lat = map(float, c[6:-1].split())
                found.setdefault(b["name"]["value"], [lat, lon])
        done.update(part)
        if (s // chunk) % 20 == 0:
            path.write_text(json.dumps(found))
            done_path.write_text(json.dumps(sorted(done)))
            print(f"  {s + len(part)}/{len(names)} categories, {len(found)} with coordinates ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(0.5)
    path.write_text(json.dumps(found))
    done_path.write_text(json.dumps(sorted(done)))
    print(f"{len(cats)} landmarks, {len(set(cats.values()))} categories, {len(found)} with coordinates")


def _xyz(ll: np.ndarray) -> np.ndarray:
    la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def _chord(km: float) -> float:
    return 2 * np.sin(km / 6371.0 / 2)


def ceiling() -> None:
    from scipy.spatial import cKDTree

    from geo_search_env.models.pinpoint_reranker import _discover_index

    cats = {int(r["landmark_id"]): urllib.parse.unquote(r["category"].rsplit("Category:", 1)[-1]).replace("_", " ")
            for r in csv.DictReader(open(OUT / "train_label_to_category.csv"))}
    found = json.loads((OUT / "coords.json").read_text())
    n_img = {}
    for r in csv.DictReader(open(OUT / "train.csv")):
        n_img[int(r["landmark_id"])] = n_img.get(int(r["landmark_id"]), 0) + 1
    clean = {int(r["landmark_id"]): len(r["images"].split()) for r in csv.DictReader(open(OUT / "train_clean.csv"))}
    ids = np.asarray([k for k in cats if cats[k] in found])
    ll = np.asarray([found[cats[k]] for k in ids])
    imgs = np.asarray([n_img.get(k, 0) for k in ids])
    cl = np.asarray([clean.get(k, 0) for k in ids])
    print(f"GLDv2 train: {len(cats)} landmarks, {sum(n_img.values())} images; with coordinates {len(ids)} landmarks ({100 * len(ids) / len(cats):.1f}%), "
          f"{imgs.sum()} images ({100 * imgs.sum() / sum(n_img.values()):.1f}%); in the clean subset {np.sum(cl > 0)} landmarks, {cl.sum()} images")
    tree = cKDTree(_xyz(ll))

    index_dir = _discover_index()
    man = json.loads((index_dir / "manifest.json").read_text())
    mp = np.fromfile(index_dir / man["files"]["latlon_deg"], dtype=np.float32).reshape(-1, 2).astype(np.float64)
    mtree = cKDTree(_xyz(mp))

    tag, key = "commons26dev_setft_own", "commons26dev:setft_own|RL policy, lambda 0"
    photos = json.loads((Path("artifacts/photo_sets") / tag / "photos.json").read_text())
    km = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")[key]
    bm = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
    tier = np.asarray([bm[e["image_id"]]["tier"] for e in photos])
    q = _xyz(np.asarray([e["truth"] for e in photos], np.float64))
    feats = {}
    for r in (0.3, 1, 5, 25):
        near = tree.query_ball_point(q, _chord(r))
        feats[f"lm{r}"] = np.asarray([len(x) for x in near])
        feats[f"img{r}"] = np.asarray([imgs[x].sum() if x else 0 for x in near])
        feats[f"clean{r}"] = np.asarray([cl[x].sum() if x else 0 for x in near])
        feats[f"mp{r}"] = np.asarray([len(x) for x in mtree.query_ball_point(q, _chord(r))])
    d_near, _ = tree.query(q)
    d_km = 2 * 6371.0 * np.arcsin(np.clip(d_near / 2, 0, 1))
    np.savez(OUT / "ceiling_commons26dev.npz", tier=tier, km=km, d_km=d_km, **feats)

    def row(label, m):
        print(f"  {label:34s} {m.sum():4d} | nearest landmark median {np.median(d_km[m]):6.1f} km | landmark within 0.3 / 1 / 5 / 25 km "
              + " / ".join(f"{100 * np.mean(feats[f'lm{r}'][m] > 0):4.1f}" for r in (0.3, 1, 5, 25))
              + f" | images within 1 km median {np.median(feats['img1'][m]):3.0f} | MP16 photos within 1 km: none {100 * np.mean(feats['mp1'][m] == 0):4.1f}%, "
              f"median {np.median(feats['mp1'][m]):4.0f}")

    print("commons26 dev, % of photos (GLDv2 train landmarks with Wikidata coordinates)")
    wrong = km >= 25
    for t in ("landmark", "city", "region", "none"):
        row(f"tier {t}", tier == t)
        row(f"tier {t}, pipeline wrong", (tier == t) & wrong)
    geo = tier != "none"
    row("geolocatable", geo)
    row("geolocatable, pipeline wrong", geo & wrong)
    row("geolocatable, wrong, no MP16 photo < 1 km", geo & wrong & (feats["mp1"] == 0))
    for r in (0.3, 1):
        m = geo & wrong & (feats[f"lm{r}"] > 0)
        print(f"  pipeline wrong with a GLDv2 landmark within {r} km: {m.sum()} photos = {100 * m.sum() / len(photos):.1f} pts of all dev "
              f"({100 * (m & (tier == 'landmark')).sum() / len(photos):.1f} landmark tier)")


if __name__ == "__main__":
    {"coords": coords, "ceiling": ceiling}[sys.argv[1]]()
