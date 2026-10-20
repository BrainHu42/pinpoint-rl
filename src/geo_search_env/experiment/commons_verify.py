# Wikimedia Commons as a live reference gallery (retrieval track, lesson R7): does imagery near each candidate place tell the right one?
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_verify {pairs,fetch,embed,report}

"""pairs: commons26 dev photos tagged geolocatable (tier landmark / city / region). Place 0 = the pipeline's answer (action_agent, RL lambda 0,
--one-km 0.5); places 1..k = the top-k distinct text-search places (lesson 119's clusters, ordered by share / log2(2 + n) as text_verify,
> 25 km apart and from place 0).
fetch: per place, Commons files within 2 km (list=geosearch, namespace File), the 10 nearest images uploaded before 2026 (every commons26 photo
was taken in 2026) and not by the photo's author; 400 px thumbnails -> artifacts/retrieval/commons_verify/.
embed: SigLIP2 (embed_cache) and MegaLoc (megaloc_gate) for the queries and the thumbnails.
report: per place the best / top-3 similarity to the query (Commons SigLIP2, Commons MegaLoc; MP16 index photos within 2 km, SigLIP2, as the
offline baseline); AUC of right vs wrong places, and the switch "to the best text place when it beats the pipeline's answer by a margin" as
fixes / breaks in points of the geolocatable and of all dev photos."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .commons_bench import USER_AGENT, _request

OUT = Path("artifacts/retrieval/commons_verify")
TAG, KEY = "commons26dev_setft_own", "commons26dev:setft_own|RL policy, lambda 0"
APPLY = Path("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
CANDS = Path("artifacts/agent/cands_commons26_dev.jsonl")
RELEASE = Path("data/pinpoint/commons26/release")
RADIUS_M, PER_PLACE, THUMB = 2000, 10, 400
BEFORE = "2026-01-01T00:00:00Z"
N_ALL = 977


def _hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def pairs(k_text: int) -> None:
    photos = json.loads((Path("artifacts/photo_sets") / TAG / "photos.json").read_text())
    z = np.load(APPLY)
    centre = z[f"{KEY}|center"]
    bm = {r["IMG_ID"]: r for r in csv.DictReader(open(RELEASE / "benchmark.csv"))}
    artist = {r["IMG_ID"]: r["artist"] for r in csv.DictReader(open(RELEASE / "attribution.csv"))}
    cands = {}
    for line in CANDS.read_text().splitlines():
        rec = json.loads(line)
        items = [(c[0], c[1], c[3] / math.log2(2 + r["n"])) for q in rec["queries"] for r in q["src"].values() for c in r["clusters"]]
        cands[rec["image_id"]] = sorted(items, key=lambda x: -x[2])
    out = []
    for i, e in enumerate(photos):
        if bm[e["image_id"]]["tier"] == "none":
            continue
        places = [(float(centre[i][0]), float(centre[i][1]))]
        for la, lo, _ in cands.get(e["image_id"], []):
            if len(places) > k_text:
                break
            if all(_hav(la, lo, p[0], p[1]) > 25 for p in places):
                places.append((float(la), float(lo)))
        t = e["truth"]
        out.append({"image_id": e["image_id"], "artist": artist.get(e["image_id"], ""), "places": places,
                    "right": [bool(_hav(t[0], t[1], la, lo) < 25) for la, lo in places]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs.json").write_text(json.dumps(out, indent=0))
    base = np.asarray([p["right"][0] for p in out])
    alt = np.asarray([any(p["right"][1:]) for p in out])
    print(f"{len(out)} geolocatable photos, {np.mean([len(p['places']) for p in out]):.1f} places each; pipeline right {100 * base.mean():.1f}%, "
          f"a text place right where the pipeline is wrong {100 * (alt & ~base).mean():.1f}% (oracle {100 * (base | alt).mean():.1f}%)")


def _near(lat: float, lon: float, artist: str) -> list[dict]:
    geo = _request({"action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}", "gsradius": RADIUS_M, "gslimit": 100, "gsnamespace": 6})
    hits = sorted(geo.get("query", {}).get("geosearch", []), key=lambda h: h["dist"])
    keep = []
    for s in range(0, len(hits), 50):
        chunk = hits[s : s + 50]
        info = _request({"action": "query", "titles": "|".join(h["title"] for h in chunk), "prop": "imageinfo",
                         "iiprop": "url|timestamp|user|mime", "iiurlwidth": THUMB}, post=True)
        pages = {p.get("title"): p for p in info.get("query", {}).get("pages", {}).values()}
        for h in chunk:
            ii = (pages.get(h["title"], {}).get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png") or not ii.get("thumburl") or ii.get("timestamp", "9") >= BEFORE:
                continue
            if artist and ii.get("user", "").lower() == artist.lower():
                continue
            keep.append({"title": h["title"], "lat": h["lat"], "lon": h["lon"], "dist": h["dist"], "ts": ii["timestamp"], "url": ii["thumburl"]})
        if len(keep) >= PER_PLACE:
            break
    return keep[:PER_PLACE]


def _download(job):
    url, dst = job
    if dst.exists():
        return True
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
                dst.write_bytes(r.read())
            return True
        except Exception as error:  # noqa: BLE001
            code = getattr(error, "code", None)
            if code == 404:
                return False
            time.sleep(2 ** attempt * (5 if code == 429 else 1))
    return False


def fetch(workers: int) -> None:
    ps = json.loads((OUT / "pairs.json").read_text())
    refs_path = OUT / "refs.json"
    refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    jobs = [(f"{p['image_id']}|{j}", place, p["artist"]) for p in ps for j, place in enumerate(p["places"]) if f"{p['image_id']}|{j}" not in refs]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for k, (key, found) in enumerate(zip([j[0] for j in jobs], pool.map(lambda j: _near(j[1][0], j[1][1], j[2]), jobs))):
            refs[key] = found
            if k % 100 == 0:
                refs_path.write_text(json.dumps(refs))
                print(f"  places {k}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    refs_path.write_text(json.dumps(refs))
    thumbs = OUT / "thumbs"
    thumbs.mkdir(exist_ok=True)
    files = {r["url"]: thumbs / f"{abs(hash(r['url'])) % 10 ** 12:012d}.jpg" for rs in refs.values() for r in rs}
    with ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(_download, files.items()))
    (OUT / "files.json").write_text(json.dumps({u: str(p) for (u, p), k in zip(files.items(), ok) if k}))
    print(f"{len(refs)} places, {sum(len(r) for r in refs.values())} references ({np.mean([len(r) for r in refs.values()]):.1f} per place, "
          f"{sum(not r for r in refs.values())} places with none); downloaded {sum(ok)}/{len(ok)} ({time.time() - t0:.0f}s)")


def embed() -> None:
    import torch
    from PIL import Image

    from .embed_cache import _embed, _load_model, _pixels
    from .megaloc_gate import SIZE, _model

    ps = json.loads((OUT / "pairs.json").read_text())
    files = json.loads((OUT / "files.json").read_text())
    paths = [str(RELEASE / "images" / p["image_id"]) for p in ps] + sorted(set(files.values()))
    blobs = [Path(p).read_bytes() for p in paths]
    siglip, processor = _load_model()
    s_out = np.zeros((len(paths), 1536), np.float16)
    for s in range(0, len(paths), 64):
        px = [_pixels(processor, b) for b in blobs[s : s + 64]]
        ok = [k for k, x in enumerate(px) if x is not None]
        if ok:
            s_out[[s + k for k in ok]] = _embed(siglip, torch.stack([px[k] for k in ok]))
    del siglip
    torch.cuda.empty_cache()
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

    def tf(b):
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            return torch.from_numpy(((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy())
        except Exception:  # noqa: BLE001
            return torch.zeros(3, SIZE, SIZE)

    megaloc = _model()
    m_out = np.zeros((len(paths), 8448), np.float16)
    with torch.inference_mode():
        for s in range(0, len(paths), 32):
            m_out[s : s + 32] = megaloc(torch.stack([tf(b) for b in blobs[s : s + 32]]).cuda()).float().cpu().numpy()
    np.savez(OUT / "embeddings.npz", paths=np.asarray(paths), siglip=s_out, megaloc=m_out)
    print(f"embedded {len(paths)} images ({len(ps)} queries)")


def report(margin: float) -> None:
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import _discover_index

    ps = json.loads((OUT / "pairs.json").read_text())
    refs = json.loads((OUT / "refs.json").read_text())
    files = json.loads((OUT / "files.json").read_text())
    z = np.load(OUT / "embeddings.npz")
    at = {p: i for i, p in enumerate(z["paths"])}
    unit = lambda x: x.astype(np.float32) / np.maximum(np.linalg.norm(x.astype(np.float32), axis=-1, keepdims=True), 1e-8)
    S, M = unit(z["siglip"]), unit(z["megaloc"])
    # MP16 baseline: index photos within 2 km of each place, raw SigLIP2 (the index's int8 part)
    index_dir = _discover_index()
    man = json.loads((index_dir / "manifest.json").read_text())
    count = man["num_samples"]
    ll = np.fromfile(index_dir / man["files"]["latlon_deg"], dtype=np.float32).reshape(-1, 2).astype(np.float64)
    raw = np.memmap(index_dir / man["files"]["raw_image_embeddings"], dtype=np.int8, mode="r", shape=(count, 1536))
    xyz = lambda a: np.stack([np.cos(np.radians(a[:, 0])) * np.cos(np.radians(a[:, 1])), np.cos(np.radians(a[:, 0])) * np.sin(np.radians(a[:, 1])),
                              np.sin(np.radians(a[:, 0]))], 1)
    tree = cKDTree(xyz(ll))
    n, P = len(ps), max(len(p["places"]) for p in ps)
    names = ("commons siglip", "commons siglip top3", "commons megaloc", "commons megaloc top3", "mp16 siglip", "mp16 siglip top3")
    feats = {k: np.full((n, P), np.nan) for k in names}
    counts = np.zeros((n, P), int)
    right, valid = np.zeros((n, P), bool), np.zeros((n, P), bool)
    for i, p in enumerate(ps):
        q = at[str(RELEASE / "images" / p["image_id"])]
        right[i, : len(p["right"])], valid[i, : len(p["right"])] = p["right"], True
        for j, place in enumerate(p["places"]):
            rows = [at[files[r["url"]]] for r in refs.get(f"{p['image_id']}|{j}", []) if r["url"] in files]
            counts[i, j] = len(rows)
            if rows:
                for name, E in (("siglip", S), ("megaloc", M)):
                    sims = np.sort(E[rows] @ E[q])[::-1]
                    feats[f"commons {name}"][i, j], feats[f"commons {name} top3"][i, j] = sims[0], sims[:3].mean()
            near = tree.query_ball_point(xyz(np.asarray([place]))[0], 2 * math.sin(2.0 / 6371.0 / 2))[:2000]
            if near:
                sims = np.sort(unit(np.asarray(raw[np.sort(near)], dtype=np.float32)) @ S[q])[::-1]
                feats["mp16 siglip"][i, j], feats["mp16 siglip top3"][i, j] = sims[0], sims[:3].mean()
    base, alt = right[:, 0], right[:, 1:].any(1)
    print(f"{n} geolocatable photos, up to {P} places; pipeline {100 * base.mean():.1f}%, oracle {100 * (base | alt).mean():.1f}%; Commons references "
          f"per place: median {np.median(counts[valid]):.0f}; places with none: right {100 * np.mean(counts[right] == 0):.1f}%, wrong "
          f"{100 * np.mean(counts[valid & ~right] == 0):.1f}%")
    for name, f in feats.items():
        g = np.where(np.isfinite(f) & valid, f, -np.inf)
        rs, ws = g[right & np.isfinite(g)], g[valid & ~right & np.isfinite(g)]
        auc = (rs[:, None] > ws[None]).mean() + 0.5 * (rs[:, None] == ws[None]).mean()
        best = g[:, 1:].argmax(1) + 1
        gain = g[np.arange(n), best] - g[:, 0]
        lines = []
        for m in sorted({margin, 0.05, 0.1}):
            switch = np.isfinite(gain) & (gain > m)
            new = np.where(switch, right[np.arange(n), best], base)
            fixes, breaks = int((new & ~base).sum()), int((~new & base).sum())
            lines.append(f"margin {m}: switch {switch.sum()} fixes {fixes} breaks {breaks} = {100 * (fixes - breaks) / n:+.1f} geo / "
                         f"{100 * (fixes - breaks) / N_ALL:+.2f} all")
        print(f"  {name:20s} AUC {auc:.3f} | " + " | ".join(lines))
    np.savez(OUT / "features.npz", right=right, valid=valid, counts=counts, **{k.replace(" ", "_"): v for k, v in feats.items()})


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pairs")
    pp.add_argument("--k-text", type=int, default=5)
    f = sub.add_parser("fetch")
    f.add_argument("--workers", type=int, default=4)
    sub.add_parser("embed")
    r = sub.add_parser("report")
    r.add_argument("--margin", type=float, default=0.0)
    a = p.parse_args(argv)
    if a.cmd == "pairs":
        pairs(a.k_text)
    elif a.cmd == "fetch":
        fetch(a.workers)
    elif a.cmd == "embed":
        embed()
    else:
        report(a.margin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
