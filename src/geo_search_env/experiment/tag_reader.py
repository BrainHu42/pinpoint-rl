# Gate for the tool-reading agent (lesson 86): can the 4B choose among Pinpoint's areas by reading the titles / tags of each area's photos?
# Per dev3k photo, Pinpoint's areas (hypotheses.py, source 0: ~4.6 per photo, each with its restricted search, photographer excluded); per
# area its label (nearest town) and the YFCC titles + tags of its restricted candidates (data/yfcc100m/geo.npz). The 4B sees the photo and the
# areas as letters (A = Pinpoint's top area) and answers one letter; P(letter) from the first token's log-probabilities. Conditions: labels
# only (control) and labels + tags. An area's answer is its restricted top-1 (the 51.7% oracle of lesson 85).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.tag_reader prepare
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.tag_reader ask --cond tags      (VLM served at :8765; also --cond labels)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.tag_reader report

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
TAG = "dev3k_setft_a"
AREAS = OUT / f"tag_reader_{TAG}.json"
LETTERS = "ABCDEFGH"
PER_AREA = 8  # neighbour titles shown per area
CHARS = 90
PROMPTS = {
    "tags": ("This photo was taken in one of the candidate areas below. For each area, the titles and tags of photos taken there that look "
             "similar to this one are listed. Which area is the photo from? Use what you see in the photo and what the listed photos say about "
             "each area. Answer with the letter only.\n\n{areas}"),
    "labels": ("This photo was taken in one of the candidate areas below. Which area is the photo from? Use what you see in the photo. Answer "
               "with the letter only.\n\n{areas}"),
}


def prepare() -> None:
    from ..models.pinpoint_reranker import _discover_index
    from .strategy_search import MP16_CSV, _haversine_km, _memmap_gallery, MP16_EMBED
    from .text_search import TextSearch

    photos = json.loads((Path("artifacts/photo_sets") / TAG / "photos.json").read_text(encoding="utf-8"))
    z = np.load(OUT / f"hypotheses_{TAG}.npz")
    keep = z["source"] == 0
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    print("MP16 photo ids", flush=True)
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        i = next(reader).index("IMG_ID")
        csv_ids = np.asarray([int(r[i].rsplit("_", 1)[-1].split(".")[0]) for r in reader], np.int64)
    rows = z["rows"][keep]
    needed = np.unique(rows[rows >= 0])
    flickr = dict(zip(needed.tolist(), csv_ids[row_index[cache[needed]]].tolist()))
    print("YFCC titles / tags", flush=True)
    g = np.load("data/yfcc100m/geo.npz")
    pid, offsets = g["photo_id"], g["offsets"]
    want = np.asarray(sorted(set(flickr.values())), np.int64)
    pos = np.searchsorted(pid, want)
    ok = (pos < len(pid)) & (pid[np.minimum(pos, len(pid) - 1)] == want)
    blob = g["blob"]
    text = {int(w): bytes(blob[offsets[p]:offsets[p + 1] - 1]).decode("utf-8", "replace").strip() for w, p, o in zip(want, pos, ok) if o}
    print(f"titles for {len(text)} of {len(want)} neighbour photos", flush=True)
    ts = TextSearch(yfcc=False)
    out = []
    for p, e in enumerate(photos):
        idx = np.flatnonzero(keep & (z["photo"] == p))
        idx = idx[np.argsort(z["rank"][idx])][: len(LETTERS)]
        areas = []
        for j in idx:
            ll = z["latlon"][j]
            v = np.flatnonzero(z["rows"][j] >= 0)
            if not len(v):
                continue
            top = ll[v[0]].astype(np.float64)
            lines = []
            for r in z["rows"][j][v]:
                t = text.get(flickr.get(int(r), -1), "")
                if t and t[:CHARS] not in lines:
                    lines.append(t[:CHARS])
            areas.append({"label": ts.label(*top), "answer": top.tolist(), "km": float(_haversine_km(*e["truth"], top[None])[0]),
                          "texts": lines[:PER_AREA]})
        out.append({"image_id": e["image_id"], "index": p, "truth": e["truth"], "areas": areas,
                    "top1_km": float(_haversine_km(*e["truth"], np.asarray(e["candidates"][0][1:], np.float64)[None])[0])})
    AREAS.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {AREAS}: {len(out)} photos, {np.mean([len(o['areas']) for o in out]):.1f} areas, "
          f"{np.mean([len(a['texts']) for o in out for a in o['areas']]):.1f} titles per area")


def _areas_text(o: dict, tags: bool) -> str:
    lines = []
    for k, a in enumerate(o["areas"]):
        lines.append(f"{LETTERS[k]}) near {a['label']}")
        if tags:
            lines += [f"   - {t}" for t in a["texts"]] or ["   (no titles)"]
    return "\n".join(lines)


def ask(cond: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from .name_score import _post, _read
    from .sft_data import MP16Images

    rows, images = json.loads(AREAS.read_text(encoding="utf-8")), MP16Images()

    def one(o: dict) -> list[float]:
        n = len(o["areas"])
        if n < 2:
            return [1.0] * n
        text = PROMPTS[cond].format(areas=_areas_text(o, cond == "tags"))
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_read(o, images)).decode()}},
                    {"type": "text", "text": text}]}]}
        try:
            top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception:  # noqa: BLE001
            return [float("nan")] * n
        p = [0.0] * n
        for t in top:
            k = LETTERS.find(t["token"].strip())
            if 0 <= k < n and len(t["token"].strip()) == 1:
                p[k] += math.exp(t["logprob"])
        s = sum(p)
        return [x / s for x in p] if s > 0 else [float("nan")] * n

    with ThreadPoolExecutor(workers) as pool:
        probs = list(pool.map(one, rows))
    dst = OUT / f"tag_reader_{TAG}_{cond}.json"
    dst.write_text(json.dumps(probs) + "\n", encoding="utf-8")
    print(f"{cond}: {sum(any(math.isnan(x) for x in p) for p in probs)} failed -> {dst}")


def report() -> None:
    rows = json.loads(AREAS.read_text(encoding="utf-8"))
    km = [np.asarray([a["km"] for a in o["areas"]]) for o in rows]
    n = len(rows)
    top1 = np.asarray([o["top1_km"] < 25 for o in rows])
    has = np.asarray([len(k) > 0 for k in km])
    area_a = np.asarray([k[0] < 25 if len(k) else t for k, t in zip(km, top1)])
    oracle = np.asarray([(k < 25).any() if len(k) else t for k, t in zip(km, top1)])
    multi = np.asarray([len(k) >= 2 for k in km])
    print(f"{n} photos ({multi.mean():.0%} with >= 2 areas); % < 25 km: retrained top-1 {top1.mean():.1%}, area A (Pinpoint's top area, its "
          f"restricted top-1) {area_a.mean():.1%}, best area {oracle.mean():.1%}; the answer head (lesson 85) 38.7%")
    for cond in ("labels", "tags"):
        path = OUT / f"tag_reader_{TAG}_{cond}.json"
        if not path.exists():
            continue
        probs = json.loads(path.read_text(encoding="utf-8"))
        pick = np.asarray([k[int(np.nanargmax(p))] < 25 if len(k) and not all(math.isnan(x) for x in p) else a
                           for k, p, a in zip(km, probs, area_a)])
        moved = np.asarray([len(p) > 0 and not all(math.isnan(x) for x in p) and int(np.nanargmax(p)) != 0 for p in probs])
        print(f"  4B pick, {cond:6s}: {pick.mean():.1%} ({100 * (pick.mean() - area_a.mean()):+.1f} vs area A); picks another area on "
              f"{moved.mean():.1%} of photos: fixes {(pick & ~area_a).sum()}, breaks {(~pick & area_a).sum()}")
        for t in (0.5, 0.7, 0.9):  # switch only when the 4B is confident in another area
            conf = np.asarray([len(p) > 0 and not all(math.isnan(x) for x in p) and int(np.nanargmax(p)) != 0 and np.nanmax(p) >= t for p in probs])
            g = np.where(conf, pick, area_a)
            print(f"    switch only at P >= {t}: {g.mean():.1%} ({100 * (g.mean() - area_a.mean()):+.1f}), switches {conf.mean():.1%}, "
                  f"fixes {(g & ~area_a).sum()}, breaks {(~g & area_a).sum()}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("prepare", "ask", "report"))
    parser.add_argument("--cond", choices=("labels", "tags"), default="tags")
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "ask":
        ask(args.cond)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
