# Is choosing among candidates limited by evidence? Zero-shot VLM with per-candidate photos or nearby place names.
# Usage: .venv/bin/python -m geo_search_env.experiment.evidence_test landmarks   (once: GeoNames landmark index)
#        .venv/bin/python -m geo_search_env.experiment.evidence_test run --evidence {none,photos,places}   (vLLM on :8765)

"""Same model, same eval photos and the reranker's top-10 candidates, three prompts:

none:   candidate name and coordinates (the default prompt; base-model baseline)
photos: plus one database photo per candidate (the most similar retrieved MP16 photo within 1 km of it, else the
        nearest MP16 photo; never the query's photographer), as a 256 px thumbnail
places: plus up to 4 named landmarks within 1 km of each candidate from GeoNames (hotels, schools and generic
        buildings dropped)

Greedy answers on the full benchmark eval halves; unparseable answers fall back to the reranker's #1.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import time
import urllib.request
from typing import Any, Sequence

import numpy as np

from ..data.benchmarks import compute_metrics
from .pivot_diagnostics import PROMPT, parse_coordinates
from .sft_data import MP16Images
from .strategy_search import BENCHMARK_NAMES, EARTH_KM, MP16_EMBED, THRESHOLDS_KM, _xyz, load_world
from .verifiers import _place_names, reranker_ranking


ROOT = Path("artifacts/strategy_search")
GEONAMES = Path("/data/pinpoint/geonames")
LANDMARKS = GEONAMES / "landmarks.npz"
CANDIDATES = 10
NEAR_KM = 1.0
MAX_PLACES = 4
THUMBNAIL = 256
# Common and uninformative feature codes (class S/L) left out of the nearby lists.
SKIP_CODES = {"S.HTL", "S.SCH", "S.BLDG", "S.HSP", "S.HSPC", "S.HSPD", "S.PO", "S.BANK", "S.REST", "S.SCHC", "S.MALL",
              "S.OFF", "S.ADMF", "S.PS", "S.RECG", "S.FCL", "S.MTRO", "S.", "L.", "S.CMTY", "S.FRM", "S.HMSD", "S.HSE",
              "S.MFG", "S.WALL", "S.RSRT", "S.SHPF", "S.CTRB", "L.LCTY", "L.AREA", "L.RGN", "L.PRT", "S.SCHT"}
INTRO = {
    "none": "Geolocate this photo. A retrieval system proposed these candidate locations, best first (all of them may be wrong):",
    "photos": "Geolocate the first photo. A retrieval system proposed these candidate locations, best first (all of them may be wrong). "
              "Each candidate is followed by a database photo taken near it:",
    "places": "Geolocate this photo. A retrieval system proposed these candidate locations, best first (all of them may be wrong). "
              "Each lists named places within 1 km of it, where known:",
}
OUTRO = PROMPT.split("{options}\n", 1)[1]


def landmarks() -> None:
    """Class S/L GeoNames entries minus uninformative codes, with readable type names."""

    types = {}
    for line in (GEONAMES / "featureCodes_en.txt").read_text(encoding="utf-8").splitlines():
        code, name, *_ = line.split("\t") + [""]
        types[code] = name
    lat, lon, name, code = [], [], [], []
    with (GEONAMES / "allCountries.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            f = line.split("\t", 9)
            c = f[6] + "." + f[7]
            if f[6] in ("S", "L") and c not in SKIP_CODES:
                lat.append(float(f[4])); lon.append(float(f[5])); name.append(f[2] or f[1]); code.append(c)
    np.savez(LANDMARKS, lat=np.asarray(lat), lon=np.asarray(lon), name=np.asarray(name), type=np.asarray([types.get(c, c) for c in code]))
    print(f"{len(lat)} landmarks")


def _thumbnail(data: bytes) -> str:
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((THUMBNAIL, THUMBNAIL))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode()


def _image_part(b64: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}


def _ask(server: str, model: str, content: list[dict[str, Any]]) -> tuple[float, float] | None:
    body = {
        "model": model, "temperature": 0.0, "seed": 0, "max_tokens": 600,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return parse_coordinates(json.loads(response.read())["choices"][0]["message"]["content"] or "")
        except Exception:
            time.sleep(5 * 2**attempt)
    return None


def run(evidence: str, *, server: str, model: str, workers: int = 64) -> None:
    from scipy.spatial import cKDTree

    world = load_world()
    coords, valid, distance, ranking = reranker_ranking(world, ROOT)
    members = [q for q, x in enumerate(world.queries) if x["split"] == "eval"]
    shown = {q: [int(c) for c in ranking[q, :CANDIDATES] if valid[q, c]] for q in members}
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    nearest = {(q, c): int(world.mp16["row_index"][tree.query(_xyz(coords[q, c]))[1]]) for q in members for c in shown[q]}
    names = _place_names(set(nearest.values()))

    extra: dict[tuple[int, int], Any] = {}
    if evidence == "photos":
        image_ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
        with np.load(ROOT / "neighbors.npz") as saved:
            idx, sim = saved["mp16_raw_idx"], saved["mp16_raw_sim"]
        cos_near = math.cos(NEAR_KM / EARTH_KM)
        kinds = {"most similar within 1 km": 0, "nearest photo": 0}
        for q in members:
            finite = np.isfinite(sim[q])
            rows = idx[q][finite]
            hit_xyz = _xyz(world.mp16["latlon"][rows])
            for c in shown[q]:
                near = np.flatnonzero(hit_xyz @ _xyz(coords[q, c]) >= cos_near)
                if len(near):
                    row = int(rows[near[0]])  # neighbours are sorted by similarity
                    kinds["most similar within 1 km"] += 1
                else:
                    _, found = tree.query(_xyz(coords[q, c]), k=16)
                    other = [int(r) for r in found if world.mp16["author"][r] != world.query_author[q]]
                    row = other[0] if other else int(found[-1])
                    kinds["nearest photo"] += 1
                extra[(q, c)] = image_ids[row]
        print("exemplars:", kinds, flush=True)
    elif evidence == "places":
        saved = dict(np.load(LANDMARKS))  # NpzFile re-reads an array on every key access
        lm_tree = cKDTree(_xyz(np.c_[saved["lat"], saved["lon"]]))
        radius = 2 * math.sin(NEAR_KM / EARTH_KM / 2)
        listed = 0
        for q in members:
            for c in shown[q]:
                hits = np.asarray(lm_tree.query_ball_point(_xyz(coords[q, c]), radius), dtype=np.int64)
                gap = ((_xyz(np.c_[saved["lat"][hits], saved["lon"][hits]]) - _xyz(coords[q, c])) ** 2).sum(1)
                order = hits[np.argsort(gap)]
                seen, entries = set(), []
                for j in order:
                    if saved["name"][j] not in seen:
                        seen.add(saved["name"][j])
                        entries.append(f"{saved['name'][j]} ({saved['type'][j]})")
                    if len(entries) == MAX_PLACES:
                        break
                extra[(q, c)] = entries
                listed += bool(entries)
        print(f"candidates with nearby places: {listed / sum(map(len, shown.values())):.0%}", flush=True)

    images = MP16Images()

    def content(q: int) -> list[dict[str, Any]]:
        parts = [_image_part(base64.b64encode(Path(world.queries[q]["path"]).read_bytes()).decode()), {"type": "text", "text": INTRO[evidence]}]
        for rank, c in enumerate(shown[q], start=1):
            line = f"{rank}. {names[nearest[(q, c)]]} ({coords[q, c, 0]:.3f}, {coords[q, c, 1]:.3f})"
            if evidence == "places":
                line += "\n   nearby: " + (", ".join(extra[(q, c)]) if extra[(q, c)] else "none listed")
            parts.append({"type": "text", "text": line})
            if evidence == "photos":
                data = images.read(extra[(q, c)])
                if data:
                    parts.append(_image_part(_thumbnail(data)))
        parts.append({"type": "text", "text": OUTRO})
        return parts

    if evidence == "none":  # one text block, exactly the default prompt
        def request(q):
            options = "\n".join(p["text"] for p in content(q)[2:-1])
            return _ask(server, model, [content(q)[0], {"type": "text", "text": PROMPT.format(options=options)}])
    else:
        def request(q):
            return _ask(server, model, content(q))

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        answers = dict(zip(members, pool.map(request, members)))
    print(f"requests took {time.time() - started:.0f}s", flush=True)

    report: dict[str, Any] = {"evidence": evidence, "model": model}
    for name in (*BENCHMARK_NAMES, "both"):
        qs = [q for q in members if name == "both" or world.queries[q]["benchmark"] == name]
        truth = world.query_latlon[qs]
        points = np.asarray([answers[q] or tuple(coords[q, shown[q][0]]) for q in qs])
        report[name] = {
            "n": len(qs), "unparsed": float(np.mean([answers[q] is None for q in qs])),
            "reranker top-1": compute_metrics(np.asarray([coords[q, shown[q][0]] for q in qs]), truth),
            "VLM greedy": compute_metrics(points, truth),
        }
        print(f"{name:9s} n={len(qs)} unparsed {report[name]['unparsed']:.1%}  " + "  ".join(
            f"{arm}: " + "/".join(f"{report[name][arm][f'Under_{int(t)}_km']:.1%}" for t in THRESHOLDS_KM[:2]) for arm in ("reranker top-1", "VLM greedy")))
    (ROOT / f"evidence_test_{model}_{evidence}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / f"evidence_test_{model}_{evidence}_answers.json").write_text(json.dumps({world.queries[q]["image_id"]: answers[q] for q in members}) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("landmarks", "run"))
    parser.add_argument("--evidence", choices=tuple(INTRO), default="none")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="vlm")
    args = parser.parse_args(argv)
    if args.node == "landmarks":
        landmarks()
    else:
        run(args.evidence, server=args.server, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
