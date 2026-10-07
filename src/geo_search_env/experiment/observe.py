# AGENT_DIRECTION.md "observe" step: a VLM reads each photo into structured clues (verbatim text, language / script, driving side, visible
# features from a fixed vocabulary that maps to GeoNames feature codes, readable place names, a country guess), for the constraint tools.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.observe extract --tag dev --name base   (VLM served at :8765)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.observe report --tag dev --name base

"""extract: greedy JSON answer per photo of a set (dev / val MP16 photos, full = benchmark eval halves); saved to artifacts/agent/observe_{tag}_{name}.json.
report:  parse rate, how often each field is filled, feature frequencies, and on the fixable photos (agent_targets.py) how often text / place names /
         features are present."""

from __future__ import annotations

import argparse
import base64
import collections
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.request
from typing import Sequence

from .name_score import _read
from .query_evidence import ROOT
from .sft_data import MP16Images

OUT = Path("artifacts/agent")

# visible feature -> GeoNames feature codes (class.code) it can be checked against
VOCAB: dict[str, tuple[str, ...]] = {
    "church": ("S.CH", "S.CVNT", "S.MSTY"), "mosque": ("S.MSQE",), "temple": ("S.TMPL", "S.PGDA"), "shrine": ("S.SHRN",),
    "synagogue": ("S.SYG",), "castle": ("S.CSTL",), "palace": ("S.PAL",), "fort": ("S.FT",), "tower": ("S.TOWR",),
    "lighthouse": ("S.LTHSE",), "bridge": ("S.BDG",), "dam": ("S.DAM",), "railway station": ("S.RSTN", "S.RSTP"),
    "railway": ("R.RR", "S.RSTN", "S.RSTP"), "airport": ("S.AIRP",), "harbour": ("H.HBR", "L.PRT", "S.MAR", "S.PIER", "S.BRKW"),
    "stadium": ("S.STDM",), "university": ("S.UNIV",), "museum": ("S.MUS",), "monument": ("S.MNMT", "S.ARCH"), "ruins": ("S.RUIN",),
    "square": ("S.SQR",), "market": ("S.MKT",), "theatre": ("S.THTR", "S.OPRA"), "cemetery": ("S.CMTY",), "garden": ("S.GDN",),
    "zoo": ("S.ZOO",), "observatory": ("S.OBS",), "windmill": ("S.MLWND",), "cave": ("S.CAVE",), "park": ("L.PRK",),
    "river": ("H.STM",), "canal": ("H.CNL",), "lake": ("H.LK", "H.RSV"), "waterfall": ("H.FLLS",), "bay": ("H.BAY",),
    "glacier": ("H.GLCR",), "beach": ("T.BCH",), "cliff": ("T.CLF", "T.CAPE"), "island": ("T.ISL",), "mountain": ("T.MT", "T.PK"),
    "volcano": ("T.VLC",), "hill": ("T.HLL",), "valley": ("T.VAL", "T.GRGE"), "desert": ("T.DSRT", "T.DUNE"), "mountain pass": ("T.PASS",),
    "forest": ("V.FRST",), "vineyard": ("V.VIN",), "tea plantation": ("S.ESTT",), "tunnel": ("R.TNL",),
}

PROMPTS = {
    "v1": """You are a professional geolocator. Study this photo and report the clues in it. Answer with JSON only, in this form:
{"text": ["every piece of readable text, verbatim, e.g. signs, shop names, plates"],
 "language": "language of the text, or null",
 "script": "writing system (Latin, Cyrillic, Arabic, Han, Hangul, Kana, Devanagari, Thai, Greek, Hebrew, ...), or null",
 "driving_side": "left, right or unknown",
 "features": ["visible features, only from this list: FEATURES"],
 "place_names": ["names of specific places you can read or recognise (towns, streets, landmarks, businesses)"],
 "country": "your best guess of the country, or null"}
Report only what is actually visible; use empty lists and null when there is nothing.""",
    # v2: short fields first, text last and capped, so a repetition loop in the text can't cost the other fields
    "v2": """You are a professional geolocator. Study this photo and report the clues in it. Answer with JSON only, in this form:
{"country": "your best guess of the country, or null",
 "driving_side": "left, right or unknown",
 "script": "writing system of any text (Latin, Cyrillic, Arabic, Han, Hangul, Kana, Devanagari, Thai, Greek, Hebrew, ...), or null",
 "language": "language of the text, or null",
 "features": ["visible features, only from this list: FEATURES"],
 "place_names": ["names of specific places you can read or recognise (towns, streets, landmarks, businesses)"],
 "text": ["the most informative readable text, verbatim (signs, shop names, plates); at most 15 items, no repeats"]}
Report only what is actually visible; use empty lists and null when there is nothing.""",
}
PROMPT_VERSION = "v2"


def prompt(version: str = PROMPT_VERSION) -> str:
    return PROMPTS[version].replace("FEATURES", ", ".join(VOCAB))


def _salvage(answer: str) -> dict | None:
    """Field by field from broken JSON (e.g. a text list cut off by a repetition loop): scalars and the items of every list, closed or not."""

    obj: dict = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*(\[|"((?:[^"\\]|\\.)*)"|null)', answer):
        key = m.group(1)
        if m.group(2) == "[":
            end = answer.find("]", m.end())
            body = answer[m.end(): end if end >= 0 else len(answer)]
            obj[key] = list(dict.fromkeys(re.findall(r'"((?:[^"\\]|\\.)*)"', body)))
        else:
            obj[key] = m.group(3)
    return obj or None


def parse(answer: str | None) -> dict | None:
    if not answer:
        return None
    match = re.search(r"\{.*\}", answer, re.S)
    try:
        obj = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        obj = _salvage(answer)
    if not isinstance(obj, dict):
        return None
    lists = {k: list(dict.fromkeys(str(x) for x in obj.get(k) or [] if isinstance(x, (str, int, float)) and str(x).strip()))
             for k in ("text", "features", "place_names")}
    lists["features"] = [f.lower().strip() for f in lists["features"] if f.lower().strip() in VOCAB]
    scalar = {k: (str(obj[k]).strip() if obj.get(k) not in (None, "", "null") else None) for k in ("language", "script", "driving_side", "country")}
    return lists | scalar


def extract(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 48) -> None:
    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    images = MP16Images()
    text = prompt()

    def run(e: dict) -> str | None:
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 400, "repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_read(e, images)).decode()}},
                    {"type": "text", "text": text}]}]}
        request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"]
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(run, photos))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"observe_{tag}_{name}.json"
    path.write_text(json.dumps({e["image_id"]: {"raw": a, "parsed": parse(a)} for e, a in zip(photos, answers)}) + "\n", encoding="utf-8")
    print(f"{tag}: {len(answers)} answers, {sum(a is None for a in answers)} failed, {sum(parse(a) is None for a in answers)} unparsed -> {path}")


CROP_PROMPT = ('Read all text visible in this image crop (signs, shop names, plates, posters). Answer with JSON only: {"text": ["verbatim text", ...]}, '
               'at most 10 items, no repeats; {"text": []} if there is none.')


def crops(tag: str, base: str, out: str, server: str = "http://127.0.0.1:8765", workers: int = 48) -> None:
    """Text from 4 overlapping tiles per photo (each 60% of the width and height), merged into the `base` clues -> observe_{tag}_{out}.json."""

    import io

    from PIL import Image

    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    clues = json.loads((OUT / f"observe_{tag}_{base}.json").read_text(encoding="utf-8"))
    images = MP16Images()

    def tiles(e: dict) -> list[str]:
        im = Image.open(io.BytesIO(_read(e, images))).convert("RGB")
        w, h = im.size
        out = []
        for x0, y0 in ((0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)):
            buf = io.BytesIO()
            im.crop((int(x0 * w), int(y0 * h), int((x0 + 0.6) * w), int((y0 + 0.6) * h))).save(buf, "JPEG", quality=92)
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out

    def run(job: tuple[int, str]) -> str | None:
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 200, "repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + job[1]}},
                                                          {"type": "text", "text": CROP_PROMPT}]}]}
        request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"]
        except Exception:
            return None

    jobs = [(i, t) for i, e in enumerate(photos) for t in tiles(e)]
    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(run, jobs))
    extra: dict[int, list[str]] = {}
    for (i, _), a in zip(jobs, answers):
        extra.setdefault(i, []).extend((parse(a) or {}).get("text") or [])
    merged = {}
    for i, e in enumerate(photos):
        c = clues[e["image_id"]]
        parsed = dict(c["parsed"] or {})
        parsed["text"] = list(dict.fromkeys((parsed.get("text") or []) + extra.get(i, [])))
        merged[e["image_id"]] = {"raw": c["raw"], "parsed": parsed}
    (OUT / f"observe_{tag}_{out}.json").write_text(json.dumps(merged) + "\n", encoding="utf-8")
    added = sum(len(set(extra.get(i, [])) - set((clues[e["image_id"]]["parsed"] or {}).get("text") or [])) > 0 for i, e in enumerate(photos))
    print(f"{tag}: {len(jobs)} crops, {sum(a is None for a in answers)} failed; new text on {added} photos -> {OUT / f'observe_{tag}_{out}.json'}")


LANDMARK_PROMPT = ('Does this photo show a specific, recognisable landmark or named place (a particular building, monument, castle, church, museum, '
                   'bridge, square, natural landmark)? Answer with JSON only: {"landmark": "its name, or null", "city": "city or null", '
                   '"region": "state / province / region, or null", "country": "country or null", "confidence": "high, medium or low"}. '
                   'Use null for landmark if it is a generic scene.')


def landmarks(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 48) -> None:
    """Per photo: the landmark the VLM recognises, with city / region / country -> artifacts/agent/landmark_{tag}_{name}.json."""

    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    images = MP16Images()

    def run(e: dict) -> dict | None:
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 150, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_read(e, images)).decode()}},
                    {"type": "text", "text": LANDMARK_PROMPT}]}]}
        request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = json.loads(response.read())["choices"][0]["message"]["content"]
        except Exception:
            return None
        obj = _salvage(raw) or {}
        clean = {k: (str(obj[k]).strip() if obj.get(k) not in (None, "", "null", "None") else None) for k in ("landmark", "city", "region", "country", "confidence")}
        return clean | {"raw": raw}

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(run, photos))
    path = OUT / f"landmark_{tag}_{name}.json"
    path.write_text(json.dumps({e["image_id"]: a for e, a in zip(photos, answers)}) + "\n", encoding="utf-8")
    named = sum(bool(a and a["landmark"]) for a in answers)
    print(f"{tag}: {len(answers)} answers, {sum(a is None for a in answers)} failed, landmark named on {named} -> {path}")


def report(tag: str, name: str) -> None:
    obs = json.loads((OUT / f"observe_{tag}_{name}.json").read_text(encoding="utf-8"))
    targets = {r["image_id"]: r for r in json.loads((OUT / f"targets_{tag}.json").read_text(encoding="utf-8"))}
    parsed = {k: v["parsed"] for k, v in obs.items()}
    ok = {k: v for k, v in parsed.items() if v}
    print(f"{tag} {name}: {len(ok)} / {len(parsed)} parsed")
    for label, keep in (("all", lambda r: True), ("fixable", lambda r: r["fixable"]), ("reranker right", lambda r: r["top1_km"] < 25)):
        rows = [v for k, v in ok.items() if keep(targets[k])]
        if not rows:
            continue
        has = lambda f: sum(bool(v[f]) for v in rows) / len(rows)
        side = collections.Counter(v["driving_side"] for v in rows)
        print(f"  {label:15s} n={len(rows)}: text {has('text'):.0%}, place names {has('place_names'):.0%}, features {has('features'):.0%} "
              f"(mean {sum(len(v['features']) for v in rows) / len(rows):.1f}), language {has('language'):.0%}, driving side known "
              f"{1 - (side.get('unknown', 0) + side.get(None, 0)) / len(rows):.0%}")
    freq = collections.Counter(f for v in ok.values() for f in v["features"])
    print("  features: " + ", ".join(f"{f} {n}" for f, n in freq.most_common(25)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("extract", "crops", "landmarks", "report"))
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--name", default="base")
    parser.add_argument("--base", default="9b", help="crops: the clues the crop text is merged into")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    if args.node == "extract":
        extract(args.tag, args.name, args.server)
    elif args.node == "crops":
        crops(args.tag, args.base, args.name, args.server)
    elif args.node == "landmarks":
        landmarks(args.tag, args.name, args.server)
    else:
        report(args.tag, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
