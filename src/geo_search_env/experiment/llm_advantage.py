# Where could the VLM beat the reranker? Photo-content slices of the benchmark eval halves and gazetteer coverage.
# Usage: set -a; . ./.env; set +a
#        .venv/bin/python -m geo_search_env.experiment.llm_advantage label    (Gemini labels, eval photos only)
#        .venv/bin/python -m geo_search_env.experiment.llm_advantage slices --answers <pivot_vlm_answers_...json>
#        .venv/bin/python -m geo_search_env.experiment.llm_advantage gazetteer

"""Photo-content slices for the eval halves, and whether a place name resolves to the photo's location offline.

`label` asks an API model (evaluation only, never training data) whether each eval photo has readable location text
and which named place it shows. `slices` compares a model's saved answers with the reranker per slice. `gazetteer`
checks whether those place names resolve, via GeoNames, to within 1 / 25 km of the truth: the upper bound for a
name-then-geocode answer format.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Sequence

import numpy as np

from .strategy_search import _haversine_km, _xyz, load_world
from .verifiers import reranker_ranking


ROOT = Path("artifacts/strategy_search")
LABELS = ROOT / "llm_advantage_labels.json"
GEONAMES = Path("/data/pinpoint/geonames/allCountries.txt")
LABEL_MODEL = "google/gemini-3.8-flash"
LABEL_PROMPT = """Describe what in this photo could identify where it was taken. Reply with only this JSON:
{"readable_text": true or false (legible text that hints at the location: signs, shop or street names, plates, banners),
 "named_place": "the most specific named place you can identify in the photo (landmark, venue, building, park, street), with its city and country, or null",
 "confidence": "high", "medium" or "low" (for named_place),
 "scene": one of "landmark", "urban", "building", "interior", "nature", "event", "other"}"""


def _label_one(api_key: str, path: str) -> dict[str, Any] | None:
    body = {
        "model": LABEL_MODEL, "temperature": 0.0, "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()}},
            {"type": "text", "text": LABEL_PROMPT},
        ]}],
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                text = json.loads(response.read())["choices"][0]["message"]["content"] or ""
            match = re.search(r"\{.*\}", text, re.S)
            return json.loads(match.group(0)) if match else None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            time.sleep(5 * 2**attempt)
    return None


def label(*, workers: int = 8) -> None:
    world = load_world()
    eval_queries = [q for q in world.queries if q["split"] == "eval"]
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    todo = [q for q in eval_queries if labels.get(q["image_id"]) is None]
    api_key = os.environ["OPENROUTER_API_KEY"]
    with ThreadPoolExecutor(workers) as pool:
        for done, (q, result) in enumerate(zip(todo, pool.map(lambda q: _label_one(api_key, q["path"]), todo)), start=1):
            labels[q["image_id"]] = result
            if done % 200 == 0 or done == len(todo):
                LABELS.write_text(json.dumps(labels) + "\n", encoding="utf-8")
                print(f"  labelled {done}/{len(todo)}", flush=True)
    print(f"unlabelled: {sum(labels.get(q['image_id']) is None for q in eval_queries)}/{len(eval_queries)}")


def _greedy_distances(world, answers: dict[str, dict], coords, valid, distance, ranking) -> tuple[list[int], np.ndarray, np.ndarray]:
    index = {q["image_id"]: i for i, q in enumerate(world.queries)}
    members, model_km, reranker_km = [], [], []
    for image_id, answer in answers.items():
        q = index[image_id]
        point = answer.get("greedy") or tuple(coords[q, np.flatnonzero(valid[q])[0]])
        members.append(q)
        model_km.append(_haversine_km(*world.query_latlon[q], np.asarray([point]))[0])
        reranker_km.append(distance[q, ranking[q, 0]])
    return members, np.asarray(model_km), np.asarray(reranker_km)


def slices(answers_path: Path) -> None:
    world = load_world()
    coords, valid, distance, ranking = reranker_ranking(world, ROOT)
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    answers = {k: v for k, v in json.loads(answers_path.read_text(encoding="utf-8")).items() if labels.get(k)}  # labelled photos only
    members, model_km, reranker_km = _greedy_distances(world, answers, coords, valid, distance, ranking)
    info = [labels.get(world.queries[q]["image_id"]) or {} for q in members]
    named = np.asarray([bool(x.get("named_place")) and x.get("confidence") == "high" for x in info])
    groups = {
        "all": np.ones(len(members), dtype=bool),
        "readable text": np.asarray([x.get("readable_text") is True for x in info]),
        "named place (high confidence)": named,
        "text or named place": np.asarray([x.get("readable_text") is True for x in info]) | named,
        "neither": ~(np.asarray([x.get("readable_text") is True for x in info]) | named),
        **{f"scene: {s}": np.asarray([x.get("scene") == s for x in info]) for s in ("landmark", "urban", "building", "interior", "nature", "event", "other")},
    }
    report: dict[str, Any] = {"answers": str(answers_path)}
    print(f"{'slice':32s} {'n':>5s}   {'<1 km: model / reranker / m-only / r-only':44s} {'<25 km: model / reranker / m-only / r-only'}")
    for name, mask in groups.items():
        entry: dict[str, Any] = {"n": int(mask.sum())}
        line = f"{name:32s} {mask.sum():5d}"
        for km in (1, 25):
            m, r = model_km[mask] < km, reranker_km[mask] < km
            entry[f"{km}km"] = {"model": float(m.mean()), "reranker": float(r.mean()), "model only": float((m & ~r).mean()), "reranker only": float((r & ~m).mean())}
            line += "   " + " / ".join(f"{v:5.1%}" for v in entry[f"{km}km"].values())
        report[name] = entry
        print(line)
    (ROOT / f"llm_advantage_slices_{answers_path.stem.removeprefix('pivot_vlm_answers_')}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"^the\s+", "", text)).strip()


def gazetteer() -> None:
    """Resolve each high-confidence place name with GeoNames; report hits within 1 / 25 km of the truth."""

    world = load_world()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    index = {q["image_id"]: i for i, q in enumerate(world.queries)}
    wanted: dict[str, list[tuple[str, int]]] = {}  # normalized first component -> [(image_id, query)]
    for image_id, x in labels.items():
        if x and x.get("named_place") and x.get("confidence") == "high":
            wanted.setdefault(_normalize(x["named_place"].split(",")[0]), []).append((image_id, index[image_id]))
    print(f"high-confidence named places: {sum(map(len, wanted.values()))}; scanning GeoNames", flush=True)
    matches: dict[str, list[tuple[float, float, int, str]]] = {name: [] for name in wanted}
    with GEONAMES.open("r", encoding="utf-8") as stream:
        for line in stream:
            f = line.rstrip("\n").split("\t")
            names = {_normalize(f[1]), _normalize(f[2])} | {_normalize(n) for n in f[3].split(",") if n}
            for name in names & matches.keys():
                matches[name].append((float(f[4]), float(f[5]), int(f[14] or 0), f[6] + "." + f[7]))
    rows = []
    for name, users in wanted.items():
        found = np.asarray([(lat, lon) for lat, lon, *_ in matches[name]]) if matches[name] else np.zeros((0, 2))
        for image_id, q in users:
            truth = world.query_latlon[q]
            place = labels[image_id]["named_place"]
            # Rule-based pick without the truth: prefer matches whose surroundings contain the stated city
            # (approximated by the most populous match; ties to the first), i.e. what a simple geocoder would return.
            best_km = float(_haversine_km(*truth, found).min()) if len(found) else np.inf
            pick = max(matches[name], key=lambda m: m[2]) if matches[name] else None
            pick_km = float(_haversine_km(*truth, np.asarray([pick[:2]]))[0]) if pick else np.inf
            rows.append({"image_id": image_id, "place": place, "n_matches": len(found), "oracle_km": best_km, "pick_km": pick_km})
    n = len(rows)
    oracle, pick = np.asarray([r["oracle_km"] for r in rows]), np.asarray([r["pick_km"] for r in rows])
    report = {
        "high-confidence named places": n,
        "any GeoNames match": float(np.mean([r["n_matches"] > 0 for r in rows])),
        **{f"oracle match <{km} km": float((oracle < km).mean()) for km in (1, 25)},
        **{f"most-populous match <{km} km": float((pick < km).mean()) for km in (1, 25)},
    }
    (ROOT / "llm_advantage_gazetteer.json").write_text(json.dumps({"summary": report, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("label", "slices", "gazetteer"))
    parser.add_argument("--answers", type=Path, default=ROOT / "pivot_vlm_answers_sft-34k-retrieval_sft-retrieval-prompt_t0.7_full-eval.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.node == "label":
        label(workers=args.workers)
    elif args.node == "slices":
        slices(args.answers)
    else:
        gazetteer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
