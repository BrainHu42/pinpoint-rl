# Reader comparison (lesson 88): does a stronger open VLM produce the countries and place names the 4B misses, on the photos the pipeline gets
# wrong? Same prompts as observe.py (extract: clues + country guess; queries: searchable names; landmarks: a recognised place) on dev3k; every
# reader's phrases resolved with text_search (top cluster, photo and photographer excluded; tools.maps is too slow for the 1-hour budget). Earlier readers'
# outputs are cached (4b / 9b observe + queries, 9b / 27b = Qwen3.6-27B landmarks, lesson 59).
# Usage: NAME=38_27b MODEL=<snapshot dir> bash scripts/reader_compare.sh          (serves the model, runs the three prompts)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.reader_compare resolve --name 38_27b
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.reader_compare report --names 4b 9b 38_27b

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
TAG = "dev3k"
KM = 25.0
SOURCES = ("text", "place_names", "queries")  # landmark phrases are kept in phrases() but no longer scored (lesson 59)


def _rows() -> list[dict]:
    rows = [json.loads(line) for line in (OUT / "clue_grpo_data.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "dev"]
    areas = {o["image_id"]: o for o in json.loads((OUT / "tag_reader_dev3k_setft_a.json").read_text(encoding="utf-8"))}
    for r in rows:
        km = [a["km"] for a in areas[r["image_id"]]["areas"]]
        r["area_right"] = any(k < KM for k in km)
    return rows


def _load(kind: str, name: str) -> dict:
    path = OUT / f"{kind}_{TAG}_{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def phrases(name: str) -> dict[str, dict[str, list[str]]]:
    """Per photo, the reader's searchable phrases by source (missing prompts give no phrases)."""

    obs, qs, lm = _load("observe", name), _load("queries", name), _load("landmark", name)
    out: dict[str, dict[str, list[str]]] = {}
    for r in _rows():
        i = r["image_id"]
        p = (obs.get(i) or {}).get("parsed") or {}
        a = lm.get(i) or {}
        land = [a["landmark"]] if a.get("landmark") else []
        if land and a.get("city"):
            land.append(f"{a['landmark']} {a['city']}")
        out[i] = {"text": [t for t in p.get("text", []) if len(t.strip()) >= 4][:15], "place_names": p.get("place_names", [])[:10],
                  "queries": (qs.get(i) or [])[:5], "landmark": land}
    return out


def resolve(name: str, workers: int = 8) -> None:
    from .region_evidence import _owns
    from .text_search import TextSearch, haversine_km

    rows = {r["image_id"]: r for r in _rows()}
    owns = _owns([{"image_id": i} for i in rows])
    cache_path = OUT / f"reader_resolve_{TAG}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    jobs = sorted({f"{i}\t{q.strip()}" for i, src in phrases(name).items() for k in SOURCES for q in src[k] if q.strip()} - set(cache))
    print(f"{name}: {len(jobs)} new (photo, phrase) pairs, {len(cache)} cached", flush=True)
    ts = TextSearch()

    def run(job: str) -> dict:
        i, q = job.split("\t", 1)
        t = rows[i]["truth"]
        out = {}
        try:
            c = ts.search(q, exclude=owns.get(i)).clusters
            if c:
                out["text"] = float(haversine_km(t[0], t[1], np.asarray([c[0].lat]), np.asarray([c[0].lon]))[0])
        except Exception:  # noqa: BLE001
            pass
        return out

    with ThreadPoolExecutor(workers) as pool:
        for k, (job, res) in enumerate(zip(jobs, pool.map(run, jobs))):
            cache[job] = res
            if k % 2000 == 1999:
                print(f"  {k + 1} / {len(jobs)}", flush=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {cache_path}")


def report(names: Sequence[str]) -> None:
    from .nested_passk import _Names

    rows = _rows()
    names_map = _Names(np.asarray([r["truth_country"] for r in rows] + [r["pin_country"] for r in rows]),
                       np.asarray([r["truth_region"] for r in rows] + [r["pin_region"] for r in rows]))
    cache = json.loads((OUT / f"reader_resolve_{TAG}.json").read_text(encoding="utf-8"))
    top1_right = np.asarray([_km(r["truth"], r["top1"]) < KM for r in rows])
    area_right = np.asarray([r["area_right"] for r in rows])
    pin_c = np.asarray([r["pin_country"] == r["truth_country"] for r in rows])
    groups = {"all": np.ones(len(rows), bool), "top-1 wrong": ~top1_right, "no right area": ~area_right}
    print(f"dev3k {len(rows)} photos; retrained top-1 right {top1_right.mean():.1%}, some Pinpoint area right {area_right.mean():.1%}, "
          f"Pinpoint's country right {pin_c.mean():.1%}")
    for name in names:
        obs, ph = _load("observe", name), phrases(name)

        def hit(i: str, srcs: Sequence[str]) -> bool:
            return any(min(cache.get(f"{i}\t{q.strip()}", {}).values(), default=1e9) < KM for s in srcs for q in ph[i][s])

        country = [names_map.country(((obs.get(r["image_id"]) or {}).get("parsed") or {}).get("country")) for r in rows]
        c_right = np.asarray([c is not None and c == r["truth_country"] for c, r in zip(country, rows)])
        c_moved = np.asarray([c is not None and c != r["pin_country"] for c, r in zip(country, rows)])
        print(f"\n{name}: country answered {np.mean([c is not None for c in country]):.0%} (when observe ran), right {c_right.mean():.1%}; "
              f"differs from Pinpoint's country {c_moved.mean():.1%} of photos, right there {c_right[c_moved].mean() if c_moved.any() else 0:.1%} "
              f"(fixes {(c_right & ~pin_c).sum()}, breaks {(c_moved & pin_c).sum()})")
        for g, m in groups.items():
            parts = [f"{s} {np.mean([hit(r['image_id'], (s,)) for r, k in zip(rows, m) if k]):.1%}" for s in SOURCES]
            anyhit = np.asarray([hit(r["image_id"], SOURCES) for r in rows])
            has = np.asarray([any(ph[r["image_id"]][s] for s in SOURCES) for r in rows])
            print(f"  {g:13s} n={m.sum():4d}: any phrase within {KM:.0f} km {anyhit[m].mean():.1%} (phrases on {has[m].mean():.0%}); by source: "
                  + ", ".join(parts) + f"; country right {c_right[m].mean():.1%}")


def _km(a: Sequence[float], b: Sequence[float]) -> float:
    from .text_search import haversine_km

    return float(haversine_km(a[0], a[1], np.asarray([b[0]]), np.asarray([b[1]]))[0])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("resolve", "report"))
    parser.add_argument("--name", default="38_27b")
    parser.add_argument("--names", nargs="+", default=["4b", "9b", "38_27b"])
    args = parser.parse_args(argv)
    if args.cmd == "resolve":
        resolve(args.name)
    else:
        report(args.names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
