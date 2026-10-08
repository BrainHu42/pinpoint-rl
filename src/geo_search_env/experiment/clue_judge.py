# Can a VLM tell a clue that belongs to the photo's place from one that refers to somewhere else? (TEXT_SEARCH_PLAN.md, after the region-evidence
# eye check: text read in a photo often names elsewhere, a brand, a shirt, a poster, and recalled landmark names are often look-alikes.) For every
# specific text-search result of a photo (region_evidence), the VLM sees the photo, the clue and where the search points, and answers Yes / No;
# P(Yes) then filters the clues. dev sets only.
# Usage: MODEL=... NAME=4b bash scripts/clue_judge.sh                                  (serves the VLM, then `judge`)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_judge report --tag dev3k --name 4b
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_judge picks --tag dev3k --name 4b --threshold 0.5   (for pinpoint_constrained)

"""judge:  P(Yes) for each region_evidence result with a top place, a labelled truth and cluster share >= MIN_SHARE (the specific ones)
        -> artifacts/agent/clue_judge_{tag}_{name}.json.
report: by P(Yes) threshold, on (photo, region) pairs that differ from Pinpoint's top-1 region: clue right vs Pinpoint right, overall and by
        source; the threshold is also chosen on one half of the photos and counted on the other (2 folds) so the choice isn't fitted to the count.
picks:  per photo, the accepted clue with the highest P(Yes) whose region differs from Pinpoint's -> text_search_picks_{tag}_{name}.json, the
        format pinpoint_constrained reads (--picks)."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
MIN_SHARE = 0.3
PROMPT = """A geolocation tool found the name "{query}" for this photo ({how}). Searching for it points to {region}, {country}.

Is this name evidence of where THIS photo was taken? Answer Yes only if the photo itself ties the name to that place: it is written on a shop front, \
street or station sign, building, plaque or local notice, or it names a unique landmark you can clearly recognise in the photo. Answer No if the name \
could refer to somewhere else (a brand, product, logo, clothing, a poster or advert for another place, a person's name, generic words) or the place \
is only a similar-looking one.

Answer with Yes or No."""
HOW = {"text 9b": "text read in the photo", "place name 9b": "a place name read in the photo",
       "landmark 27b": "a landmark name suggested from the photo's appearance", "landmark 9b": "a landmark name suggested from the photo's appearance",
       "query 9b": "a place name suggested from the photo"}


def _items(tag: str) -> list[dict[str, Any]]:
    rows = json.loads((OUT / f"region_evidence_{tag}.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    return [r for r in rows if "region" in r and r["share"] >= MIN_SHARE and labels[r["image_id"]]["truth"].get("region")]


def judge(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from .name_score import _post
    from .sft_data import MP16Images

    items = _items(tag)
    images = MP16Images()
    cache: dict[str, str] = {}

    def ask(r: dict) -> float | None:
        if r["image_id"] not in cache:
            cache[r["image_id"]] = base64.b64encode(images.read(r["image_id"])).decode()
        text = PROMPT.format(query=r["query"], how=HOW.get(r["source"], "a name found for the photo"), region=r["region"][1] or "?", country=r["region"][0] or "?")
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + cache[r["image_id"]]}},
                                                          {"type": "text", "text": text}]}]}
        try:
            top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception as error:  # noqa: BLE001 - a failed request counts as unjudged
            print(f"  failed: {error}", flush=True)
            return None
        p = {"yes": 0.0, "no": 0.0}
        for t in top:
            key = t["token"].strip().lower()
            if key in p:
                p[key] += math.exp(t["logprob"])
        return p["yes"] / (p["yes"] + p["no"]) if p["yes"] + p["no"] > 0 else None

    with ThreadPoolExecutor(workers) as pool:
        scores = list(pool.map(ask, items))
    out = [{"image_id": r["image_id"], "query": r["query"], "p_yes": s} for r, s in zip(items, scores)]
    path = OUT / f"clue_judge_{tag}_{name}.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{tag} / {name}: {len(out)} clues judged, {sum(s is None for s in scores)} failed -> {path}")


def _joined(tag: str, name: str) -> list[dict[str, Any]]:
    from .agent_targets import same_region

    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    scores = {(s["image_id"], s["query"]): s["p_yes"] for s in json.loads((OUT / f"clue_judge_{tag}_{name}.json").read_text(encoding="utf-8"))}
    reg = lambda r: {"country": r[0], "region": r[1]}
    out = []
    for r in _items(tag):
        p = scores.get((r["image_id"], r["query"]))
        if p is None:
            continue
        lab = labels[r["image_id"]]
        truth, pin = lab["truth"], lab["pool"][0]
        out.append(r | {"p_yes": p, "right": same_region(reg(r["region"]), truth), "pin_right": same_region(pin, truth),
                        "differs": not same_region(reg(r["region"]), pin)})
    return out


def _count(rows: list[dict], threshold: float) -> tuple[int, int, int]:
    """(pairs, clue right, Pinpoint right) over (photo, region) pairs differing from Pinpoint, clues with P(Yes) >= threshold."""

    pairs = {}
    for r in rows:
        if r["differs"] and r["p_yes"] >= threshold:
            pairs.setdefault((r["image_id"], tuple(r["region"])), r)
    return len(pairs), sum(r["right"] for r in pairs.values()), sum(r["pin_right"] for r in pairs.values())


def report(tag: str, name: str) -> None:
    rows = _joined(tag, name)
    print(f"{tag} / {name}: {len(rows)} judged specific clues; P(Yes) >= 0.5 on {np.mean([r['p_yes'] >= 0.5 for r in rows]):.0%}")
    print("  clues accepted at P(Yes) >= t, on (photo, region) pairs that differ from Pinpoint's region: clue right / Pinpoint right")
    thresholds = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
    for t in thresholds:
        n, fix, brk = _count(rows, t)
        print(f"    t = {t:.2f}: pairs {n:5d}, clue right {fix:4d}, Pinpoint right {brk:4d}")
    for src in sorted({r["source"] for r in rows}):
        sel = [r for r in rows if r["source"] == src]
        line = ", ".join("t={:.1f}: {}/{}".format(t, *_count(sel, t)[1:]) for t in (0.0, 0.5, 0.9))
        print(f"    {src:14s} {line}")
    ids = sorted({r["image_id"] for r in rows})
    fold = {i: k % 2 for k, i in enumerate(np.random.default_rng(0).permutation(ids))}
    total = [0, 0]
    for f in (0, 1):
        fit = [r for r in rows if fold[r["image_id"]] != f]
        held = [r for r in rows if fold[r["image_id"]] == f]
        best = max(thresholds, key=lambda t: _count(fit, t)[1] - _count(fit, t)[2])
        _, fix, brk = _count(held, best)
        total[0] += fix
        total[1] += brk
        print(f"  fold {f}: threshold chosen on the other half {best:.2f} -> clue right {fix}, Pinpoint right {brk}")
    print(f"  held-out total: clue right {total[0]}, Pinpoint right {total[1]}")


def picks(tag: str, name: str, threshold: float) -> None:

    rows = [r for r in _joined(tag, name) if r["differs"] and r["p_yes"] >= threshold]
    best: dict[str, dict] = {}
    for r in rows:
        if r["image_id"] not in best or r["p_yes"] > best[r["image_id"]]["p_yes"]:
            best[r["image_id"]] = r
    out = {k: {"lat": r["lat"], "lon": r["lon"], "weight": r["weight"], "share": r["share"], "query": r["query"], "p_yes": r["p_yes"]} for k, r in best.items()}
    path = OUT / f"text_search_picks_{tag}_{name}.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(out)} photos with an accepted clue (P(Yes) >= {threshold}) -> {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("judge", "report", "picks"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="4b")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.cmd == "judge":
        judge(args.tag, args.name)
    elif args.cmd == "report":
        report(args.tag, args.name)
    else:
        picks(args.tag, args.name, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
