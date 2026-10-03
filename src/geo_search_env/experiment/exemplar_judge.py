# Can a VLM tell the right candidate from the wrong top-1 by comparing the query photo with an exemplar photo of each?
# Usage: .venv/bin/python -m geo_search_env.experiment.exemplar_judge pairs          (CPU, ~3 min)
#        .venv/bin/python -m geo_search_env.experiment.exemplar_judge judge          (vLLM on :8765, 2 images per prompt)
#        .venv/bin/python -m geo_search_env.experiment.exemplar_judge report

"""Screen on "choosing" photos (dev and val: the pool holds a candidate within 25 km, the reranker's top-1 is wrong).

pairs:  for the right candidate (the pool candidate nearest the truth) and the wrong top-1, up to EXEMPLARS database photos taken within
        1 km of it (not by the query's photographer), the most similar to the query first; the embedding similarity is kept.
judge:  the VLM sees the query photo and one exemplar and says whether they show the same place; the score is P(yes) / (P(yes) + P(no)).
report: how often the right candidate's best exemplar scores higher than the wrong top-1's, for the VLM and for the embedding similarity
        (the reference), with bootstrap intervals; also with only the single most similar exemplar.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import urllib.request
from typing import Any, Sequence

import numpy as np

from .evidence_ranker import load_set
from .query_evidence import ROOT
from .sft_data import MP16Images
from .stage1_eval import _bootstrap
from .strategy_search import EARTH_KM, MP16_EMBED, _xyz, load_world

EXEMPLARS = 2
EXEMPLAR_KM = 1.0
SIZE = 448
PROMPT = "Were these two photos taken at the same place? Answer yes or no."
PAIRS = ROOT / "exemplar_pairs.json"
SCORES = ROOT / "exemplar_scores.json"
TOPK = 8  # reranker ranks scored per photo in the full test
TOPK_PAIRS = ROOT / "exemplar_topk_pairs.json"
TOPK_SCORES = ROOT / "exemplar_topk_scores.json"
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _jpeg(data: bytes) -> str:
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((SIZE, SIZE))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=88)
    return base64.b64encode(out.getvalue()).decode()


def pairs() -> None:
    from scipy.spatial import cKDTree

    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    tree = None
    for tag in ("dev", "val"):
        s = load_set(tag)
        photos = s["photos"]
        world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos]) if tag == "dev" else load_world()
        if tree is None:
            tree = cKDTree(_xyz(world.mp16["latlon"]))
        excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8"))) if tag == "val" else set()
        chord = 2 * math.sin(EXEMPLAR_KM / EARTH_KM / 2)

        def exemplars(latlon: np.ndarray, q: np.ndarray, author: int) -> list[dict[str, Any]]:
            rows = [r for r in tree.query_ball_point(_xyz(latlon), chord) if world.mp16["author"][r] != author]
            if not rows:
                return []
            rows = np.sort(rows)
            emb = np.asarray(world.mp16["embeddings"][rows], dtype=np.float32)
            sims = emb @ q / np.linalg.norm(emb, axis=1)
            order = np.argsort(-sims)[:EXEMPLARS]
            return [{"id": ids[rows[i]], "sim": float(sims[i])} for i in order]

        for m, e in enumerate(photos):
            if e["image_id"] in excluded:
                continue
            d = s["distance"][m]
            top = [c for c in range(d.shape[0]) if s["valid"][m, c] and np.allclose(s["coords"][m, c], e["pool"][0], atol=1e-6)]
            right = [c for c in range(d.shape[0]) if s["valid"][m, c] and d[c] < 25]
            if not top or not right or d[top[0]] < 25:
                continue  # not a choosing photo
            best = min(right, key=lambda c: d[c])
            pos = e["index"] if tag == "val" else m
            q = world.query_embeddings[pos] / np.linalg.norm(world.query_embeddings[pos])
            author = int(world.query_author[pos])
            right_ex, wrong_ex = exemplars(s["coords"][m, best], q, author), exemplars(s["coords"][m, top[0]], q, author)
            if right_ex and wrong_ex:
                out.append({"tag": tag, "image_id": e["image_id"], "path": e.get("path"), "benchmark": e.get("benchmark", "mp16"),
                            "right_km": float(d[best]), "wrong_km": float(d[top[0]]), "right": right_ex, "wrong": wrong_ex})
        print(f"{tag}: choosing photos so far {sum(p['tag'] == tag for p in out)}", flush=True)
    PAIRS.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(out)} photos with exemplars for both candidates -> {PAIRS}")


def _p_same(server: str, query_b64: str, exemplar_b64: str) -> float | None:
    body = {
        "model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 12, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + query_b64}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + exemplar_b64}},
            {"type": "text", "text": PROMPT}]}],
    }
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            top = json.loads(response.read())["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception:
        return None
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else None


def judge(server: str) -> None:
    photos = json.loads(PAIRS.read_text(encoding="utf-8"))
    images = MP16Images()
    jobs = [(i, side, j) for i, p in enumerate(photos) for side in ("right", "wrong") for j in range(len(p[side]))]

    def run(job: tuple[int, str, int]) -> float | None:
        i, side, j = job
        p = photos[i]
        query = Path(p["path"]).read_bytes() if p["path"] else images.read(p["image_id"])
        return _p_same(server, _jpeg(query), _jpeg(images.read(p[side][j]["id"])))

    with ThreadPoolExecutor(32) as pool:
        scores = list(pool.map(run, jobs))
    for (i, side, j), score in zip(jobs, scores):
        photos[i][side][j]["p_same"] = score
    SCORES.write_text(json.dumps(photos) + "\n", encoding="utf-8")
    print(f"{len(jobs)} comparisons, {sum(s is None for s in scores)} failed -> {SCORES}")


def report() -> None:
    photos = json.loads(SCORES.read_text(encoding="utf-8"))

    def favour(p: dict[str, Any], key: str, k: int) -> float | None:
        r, w = [x[key] for x in p["right"][:k] if x.get(key) is not None], [x[key] for x in p["wrong"][:k] if x.get(key) is not None]
        if not r or not w:
            return None
        return 1.0 if max(r) > max(w) else 0.5 if max(r) == max(w) else 0.0

    print("Share of choosing photos where the RIGHT candidate's best exemplar scores higher than the WRONG top-1's (chance 50%)")
    print(f"{'subset':16s} {'n':>4s}   VLM, 1 exemplar   VLM, 2 exemplars   embedding sim (1)   embedding sim (2)")
    groups = {"dev": lambda p: p["tag"] == "dev", "val": lambda p: p["tag"] == "val", "all": lambda p: True,
              "within 1 km": lambda p: p["right_km"] < 1, "right 1-25 km": lambda p: 1 <= p["right_km"] < 25}
    summary = {}
    for name, keep in groups.items():
        chosen = [p for p in photos if keep(p)]
        cells = []
        for key, k in (("p_same", 1), ("p_same", 2), ("sim", 1), ("sim", 2)):
            values = np.asarray([v for v in (favour(p, key, k) for p in chosen) if v is not None])
            mean, lo, hi = _bootstrap(values) if len(values) else (float("nan"),) * 3
            cells.append(f"{mean:6.1%} [{lo:.0%},{hi:.0%}]")
            summary[f"{name} {key} {k}"] = [mean, lo, hi]
        print(f"{name:16s} {len(chosen):4d}   " + "   ".join(cells))
    scored = [x["p_same"] for p in photos for side in ("right", "wrong") for x in p[side] if x.get("p_same") is not None]
    print(f"\nP(same place) over all comparisons: median {np.median(scored):.2f}, share above 0.5: {np.mean(np.asarray(scored) > 0.5):.0%}")
    (ROOT / "exemplar_screen.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def topk_pairs() -> None:
    """For every dev and val photo, the best exemplar (within 1 km of the candidate, not by the query's photographer) of each of its top TOPK candidates."""

    from scipy.spatial import cKDTree

    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    tree = None
    chord = 2 * math.sin(EXEMPLAR_KM / EARTH_KM / 2)
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos]) if tag == "dev" else load_world()
        if tree is None:
            tree = cKDTree(_xyz(world.mp16["latlon"]))
        for m, e in enumerate(photos):
            pos = e["index"] if tag == "val" else m
            q = world.query_embeddings[pos] / np.linalg.norm(world.query_embeddings[pos])
            author = int(world.query_author[pos])
            found: list[str | None] = []
            for latlon in e["pool"][:TOPK]:
                rows = [r for r in tree.query_ball_point(_xyz(np.asarray(latlon)), chord) if world.mp16["author"][r] != author]
                if not rows:
                    found.append(None)
                    continue
                rows = np.sort(rows)
                emb = np.asarray(world.mp16["embeddings"][rows], dtype=np.float32)
                found.append(ids[rows[int(np.argmax(emb @ q / np.linalg.norm(emb, axis=1)))]])
            out.append({"tag": tag, "image_id": e["image_id"], "path": e.get("path"), "exemplars": found})
        print(f"{tag}: {sum(p['tag'] == tag for p in out)} photos; candidates with an exemplar "
              f"{np.mean([x is not None for p in out if p['tag'] == tag for x in p['exemplars']]):.0%}", flush=True)
    TOPK_PAIRS.write_text(json.dumps(out) + "\n", encoding="utf-8")


def topk_judge(server: str) -> None:
    photos = json.loads(TOPK_PAIRS.read_text(encoding="utf-8"))
    images = MP16Images()
    jobs = [(i, r) for i, p in enumerate(photos) for r, x in enumerate(p["exemplars"]) if x is not None]

    def run(job: tuple[int, int]) -> float | None:
        i, r = job
        p = photos[i]
        query = Path(p["path"]).read_bytes() if p["path"] else images.read(p["image_id"])
        return _p_same(server, _jpeg(query), _jpeg(images.read(p["exemplars"][r])))

    with ThreadPoolExecutor(32) as pool:
        scores = list(pool.map(run, jobs))
    for p in photos:
        p["p_same"] = [None] * len(p["exemplars"])
    for (i, r), score in zip(jobs, scores):
        photos[i]["p_same"][r] = score
    TOPK_SCORES.write_text(json.dumps(photos) + "\n", encoding="utf-8")
    print(f"{len(jobs)} comparisons, {sum(s is None for s in scores)} failed -> {TOPK_SCORES}")


def topk_report() -> None:
    """Top-1 accuracy of score = -rank + w * logit(P(same place)) over each photo's top candidates, w fitted on dev; shuffled control."""

    from .wiki_backend import _km

    scored = json.loads(TOPK_SCORES.read_text(encoding="utf-8"))
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    data: dict[str, dict[str, Any]] = {}
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        mine = [p for p in scored if p["tag"] == tag]
        keep = [m for m, e in enumerate(photos) if e["image_id"] not in excluded]
        dist = [_km(np.asarray(photos[m]["pool"][:TOPK]), *photos[m]["truth"]) for m in keep]
        logit = []
        for m in keep:
            raw = np.asarray([np.nan if x is None else x for x in mine[m]["p_same"]], dtype=np.float64)
            raw = np.where(np.isnan(raw), 0.5, np.clip(raw, 0.02, 0.98))  # no exemplar or failed call: neutral
            logit.append(np.log(raw / (1 - raw)))
        data[tag] = {"dist": dist, "logit": logit, "bench": np.asarray([photos[m].get("benchmark", "mp16") for m in keep])}

    def top1(tag: str, w: float, shuffle: bool = False) -> np.ndarray:
        d, lg = data[tag]["dist"], data[tag]["logit"]
        if shuffle:
            lg = [lg[i] for i in np.random.default_rng(0).permutation(len(lg))]
        fit = lambda l, n: np.pad(l[:n], (0, max(0, n - len(l))))  # photos have 8 or fewer candidates; a shuffled vector may be shorter or longer
        return np.asarray([x[int(np.argmax(-np.arange(len(x)) + w * fit(l, len(x))))] for x, l in zip(d, lg)])

    w = max(WEIGHTS, key=lambda v: ((top1("dev", v) < 25).mean(), -v))
    print(f"w fitted on dev for top-1 <25 km: {w:g}; candidates scored per photo: top {TOPK}\n")
    print(f"{'set':14s} {'n':>4s}  reranker top-1 <1/<25/<200 km    + exemplar judge (w={w:g})     change <1/<25/<200 km   <25 km 95% CI     shuffled-judge control <25 km")
    for tag, bench in (("dev", None), ("val", None), ("val", "im2gps3k"), ("val", "yfcc4k")):
        sel = np.ones(len(data[tag]["dist"]), bool) if bench is None else data[tag]["bench"] == bench
        base, with_j, shuf = top1(tag, 0.0)[sel], top1(tag, w)[sel], top1(tag, w, shuffle=True)[sel]
        line = f"{tag + (' ' + bench if bench else ''):14s} {int(sel.sum()):4d}  "
        line += " ".join(f"{(base < t).mean():6.1%}" for t in (1, 25, 200)) + "      " + " ".join(f"{(with_j < t).mean():6.1%}" for t in (1, 25, 200)) + "      "
        line += " ".join(f"{(with_j < t).mean() - (base < t).mean():+6.1%}" for t in (1, 25, 200))
        ci = _bootstrap((with_j < 25).astype(float) - (base < 25))
        print(line + f"    [{ci[1]:+.1%}, {ci[2]:+.1%}]    {(shuf < 25).mean() - (base < 25).mean():+.1%}")
    for v in WEIGHTS[1:]:
        print(f"  w={v:<4g} change <25 km: dev {(top1('dev', v) < 25).mean() - (top1('dev', 0.0) < 25).mean():+.1%}, val {(top1('val', v) < 25).mean() - (top1('val', 0.0) < 25).mean():+.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("pairs", "judge", "report", "topk-pairs", "topk-judge", "topk-report"))
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    {"pairs": pairs, "judge": lambda: judge(args.server), "report": report, "topk-pairs": topk_pairs,
     "topk-judge": lambda: topk_judge(args.server), "topk-report": topk_report}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
