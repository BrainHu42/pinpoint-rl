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
PAIR_K = 4  # candidates compared pairwise (12 ordered comparisons per photo)
PAIR_PROMPT = "Which of photos 2 and 3 was taken at the same place as photo 1? Answer 2 or 3."


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


def _scores_path(name: str) -> Path:
    """Scores of the zero-shot 9B keep their original file name; other judges are stored by name."""

    return TOPK_SCORES if name == "zeroshot-9b" else ROOT / f"exemplar_topk_scores_{name}.json"


def topk_judge(server: str, name: str) -> None:
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
    _scores_path(name).write_text(json.dumps(photos) + "\n", encoding="utf-8")
    print(f"{len(jobs)} comparisons, {sum(s is None for s in scores)} failed -> {_scores_path(name)}")


def _load_topk(name: str) -> list[dict[str, Any]]:
    """Per dev / val photo (placeholders dropped): its top candidates' distances to the truth in reranker order and the judge's P(same) (NaN: none)."""

    from .wiki_backend import _km

    scored = json.loads(_scores_path(name).read_text(encoding="utf-8"))
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    out = []
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        mine = [p for p in scored if p["tag"] == tag]
        for m, e in enumerate(photos):
            if e["image_id"] in excluded:
                continue
            d = _km(np.asarray(e["pool"][:TOPK]), *e["truth"])
            p = np.asarray([np.nan if x is None else x for x in mine[m]["p_same"]][: len(d)], dtype=np.float64)
            out.append({"tag": tag, "bench": e.get("benchmark", "mp16"), "dist": d, "p": p})
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.where(np.isnan(p), 0.5, np.clip(p, 0.02, 0.98))  # no exemplar or a failed call: neutral
    return np.log(p / (1 - p))


def topk_report(name: str) -> None:
    """Top-1 accuracy of score = -rank + w * logit(P(same place)) over each photo's top candidates, w fitted on dev; shuffled control."""

    photos = _load_topk(name)
    data = {tag: {"dist": [p["dist"] for p in photos if p["tag"] == tag], "logit": [_logit(p["p"]) for p in photos if p["tag"] == tag],
                  "bench": np.asarray([p["bench"] for p in photos if p["tag"] == tag])} for tag in ("dev", "val")}

    def top1(tag: str, w: float, shuffle: bool = False) -> np.ndarray:
        d, lg = data[tag]["dist"], data[tag]["logit"]
        if shuffle:
            lg = [lg[i] for i in np.random.default_rng(0).permutation(len(lg))]
        fit = lambda l, n: np.pad(l[:n], (0, max(0, n - len(l))))  # photos have 8 or fewer candidates; a shuffled vector may be shorter or longer
        return np.asarray([x[int(np.argmax(-np.arange(len(x)) + w * fit(l, len(x))))] for x, l in zip(d, lg)])

    w = max(WEIGHTS, key=lambda v: ((top1("dev", v) < 25).mean(), -v))
    print(f"judge {name}: w fitted on dev for top-1 <25 km: {w:g}; candidates scored per photo: top {TOPK}\n")
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


def topk_diagnose(name: str) -> None:
    """Candidate-level separation and within-photo ranking of the judge, and how often combining flips the top-1 right or wrong."""

    photos = _load_topk(name)

    def auc(pos: list[float], neg: list[float]) -> float:
        a, b = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
        return float((a > b).mean() + 0.5 * (a == b).mean())

    print(f"judge {name}")
    for tag in ("dev", "val"):
        mine = [p for p in photos if p["tag"] == tag]
        pick = lambda lo, hi: [x for p in mine for x, k in zip(p["p"], p["dist"]) if lo <= k < hi and not np.isnan(x)]
        near, mid, far = pick(0, 1), pick(1, 25), pick(25, 1e9)
        print(f"{tag}: mean P(same) for candidates <1 km from the truth {np.mean(near):.2f} (n={len(near)}), 1-25 km {np.mean(mid):.2f} (n={len(mid)}), >=25 km {np.mean(far):.2f} (n={len(far)})")
        print(f"     candidate-level AUC: <1 km vs >=25 km {auc(near, far):.3f}; 1-25 km vs >=25 km {auc(mid, far):.3f}")
        with_near = [p for p in mine if (p["dist"] < 1).any() and not np.isnan(p["p"]).all()]
        print(f"     photos with a candidate within 1 km among the top {TOPK} (n={len(with_near)}): the judge's top pick is within 1 km in "
              f"{np.mean([p['dist'][int(np.nanargmax(p['p']))] < 1 for p in with_near]):.0%}, the reranker top-1 in {np.mean([p['dist'][0] < 1 for p in with_near]):.0%}")
        for w in (0.5, 1.0, 2.0):
            fixed = broke = 0
            for p in mine:
                pick_i = int(np.argmax(-np.arange(len(p["dist"])) + w * _logit(p["p"])))
                fixed += (p["dist"][pick_i] < 25) and not (p["dist"][0] < 25)
                broke += (p["dist"][0] < 25) and not (p["dist"][pick_i] < 25)
            print(f"     w={w}: top-1 flips to a right answer {fixed}, away from a right answer {broke}")


def topk_cv(name: str) -> None:
    """Cross-validated learned combiner of the reranker rank and judge features over dev + val (5 folds, 3 seeds): an upper bound for the scalar rule."""

    import torch

    photos = _load_topk(name)
    D, F, tags = [], [], np.asarray([p["tag"] for p in photos])
    for p in photos:
        lg, n = _logit(p["p"]), len(p["dist"])
        rank = np.arange(n, dtype=float)
        order = (-lg).argsort().argsort().astype(float)  # the judge's rank within the photo
        f = np.stack([rank, (rank == 0).astype(float), (rank < 3).astype(float), lg, lg * rank, lg - lg.max(), order, (order == 0).astype(float), lg * (rank == 0)], axis=1)
        D.append(np.pad(p["dist"], (0, TOPK - n), constant_values=1e5)); F.append(np.pad(f, ((0, TOPK - n), (0, 0))))
    D, F = np.asarray(D), np.asarray(F, dtype=np.float32)
    valid = D < 9e4
    reward = np.where(valid, (D < 25) * 1.0 + (D < 1) * 0.5, 0.0).astype(np.float32)
    mean, std = F[valid].mean(0), F[valid].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(valid), torch.as_tensor(reward)

    def fit(idx: np.ndarray, seed: int):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
        idx_t = torch.as_tensor(idx)
        for _ in range(300):
            loss = -(torch.softmax(net(X[idx_t]).squeeze(-1).masked_fill(~M[idx_t], float("-inf")), -1) * R[idx_t]).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return net

    folds = np.array_split(np.random.default_rng(0).permutation(len(D)), 5)
    hits = {t: np.zeros(len(D)) for t in (1.0, 25.0, 200.0)}
    for seed in range(3):
        for f in folds:
            net = fit(np.setdiff1d(np.arange(len(D)), f), seed)
            with torch.no_grad():
                pick = net(X[f]).squeeze(-1).masked_fill(~M[f], float("-inf")).argmax(1).numpy()
            for t in hits:
                hits[t][f] += (D[f, pick] < t) / 3
    base = {t: (D[:, 0] < t).astype(float) for t in hits}
    print(f"judge {name}: learned combiner of rank + judge features, 5-fold CV, 3 seeds, {len(D)} photos, vs reranker top-1")
    for label, sel in (("all", np.ones(len(D), bool)), ("dev", tags == "dev"), ("val", tags == "val")):
        ci = _bootstrap(hits[25.0][sel] - base[25.0][sel])
        print(f"  {label:4s} n={int(sel.sum()):4d}  reranker " + " ".join(f"{base[t][sel].mean():6.1%}" for t in hits) + "   learned " + " ".join(f"{hits[t][sel].mean():6.1%}" for t in hits)
              + f"   change <25 km {ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]")


def _p_third(server: str, query_b64: str, second_b64: str, third_b64: str) -> float | None:
    """P(the third photo is the match) = P("3") / (P("2") + P("3")) for the pairwise comparator."""

    body = {
        "model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 12, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}} for b64 in (query_b64, second_b64, third_b64)
        ] + [{"type": "text", "text": PAIR_PROMPT}]}],
    }
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            top = json.loads(response.read())["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception:
        return None
    two = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == "2")
    three = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == "3")
    return three / (two + three) if two + three > 0 else None


def pair_judge(server: str, name: str) -> None:
    """Every ordered pair among each photo's top PAIR_K candidates (those with an exemplar): s[i][j] = P(candidate i matches) with i shown third and j second."""

    photos = json.loads(TOPK_PAIRS.read_text(encoding="utf-8"))
    images = MP16Images()
    jobs = [(m, i, j) for m, p in enumerate(photos) for i in range(PAIR_K) for j in range(PAIR_K)
            if i != j and i < len(p["exemplars"]) and j < len(p["exemplars"]) and p["exemplars"][i] and p["exemplars"][j]]

    def run(job: tuple[int, int, int]) -> float | None:
        m, i, j = job
        p = photos[m]
        query = Path(p["path"]).read_bytes() if p["path"] else images.read(p["image_id"])
        return _p_third(server, _jpeg(query), _jpeg(images.read(p["exemplars"][j])), _jpeg(images.read(p["exemplars"][i])))

    with ThreadPoolExecutor(32) as pool:
        scores = list(pool.map(run, jobs))
    for p in photos:
        p["s"] = [[None] * PAIR_K for _ in range(PAIR_K)]
    for (m, i, j), score in zip(jobs, scores):
        photos[m]["s"][i][j] = score
    _pair_path(name).write_text(json.dumps(photos) + "\n", encoding="utf-8")
    print(f"{len(jobs)} ordered comparisons, {sum(s is None for s in scores)} failed -> {_pair_path(name)}")


def _pair_path(name: str) -> Path:
    return ROOT / f"exemplar_pair_scores_{name}.json"


def _borda(s: list[list[float | None]], n: int) -> np.ndarray:
    """Per candidate: the mean over the others of P(it beats the other), each pair judged in both orders (missing -> 0.5)."""

    out = np.full(n, 0.5)
    for i in range(n):
        wins = []
        for j in range(n):
            if i == j:
                continue
            a, b = s[i][j], s[j][i]
            wins.append(0.5 * ((0.5 if a is None else a) + 1 - (0.5 if b is None else b)))
        out[i] = np.mean(wins) if wins else 0.5
    return out


def pair_report(name: str) -> None:
    """Pairwise accuracy on choosing photos, and top-1 of -rank + w * logit(Borda score) over the top PAIR_K candidates (w fitted on dev), with a shuffled control."""

    from .wiki_backend import _km

    scored = json.loads(_pair_path(name).read_text(encoding="utf-8"))
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    data: dict[str, dict[str, list]] = {"dev": {"dist": [], "borda": [], "bench": []}, "val": {"dist": [], "borda": [], "bench": []}}
    right_vs_wrong: dict[str, list[float]] = {"dev": [], "val": []}
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        mine = [p for p in scored if p["tag"] == tag]
        for m, e in enumerate(photos):
            if e["image_id"] in excluded:
                continue
            d = _km(np.asarray(e["pool"][:PAIR_K]), *e["truth"])
            borda = _borda(mine[m]["s"], len(d))
            data[tag]["dist"].append(d); data[tag]["borda"].append(borda); data[tag]["bench"].append(e.get("benchmark", "mp16"))
            if d[0] >= 25 and (d[1:] < 25).any():  # top-1 wrong, a right candidate among the next ones: does the judge prefer the right one to the top-1?
                right = int(np.argmin(np.where(d < 25, d, 1e9)))
                right_vs_wrong[tag].append(float(borda[right] > borda[0]) + 0.5 * float(borda[right] == borda[0]))
    for tag in ("dev", "val"):
        v = np.asarray(right_vs_wrong[tag])
        print(f"{tag}: top-1 wrong with a right candidate among ranks 2-{PAIR_K} (n={len(v)}): the judge's Borda score prefers the right candidate to the top-1 in {v.mean():.1%}")

    logit = lambda b: np.log(np.clip(b, 0.02, 0.98) / (1 - np.clip(b, 0.02, 0.98)))

    def top1(tag: str, w: float, shuffle: bool = False) -> np.ndarray:
        d, b = data[tag]["dist"], data[tag]["borda"]
        if shuffle:
            b = [b[i] for i in np.random.default_rng(0).permutation(len(b))]
        fit = lambda x, n: np.pad(x[:n], (0, max(0, n - len(x))), constant_values=0.5)
        return np.asarray([x[int(np.argmax(-np.arange(len(x)) + w * logit(fit(l, len(x)))))] for x, l in zip(d, b)])

    w = max(WEIGHTS, key=lambda v: ((top1("dev", v) < 25).mean(), -v))
    print(f"\njudge {name} (pairwise, top {PAIR_K}): w fitted on dev for top-1 <25 km: {w:g}")
    print(f"{'set':14s} {'n':>4s}  reranker top-1 <1/<25/<200 km    + pairwise judge (w={w:g})    change <1/<25/<200 km   <25 km 95% CI     shuffled control <25 km")
    for tag, bench in (("dev", None), ("val", None), ("val", "im2gps3k"), ("val", "yfcc4k")):
        sel = np.ones(len(data[tag]["dist"]), bool) if bench is None else np.asarray(data[tag]["bench"]) == bench
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
    parser.add_argument("node", choices=("pairs", "judge", "report", "topk-pairs", "topk-judge", "topk-report", "topk-diagnose", "topk-cv", "pair-judge", "pair-report"))
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--name", default="zeroshot-9b", help="which judge's top-k scores to write or analyse")
    args = parser.parse_args(argv)
    {"pairs": pairs, "judge": lambda: judge(args.server), "report": report, "topk-pairs": topk_pairs, "topk-judge": lambda: topk_judge(args.server, args.name),
     "topk-report": lambda: topk_report(args.name), "topk-diagnose": lambda: topk_diagnose(args.name), "topk-cv": lambda: topk_cv(args.name),
     "pair-judge": lambda: pair_judge(args.server, args.name), "pair-report": lambda: pair_report(args.name)}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
