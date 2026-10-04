# Does the comparator get better when it sees several exemplar photos of each candidate instead of one? (no training; comparator-a as is)
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.multi_exemplar pairs              (CPU, a few min)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.multi_exemplar judge --name comparator-a   (vLLM on :8765; scripts/multi_exemplar.sh)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.multi_exemplar report --name comparator-a
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.multi_exemplar full-pairs      (CPU; every im2gps3k / yfcc4k eval-half photo)
#        RUN=comparator-b bash scripts/multi_exemplar_full.sh                                          (full-judge, then full-report)

"""pairs:  for each dev / val photo's top TOPK candidates, up to EXEMPLARS database photos within 1 km (not by the query's photographer), at most one
        per photographer, most similar to the query first. Exemplar 0 is the one exemplar_judge.topk_pairs picks, so "first" reproduces lesson 24.
judge:  P(same place) for every (query, exemplar) pair.
report: top-1 <25 km of -rank + w * aggregate(logit P) with w fitted on dev, for aggregates first / max / mean / mean of the best 2, and the
        cross-validated listwise combiner (category_evidence._cv) with one-exemplar vs all-exemplar features, against the reranker top-1.
full-*: the same on all 3,795 benchmark eval-half photos (placeholders dropped); nothing is fitted on them: the aggregate and weight are chosen on the
        MP16 dev photos, and the listwise combiner is trained on the dev photos only."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .category_evidence import _cv
from .exemplar_judge import EXEMPLAR_KM, PROMPT, TOPK, WEIGHTS, _jpeg, _logit
from .query_evidence import ROOT
from .sft_data import MP16Images
from .stage1_eval import _bootstrap
from .strategy_search import EARTH_KM, MP16_EMBED, _xyz, load_world

EXEMPLARS = 4
PAIRS = ROOT / "multi_exemplar_pairs.json"


def _is_wikimedia(tag: str) -> bool:
    return tag.startswith("wikimedia")


def _pairs_path(tag: str) -> Path:
    return ROOT / f"multi_exemplar_{tag}_pairs.json"


def _scores_path(name: str, tag: str | None = None) -> Path:
    return ROOT / f"multi_exemplar_{tag + '_' if tag else ''}scores_{name}.json"


def _exemplars(world: Any, tree: Any, ids: list[str], chord: float, q: np.ndarray, author: int, latlon: Any) -> list[str]:
    """Up to EXEMPLARS database photos within 1 km of the candidate, not by the query's photographer, at most one per photographer, most similar first."""

    rows = np.sort([r for r in tree.query_ball_point(_xyz(np.asarray(latlon)), chord) if world.mp16["author"][r] != author])
    chosen: list[str] = []
    if len(rows):
        emb = np.asarray(world.mp16["embeddings"][rows], dtype=np.float32)
        seen: set[int] = set()
        for i in np.argsort(-(emb @ q / np.linalg.norm(emb, axis=1))):
            a = int(world.mp16["author"][rows[i]])
            if a not in seen:
                seen.add(a)
                chosen.append(ids[rows[i]])
                if len(chosen) == EXEMPLARS:
                    break
    return chosen


def pairs() -> None:
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
            found = [_exemplars(world, tree, ids, chord, q, author, latlon) for latlon in e["pool"][:TOPK]]
            out.append({"tag": tag, "image_id": e["image_id"], "path": e.get("path"), "exemplars": found})
        mine = [len(x) for p in out if p["tag"] == tag for x in p["exemplars"]]
        print(f"{tag}: {sum(p['tag'] == tag for p in out)} photos; exemplars per candidate: mean {np.mean(mine):.2f}, "
              f"none {np.mean(np.asarray(mine) == 0):.1%}, all {EXEMPLARS} {np.mean(np.asarray(mine) == EXEMPLARS):.0%}", flush=True)
    PAIRS.write_text(json.dumps(out) + "\n", encoding="utf-8")


def full_pairs(tag: str = "full") -> None:
    """Exemplars of the top TOPK candidates of every `full` photo; the Flickr "photo no longer available" placeholders (cosine >= 0.95 to the known one) get none."""

    from scipy.spatial import cKDTree

    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    world = load_world(benchmarks=("wikimedia",)) if _is_wikimedia(tag) else load_world()
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    chord = 2 * math.sin(EXEMPLAR_KM / EARTH_KM / 2)
    index = {q["image_id"]: i for i, q in enumerate(world.queries)}
    ref = None
    if not _is_wikimedia(tag):  # the Flickr "photo no longer available" placeholder, known from the benchmark halves
        ref_id = json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8"))[0]
        ref = world.query_embeddings[index[ref_id]] / np.linalg.norm(world.query_embeddings[index[ref_id]])
    out = []
    for m, e in enumerate(photos):
        q = world.query_embeddings[e["index"]] / np.linalg.norm(world.query_embeddings[e["index"]])
        placeholder = bool(ref is not None and q @ ref >= 0.95)
        author = int(world.query_author[e["index"]])
        found = [] if placeholder else [_exemplars(world, tree, ids, chord, q, author, latlon) for latlon in e["pool"][:TOPK]]
        out.append({"image_id": e["image_id"], "path": e["path"], "benchmark": e["benchmark"], "placeholder": placeholder, "exemplars": found})
        if (m + 1) % 1000 == 0:
            print(f"  {m + 1}/{len(photos)} photos", flush=True)
    _pairs_path(tag).write_text(json.dumps(out) + "\n", encoding="utf-8")
    counts = [len(x) for p in out for x in p["exemplars"]]
    print(f"{len(out)} photos, {sum(p['placeholder'] for p in out)} placeholders; exemplars per candidate mean {np.mean(counts):.2f}, none {np.mean(np.asarray(counts) == 0):.1%}")


def _post(server: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(f"{server}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def _p_same(server: str, query_b64: str, exemplar_b64: str, yes_no: tuple[int, int]) -> float | None:
    """P(yes) / (P(yes) + P(no)) with the output restricted to the two answer tokens (the server runs with --logprobs-mode processed_logprobs, so the
    returned logprobs are renormalised over them). A model whose Yes / No logits drifted below its other tokens (comparator-b) still scores exactly."""

    body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 2, "allowed_token_ids": list(yes_no),
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + query_b64}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + exemplar_b64}},
                {"type": "text", "text": PROMPT}]}]}
    try:
        top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception:
        return None
    by_token = {t["token"]: t["logprob"] for t in top}
    if "Yes" not in by_token or "No" not in by_token:
        return None
    return 1.0 / (1.0 + math.exp(by_token["No"] - by_token["Yes"]))


def judge(server: str, name: str, tag: str | None = None) -> None:
    photos = json.loads((_pairs_path(tag) if tag else PAIRS).read_text(encoding="utf-8"))
    images = MP16Images()
    yes_no = tuple(_post(server, "/tokenize", {"model": "vlm", "prompt": w, "add_special_tokens": False})["tokens"][0] for w in ("Yes", "No"))
    jobs = [(i, r, j) for i, p in enumerate(photos) for r, xs in enumerate(p["exemplars"]) for j in range(len(xs))]

    def run(job: tuple[int, int, int]) -> float | None:
        i, r, j = job
        p = photos[i]
        query = Path(p["path"]).read_bytes() if p["path"] else images.read(p["image_id"])
        return _p_same(server, _jpeg(query), _jpeg(images.read(p["exemplars"][r][j])), yes_no)

    with ThreadPoolExecutor(32) as pool:
        scores = list(pool.map(run, jobs))
    for p in photos:
        p["p_same"] = [[None] * len(xs) for xs in p["exemplars"]]
    for (i, r, j), score in zip(jobs, scores):
        photos[i]["p_same"][r][j] = score
    path = _scores_path(name, tag)
    path.write_text(json.dumps(photos) + "\n", encoding="utf-8")
    print(f"{len(jobs)} comparisons, {sum(s is None for s in scores)} failed -> {path}")


def _load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distances [photo, TOPK] (1e5 = no candidate), logits [photo, TOPK, EXEMPLARS] (NaN = no exemplar), tags and benchmarks; placeholders dropped."""

    from .wiki_backend import _km

    scored = json.loads(_scores_path(name).read_text(encoding="utf-8"))
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    D, L, tags, bench = [], [], [], []
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        mine = [p for p in scored if p["tag"] == tag]
        assert len(mine) == len(photos)
        for e, p in zip(photos, mine):
            if e["image_id"] in excluded:
                continue
            d = _km(np.asarray(e["pool"][:TOPK]), *e["truth"])
            lg = np.full((TOPK, EXEMPLARS), np.nan)
            for r, xs in enumerate(p["p_same"][: len(d)]):
                for j, x in enumerate(xs):
                    if x is not None:
                        lg[r, j] = _logit(np.asarray([x]))[0]
            D.append(np.pad(d, (0, TOPK - len(d)), constant_values=1e5))
            L.append(lg)
            tags.append(tag)
            bench.append(e.get("benchmark", "mp16"))
    return np.asarray(D), np.asarray(L), np.asarray(tags), np.asarray(bench)


def _aggregates(L: np.ndarray) -> dict[str, np.ndarray]:
    """Per candidate, one score from its exemplars' logits (0 = neutral when it has none)."""

    has = ~np.isnan(L)
    filled = np.where(has, L, -np.inf)
    count = has.sum(-1)
    best2 = -np.sort(-filled, axis=-1)[..., :2]
    best2 = np.where(np.isinf(best2), np.nan, best2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "first": np.where(has[..., 0], L[..., 0], 0.0),
            "max": np.where(count > 0, filled.max(-1), 0.0),
            "mean": np.where(count > 0, np.nansum(L, -1) / np.maximum(count, 1), 0.0),
            "mean of best 2": np.where(count > 0, np.nan_to_num(np.nanmean(best2, -1)), 0.0),
        }


def _features(L: np.ndarray, kind: str) -> np.ndarray:
    """Listwise combiner features per candidate [photo, TOPK, F]: the reranker rank and the judge's score (kind "first": exemplar 0 only; "all": max, mean and
    exemplar 0 of each candidate's exemplars plus their count)."""

    aggs = _aggregates(L)
    rank = np.broadcast_to(np.arange(TOPK, dtype=float), L.shape[:2])
    rank_f = [rank, (rank == 0).astype(float), (rank < 3).astype(float)]

    def judge_f(a: np.ndarray) -> list[np.ndarray]:
        order = (-a).argsort(1).argsort(1).astype(float)
        return [a, a * rank, a - a.max(1, keepdims=True), order, (order == 0).astype(float), a * (rank == 0)]

    feats = rank_f + judge_f(aggs["first"]) if kind == "first" else rank_f + judge_f(aggs["max"]) + judge_f(aggs["mean"]) + [aggs["first"], (~np.isnan(L)).sum(-1).astype(float)]
    return np.stack(feats, -1).astype(np.float32)


def _fit_apply(D: np.ndarray, F: np.ndarray, valid: np.ndarray, F_new: np.ndarray, valid_new: np.ndarray) -> np.ndarray:
    """Pick per photo of the listwise combiner trained on (D, F) only, applied to new photos: scores averaged over 3 seeds. Same net, reward and optimiser as category_evidence._cv."""

    import torch

    reward = np.where(valid, (D < 25) * 1.0 + (D < 1) * 0.5, 0.0).astype(np.float32)
    mean, std = F[valid].mean(0), F[valid].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(valid), torch.as_tensor(reward)
    X_new = torch.as_tensor((F_new - mean) / std)
    total = torch.zeros(F_new.shape[:2])
    for seed in range(3):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
        for _ in range(300):
            loss = -(torch.softmax(net(X).squeeze(-1).masked_fill(~M, float("-inf")), -1) * R).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            total += net(X_new).squeeze(-1)
    return total.masked_fill(~torch.as_tensor(valid_new), float("-inf")).argmax(1).numpy()


def report(name: str) -> None:
    D, L, tags, bench = _load(name)
    valid = D < 9e4
    rank = np.broadcast_to(np.arange(TOPK, dtype=float), D.shape)
    base = (D[:, 0] < 25).astype(float)
    count = (~np.isnan(L)).sum(-1)
    print(f"{name}: {len(D)} photos (dev {int((tags == 'dev').sum())}, val {int((tags == 'val').sum())}); exemplars per candidate mean "
          f"{count[valid].mean():.2f}; reranker top-1 <25 km dev {base[tags == 'dev'].mean():.1%}, val {base[tags == 'val'].mean():.1%}\n")

    def top1(score: np.ndarray) -> np.ndarray:
        return (D[np.arange(len(D)), np.where(valid, score, -np.inf).argmax(1)] < 25).astype(float)

    out: dict[str, Any] = {"n": len(D), "rule": {}, "cv": {}}
    print("score = -rank + w * aggregate(logit P(same)), w fitted on dev; change in top-1 <25 km vs the reranker [95% CI]")
    for agg, a in _aggregates(L).items():
        w = max(WEIGHTS, key=lambda v: (top1(-rank + v * a)[tags == "dev"].mean(), -v))
        hit = top1(-rank + w * a)
        cells = {}
        for label, sel in (("dev", tags == "dev"), ("val", tags == "val"), ("im2gps3k", bench == "im2gps3k"), ("yfcc4k", bench == "yfcc4k")):
            cells[label] = _bootstrap(hit[sel] - base[sel])
        out["rule"][agg] = {"w": w, **cells}
        print(f"  {agg:15s} w={w:<4g} " + "  ".join(f"{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" for k, v in cells.items()))

    configs = {"rank + first exemplar": _features(L, "first"), "rank + all exemplars": _features(L, "all")}
    print("\ncross-validated listwise combiner (5 folds, 3 seeds, dev + val), change in top-1 <25 km vs the reranker [95% CI]")
    for label, feats in configs.items():
        hit = _cv(D, feats, valid)
        cells = {k: _bootstrap(hit[s] - base[s]) for k, s in (("all", np.ones(len(D), bool)), ("dev", tags == "dev"), ("val", tags == "val"))}
        out["cv"][label] = cells
        print(f"  {label:24s} " + "  ".join(f"{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" for k, v in cells.items()))
    (ROOT / f"multi_exemplar_report_{name}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def full_report(name: str, tag: str = "full") -> None:
    """Top-1 on every im2gps3k / yfcc4k eval-half photo (placeholders dropped) against the reranker, with everything chosen or trained on the MP16 dev photos:
    the (aggregate, weight) of the scalar rule is the pair with the best dev top-1 < 25 km, the listwise combiner is trained on dev only."""

    from .wiki_backend import _km

    D, L, tags, _ = _load(name)
    dev = tags == "dev"
    D_dev, L_dev = D[dev], L[dev]
    valid_dev = D_dev < 9e4
    rank_dev = np.broadcast_to(np.arange(TOPK, dtype=float), D_dev.shape)

    def top1(Dx: np.ndarray, valid: np.ndarray, score: np.ndarray) -> np.ndarray:
        return (Dx[np.arange(len(Dx)), np.where(valid, score, -np.inf).argmax(1)] < 25).astype(float)

    aggs_dev = _aggregates(L_dev)
    fitted = {agg: max(WEIGHTS, key=lambda v: (top1(D_dev, valid_dev, -rank_dev + v * a).mean(), -v)) for agg, a in aggs_dev.items()}
    chosen = max(fitted, key=lambda agg: top1(D_dev, valid_dev, -rank_dev + fitted[agg] * aggs_dev[agg]).mean())

    scored = json.loads(_scores_path(name, tag).read_text(encoding="utf-8"))
    photos = {e["image_id"]: e for e in json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))}
    with np.load(Path("artifacts/wikimedia" if _is_wikimedia(tag) else "artifacts/strategy_search") / "neighbors.npz") as cache:  # top raw neighbour similarity after same-photographer exclusion
        raw = np.maximum(cache["mp16_raw_sim"][:, 0], cache["osv_raw_sim"][:, 0])
    keep = [p for p in scored if not p["placeholder"]]
    Df, Lf, bench, near = [], [], [], []
    for p in keep:
        e = photos[p["image_id"]]
        d = _km(np.asarray(e["pool"][:TOPK]), *e["truth"])
        lg = np.full((TOPK, EXEMPLARS), np.nan)
        for r, xs in enumerate(p["p_same"][: len(d)]):
            for j, x in enumerate(xs):
                if x is not None:
                    lg[r, j] = _logit(np.asarray([x]))[0]
        Df.append(np.pad(d, (0, TOPK - len(d)), constant_values=1e5))
        Lf.append(lg)
        bench.append(p["benchmark"])
        near.append(bool(raw[e["index"]] >= 0.95))
    Df, Lf, bench, near = np.asarray(Df), np.asarray(Lf), np.asarray(bench), np.asarray(near)
    valid = Df < 9e4
    rank = np.broadcast_to(np.arange(TOPK, dtype=float), Df.shape)
    base = (Df[:, 0] < 25).astype(float)
    base_t = {t: (Df[:, 0] < t).astype(float) for t in (1.0, 25.0, 200.0)}
    arms: dict[str, np.ndarray] = {}
    for agg, a in _aggregates(Lf).items():
        arms[f"rule: {agg} (w={fitted[agg]:g}){'  <- chosen on dev' if agg == chosen else ''}"] = np.where(valid, -rank + fitted[agg] * a, -np.inf).argmax(1)
    for kind in ("first", "all"):
        arms[f"combiner trained on dev: rank + {kind} exemplar{'s' if kind == 'all' else ''}"] = _fit_apply(D_dev, _features(L_dev, kind), valid_dev, _features(Lf, kind), valid)
    print(f"{name} on {tag}: {len(Df)} photos (placeholders dropped: {sum(p['placeholder'] for p in scored)}); exemplars per candidate "
          f"{(~np.isnan(Lf)).sum(-1)[valid].mean():.2f}; nothing fitted on them. Reranker top-1 <1/<25/<200 km: " + " / ".join(f"{base_t[t].mean():.1%}" for t in base_t) + "\n")
    out: dict[str, Any] = {"n": len(Df), "reranker": {f"<{t:g} km": float(v.mean()) for t, v in base_t.items()}, "arms": {}}
    sets = (("all", np.ones(len(Df), bool)), ("no near-dup", ~near), ("near-dup", near)) if _is_wikimedia(tag) else \
        (("all", np.ones(len(Df), bool)), ("im2gps3k", bench == "im2gps3k"), ("yfcc4k", bench == "yfcc4k"), ("no near-dup", ~near), ("near-dup", near))
    for label, pick in arms.items():
        hits = {t: (Df[np.arange(len(Df)), pick] < t).astype(float) for t in base_t}
        cells = {k: _bootstrap(hits[25.0][s] - base[s]) for k, s in sets}
        out["arms"][label] = {"top1": {f"<{t:g} km": float(h.mean()) for t, h in hits.items()}, "change <25 km": cells}
        print(f"{label}\n    top-1 <1/<25/<200 km: " + " / ".join(f"{hits[t].mean():.1%}" for t in hits) + "   change: " + "  ".join(f"<{t:g} {100 * (hits[t].mean() - base_t[t].mean()):+.1f}" for t in hits))
        print("    <25 km change [95% CI]: " + "  ".join(f"{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}] (n={int(s.sum())})" for (k, s), v in zip(sets, cells.values())))
    (ROOT / f"multi_exemplar_{tag}_report_{name}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("pairs", "judge", "report", "full-pairs", "full-judge", "full-report"))
    parser.add_argument("--name", default="comparator-a")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--tag", default="full", help="photo set for full-*: full (im2gps3k + yfcc4k eval halves), wikimedia or wikimedia_balanced")
    args = parser.parse_args(argv)
    if args.node == "pairs":
        pairs()
    elif args.node == "judge":
        judge(args.server, args.name)
    elif args.node == "report":
        report(args.name)
    elif args.node == "full-pairs":
        full_pairs(args.tag)
    elif args.node == "full-judge":
        judge(args.server, args.name, args.tag)
    else:
        full_report(args.name, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
