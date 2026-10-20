# Track S (SEARCH_PLAN.md), step 1: the search-tree ceiling with a luck control. Per photo, a discrete constraint vocabulary from its own pool
# (countries and regions of the ~46 candidates, +-25 km boxes around candidates outside every area); Pinpoint (tools.pinpoint) restricted to
# each; how often is some restricted top-1 < 25 km beyond the area oracle, per loss bucket (lesson 129), against the same vocabulary scored on
# another photo's truth (lesson 120). Also the search speed of tools.pinpoint per constraint kind.
# Usage (PINPOINT_KEEP_OWN=1, PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_own/checkpoints/ckpt_final.pt; GPU):
#   PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.search_tree speed
#   PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.search_tree vocab --tag dev3k_setft_own      (CPU)
#   PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.search_tree run --tag dev3k_setft_own        (resumable parts)
#   PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.search_tree ceiling --tag dev3k_setft_own    (CPU)

"""speed:   ms / query of tools.pinpoint by constraint kind and batch size, peak GPU memory.
vocab:   per photo the constraints (kind, label / box, pool mass = softmax(score / 0.2) share of pool candidates inside) ->
         artifacts/search/vocab_{tag}.json; also the true country / region as reference searches (not in the vocabulary).
run:     every vocabulary constraint searched -> artifacts/search/searches_{tag}.npz (top TOP restricted candidates per search).
ceiling: oracle of restricted top-1s beyond the area oracle, per family and loss bucket, by search budget (constraints in pool-mass order),
         against two luck controls (another photo's truth: any photo; a photo in the same true country)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/search")
PHOTO_SETS = Path("artifacts/photo_sets")
AGENT = Path("artifacts/agent")
APPLY = AGENT / "action_agent_apply_km_setft_own_1km0.5_commons26dev.npz"  # the pipeline's cached answers (lesson 129's)
BOX_KM, TEMP, TOP, PART = 25.0, 0.2, 16, 4096
KINDS = ("country", "region", "box")


def _km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Haversine km between broadcastable [..., 2] lat / lon arrays (NaN stays NaN)."""

    a, b = np.radians(a), np.radians(b)
    h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _box(c: Sequence[float]) -> tuple[float, float, float, float]:
    """The pipeline's zoom box (hypotheses.restricted_search): +-BOX_KM around a centre."""

    dlat = BOX_KM / 111.0
    dlon = dlat / max(np.cos(np.radians(c[0])), 0.05)
    return (c[0] - dlat, c[0] + dlat, c[1] - dlon, c[1] + dlon)


def _photos(tag: str) -> list[dict]:
    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    for e in photos:
        for k in ("truth", "pool", "pool_rows"):
            if isinstance(e.get(k), str):
                e[k] = json.loads(e[k])
    return photos


def _tool():
    if os.environ.get("PINPOINT_KEEP_OWN") != "1" or "setft_own" not in os.environ.get("PINPOINT_RERANKER_CKPT", ""):
        raise SystemExit("set PINPOINT_KEEP_OWN=1 and PINPOINT_RERANKER_CKPT to the setft_own checkpoint (the primary protocol's pipeline)")
    from ..tools.pinpoint import Pinpoint

    return Pinpoint()


def _queries(tool, photos: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(raw SigLIP2 embeddings, author ids) of the photos; KEEP_OWN: each MP16 photo is its own author, so only the query itself is dropped."""

    from .strategy_search import load_world

    rows = np.asarray([int(e["row"]) for e in photos])
    return np.asarray(load_world().mp16["embeddings"][rows], np.float32), tool.mp16_author(rows)


# ---------------------------------------------------------------- speed

def speed(tag: str, sizes: Sequence[int] = (16, 256, 1024)) -> None:
    import torch

    photos = _photos(tag)
    tool = _tool()
    emb, authors = _queries(tool, photos[: max(sizes)])
    country, region = tool.labels()
    pool_rows = np.asarray([e["pool_rows"][0] for e in photos[: max(sizes)]])  # each photo's pool top-1: a constraint that contains a match
    ll = np.asarray(tool.reranker.latlon)
    from ..tools.pinpoint import Constraint

    def make(kind: str, i: int):
        if kind == "none":
            return None
        if kind == "box":
            return Constraint(box=_box(ll[pool_rows[i]]))
        if kind == "country":
            return Constraint(country=country[pool_rows[i]])
        if kind == "region":
            return Constraint(region=region[pool_rows[i]])
        if kind == "near":
            return Constraint(near=(ll[pool_rows[i]][None].astype(np.float64), BOX_KM))
        b = _box(ll[pool_rows[i]])
        return Constraint(rows=np.flatnonzero((ll[:, 0] >= b[0]) & (ll[:, 0] <= b[1]) & (ll[:, 1] >= b[2]) & (ll[:, 1] <= b[3])))

    tool.search(emb[:4], authors=authors[:4])  # warm-up: tower, OSV tree, page cache
    lines = [f"tools.pinpoint search speed ({tag} photos; index {tool.reranker.index_dir}; {torch.cuda.get_device_name()})"]
    free = tool.search(emb[:64], authors=authors[:64])
    saved = np.asarray([e["pool"][0] for e in photos[:64]])
    lines.append(f"  check: unconstrained top-1 = the saved {tag} pool top-1 (< 0.1 km) on {np.mean(_km(free.latlon[:, 0], saved) < 0.1):.0%} of 64")
    for kind in ("none", "box", "country", "region", "near", "rows"):
        for n in (sizes if kind in ("none", "box", "country", "region") else (16, 256)):
            cons = [make(kind, i) for i in range(n)]
            if kind in ("near", "rows") and not tool._resident:
                tool.search(emb[:1], [make(kind, 0)], authors[:1])  # loads the resident index (not timed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            tool.search(emb[:n], cons, authors[:n])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            line = (f"  {kind:8s} batch {n:5d}: {dt:7.1f} s, {1000 * dt / n:8.1f} ms / query; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            if kind in ("near", "rows"):
                line += f" (rows per query {np.mean([len(c.rows) if c.rows is not None else len(tool.rows_near(*c.near)) for c in cons]):.0f})"
            lines.append(line)
            print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speed.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- vocabulary

def _areas(tag: str) -> dict[int, np.ndarray]:
    """Area centres per photo: the pipeline's Pinpoint areas and clue places (hypotheses sources 0, 1)."""

    centres: dict[int, list] = {}
    for name in (f"hypotheses_{tag}.npz", f"hypotheses_{tag}_cluehyps.npz"):
        z = np.load(AGENT / name)
        for p, c in zip(z["photo"], z["center"]):
            centres.setdefault(int(p), []).append(c)
    return {p: np.asarray(c, np.float64).reshape(-1, 2) for p, c in centres.items()}


def vocab(tag: str) -> None:
    from ..tools.pinpoint import LABELS_NPZ
    from .pinpoint_reference import load_pools

    photos = _photos(tag)
    pools = load_pools(photos[0]["pools"])
    z = np.load(LABELS_NPZ)
    country = dict(zip(z["row"].tolist(), z["country"].tolist()))
    region = {r: f"{c}|{g}" for r, c, g in zip(z["row"].tolist(), z["country"].tolist(), z["region"].tolist()) if c and g}
    areas = _areas(tag)
    out = []
    for p, e in enumerate(photos):
        v = pools["valid"][e["index"]]
        cache = pools["cache_rows"][e["index"]][v]
        ll = pools["latlon"][e["index"]][v].astype(np.float64)
        s = pools["scores"][e["index"]][v].astype(np.float64)
        w = np.exp((s - s.max()) / TEMP)
        w /= w.sum()
        cons = []
        for kind, lab in (("country", country), ("region", region)):
            labels = np.asarray([lab.get(int(c), "") for c in cache], dtype=object)
            for name in dict.fromkeys(labels[np.argsort(-w, kind="stable")]):
                if name:
                    cons.append({"kind": kind, "label": name, "mass": float(w[labels == name].sum())})
        here = areas.get(p, np.zeros((0, 2)))
        outside = np.flatnonzero((_km(ll[:, None], here[None]) >= BOX_KM).all(1)) if len(here) else np.arange(len(ll))
        kept: list[int] = []
        for j in outside[np.argsort(-w[outside], kind="stable")]:  # boxes in score order, one per 25 km
            if all(_km(ll[j], ll[k]) >= BOX_KM for k in kept):
                kept.append(int(j))
        for j in kept:
            cons.append({"kind": "box", "center": ll[j].tolist(), "mass": float(w[_km(ll, ll[j]) < BOX_KM].sum())})
        tc = int(e["row"])
        out.append({"photo": p, "truth_country": country.get(tc, ""), "truth_region": region.get(tc, ""), "constraints": cons})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"vocab_{tag}.json").write_text(json.dumps(out), encoding="utf-8")
    n = {k: np.asarray([sum(c["kind"] == k for c in r["constraints"]) for r in out]) for k in KINDS}
    print(f"{tag}: {len(out)} photos; constraints per photo (mean / median / max): " +
          ", ".join(f"{k} {n[k].mean():.1f} / {np.median(n[k]):.0f} / {n[k].max()}" for k in KINDS) +
          f"; total {sum(int(v.sum()) for v in n.values())}; true country labelled {np.mean([bool(r['truth_country']) for r in out]):.1%}, "
          f"region {np.mean([bool(r['truth_region']) for r in out]):.1%}")


# ---------------------------------------------------------------- searches

def run(tag: str) -> None:
    from ..tools.pinpoint import Constraint

    photos = _photos(tag)
    voc = json.loads((OUT / f"vocab_{tag}.json").read_text(encoding="utf-8"))
    items = []  # (photo, kind code, constraint); reference searches (true country / region) get kinds 3 / 4
    for r in voc:
        for c in r["constraints"]:
            items.append((r["photo"], KINDS.index(c["kind"]), Constraint(box=_box(c["center"])) if c["kind"] == "box" else
                          Constraint(**{c["kind"]: c["label"]})))
        if r["truth_country"]:
            items.append((r["photo"], 3, Constraint(country=r["truth_country"])))
        if r["truth_region"]:
            items.append((r["photo"], 4, Constraint(region=r["truth_region"])))
    tool = _tool()
    emb, authors = _queries(tool, photos)
    parts = OUT / f"parts_{tag}"
    parts.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(items)), key=lambda i: (items[i][1] % 3, items[i][0]))  # one kind per part (box / label kinds take different paths)
    print(f"{tag}: {len(items)} searches in {-(-len(items) // PART)} parts", flush=True)
    for start in range(0, len(order), PART):
        dst = parts / f"{start // PART:04d}.npz"
        if dst.exists():
            continue
        idx = np.asarray(order[start: start + PART])
        p = np.asarray([items[i][0] for i in idx])
        t = time.perf_counter()
        found = tool.search(emb[p], [items[i][2] for i in idx], authors[p])
        np.savez(dst, item=idx, latlon=found.latlon[:, :TOP].astype(np.float32), scores=found.scores[:, :TOP].astype(np.float32),
                 features=found.features[:, :TOP].astype(np.float16), rows=found.rows[:, :TOP].astype(np.int32))
        print(f"  part {start // PART}: {len(idx)} searches in {time.perf_counter() - t:.0f} s", flush=True)
    merged = {k: [] for k in ("item", "latlon", "scores", "features", "rows")}
    for f in sorted(parts.glob("*.npz")):
        z = np.load(f)
        for k in merged:
            merged[k].append(z[k])
    merged = {k: np.concatenate(v) for k, v in merged.items()}
    o = np.argsort(merged["item"])
    assert np.array_equal(merged["item"][o], np.arange(len(items))), "parts do not cover the vocabulary (stale parts?)"
    np.savez(OUT / f"searches_{tag}.npz", photo=np.asarray([it[0] for it in items]), kind=np.asarray([it[1] for it in items]),
             **{k: v[o] for k, v in merged.items() if k != "item"})
    print(f"-> {OUT / f'searches_{tag}.npz'}", flush=True)


# ---------------------------------------------------------------- ceiling

def _buckets(tag: str, photos: list[dict]):
    """Lesson 129's per-photo stage buckets (scripts/pipeline_diagnostics.py): pipeline right, wrong area chosen, never an area, retrieval miss."""

    from .area_model import _paths
    from .hypothesis_agent import SOURCES
    from .pinpoint_reference import load_pools

    base, ck = tag.split("_", 1)
    km = np.load(APPLY)[f"{base}|RL policy, lambda 0"]
    pools = load_pools(photos[0]["pools"])
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    pool_ok = np.asarray([(_km(truth[i], pools["latlon"][e["index"]][pools["valid"][e["index"]]].astype(np.float64)) < 25).any()
                          for i, e in enumerate(photos)])
    f = np.load(_paths(base, ck)[0])
    keep = np.isin(f["source"], SOURCES)
    area_ok = np.zeros(len(photos), bool)
    np.logical_or.at(area_ok, f["photo"][keep], f["km_agree"][keep] < 25)
    ok = km < 25
    bucket = np.where(ok, 0, np.where(area_ok, 1, np.where(pool_ok, 2, 3)))
    return ok, area_ok | ok, pool_ok, bucket


BUCKETS = ("pipeline right", "wrong area chosen", "never an area", "retrieval miss")


def ceiling(tag: str, seed: int = 0) -> None:
    photos = _photos(tag)
    n = len(photos)
    voc = json.loads((OUT / f"vocab_{tag}.json").read_text(encoding="utf-8"))
    z = np.load(OUT / f"searches_{tag}.npz")
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    ok, area_ok, pool_ok, bucket = _buckets(tag, photos)
    # the vocabulary's search index per (photo, constraint), in vocabulary order; mass per search
    kind, photo = z["kind"], z["photo"]
    mass = np.full(len(kind), np.nan)
    pos = np.full(len(kind), -1)
    i = 0
    for r in voc:
        for j, c in enumerate(r["constraints"]):
            mass[i], pos[i] = c["mass"], j
            i += 1
        i += bool(r["truth_country"]) + bool(r["truth_region"])
    top1 = z["latlon"][:, 0].astype(np.float64)
    vocab_items = kind < 3
    # area oracle per photo against any truth: the pipeline's area searches (zoom top-1 of each area) + the pipeline's answer
    hyp = {}
    for name in (f"hypotheses_{tag}.npz", f"hypotheses_{tag}_cluehyps.npz"):
        h = np.load(AGENT / name)
        for p, ll in zip(h["photo"], h["latlon"][:, 0]):
            hyp.setdefault(int(p), []).append(ll)
    answer = np.load(APPLY)[f"{tag.split('_')[0]}|RL policy, lambda 0|center"]
    area_pts = [np.asarray(hyp.get(p, []) + [answer[p]], np.float64).reshape(-1, 2) for p in range(n)]
    pool_pts = [e["pool"] for e in photos]

    def hit(points_of, against):
        return np.asarray([bool(len(points_of[p])) and bool((_km(np.asarray(points_of[p], np.float64), against[p]) < 25).any()) for p in range(n)])

    rng = np.random.default_rng(seed)
    derange = rng.permutation(n)
    while (derange == np.arange(n)).any():
        derange = rng.permutation(n)
    same = np.arange(n)  # a different photo with the same true country label
    tc = np.asarray([r["truth_country"] for r in voc], dtype=object)
    for c in set(tc) - {""}:
        g = rng.permutation(np.flatnonzero(tc == c))
        if len(g) > 1:
            same[g] = np.roll(g, 1)
    same_ok = (same != np.arange(n))
    far = np.arange(n)  # same true country, truth > 100 km away (no same-city pairs)
    for i in range(n):
        g = np.flatnonzero((tc == tc[i]) & (tc[i] != ""))
        g = g[_km(truth[i], truth[g]) > 100]
        if len(g):
            far[i] = rng.choice(g)
    far_ok = far != np.arange(n)
    truths = {"real truth": truth, "luck: random other photo": truth[derange], "luck: same-country photo": truth[same],
              "luck: same-country photo > 100 km away": truth[far]}

    def search_points(sel):
        pts = [[] for _ in range(n)]
        for k in np.flatnonzero(sel):
            if np.isfinite(top1[k, 0]):
                pts[photo[k]].append(top1[k])
        return pts

    lines = [f"Search-tree ceiling, {tag} ({n} photos; restricted top-1 < 25 km). Pipeline {100 * ok.mean():.1f}, area oracle "
             f"(lesson 129) {100 * area_ok.mean():.1f}, pool oracle {100 * pool_ok.mean():.1f}"]
    counts = {k: np.bincount(photo[kind == KINDS.index(k)], minlength=n) for k in KINDS}
    lines.append("searches per photo (mean / median / p90): " + ", ".join(
        f"{k} {counts[k].mean():.1f} / {np.median(counts[k]):.0f} / {np.quantile(counts[k], 0.9):.0f}" for k in KINDS) +
        f"; all {sum(counts.values()).mean():.1f}")
    fams = {"country": kind == 0, "region": kind == 1, "box": kind == 2, "all": vocab_items}
    for tname, tr in truths.items():
        sub = (far_ok if "100 km" in tname else same_ok) if "same-country" in tname else np.ones(n, bool)
        a = hit(area_pts, tr)
        po = hit(pool_pts, tr)
        lines.append(f"\n== {tname} (photos {sub.sum()}): area oracle {100 * a[sub].mean():.1f}, pool oracle {100 * po[sub].mean():.1f}")
        for fam, sel in fams.items():
            s = hit(search_points(sel), tr)
            lines.append(f"  {fam:8s}: some restricted top-1 {100 * s[sub].mean():5.1f}; beyond the area oracle +{100 * (s & ~a)[sub].mean():.1f} "
                         f"(beyond area + pool oracle +{100 * (s & ~a & ~po)[sub].mean():.1f})")
        if tname == "real truth":
            ref = {"true country": kind == 3, "true region": kind == 4}
            for nm, sel in ref.items():
                s = hit(search_points(sel), tr)
                lines.append(f"  reference, {nm} search (not in the vocabulary): top-1 {100 * s.mean():.1f}; beyond area oracle +{100 * (s & ~a).mean():.1f}")
            s_all = hit(search_points(vocab_items), tr)
            lines.append("  per loss bucket (% of the bucket / pts of the set fixed by some vocabulary search):")
            for b, nm in enumerate(BUCKETS):
                m = bucket == b
                lines.append(f"    {nm:18s} {100 * m.mean():5.1f}% of photos: " + ", ".join(
                    f"{fam} {100 * hit(search_points(sel), tr)[m].mean():.1f}% / {100 * (hit(search_points(sel), tr) & m).mean():.1f}"
                    for fam, sel in fams.items()))
        # budget: the first k vocabulary searches by pool mass (fixed recipe order), all families pooled
        line = "  budget k (top-k constraints by pool mass): "
        order = {}
        for k in np.flatnonzero(vocab_items):
            order.setdefault(int(photo[k]), []).append(k)
        for p in order:
            order[p].sort(key=lambda k: -mass[k])
        for budget in (1, 2, 4, 8, 16, 32, 64):
            sel = np.zeros(len(kind), bool)
            for p, ks in order.items():
                sel[ks[:budget]] = True
            s = hit(search_points(sel), tr)
            line += f"{budget}: +{100 * (s & ~a)[sub].mean():.1f}  "
        lines.append(line + "(beyond the area oracle)")
        for fam in KINDS:  # per family, in mass order
            line = f"    {fam:8s} only, budget k: "
            for budget in (1, 2, 4, 8, 16):
                sel = np.zeros(len(kind), bool)
                for p, ks in order.items():
                    sel[[k for k in ks if kind[k] == KINDS.index(fam)][:budget]] = True
                s = hit(search_points(sel), tr)
                line += f"{budget}: +{100 * (s & ~a)[sub].mean():.1f}  "
            lines.append(line)
    text = "\n".join(lines)
    (OUT / f"ceiling_{tag}.txt").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------- step 2: is a search's result evidence?

A_NAMES = ("is_country", "is_region", "is_box", "log_mass", "log_n_inside", "best_inside_vs_top", "log_rank_inside", "has_pool_top1",
           "pool_top_score", "pool_top_mass")
B_NAMES = ("r1_score", "r1_vs_pool_top", "r1_margin", "r_agree", "r_valid", "sim_image", "sim_raw", "sim_gps", "rank_image", "rank_raw",
           "rank_gps", "in_image", "in_raw", "in_gps", "log_km_pool", "log_km_pool_top1", "r1_in_pool", "log_n_agree_searches",
           "agree_searches_mass")


def _features(tag: str, budget: int | None) -> dict:
    """Per vocabulary search (the first `budget` of each photo in pool-mass order; None: all): prior features A (what the pool already says
    about the constraint), result features B (what the restricted search returned, incl. agreement with the photo's other searches in the
    budget), the label (restricted top-1 < 25 km), the photo and the restricted top-1."""

    from ..tools.pinpoint import LABELS_NPZ
    from .pinpoint_reference import load_pools

    photos = _photos(tag)
    pools = load_pools(photos[0]["pools"])
    voc = json.loads((OUT / f"vocab_{tag}.json").read_text(encoding="utf-8"))
    z = np.load(OUT / f"searches_{tag}.npz")
    LL, SC, FE, RO = z["latlon"], z["scores"], z["features"], z["rows"]
    lab = np.load(LABELS_NPZ)
    country = dict(zip(lab["row"].tolist(), lab["country"].tolist()))
    region = {r: f"{c}|{g}" for r, c, g in zip(lab["row"].tolist(), lab["country"].tolist(), lab["region"].tolist()) if c and g}
    A, B, y, ph, top, search = [], [], [], [], [], []
    i = 0
    for r, e in zip(voc, photos):
        cons = r["constraints"]
        ks = np.arange(i, i + len(cons))
        i += len(cons) + bool(r["truth_country"]) + bool(r["truth_region"])
        if not len(cons):
            continue
        chosen = np.argsort([-c["mass"] for c in cons], kind="stable")[:budget]
        v = pools["valid"][e["index"]]
        ll = pools["latlon"][e["index"]][v].astype(np.float64)
        s = pools["scores"][e["index"]][v].astype(np.float64)
        w = np.exp((s - s.max()) / TEMP)
        w /= w.sum()
        cache = pools["cache_rows"][e["index"]][v]
        prow = set(pools["rows"][e["index"]][v].tolist())
        labels = {"country": np.asarray([country.get(int(c), "") for c in cache], object),
                  "region": np.asarray([region.get(int(c), "") for c in cache], object)}
        r1 = LL[ks[chosen], 0].astype(np.float64)
        ok1 = np.isfinite(r1[:, 0])
        pair = _km(np.where(ok1[:, None], r1, 0)[:, None], np.where(ok1[:, None], r1, 0)[None]) < BOX_KM
        pair &= ok1[:, None] & ok1[None]
        np.fill_diagonal(pair, False)
        masses = np.asarray([cons[j]["mass"] for j in chosen])
        truth = np.asarray(e["truth"], np.float64)
        for q, j in enumerate(chosen):
            c, k = cons[j], ks[j]
            inside = _km(ll, np.asarray(c["center"])) < BOX_KM if c["kind"] == "box" else labels[c["kind"]] == c["label"]
            n_in = int(inside.sum())
            A.append((c["kind"] == "country", c["kind"] == "region", c["kind"] == "box", np.log(c["mass"] + 1e-6), np.log1p(n_in),
                      (s[inside].max() - s.max()) if n_in else -10.0, np.log1p(np.flatnonzero(inside)[0]) if n_in else np.log1p(len(s)),
                      bool(inside[0]), s.max(), w.max()))
            rs = SC[k].astype(np.float64)
            valid = np.isfinite(rs)
            if not ok1[q]:
                B.append((-10.0, -10.0 - s.max(), 0.0, 0.0, 0.0) + (0.0,) * 9 + (np.log1p(20000.0),) * 2 + (0.0, 0.0, 0.0))
            else:
                wr = np.exp((rs[valid] - rs[valid].max()) / TEMP)
                wr /= wr.sum()
                agree = float(wr[_km(LL[k][valid].astype(np.float64), r1[q]) < BOX_KM].sum())
                B.append((rs[0], rs[0] - s.max(), rs[0] - rs[1] if valid.sum() > 1 else 0.0, agree, valid.mean(),
                          *FE[k, 0].astype(np.float64), np.log1p(_km(ll, r1[q]).min()), np.log1p(_km(ll[0], r1[q])), int(RO[k, 0]) in prow,
                          np.log1p(pair[q].sum()), float(masses[pair[q]].sum())))
            y.append(ok1[q] and _km(truth, r1[q]) < 25)
            ph.append(r["photo"])
            top.append(r1[q])
            search.append(k)
    return {"A": np.asarray(A, np.float32), "B": np.asarray(B, np.float32), "y": np.asarray(y, bool), "photo": np.asarray(ph),
            "top": np.asarray(top, np.float64), "search": np.asarray(search)}


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    if y.all() or not y.any():
        return float("nan")
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def _fit(x: np.ndarray, y: np.ndarray, epochs: int = 4, seed: int = 0):
    """A small MLP (64 x 2, BCE) on standardized features; returns predict(x) -> P(right)."""

    import torch

    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.as_tensor((x - mu) / sd), torch.as_tensor(y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        for b in torch.randperm(len(xt)).split(4096):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(x2: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(net(torch.as_tensor((x2 - mu) / sd)).squeeze(1)).numpy()

    return predict


def _context(tag: str) -> dict:
    photos = _photos(tag)
    ok, area_ok, pool_ok, bucket = _buckets(tag, photos)
    apply = np.load(APPLY)
    base = tag.split("_")[0]
    return {"n": len(photos), "ok": ok, "area_ok": area_ok, "bucket": bucket, "answer": apply[f"{base}|RL policy, lambda 0|center"],
            "conf": apply[f"{base}|RL policy, lambda 0|conf"], "truth": np.asarray([e["truth"] for e in photos], np.float64)}


def _switch_lines(name: str, p: np.ndarray, te: dict, cx: dict, seeds: int = 20) -> list[str]:
    """AUC (all, among alternatives = searches whose top-1 is >= 25 km from the pipeline's answer, the only ones that can change it) and the
    switch rule pipeline answer -> the best alternative's top-1 when P (or P - the pipeline's confidence) beats a threshold; thresholds
    cross-fitted on random photo halves (seeds x 2), plus the hindsight-best threshold and the 30 most confident switches."""

    n, ok, answer, conf, truth = cx["n"], cx["ok"], cx["answer"], cx["conf"], cx["truth"]
    wrong = ~ok[te["photo"]]
    alt = _km(te["top"], answer[te["photo"]]) >= 25
    best = np.full(n, -1)
    for k in np.flatnonzero(alt)[np.argsort(p[alt])]:
        best[te["photo"][k]] = k  # the last write is the highest P
    has = best >= 0
    pb, tb = np.where(has, p[np.maximum(best, 0)], -1.0), te["top"][np.maximum(best, 0)]
    right_b = has & (_km(tb, truth) < 25)
    lines = [f"  {name:22s}: AUC {_auc(p, te['y']):.3f} (pipeline-wrong photos {_auc(p[wrong], te['y'][wrong]):.3f}); among alternatives "
             f"{_auc(p[alt], te['y'][alt]):.3f} (pipeline-wrong photos {_auc(p[alt & wrong], te['y'][alt & wrong]):.3f}); best alternative "
             f"right on {100 * right_b.mean():.1f}% of photos (pipeline wrong & it right {100 * (right_b & ~ok).mean():.1f}, "
             f"some alternative right {100 * np.isin(np.arange(n), te['photo'][alt & te['y']]).mean():.1f})"]
    for rule, gate in (("P(best) > t", pb), ("P(best) - conf > t", pb - conf)):
        grid = np.append(np.quantile(gate[has], np.linspace(0, 1, 101)) if has.any() else [], np.inf)  # inf: never switch

        def net(m, t):
            sw = m & has & (gate > t)
            return int((sw & right_b & ~ok).sum()) - int((sw & ~right_b & ok).sum()), sw

        h_net, h_sw = net(np.ones(n, bool), max(grid, key=lambda t: net(np.ones(n, bool), t)[0]))
        fixes, breaks = [], []
        rng = np.random.default_rng(0)
        for _ in range(seeds):
            half = rng.permutation(n) < n // 2
            for fit_m, use_m in ((half, ~half), (~half, half)):
                _, sw = net(use_m, max(grid, key=lambda t: net(fit_m, t)[0]))
                fixes.append((sw & right_b & ~ok).sum())
                breaks.append((sw & ~right_b & ok).sum())
        nf, nb = 2 * np.mean(fixes), 2 * np.mean(breaks)  # per full set (each half is half the photos)
        top30 = np.flatnonzero(has)[np.argsort(-gate[has])][:30]
        lines.append(f"    switch {rule:20s}: cross-fitted fixes {nf:.0f} / breaks {nb:.0f}, net {100 * (nf - nb) / n:+.2f} pts "
                     f"(precision {nf / max(nf + nb, 1):.0%}); hindsight best t: net {100 * h_net / n:+.2f} pts on {h_sw.sum()} switches; "
                     f"30 most confident: {int((right_b & ~ok)[top30].sum())} fixes / {int((~right_b & ok)[top30].sum())} breaks")
    return lines


def _bucket_lines(preds: dict, te: dict, cx: dict) -> list[str]:
    out = []
    for b, nm in enumerate(BUCKETS):
        m = cx["bucket"][te["photo"]] == b
        out.append(f"  AUC within bucket '{nm}': " + ", ".join(f"{name.split()[0]} {_auc(p[m], te['y'][m]):.3f}" for name, p in preds.items()))
    return out


def evidence(tag: str, train_tag: str = "train_setft_own") -> None:
    """Fit P(restricted top-1 right) on train searches with prior features A and with A + result features B; on `tag` (dev3k): AUC and the
    switch rule (_switch_lines; the pipeline's confidence on train is in-sample, so thresholds are cross-fitted on `tag`)."""

    cx = _context(tag)
    lines = [f"Step 2, is a search's result evidence? fit on {train_tag}, evaluated on {tag} ({cx['n']} photos; pipeline "
             f"{100 * cx['ok'].mean():.1f}, area oracle {100 * cx['area_ok'].mean():.1f})"]
    for budget in (8, 16, None):
        tr, te = _features(train_tag, budget), _features(tag, budget)
        fams = {"A (prior)": (tr["A"], te["A"]), "A+B (prior + result)": (np.hstack([tr["A"], tr["B"]]), np.hstack([te["A"], te["B"]]))}
        lines.append(f"\n== budget {budget or 'all'}: {len(tr['y'])} train / {len(te['y'])} {tag} searches, right {100 * tr['y'].mean():.1f} / "
                     f"{100 * te['y'].mean():.1f}%")
        preds = {}
        for name, (xtr, xte) in fams.items():
            preds[name] = _fit(xtr, tr["y"])(xte)
            lines += _switch_lines(name, preds[name], te, cx)
        lines += _bucket_lines(preds, te, cx)
    text = "\n".join(lines)
    (OUT / f"evidence_{tag}.txt").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------- step 2b: MegaLoc of the restricted top-1

def megaloc(tag: str, batch: int = 64, workers: int = 3) -> None:
    """MegaLoc similarity (query, restricted top-1) for every search of `tag` -> artifacts/search/megaloc_{tag}.npz (NaN: empty search or
    unreadable photo). Queries: the cached artifacts/agent/megaloc_queries_{tag}.npy (megaloc_gate, same photo order)."""

    import torch
    from torch.utils.data import DataLoader

    from . import megaloc_gate as mg

    base, mg.CKPT_TAG = tag.split("_", 1)
    z = np.load(OUT / f"searches_{tag}.npz")
    r1, photo = z["rows"][:, 0], z["photo"]
    need = np.unique(r1[r1 >= 0])
    ids, nq, dataset = mg._image_ids(base, need)
    q = torch.as_tensor(np.load(AGENT / f"megaloc_queries_{tag}.npy"), device="cuda").float()
    assert len(q) == nq
    model = mg._model()
    emb = torch.zeros((len(need), q.shape[1]), dtype=torch.float16)
    blank = np.zeros(len(need), bool)
    t = time.perf_counter()
    with torch.inference_mode():
        for b, (i, x) in enumerate(DataLoader(dataset(ids[nq:]), batch_size=batch, num_workers=workers)):
            blank[i.numpy()] = (x.flatten(1) == 0).all(1).numpy()
            emb[i] = model(x.cuda()).half().cpu()
            if b % 100 == 0:
                print(f"  {(b + 1) * batch} / {len(need)} photos, {time.perf_counter() - t:.0f} s", flush=True)
    sim = np.full(len(r1), np.nan, np.float32)
    has = r1 >= 0
    pos = np.searchsorted(need, r1[has])
    e = emb[torch.as_tensor(pos)].float().cuda()
    sim[has] = (e * q[torch.as_tensor(photo[has], device="cuda")]).sum(1).cpu().numpy()
    sim[np.flatnonzero(has)[blank[pos]]] = np.nan
    np.savez(OUT / f"megaloc_{tag}.npz", sim=sim)
    print(f"-> {OUT / f'megaloc_{tag}.npz'}: {len(need)} photos ({blank.sum()} unreadable) in {time.perf_counter() - t:.0f} s", flush=True)


def evidence_megaloc(tag: str, folds: int = 5) -> None:
    """Step 2b on `tag` alone (no train MegaLoc): A+B vs A+B+M (MegaLoc of the restricted top-1, a missing flag, and minus the best MegaLoc
    among the photo's searches in the budget), cross-fitted over photo folds; same switch report as evidence."""

    cx = _context(tag)
    sim_all = np.load(OUT / f"megaloc_{tag}.npz")["sim"]
    lines = [f"Step 2b, MegaLoc of the restricted top-1 as result evidence, {tag} ({cx['n']} photos), {folds}-fold cross-fitting by photo"]
    fold = np.random.default_rng(0).permutation(cx["n"]) % folds
    for budget in (8, 16, None):
        te = _features(tag, budget)
        sim = sim_all[te["search"]]
        filled = np.nan_to_num(sim, nan=0.0)
        best_in_photo = np.full(cx["n"], -1.0)
        np.maximum.at(best_in_photo, te["photo"], filled)
        M = np.stack([filled, np.isnan(sim), filled - best_in_photo[te["photo"]]], 1).astype(np.float32)
        ab = np.hstack([te["A"], te["B"]])
        fams = {"A+B (prior + result)": ab, "A+B+M (+ MegaLoc)": np.hstack([ab, M])}
        lines.append(f"\n== budget {budget or 'all'}: {len(te['y'])} searches, right {100 * te['y'].mean():.1f}%; MegaLoc AUC alone "
                     f"{_auc(filled, te['y']):.3f}")
        preds = {}
        for name, x in fams.items():
            p = np.zeros(len(x), np.float32)
            for f in range(folds):
                held = fold[te["photo"]] == f
                p[held] = _fit(x[~held], te["y"][~held])(x[held])
            preds[name] = p
            lines += _switch_lines(name, p, te, cx)
        lines += _bucket_lines(preds, te, cx)
    text = "\n".join(lines)
    (OUT / f"evidence_megaloc_{tag}.txt").write_text(text + "\n")
    print(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cmds = {"speed": speed, "vocab": vocab, "run": run, "ceiling": ceiling, "evidence": evidence, "megaloc": megaloc,
            "evidence-megaloc": evidence_megaloc}
    parser.add_argument("cmd", choices=tuple(cmds))
    parser.add_argument("--tag", default="dev3k_setft_own")
    args = parser.parse_args(argv)
    cmds[args.cmd](args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
