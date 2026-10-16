# Learned ranking of text-search candidates (lesson 116-118 follow-up): on commons26 dev a right place (< 25 km) is among the search results of
# 43% of photos (phrases, 1-3 word spans, OSM names) but the pipeline's answer is right on 25%; 23% have a right candidate and a wrong answer.
# Candidates come from four sources searched separately (GeoNames, Overture, Flickr titles / tags, OSM names), each query's matches clustered per
# source, then merged per photo within 25 km; a listwise ranker (with a 'none of them' option) is cross-validated on dev, grouped by uploader.
# Tuned on commons26 dev (user, 2026-10-16); dev3k for regressions.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.candidate_ranker gather --tag commons26 --split dev   (CPU, ~1 h)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.candidate_ranker cv --tag commons26 --split dev

"""gather: per photo, every 4B phrase (text, place names, queries) and its spans searched in each source -> per (query, source) its top clusters
          (artifacts/agent/cands_{tag}_{split}.jsonl, appended per photo, resumable).
cv:       candidates, features, the ranker under grouped 5-fold CV; top-1 accuracy, a confidence switch against the pipeline's answer, source ablations."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
SOURCES = ("GeoNames", "Overture", "Flickr", "OSM")
CLUSTER_KM = 25.0
TOP = 8  # clusters kept per (query, source)
SAMPLE = 500  # matches clustered per (query, source)
MAX_MATCHES = 10_000  # matches counted per (query, source)
MAX_CANDIDATES = 64  # per photo, by support (ranking cost)


def _hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _greedy(lat: np.ndarray, lon: np.ndarray, w: np.ndarray, top: int = TOP) -> list[tuple[int, np.ndarray]]:
    """text_search._cluster's greedy 25 km clustering: (centre index, member indices) by weight."""

    d = _hav(lat[:, None], lon[:, None], lat[None], lon[None]) < CLUSTER_KM
    alive, out = np.ones(len(lat), bool), []
    while alive.any() and len(out) < top:
        score = d @ (w * alive)
        score[~alive] = -1
        c = int(score.argmax())
        members = np.flatnonzero(d[c] & alive)
        alive[members] = False
        out.append((c, members))
    return out


def _sample(n: int, key: str) -> np.ndarray:
    if n <= SAMPLE:
        return np.arange(n)
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    return np.sort(rng.choice(n, SAMPLE, replace=False))


class _Searcher:
    def __init__(self) -> None:
        import sqlite3

        from ..tools.maps import _osm_path
        from .text_search import TextSearch

        self.ts = TextSearch()
        self.osm = sqlite3.connect(f"file:{_osm_path()}?mode=ro", uri=True, check_same_thread=False)

    def _osm(self, words: list[str]) -> tuple[list, int, bool]:
        for loose in (False, True):
            if loose and len(words) < 2:
                break
            match = " ".join(words) if loose else '"' + " ".join(words) + '"'
            try:
                ids = [r for (r,) in self.osm.execute(f"SELECT rowid FROM names WHERE names MATCH ? LIMIT {MAX_MATCHES}", (match,)).fetchall()]
            except Exception:  # noqa: BLE001  (FTS syntax)
                return [], 0, False
            if ids:
                n = len(ids)
                ids = [ids[i] for i in _sample(n, match)]
                rows = []
                for s in range(0, len(ids), 900):
                    chunk = ids[s:s + 900]
                    rows += self.osm.execute(f"SELECT key, lat, lon FROM places WHERE rowid IN ({','.join('?' * len(chunk))})", chunk).fetchall()
                return rows, n, loose
        return [], 0, False

    def query(self, q: str, exclude) -> dict:
        """Per source: match count, loose flag, top clusters [lat, lon, weight, share, rank, exact, cc]."""

        from .text_search import norm

        kind, places, photos, counts, hint, loose = self.ts.matches(q, None, exclude)
        qn = norm(q)
        out = {"kind": kind, "hint": hint, "src": {}}
        n_places = counts.get("places", 0) or next((v for k, v in counts.items() if k.startswith("places")), 0)
        groups = {"GeoNames": [p for p in places if not p[5]], "Overture": [p for p in places if p[5]]}
        pts = {s: (np.asarray([p[1] for p in g], float), np.asarray([p[2] for p in g], float), np.ones(len(g)),
                   np.asarray([norm(p[0]) == qn for p in g]), round(n_places * len(g) / max(len(places), 1)), loose) for s, g in groups.items()}
        if photos:
            per_user = collections.Counter(p[1] for p in photos)
            pts["Flickr"] = (np.asarray([p[2] for p in photos], float), np.asarray([p[3] for p in photos], float),
                             np.asarray([1.0 / per_user[p[1]] for p in photos]), np.zeros(len(photos), bool), counts.get("Flickr photos", len(photos)), loose)
        if kind == "name" and qn:
            rows, n, lo = self._osm(qn.split())
            if rows:
                pts["OSM"] = (np.asarray([r[1] for r in rows], float), np.asarray([r[2] for r in rows], float), np.ones(len(rows)),
                              np.asarray([r[0] == qn for r in rows]), n, lo)
        for s, (la, lon, w, exact, n, lo) in pts.items():
            if not len(la):
                continue
            total, cl = w.sum(), []
            for rank, (c, m) in enumerate(_greedy(la, lon, w)):
                lab = self.ts.label(float(la[c]), float(lon[c]))
                cl.append([round(float(la[c]), 5), round(float(lon[c]), 5), round(float(w[m].sum()), 3), round(float(w[m].sum() / total), 4), rank,
                           int(exact[m].sum()), lab.rsplit(", ", 1)[-1] if ", " in lab else ""])
            out["src"][s] = {"n": int(n), "loose": bool(lo), "clusters": cl}
        return out


def _work(job):
    global _S
    try:
        _S
    except NameError:
        _S = _Searcher()
    image_id, queries, own = job
    return {"image_id": image_id, "queries": [{"q": q, "from": src, **_S.query(q, own)} for q, src in queries]}


def _queries(phrases: dict[str, str]) -> list[tuple[str, str]]:
    from .read_constrain import _spans
    from .text_search import norm

    out, seen = [], set()
    for q, src in phrases.items():
        out.append((q, src))
        seen.add(norm(q))
    for q in phrases:
        for s in _spans(q):
            if norm(s) and norm(s) not in seen:
                seen.add(norm(s))
                out.append((s, "span"))
    return out


def gather(tag: str, split: str | None, workers: int = 4) -> None:
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    from .region_evidence import _owns, _raw_phrases

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    if split:
        sp = {e["image_id"]: e.get("split") for e in json.loads((PHOTO_SETS / f"{tag}_setft_own" / "photos.json").read_text(encoding="utf-8"))}
        photos = [e for e in photos if sp.get(e["image_id"]) == split]
    phrases = _raw_phrases(tag, "4b")
    owns = _owns(photos)
    dst = OUT / f"cands_{tag}_{split or 'all'}.jsonl"
    done = {json.loads(line)["image_id"] for line in dst.read_text().splitlines()} if dst.exists() else set()
    jobs = [(e["image_id"], _queries(phrases.get(e["image_id"], {})), owns[e["image_id"]]) for e in photos if e["image_id"] not in done]
    print(f"{len(jobs)} photos to search ({len(done)} done), {sum(len(j[1]) for j in jobs)} queries", flush=True)
    t0 = time.time()
    with dst.open("a") as f, ProcessPoolExecutor(workers) as pool:
        for n, rec in enumerate(pool.map(_work, jobs, chunksize=2), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"{n} photos {time.time() - t0:.0f}s", flush=True)
    print(f"-> {dst}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------- ranking

FEATURES = ["queries", "phrase queries", "span queries", "sources", *[f"has {s}" for s in SOURCES], *[f"weight {s}" for s in SOURCES],
            "specificity", "sharpness", "max share", "top of a query", "exact name", "loose only", "words", "from place name", "from text",
            "from query", "km to area", "near area 1", "near an area", "Pinpoint mass", "country = 4B", "country != 4B", "population",
            "phone / domain", "support", "support rank", "candidates", "photo queries"]


def _context(tag: str, split: str, ckpt: str = "setft_own") -> dict:
    """Everything per photo besides the search results: truth, uploader group, Pinpoint's areas and set probabilities, the 4B's country, the
    pipeline's errors, and a town-population lookup."""

    from scipy.spatial import cKDTree

    from .read_constrain import _country_codes
    from .text_search import TextSearch

    photos = json.loads((PHOTO_SETS / f"{tag}_{ckpt}" / "photos.json").read_text(encoding="utf-8"))
    pidx = {e["image_id"]: k for k, e in enumerate(photos)}
    bench = {r["IMG_ID"]: r for r in csv.DictReader(open(f"data/pinpoint/{tag}/release/benchmark.csv", encoding="utf-8"))}
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    areas = collections.defaultdict(list)
    for j in np.flatnonzero(zm["source"] == 0):
        areas[int(zm["photo"][j])].append((int(zm["rank"][j]), zm["center"][j].astype(np.float64)))
    areas = {k: np.asarray([c for _, c in sorted(v, key=lambda x: x[0])]) for k, v in areas.items()}
    sp = np.load(Path("artifacts/pinpoint_set") / f"set_probs_{tag}_{ckpt}.npz")
    probs, plat, valid = sp[f"{tag}_{ckpt}_probs"], sp[f"{tag}_{ckpt}_latlon"], sp[f"{tag}_{ckpt}_valid"]
    cc_of = _country_codes()
    guess = {k: cc_of.get(str((v.get("parsed") or {}).get("country") or "").strip().lower())
             for k, v in json.loads((OUT / f"observe_{tag}_4b.json").read_text(encoding="utf-8")).items()}
    km = np.load(OUT / f"action_agent_apply_km_{ckpt}_1km0.5_{tag}.npz")
    final = km[f"{tag}:{ckpt}|RL policy, lambda 0"]
    t = np.asarray(TextSearch().places.execute("SELECT lat, lon, pop FROM towns").fetchall(), np.float64)
    return {"photos": photos, "pidx": pidx, "bench": bench, "areas": areas, "probs": probs * valid, "plat": plat, "guess": guess, "final": final,
            "towns": (cKDTree(_xyz(t[:, 0], t[:, 1])), np.log10(1 + t[:, 2]))}


def _xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _candidates(rec: dict, ctx: dict, sources: tuple[str, ...] = SOURCES) -> tuple[np.ndarray, np.ndarray]:
    """One photo's merged candidates: (centres [n, 2], features [n, len(FEATURES)])."""

    items = []  # (query index, source, lat, lon, weight, share, rank, exact, cc, n, loose)
    for qi, q in enumerate(rec["queries"]):
        for s, r in q["src"].items():
            if s in sources:
                items += [(qi, s, c[0], c[1], c[2], c[3], c[4], c[5], c[6], r["n"], r["loose"]) for c in r["clusters"]]
    if not items:
        return np.zeros((0, 2)), np.zeros((0, len(FEATURES)), np.float32)
    items.sort(key=lambda x: -x[5] / math.log2(2 + x[9]))
    lat = np.asarray([x[2] for x in items])
    lon = np.asarray([x[3] for x in items])
    centre, owner = [], np.empty(len(items), int)
    for k in range(len(items)):  # greedy: each cluster joins the first candidate within 25 km, else starts one
        if centre:
            c = np.asarray(centre)
            d = _hav(lat[k], lon[k], c[:, 0], c[:, 1])
            j = int(d.argmin())
            if d[j] < CLUSTER_KM:
                owner[k] = j
                continue
        owner[k] = len(centre)
        centre.append((lat[k], lon[k]))
    C = np.asarray(centre)
    i = ctx["pidx"][rec["image_id"]]
    areas = ctx["areas"].get(i, np.zeros((0, 2)))
    guess = ctx["guess"].get(rec["image_id"])
    tree, logpop = ctx["towns"]
    queries = rec["queries"]
    X = np.zeros((len(C), len(FEATURES)), np.float32)
    support = np.zeros(len(C))
    for j in range(len(C)):
        m = [items[k] for k in np.flatnonzero(owner == j)]
        qs = {x[0] for x in m}
        phr = {qi for qi in qs if queries[qi]["from"] != "span"}
        src = {x[1] for x in m}
        sharp = [x[5] / math.log2(2 + x[9]) for x in m]
        best_q = collections.defaultdict(float)
        for x, s in zip(m, sharp):
            best_q[x[0]] = max(best_q[x[0]], s)
        support[j] = sum(best_q.values())
        ccs = collections.Counter(x[8] for x in m if x[8])
        cc = ccs.most_common(1)[0][0] if ccs else ""
        d_area = _hav(C[j, 0], C[j, 1], areas[:, 0], areas[:, 1]) if len(areas) else np.full(1, 2e4)
        pm = float(ctx["probs"][i][_hav(C[j, 0], C[j, 1], ctx["plat"][i][:, 0], ctx["plat"][i][:, 1]) < CLUSTER_KM].sum())
        near = tree.query_ball_point(_xyz(C[j, 0], C[j, 1]), CLUSTER_KM / 6371.0)
        froms = {queries[qi]["from"].split(" ")[0] for qi in qs}
        X[j] = [math.log1p(len(qs)), math.log1p(len(phr)), math.log1p(len(qs - phr)), len(src), *[s in src for s in SOURCES],
                *[math.log1p(sum(x[4] for x in m if x[1] == s)) for s in SOURCES],
                min(math.log10(1 + x[9]) for x in m), max(sharp), max(x[5] for x in m), any(x[6] == 0 for x in m), any(x[7] > 0 for x in m),
                all(x[10] for x in m), max(len(queries[qi]["q"].split()) for qi in qs), "place" in froms, "text" in froms, "query" in froms,
                math.log1p(min(float(d_area.min()), 2e4)), bool(len(areas)) and d_area[0] < CLUSTER_KM, bool((d_area < CLUSTER_KM).any()), pm,
                bool(guess) and cc == guess, bool(guess) and bool(cc) and cc != guess, float(logpop[near].max()) if near else 0.0,
                any(queries[qi]["kind"] != "name" for qi in qs), support[j], 0, math.log1p(len(C)), math.log1p(len(queries))]
    X[:, FEATURES.index("support rank")] = np.log1p(np.argsort(np.argsort(-support)))
    keep = np.argsort(-support, kind="stable")[:MAX_CANDIDATES]
    return C[keep], X[keep]


def _fit_predict(Xtr, Mtr, Ytr, Xte, Mte, hidden: int, seed: int = 0, epochs: int = 400, wd: float = 1e-3) -> np.ndarray:
    """Listwise softmax over each photo's candidates plus a 'none' option; returns P(candidate) [photos, max candidates] for the test photos."""

    import torch

    torch.manual_seed(seed)
    mu, sd = Xtr[Mtr].mean(0), Xtr[Mtr].std(0) + 1e-6
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    d = Xtr.shape[-1]
    net = torch.nn.Linear(d, 1) if not hidden else torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    none = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([*net.parameters(), none], lr=0.01, weight_decay=wd)
    X, M, Y = f(Xtr), torch.tensor(Mtr), torch.tensor(Ytr)
    target = torch.cat([Y.float(), (~Y.any(1, keepdim=True)).float()], 1)
    for _ in range(epochs):
        logits = torch.cat([net(X).squeeze(-1).masked_fill(~M, -1e9), none.expand(len(X), 1)], 1)
        logp = logits.log_softmax(1)
        loss = -(torch.logsumexp(logp.masked_fill(target == 0, -1e9), 1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = torch.cat([net(f(Xte)).squeeze(-1).masked_fill(~torch.tensor(Mte), -1e9), none.expand(len(Xte), 1)], 1)
        return logits.softmax(1)[:, :-1].numpy()


def _pad(Xs: list[np.ndarray], Ys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max(1, max(len(x) for x in Xs))
    X = np.zeros((len(Xs), n, len(FEATURES)), np.float32)
    M = np.zeros((len(Xs), n), bool)
    Y = np.zeros((len(Xs), n), bool)
    for k, (x, y) in enumerate(zip(Xs, Ys)):
        X[k, :len(x)], M[k, :len(x)], Y[k, :len(y)] = x, True, y
    return X, M, Y


def _folds(groups: list[str], k: int = 5) -> np.ndarray:
    return np.asarray([int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % k for g in groups])


def cv(tag: str, split: str) -> None:
    ctx = _context(tag, split)
    recs = [json.loads(line) for line in (OUT / f"cands_{tag}_{split}.jsonl").read_text().splitlines()]
    ids = [r["image_id"] for r in recs]
    truth = np.asarray([ctx["photos"][ctx["pidx"][i]]["truth"] for i in ids], np.float64)
    fin_right = np.asarray([ctx["final"][ctx["pidx"][i]] < CLUSTER_KM for i in ids])
    fold = _folds([ctx["bench"][i]["group"] or i for i in ids])
    tiers = np.asarray([ctx["bench"][i]["tier"] for i in ids])
    n = len(ids)
    print(f"{tag} {split}: {n} photos; pipeline (RL lambda 0) < 25 km {fin_right.mean():.1%}")

    def run(sources, hidden_list=(0, 16), report=True):
        Cs, Xs, Ys = [], [], []
        for r, t in zip(recs, truth):
            C, X = _candidates(r, ctx, sources)
            Cs.append(C)
            Xs.append(X)
            Ys.append(_hav(t[0], t[1], C[:, 0], C[:, 1]) < CLUSTER_KM if len(C) else np.zeros(0, bool))
        X, M, Y = _pad(Xs, Ys)
        oracle = Y.any(1)
        heur = np.asarray([bool(len(y)) and y[int(np.argmax(x[:, FEATURES.index("support")]))] for x, y in zip(Xs, Ys)])
        out = {}
        for hidden in hidden_list:
            P = np.zeros(M.shape, np.float32)
            for k in range(5):
                tr, te = fold != k, fold == k
                P[te] = np.mean([_fit_predict(X[tr], M[tr], Y[tr], X[te], M[te], hidden, seed) for seed in range(1)], 0)
            top = P.argmax(1)
            top_right = Y[np.arange(n), top] & M[np.arange(n), top]
            out[hidden] = (P, top_right)
        if report:
            print(f"\nsources {'+'.join(sources)}: candidates per photo median {np.median([len(x) for x in Xs]):.0f}; some candidate right {oracle.mean():.1%}; "
                  f"top by support (heuristic) {heur.mean():.1%}")
            for hidden, (P, top_right) in out.items():
                print(f"  ranker {'linear' if not hidden else f'MLP {hidden}'}: top candidate right {top_right.mean():.1%} "
                      f"(of photos with a right candidate {top_right[oracle].mean():.1%})")
        return out, Y, M, oracle

    out, Y, M, oracle = run(SOURCES)
    best = max(out, key=lambda h: out[h][1].mean())
    P, top_right = out[best]
    conf = P.max(1)
    # calibration
    print(f"\ncalibration ({'linear' if not best else f'MLP {best}'}): confidence bin -> photos, top candidate right, pipeline right")
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .7), (.7, 1.01)):
        s = (conf >= lo) & (conf < hi)
        if s.any():
            print(f"  [{lo:.1f}, {hi:.1f}): {s.sum():4d}  {top_right[s].mean():5.1%}  {fin_right[s].mean():5.1%}")
    # switch: answer the ranker's top candidate when its confidence > t; t chosen on the other folds' out-of-fold predictions
    grid = np.linspace(0.05, 0.95, 37)
    gain_of = lambda s, t: (np.where(conf[s] > t, top_right[s], fin_right[s]).astype(float) - fin_right[s]).sum()
    new = fin_right.astype(float).copy()
    ts = []
    for k in range(5):
        other = fold != k
        t = grid[int(np.argmax([gain_of(other, t) for t in grid]))]
        ts.append(t)
        te = fold == k
        new[te] = np.where(conf[te] > t, top_right[te], fin_right[te])
    diff = new - fin_right
    print(f"\nswitch (threshold chosen on other folds, {', '.join(f'{t:.2f}' for t in ts)}): {fin_right.mean():.1%} -> {new.mean():.1%} "
          f"({100 * diff.mean():+.1f} pts, SE {100 * diff.std() / math.sqrt(n):.1f}); switched {np.mean([c > t for c, t in zip(conf, np.asarray(ts)[fold])]):.1%} of photos, "
          f"fixes {(diff > 0).sum()}, breaks {(diff < 0).sum()}")
    print("  threshold curve (all photos, in-sample choice):", ", ".join(f"{t:.2f}: {100 * gain_of(np.ones(n, bool), t) / n:+.1f}" for t in grid[::4]))
    for g in ("landmark", "city", "region", "none"):
        s = tiers == g
        print(f"  tier {g:8s} ({s.sum()}): {fin_right[s].mean():.1%} -> {new[s].mean():.1%}")
    # linear weights for reading
    def switch_gain(P, top_right):
        conf, grid, new = P.max(1), np.linspace(0.05, 0.95, 37), fin_right.astype(float).copy()
        for k in range(5):
            other, te = fold != k, fold == k
            t = grid[int(np.argmax([(np.where(conf[other] > t, top_right[other], fin_right[other]).astype(float) - fin_right[other]).sum() for t in grid]))]
            new[te] = np.where(conf[te] > t, top_right[te], fin_right[te])
        return 100 * (new - fin_right).mean()

    print(f"\nsource ablations: top candidate right, switch gain (all four: {top_right.mean():.1%}, {switch_gain(P, top_right):+.1f} pts)")
    for drop in SOURCES:
        o, _, _, orc = run(tuple(s for s in SOURCES if s != drop), (best,), report=False)
        print(f"  without {drop:9s}: some candidate right {orc.mean():.1%}; top right {o[best][1].mean():.1%}; switch {switch_gain(*o[best]):+.1f} pts")
    for only in SOURCES:
        o, _, _, orc = run((only,), (best,), report=False)
        print(f"  {only:9s} only: some candidate right {orc.mean():.1%}; top right {o[best][1].mean():.1%}; switch {switch_gain(*o[best]):+.1f} pts")
    np.savez(OUT / f"cand_ranker_{tag}_{split}.npz", ids=np.asarray(ids), conf=conf, top_right=top_right, oracle=oracle, fin_right=fin_right, switched=new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gather", "cv"])
    ap.add_argument("--tag", default="commons26")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "gather":
        gather(a.tag, a.split or None, a.workers)
    else:
        cv(a.tag, a.split)


if __name__ == "__main__":
    main()
