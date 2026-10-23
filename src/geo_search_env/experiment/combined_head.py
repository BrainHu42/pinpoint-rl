"""Combined answer head, step A (plan of 2026-10-21): one answer head over Pinpoint's areas, the clue areas and text-search places, with text
evidence as per-area features, trained on MP16 plus fresh Commons photos. Cross-validated on commons26 dev (folds by uploader), dev3k for
regressions.

Text evidence (lessons 118-121; computed from the cached four-source text searches, artifacts/agent/cands_commons26_dev.jsonl):
  - per token (area, clue area or text place) at its centre: rare-name support (distinct 4B phrases / spans whose GeoNames / Overture / OSM
    matches number <= 5 / 20 / 100 worldwide and have a match within 25 km; lesson 118's rule as features, the head learns the cut), the text
    ranker's probability mass and best probability within 25 km and whether its top place is within 25 km (lesson 119's listwise ranker,
    linear, refit per fold so no fold sees its own labels), a text-place flag and a "text features available" flag;
  - text-place tokens: the ranker's top 3 places that are not within 25 km of an existing token, answering like an unzoomed area (the global
    pool's best candidate within 25 km of the place, else the place). Like clue areas they appear only once the photo is read, and the text
    columns are hidden before the read (action_agent.view).
MP16 train photos have no text features (flag 0); the head learns the text columns from the commons26 dev photos of the training folds.

Cells (each the lesson-111 action agent, --one-km 0.5, setft_own: answer head + RL policy lambda 0, and the fixed "take everything" recipe):
  base            trained on MP16 train only (the current pipeline);
  +commons        MP16 train + commons26 dev training folds (oversampled --weight x), no text features: the in-domain effect;
  +commons+text   the same with the text features and text-place tokens.

  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.combined_head features   (CPU, ~2 min; caches the text candidates)
  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.combined_head cv [--weight 10] [--seeds 2]

stack: the MP16 side of the plan, the earlier sub-threshold head inputs together on train / dev3k (graph diffusion mass, lesson 127; crop
views vs each area's photos, lesson 89, `crop_look features` with CKPT_TAG setft_own), each visible in every state.
  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.combined_head stack [--seeds 3]

adapter: the MP16-trained head frozen and ~10 parameters fit on commons26 dev (a scale on its logits, a weight per text column, a
text-place bias), cross-fitted by uploader: whether the text evidence carries signal once estimation noise is small.
  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.combined_head adapter [--seeds 2]
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import action_agent as aa

OUT = Path("artifacts/agent")
TAG, CKPT = "commons26dev", "setft_own"  # the evaluation set
# per Commons photo set: (cached text searches, benchmark CSV, 4B clue file); commons26 dev's searches are lesson 119's
SETS = {"commons26dev": (OUT / "cands_commons26_dev.jsonl", Path("data/pinpoint/commons26/release/benchmark.csv"), OUT / "observe_commons26_4b.json"),
        "commons_train": (OUT / "cands_commons_train.jsonl", Path("data/pinpoint/commons_train/release/benchmark.csv"), OUT / "observe_commons_train_4b.json"),
        "dev3k": (OUT / "cands_dev3k.jsonl", None, OUT / "observe_dev3k_4b.json")}  # MP16 photos: no CSV (ranker folds by photo, no continent)
SOURCES = ("GeoNames", "Overture", "Flickr", "OSM")
NAME_SOURCES = ("GeoNames", "Overture", "OSM")
CLUSTER_KM = 25.0
MAX_CANDIDATES = 64
RARE = (5, 20, 100)
N_TEXT_TOKENS = 3
TEXT_COLS = [*[f"rare names <= {t}" for t in RARE], "ranker mass", "ranker max", "ranker top here", "text place", "text available"]
RANKER_FEATURES = ["queries", "phrase queries", "span queries", "sources", *[f"has {s}" for s in SOURCES], *[f"weight {s}" for s in SOURCES],
                   "specificity", "sharpness", "max share", "top of a query", "exact name", "loose only", "words", "from place name", "from text",
                   "from query", "km to area", "near area 1", "near an area", "Pinpoint mass", "country = 4B", "country != 4B", "population",
                   "phone / domain", "support", "support rank", "candidates", "photo queries"]


def _hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _country_codes() -> dict[str, str]:
    out = {}
    for line in Path("data/pinpoint/geonames/countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split("\t")
            out[f[4].lower()] = f[0]
    out |= {"united states of america": "US", "usa": "US", "uk": "GB", "england": "GB", "scotland": "GB", "wales": "GB", "northern ireland": "GB",
            "russia": "RU", "south korea": "KR", "korea": "KR", "czechia": "CZ", "czech republic": "CZ", "vietnam": "VN", "türkiye": "TR",
            "turkey": "TR", "taiwan": "TW", "the netherlands": "NL", "holland": "NL", "iran": "IR", "syria": "SY", "laos": "LA"}
    return out


def _cache(tag: str) -> Path:
    return OUT / f"combined_text_{tag}_{CKPT}.npz"


def _photos(tag: str = TAG) -> list[dict]:
    return json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT}" / "photos.json").read_text(encoding="utf-8"))


def _context(photos: list[dict], tag: str) -> dict:
    """What lesson 119's ranker features need besides the search results: Pinpoint's areas and set probabilities, the 4B's country, towns."""

    from scipy.spatial import cKDTree

    from .text_search import TextSearch

    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT}.npz")
    areas = collections.defaultdict(list)
    for j in np.flatnonzero(zm["source"] == 0):
        areas[int(zm["photo"][j])].append((int(zm["rank"][j]), zm["center"][j].astype(np.float64)))
    areas = {k: np.asarray([c for _, c in sorted(v, key=lambda x: x[0])]) for k, v in areas.items()}
    sp = np.load(Path("artifacts/pinpoint_set") / f"set_probs_{tag}_{CKPT}.npz")
    key = f"{tag}_{CKPT}"
    cc_of = _country_codes()
    guess = {k: cc_of.get(str((v.get("parsed") or {}).get("country") or "").strip().lower())
             for k, v in json.loads(SETS[tag][2].read_text(encoding="utf-8")).items()}
    t = np.asarray(TextSearch().places.execute("SELECT lat, lon, pop FROM towns").fetchall(), np.float64)
    return {"areas": areas, "probs": sp[f"{key}_probs"] * sp[f"{key}_valid"], "plat": sp[f"{key}_latlon"], "guess": guess,
            "towns": (cKDTree(_xyz(t[:, 0], t[:, 1])), np.log10(1 + t[:, 2]))}


def _candidates(rec: dict, i: int, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """One photo's merged text candidates (lesson 119, candidate_ranker._candidates at 7775c85): centres [n, 2], features [n, 37]."""

    items = []  # (query index, source, lat, lon, weight, share, rank, exact, cc, n, loose)
    for qi, q in enumerate(rec["queries"]):
        for s, r in q["src"].items():
            items += [(qi, s, c[0], c[1], c[2], c[3], c[4], c[5], c[6], r["n"], r["loose"]) for c in r["clusters"]]
    if not items:
        return np.zeros((0, 2)), np.zeros((0, len(RANKER_FEATURES)), np.float32)
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
    areas = ctx["areas"].get(i, np.zeros((0, 2)))
    guess = ctx["guess"].get(rec["image_id"])
    tree, logpop = ctx["towns"]
    queries = rec["queries"]
    X = np.zeros((len(C), len(RANKER_FEATURES)), np.float32)
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
    X[:, RANKER_FEATURES.index("support rank")] = np.log1p(np.argsort(np.argsort(-support)))
    keep = np.argsort(-support, kind="stable")[:MAX_CANDIDATES]
    return C[keep], X[keep]


def features(tag: str = TAG) -> None:
    """Text candidates, ranker features and name matches of every photo of a Commons photo set (photo-set order) -> _cache(tag)."""

    cands, bench_csv, _ = SETS[tag]
    photos = _photos(tag)
    pidx = {e["image_id"]: k for k, e in enumerate(photos)}
    recs = {}
    for line in cands.read_text().splitlines():
        r = json.loads(line)
        recs[r["image_id"]] = r
    ctx = _context(photos, tag)
    C, X, Y, coff = [], [], [], [0]
    names, noff = [], [0]  # (query index, match count, lat, lon) of every name-source cluster
    for i, e in enumerate(photos):
        r = recs.get(e["image_id"], {"image_id": e["image_id"], "queries": []})
        c, x = _candidates(r, i, ctx)
        C.append(c)
        X.append(x)
        Y.append(_hav(e["truth"][0], e["truth"][1], c[:, 0], c[:, 1]) < CLUSTER_KM if len(c) else np.zeros(0, bool))
        coff.append(coff[-1] + len(c))
        nm = [(qi, s["n"], cl[0], cl[1]) for qi, q in enumerate(r["queries"]) for src, s in q["src"].items() if src in NAME_SOURCES for cl in s["clusters"]]
        names += nm
        noff.append(noff[-1] + len(nm))
    bench = {}
    if bench_csv:
        with bench_csv.open(encoding="utf-8") as fh:
            bench = {r["IMG_ID"]: r for r in csv.DictReader(fh)}
    groups = [bench.get(e["image_id"], {}).get("group") or e["image_id"] for e in photos]
    np.savez(_cache(tag), C=np.concatenate(C), X=np.concatenate(X), Y=np.concatenate(Y), coff=np.asarray(coff), names=np.asarray(names, np.float64).reshape(-1, 4),
             noff=np.asarray(noff), fold=np.asarray([int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % 5 for g in groups]),
             continent=np.asarray([bench.get(e["image_id"], {}).get("continent", "") for e in photos]), ids=np.asarray([e["image_id"] for e in photos]))
    print(f"{len(photos)} photos ({len(pidx)} ids, {sum(1 for e in photos if e['image_id'] in recs)} with searches); "
          f"candidates per photo median {np.median(np.diff(coff)):.0f}, some right {np.mean([y.any() for y in Y]):.1%} -> {_cache(tag)}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------- searching
# lesson 119's text searches (candidate_ranker gather at 7775c85; spans from read_constrain at c9a8944): every 4B phrase and its 1-3 word spans
# in GeoNames, Overture, Flickr titles / tags (text_search) and OSM names (tools.maps), each query's matches clustered per source

TOP = 8  # clusters kept per (query, source)
SAMPLE = 500  # matches clustered per (query, source)
MAX_MATCHES = 10_000  # matches counted per (query, source)
GENERIC = {"the", "of", "and", "de", "la", "le", "les", "du", "des", "del", "della", "di", "da", "do", "dos", "das", "der", "die", "und", "von", "zu", "am", "im", "an", "in", "on", "at", "by", "for", "to", "a", "monument", "museum", "station", "railway", "church", "cathedral", "chapel", "temple", "mosque", "shrine", "square", "street", "road", "avenue", "park", "garden", "bridge", "tower", "castle", "palace", "center", "centre", "shopping", "mall", "market", "hotel", "restaurant", "cafe", "bar", "school", "university", "college", "hospital", "library", "hall", "house", "building", "memorial", "statue", "fountain", "gate", "lake", "river", "mountain", "beach", "island", "port", "harbour", "harbor", "airport", "bus", "stop", "city", "town", "village", "old", "new", "north", "south", "east", "west", "central", "national", "state", "public", "main", "saint", "st", "plaza", "place", "via", "rue", "strasse", "calle", "piazza", "location", "photo", "view", "interior", "exterior", "entrance", "sign", "front", "festival", "fest", "event", "collection", "exhibition", "gallery", "theatre", "theater", "cinema", "office", "police", "fire", "community", "club"}


def _spans(phrase: str, max_len: int = 3) -> list[str]:
    """Contiguous word spans of a phrase (1-3 words) that aren't only generic words."""

    words = [w.strip(".,;:!?()[]\"'«»“”") for w in phrase.split()]
    words = [w for w in words if w]
    out = []
    for n in range(1, max_len + 1):
        for i in range(len(words) - n + 1):
            span = words[i:i + n]
            if all(w.lower() in GENERIC or len(w) < 3 for w in span):
                continue
            out.append(" ".join(span))
    return out


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
        _S  # noqa: B018
    except NameError:
        _S = _Searcher()
    image_id, queries, own = job
    return {"image_id": image_id, "queries": [{"q": q, "from": src, **_S.query(q, own)} for q, src in queries]}


def _queries(phrases: dict[str, str]) -> list[tuple[str, str]]:
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


def gather(tag: str, workers: int = 4) -> None:
    """The text searches of every photo of a Commons photo set (its 4B clues, observe extract + queries on the base set) -> SETS[tag][0],
    appended per photo, resumable."""

    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    from .region_evidence import _owns, _raw_phrases

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    photos = _photos(tag)
    phrases = _raw_phrases(tag, "4b")
    owns = _owns(photos)
    dst = SETS[tag][0]
    done = {json.loads(line)["image_id"] for line in dst.read_text().splitlines()} if dst.exists() else set()
    jobs = [(e["image_id"], _queries(phrases.get(e["image_id"], {})), owns[e["image_id"]]) for e in photos if e["image_id"] not in done]
    print(f"{len(jobs)} photos to search ({len(done)} done), {sum(len(j[1]) for j in jobs)} queries", flush=True)
    t0 = time.time()
    with dst.open("a") as f, ProcessPoolExecutor(workers) as pool:
        for n, rec in enumerate(pool.map(_work, jobs, chunksize=2), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 500 == 0:
                print(f"{n} photos {time.time() - t0:.0f}s", flush=True)
    print(f"-> {dst}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------- the ranker


def _fit_predict(Xtr, Mtr, Ytr, Xte, Mte, seed: int = 0, epochs: int = 400, wd: float = 1e-3) -> np.ndarray:
    """Lesson 119's linear listwise ranker with a 'none' option; P(candidate) [photos, max candidates] for the test photos."""

    import torch

    torch.manual_seed(seed)
    mu, sd = Xtr[Mtr].mean(0), Xtr[Mtr].std(0) + 1e-6
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    net = torch.nn.Linear(Xtr.shape[-1], 1)
    none = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([*net.parameters(), none], lr=0.01, weight_decay=wd)
    X, M, Y = f(Xtr), torch.tensor(Mtr), torch.tensor(Ytr)
    target = torch.cat([Y.float(), (~Y.any(1, keepdim=True)).float()], 1)
    for _ in range(epochs):
        logits = torch.cat([net(X).squeeze(-1).masked_fill(~M, -1e9), none.expand(len(X), 1)], 1)
        loss = -(torch.logsumexp(logits.log_softmax(1).masked_fill(target == 0, -1e9), 1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = torch.cat([net(f(Xte)).squeeze(-1).masked_fill(~torch.tensor(Mte), -1e9), none.expand(len(Xte), 1)], 1)
        return logits.softmax(1)[:, :-1].numpy()


class Text:
    """The cached text candidates as padded arrays, ranker probabilities that never saw given folds, and per-token text features."""

    def __init__(self, tag: str = TAG) -> None:
        z = dict(np.load(_cache(tag)))  # NpzFile re-reads an array on every access
        self.fold, self.continent, self.ids = z["fold"], z["continent"], z["ids"]
        coff, noff = z["coff"], z["noff"]
        self.n = len(coff) - 1
        w = max(1, int(np.diff(coff).max()))
        self.C = np.zeros((self.n, w, 2))
        self.X = np.zeros((self.n, w, len(RANKER_FEATURES)), np.float32)
        self.M = np.zeros((self.n, w), bool)
        self.Y = np.zeros((self.n, w), bool)
        for i in range(self.n):
            a, b = coff[i], coff[i + 1]
            self.C[i, :b - a], self.X[i, :b - a], self.M[i, :b - a], self.Y[i, :b - a] = z["C"][a:b], z["X"][a:b], True, z["Y"][a:b]
        self.names = [z["names"][noff[i]:noff[i + 1]] for i in range(self.n)]
        self._p: dict = {}

    def probs(self, held: tuple[int, ...]) -> np.ndarray:
        """P per candidate for every photo, from rankers that never saw the folds in `held` nor the photo's own fold."""

        if held not in self._p:
            P = np.zeros(self.M.shape, np.float32)
            for k in range(5):
                te = self.fold == k
                tr = ~np.isin(self.fold, (k, *held))
                if te.any():
                    P[te] = _fit_predict(self.X[tr], self.M[tr], self.Y[tr], self.X[te], self.M[te])
            self._p[held] = np.where(self.M, P, 0.0)
        return self._p[held]

    def at(self, i: int, centre: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Text columns of a token of photo i at `centre` (lat, lon)."""

        out = np.zeros(len(TEXT_COLS), np.float32)
        nm = self.names[i]
        if len(nm):
            near = _hav(centre[0], centre[1], nm[:, 2], nm[:, 3]) < CLUSTER_KM
            for k, t in enumerate(RARE):
                out[k] = math.log1p(len(set(nm[near & (nm[:, 1] <= t), 0].tolist())))
        m = self.M[i]
        if m.any():
            d = _hav(centre[0], centre[1], self.C[i, m, 0], self.C[i, m, 1]) < CLUSTER_KM
            p = P[i, m]
            out[3], out[4] = p[d].sum(), p[d].max(initial=0.0)
            out[5] = float(d[int(p.argmax())])
        out[7] = 1.0
        return out


# ---------------------------------------------------------------------------------------------------------------------------- data


SLOT_KEYS = ("pre", "pre_noclue", "post", "hidden", "cand", "cvalid", "mask", "km_zoom", "km_zoom_set", "km_unzoom", "is_area", "is_osv", "rank", "center")


def _with_text(d: dict, cols: np.ndarray | None, tokens: list[list[tuple]] | None) -> dict:
    """The agent's batch dict with the text columns inserted before the MegaLoc columns (zeros if `cols` is None: MP16 photos) and up to
    N_TEXT_TOKENS text-place tokens per photo appended (tokens[i] = [(lat, lon, km, text columns)])."""

    import torch

    n, h = d["mask"].shape
    k = len(TEXT_COLS)
    x = torch.zeros((n, h, k), device="cuda") if cols is None else torch.as_tensor(cols, device="cuda")
    e = dict(d)
    for key in ("pre", "pre_noclue"):
        v = d[key]
        add = x if key == "pre" else torch.zeros_like(x)
        e[key] = torch.cat([v[..., :-aa.N_ML], add, v[..., -aa.N_ML:]], -1)
    if tokens is None:
        return e
    extra = N_TEXT_TOKENS
    for key in SLOT_KEYS:
        v = e[key]
        fill = {"km_zoom": float("inf"), "km_zoom_set": float("inf"), "km_unzoom": float("inf"), "rank": 99.0}.get(key, 0)
        pad = torch.full((n, extra) + tuple(v.shape[2:]), fill, dtype=v.dtype, device=v.device)
        e[key] = torch.cat([v, pad], 1)
    for i, toks in enumerate(tokens):
        for j, (lat, lon, km, tc) in enumerate(toks[:extra]):
            s = h + j
            e["pre"][i, s, -aa.N_ML - k:-aa.N_ML] = torch.as_tensor(tc, device="cuda")
            e["cvalid"][i, s, 0] = True
            e["mask"][i, s] = True
            for key in ("km_zoom", "km_zoom_set", "km_unzoom"):
                e[key][i, s] = km
            e["center"][i, s] = torch.as_tensor([lat, lon], dtype=e["center"].dtype, device="cuda")
    return e


def _concat(a: dict, b: dict, consume: bool = False) -> dict:
    """Two batch dicts as one (slot dimension padded to the larger). consume: pop each key from a and b as it is joined (temporaries; lower
    peak GPU memory)."""

    import torch

    h = max(a["mask"].shape[1], b["mask"].shape[1])
    out = {}
    for key in list(a):
        va, vb = (a.pop(key), b.pop(key)) if consume else (a[key], b[key])
        if key in SLOT_KEYS:
            fill = {"km_zoom": float("inf"), "km_zoom_set": float("inf"), "km_unzoom": float("inf"), "rank": 99.0}.get(key, 0)
            padded = []
            for v in (va, vb):
                p = torch.full((v.shape[0], h - v.shape[1]) + tuple(v.shape[2:]), fill, dtype=v.dtype, device=v.device)
                padded.append(torch.cat([v, p], 1) if h > v.shape[1] else v)
            del va, vb
            out[key] = torch.cat(padded, 0)
            del padded
        elif torch.is_tensor(va):
            out[key] = torch.cat([va, vb], 0)
        elif isinstance(va, np.ndarray):
            out[key] = np.concatenate([va, vb])
        elif key == "n":
            out[key] = va + vb
        else:
            out[key] = va
    return out


def _text_inputs(d: dict, text: Text, P: np.ndarray, photos: list[dict]) -> tuple[np.ndarray, list]:
    """Per-token text columns [n, h, k] for the existing tokens, and each photo's text-place tokens (the ranker's top places not within 25 km
    of a token), answering like an unzoomed area."""

    mask = d["mask"].cpu().numpy()
    centre = d["center"].cpu().numpy().astype(np.float64)
    n, h = mask.shape
    cols = np.zeros((n, h, len(TEXT_COLS)), np.float32)
    tokens = []
    for i in range(n):
        for s in np.flatnonzero(mask[i]):
            cols[i, s] = text.at(i, centre[i, s], P)
        toks = []
        m = text.M[i]
        if m.any():
            cs, p = text.C[i, m], P[i, m]
            have = centre[i, mask[i]]
            pool = np.asarray(photos[i]["pool"], np.float64)
            for j in np.argsort(-p):
                if len(toks) == N_TEXT_TOKENS or p[j] <= 0:
                    break
                c = cs[j]
                if len(have) and (_hav(c[0], c[1], have[:, 0], have[:, 1]) < CLUSTER_KM).any():
                    continue
                inside = np.flatnonzero(_hav(c[0], c[1], pool[:, 0], pool[:, 1]) < CLUSTER_KM)
                point = pool[inside[0]] if len(inside) else c
                km = float(_hav(photos[i]["truth"][0], photos[i]["truth"][1], point[0], point[1]))
                tc = text.at(i, c, P)
                tc[TEXT_COLS.index("text place")] = 1.0
                toks.append((float(point[0]), float(point[1]), km, tc))
                have = np.concatenate([have, c[None]])
        tokens.append(toks)
    return cols, tokens


# ---------------------------------------------------------------------------------------------------------------------------- CV


def _fit(d: dict, photos: np.ndarray, seed: int):
    net = aa.fit_answer(d, photos, seed=seed)
    pol = aa.fit_policy(net, d, photos, 0.0, seed=seed + 4)
    return net, {0.0: pol}


def _keep(rows) -> dict[str, np.ndarray]:
    return {r[0]: r[1] for r in rows if r[0] in ("RL policy, lambda 0", "fixed: read 1, recognise 1, zoom all")}


def cv(weight: int, seeds: int, cells: tuple[str, ...] = ("+commons", "+commons+text"), tokens_on: bool = True) -> None:
    import torch

    aa.CKPT_TAG, aa.ONE_KM = CKPT, 0.5
    text = Text()
    photos = _photos()
    tr = aa._data("train")
    cm = aa._data(TAG, stats=tr["stats"])
    dv = aa._data("dev3k", stats=tr["stats"])
    cm["base_km"] = aa._released_own_km(TAG, CKPT)
    assert cm["n"] == text.n and all(e["image_id"] == i for e, i in zip(photos, text.ids))
    n_tr, n_c = tr["n"], cm["n"]
    print(f"combined head: {n_tr} MP16 train, {n_c} commons26 dev photos (folds {np.bincount(text.fold).tolist()}), weight {weight}, {seeds} seed(s)",
          flush=True)
    res = collections.defaultdict(dict)  # cell -> recipe -> km [n_c]
    dev3k = collections.defaultdict(dict)
    mp16 = np.arange(n_tr)
    # base: the current pipeline (MP16 only; fold-independent)
    tr_t, dv_t = _with_text(tr, None, None), _with_text(dv, None, None)
    cm_t0 = _with_text(cm, None, None)
    for s in range(seeds):
        net, pols = _fit(tr_t, mp16, 7 + 100 * s)
        for k, v in _keep(aa.evaluate(net, pols, cm_t0, np.arange(n_c), cm["base_km"])).items():
            res["base"].setdefault(k, []).append(v)
        for k, v in _keep(aa.evaluate(net, pols, dv_t, np.arange(dv["n"]), dv["base_km"])).items():
            dev3k["base"].setdefault(k, []).append(v)
    print("base done", flush=True)
    for cell in cells:
        out = {}
        for k in range(5):
            P = text.probs((k,))
            if cell == "+commons+text":
                cols, tokens = _text_inputs(cm_t0, text, P, photos)
                tokens = tokens if tokens_on else [[] for _ in tokens]
                cm_k = _with_text(cm, cols, tokens)
                mix = _concat(_with_text(tr, None, [[] for _ in range(n_tr)]), cm_k)
            else:
                cm_k = cm_t0
                mix = _concat(tr_t, cm_t0)
            fit_idx = np.concatenate([mp16, np.repeat(n_tr + np.flatnonzero(text.fold != k), weight)])
            held = np.flatnonzero(text.fold == k)
            for s in range(seeds):
                net, pols = _fit(mix, fit_idx, 7 + 100 * s + k)
                for key, v in _keep(aa.evaluate(net, pols, mix, n_tr + held, cm["base_km"][held])).items():
                    out.setdefault((key, s), np.zeros(n_c))[held] = v
            torch.cuda.empty_cache()
            print(f"{cell} fold {k} done", flush=True)
        for (key, s), v in out.items():
            res[cell].setdefault(key, []).append(v)
        # dev3k regressions: trained on all of commons26 dev (rankers out of fold)
        if cell == "+commons+text":
            cols, tokens = _text_inputs(cm_t0, text, text.probs(()), photos)
            tokens = tokens if tokens_on else [[] for _ in tokens]
            mix = _concat(_with_text(tr, None, [[] for _ in range(n_tr)]), _with_text(cm, cols, tokens))
            dv_k = _with_text(dv, None, [[] for _ in range(dv["n"])])
        else:
            mix, dv_k = _concat(tr_t, cm_t0), dv_t
        fit_idx = np.concatenate([mp16, np.repeat(n_tr + np.arange(n_c), weight)])
        for s in range(seeds):
            net, pols = _fit(mix, fit_idx, 7 + 100 * s)
            for key, v in _keep(aa.evaluate(net, pols, dv_k, np.arange(dv["n"]), dv["base_km"])).items():
                dev3k[cell].setdefault(key, []).append(v)
    report(res, dev3k, text, cm, dv, weight, seeds, "" if tokens_on else "_notokens")


def report(res, dev3k, text, cm, dv, weight, seeds, suffix: str = "") -> None:
    lines = [f"combined head, step A (commons26 dev CV by uploader, weight {weight}, {seeds} seed(s), text-place tokens {'off' if suffix else 'on'}; km per seed averaged as accuracy)"]
    conts = sorted(set(text.continent))
    for name, r, d in (("commons26 dev", res, cm), ("dev3k", dev3k, dv)):
        lines.append(f"\n {name}: released Pinpoint {np.mean(d['base_km'] < 25):.1%}, retrained top-1 {np.mean(d['rt_base_km'] < 25):.1%} (< 25 km)")
        base = {k: np.mean([v < 25 for v in vs], 0) for k, vs in r["base"].items()}
        for cell in r:
            for key, vs in r[cell].items():
                ok = np.mean([v < 25 for v in vs], 0)
                ok1 = np.mean([v < 1 for v in vs], 0)
                diff = ok - base[key]
                se = diff.std() / math.sqrt(len(diff))
                bal = np.mean([ok[text.continent == c].mean() for c in conts]) if name == "commons26 dev" else float("nan")
                lines.append(f"   {cell:14s} {key:38s} < 25 km {ok.mean():6.1%} ({100 * diff.mean():+.2f}, SE {100 * se:.2f}; fixes {np.sum(diff > 0):.0f}, "
                             f"breaks {np.sum(diff < 0):.0f})  < 1 km {ok1.mean():5.1%}" + (f"  continent-balanced {bal:5.1%}" if name == "commons26 dev" else ""))
    text_ = "\n".join(lines)
    print(text_, flush=True)
    (OUT / f"combined_head_{TAG}_w{weight}{suffix}.txt").write_text(text_ + "\n")


# ---------------------------------------------------------------------------------------------------------------------------- text adapter


def _final_options(net, pol, d: dict):
    """The greedy lambda-0 policy's final state on every photo: answer logits over (area, unzoomed point) + (area, zoomed point) [n, 2h], their
    km, and whether the photo was read."""

    import torch

    seen = []
    orig = aa._km

    def grab(net_, v, zoomed):
        seen.append((v, zoomed))
        return orig(net_, v, zoomed)

    aa._km = grab
    try:
        with torch.no_grad():
            b = aa._sub(d, torch.arange(d["n"], device="cuda"))
            aa.rollout(net, pol, b, 0.0, "greedy")
            v, zoomed = seen[-1]
            logits = aa._options(net, v, zoomed)
    finally:
        aa._km = orig
    km = torch.cat([v["km_unzoom"], v["km_zoom"]], 1)
    read = v["pre"][:, 0, -2] > 0.5  # view's flag columns: read, recognise
    return logits.cpu().numpy(), km.cpu().numpy(), read.cpu().numpy()


def _fit_adapter(L, F, K, R, TL, TF, TK, photos, l2: float = 1e-2, steps: int = 300):
    """A few parameters on top of the frozen head: score = a * logit + w . text columns for area options, b + w . columns for text-place
    tokens; fit by the head's own loss (log-probability of the right options) on `photos`. Returns a function of photo indices -> chosen km."""

    import torch

    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    k = F.shape[-1]
    w = torch.zeros(k, requires_grad=True)
    a = torch.ones(1, requires_grad=True)
    b = torch.full((1,), -2.0, requires_grad=True)

    def scores(idx, L=L, F=F, R=R, TL=TL, TF=TF):
        finite = np.isfinite(L[idx])
        r = t(R[idx])[:, None, None]
        area = torch.where(torch.as_tensor(finite), a * t(np.where(finite, L[idx], 0.0)) + (t(F[idx]) * r) @ w, -1e9)
        tok = torch.where(torch.as_tensor((TL[idx] > 0.5) & R[idx][:, None]), b + t(TF[idx]) @ w, -1e9)
        return torch.cat([area, tok], 1)

    km = np.concatenate([K, TK], 1)
    idx = np.asarray(photos)
    ok = torch.as_tensor(km[idx] < CLUSTER_KM)
    s0 = scores(idx)
    valid = s0 > -1e8
    use = (ok & valid).any(1) & ~((ok | ~valid).all(1))
    opt = torch.optim.Adam([w, a, b], lr=0.05)
    for _ in range(steps):
        logp = scores(idx).log_softmax(1)
        loss = -torch.logsumexp(logp.masked_fill(~ok, -1e9), 1)[use].mean() + l2 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def choose(sel, other=None):
        """Chosen km of photos `sel` of the fitted set, or of another set's (L, F, K, R, TL, TF, TK)."""
        L_, F_, K_, R_, TL_, TF_, TK_ = other or (L, F, K, R, TL, TF, TK)
        with torch.no_grad():
            pick = scores(np.asarray(sel), L_, F_, R_, TL_, TF_).argmax(1).numpy()
        return np.concatenate([K_, TK_], 1)[np.asarray(sel), pick]

    return choose, (w.detach().numpy(), a.item(), b.item())


def _adapter_arrays(cols: np.ndarray, tokens: list) -> tuple:
    """_text_inputs' output as the adapter's arrays: columns per (area, unzoomed / zoomed) option, text-place token flags, columns, km."""

    n = len(tokens)
    TL = np.zeros((n, N_TEXT_TOKENS))
    TF = np.zeros((n, N_TEXT_TOKENS, len(TEXT_COLS)), np.float32)
    TK = np.full((n, N_TEXT_TOKENS), np.inf)
    for i, toks in enumerate(tokens):
        for j, (_, _, km, tc) in enumerate(toks):
            TL[i, j], TF[i, j], TK[i, j] = 1.0, tc, km
    return np.concatenate([cols, cols], 1), TL, TF, TK


def adapter(seeds: int) -> None:
    """The MP16-trained head frozen; a text adapter (lesson 121's override as ~10 parameters over every option) cross-fitted on commons26 dev."""

    import torch

    aa.CKPT_TAG, aa.ONE_KM = CKPT, 0.5
    text = Text()
    photos = _photos()
    tr = aa._data("train")
    cm = aa._data(TAG, stats=tr["stats"])
    tr_t, cm_t0 = _with_text(tr, None, None), _with_text(cm, None, None)
    n = cm["n"]
    h = cm_t0["mask"].shape[1]
    lines = [f"text adapter on the frozen MP16 head (commons26 dev, uploader folds, {seeds} seed(s)); % < 25 km"]
    tot_base, tot_new = [], []
    for s in range(seeds):
        net, pols = _fit(tr_t, np.arange(tr["n"]), 7 + 100 * s)
        L, K, R = _final_options(net, pols[0.0], cm_t0)
        new = np.zeros(n)
        weights = []
        for k in range(5):
            F, TL, TF, TK = _adapter_arrays(*_text_inputs(cm_t0, text, text.probs((k,)), photos))
            assert F.shape[1] == 2 * h
            choose, wt = _fit_adapter(L, F, K, R, TL, TF, TK, np.flatnonzero(text.fold != k))
            held = np.flatnonzero(text.fold == k)
            new[held] = choose(held)
            weights.append(wt)
        base_km = K[np.arange(n), np.nan_to_num(L, neginf=-1e9).argmax(1)]
        tot_base.append(base_km < CLUSTER_KM)
        tot_new.append(new < CLUSTER_KM)
        w = np.mean([x[0] for x in weights], 0)
        lines.append(f"  seed {s}: base {np.mean(base_km < CLUSTER_KM):.1%} -> adapter {np.mean(new < CLUSTER_KM):.1%}; read {R.mean():.0%} of photos; "
                     f"mean weights " + ", ".join(f"{c} {x:+.2f}" for c, x in zip(TEXT_COLS, w)) + f"; a {np.mean([x[1] for x in weights]):.2f}, "
                     f"b {np.mean([x[2] for x in weights]):+.2f}")
        torch.cuda.empty_cache()
    b, nw = np.mean(tot_base, 0), np.mean(tot_new, 0)
    diff = nw - b
    lines.append(f"  mean over seeds: {b.mean():.1%} -> {nw.mean():.1%} ({100 * diff.mean():+.2f}, SE {100 * diff.std() / math.sqrt(n):.2f}; "
                 f"fixes {np.sum(diff > 0):.0f}, breaks {np.sum(diff < 0):.0f}); continent-balanced "
                 f"{np.mean([b[text.continent == c].mean() for c in sorted(set(text.continent))]):.1%} -> "
                 f"{np.mean([nw[text.continent == c].mean() for c in sorted(set(text.continent))]):.1%}")
    text_ = "\n".join(lines)
    print(text_, flush=True)
    (OUT / f"combined_head_adapter_{TAG}.txt").write_text(text_ + "\n")


# ---------------------------------------------------------------------------------------------------------------------------- step B


def transfer(weight: int, seeds: int) -> None:
    """Step B: Commons training photos from the commons_train pool (no commons26 uploader), evaluated on all of commons26 dev (no CV) and on
    dev3k. Cells: the MP16 head (base); MP16 + commons_train rows (x weight) without and with text columns / text-place tokens; the frozen MP16
    head with the text adapter fitted on commons_train. Text rankers: out of fold (by uploader) on commons_train, trained on all of
    commons_train for commons26 dev."""

    import torch

    aa.CKPT_TAG, aa.ONE_KM = CKPT, 0.5
    ct_text, cd_text = Text("commons_train"), Text(TAG)
    ct_photos, cd_photos = _photos("commons_train"), _photos(TAG)
    tr = aa._data("train")
    ct = aa._data("commons_train", stats=tr["stats"])
    cd = aa._data(TAG, stats=tr["stats"])
    dv = aa._data("dev3k", stats=tr["stats"])
    cd["base_km"] = aa._released_own_km(TAG, CKPT)
    assert ct["n"] == ct_text.n and cd["n"] == cd_text.n
    n_tr, n_ct, n_cd = tr["n"], ct["n"], cd["n"]
    P_ct = ct_text.probs(())
    P_cd = np.where(cd_text.M, _fit_predict(ct_text.X, ct_text.M, ct_text.Y, cd_text.X, cd_text.M), 0.0)
    dv_text = Text("dev3k") if _cache("dev3k").exists() else None  # dev3k's own text searches (else zeros: no text on dev3k)
    if dv_text is not None:
        assert dv_text.n == dv["n"]
        P_dv = np.where(dv_text.M, _fit_predict(ct_text.X, ct_text.M, ct_text.Y, dv_text.X, dv_text.M), 0.0)
    print(f"step B: {n_tr} MP16 train + {n_ct} commons_train photos ({len(set(ct_text.fold.tolist()))} folds), weight {weight}; "
          f"eval {n_cd} commons26 dev, {dv['n']} dev3k; {seeds} seed(s)", flush=True)
    for name, t, P in (("commons_train", ct_text, P_ct), ("commons26 dev", cd_text, P_cd)) + ((("dev3k", dv_text, P_dv),) if dv_text else ()):
        top = np.where(t.M.any(1), t.Y[np.arange(t.n), np.where(t.M, P, -1).argmax(1)], False)
        print(f"  text ranker on {name}: a right candidate {t.Y.any(1).mean():.1%}, top-1 right {top.mean():.1%}", flush=True)
    res = collections.defaultdict(dict)
    dev3k = collections.defaultdict(dict)
    mp16 = np.arange(n_tr)
    ct_cols, ct_tokens = _text_inputs(_with_text(ct, None, None), ct_text, P_ct, ct_photos)
    cd_cols, cd_tokens = _text_inputs(_with_text(cd, None, None), cd_text, P_cd, cd_photos)
    if dv_text is not None:
        dv_cols, dv_tokens = _text_inputs(_with_text(dv, None, None), dv_text, P_dv, _photos("dev3k"))
    no_tok = lambda d: [[] for _ in range(d["n"])]
    def cell_data(cell):  # built one cell at a time (the GPU is shared)
        if cell == "base":
            return _with_text(tr, None, None), mp16, _with_text(cd, None, None), _with_text(dv, None, None)
        if cell == "+ctrain":
            return (_concat(_with_text(tr, None, None), _with_text(ct, None, None), consume=True), None, _with_text(cd, None, None),
                    _with_text(dv, None, None))
        return (_concat(_with_text(tr, None, no_tok(tr)), _with_text(ct, ct_cols, ct_tokens), consume=True), None, _with_text(cd, cd_cols, cd_tokens),
                _with_text(dv, dv_cols, dv_tokens) if dv_text else _with_text(dv, None, no_tok(dv)))

    fit_mix = np.concatenate([mp16, np.repeat(n_tr + np.arange(n_ct), weight)])
    suffix = "_dev3ktext" if dv_text else ""
    partial = OUT / f"combined_head_stepB_w{weight}{suffix}_partial.pkl"  # finished cells (a rerun skips them)
    import pickle

    if partial.exists():
        res, dev3k = (collections.defaultdict(dict, x) for x in pickle.loads(partial.read_bytes()))
    elif suffix and (old := OUT / f"combined_head_stepB_w{weight}_partial.pkl").exists():  # the +ctrain cell doesn't see text: reuse it
        r0, d0 = pickle.loads(old.read_bytes())
        res["+ctrain"], dev3k["+ctrain"] = r0["+ctrain"], d0["+ctrain"]
    for cell in ("+ctrain+text", "+ctrain", "base"):  # the largest batch first, on a fresh allocator
        if cell in res and len(next(iter(res[cell].values()))) == seeds:
            continue
        res.pop(cell, None)
        dev3k.pop(cell, None)
        if cell == "base":
            res.pop("adapter", None)
        train_d, idx, cd_d, dv_d = cell_data(cell)
        for s in range(seeds):
            net, pols = _fit(train_d, fit_mix if idx is None else idx, 7 + 100 * s)
            for key, v in _keep(aa.evaluate(net, pols, cd_d, np.arange(n_cd), cd["base_km"])).items():
                res[cell].setdefault(key, []).append(v)
            for key, v in _keep(aa.evaluate(net, pols, dv_d, np.arange(dv["n"]), dv["base_km"])).items():
                dev3k[cell].setdefault(key, []).append(v)
            if cell == "base":  # the text adapter on this frozen head: fitted on commons_train's final options, applied to commons26 dev
                L, K, R = _final_options(net, pols[0.0], _with_text(ct, None, None))
                F, TL, TF, TK = _adapter_arrays(ct_cols, ct_tokens)
                choose, wt = _fit_adapter(L, F, K, R, TL, TF, TK, np.arange(n_ct))
                Lc, Kc, Rc = _final_options(net, pols[0.0], cd_d)
                Fc, TLc, TFc, TKc = _adapter_arrays(cd_cols, cd_tokens)
                km = choose(np.arange(n_cd), (Lc, Fc, Kc, Rc, TLc, TFc, TKc))
                res["adapter"].setdefault("RL policy, lambda 0", []).append(km)
                if dv_text is not None:
                    Ld, Kd, Rd = _final_options(net, pols[0.0], dv_d)
                    Fd, TLd, TFd, TKd = _adapter_arrays(dv_cols, dv_tokens)
                    dev3k["adapter"].setdefault("RL policy, lambda 0", []).append(choose(np.arange(dv["n"]), (Ld, Fd, Kd, Rd, TLd, TFd, TKd)))
                print(f"  seed {s} adapter weights " + ", ".join(f"{c} {x:+.2f}" for c, x in zip(TEXT_COLS, wt[0])) + f"; a {wt[1]:.2f}, b {wt[2]:+.2f}",
                      flush=True)
            del net, pols
            torch.cuda.empty_cache()
            print(f"seed {s} {cell} done", flush=True)
        del train_d, cd_d, dv_d
        torch.cuda.empty_cache()
        partial.write_bytes(pickle.dumps((dict(res), dict(dev3k))))
    lines = [f"combined head, step B (train MP16 + commons_train x{weight}; eval all of commons26 dev and dev3k; {seeds} seed(s); % < 25 km)"]
    conts = sorted(set(cd_text.continent))
    for name, r, d in (("commons26 dev", res, cd), ("dev3k", dev3k, dv)):
        lines.append(f"\n {name}: released Pinpoint {np.mean(d['base_km'] < 25):.1%}, retrained top-1 {np.mean(d['rt_base_km'] < 25):.1%}")
        base = {k: np.mean([v < 25 for v in vs], 0) for k, vs in r["base"].items()}
        for cell in [c for c in ("base", "+ctrain", "+ctrain+text", "adapter") if c in r]:
            for key, vs in r[cell].items():
                ok, ok1 = np.mean([v < 25 for v in vs], 0), np.mean([v < 1 for v in vs], 0)
                diff = ok - base[key]
                bal = f"  continent-balanced {np.mean([ok[cd_text.continent == c].mean() for c in conts]):5.1%}" if name == "commons26 dev" else ""
                lines.append(f"   {cell:14s} {key:38s} < 25 km {ok.mean():6.1%} ({100 * diff.mean():+.2f}, SE {100 * diff.std() / math.sqrt(len(diff)):.2f}; "
                             f"fixes {np.sum(diff > 0):.0f}, breaks {np.sum(diff < 0):.0f})  < 1 km {ok1.mean():5.1%}{bal}")
    text_ = "\n".join(lines)
    print(text_, flush=True)
    (OUT / f"combined_head_stepB_w{weight}{suffix}.txt").write_text(text_ + "\n")


# ---------------------------------------------------------------------------------------------------------------------------- stacking (dev3k)


def _row_cols(tag: str, group: str) -> np.ndarray:
    """Per zoom-file row of an MP16 photo set: the columns of one earlier sub-threshold head input (setft_own caches)."""

    if group == "diffusion":  # lesson 127: personalised PageRank mass within 25 km of the area, diffused and seeds only
        z = np.load(Path("artifacts/diffusion") / f"diffusion_{tag}_{CKPT}.npz")
        return np.log(np.stack([z["mass_diff"], z["mass_seed"]], 1) + 1e-4)
    z = np.load(OUT / f"cropcheck_{tag}_{CKPT}.npz")  # lesson 89: the full view and 9 crops vs each area's restricted-search photos
    return z["x"][:, np.isin(z["groups"], ("global", "crop"))]


def _slot_pad(tag: str, x: np.ndarray, d: dict) -> np.ndarray:
    """Zoom-file rows -> the batch's [photo, slot] layout (hypothesis_agent.load's slot order: kept sources, file order within a photo)."""

    from .area_model import _paths
    from .hypothesis_agent import SOURCES as HSOURCES

    zm = np.load(_paths(tag, CKPT)[1])
    rows = np.flatnonzero(np.isin(zm["source"], HSOURCES))
    photo = zm["photo"][rows]
    slot = np.zeros(len(rows), int)
    seen = collections.Counter()
    for j, p in enumerate(photo):
        slot[j] = seen[p]
        seen[p] += 1
    n, h = d["mask"].shape
    out = np.zeros((n, h, x.shape[1]), np.float32)
    out[photo, slot] = x[rows]
    return out


def _with_cols(d: dict, cols: np.ndarray) -> dict:
    """Columns visible in every state (before and after the read), inserted before the MegaLoc columns."""

    import torch

    x = torch.as_tensor(cols, device="cuda")
    e = dict(d)
    for key in ("pre", "pre_noclue"):
        e[key] = torch.cat([d[key][..., :-aa.N_ML], x, d[key][..., -aa.N_ML:]], -1)
    return e


def stack(seeds: int) -> None:
    """The lesson-111 agent on train / dev3k with the cached sub-threshold inputs added (diffusion, lesson 127; crop views, lesson 89)."""

    aa.CKPT_TAG, aa.ONE_KM = CKPT, 0.5
    tr = aa._data("train")
    dv = aa._data("dev3k", stats=tr["stats"])
    raw = {g: (_slot_pad("train", _row_cols("train", g), tr), _slot_pad("dev3k", _row_cols("dev3k", g), dv)) for g in ("diffusion", "crops")}
    mask_tr = tr["mask"].cpu().numpy()
    norm = {}
    for g, (a, b) in raw.items():
        mu, sd = a[mask_tr].mean(0), a[mask_tr].std(0) + 1e-6
        norm[g] = ((a - mu) / sd * mask_tr[..., None], (b - mu) / sd * dv["mask"].cpu().numpy()[..., None])
    variants = {"base": (), "+diffusion": ("diffusion",), "+crops": ("crops",), "+diffusion+crops": ("diffusion", "crops")}
    res = collections.defaultdict(dict)
    for name, groups in variants.items():
        if groups:
            tr_v = _with_cols(tr, np.concatenate([norm[g][0] for g in groups], -1))
            dv_v = _with_cols(dv, np.concatenate([norm[g][1] for g in groups], -1))
        else:
            tr_v, dv_v = tr, dv
        for s in range(seeds):
            net, pols = _fit(tr_v, np.arange(tr["n"]), 7 + 100 * s)
            for key, v in _keep(aa.evaluate(net, pols, dv_v, np.arange(dv["n"]), dv["base_km"])).items():
                res[name].setdefault(key, []).append(v)
        print(f"{name} done", flush=True)
    lines = [f"stacked sub-threshold inputs, dev3k ({seeds} seeds; accuracy averaged over seeds)"]
    base = {k: np.mean([v < 25 for v in vs], 0) for k, vs in res["base"].items()}
    for name in res:
        for key, vs in res[name].items():
            ok, ok1 = np.mean([v < 25 for v in vs], 0), np.mean([v < 1 for v in vs], 0)
            diff = ok - base[key]
            per_seed = ", ".join(f"{100 * np.mean(v < 25):.1f}" for v in vs)
            lines.append(f"   {name:18s} {key:38s} < 25 km {ok.mean():6.1%} ({100 * diff.mean():+.2f}, SE {100 * diff.std() / math.sqrt(len(diff)):.2f}; "
                         f"seeds {per_seed})  < 1 km {ok1.mean():5.1%}")
    text_ = "\n".join(lines)
    print(text_, flush=True)
    (OUT / "combined_head_stack_dev3k.txt").write_text(text_ + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("gather", "features", "cv", "stack", "adapter", "transfer"))
    ap.add_argument("--tag", default=TAG, choices=tuple(SETS), help="gather / features: the Commons photo set")
    ap.add_argument("--weight", type=int, default=10, help="oversampling of the commons26 dev training photos")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cells", nargs="*", default=("+commons", "+commons+text"))
    ap.add_argument("--no-tokens", action="store_true", help="text columns only, no text-place tokens")
    a = ap.parse_args()
    if a.cmd == "gather":
        gather(a.tag)
    elif a.cmd == "features":
        features(a.tag)
    elif a.cmd == "stack":
        stack(a.seeds)
    elif a.cmd == "adapter":
        adapter(a.seeds)
    elif a.cmd == "transfer":
        transfer(a.weight, a.seeds)
    else:
        cv(a.weight, a.seeds, tuple(a.cells), not a.no_tokens)


if __name__ == "__main__":
    main()
