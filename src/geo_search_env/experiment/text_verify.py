# Track S (SEARCH_PLAN.md), S3: propose from text, verify by search. Lesson S2: constraints taken from the photo's own pool can't be verified by
# a restricted search, since each contains a look-alike by construction. A place proposed by read text has no such selection: if it is a wrong
# homonym, Pinpoint restricted there need not find anything like the photo, so the restricted search is a visual check independent of the text.
# On commons26 dev (reading is the gain there; lessons 115-121): lesson 119's text-search candidates (<= 64 per photo; its 37 text features,
# ported verbatim from candidate_ranker.py at 7775c85) and, as a same-photo control, boxes around the photo's own pool candidates; Pinpoint
# restricted to +-25 km around each; does the result separate right from wrong for text candidates where it couldn't for pool ones, and does
# it lift the ranker and a switch against the pipeline (grouped 5-fold by uploader, as lessons 119 and 121)?
# Usage (PINPOINT_KEEP_OWN=1, PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_own/checkpoints/ckpt_final.pt):
#   PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_verify build      (GPU, minutes)
#   PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_verify megaloc    (GPU, minutes)
#   PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_verify evaluate   (CPU)

"""build:    candidates (text: lesson 119's; pool: the photo's pool candidates, one per 25 km) with their restricted searches, index density
          around each, and each photo's unconstrained search -> artifacts/search/textverify_commons26dev.npz.
megaloc:  MegaLoc similarity of the query and each restricted top-1 -> artifacts/search/textverify_megaloc_commons26dev.npz.
evaluate: verification AUC for text vs pool candidates, the listwise ranker with text features T vs T + verification V, switches against the
          pipeline's answer -> artifacts/search/textverify_commons26dev.txt."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .search_tree import BOX_KM, TEMP, TOP, _auc, _box, _km, _photos

OUT = Path("artifacts/search")
AGENT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
TAG, FULL, SPLIT = "commons26dev_setft_own", "commons26_setft_own", "dev"  # the dev photo set; the commons26 set its zoom / set-prob caches index
CANDS = AGENT / "cands_commons26_dev.jsonl"
APPLY = AGENT / "action_agent_apply_km_setft_own_1km0.5_commons26dev.npz"
KEY = "commons26dev:setft_own|RL policy, lambda 0"
BUILT, MEGALOC, REPORT = OUT / "textverify_commons26dev.npz", OUT / "textverify_megaloc_commons26dev.npz", OUT / "textverify_commons26dev.txt"

# ---------------------------------------------------------------- lesson 119's candidates and text features (candidate_ranker.py, 7775c85)
SOURCES = ("GeoNames", "Overture", "Flickr", "OSM")
CLUSTER_KM = 25.0
MAX_CANDIDATES = 64
FEATURES = ["queries", "phrase queries", "span queries", "sources", *[f"has {s}" for s in SOURCES], *[f"weight {s}" for s in SOURCES],
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


def _context() -> dict:
    """Per photo besides the search results (lesson 119's _context): Pinpoint's areas and set probabilities, the 4B's country, town populations."""

    from scipy.spatial import cKDTree

    from .text_search import TextSearch

    photos = json.loads((PHOTO_SETS / FULL / "photos.json").read_text(encoding="utf-8"))
    pidx = {e["image_id"]: k for k, e in enumerate(photos)}
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{FULL}.npz")
    src, ph, rk, ce = zm["source"], zm["photo"], zm["rank"], zm["center"]
    areas = collections.defaultdict(list)
    for j in np.flatnonzero(src == 0):
        areas[int(ph[j])].append((int(rk[j]), ce[j].astype(np.float64)))
    areas = {k: np.asarray([c for _, c in sorted(v, key=lambda x: x[0])]) for k, v in areas.items()}
    sp = np.load(Path("artifacts/pinpoint_set") / f"set_probs_{FULL}.npz")
    probs, plat, valid = sp[f"{FULL}_probs"], sp[f"{FULL}_latlon"], sp[f"{FULL}_valid"]
    cc_of = _country_codes()
    guess = {k: cc_of.get(str((v.get("parsed") or {}).get("country") or "").strip().lower())
             for k, v in json.loads((AGENT / "observe_commons26_4b.json").read_text(encoding="utf-8")).items()}
    t = np.asarray(TextSearch().places.execute("SELECT lat, lon, pop FROM towns").fetchall(), np.float64)
    return {"pidx": pidx, "areas": areas, "probs": probs * valid, "plat": plat, "guess": guess,
            "towns": (cKDTree(_xyz(t[:, 0], t[:, 1])), np.log10(1 + t[:, 2]))}


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


def _folds(groups: list[str], k: int = 5) -> np.ndarray:
    return np.asarray([int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % k for g in groups])


# ---------------------------------------------------------------- build: candidates and their restricted searches

def build() -> None:
    from ..tools.pinpoint import Constraint
    from .hypotheses import photo_world
    from .search_tree import _tool

    photos = _photos(TAG)
    pos ={e["image_id"]: p for p, e in enumerate(photos)}
    recs = [json.loads(line) for line in CANDS.read_text(encoding="utf-8").splitlines()]
    assert {r["image_id"] for r in recs} <= set(pos), "candidates for photos outside the dev set"
    ctx = _context()
    photo, source, centre, X = [], [], [], []
    for r in recs:  # text candidates (lesson 119)
        C, F = _candidates(r, ctx)
        photo += [pos[r["image_id"]]] * len(C)
        source += [0] * len(C)
        centre += list(C)
        X += list(F)
    for p, e in enumerate(photos):  # same-photo control: the pool's candidates in score order, one per 25 km
        kept: list[np.ndarray] = []
        for c in np.asarray(e["pool"], np.float64):
            if np.isfinite(c).all() and all(_km(c, k) >= BOX_KM for k in kept):
                kept.append(c)
        photo += [p] * len(kept)
        source += [1] * len(kept)
        centre += kept
        X += [np.zeros(len(FEATURES), np.float32)] * len(kept)
    photo, source, centre = np.asarray(photo), np.asarray(source), np.asarray(centre, np.float64)
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    print(f"{TAG}: {len(photos)} photos; {np.sum(source == 0)} text candidates on {len(set(photo[source == 0]))} photos, "
          f"{np.sum(source == 1)} pool boxes", flush=True)
    tool = _tool()
    world, qrows = photo_world(photos)
    emb = np.asarray(world.query_embeddings[qrows], np.float32)
    free = tool.search(emb)  # benchmark queries: no photographer (original protocol)
    found = {k: [] for k in ("latlon", "scores", "features", "rows")}
    for s in range(0, len(photo), 4096):
        part = slice(s, s + 4096)
        f = tool.search(emb[photo[part]], [Constraint(box=_box(c)) for c in centre[part]])
        for k in found:
            found[k].append(getattr(f, k)[:, :TOP])
        print(f"  {min(s + 4096, len(photo))} / {len(photo)} searches", flush=True)
    tool.rows_near(np.zeros((1, 2)), 0.0)  # builds the KD-tree over index locations
    from ..models.pinpoint_reranker import _unit_xyz

    density = np.asarray(tool._tree.query_ball_point(_unit_xyz(centre), 2 * math.sin(BOX_KM / 6371.0 / 2), return_length=True))
    np.savez(BUILT, photo=photo, source=source, centre=centre, X=np.asarray(X, np.float32), truth=truth,
             y=_km(centre, truth[photo]) < BOX_KM, density=density, free_latlon=free.latlon[:, :TOP], free_scores=free.scores[:, :TOP],
             free_rows=free.rows[:, :TOP], latlon=np.concatenate(found["latlon"]), scores=np.concatenate(found["scores"]),
             features=np.concatenate(found["features"]).astype(np.float16), rows=np.concatenate(found["rows"]))
    print(f"-> {BUILT}", flush=True)


def megaloc(batch: int = 64, workers: int = 3) -> None:
    import time

    import torch
    from torch.utils.data import DataLoader

    from . import megaloc_gate as mg

    z = np.load(BUILT)
    photos = json.loads((PHOTO_SETS / TAG / "photos.json").read_text(encoding="utf-8"))
    # megaloc_gate's query embeddings in this set's photo order; megaloc_queries_commons26_setft_own.npy predates the benchmark-path fix (bd2d12d)
    # and is blank (all photos ~one vector), so only the dev file is usable
    q = torch.as_tensor(np.load(AGENT / f"megaloc_queries_{TAG}.npy"), device="cuda").float()
    assert len(q) == len(photos) and float((q[:100] @ q[:100].T).mean()) < 0.5, "blank MegaLoc query embeddings"
    r1 = np.concatenate([z["rows"][:, 0], z["free_rows"][:, 0]])
    owner = np.concatenate([z["photo"], np.arange(len(photos))])
    need = np.unique(r1[r1 >= 0])
    mg.CKPT_TAG = "setft_own"
    ids, nq, dataset = mg._image_ids("commons26dev", need)
    model = mg._model()
    emb = torch.zeros((len(need), q.shape[1]), dtype=torch.float16)
    t = time.perf_counter()
    with torch.inference_mode():
        for b, (i, x) in enumerate(DataLoader(dataset(ids[nq:]), batch_size=batch, num_workers=workers)):
            emb[i] = model(x.cuda()).half().cpu()
            if b % 100 == 0:
                print(f"  {(b + 1) * batch} / {len(need)} photos, {time.perf_counter() - t:.0f} s", flush=True)
    sim = np.full(len(r1), np.nan, np.float32)
    has = r1 >= 0
    e = emb[torch.as_tensor(np.searchsorted(need, r1[has]))].float().cuda()
    sim[has] = (e * q[torch.as_tensor(owner[has], device="cuda")]).sum(1).cpu().numpy()
    n = len(z["photo"])
    np.savez(MEGALOC, sim=sim[:n], free_sim=sim[n:])
    print(f"-> {MEGALOC}: {len(need)} photos in {time.perf_counter() - t:.0f} s", flush=True)


# ---------------------------------------------------------------- evaluate

V_NAMES = ("r1_score", "r1_vs_free_top", "r1_margin", "r_agree", "r_valid", "sim_image", "sim_raw", "sim_gps", "log_density", "log_km_r1_centre",
           "r1_near_free_top1", "log_km_r1_free_top1", "free_top_score", "megaloc", "megaloc_vs_free_top1")


def _verification(z, ml) -> np.ndarray:
    """Per candidate, what its restricted search returned, relative to the photo's unconstrained search."""

    n = len(z["photo"])
    p = z["photo"]
    sc, ll = z["scores"].astype(np.float64), z["latlon"].astype(np.float64)
    ftop, fll = z["free_scores"][:, 0].astype(np.float64)[p], z["free_latlon"][:, 0].astype(np.float64)[p]
    V = np.zeros((n, len(V_NAMES)), np.float32)
    ok = np.isfinite(sc[:, 0])
    r1 = np.where(ok[:, None], ll[:, 0], 0.0)
    wr = np.exp(np.where(np.isfinite(sc), sc - np.where(ok, sc[:, 0], 0)[:, None], -np.inf) / TEMP)
    wr /= np.maximum(wr.sum(1, keepdims=True), 1e-12)
    close = _km(np.nan_to_num(ll), r1[:, None]) < BOX_KM
    V[:, 0] = np.where(ok, sc[:, 0], -10)
    V[:, 1] = np.where(ok, sc[:, 0] - ftop, -10 - ftop)
    V[:, 2] = np.where(np.isfinite(sc[:, 1]), sc[:, 0] - sc[:, 1], 0)
    V[:, 3] = np.where(ok, (wr * close).sum(1), 0)
    V[:, 4] = np.isfinite(sc).mean(1)
    V[:, 5:8] = np.where(ok[:, None], z["features"][:, 0, :3].astype(np.float32), 0)
    V[:, 8] = np.log1p(z["density"])
    V[:, 9] = np.where(ok, np.log1p(_km(r1, z["centre"])), np.log1p(2e4))
    d = _km(r1, fll)
    V[:, 10] = ok & (d < BOX_KM)
    V[:, 11] = np.where(ok, np.log1p(d), np.log1p(2e4))
    V[:, 12] = ftop
    if ml is not None:
        V[:, 13] = np.nan_to_num(ml["sim"], nan=0.0)
        V[:, 14] = V[:, 13] - np.nan_to_num(ml["free_sim"], nan=0.0)[p]
    return V


def evaluate() -> None:
    z = np.load(BUILT)
    ml = np.load(MEGALOC) if MEGALOC.exists() else None
    photos = json.loads((PHOTO_SETS / TAG / "photos.json").read_text(encoding="utf-8"))
    n = len(photos)
    bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
    fold_of_photo = _folds([bench[e["image_id"]]["group"] or e["image_id"] for e in photos])
    apply = np.load(APPLY)
    km, conf, answer = apply[KEY], apply[f"{KEY}|conf"], apply[f"{KEY}|center"]
    ok = km < 25
    truth = z["truth"]
    photo, source, y = z["photo"], z["source"], z["y"]
    V = _verification(z, ml)
    vn = list(V_NAMES) if ml is not None else list(V_NAMES[:13])
    V = V[:, : len(vn)]
    r1 = np.nan_to_num(z["latlon"][:, 0].astype(np.float64))
    r1_right = np.isfinite(z["scores"][:, 0]) & (_km(r1, truth[photo]) < BOX_KM)
    free_right = _km(np.nan_to_num(z["free_latlon"][:, 0].astype(np.float64)), truth) < BOX_KM
    lines = [f"S3, propose from text, verify by search: {TAG} ({n} photos; pipeline < 25 km {100 * ok.mean():.1f}, unconstrained top-1 "
             f"{100 * free_right.mean():.1f}); label = candidate centre < 25 km of the truth"]
    for s, nm in ((0, "text candidates"), (1, "pool boxes")):
        m = source == s
        some = np.zeros(n, bool)
        np.logical_or.at(some, photo[m], y[m])
        some_r1 = np.zeros(n, bool)
        np.logical_or.at(some_r1, photo[m], r1_right[m])
        lines.append(f"  {nm}: {m.sum()} on {len(set(photo[m]))} photos; some candidate right {100 * some.mean():.1f}% (its restricted top-1 "
                     f"{100 * some_r1.mean():.1f}%); right & pipeline wrong {100 * (some & ~ok).mean():.1f}%")
    # luck: the text candidates against another photo's truth
    rng = np.random.default_rng(0)
    der = rng.permutation(n)
    while (der == np.arange(n)).any():
        der = rng.permutation(n)
    luck = np.zeros(n, bool)
    m = source == 0
    np.logical_or.at(luck, photo[m], _km(z["centre"][m], truth[der][photo[m]]) < BOX_KM)
    lines.append(f"  luck: text candidates vs a random other photo's truth {100 * luck.mean():.1f}%")

    # (1) selection-bias test: verification features alone, text vs pool candidates, pointwise MLP cross-fitted by uploader
    from .search_tree import _fit

    lines.append("\n(1) verification alone (restricted-search features V, pointwise MLP, uploader 5-fold): AUC right vs wrong candidates")
    alt_free = V[:, vn.index("r1_near_free_top1")] == 0
    for s, nm in ((0, "text candidates"), (1, "pool boxes")):
        m = source == s
        p = np.zeros(m.sum(), np.float32)
        f = fold_of_photo[photo[m]]
        for k in range(5):
            p[f == k] = _fit(V[m][f != k], y[m][f != k])(V[m][f == k])
        a = alt_free[m]
        lines.append(f"  {nm:16s}: V {_auc(p, y[m]):.3f}; candidates whose restricted top-1 is not at the free top-1 {_auc(p[a], y[m][a]):.3f}; "
                     f"single features: " + ", ".join(f"{v} {_auc(V[m][:, vn.index(v)], y[m]):.3f}" for v in ("r1_vs_free_top", "r_agree", "log_density")
                                                      + (("megaloc",) if ml is not None else ())))

    # (2) the listwise ranker on text candidates: T (lesson 119) vs T + V
    m = source == 0
    tp = [np.flatnonzero(m & (photo == p)) for p in range(n)]
    width = max(len(t) for t in tp)

    def pad(F):
        out = np.zeros((n, width, F.shape[1]), np.float32)
        for p, t in enumerate(tp):
            out[p, : len(t)] = F[t]
        return out

    M = np.zeros((n, width), bool)
    Y = np.zeros((n, width), bool)
    idx = np.full((n, width), -1)
    for p, t in enumerate(tp):
        M[p, : len(t)], Y[p, : len(t)], idx[p, : len(t)] = True, y[t], t
    sets = {"T (text, lesson 119)": pad(z["X"]), "T+V (+ verification)": pad(np.hstack([z["X"], V]))}
    lines.append(f"\n(2) listwise ranker over text candidates (lesson 119's, uploader 5-fold): photos with a right candidate {100 * Y.any(1).mean():.1f}%")
    res = {}
    for nm, X in sets.items():
        for hidden in (0, 16):
            P = np.zeros(M.shape, np.float32)
            for k in range(5):
                tr, te = fold_of_photo != k, fold_of_photo == k
                P[te] = _fit_predict(X[tr], M[tr], Y[tr], X[te], M[te], hidden)
            top = P.argmax(1)
            has = M.any(1)
            top_right = has & Y[np.arange(n), top]
            res[(nm, hidden)] = (P, top_right)
            top_ll = np.where(has[:, None], z["centre"][np.maximum(idx[np.arange(n), top], 0)], np.nan)
            disagree = has & (_km(np.nan_to_num(top_ll), answer) >= BOX_KM)
            flat = M.reshape(-1)
            lines.append(f"  {nm:22s} {'linear' if not hidden else 'MLP 16':7s}: top right {100 * top_right.mean():.1f}% (of photos with a right one "
                         f"{100 * top_right[Y.any(1)].mean():.1f}); AUC {_auc(P.reshape(-1)[flat], Y.reshape(-1)[flat]):.3f}; top right & pipeline wrong "
                         f"{100 * (top_right & ~ok).mean():.1f}, pipeline right & top wrong {100 * (ok & ~top_right).mean():.1f}; where top disagrees "
                         f"with the pipeline ({100 * disagree.mean():.1f}% of photos): top right {100 * top_right[disagree].mean():.1f}%, pipeline "
                         f"{100 * ok[disagree].mean():.1f}%")

    # (3) switch: the ranker's top replaces the pipeline's answer when (ranker conf > a and pipeline conf < b); thresholds from the other folds
    lines.append("\n(3) switch to the ranker's top when its confidence > a and the pipeline's < b (thresholds from the other folds; SE over photos)")
    A_GRID, B_GRID = np.linspace(0.05, 0.95, 19), np.asarray([0.3, 0.5, 0.7, 0.9, 1.01])
    for (nm, hidden), (P, top_right) in res.items():
        c = P.max(1)
        gain = lambda s, a, b: (np.where((c[s] > a) & (conf[s] < b), top_right[s], ok[s]).astype(float) - ok[s]).sum()
        new = ok.astype(float).copy()
        for k in range(5):
            other, te = fold_of_photo != k, fold_of_photo == k
            a, b = max(((a, b) for a in A_GRID for b in B_GRID), key=lambda ab: gain(other, *ab))
            new[te] = np.where((c[te] > a) & (conf[te] < b), top_right[te], ok[te])
        d = new - ok
        lines.append(f"  {nm:22s} {'linear' if not hidden else 'MLP 16':7s}: {100 * ok.mean():.1f} -> {100 * new.mean():.1f} ({100 * d.mean():+.2f} pts, "
                     f"SE {100 * d.std() / math.sqrt(n):.2f}); fixes {(d > 0).sum()}, breaks {(d < 0).sum()}")
    # (4) why: on photos with a right text candidate, its verification vs the T ranker's wrong top, by whether the pipeline is right
    P, top_right = res[("T (text, lesson 119)", 0)]
    top_k = idx[np.arange(n), P.argmax(1)]
    lines.append("\n(4) verification of the best right text candidate (highest T score among right ones) vs the T ranker's top when that top "
                 "is wrong; medians")
    show = [v for v in ("r1_vs_free_top", "r_agree", "log_density", "r1_near_free_top1", "megaloc") if v in vn]
    for nm, sel in (("pipeline wrong (fixable)", Y.any(1) & ~top_right & ~ok), ("pipeline right", Y.any(1) & ~top_right & ok)):
        right_k = np.asarray([idx[p][M[p] & Y[p]][np.argmax(P[p][M[p] & Y[p]])] for p in np.flatnonzero(sel)], int)
        wrong_k = top_k[sel]
        lines.append(f"  {nm} ({sel.sum()} photos): " + "; ".join(
            f"{v} right {np.median(V[right_k, vn.index(v)]):.2f} vs top {np.median(V[wrong_k, vn.index(v)]):.2f} "
            f"(right > top on {100 * np.mean(V[right_k, vn.index(v)] > V[wrong_k, vn.index(v)]):.0f}%)" for v in show))
    text = "\n".join(lines)
    REPORT.write_text(text + "\n")
    print(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("build", "megaloc", "evaluate"))
    args = parser.parse_args(argv)
    {"build": build, "megaloc": megaloc, "evaluate": evaluate}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
