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
TAG, CKPT = "commons26dev", "setft_own"
CANDS = OUT / "cands_commons26_dev.jsonl"
CACHE = OUT / f"combined_text_{TAG}_{CKPT}.npz"
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


def _photos() -> list[dict]:
    return json.loads((Path("artifacts/photo_sets") / f"{TAG}_{CKPT}" / "photos.json").read_text(encoding="utf-8"))


def _context(photos: list[dict]) -> dict:
    """What lesson 119's ranker features need besides the search results: Pinpoint's areas and set probabilities, the 4B's country, towns."""

    from scipy.spatial import cKDTree

    from .text_search import TextSearch

    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{TAG}_{CKPT}.npz")
    areas = collections.defaultdict(list)
    for j in np.flatnonzero(zm["source"] == 0):
        areas[int(zm["photo"][j])].append((int(zm["rank"][j]), zm["center"][j].astype(np.float64)))
    areas = {k: np.asarray([c for _, c in sorted(v, key=lambda x: x[0])]) for k, v in areas.items()}
    sp = np.load(Path("artifacts/pinpoint_set") / f"set_probs_{TAG}_{CKPT}.npz")
    key = f"{TAG}_{CKPT}"
    cc_of = _country_codes()
    guess = {k: cc_of.get(str((v.get("parsed") or {}).get("country") or "").strip().lower())
             for k, v in json.loads((OUT / "observe_commons26_4b.json").read_text(encoding="utf-8")).items()}
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


def features() -> None:
    """Text candidates, ranker features and name matches of every commons26 dev photo (photo-set order) -> CACHE."""

    photos = _photos()
    pidx = {e["image_id"]: k for k, e in enumerate(photos)}
    recs = {}
    for line in CANDS.read_text().splitlines():
        r = json.loads(line)
        recs[r["image_id"]] = r
    ctx = _context(photos)
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
    bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
    groups = [bench[e["image_id"]]["group"] or e["image_id"] for e in photos]
    np.savez(CACHE, C=np.concatenate(C), X=np.concatenate(X), Y=np.concatenate(Y), coff=np.asarray(coff), names=np.asarray(names, np.float64).reshape(-1, 4),
             noff=np.asarray(noff), fold=np.asarray([int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % 5 for g in groups]),
             continent=np.asarray([bench[e["image_id"]]["continent"] for e in photos]), ids=np.asarray([e["image_id"] for e in photos]))
    print(f"{len(photos)} photos ({len(pidx)} ids, {sum(1 for e in photos if e['image_id'] in recs)} with searches); "
          f"candidates per photo median {np.median(np.diff(coff)):.0f}, some right {np.mean([y.any() for y in Y]):.1%} -> {CACHE}", flush=True)


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

    def __init__(self) -> None:
        z = np.load(CACHE)
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


def _concat(a: dict, b: dict) -> dict:
    """Two batch dicts as one (slot dimension padded to the larger)."""

    import torch

    h = max(a["mask"].shape[1], b["mask"].shape[1])
    out = {}
    for key in a:
        va, vb = a[key], b[key]
        if key in SLOT_KEYS:
            fill = {"km_zoom": float("inf"), "km_zoom_set": float("inf"), "km_unzoom": float("inf"), "rank": 99.0}.get(key, 0)
            padded = []
            for v in (va, vb):
                p = torch.full((v.shape[0], h - v.shape[1]) + tuple(v.shape[2:]), fill, dtype=v.dtype, device=v.device)
                padded.append(torch.cat([v, p], 1))
            out[key] = torch.cat(padded, 0)
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


def cv(weight: int, seeds: int) -> None:
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
    for cell in ("+commons", "+commons+text"):
        out = {}
        for k in range(5):
            P = text.probs((k,))
            if cell == "+commons+text":
                cols, tokens = _text_inputs(cm_t0, text, P, photos)
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
            mix = _concat(_with_text(tr, None, [[] for _ in range(n_tr)]), _with_text(cm, cols, tokens))
            dv_k = _with_text(dv, None, [[] for _ in range(dv["n"])])
        else:
            mix, dv_k = _concat(tr_t, cm_t0), dv_t
        fit_idx = np.concatenate([mp16, np.repeat(n_tr + np.arange(n_c), weight)])
        for s in range(seeds):
            net, pols = _fit(mix, fit_idx, 7 + 100 * s)
            for key, v in _keep(aa.evaluate(net, pols, dv_k, np.arange(dv["n"]), dv["base_km"])).items():
                dev3k[cell].setdefault(key, []).append(v)
    report(res, dev3k, text, cm, dv, weight, seeds)


def report(res, dev3k, text, cm, dv, weight, seeds) -> None:
    lines = [f"combined head, step A (commons26 dev CV by uploader, weight {weight}, {seeds} seed(s); km per seed averaged as accuracy)"]
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
    (OUT / f"combined_head_{TAG}_w{weight}.txt").write_text(text_ + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("features", "cv"))
    ap.add_argument("--weight", type=int, default=10, help="oversampling of the commons26 dev training photos")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "features":
        features()
    else:
        cv(a.weight, a.seeds)


if __name__ == "__main__":
    main()
