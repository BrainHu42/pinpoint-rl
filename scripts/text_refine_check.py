# Step 1 for an RL search agent over the text tools: does a search conditioned on another search's result rank the right place first more often
# than one-shot searches? Scripted refinement rules on commons26 dev (977 photos), the 4B's phrases (text / place names / queries) and their
# 1-3 word spans (lesson 116). Every query's matches are fetched once, uncapped up to text_search.MAX_MATCHES per source, so "search B near A's
# result" is a lookup. Rules (no truth, no photographer):
#   single     each phrase's top cluster, the one with the highest share / log2(2 + matches) committed (lesson 119's support order)
#   +spans     phrases and their spans, same commit rule (the mechanical relaxation of lesson 116)
#   +country   each phrase's top cluster inside the 4B's country guess (unconstrained top if none there)
#   anchor     every top-8 cluster of every phrase / span; support = other phrases (sharing no specific word) with a match within 25 km;
#              commit the most supported (then the support order); no support -> the single rule
#   anchor+cc  anchor, ties broken by the 4B country first
# Reported: top-1 right (< 25 km), the candidate set's ceiling with lesson 120's luck control (a random other photo's truth, and a same-country
# photo > 100 km away), and against the pipeline (25.2%) the switch "commit when support >= k" for pre-declared k = 1, 2, 3.
# Usage: PYTHONPATH=src .venv/bin/python scripts/text_refine_check.py fetch [--workers 4]
#        PYTHONPATH=src .venv/bin/python scripts/text_refine_check.py report
import argparse
import csv
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from geo_search_env.experiment import text_search as ts  # noqa: E402
from geo_search_env.experiment.region_evidence import _raw_phrases  # noqa: E402

TAG = "commons26dev_setft_own"
OUT = Path("artifacts/agent/text_refine")
R = 6371.0
GENERIC = set("""the of and de la le les du des del della di da do dos das der die und von zu am im an in on at by for to a
monument museum station railway church cathedral chapel temple mosque shrine square street road avenue park garden bridge tower castle palace
center centre shopping mall market hotel restaurant cafe bar school university college hospital library hall house building memorial statue
fountain gate lake river mountain beach island port harbour harbor airport bus stop city town village old new north south east west central
national state public main saint st plaza place via rue strasse calle piazza location photo view interior exterior entrance sign front
festival fest event collection exhibition gallery theatre theater cinema office police fire community club""".split())  # read_constrain (tag pre-merge-cleanup-2026-10-16)


def _spans(phrase: str, max_len: int = 3) -> list[str]:
    words = [w.strip(".,;:!?()[]\"'«»“”") for w in phrase.split()]
    words = [w for w in words if w]
    out = []
    for n in range(1, max_len + 1):
        for i in range(len(words) - n + 1):
            span = words[i:i + n]
            if not all(w.lower() in GENERIC or len(w) < 3 for w in span):
                out.append(" ".join(span))
    return out


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
    return json.loads((Path("artifacts/photo_sets") / TAG / "photos.json").read_text(encoding="utf-8"))


def _queries() -> dict[str, dict[str, list[str]]]:
    """image id -> raw phrase -> [the phrase, its spans]."""
    out = {}
    for image_id, table in _raw_phrases(TAG, "4b").items():
        out[image_id] = {p: list(dict.fromkeys([p] + _spans(p))) for p in table}
    return out


def _fetch_one(tool: ts.TextSearch, q: str):
    """All matches of a query: (kind, loose, lat, lon, weight, cc) with places weighing 1 and each Flickr user 1 in total."""
    kind = "name"
    words = ts.norm(q).split()
    if ts.phone_digits(q) or ts.domain_of(q, ts.GENERIC_TLDS | set(tool.tld)) or not words:
        kind, places, photos, _, _, loose = tool.matches(q)
        lat = [p[1] for p in places] + [p[2] for p in photos]
        lon = [p[2] for p in places] + [p[3] for p in photos]
        cc = [p[3] or "" for p in places] + [""] * len(photos)
        users = [p[1] for p in photos]
    else:
        lat, lon, cc, users, loose = [], [], [], [], False
        for lz in (False, True):
            if lz and len(words) < 2:
                break
            match = " ".join(words) if lz else '"' + " ".join(words) + '"'
            ids = tool._fts(tool.places, "names", match, None, "places")
            rows = []
            for s in range(0, len(ids), 900):
                chunk = ids[s:s + 900]
                rows += tool.places.execute(f"SELECT lat, lon, cc FROM places WHERE rowid IN ({','.join('?' * len(chunk))})", chunk).fetchall()
            ph = []
            if tool.yfcc is not None:
                ids = tool._fts(tool.yfcc, "titles", match, None, "meta")
                for s in range(0, len(ids), 900):
                    chunk = ids[s:s + 900]
                    ph += tool.yfcc.execute(f"SELECT uid, lat, lon, accuracy FROM meta WHERE rowid IN ({','.join('?' * len(chunk))})", chunk).fetchall()
                ph = [r for r in ph if r[3] >= tool.min_accuracy]  # Commons 2026 photos are not in YFCC: nothing to exclude
            if rows or ph:
                lat = [r[0] for r in rows] + [r[1] for r in ph]
                lon = [r[1] for r in rows] + [r[2] for r in ph]
                cc = [r[2] or "" for r in rows] + [""] * len(ph)
                users, loose = [r[0] for r in ph], lz
                break
    n_pl = len(lat) - len(users)
    per = {}
    for u in users:
        per[u] = per.get(u, 0) + 1
    w = np.asarray([1.0] * n_pl + [1.0 / per[u] for u in users], np.float32)
    return kind, loose, np.asarray(lat, np.float32), np.asarray(lon, np.float32), w, np.asarray(cc)


def fetch(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qs = _queries()
    uniq = sorted({q for t in qs.values() for v in t.values() for q in v})
    print(f"{len(qs)} photos with phrases, {sum(len(t) for t in qs.values())} phrases, {len(uniq)} unique queries", flush=True)
    tools = [ts.TextSearch() for _ in range(workers)]
    chunks = [uniq[k::workers] for k in range(workers)]

    def run(k):
        return {q: _fetch_one(tools[k], q) for q in chunks[k]}

    res = {}
    with ThreadPoolExecutor(workers) as ex:
        for part in ex.map(run, range(workers)):
            res |= part
    with open(OUT / "matches.pkl", "wb") as f:
        pickle.dump(res, f, protocol=5)
    print(f"-> {OUT / 'matches.pkl'}", flush=True)


def _xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _hav(a, b):
    a, b = np.radians(np.asarray(a, np.float64)), np.radians(np.asarray(b, np.float64))
    h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


CHORD = 2 * np.sin(25.0 / R / 2)


class Town:
    def __init__(self):
        from scipy.spatial import cKDTree

        t = ts.TextSearch(yfcc=False)
        rows = t.places.execute("SELECT lat, lon, cc FROM towns").fetchall()
        self.cc = np.asarray([r[2] for r in rows])
        self.tree = cKDTree(_xyz([r[0] for r in rows], [r[1] for r in rows]))

    def of(self, lat, lon):
        return self.cc[self.tree.query(_xyz(lat, lon))[1]]


def _clusters(m, town: Town, top: int = 8, sample: int = 1000):
    """Greedy 25 km clusters of a query's matches (as text_search._cluster, which samples 500 per source): [(lat, lon, share, cc)], match count."""
    kind, loose, lat, lon, w, cc = m
    n = len(lat)
    if not n:
        return [], 0
    idx = np.arange(n) if n <= sample else np.sort(np.random.default_rng(n).choice(n, sample, replace=False))
    lat, lon, w, cc = lat[idx], lon[idx], w[idx], cc[idx].copy()
    if (cc == "").any():
        cc[cc == ""] = town.of(lat[cc == ""], lon[cc == ""])
    x = _xyz(lat, lon)
    within = ((x[:, None, :] - x[None]) ** 2).sum(-1) < CHORD ** 2
    alive = np.ones(len(lat), bool)
    total, out = w.sum(), []
    while alive.any() and len(out) < top:
        score = within @ (w * alive)
        score[~alive] = -1
        c = int(score.argmax())
        mem = np.flatnonzero(within[c] & alive)
        alive[mem] = False
        vals, inv = np.unique(cc[mem], return_inverse=True)
        out.append((float(lat[c]), float(lon[c]), float(w[mem].sum() / total), str(vals[np.bincount(inv, w[mem]).argmax()])))
    return out, n


def _specific(q: str) -> set[str]:
    return {w for w in ts.norm(q).split() if w not in GENERIC and len(w) >= 3}


def report() -> None:
    from scipy.spatial import cKDTree

    photos = _photos()
    qs = _queries()
    res = pickle.load(open(OUT / "matches.pkl", "rb"))
    town = Town()
    cache = OUT / "clusters.pkl"
    if cache.exists():
        cl = pickle.load(open(cache, "rb"))
    else:
        print("clustering", flush=True)
        cl = {q: _clusters(m, town) for q, m in res.items()}
        pickle.dump(cl, open(cache, "wb"))
    trees = {}

    def tree(q):
        if q not in trees:
            m = res[q]
            trees[q] = cKDTree(_xyz(m[2], m[3])) if len(m[2]) else None
        return trees[q]

    bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
    cc_of = _country_codes()
    obs = json.loads((Path("artifacts/agent") / f"observe_{TAG}_4b.json").read_text(encoding="utf-8"))
    z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
    pipe_km = z["commons26dev:setft_own|RL policy, lambda 0"]
    assert len(pipe_km) == len(photos)
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    true_cc = np.asarray([bench[e["image_id"]]["country"] for e in photos])
    tier = np.asarray([bench[e["image_id"]]["tier"] for e in photos])
    n = len(photos)
    caps = {"anchor": None, "anchor<=500": 500, "anchor<=50": 50}  # support only from phrases with at most this many matches (pre-declared)
    rules = ("single", "+spans", "+country") + tuple(caps) + ("anchor+cc",)
    commit = {r: np.full((n, 2), np.nan) for r in rules}
    cand_sets = {r: [np.zeros((0, 2))] * n for r in rules}
    support = {r: np.zeros(n, int) for r in caps}
    n_indep = np.zeros(n, int)
    for i, e in enumerate(photos):
        t = qs.get(e["image_id"])
        if not t:
            continue
        guess = cc_of.get(str((obs.get(e["image_id"], {}).get("parsed") or {}).get("country") or "").strip().lower())
        order = lambda c, nm: c[2] / np.log2(2 + nm)
        singles, spans_, country, allc = [], [], [], []
        for p, qlist in t.items():
            c, nm = cl[p]
            if c:
                singles.append((order(c[0], nm), c[0][:2], p))
                inc = [x for x in c if x[3] == guess]
                country.append((order(inc[0] if inc else c[0], nm), (inc[0] if inc else c[0])[:2], p))
            for q in qlist:
                cq, nq = cl[q]
                if cq:
                    spans_.append((order(cq[0], nq), cq[0][:2], p))
                    allc += [(order(x, nq), x[:2], p, x[3] == guess) for x in cq]
        # independent phrases: share no specific word
        spec = {p: _specific(p) for p in t}
        n_indep[i] = max((sum(1 for b in t if b != a and not (spec[a] & spec[b])) for a in t), default=0)
        scored = []  # (support per cap, in the 4B country, support order, point)
        for sc, ll, p, in_cc in allc:
            x = _xyz(*ll)
            near = [b for b in t if b != p and not (spec[p] & spec[b]) and tree(b) is not None
                    and tree(b).query_ball_point(x, CHORD, return_length=True) > 0]
            s = {r: sum(1 for b in near if cap is None or len(res[b][2]) <= cap) for r, cap in caps.items()}
            scored.append((s, in_cc, sc, ll))
        for r, lst in (("single", singles), ("+spans", spans_), ("+country", country)):
            if lst:
                commit[r][i] = max(lst, key=lambda x: x[0])[1]
                cand_sets[r][i] = np.asarray([x[1] for x in lst])
        if scored:
            for r in caps:
                a1 = max(scored, key=lambda x: (x[0][r], x[2]))
                support[r][i] = a1[0][r]
                commit[r][i] = a1[3] if a1[0][r] > 0 else commit["single"][i]
                cand_sets[r][i] = np.asarray([x[3] for x in scored])
            a2 = max(scored, key=lambda x: (x[0]["anchor"], x[1], x[2]))
            commit["anchor+cc"][i] = a2[3] if a2[0]["anchor"] > 0 else commit["single"][i]
            cand_sets["anchor+cc"][i] = cand_sets["anchor"][i]
    # luck controls: a random other photo; a same-country photo > 100 km away
    rng = np.random.default_rng(0)
    other = rng.permutation(n)
    same = np.full(n, -1)
    for i in range(n):
        pool = np.flatnonzero((true_cc == true_cc[i]) & (_hav(truth[i], truth) > 100))
        if len(pool):
            same[i] = rng.choice(pool)

    def hit(sets, tr, rows):
        return np.asarray([len(sets[i]) > 0 and j >= 0 and (_hav(tr[j], sets[i]) < 25).any() for i, j in zip(range(n), rows)])

    pipe = pipe_km < 25
    geo = np.isin(tier, ("landmark", "city", "region"))
    lines = [f"text refinement check, commons26 dev ({n} photos; {len(qs)} with phrases; pipeline {pipe.mean():.1%}, geolocatable tiers "
             f"{geo.sum()} photos, pipeline {pipe[geo].mean():.1%})",
             f"photos with >= 2 independent phrases (no shared specific word): {np.mean(n_indep >= 1):.1%}; anchor commit with support >= 1: "
             f"{np.mean(support['anchor'] >= 1):.1%}, >= 2: {np.mean(support['anchor'] >= 2):.1%}",
             f"  {'rule':11s} {'top-1':>6s} {'geo':>6s} {'ceiling':>8s} {'luck rnd':>9s} {'luck cc':>8s}  top-1 on pipeline misses"]
    for r in rules:
        ok = np.isfinite(commit[r][:, 0])
        right = np.zeros(n, bool)
        right[ok] = _hav(truth[ok], commit[r][ok]) < 25
        ceil = hit(cand_sets[r], truth, np.arange(n))
        lr, lc = hit(cand_sets[r], truth, other), hit(cand_sets[r], truth, same)
        lines.append(f"  {r:11s} {right.mean():6.1%} {right[geo].mean():6.1%} {ceil.mean():8.1%} {lr.mean():9.1%} {lc[same >= 0].mean():8.1%}  "
                     f"{right[~pipe].mean():.1%} ({(right & ~pipe).sum()} photos)")
        commit[r + "_right"] = right
    for r in caps:
        a = commit[r + "_right"]
        lines.append(f"  switch to the {r} commit when its support >= k (pre-declared k), vs the pipeline:")
        for k in (1, 2, 3):
            sw = support[r] >= k
            fx, br = (sw & a & ~pipe).sum(), (sw & ~a & pipe).sum()
            d = (np.where(sw, a, pipe).astype(float) - pipe)
            lines.append(f"    k {k}: switches {sw.sum():4d}, right {a[sw].mean() if sw.any() else 0:.1%}; {fx} fixes / {br} breaks, "
                         f"{100 * d.mean():+.2f} pts (SE {100 * d.std() / np.sqrt(n):.2f}); geolocatable {100 * d[geo].mean():+.2f}")
    text = "\n".join(lines)
    print(text, flush=True)
    (OUT / "report.txt").write_text(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fetch", "report"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    fetch(a.workers) if a.cmd == "fetch" else report()


if __name__ == "__main__":
    main()
