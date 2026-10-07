# AGENT_DIRECTION.md constraint tools over GeoNames, scored per retrieval candidate from the observe.py clues: features_near (do the listed visible
# features exist near the candidate?), geocode (does read text / a place name match a GeoNames name near it?), country rules (language, script,
# driving side, the VLM's country guess vs the candidate's country). Then the tools as candidate scorers: within-photo AUC and a CV listwise combiner
# of rank + tool features vs the reranker top-1, with fixes / breaks on the fixable photos (agent_targets.py).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.geo_tools index                (CPU, ~10 min, once)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.geo_tools score --tag dev --name base
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.geo_tools report --tags dev val --name base
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.geo_tools transfer --fit dev val --tag full --name base   (test set: once per method)

"""index:    every GeoNames entry: lat / lon, feature code, country code, and all its names (name, ascii name, alternate names) normalised, in
          artifacts/agent/geonames_index.npz (names as one blob + offsets).
score:    per photo and pool candidate, tool features (see FEATURES) -> artifacts/agent/tools_{tag}_{name}.npz.
report:   per feature the within-photo AUC (candidate < 25 km vs >= 25 km) on all / fixable photos; CV combiner (5 folds x 3 seeds) of rank only,
          rank + each tool group, rank + all tools: change in top-1 < 1 / 25 km vs the reranker top-1, fixes and breaks (overall and on fixable).
transfer: the combiner fitted on the fit sets, applied once to another set (the benchmark eval halves)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata
from typing import Sequence

import numpy as np

from ..paths import DATA_ROOT
from .observe import OUT, VOCAB
from .query_evidence import ROOT

GEONAMES = DATA_ROOT / "geonames"
INDEX = OUT / "geonames_index.npz"
RADII = (1.0, 5.0, 25.0)
MIN_NAME = 4  # characters (after normalisation) for a name to count as a text match; CJK names count double
EARTH_KM = 6371.0088

LEFT_DRIVING = set("AG AI AU BB BD BM BN BS BT BW CC CK CX CY DM FJ FK GB GD GG GY HK ID IE IM IN JE JM JP KE KI KN KY LC LK LS MO MS MT MU MV "
                   "MW MY MZ NA NF NP NR NU NZ PG PK PN SB SC SG SH SR SZ TC TH TK TL TO TT TV TZ UG VC VG VI WS ZA ZM ZW".split())
LANGUAGES = {  # language name (as a VLM writes it) -> ISO 639-1 / -3 code used in countryInfo
    "english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "ukrainian": "uk", "polish": "pl", "czech": "cs", "slovak": "sk", "hungarian": "hu", "romanian": "ro", "bulgarian": "bg", "serbian": "sr",
    "croatian": "hr", "slovenian": "sl", "greek": "el", "turkish": "tr", "arabic": "ar", "hebrew": "he", "persian": "fa", "farsi": "fa",
    "hindi": "hi", "bengali": "bn", "urdu": "ur", "tamil": "ta", "thai": "th", "vietnamese": "vi", "indonesian": "id", "malay": "ms",
    "chinese": "zh", "mandarin": "zh", "cantonese": "zh", "japanese": "ja", "korean": "ko", "swedish": "sv", "norwegian": "no", "danish": "da",
    "finnish": "fi", "icelandic": "is", "estonian": "et", "latvian": "lv", "lithuanian": "lt", "catalan": "ca", "basque": "eu", "galician": "gl",
    "irish": "ga", "welsh": "cy", "maltese": "mt", "albanian": "sq", "macedonian": "mk", "georgian": "ka", "armenian": "hy", "khmer": "km",
    "lao": "lo", "burmese": "my", "nepali": "ne", "sinhala": "si", "amharic": "am", "swahili": "sw", "tagalog": "tl", "filipino": "tl",
    "afrikaans": "af", "mongolian": "mn", "kazakh": "kk", "uzbek": "uz", "belarusian": "be", "bosnian": "bs", "luxembourgish": "lb",
}
SCRIPTS = {  # writing system -> languages written in it (Latin is too common to constrain)
    "cyrillic": {"ru", "uk", "bg", "sr", "mk", "be", "kk", "ky", "mn", "tg"}, "han": {"zh", "ja"}, "chinese": {"zh", "ja"},
    "hangul": {"ko"}, "kana": {"ja"}, "hiragana": {"ja"}, "katakana": {"ja"}, "japanese": {"ja"}, "thai": {"th"}, "greek": {"el"},
    "hebrew": {"he"}, "arabic": {"ar", "fa", "ur", "ps"}, "devanagari": {"hi", "ne", "mr"}, "bengali": {"bn"}, "georgian": {"ka"},
    "armenian": {"hy"}, "tamil": {"ta"}, "khmer": {"km"}, "lao": {"lo"}, "burmese": {"my"}, "myanmar": {"my"}, "sinhala": {"si"},
    "ethiopic": {"am"}, "ge'ez": {"am"},
}
FEATURES = (["feat frac 1", "feat frac 5", "feat frac 25", "feat idf 1", "feat idf 5", "feat idf 25", "text match 1", "text match 5", "text match 25",
             "name match 1", "name match 5", "name match 25", "language ok", "script ok", "driving ok", "country guess",
             "text region", "text country", "name region", "name country"])
GROUPS = {"features": [f for f in FEATURES if f.startswith("feat")], "text": [f for f in FEATURES if f.startswith(("text match", "name match"))],
          "rules": ["language ok", "script ok", "driving ok"], "country guess": ["country guess"],
          "text region": ["text region", "text country", "name region", "name country"]}
MAX_NGRAM = 4  # words; CJK runs: substrings of 2-6 characters


def _cjk(c: str) -> bool:
    return "぀" <= c <= "鿿" or "가" <= c <= "힯"


def ngrams(items: Sequence[str]) -> set[str]:
    """Normalised word n-grams (1-4 words, >= MIN_NAME characters, not just digits) and CJK substrings (2-6 characters) of the read strings."""

    out = set()
    for item in items:
        words = norm(item).split()
        for n in range(1, MAX_NGRAM + 1):
            for s in range(len(words) - n + 1):
                gram = " ".join(words[s:s + n])
                if len(gram) >= MIN_NAME and not gram.replace(" ", "").isdigit() and not any(_cjk(c) for c in gram):
                    out.add(gram)
        for run in re.findall(r"[぀-鿿가-힯]+", norm(item)):
            out |= {run[a:b] for a in range(len(run)) for b in range(a + 2, min(len(run), a + 6) + 1)}
    return out


def textgeo(name: str, tags: Sequence[str]) -> None:
    """For every n-gram of the read text and place names of the given photo sets: the GeoNames regions ("cc.admin1") and countries having an
    entry with that name -> artifacts/agent/textgeo_{name}.json."""

    grams: set[str] = set()
    for tag in tags:
        for v in json.loads((OUT / f"observe_{tag}_{name}.json").read_text(encoding="utf-8")).values():
            o = v.get("parsed") or {}
            grams |= ngrams((o.get("text") or []) + (o.get("place_names") or []))
    regions: dict[str, set[str]] = {g: set() for g in grams}
    with (GEONAMES / "allCountries.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            f = line.rstrip("\n").split("\t")
            for n in {norm(x) for x in [f[1], f[2]] + (f[3].split(",") if f[3] else [])} & grams:
                regions[n].add(f"{f[8]}.{f[10]}")
    found = {g: sorted(r) for g, r in regions.items() if r}
    (OUT / f"textgeo_{name}.json").write_text(json.dumps(found) + "\n", encoding="utf-8")
    print(f"{len(grams)} n-grams, {len(found)} found in GeoNames -> {OUT / f'textgeo_{name}.json'}")


def _region_scores(items: Sequence[str], table: dict[str, list[str]]) -> tuple[dict[str, float], dict[str, float]]:
    """Evidence per region and per country from the read strings: each matching n-gram adds len(n-gram) / (number of regions or countries it
    names); an n-gram inside a longer matching one is dropped."""

    grams = [g for g in ngrams(items) if g in table]
    grams = [g for g in grams if not any(g != h and g in h for h in grams)]
    reg, cty = {}, {}
    for g in grams:
        regions = table[g]
        countries = {r.split(".")[0] for r in regions}
        for r in regions:
            reg[r] = reg.get(r, 0.0) + len(g) / len(regions)
        for c in countries:
            cty[c] = cty.get(c, 0.0) + len(g) / len(countries)
    return reg, cty


def norm(text: str) -> str:
    """Casefolded, accents stripped (Latin), punctuation -> space, single spaces; CJK and other scripts kept."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", text).strip()


def _xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _chord(km: float) -> float:
    return 2 * np.sin(km / EARTH_KM / 2)


def index() -> None:
    lat, lon, code, cc, a1, offsets, blob = [], [], [], [], [], [0], bytearray()
    with (GEONAMES / "allCountries.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            f = line.rstrip("\n").split("\t")
            names = {norm(n) for n in [f[1], f[2]] + (f[3].split(",") if f[3] else [])}
            names.discard("")
            lat.append(float(f[4])); lon.append(float(f[5])); code.append(f[6] + "." + f[7]); cc.append(f[8]); a1.append(f[10])
            blob += ("\x1f".join(sorted(names))).encode("utf-8")
            offsets.append(len(blob))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX, lat=np.asarray(lat, np.float32), lon=np.asarray(lon, np.float32), code=np.asarray(code), cc=np.asarray(cc),
             a1=np.asarray(a1), offsets=np.asarray(offsets, np.int64), blob=np.frombuffer(bytes(blob), np.uint8))
    print(f"{len(lat)} entries, {len(blob) / 1e9:.2f} GB of names -> {INDEX}")


class Gazetteer:
    def __init__(self) -> None:
        from scipy.spatial import cKDTree

        z = np.load(INDEX)
        self.code, self.cc, self.a1, self.offsets, self.blob = z["code"], z["cc"], z["a1"], z["offsets"], z["blob"].tobytes()
        self.tree = cKDTree(_xyz(z["lat"].astype(np.float64), z["lon"].astype(np.float64)))
        self.wanted = {c for codes in VOCAB.values() for c in codes}
        populated = np.char.startswith(self.code.astype(str), "P.")
        self.ptree, self.pidx = cKDTree(self.tree.data[populated]), np.flatnonzero(populated)

    def names(self, i: int) -> list[str]:
        return self.blob[self.offsets[i]:self.offsets[i + 1]].decode("utf-8").split("\x1f")

    def near(self, lat: float, lon: float, km: float) -> np.ndarray:
        return np.asarray(self.tree.query_ball_point(_xyz(np.array(lat), np.array(lon)), _chord(km)), np.int64)

    def country(self, lat: float, lon: float) -> str:
        return str(self.cc[self.pidx[self.ptree.query(_xyz(np.array(lat), np.array(lon)))[1]]])

    def region(self, lat: float, lon: float) -> tuple[str, str]:
        """(country code, "country.admin1") of the nearest populated place."""

        j = self.pidx[self.ptree.query(_xyz(np.array(lat), np.array(lon)))[1]]
        return str(self.cc[j]), f"{self.cc[j]}.{self.a1[j]}"


def _countries() -> tuple[dict[str, set[str]], dict[str, str]]:
    """country code -> language codes (2-letter prefixes too); normalised country name -> code."""

    langs, names = {}, {}
    for line in (GEONAMES / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        codes = {c.strip() for c in f[15].split(",") if c.strip()}
        langs[f[0]] = codes | {c.split("-")[0] for c in codes}
        names[norm(f[4])] = f[0]
    names |= {"usa": "US", "united states of america": "US", "uk": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
              "south korea": "KR", "korea": "KR", "russia": "RU", "czech republic": "CZ", "the netherlands": "NL", "holland": "NL"}
    return langs, names


def _hit(name: str, blob: str) -> float:
    """Match strength of a gazetteer name in the photo's text blob: its length if it occurs as a whole word (CJK: as a substring), else 0."""

    cjk = any("぀" <= c <= "鿿" or "가" <= c <= "힯" for c in name)
    if (len(name) * (2 if cjk else 1)) < MIN_NAME:
        return 0.0
    if cjk:
        return float(len(name) * 2) if name in blob else 0.0
    return float(len(name)) if f" {name} " in blob else 0.0


def score(tag: str, name: str) -> None:
    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    obs = json.loads((OUT / f"observe_{tag}_{name}.json").read_text(encoding="utf-8"))
    g = Gazetteer()
    langs, country_names = _countries()
    C = max(len(e["pool"]) for e in photos)
    F = np.zeros((len(photos), C, len(FEATURES)), np.float32)
    # global feature rarity: share of a random sample of candidates with each vocabulary feature within each radius
    rng = np.random.default_rng(0)
    sample = [c for e in rng.choice(photos, min(300, len(photos)), replace=False) for c in e["pool"][:5]]
    present = {r: {f: 0 for f in VOCAB} for r in RADII}
    for lat, lon in sample:
        for r in RADII:
            codes = set(g.code[g.near(lat, lon, r)])
            for f, fc in VOCAB.items():
                present[r][f] += bool(codes & set(fc))
    idf = {r: {f: -np.log((present[r][f] + 1) / (len(sample) + 2)) for f in VOCAB} for r in RADII}
    col = {f: i for i, f in enumerate(FEATURES)}
    table_path = OUT / f"textgeo_{name}.json"
    table = json.loads(table_path.read_text(encoding="utf-8")) if table_path.exists() else {}
    for i, e in enumerate(photos):
        o = (obs.get(e["image_id"]) or {}).get("parsed") or {}
        text_reg, text_cty = _region_scores(o.get("text") or [], table)
        name_reg, name_cty = _region_scores(o.get("place_names") or [], table)
        feats = o.get("features") or []
        text_blob = " " + " ".join(norm(t) for t in o.get("text") or []) + " "
        name_blob = " " + " ".join(norm(t) for t in o.get("place_names") or []) + " "
        lang = LANGUAGES.get(norm(o.get("language") or ""))
        script_langs = SCRIPTS.get(norm(o.get("script") or ""))
        side = norm(o.get("driving_side") or "")
        guess = country_names.get(norm(o.get("country") or ""))
        for k, (lat, lon) in enumerate(e["pool"]):
            near25 = g.near(lat, lon, 25.0)
            if len(near25):
                d = 2 * np.arcsin(np.clip(np.linalg.norm(g.tree.data[near25] - _xyz(np.array(lat), np.array(lon)), axis=1) / 2, 0, 1)) * EARTH_KM
            else:
                d = np.zeros(0)
            for r in RADII:
                ids = near25[d <= r]
                if feats:
                    codes = set(g.code[ids])
                    has = [f for f in feats if codes & set(VOCAB[f])]
                    F[i, k, col[f"feat frac {r:g}"]] = len(has) / len(feats)
                    F[i, k, col[f"feat idf {r:g}"]] = sum(idf[r][f] for f in has) - sum(idf[r][f] for f in feats if f not in has) * 0.25
            for blob, prefix in ((text_blob, "text match"), (name_blob, "name match")):
                if blob.strip():
                    best = {r: 0.0 for r in RADII}
                    for j, dist in zip(near25, d):
                        s = max((_hit(n, blob) for n in g.names(int(j))), default=0.0)
                        if s:
                            for r in RADII:
                                if dist <= r:
                                    best[r] = max(best[r], s)
                    for r in RADII:
                        F[i, k, col[f"{prefix} {r:g}"]] = best[r]
            cc, region = g.region(lat, lon)
            F[i, k, col["text region"]], F[i, k, col["text country"]] = text_reg.get(region, 0.0), text_cty.get(cc, 0.0)
            F[i, k, col["name region"]], F[i, k, col["name country"]] = name_reg.get(region, 0.0), name_cty.get(cc, 0.0)
            F[i, k, col["language ok"]] = 0.0 if lang is None else (1.0 if lang in langs.get(cc, set()) else -1.0)
            F[i, k, col["script ok"]] = 0.0 if script_langs is None else (1.0 if script_langs & langs.get(cc, set()) else -1.0)
            F[i, k, col["driving ok"]] = 0.0 if side not in ("left", "right") else (1.0 if (cc in LEFT_DRIVING) == (side == "left") else -1.0)
            F[i, k, col["country guess"]] = 0.0 if guess is None else (1.0 if guess == cc else -1.0)
        if i % 200 == 0:
            print(f"  {tag}: {i} / {len(photos)}", flush=True)
    np.savez(OUT / f"tools_{tag}_{name}.npz", F=F, image_ids=np.asarray([e["image_id"] for e in photos]))
    print(f"{tag}: tool features for {len(photos)} photos -> {OUT / f'tools_{tag}_{name}.npz'}")


def _load(tag: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """[photos, C, 3 + 2 * len(FEATURES)] features (rank, log rank, top-1, tools, tools minus pool mean), km to truth, validity, fixable, top-1 km."""

    from .strategy_search import _haversine_km

    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    targets = {r["image_id"]: r for r in json.loads((OUT / f"targets_{tag}.json").read_text(encoding="utf-8"))}
    T = np.load(OUT / f"tools_{tag}_{name}.npz")["F"]
    n, C = len(photos), T.shape[1]
    D, V = np.full((n, C), 1e5), np.zeros((n, C), bool)
    for i, e in enumerate(photos):
        D[i, : len(e["pool"])] = _haversine_km(*e["truth"], np.asarray(e["pool"]))
        V[i, : len(e["pool"])] = True
    k = np.arange(C)[None, :].repeat(n, 0)
    rank = np.stack([k, np.log1p(k), (k == 0)], -1).astype(np.float32)
    mean = (T * V[..., None]).sum(1, keepdims=True) / V.sum(1)[:, None, None]
    F = np.concatenate([rank, T, T - mean], -1).astype(np.float32)
    fix = np.array([targets[e["image_id"]]["fixable"] for e in photos])
    return F, D, V, fix, D[:, 0]


def _cols(group: str) -> list[int]:
    base = [0, 1, 2]
    if group == "rank":
        return base
    names = FEATURES if group == "all" else GROUPS[group]
    idx = [FEATURES.index(f) for f in names]
    return base + [3 + j for j in idx] + [3 + len(FEATURES) + j for j in idx]


def _auc(F: np.ndarray, D: np.ndarray, V: np.ndarray, j: int, rows: np.ndarray) -> float:
    wins = pairs = 0.0
    for i in np.flatnonzero(rows):
        a, b = F[i, V[i] & (D[i] < 25), j], F[i, V[i] & (D[i] >= 25), j]
        if len(a) and len(b):
            wins += (a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()
            pairs += len(a) * len(b)
    return wins / max(pairs, 1)


def _summary(label: str, hits: dict, base: np.ndarray, fix: np.ndarray) -> str:
    from .stage1_eval import _bootstrap

    parts = []
    for km in (1, 25):
        c = _bootstrap(hits[km] - (base < km))
        parts.append(f"< {km} km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
    fixes, breaks = hits[25][base >= 25].sum(), (1 - hits[25][base < 25]).sum()
    return f"  {label:22s}: " + "; ".join(parts) + f"; 25 km fixes {fixes:.0f} (on fixable {hits[25][fix].sum():.0f} of {fix.sum()}), breaks {breaks:.0f}"


def report(tags: Sequence[str], name: str) -> None:
    from .name_score import _fit

    parts = [_load(t, name) for t in tags]
    F, D, V, fix, base = (np.concatenate([p[j] for p in parts]) for j in range(5))
    n = len(D)
    print(f"{'+'.join(tags)} (n={n}), {name}: reranker top-1 < 1 / 25 km {(base < 1).mean():.1%} / {(base < 25).mean():.1%}; fixable {fix.sum()}")
    print("within-photo AUC (candidate < 25 km vs >= 25 km): all photos / fixable photos; share of candidates with a nonzero value")
    for j, f in enumerate(FEATURES):
        nz = (F[..., 3 + j][V] != 0).mean()
        print(f"  {f:16s} {_auc(F, D, V, 3 + j, np.ones(n, bool)):.3f} / {_auc(F, D, V, 3 + j, fix):.3f}   nonzero {nz:.0%}")
    folds = np.array_split(np.random.default_rng(0).permutation(n), 5)
    for group in ["rank", *GROUPS, "all"]:
        cols = _cols(group)
        hits = {km: np.zeros(n) for km in (1, 25)}
        for seed in range(3):
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                pick = np.argmax(_fit(F[tr][..., cols], D[tr], V[tr], seed)(F[f][..., cols], V[f]), 1)
                for km in hits:
                    hits[km][f] += (D[f, pick] < km) / 3
        print(_summary(f"CV rank + {group}" if group != "rank" else "CV rank only", hits, base, fix))


def transfer(fit_tags: Sequence[str], tag: str, name: str, group: str = "all") -> None:
    from .name_score import _fit

    parts = [_load(t, name) for t in fit_tags]
    C = max(p[0].shape[1] for p in parts + [_load(tag, name)])
    pad = lambda a, fill: np.concatenate([a, np.full((a.shape[0], C - a.shape[1], *a.shape[2:]), fill, a.dtype)], 1)
    Ff, Df, Vf = (np.concatenate([pad(p[j], fill) for p in parts]) for j, fill in ((0, 0), (1, 1e5), (2, False)))
    F, D, V, fix, base = _load(tag, name)
    F, D, V = pad(F, 0), pad(D, 1e5), pad(V, False)
    cols = _cols(group)
    n = len(D)
    hits = {km: np.zeros(n) for km in (1, 25)}
    for seed in range(3):
        pick = np.argmax(_fit(Ff[..., cols], Df, Vf, seed)(F[..., cols], V), 1)
        for km in hits:
            hits[km] += (D[np.arange(n), pick] < km) / 3
    print(f"fit on {'+'.join(fit_tags)} (n={len(Df)}), applied to {tag} (n={n}), tools '{group}', {name}: reranker top-1 < 1 / 25 km "
          f"{(base < 1).mean():.1%} / {(base < 25).mean():.1%}")
    print(_summary(f"rank + {group}", hits, base, fix))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("index", "textgeo", "score", "report", "transfer"))
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--tags", nargs="+", default=["dev"])
    parser.add_argument("--fit", nargs="+", default=["dev", "val"])
    parser.add_argument("--name", default="base")
    parser.add_argument("--group", default="all", choices=["rank", *GROUPS, "all"])
    args = parser.parse_args(argv)
    if args.node == "index":
        index()
    elif args.node == "textgeo":
        textgeo(args.name, args.tags)
    elif args.node == "score":
        score(args.tag, args.name)
    elif args.node == "report":
        report(args.tags, args.name)
    else:
        transfer(args.fit, args.tag, args.name, args.group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
