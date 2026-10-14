# Gate for nested proposals (lesson 83; follows lesson 82): the base 4B proposes {country, region, place?} or keep, blind (photo only), K samples.
# Each sample becomes one answer by the fallback rule: a place whose map matches fall inside the proposed region (else country) -> Pinpoint
# restricted to photos near those matches; else the region -> Pinpoint restricted to it; else the country; else (keep / nothing usable) Pinpoint's
# top-1. Areas are the Overture labels of lesson 64 (country, first-level region), the 4B's names matched to them by name. Photos and groups
# as lesson 82 (302 fixable dev3k + 300 random others), reference = the retrained reranker's top-1 (photo set dev3k_setft_a).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.nested_passk sample      (VLM served at :8765)
#        PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.nested_passk evaluate
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.nested_passk report

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .query_passk import K, KM, OUT, _photos

SAMPLES = OUT / "nested_passk_dev3k.json"
ANSWERS = OUT / "nested_passk_dev3k_answers.npz"
PROMPT = ("Where was this photo taken? Use what you see (text, language, architecture, vegetation, landmarks, vehicles, road markings) and what "
          "you know. Give the country, the first-level region (state / province / region) and, only if the photo shows a specific identifiable "
          "place (a landmark, building, street, business), its name as you would search it on a map. If you cannot tell even the country, "
          'answer keep. Answer with JSON only: {"country": "...", "region": "...", "place": "..."} (null for parts you cannot tell) or '
          '{"keep": true}.')
PLACE_KM = 5.0  # a place's map matches: Pinpoint restricted to index photos within this distance
ALIASES = {"usa": "United States", "us": "United States", "united states of america": "United States", "america": "United States",
           "uk": "United Kingdom", "great britain": "United Kingdom", "britain": "United Kingdom", "england": "United Kingdom",
           "scotland": "United Kingdom", "wales": "United Kingdom", "northern ireland": "United Kingdom", "czech republic": "Czechia",
           "holland": "Netherlands", "the netherlands": "Netherlands", "south korea": "South Korea", "korea": "South Korea",
           "russian federation": "Russia", "turkey": "Türkiye", "ivory coast": "Côte d'Ivoire", "burma": "Myanmar", "uae": "United Arab Emirates"}
GENERIC = re.compile(r"\b(region|province|state|county|prefecture|oblast|voivodeship|department|governorate|autonomous community|of|the|"
                     r"metropolitan|city|district|kraj|land|territory)\b")


def sample(server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from .name_score import _post, _read
    from .sft_data import MP16Images

    photos, images = _photos(), MP16Images()

    def ask(e: dict) -> dict:
        body = {"model": "vlm", "temperature": 1.0, "top_p": 1.0, "n": K, "max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_read(e, images)).decode()}},
                    {"type": "text", "text": PROMPT}]}]}
        try:
            raw = [c["message"]["content"] for c in _post(server, "/v1/chat/completions", body)["choices"]]
        except Exception:  # noqa: BLE001
            raw = []
        out = []
        for r in raw:
            m = re.search(r"\{.*\}", r or "", re.S)
            try:
                obj = json.loads(m.group(0)) if m else {}
            except Exception:  # noqa: BLE001
                obj = {}
            out.append({k: (str(obj[k]).strip() if obj.get(k) not in (None, "", "null") else None) for k in ("country", "region", "place")})
        return {"image_id": e["image_id"], "row": int(e["row"]), "group": e["group"], "truth": e["truth"], "top1": e["candidates"][0][1:],
                "samples": out}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(ask, photos))
    SAMPLES.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} photos, {sum(not r['samples'] for r in rows)} failed -> {SAMPLES}")


class _Names:
    """The 4B's country / region names -> the Overture labels' strings."""

    def __init__(self, country: np.ndarray, region: np.ndarray) -> None:
        from .text_search import norm

        self.norm = norm
        self.countries = {norm(c): c for c in set(country) - {""}}
        self.regions: dict[str, dict[str, str]] = {}
        for cr in set(region) - {""}:
            c, r = cr.split("|", 1)
            self.regions.setdefault(c, {})[self._key(r)] = cr

    def _key(self, name: str) -> str:
        return re.sub(r"\s+", " ", GENERIC.sub(" ", self.norm(name))).strip()

    def country(self, name: str | None) -> str | None:
        if not name:
            return None
        k = self.norm(name)
        if k in ALIASES:
            return ALIASES[k]
        if k in self.countries:
            return self.countries[k]
        m = difflib.get_close_matches(k, list(self.countries), n=1, cutoff=0.85)
        return self.countries[m[0]] if m else None

    def region(self, country: str | None, name: str | None, raw_country: str | None = None) -> str | None:
        if not country:
            return None
        table = self.regions.get(country, {})
        candidates = [name] if name else []
        if raw_country and self.norm(raw_country) != self.norm(country):  # "England" given as the country: the UK's region
            candidates.append(raw_country)
        for n in candidates:
            k = self._key(n)
            if not k:
                continue
            if k in table:
                return table[k]
            m = difflib.get_close_matches(k, list(table), n=1, cutoff=0.8)
            if m:
                return table[m[0]]
            hits = [v for t, v in table.items() if len(t) >= 4 and (t in k or k in t)]
            if len(hits) == 1:
                return hits[0]
        return None


def evaluate() -> None:
    from ..tools.maps import MapsSearch
    from ..tools.pinpoint import Constraint, Pinpoint
    from .strategy_search import load_world

    rows = json.loads(SAMPLES.read_text(encoding="utf-8"))
    tool = Pinpoint()
    country_lab, region_lab = tool.labels()
    names = _Names(country_lab, region_lab)
    ms = MapsSearch()

    # each sample -> (kind, key): ("place", ...), ("region", r), ("country", c) or ("keep", "")
    plan, jobs = [], {}
    place_cache: dict[tuple[str, str], np.ndarray | None] = {}
    n_place = n_region = n_country = 0
    for i, r in enumerate(rows):
        per = []
        for s in r["samples"]:
            c = names.country(s["country"])
            reg = names.region(c, s["region"], s["country"])
            kind, key = "keep", ""
            if s["place"]:
                area = reg or c or ""
                ck = (s["place"], area)
                if ck not in place_cache:
                    try:
                        found = ms.query(s["place"])
                    except Exception:  # noqa: BLE001
                        found = []
                    pts = np.asarray([[p.lat, p.lon] for p in found], np.float64).reshape(-1, 2)
                    if len(pts) and area:
                        pc, pr = tool.labels_at(pts)
                        pts = pts[(pr == area) if reg else (pc == area)]
                    elif not area:
                        pts = pts[:1]  # no area to check against: only the top match
                    place_cache[ck] = pts[:5] if len(pts) else None
                if place_cache[ck] is not None:
                    kind, key = "place", f"{s['place']}\t{area}"
                    n_place += 1
            if kind == "keep" and reg:
                kind, key = "region", reg
                n_region += 1
            elif kind == "keep" and c:
                kind, key = "country", c
                n_country += 1
            per.append((kind, key))
            if kind != "keep":
                jobs.setdefault((i, kind, key), None)
        plan.append(per)
    print(f"samples: place {n_place}, region {n_region}, country {n_country}; {len(jobs)} distinct restricted searches", flush=True)

    world = load_world()
    keys = list(jobs)
    photo_rows = np.asarray([rows[i]["row"] for i, _, _ in keys])
    emb = np.asarray(world.mp16["embeddings"][photo_rows], np.float32)
    authors = tool.mp16_author(photo_rows)
    cons = []
    for i, kind, key in keys:
        if kind == "place":
            place, area = key.split("\t")
            cons.append(Constraint(near=(place_cache[(place, area)], PLACE_KM)))
        else:
            cons.append(Constraint(**{kind: key}))
    found = tool.search(emb, cons, authors)
    top, top_score = found.latlon[:, 0], found.scores[:, 0]
    answer = {k: top[j] for j, k in enumerate(keys)}
    score = {k: top_score[j] for j, k in enumerate(keys)}
    own_rows = np.asarray([r["row"] for r in rows])
    free = tool.search(np.asarray(world.mp16["embeddings"][own_rows], np.float32), None, tool.mp16_author(own_rows))
    base_score = free.scores[:, 0]  # Pinpoint's own top-1 score (the reranker's verification signal, lesson 79)

    truth_c, truth_r = tool.mp16_labels(np.asarray([r["row"] for r in rows]))
    ll = np.full((len(rows), K, 2), np.nan)
    kinds = np.full((len(rows), K), "", dtype=object)
    keyed = np.full((len(rows), K), "", dtype=object)
    sc = np.full((len(rows), K), -np.inf)
    for i, per in enumerate(plan):
        for j, (kind, key) in enumerate(per[:K]):
            kinds[i, j] = kind
            ll[i, j] = rows[i]["top1"] if kind == "keep" else answer[(i, kind, key)]
            keyed[i, j] = f"{kind}\t{key}"
            sc[i, j] = base_score[i] if kind == "keep" else score[(i, kind, key)]
            if not np.isfinite(ll[i, j]).all():  # empty restricted search: keep Pinpoint's answer
                ll[i, j] = rows[i]["top1"]
    sample_c = np.asarray([[names.country(s["country"]) or "" for s in r["samples"][:K]] + [""] * (K - len(r["samples"][:K])) for r in rows])
    sample_r = np.asarray([[names.region(names.country(s["country"]), s["region"], s["country"]) or "" for s in r["samples"][:K]]
                           + [""] * (K - len(r["samples"][:K])) for r in rows])
    np.savez(ANSWERS, ll=ll, kinds=kinds.astype(str), keys=keyed.astype(str), score=sc, base_score=base_score, truth_country=truth_c.astype(str), truth_region=truth_r.astype(str),
             sample_country=sample_c, sample_region=sample_r)
    print(f"-> {ANSWERS}")


def report() -> None:
    from .text_search import haversine_km

    rows = json.loads(SAMPLES.read_text(encoding="utf-8"))
    z = np.load(ANSWERS)
    ll, kinds = z["ll"], z["kinds"]
    truth = np.asarray([r["truth"] for r in rows], np.float64)
    top1 = np.asarray([r["top1"] for r in rows], np.float64)

    def km(a, b):
        return np.asarray([haversine_km(a[i, 0], a[i, 1], b[i:i + 1, 0], b[i:i + 1, 1])[0] for i in range(len(a))])

    base = km(top1, truth) < KM
    hit = np.stack([km(np.nan_to_num(ll[:, j]), truth) < KM for j in range(K)], 1)
    group = np.asarray([r["group"] for r in rows])
    weight = {"fixable": 302 / 3000, "control": 2698 / 3000}
    scaled = {}
    for g in ("fixable", "control"):
        m = group == g
        wrong = m & ~base
        c_ok = z["sample_country"] == z["truth_country"][:, None]
        r_ok = z["sample_region"] == z["truth_region"][:, None]
        print(f"\n{g}: {m.sum()} photos; Pinpoint (retrained) top-1 < {KM:.0f} km {base[m].mean():.1%}")
        print("  sample kinds: " + ", ".join(f"{k} {(kinds[m] == k).mean():.0%}" for k in ("place", "region", "country", "keep")))
        print(f"  4B country right (share of samples) {c_ok[m].mean():.1%}, region right {r_ok[m].mean():.1%}")
        print(f"  top-1 wrong ({wrong.sum()}): true country in 16 samples {c_ok[wrong].any(1).mean():.1%}, true region {r_ok[wrong].any(1).mean():.1%}")
        oracle = base | hit.any(1)
        first = hit[:, 0]
        mean = hit.mean(1)
        print(f"  < {KM:.0f} km: Pinpoint {base[m].mean():.1%}; first sample {first[m].mean():.1%} ({first[m].mean() - base[m].mean():+.1%}); "
              f"mean over samples {mean[m].mean():.1%} ({mean[m].mean() - base[m].mean():+.1%}); best of Pinpoint + 16 samples {oracle[m].mean():.1%} "
              f"({oracle[m].mean() - base[m].mean():+.1%})")
        for kind in ("place", "region", "country"):
            sel = (kinds == kind) & m[:, None]
            if sel.any():
                b = np.broadcast_to(base[:, None], hit.shape)
                print(f"    {kind} samples: fixes {(hit & ~b & sel).sum()}, breaks {(~hit & b & sel).sum()} (of {sel.sum()})")
        scaled[g] = (first[m].mean() - base[m].mean(), mean[m].mean() - base[m].mean(), oracle[m].mean() - base[m].mean())
    s = [sum(weight[g] * scaled[g][k] for g in weight) for k in range(3)]
    print(f"\ndev3k-weighted change at {KM:.0f} km: first sample {s[0]:+.2%}, mean over samples {s[1]:+.2%}, best of 16 (ceiling) {s[2]:+.2%}")
    _selector(z, hit, base, group, weight)


def _selector(z, hit, base, group, weight) -> None:
    """No-training selection: per photo the most voted non-keep answer among the samples (ties: higher restricted score); override Pinpoint
    when its vote share >= v and its restricted top-1 score minus Pinpoint's own top-1 score >= g. (v, g) chosen on one half of the photos,
    measured on the other (2-fold), dev3k-weighted change at 25 km."""

    n = len(base)
    votes, gap, fix = np.zeros(n), np.full(n, -np.inf), np.zeros(n, bool)
    for i in range(n):
        keys = [k for k in z["keys"][i] if not k.startswith("keep")]
        if not keys:
            continue
        uniq = sorted(set(keys), key=lambda k: (-keys.count(k), -z["score"][i][list(z["keys"][i]).index(k)]))
        j = list(z["keys"][i]).index(uniq[0])
        votes[i], gap[i], fix[i] = keys.count(uniq[0]) / len(z["keys"][i]), z["score"][i, j] - z["base_score"][i], hit[i, j]
    w = np.where(group == "fixable", weight["fixable"] / (group == "fixable").sum(), weight["control"] / (group == "control").sum())

    def change(mask, sub=None):
        sub = np.ones(n, bool) if sub is None else sub
        return float((w * (mask & sub) * (fix.astype(float) - base)).sum() / w[sub].sum())

    finite = gap[np.isfinite(gap)]
    grid = [(v, g) for v in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75) for g in np.quantile(finite, np.linspace(0, 0.95, 20))]
    fold = np.arange(n) % 2
    total, chosen = 0.0, []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        best = max(grid, key=lambda vg: change((votes >= vg[0]) & (gap >= vg[1]), tr))
        chosen.append(best)
        total += change((votes >= best[0]) & (gap >= best[1]), te) / 2
    full = max(grid, key=lambda vg: change((votes >= vg[0]) & (gap >= vg[1])))
    m = (votes >= full[0]) & (gap >= full[1])
    print(f"selector (votes, score gap): 2-fold held-out {total:+.2%} (choices {[(v, round(float(g), 2)) for v, g in chosen]}); "
          f"in-sample best {change(m):+.2%} at {full[0]}, {full[1]:.2f}: overrides {m.mean():.1%} of photos, fixes {(m & fix & ~base.astype(bool)).sum()}, "
          f"breaks {(m & ~fix & base.astype(bool)).sum()}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("sample", "evaluate", "report"))
    args = parser.parse_args(argv)
    {"sample": sample, "evaluate": evaluate, "report": report}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
