# Is the VLM's country guess better than Pinpoint's? (TEXT_SEARCH_PLAN.md option 1, "country first".) Compares the 9B's country (observe.py
# extract, `country` field) with the country of Pinpoint's top-1 on dev3k, and writes a picks file so pinpoint_constrained can restrict Pinpoint to
# the VLM's country (its "pick country" arm). dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.vlm_country compare --tag dev3k
#        RUN="vlm_country sample" NAME=4b bash scripts/clue_judge.sh      (8 sampled country answers per photo; then `confidence`)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.vlm_country confidence --tag dev3k --name 4b
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pinpoint_constrained run --tag dev3k --picks vlmcountry_9b

"""compare: country accuracy of the VLM and of Pinpoint's top-1 (Overture labels; VLM names mapped to them by alias / exact match), where they
         disagree who is right, and by Pinpoint's top-1 score quintile. Writes text_search_picks_{tag}_vlmcountry_9b.json: for each photo whose
         VLM country differs from Pinpoint's, the coordinate of an index photo labelled with that country (so the nearest-index label is it)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
ALIASES = {"usa": "United States", "us": "United States", "united states of america": "United States", "america": "United States",
           "uk": "United Kingdom", "england": "United Kingdom", "scotland": "United Kingdom", "wales": "United Kingdom",
           "northern ireland": "United Kingdom", "great britain": "United Kingdom", "britain": "United Kingdom",
           "czech republic": "Czechia", "korea": "South Korea", "republic of korea": "South Korea", "holland": "Netherlands",
           "the netherlands": "Netherlands", "hong kong": "China", "macau": "China", "vatican": "Vatican City", "burma": "Myanmar",
           "uae": "United Arab Emirates", "russian federation": "Russia", "viet nam": "Vietnam", "turkiye": "Turkey"}


def _country(name: str | None, known: set[str]) -> str | None:
    if not name or str(name).strip().lower() in ("none", "null", "unknown", ""):
        return None
    n = str(name).strip()
    n = ALIASES.get(n.lower(), n)
    return n if n in known else next((k for k in known if k.lower() == n.lower()), None)


def compare(tag: str, source: str = "observe_{tag}_9b.json") -> None:
    from .pinpoint_constrained import LABELS_NPZ
    from .pinpoint_reference import load_pools

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    observed = json.loads((OUT / source.format(tag=tag)).read_text(encoding="utf-8"))
    z = np.load(LABELS_NPZ)
    known = set(z["country"].tolist()) - {""}
    truth = [labels[e["image_id"]]["truth"].get("country") for e in photos]
    pin = [labels[e["image_id"]]["pool"][0].get("country") if labels[e["image_id"]]["pool"] else None for e in photos]
    raw = [(observed.get(e["image_id"], {}).get("parsed") or {}).get("country") for e in photos]
    vlm = [_country(r, known) for r in raw]
    pools = load_pools("mp16")
    score = np.asarray([pools["scores"][e["index"]][0] for e in photos])
    has = np.asarray([t is not None for t in truth])
    v_right = np.asarray([v is not None and v == t for v, t in zip(vlm, truth)])
    p_right = np.asarray([p is not None and p == t for p, t in zip(pin, truth)])
    answered = np.asarray([v is not None for v in vlm])
    unmapped = sum(r not in (None, "None") and str(r).lower() not in ("none", "null", "unknown") and v is None for r, v in zip(raw, vlm))
    print(f"{tag}: {len(photos)} photos, {has.sum()} with a truth country; VLM names a country on {answered.sum()} ({unmapped} names not mapped)")
    print(f"  country right: Pinpoint top-1 {p_right[has].mean():.1%} of photos; VLM {v_right[has].mean():.1%} of photos, "
          f"{v_right[has & answered].mean():.1%} of those it answers (Pinpoint {p_right[has & answered].mean():.1%} on the same photos)")
    differ = has & answered & np.asarray([v != p for v, p in zip(vlm, pin)])
    print(f"  VLM and Pinpoint disagree on {differ.sum()} photos: VLM right {(v_right & differ).sum()}, Pinpoint right {(p_right & differ).sum()}, "
          f"neither {(differ & ~v_right & ~p_right).sum()}")
    print("  by Pinpoint's top-1 score quintile (low = unsure): disagreements, VLM right, Pinpoint right")
    edges = np.quantile(score, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for k in range(5):
        q = differ & (score >= edges[k]) & (score <= edges[k + 1])
        print(f"    q{k + 1}: {q.sum():4d}  {(v_right & q).sum():4d}  {(p_right & q).sum():4d}")
    from .region_evidence import _Labeller

    lab = _Labeller()
    # an index photo labelled with each country, so pinpoint_constrained's nearest-index labelling returns exactly that country
    example: dict[str, int] = {}
    for j, c in enumerate(lab.country):
        if c and c not in example:
            example[c] = j
    latlon = lab.tree.data
    lat = np.degrees(np.arcsin(latlon[:, 2]))
    lon = np.degrees(np.arctan2(latlon[:, 1], latlon[:, 0]))
    picks = {e["image_id"]: {"lat": float(lat[example[v]]), "lon": float(lon[example[v]]), "country": v}
             for e, v, d in zip(photos, vlm, differ | (answered & ~has & np.asarray([v != p for v, p in zip(vlm, pin)])))
             if d and v in example}
    path = OUT / f"text_search_picks_{tag}_vlmcountry_9b.json"
    path.write_text(json.dumps(picks) + "\n", encoding="utf-8")
    print(f"  {len(picks)} photos where the VLM's country differs from Pinpoint's -> {path}")


SAMPLE_PROMPT = "In which country was this photo taken? Answer with the country name only."


def sample(tag: str, name: str, n: int = 8, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    """n answers per photo at temperature 1 -> artifacts/agent/country_samples_{tag}_{name}.json {image_id: [answers]}."""

    import base64
    from concurrent.futures import ThreadPoolExecutor

    from .name_score import _post
    from .sft_data import MP16Images

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    images = MP16Images()

    def ask(e: dict) -> list[str]:
        image = base64.b64encode(images.read(e["image_id"])).decode()
        body = {"model": "vlm", "temperature": 1.0, "n": n, "max_tokens": 12, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
                                                          {"type": "text", "text": SAMPLE_PROMPT}]}]}
        try:
            return [c["message"]["content"].strip().strip(".") for c in _post(server, "/v1/chat/completions", body)["choices"]]
        except Exception as error:  # noqa: BLE001
            print(f"  failed: {error}", flush=True)
            return []

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(ask, photos))
    path = OUT / f"country_samples_{tag}_{name}.json"
    path.write_text(json.dumps({e["image_id"]: a for e, a in zip(photos, answers)}) + "\n", encoding="utf-8")
    print(f"{tag} / {name}: {len(photos)} photos x {n} answers -> {path}")


def confidence(tag: str, name: str) -> None:
    """The modal sampled country and its share (agreement) as confidence: where it differs from Pinpoint's country, who is right, by agreement."""

    import collections

    from .pinpoint_constrained import LABELS_NPZ

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    samples = json.loads((OUT / f"country_samples_{tag}_{name}.json").read_text(encoding="utf-8"))
    known = set(np.load(LABELS_NPZ)["country"].tolist()) - {""}
    rows = []
    for e in photos:
        lab = labels[e["image_id"]]
        truth, pin = lab["truth"].get("country"), (lab["pool"][0].get("country") if lab["pool"] else None)
        mapped = [c for c in (_country(a, known) for a in samples.get(e["image_id"], [])) if c]
        if not truth or not mapped:
            continue
        mode, count = collections.Counter(mapped).most_common(1)[0]
        rows.append((e["image_id"], mode, count / len(samples[e["image_id"]]), mode == truth, pin == truth, mode != pin))
    v = np.asarray([r[3] for r in rows]); p = np.asarray([r[4] for r in rows]); d = np.asarray([r[5] for r in rows]); a = np.asarray([r[2] for r in rows])
    print(f"{tag} / {name}: {len(rows)} photos with a truth and a mapped answer; modal country right {v.mean():.1%} (Pinpoint {p.mean():.1%} on the same)")
    print("  where the modal country differs from Pinpoint's, by agreement of the 8 samples: photos, VLM right, Pinpoint right")
    for lo, hi in ((0, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.01)):
        q = d & (a >= lo) & (a < hi)
        print(f"    agreement [{lo:.2f}, {min(hi, 1):.2f}{']' if hi > 1 else ')'}: {q.sum():4d}  {(v & q).sum():4d}  {(p & q).sum():4d}")
    picks_from = {r[0]: r[1] for r in rows if r[5] and r[2] >= 1.0}
    from .region_evidence import _Labeller

    lab = _Labeller()
    example: dict[str, int] = {}
    for j, c in enumerate(lab.country):
        if c and c not in example:
            example[c] = j
    xyz = lab.tree.data
    lat, lon = np.degrees(np.arcsin(xyz[:, 2])), np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    picks = {k: {"lat": float(lat[example[c]]), "lon": float(lon[example[c]]), "country": c} for k, c in picks_from.items() if c in example}
    path = OUT / f"text_search_picks_{tag}_vlmcountry_{name}_unanimous.json"
    path.write_text(json.dumps(picks) + "\n", encoding="utf-8")
    print(f"  {len(picks)} photos with a unanimous country that differs from Pinpoint's -> {path}")


LANGUAGE_CODES = {"english": "en", "spanish": "es", "french": "fr", "italian": "it", "german": "de", "portuguese": "pt", "chinese": "zh",
                  "russian": "ru", "japanese": "ja", "dutch": "nl", "catalan": "ca", "arabic": "ar", "greek": "el", "croatian": "hr", "czech": "cs",
                  "swedish": "sv", "finnish": "fi", "polish": "pl", "norwegian": "no", "thai": "th", "danish": "da", "hungarian": "hu",
                  "turkish": "tr", "korean": "ko", "vietnamese": "vi", "hebrew": "he", "indonesian": "id", "malay": "ms", "hindi": "hi",
                  "romanian": "ro", "bulgarian": "bg", "serbian": "sr", "slovak": "sk", "slovenian": "sl", "ukrainian": "uk", "icelandic": "is",
                  "estonian": "et", "latvian": "lv", "lithuanian": "lt", "welsh": "cy", "irish": "ga", "basque": "eu", "galician": "gl"}


def explicit(tag: str) -> None:
    """Explicit country clues as constraints: country-code domains and phone prefixes (text_search hints on region_evidence's searches) and the
    language of text read in the photo (observe.py; only photos where the 9B read some text). A clue allows a set of countries (a language:
    every country where GeoNames lists it). Counts how often the truth is allowed, how often Pinpoint's top-1 country is not, and how often
    Pinpoint is then wrong (>= 25 km); single-country clues are the ones a restricted search can use directly."""

    import collections
    import re

    from .strategy_search import _haversine_km
    from .text_search import _countries

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    _, _, iso_names = _countries()
    name_iso = {v: k for k, v in iso_names.items()} | {"Czechia": "CZ", "Russia": "RU", "South Korea": "KR", "Vietnam": "VN", "Taiwan": "TW",
                                                       "Iran": "IR", "Syria": "SY", "Laos": "LA", "Bolivia": "BO", "Venezuela": "VE", "Tanzania": "TZ"}
    lang_countries: dict[str, set[str]] = collections.defaultdict(set)
    for line in (Path("data/pinpoint/geonames") / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        for code in f[15].split(","):
            if code:
                lang_countries[code.split("-")[0].lower()].add(f[0])
    clues: dict[str, dict[str, set[str]]] = collections.defaultdict(dict)  # clue type -> image id -> allowed countries
    for r in json.loads((OUT / f"region_evidence_{tag}.json").read_text(encoding="utf-8")):
        m = re.search(r"\(([A-Z]{2})\)", r.get("hint") or "")
        if m:
            kind = "country-code domain" if r["kind"] == "domain" else "phone prefix"
            clues[kind].setdefault(r["image_id"], set()).add(m.group(1))
        elif (r.get("hint") or "").startswith("phone prefix +1 "):
            clues["phone prefix"].setdefault(r["image_id"], set()).update({"US", "CA"})
    for k, v in json.loads((OUT / f"observe_{tag}_9b.json").read_text(encoding="utf-8")).items():
        parsed = v.get("parsed") or {}
        if not parsed.get("text"):
            continue
        language = parsed.get("language")
        names = language if isinstance(language, list) else [language]
        allowed = set()
        for n in names:
            code = LANGUAGE_CODES.get(str(n).strip().strip("[]'").lower())
            if code:
                allowed |= lang_countries.get(code, set())
        if allowed:
            clues["language of read text"][k] = allowed
            if not ({"en"} & {LANGUAGE_CODES.get(str(n).strip().strip("[]'").lower()) for n in names}):
                clues["language of read text, not English"][k] = allowed
    km = {e["image_id"]: _haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] for e in photos}
    print(f"{tag}: explicit country clues as constraints (truth / Pinpoint countries from Overture labels, mapped to ISO codes)")
    print(f"  {'clue':38s} {'photos':>6s} {'truth allowed':>13s} {'Pinpoint not allowed':>20s} {'of those Pinpoint wrong':>23s} {'single country':>14s} "
          f"{'single: truth = it, Pinpoint not':>32s}")
    for kind, table in clues.items():
        n = allowed_truth = violated = violated_wrong = single = single_fix = 0
        for k, allowed in table.items():
            lab = labels.get(k)
            if lab is None:
                continue
            truth = name_iso.get(lab["truth"].get("country") or "")
            pin = name_iso.get((lab["pool"][0].get("country") if lab["pool"] else None) or "")
            if not truth:
                continue
            n += 1
            allowed_truth += truth in allowed
            if pin and pin not in allowed:
                violated += 1
                violated_wrong += km[k] >= 25
            if len(allowed) == 1:
                single += 1
                single_fix += truth in allowed and pin not in allowed
        print(f"  {kind:38s} {n:6d} {allowed_truth / max(n, 1):13.0%} {violated:20d} {violated_wrong:23d} {single:14d} {single_fix:32d}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("compare", "sample", "confidence", "explicit"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="4b")
    args = parser.parse_args(argv)
    {"compare": lambda: compare(args.tag), "sample": lambda: sample(args.tag, args.name), "confidence": lambda: confidence(args.tag, args.name),
     "explicit": lambda: explicit(args.tag)}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
