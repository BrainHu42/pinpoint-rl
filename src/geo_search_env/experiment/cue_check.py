# Perception in the model, knowledge in a table (after lessons 21 and 79: attribute / constraint checks tested as whole-pipeline features). The
# VLM reports what it sees on lesson 106's 202 dev3k photos (languages and scripts of visible text, driving side, web domains); GeoNames'
# countryInfo (languages, tld) and a left-driving list say which of the pair's two areas each cue fits. A cue decides a pair when it fits
# exactly one area's country. English, Latin script and generic domains decide nothing (they appear everywhere).
# Usage: NAME=4b_cues RUN="cue_check ask" GPU_UTIL=0.3 bash scripts/clue_judge.sh
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.cue_check report --name 4b_cues

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .names_judge import OUT, SET, _images

GEONAMES = Path("data/pinpoint/geonames/countryInfo.txt")
SCRIPTS = ["Latin", "Cyrillic", "Greek", "Arabic", "Hebrew", "Chinese", "Japanese kana", "Korean Hangul", "Thai", "Devanagari", "Georgian",
           "Armenian", "Bengali", "Tamil", "Khmer", "Lao", "Burmese", "Sinhala", "Ethiopic", "other"]
SCRIPT_OF = {"Cyrillic": "ru uk be bg sr mk kk ky mn tg", "Greek": "el", "Arabic": "ar fa ur ps ug ku", "Hebrew": "he yi", "Chinese": "zh",
             "Japanese kana": "ja", "Korean Hangul": "ko", "Thai": "th", "Devanagari": "hi ne mr", "Georgian": "ka", "Armenian": "hy",
             "Bengali": "bn", "Tamil": "ta", "Khmer": "km", "Lao": "lo", "Burmese": "my", "Sinhala": "si", "Ethiopic": "am ti"}
LANG = {"english": "en", "french": "fr", "german": "de", "spanish": "es", "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
        "ukrainian": "uk", "polish": "pl", "czech": "cs", "slovak": "sk", "hungarian": "hu", "romanian": "ro", "bulgarian": "bg", "greek": "el",
        "turkish": "tr", "arabic": "ar", "hebrew": "he", "persian": "fa", "farsi": "fa", "chinese": "zh", "mandarin": "zh", "cantonese": "zh",
        "japanese": "ja", "korean": "ko", "thai": "th", "vietnamese": "vi", "indonesian": "id", "malay": "ms", "hindi": "hi", "swedish": "sv",
        "norwegian": "no", "danish": "da", "finnish": "fi", "icelandic": "is", "estonian": "et", "latvian": "lv", "lithuanian": "lt",
        "croatian": "hr", "serbian": "sr", "slovenian": "sl", "bosnian": "bs", "albanian": "sq", "macedonian": "mk", "catalan": "ca",
        "basque": "eu", "galician": "gl", "welsh": "cy", "irish": "ga", "gaelic": "gd", "maltese": "mt", "tagalog": "tl", "filipino": "tl",
        "swahili": "sw", "afrikaans": "af", "georgian": "ka", "armenian": "hy", "urdu": "ur", "bengali": "bn", "tamil": "ta", "khmer": "km",
        "lao": "lo", "burmese": "my", "nepali": "ne", "sinhala": "si", "amharic": "am", "mongolian": "mn", "kazakh": "kk", "latin": "la"}
LEFT = set(("GB IE IM JE GG MT CY IN PK BD LK NP BT MV JP TH MY SG BN ID TL HK MO AU NZ PG FJ SB TO WS KI TV NR ZA NA BW ZW ZM MW MZ TZ KE "
            "UG LS SZ MU SC JM BS BB TT GD LC VC DM AG KN GY SR AI BM KY VG MS TC FK SH CK NU VI").split())
ALIAS = {"Türkiye": "Turkey", "Myanmar": "Myanmar", "Ivory Coast": "Ivory Coast", "Republic of the Congo": "Republic of the Congo",
         "Russian Federation": "Russia", "Vatican City": "Vatican", "Macau": "Macao", "Eswatini": "Eswatini", "Palestinian Territory": "Palestinian Territory"}
PROMPT = """Report only what you can actually see in this photo.
languages: the languages of any readable text (signs, shop names, posters, labels), not counting brand names; [] if none.
scripts: the writing systems of any visible text.
driving_side: "left" or "right" only if moving or parked vehicles on a road make it clear which side traffic drives on, else "unknown".
web_domains: country web domains seen in text (".de", ".co.uk"), [] if none."""
SCHEMA = {"type": "json_schema", "json_schema": {"name": "cues", "schema": {
    "type": "object", "properties": {
        "languages": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "scripts": {"type": "array", "items": {"type": "string", "enum": SCRIPTS}, "maxItems": 4},
        "driving_side": {"type": "string", "enum": ["left", "right", "unknown"]},
        "web_domains": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
    "required": ["languages", "scripts", "driving_side", "web_domains"], "additionalProperties": False}}}


def _lang(x: str) -> str | None:  # a language name or ISO code ("Spanish", "es", "zh-Hant") -> ISO 639-1
    x = x.lower().strip()
    return LANG.get(x) or (x.split("-")[0] if x.split("-")[0] in LANG.values() else None)


def _countries() -> dict[str, dict]:
    out = {}
    for line in GEONAMES.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        langs = {x.split("-")[0].lower() for x in f[15].split(",") if x}
        out[f[4]] = {"iso": f[0], "tld": f[9].lower(), "langs": langs,
                     "scripts": {s for s, codes in SCRIPT_OF.items() if langs & set(codes.split())}}
    return out


def ask(name: str, server: str = "http://127.0.0.1:8765", workers: int = 16) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from .name_score import _post

    set_ = json.loads(SET.read_text(encoding="utf-8"))
    imgs = _images(set_)

    def run(ex):
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 200, "response_format": SCHEMA, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": imgs[ex]}}, {"type": "text", "text": PROMPT}]}]}
        try:
            return {"ex": ex, **json.loads(_post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"])}
        except Exception as e:  # noqa: BLE001
            return {"ex": ex, "error": str(e)[:200]}

    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(run, sorted(imgs)))
    path = OUT / f"cue_check_dev3k_{name}.json"
    path.write_text(json.dumps(res), encoding="utf-8")
    print(f"-> {path}: {len(res)} photos, {sum('error' in r for r in res)} errors")


def _fits(cue: tuple[str, str], c: dict) -> bool:
    kind, v = cue
    if kind == "language":
        return v in c["langs"]
    if kind == "script":
        return v in c["scripts"]
    if kind == "driving":
        return (c["iso"] in LEFT) == (v == "left")
    return v == c["tld"]  # domain


def report(name: str) -> None:
    set_ = json.loads(SET.read_text(encoding="utf-8"))
    cands = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r"(ex\d+): A = (.*?); B = (.*)", set_["candidates"])}
    res = {r["ex"]: r for r in json.loads((OUT / f"cue_check_dev3k_{name}.json").read_text()) if "error" not in r}
    table = _countries()
    country = lambda c: table.get(ALIAS.get(c.rsplit(", ", 1)[1], c.rsplit(", ", 1)[1]))  # noqa: E731
    missing = sorted({c.rsplit(", ", 1)[1] for p in cands.values() for c in p if country(c) is None})
    print(f"{name}: {len(res)} photos; countries not in GeoNames: {missing}")
    print("cues reported: any language " + f"{np.mean([bool(r['languages']) for r in res.values()]):.0%}, non-English "
          f"{np.mean([any(_lang(x) not in (None, 'en') for x in r['languages']) for r in res.values()]):.0%}, "
          f"non-Latin script {np.mean([any(s not in ('Latin', 'other') for s in r['scripts']) for r in res.values()]):.0%}, "
          f"driving side {np.mean([r['driving_side'] != 'unknown' for r in res.values()]):.0%}, "
          f"domain {np.mean([bool(r['web_domains']) for r in res.values()]):.0%}")
    rows, kinds = [], {}
    diff = {"countries differ": 0, "language sets differ": 0, "driving side differs": 0}
    for k in set_["key"]:
        ex, r = k["ex"], res.get(k["ex"])
        a, b = (country(c) for c in cands[ex])
        if r is None or a is None or b is None:
            continue
        head, other = (a, b) if k["head"] == "A" else (b, a)
        if head["iso"] != other["iso"]:
            diff["countries differ"] += 1
            diff["language sets differ"] += head["langs"] - {"en"} != other["langs"] - {"en"}
            diff["driving side differs"] += (head["iso"] in LEFT) != (other["iso"] in LEFT)
        cues = [("language", _lang(x)) for x in r["languages"] if _lang(x) not in (None, "en")]
        cues += [("script", s) for s in r["scripts"] if s not in ("Latin", "other")]
        cues += [("driving", r["driving_side"])] if r["driving_side"] != "unknown" else []
        cues += [("domain", "." + d.lower().strip(". ").rsplit(".", 1)[-1]) for d in r["web_domains"]
                 if d.lower().strip(". ").rsplit(".", 1)[-1] not in ("com", "org", "net", "info", "eu", "")]
        vote = 0
        for cue in cues:
            fh, fo = _fits(cue, head), _fits(cue, other)
            if fh != fo:
                vote += 1 if fh else -1
                right = (fh and k["km_head"] < 25) or (fo and k["km_other"] < 25)
                kinds.setdefault(cue[0], []).append(right)
        rows.append((vote, k["km_head"], k["km_other"]))
    print("pairs (of the head's top two areas): " + ", ".join(f"{k} {v}" for k, v in diff.items()))
    for kind, v in kinds.items():
        print(f"  {kind}: decides {len(v)} pairs, picks the right area {np.mean(v):.0%}")
    rows = np.asarray(rows)
    for thr in (25, 1):
        head_ok, other_ok = rows[:, 1] < thr, rows[:, 2] < thr
        sw = rows[:, 0] < 0
        print(f"--- {thr} km: cues favour the head's area on {(rows[:, 0] > 0).sum()}, the other on {sw.sum()}; following the cues: fixes "
              f"{int((sw & other_ok & ~head_ok).sum())}, breaks {int((sw & head_ok & ~other_ok).sum())}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("ask", "report"))
    parser.add_argument("--name", default="4b_cues")
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    ask(args.name) if args.cmd == "ask" else report(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
