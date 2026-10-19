# Perception in the model, knowledge in a table (after lessons 21 and 79: attribute / constraint checks tested as whole-pipeline features). The
# VLM reports what it sees (languages and scripts of visible text, driving side, web domains, currency, phone country codes); GeoNames'
# countryInfo (languages, tld, currency, phone) and a left-driving list say which of a pair's two areas each cue fits. A cue decides a pair
# when it fits exactly one area's country. English, Latin script, generic domains and "$" decide nothing (they appear everywhere).
# Pairs: names202 = lesson 106's 202 dev3k pairs (lesson 132); grpo = the answer head's top two areas per photo from names_grpo pairs
# (artifacts/agent/names_grpo_pairs.jsonl: dev = dev3k, held = 4,693 train photos outside GRPO), chosen by a run name containing "grpo".
# Usage: NAME=4b_cues RUN="cue_check ask" GPU_UTIL=0.3 bash scripts/clue_judge.sh        (NAME=4b_cues_grpo for the grpo pairs)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.cue_check report --name 4b_cues

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .names_judge import OUT, SET, _images

GEONAMES = Path("data/pinpoint/geonames/countryInfo.txt")
GRPO_PAIRS = OUT / "names_grpo_pairs.jsonl"
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
CURRENCY = {"€": {"EUR"}, "£": {"GBP"}, "¥": {"JPY", "CNY"}, "₩": {"KRW"}, "₹": {"INR"}, "฿": {"THB"}, "₽": {"RUB"}, "₺": {"TRY"},
            "zł": {"PLN"}, "kč": {"CZK"}, "₪": {"ILS"}, "₫": {"VND"}, "₱": {"PHP"}, "ft": {"HUF"}, "rm": {"MYR"}, "r$": {"BRL"}}
LEFT = set(("GB IE IM JE GG MT CY IN PK BD LK NP BT MV JP TH MY SG BN ID TL HK MO AU NZ PG FJ SB TO WS KI TV NR ZA NA BW ZW ZM MW MZ TZ KE "
            "UG LS SZ MU SC JM BS BB TT GD LC VC DM AG KN GY SR AI BM KY VG MS TC FK SH CK NU VI").split())
ALIAS = {"Türkiye": "Turkey", "Russian Federation": "Russia", "Vatican City": "Vatican", "Macau": "Macao"}
PROMPT = """Report only what you can actually see in this photo.
languages: the languages of any readable text (signs, shop names, posters, labels), not counting brand names; [] if none.
scripts: the writing systems of any visible text.
driving_side: "left" or "right" only if moving or parked vehicles on a road make it clear which side traffic drives on, else "unknown".
web_domains: country web domains seen in text (".de", ".co.uk"), [] if none.
currency: currency symbols or codes seen with prices ("€", "£", "JPY"), [] if none.
phone_codes: international dialling codes seen in phone numbers ("+44"), [] if none."""
SCHEMA = {"type": "json_schema", "json_schema": {"name": "cues", "schema": {
    "type": "object", "properties": {
        "languages": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "scripts": {"type": "array", "items": {"type": "string", "enum": SCRIPTS}, "maxItems": 4},
        "driving_side": {"type": "string", "enum": ["left", "right", "unknown"]},
        "web_domains": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "currency": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "phone_codes": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["languages", "scripts", "driving_side", "web_domains", "currency", "phone_codes"], "additionalProperties": False}}}


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
        out[f[4]] = {"iso": f[0], "tld": f[9].lower(), "langs": langs, "currency": f[10], "phone": f[12].replace("+", "").split("-")[0],
                     "scripts": {s for s, codes in SCRIPT_OF.items() if langs & set(codes.split())}}
    return out


def _pairs(kind: str) -> list[dict]:
    """[{key, split, countries (head's area first), km (head's first)}]"""

    country = lambda c: c.rsplit(", ", 1)[1]  # noqa: E731
    if kind == "names202":
        set_ = json.loads(SET.read_text(encoding="utf-8"))
        cands = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r"(ex\d+): A = (.*?); B = (.*)", set_["candidates"])}
        out = []
        for k in set_["key"]:
            a, b = cands[k["ex"]]
            head, other = (a, b) if k["head"] == "A" else (b, a)
            out.append({"key": k["ex"], "split": "dev", "countries": (country(head), country(other)), "km": (k["km_head"], k["km_other"])})
        return out
    out = []
    for line in GRPO_PAIRS.open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] in ("dev", "held"):
            out.append({"key": r["image_id"], "split": r["split"], "countries": tuple(country(n) for n in r["names"]), "km": tuple(r["km"])})
    return out


def ask(name: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from .name_score import _post
    from .sft_data import MP16Images

    grpo = "grpo" in name
    keys = [p["key"] for p in _pairs("grpo" if grpo else "names202")]
    if grpo:
        images = MP16Images()

        def url(key):  # MP16 photo by image_id, long side 768 (as names_judge._images)
            im = Image.open(io.BytesIO(images.read(key))).convert("RGB")
            im.thumbnail((768, 768))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    else:
        url = _images(json.loads(SET.read_text(encoding="utf-8"))).__getitem__

    def run(job):
        key, image = job
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 250, "response_format": SCHEMA, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": PROMPT}]}]}
        try:
            return {"key": key, **json.loads(_post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"])}
        except Exception as e:  # noqa: BLE001
            return {"key": key, "error": str(e)[:200]}

    res = []
    with ThreadPoolExecutor(workers) as pool:  # images are read in this thread (the tar reader is not shared), in chunks
        for s in range(0, len(keys), 256):
            res += list(pool.map(run, [(k, url(k)) for k in keys[s:s + 256]]))
            print(f"  {len(res)}/{len(keys)}", flush=True)
    path = OUT / f"cue_check_dev3k_{name}.json"
    path.write_text(json.dumps(res), encoding="utf-8")
    print(f"-> {path}: {len(res)} photos, {sum('error' in r for r in res)} errors")


def _cues(r: dict) -> list[tuple[str, str]]:
    cues = [("language", _lang(x)) for x in r["languages"] if _lang(x) not in (None, "en")]
    cues += [("script", s) for s in r["scripts"] if s not in ("Latin", "other")]
    cues += [("driving", r["driving_side"])] if r["driving_side"] != "unknown" else []
    cues += [("domain", "." + d.lower().strip(". ").rsplit(".", 1)[-1]) for d in r["web_domains"]
             if d.lower().strip(". ").rsplit(".", 1)[-1] not in ("com", "org", "net", "info", "eu", "")]
    for c in r.get("currency", []):
        c = c.strip()
        codes = CURRENCY.get(c.lower()) or ({c.upper()} if re.fullmatch(r"[A-Za-z]{3}", c) and c.upper() != "USD" else None)
        if codes:
            cues.append(("currency", ",".join(sorted(codes))))
    cues += [("phone", re.sub(r"\D", "", p)) for p in r.get("phone_codes", []) if re.sub(r"\D", "", p)]
    return cues


def _fits(cue: tuple[str, str], c: dict) -> bool:
    kind, v = cue
    if kind == "language":
        return v in c["langs"]
    if kind == "script":
        return v in c["scripts"]
    if kind == "driving":
        return (c["iso"] in LEFT) == (v == "left")
    if kind == "currency":
        return c["currency"] in v.split(",")
    if kind == "phone":
        return v == c["phone"]
    return v == c["tld"]  # domain


def report(name: str) -> None:
    pairs = _pairs("grpo" if "grpo" in name else "names202")
    res = {r.get("key", r.get("ex")): r for r in json.loads((OUT / f"cue_check_dev3k_{name}.json").read_text()) if "error" not in r}
    table = _countries()
    country = lambda c: table.get(ALIAS.get(c, c))  # noqa: E731
    print(f"{name}: {len(res)} photos; countries not in GeoNames: {sorted({c for p in pairs for c in p['countries'] if country(c) is None})}")
    share = lambda rs, kind: np.mean([any(k == kind for k, _ in _cues(r)) for r in rs])  # noqa: E731
    for split in sorted({p["split"] for p in pairs}):
        ps = [p for p in pairs if p["split"] == split and p["key"] in res and all(country(c) for c in p["countries"])]
        rs = [res[p["key"]] for p in ps]
        print(f"\n=== {split}: {len(ps)} pairs; photos reporting: non-English text {share(rs, 'language'):.1%}, non-Latin script "
              f"{share(rs, 'script'):.1%}, driving side {share(rs, 'driving'):.1%}, country domain {share(rs, 'domain'):.1%}, currency "
              f"{share(rs, 'currency'):.1%}, phone code {share(rs, 'phone'):.1%}")
        kinds, rows, n_diff, dec_diff = {}, [], 0, 0
        for p, r in zip(ps, rs):
            head, other = (country(c) for c in p["countries"])
            decisive = (p["km"][0] < 25) != (p["km"][1] < 25)
            n_diff += head["iso"] != other["iso"]
            dec_diff += decisive and head["iso"] != other["iso"]
            vote = 0
            for cue in _cues(r):
                fh, fo = _fits(cue, head), _fits(cue, other)
                if fh != fo:
                    vote += 1 if fh else -1
                    if decisive:
                        kinds.setdefault(cue[0], []).append((fh and p["km"][0] < 25) or (fo and p["km"][1] < 25))
            rows.append((vote, p["km"][0], p["km"][1]))
        rows = np.asarray(rows)
        head_ok, other_ok = rows[:, 1] < 25, rows[:, 2] < 25
        dec = head_ok != other_ok
        print(f"  decisive at 25 km {dec.sum()} (head right {head_ok[dec].mean():.1%}); the two areas in different countries: {n_diff} pairs, "
              f"{dec_diff} decisive")
        for kind, v in sorted(kinds.items()):
            print(f"  {kind:9s} decides {len(v):4d} decisive pairs, picks the right area {np.mean(v):.0%}")
        fav_h, fav_o = rows[:, 0] > 0, rows[:, 0] < 0
        print(f"  cues favour the head's area on {fav_h.sum()} pairs ({(fav_h & dec).sum()} decisive, head right on "
              f"{int((fav_h & dec & head_ok).sum())}), the other area on {fav_o.sum()} ({(fav_o & dec).sum()} decisive, other right on "
              f"{int((fav_o & dec & other_ok).sum())})")
        fx, br = int((fav_o & other_ok & ~head_ok).sum()), int((fav_o & head_ok & ~other_ok).sum())
        print(f"  following the cues at 25 km: fixes {fx}, breaks {br}, net {100 * (fx - br) / len(ps):+.2f} pts of these photos")


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
