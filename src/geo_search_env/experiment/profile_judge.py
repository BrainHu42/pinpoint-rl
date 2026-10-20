# Knowledge in the prompt instead of the weights (after lessons 106, 128, 131: local models choosing between two named areas fail; lessons 28
# and 86 gave in-context text but asked for a choice). Each named area of lesson 106's 202 dev3k pairs gets a profile from offline English
# Wikipedia / Wikivoyage (data/web_search/wiki_en.db): the town's Wikipedia lead, its Wikivoyage lead, the best-known geotagged pages within
# 10 km. The VLM judges each area on its own, "taken within about 25 km of this place? yes / no", P(yes) from the first-token logprobs, with
# and without the profile; the pair margin is logit P(head's area) - logit P(other).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.profile_judge profiles
#        NAME=4b_prof RUN="profile_judge ask" GPU_UTIL=0.3 bash scripts/clue_judge.sh
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.profile_judge report --name 4b_prof

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .names_judge import OUT, SET, _images, margin_report

DB = Path("data/web_search/wiki_en.db")
PROFILES = OUT / "area_profiles_names202.json"
LEAD, VOYAGE, LANDMARKS, NEAR_KM = 700, 600, 10, 10.0

PROMPT = """Was this photo taken within about 25 km of {place}?{profile}
Answer yes or no."""
PROFILE = """
What is known about {place}:
{text}
Compare the photo with this description: landscape and terrain, vegetation, climate, buildings, language and landmarks."""


def _cands() -> dict[str, tuple[str, str]]:
    set_ = json.loads(SET.read_text(encoding="utf-8"))
    return {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r"(ex\d+): A = (.*?); B = (.*)", set_["candidates"])}


def _parse(c: str) -> tuple[str, float, float, str]:
    m = re.match(r"near (.*) \((-?[\d.]+), (-?[\d.]+)\), (.*)", c)
    return m.group(1), float(m.group(2)), float(m.group(3)), m.group(4)


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def profiles() -> None:
    from scipy.spatial import cKDTree

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT rowid, title, lat, lon, coalesce(pop, 0) FROM pages WHERE source = 'wikipedia' AND lat IS NOT NULL").fetchall()
    rid = np.asarray([r[0] for r in rows])
    title = [r[1] for r in rows]
    pop = np.asarray([r[4] for r in rows])
    tree = cKDTree(_xyz(np.asarray([r[2] for r in rows]), np.asarray([r[3] for r in rows])))
    chord = lambda km: 2 * math.sin(km / 6371 / 2)  # noqa: E731

    def body(page: int, chars: int) -> str:
        text = " ".join(zlib.decompress(b).decode("utf-8") for (b,) in con.execute("SELECT body FROM passages WHERE page = ? ORDER BY n LIMIT 3", (page,)))
        return text[:chars].rsplit(" ", 1)[0] + " ..." if len(text) > chars else text

    out = {}
    areas = sorted({c for pair in _cands().values() for c in pair})
    for k, c in enumerate(areas):
        town, lat, lon, country = _parse(c)
        here = _xyz(lat, lon)
        parts = []
        # the town's page: a geotagged page within 30 km titled the town ("Melrose" or "Melrose, Bronx"), exact title first, then nearest
        low = town.lower()
        close = tree.query_ball_point(here, chord(30))
        exact = [i for i in close if title[i].lower() == low]
        match = [int(rid[i]) for i in sorted(exact or [i for i in close if title[i].lower().startswith((low + ",", low + " ("))],
                                             key=lambda i: np.linalg.norm(tree.data[i] - here))]
        if match:
            parts.append("Wikipedia: " + body(match[0], LEAD))
        # Wikivoyage titles carry no coordinates: trust one only if the Wikipedia page of the same title is this place
        voy = con.execute("SELECT rowid FROM pages WHERE source = 'wikivoyage' AND title = ? COLLATE NOCASE", (town,)).fetchone() if exact else None
        if voy:
            parts.append("Wikivoyage: " + body(voy[0], VOYAGE))
        near = tree.query_ball_point(here, chord(NEAR_KM))
        near = [i for i in sorted(near, key=lambda i: -pop[i]) if not match or rid[i] != match[0]][:LANDMARKS]
        if near:
            parts.append("Well-known places within 10 km: " + "; ".join(title[i] for i in near) + ".")
        out[c] = {"town": town, "country": country, "lead": bool(match), "voyage": bool(voy), "landmarks": len(near),
                  "text": "\n".join(parts) or f"{town} is in {country}."}
        if k % 50 == 0:
            print(f"  {k}/{len(areas)} {c}: lead {bool(match)}, voyage {bool(voy)}, {len(near)} landmarks", flush=True)
    PROFILES.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    v = list(out.values())
    print(f"-> {PROFILES}: {len(v)} areas; Wikipedia lead {np.mean([x['lead'] for x in v]):.0%}, Wikivoyage {np.mean([x['voyage'] for x in v]):.0%}, "
          f"landmarks {np.mean([x['landmarks'] > 0 for x in v]):.0%}")


def ask(name: str, server: str = "http://127.0.0.1:8765", workers: int = 16) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from .name_score import _post

    set_ = json.loads(SET.read_text(encoding="utf-8"))
    cands = _cands()
    prof = json.loads(PROFILES.read_text(encoding="utf-8"))
    imgs = _images(set_)

    def run(job):
        ex, side, variant = job
        c = cands[ex]["AB".index(side)]
        town, _, _, country = _parse(c)
        place = f"{town}, {country}"
        text = PROMPT.format(place=place, profile=PROFILE.format(place=place, text=prof[c]["text"]) if variant == "profile" else "")
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": imgs[ex]}}, {"type": "text", "text": text}]}]}
        try:
            top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception as e:  # noqa: BLE001
            return {"ex": ex, "side": side, "variant": variant, "error": str(e)[:200]}
        p = {"yes": 0.0, "no": 0.0}
        for t in top:
            s = t["token"].strip().lower()
            if s in p:
                p[s] += math.exp(t["logprob"])
        return {"ex": ex, "side": side, "variant": variant, "p_yes": p["yes"] / max(p["yes"] + p["no"], 1e-9), "mass": p["yes"] + p["no"]}

    jobs = [(ex, s, v) for ex in cands for s in "AB" for v in ("name", "profile")]
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(run, jobs))
    path = OUT / f"profile_judge_dev3k_{name}.json"
    path.write_text(json.dumps(res), encoding="utf-8")
    print(f"-> {path}: {len(res)} answers, {sum('error' in r for r in res)} errors, mean yes+no mass "
          f"{np.mean([r['mass'] for r in res if 'mass' in r]):.2f}")


def report(name: str) -> None:
    set_ = json.loads(SET.read_text(encoding="utf-8"))
    res = {(r["ex"], r["side"], r["variant"]): r["p_yes"] for r in json.loads((OUT / f"profile_judge_dev3k_{name}.json").read_text()) if "p_yes" in r}
    logit = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))  # noqa: E731
    for variant in ("name", "profile"):
        print(f"{variant}: mean P(yes) {np.mean([v for k, v in res.items() if k[2] == variant]):.2f}")
        rows = []
        for k in set_["key"]:
            ex, other = k["ex"], "B" if k["head"] == "A" else "A"
            if (ex, "A", variant) in res and (ex, "B", variant) in res:
                rows.append((logit(res[ex, k["head"], variant]) - logit(res[ex, other, variant]), k["km_head"], k["km_other"]))
        margin_report(np.asarray(rows), f"{name} {variant}", "logits")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("profiles", "ask", "report"))
    parser.add_argument("--name", default="4b_prof")
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    profiles() if args.cmd == "profiles" else ask(args.name) if args.cmd == "ask" else report(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
