# Survey of clue types the text pipeline doesn't use yet (GEOLOCATION_CLUES.md; TEXT_SEARCH_PLAN.md "missing clue types"): on Pinpoint's dev3k
# mistakes, how often does the 4B see regional flags, regional or minority-language / bilingual signs, local-authority names, road numbers or
# kilometre markers, distance signs, postcodes, area codes, currency or units, and is the place each one implies right (vs Pinpoint)? dev sets only.
# Usage: RUN="clue_survey read" NAME=4b bash scripts/clue_judge.sh       (serves the base 4B; one JSON answer per mistake photo)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_survey report --tag dev3k

"""read:   per photo where Pinpoint's top-1 is >= 25 km off, a JSON object with one entry per clue type: what is seen (or null) and the country /
        region it implies -> artifacts/agent/clue_survey_{tag}_{name}.json.
report: per clue type, photos where it is reported; the implied country / region right; where it differs from Pinpoint's, right vs Pinpoint."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
TYPES = ("flag", "regional_language", "local_authority", "road_number", "distance_sign", "postcode", "phone_area_code", "currency_or_units",
         "regional_chain")
PROMPT = """You are a professional geolocator. Look carefully at this photo, including small and distant details, for these specific clues:
- flag: a national, regional or state flag (e.g. Catalonia, Texas, Bavaria, Scotland)
- regional_language: a sign in a regional or minority language, or a bilingual sign (e.g. Catalan, Basque, Welsh, French in Quebec, Frisian)
- local_authority: the name of a town, city or council on a sign, vehicle, bin or plaque (e.g. "Comune di Lucca", "City of Westminster")
- road_number: a road number, motorway shield or kilometre marker (e.g. "A-7", "I-95", "N-340 km 214")
- distance_sign: a sign giving distances to towns (e.g. "Toledo 12, Madrid 70")
- postcode: a postal code on a sign, shop or vehicle
- phone_area_code: a phone number with an area code
- currency_or_units: prices with a currency, or units like mph or km/h
- regional_chain: a shop, supermarket, fuel station or bank chain found only in some countries or regions (e.g. Wawa, Mercadona, Biedronka)
Answer with JSON only: an object with these keys. For each clue you can actually see, give {"seen": "what it says or shows", "country": "the
country it implies or null", "region": "the state, province or region it implies or null"}; use null for clues that are not visible."""


def read(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from .name_score import _post
    from .sft_data import MP16Images
    from .strategy_search import _haversine_km

    photos = [e for e in json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
              if _haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] >= 25]
    images = MP16Images()

    def ask(e: dict) -> dict:
        image = base64.b64encode(images.read(e["image_id"])).decode()
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 500, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
                                                          {"type": "text", "text": PROMPT}]}]}
        try:
            raw = _post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group(0)) if m else {"raw": raw}
        except Exception as error:  # noqa: BLE001
            return {"error": str(error)}

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(ask, photos))
    path = OUT / f"clue_survey_{tag}_{name}.json"
    path.write_text(json.dumps({e["image_id"]: a for e, a in zip(photos, answers)}) + "\n", encoding="utf-8")
    print(f"{tag} / {name}: {len(photos)} mistake photos, {sum('error' in a or 'raw' in a for a in answers)} unparsed -> {path}")


def report(tag: str, name: str = "4b") -> None:
    from .agent_targets import same_region
    from .knowledge_check import _right

    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    answers = json.loads((OUT / f"clue_survey_{tag}_{name}.json").read_text(encoding="utf-8"))
    print(f"{tag} / {name}: {len(answers)} of Pinpoint's mistakes surveyed")
    print(f"  {'clue type':20s} {'reported':>8s} | country given: right / Pinpoint's right / differs: clue right | region given: right / differs & clue right & Pinpoint wrong")
    for t in TYPES:
        seen = c_n = c_right = c_pin = c_diff_right = r_n = r_right = r_fix = 0
        for k, a in answers.items():
            v = a.get(t) if isinstance(a, dict) else None
            if not isinstance(v, dict) or not v.get("seen") or str(v.get("seen")).lower() in ("null", "none", ""):
                continue
            seen += 1
            truth, pin = labels[k]["truth"], (labels[k]["pool"][0] if labels[k]["pool"] else {})
            country, region = v.get("country"), v.get("region")
            if country and truth.get("country"):
                c_n += 1
                right = bool(_right(str(country), truth["country"]))
                c_right += right
                c_pin += bool(_right(pin.get("country") or "", truth["country"]))
                c_diff_right += right and not _right(str(country), pin.get("country") or "")
            if country and region and truth.get("region"):
                r_n += 1
                guess = {"country": str(country), "region": str(region)}
                right = same_region(guess, truth)
                r_right += right
                r_fix += right and not same_region(pin, truth)
        print(f"  {t:20s} {seen:8d} | {c_n:4d}: {c_right:4d} / {c_pin:4d} / {c_diff_right:4d} | {r_n:4d}: {r_right:4d} / {r_fix:4d}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("read", "report"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="4b")
    args = parser.parse_args(argv)
    read(args.tag, args.name) if args.cmd == "read" else report(args.tag, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
