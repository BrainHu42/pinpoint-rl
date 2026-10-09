# Number plates as explicit country / region clues (TEXT_SEARCH_PLAN option 3; GEOLOCATION_CLUES.md section 5): does a dev3k photo show a readable
# plate, and does the plate's country / region (as the VLM reads it) beat Pinpoint's where they disagree? dev sets only.
# Usage: RUN="plates read" NAME=9b MODEL=... bash scripts/clue_judge.sh      (one JSON answer per photo from the served VLM)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.plates report --tag dev3k --name 9b

"""read:   per photo, {"plate": bool, "text", "country", "region"} -> artifacts/agent/plates_{tag}_{name}.json.
report: how often a plate is reported (all photos, Pinpoint's mistakes); its country / region right vs Pinpoint's top-1 where they disagree."""

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
PROMPT = """Look for vehicle number plates (licence plates) in this photo, including small or distant ones.
If at least one plate is clearly visible, answer with JSON: {"plate": true, "text": "<characters you can read>", "country": "<country the plate \
is from, judged by its format, colours and characters>", "region": "<state, province or district the plate shows, e.g. a US state name, a German \
district code, a French departement number; null if it doesn't show one>"}.
If no plate is clearly visible, answer {"plate": false}. Answer with the JSON only."""


def read(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from .name_score import _post
    from .sft_data import MP16Images

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    images = MP16Images()

    def ask(e: dict) -> dict:
        image = base64.b64encode(images.read(e["image_id"])).decode()
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
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
    path = OUT / f"plates_{tag}_{name}.json"
    path.write_text(json.dumps({e["image_id"]: a for e, a in zip(photos, answers)}) + "\n", encoding="utf-8")
    print(f"{tag} / {name}: {sum(bool(a.get('plate')) for a in answers)} of {len(photos)} photos with a plate -> {path}")


def report(tag: str, name: str) -> None:
    from .agent_targets import same_region
    from .pinpoint_constrained import LABELS_NPZ
    from .strategy_search import _haversine_km
    from .vlm_country import _country

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    answers = json.loads((OUT / f"plates_{tag}_{name}.json").read_text(encoding="utf-8"))
    known = set(np.load(LABELS_NPZ)["country"].tolist()) - {""}
    wrong = {e["image_id"]: _haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] >= 25 for e in photos}
    with_plate = [e["image_id"] for e in photos if answers.get(e["image_id"], {}).get("plate") is True]
    print(f"{tag} / {name}: plate reported on {len(with_plate)} of {len(photos)} photos ({len(with_plate) / len(photos):.1%}); "
          f"on Pinpoint's mistakes {sum(wrong[k] for k in with_plate)} of {sum(wrong.values())}")
    c_n = c_right = c_pin = c_diff = c_diff_v = c_diff_p = 0
    r_n = r_right = r_diff = r_diff_v = r_diff_p = 0
    for k in with_plate:
        a, lab = answers[k], labels[k]
        truth, pin = lab["truth"], (lab["pool"][0] if lab["pool"] else {})
        country = _country(a.get("country"), known)
        if truth.get("country") and country:
            c_n += 1
            v, p = country == truth["country"], pin.get("country") == truth["country"]
            c_right += v
            c_pin += p
            if country != pin.get("country"):
                c_diff += 1
                c_diff_v += v
                c_diff_p += p
        region = a.get("region")
        if truth.get("region") and country and region and str(region).lower() not in ("null", "none"):
            r_n += 1
            guess = {"country": country, "region": str(region)}
            v, p = same_region(guess, truth), same_region(pin, truth)
            r_right += v
            if not same_region(guess, pin):
                r_diff += 1
                r_diff_v += v
                r_diff_p += p
    print(f"  plate country given on {c_n}: right {c_right} (Pinpoint {c_pin}); differs from Pinpoint's on {c_diff}: plate right {c_diff_v}, "
          f"Pinpoint right {c_diff_p}")
    print(f"  plate region given on {r_n}: right {r_right}; differs from Pinpoint's on {r_diff}: plate right {r_diff_v}, Pinpoint right {r_diff_p}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("read", "report"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="9b")
    args = parser.parse_args(argv)
    read(args.tag, args.name) if args.cmd == "read" else report(args.tag, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
