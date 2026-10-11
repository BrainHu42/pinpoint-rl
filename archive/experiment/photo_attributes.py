# The VLM describes each photo with a few coarse attributes (setting, terrain, water, vegetation, weather, text language).
# Usage: .venv/bin/python -m geo_search_env.experiment.photo_attributes run --tag dev [--limit 60]   (vLLM on :8765)

"""One short question per photo, answered as JSON with fixed choices; invalid or missing values are kept as None.

Output: artifacts/query_evidence/attributes/photo_<tag>.json  {image_id: {"raw": ..., "attrs": {key: value or None}}}
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Sequence

import numpy as np

from .query_evidence import ROOT, MP16Images, _chat, _load, _parallel, _photo

CHOICES = {
    "setting": ("city", "town", "village", "rural", "wild"),
    "terrain": ("flat", "hilly", "mountain"),
    "water": ("sea", "lake", "river", "none"),
    "vegetation": ("tropical", "desert", "grass", "temperate", "conifer", "none"),
    "weather": ("snow", "clear", "other"),
}
PROMPT = """Describe this photo for geolocation. Answer in JSON, one value each:
setting: city, town, village, rural or wild
terrain: flat, hilly or mountain
water: sea, lake, river or none
vegetation: tropical, desert, grass, temperate, conifer or none
weather: snow, clear or other
text_language: the language of any visible text, or none
```json
{{"setting": "", "terrain": "", "water": "", "vegetation": "", "weather": "", "text_language": ""}}
```"""


def parse(raw: str) -> dict[str, str | None]:
    """Each attribute's value if it is one of the allowed choices (text_language: any short lowercase word), else None."""

    attrs: dict[str, str | None] = {}
    for key, allowed in CHOICES.items():
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
        value = match.group(1).strip().lower() if match else None
        attrs[key] = value if value in allowed else None
    match = re.search(r'"text_language"\s*:\s*"([^"]*)"', raw)
    value = match.group(1).strip().lower() if match else None
    attrs["text_language"] = value if value and len(value) <= 30 else None
    return attrs


def run(tag: str, server: str, limit: int | None = None) -> None:
    photos = _load(tag, "dev.json")[:limit]
    images = MP16Images()
    raws = _parallel(lambda e: _chat(server, [_photo(images, e), {"type": "text", "text": PROMPT}], 150), photos)
    out = {e["image_id"]: {"raw": raw, "attrs": parse(raw)} for e, raw in zip(photos, raws)}
    (ROOT / "attributes").mkdir(exist_ok=True)
    name = f"photo_{tag}{'_pilot' if limit else ''}.json"
    (ROOT / "attributes" / name).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{tag}: {len(out)} photos -> {name}")
    for key in (*CHOICES, "text_language"):
        values = [v["attrs"][key] for v in out.values()]
        counts: dict[Any, int] = {}
        for x in values:
            counts[x] = counts.get(x, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:7]
        print(f"  {key:14s} invalid/missing {sum(x is None for x in values) / len(values):.0%}; " + ", ".join(f"{k}={n}" for k, n in top))
    print(f"  all five choice fields valid for {np.mean([all(v['attrs'][k] for k in CHOICES) for v in out.values()]):.0%} of photos")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("run",))
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    run(args.tag, args.server, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
