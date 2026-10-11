# Do the VLM's photo attributes agree with the map attributes at the photo's true location?
# Usage: .venv/bin/python -m geo_search_env.experiment.attribute_check   (CPU; needs photo_<tag>.json and truth_<tag>.npz)

"""For dev and val (val without the placeholder photos): the mean map attribute at the true location for each value the VLM gave
a photo (e.g. mean distance to the coast for photos it called "sea"), and how often a text language the VLM named is an official
language of the true location's country. A reliable attribute shows a clear separation; a noisy one shows the same mean for every value.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .candidate_attributes import FEATURES
from .photo_attributes import CHOICES
from .query_evidence import ROOT

ATTR = ROOT / "attributes"
LANGUAGES = {
    "english": "en", "french": "fr", "german": "de", "spanish": "es", "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "chinese": "zh", "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi", "thai": "th", "turkish": "tr", "polish": "pl",
    "swedish": "sv", "norwegian": "no", "danish": "da", "finnish": "fi", "czech": "cs", "greek": "el", "hebrew": "he", "indonesian": "id",
    "vietnamese": "vi", "hungarian": "hu", "ukrainian": "uk", "romanian": "ro", "croatian": "hr", "serbian": "sr", "bulgarian": "bg",
    "catalan": "ca", "welsh": "cy", "irish": "ga", "icelandic": "is", "malay": "ms", "persian": "fa", "urdu": "ur", "bengali": "bn",
    "tagalog": "tl", "slovak": "sk", "slovenian": "sl", "lithuanian": "lt", "latvian": "lv", "estonian": "et", "basque": "eu", "galician": "gl",
}
# attribute -> (map feature, how to show it)
SHOW = {"water": ("coast", lambda x: np.expm1(x)), "terrain": ("rugged", lambda x: np.expm1(x)), "weather": ("temp", lambda x: x),
        "vegetation": ("temp", lambda x: x), "setting": ("poi_6km", lambda x: np.expm1(x))}
UNIT = {"coast": "km to coast", "rugged": "m elevation std", "temp": "deg C", "poi_6km": "places within ~6 km"}


def main(argv: Sequence[str] | None = None) -> int:
    languages = json.loads((ATTR / "country_languages.json").read_text(encoding="utf-8"))
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        described = json.loads((ATTR / f"photo_{tag}.json").read_text(encoding="utf-8"))
        saved = np.load(ATTR / f"truth_{tag}.npz")
        num, country = saved["num"][:, 0, :], saved["country"][:, 0]
        excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8"))) if tag == "val" else set()
        keep = [m for m, e in enumerate(photos) if e["image_id"] not in excluded and e["image_id"] in described]
        attrs = [described[photos[m]["image_id"]]["attrs"] for m in keep]
        print(f"\n{tag} ({len(keep)} photos): mean map attribute at the TRUE location, by what the VLM said")
        for key, (feature, show) in SHOW.items():
            col = num[keep, FEATURES.index(feature)]
            line = []
            for value in CHOICES[key]:
                mask = np.asarray([a[key] == value for a in attrs])
                if mask.sum() >= 5:
                    line.append(f"{value} {np.median(show(col[mask])):.1f} (n={int(mask.sum())})")
            print(f"  {key:10s} median {UNIT[feature]}: " + "; ".join(line))
        has_lang = [(i, a["text_language"]) for i, a in enumerate(attrs) if a["text_language"] and a["text_language"] != "none" and a["text_language"] in LANGUAGES]
        match = [LANGUAGES[lang] in languages.get(country[keep[i]], []) for i, lang in has_lang]
        print(f"  text_language: named for {len(has_lang)} photos; an official language of the true location's country for {np.mean(match):.0%}")
        by_lang = {}
        for (i, lang), ok in zip(has_lang, match):
            by_lang.setdefault(lang, []).append(ok)
        print("    " + ", ".join(f"{k} {np.mean(v):.0%} (n={len(v)})" for k, v in sorted(by_lang.items(), key=lambda kv: -len(kv[1]))[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
