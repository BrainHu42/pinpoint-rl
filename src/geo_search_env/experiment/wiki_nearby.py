# For each candidate of the dev / val photos: the nearest geotagged Wikipedia articles (title + first sentence) within a few km.
# Usage: ~/.venvs/sft/bin/python -m geo_search_env.experiment.wiki_nearby   (PYTHONPATH=src; CPU, ~3 min; needs the Wikipedia index of wiki_backend)

"""Candidate-conditioned text knowledge: `artifacts/query_evidence/<tag>/wiki_nearby.json` maps image_id -> one list per shown candidate (dev.json order)
of up to NEAREST articles within RADIUS_KM, each {"title", "km", "lead"}; `lead` is the article's first sentence cut at 160 characters.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3

import numpy as np

from .query_evidence import ROOT
from .wiki_backend import DB

RADIUS_KM = 3.0
NEAREST = 3
EARTH_KM = 6371.0088


def _xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), axis=-1)


def first_sentence(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", text)).strip()  # drop parentheses (pronunciations, dates)
    cut = re.search(r"[.!?](\s|$)", text)
    sentence = text[: cut.end()].strip() if cut and cut.end() <= limit else text[:limit].rsplit(" ", 1)[0]
    return sentence


def main() -> int:
    from scipy.spatial import cKDTree

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ids, titles, lat, lon = zip(*db.execute("SELECT id, title, lat, lon FROM articles"))
    ids, lat, lon = np.asarray(ids), np.asarray(lat), np.asarray(lon)
    tree = cKDTree(_xyz(lat, lon))
    chord = 2 * math.sin(RADIUS_KM / EARTH_KM / 2)
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        picked: list[list[list[tuple[int, float]]]] = []
        for e in photos:
            per_photo = []
            for _, la, lo in e["candidates"]:
                rows = tree.query_ball_point(_xyz(np.asarray(la), np.asarray(lo)), chord)
                near = sorted(((int(r), float(2 * EARTH_KM * math.asin(np.linalg.norm(tree.data[r] - _xyz(np.asarray(la), np.asarray(lo))) / 2))) for r in rows), key=lambda x: x[1])
                per_photo.append(near[:NEAREST])
            picked.append(per_photo)
        need = sorted({r for p in picked for c in p for r, _ in c})
        leads = {}
        for start in range(0, len(need), 500):
            chunk = need[start : start + 500]
            for rid, lead in db.execute(f"SELECT id, lead FROM articles WHERE id IN ({','.join('?' * len(chunk))})", [int(ids[r]) for r in chunk]):
                leads[int(rid)] = first_sentence(lead)
        out = {e["image_id"]: [[{"title": titles[r], "km": round(km, 2), "lead": leads.get(int(ids[r]), "")} for r, km in c] for c in p] for e, p in zip(photos, picked)}
        (ROOT / tag / "wiki_nearby.json").write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
        covered = np.mean([bool(c) for p in picked for c in p])
        print(f"{tag}: {len(photos)} photos; candidates with at least one article within {RADIUS_KM:g} km: {covered:.0%}; articles per candidate {np.mean([len(c) for p in picked for c in p]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
