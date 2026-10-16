# Elimination evidence per hypothesis (lesson 85): does what the 4B read agree with the hypothesis's region / country, or contradict it? Per
# hypothesis of the region-hypothesis pipeline (HYP_VARIANT=regionhyps), its area label (country, region of the nearest index photo to its
# centre) is checked against the photo's clues (region_evidence rows: text_search over places + Flickr, phone / domain country hints) and
# against tools.maps lookups of the read names (all top-10 matches, not only the best cluster). Soft features for the answer head, not filters.
# Groups: "text" (text_search clue regions and hints), "maps" (maps name search). -> artifacts/agent/regioncheck_{tag}_{ckpt}.npz in the
# hypothesis_agent extra-columns format (photo, source, groups, x).
# Usage: PINPOINT_RERANKER_CKPT=... HYP_VARIANT=regionhyps PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.region_checks --tag dev3k

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
CKPT = "setft_a"
MAPS_TOP = 10
TEXT = ("read_region", "read_country", "read_contra", "recall_region", "recall_country", "recall_contra", "hint_match", "hint_contra", "has_read")
MAPS = ("maps_region", "maps_country", "maps_contra", "maps_n")


def run(tag: str) -> None:
    from ..tools.maps import MapsSearch
    from ..tools.pinpoint import Pinpoint
    from .area_model import _paths
    from .region_evidence import evidence_path
    from .text_search import norm

    feat_path, zoom_path = _paths(tag, CKPT)
    f = np.load(feat_path)
    photo, source = f["photo"], f["source"]
    center = np.load(zoom_path)["center"]
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT}" / "photos.json").read_text(encoding="utf-8"))
    position = {e["image_id"]: i for i, e in enumerate(photos)}
    tool = Pinpoint(targets=None)
    hyp_c, hyp_r = tool.labels_at(center)
    clues: dict[int, list[dict]] = {}
    for r in json.loads(evidence_path(f"{tag}_{CKPT}", "4b").read_text(encoding="utf-8")):
        if "region" in r and r["image_id"] in position:
            clues.setdefault(position[r["image_id"]], []).append(r)

    # maps: every read name's top matches, labelled by the nearest index photo
    names = sorted({r["query"] for lst in clues.values() for r in lst if not r["source"].startswith("query") and r["kind"] == "name"})
    ms = MapsSearch()

    def look(q):
        try:
            return np.asarray([[p.lat, p.lon] for p in ms.query(q, top=MAPS_TOP)], np.float64).reshape(-1, 2)
        except Exception:  # noqa: BLE001
            return np.zeros((0, 2))

    print(f"{tag}: {len(photo)} hypotheses, {len(names)} read names to look up", flush=True)
    with ThreadPoolExecutor(16) as pool:
        pts = list(pool.map(look, names))
    flat = np.concatenate([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((0, 2))
    fc, fr = tool.labels_at(flat) if len(flat) else (np.zeros(0, object), np.zeros(0, object))
    found, at = {}, 0
    for q, p in zip(names, pts):
        found[q] = (fc[at:at + len(p)], fr[at:at + len(p)])
        at += len(p)

    x = np.zeros((len(photo), len(TEXT) + len(MAPS)), np.float32)
    for i, (ph, c, reg) in enumerate(zip(photo, hyp_c, hyp_r)):
        lst = clues.get(int(ph), [])
        for r in lst:
            rc, rr = r["region"][0] or "", f"{r['region'][0]}|{r['region'][1]}" if r["region"][0] and r["region"][1] else ""
            kind = "recall" if r["source"].startswith("query") else "read"
            o = TEXT.index(f"{kind}_region")
            if reg and rr == reg:
                x[i, o] = max(x[i, o], r["share"])
            if c and rc == c:
                x[i, o + 1] = max(x[i, o + 1], r["share"])
            elif c and rc and r["share"] >= 0.5:
                x[i, o + 2] = max(x[i, o + 2], r["share"])
            if r["hint"]:
                hinted = norm(r["hint"].split("=", 1)[-1].rsplit("(", 1)[0])
                x[i, TEXT.index("hint_match" if hinted == norm(c) else "hint_contra")] = 1.0
            if kind == "read":
                x[i, TEXT.index("has_read")] = 1.0
                if r["kind"] == "name" and r["query"] in found:
                    mc, mr = found[r["query"]]
                    if len(mc):
                        o = len(TEXT)
                        x[i, o] = max(x[i, o], float(np.mean(mr == reg)) if reg else 0.0)
                        x[i, o + 1] = max(x[i, o + 1], float(np.mean(mc == c)) if c else 0.0)
                        x[i, o + 2] = max(x[i, o + 2], float(mc[0] != c) if c else 0.0)
                        x[i, o + 3] += 1
    groups = np.asarray(["text"] * len(TEXT) + ["maps"] * len(MAPS))
    dst = OUT / f"regioncheck_{tag}_{CKPT}.npz"
    np.savez(dst, photo=photo, source=source, x=x, groups=groups, names=np.asarray(TEXT + MAPS))
    for j, n in enumerate(TEXT + MAPS):
        print(f"  {n:15s} non-zero on {np.mean(x[:, j] > 0):.1%} of hypotheses")
    print(f"-> {dst}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="dev3k or train (the retrained-reranker sets)")
    args = parser.parse_args(argv)
    run(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
