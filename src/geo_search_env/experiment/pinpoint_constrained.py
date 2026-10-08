# Pinpoint as a constrained tool (TEXT_SEARCH_PLAN.md, target design): rerun Pinpoint's retrieval + reranker restricted to one area and see what it
# fixes. Ceilings restrict each photo to its true country / region; the realistic arms restrict it to the country / region of the text search tool's
# blind pick (text_search report) or of another picks file (--picks, e.g. clue_judge picks), photos without a pick keeping Pinpoint's answer.
# dev sets only.
# Usage: PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.pinpoint_constrained labels   (once; parquet -> npz)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pinpoint_constrained run --tag dev3k      (GPU, one pass over the index)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pinpoint_constrained report --tag dev3k

"""run:    every photo of the set searched once per arm (all arms in one index pass): Pinpoint unrestricted (must equal the saved pool's top-1),
        restricted to the true country, to the true (country, region), to the pick's country and to the pick's region. Areas are Overture division
        labels (place_labels: every MP16 photo in artifacts/place_labels/mp16.parquet, truths in the set's labels.json); a pick's area is the label of
        the nearest index photo. -> artifacts/agent/constrained_{tag}.npz (top-1 lat / lon per arm and photo, restriction applied or not).
report: per arm, top-1 < 1 / 25 km and fixes / breaks vs Pinpoint (bootstrap interval), and for the pick arms the same against answering with
        the pick's own coordinate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")  # photo_sets.PHOTO_SETS (package imports stay inside run / report: `labels` runs in the duckdb env)
LABELS = Path("artifacts/place_labels/mp16.parquet")
LABELS_NPZ = Path("artifacts/place_labels/mp16_country_region.npz")
SFT_ROOT = Path("artifacts/sft")
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")  # a copy of the index on the NVMe drive, if staged
ARMS = ("pinpoint", "true country", "true region", "pick country", "pick region")


def labels() -> None:
    """The MP16 labels' country / region as arrays (np.load-able, so `run` works in the repo env, which has no parquet reader)."""

    import duckdb

    row, country, region = map(np.asarray, zip(*duckdb.connect().execute(
        f"SELECT row, coalesce(country, ''), coalesce(region, '') FROM '{LABELS}'").fetchall()))
    np.savez(LABELS_NPZ, row=row.astype(np.int64), country=country.astype(str), region=region.astype(str))
    print(f"{len(row)} labelled MP16 photos -> {LABELS_NPZ}")


def _index_labels(cache_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(country, 'country|region') strings per index row from the MP16 labels ('' if unlabelled)."""

    z = np.load(LABELS_NPZ)
    row = z["row"]
    country = np.full(row.max() + 1, "", dtype=object)
    region = np.full(row.max() + 1, "", dtype=object)
    country[row] = z["country"]
    region[row] = [f"{c}|{r}" if c and r else "" for c, r in zip(z["country"], z["region"])]
    return country[cache_rows], region[cache_rows]


def run(tag: str, picks_name: str = "") -> None:
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import PinpointReranker, _unit_xyz
    from .strategy_search import load_world

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    truth_labels = {p["image_id"]: p["truth"] for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    suffix = f"_{picks_name}" if picks_name else ""
    picks = json.loads((OUT / f"text_search_picks_{tag}{suffix}.json").read_text(encoding="utf-8"))
    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    world = load_world(mp16_queries=[queries[e["index"]] for e in photos])
    reranker = PinpointReranker(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    print("labelling index rows", flush=True)
    country, region = _index_labels(reranker.cache_indices)
    vocab: dict[str, int] = {}
    ids = lambda labels: np.asarray([vocab.setdefault(x, len(vocab)) if x else -1 for x in labels], np.int64)
    index_country, index_region = ids(country), ids(region)
    # one id space for both levels: a country id never equals a region id (the strings differ), so one index array per level is enough
    tree = cKDTree(_unit_xyz(np.asarray(reranker.latlon)))

    def pick_labels(e: dict) -> tuple[str, str]:
        p = picks.get(e["image_id"])
        if p is None:
            return "", ""
        j = int(tree.query(_unit_xyz(np.asarray([[p["lat"], p["lon"]]])))[1][0])
        return country[j], region[j]

    n = len(photos)
    arm_country = {
        "pinpoint": ["" for _ in photos],
        "true country": [truth_labels[e["image_id"]].get("country") or "" for e in photos],
        "pick country": [pick_labels(e)[0] for e in photos],
    }
    arm_region = {
        "true region": [f"{t.get('country')}|{t.get('region')}" if t.get("country") and t.get("region") else "" for t in (truth_labels[e["image_id"]] for e in photos)],
        "pick region": [pick_labels(e)[1] for e in photos],
    }
    # each arm restricts on one level; run both levels' arms as one stacked query batch, with the matching index ids per level
    found_latlon = np.full((len(ARMS), n, 2), np.nan)
    restricted = np.zeros((len(ARMS), n), bool)
    for level, arms, index_ids in (("country", arm_country, index_country), ("region", arm_region, index_region)):
        names = list(arms)
        q = np.concatenate([ids(arms[a]) for a in names])
        found = reranker.rank(
            np.tile(world.query_embeddings, (len(names), 1)), query_author=np.tile(world.query_author, len(names)),
            index_author=world.mp16["author"][reranker.cache_indices], query_region=q, index_region=index_ids,
        )
        for k, a in enumerate(names):
            i = ARMS.index(a)
            found_latlon[i] = found.top1[k * n:(k + 1) * n]
            restricted[i] = q[k * n:(k + 1) * n] >= 0
        print(f"{level}: {names} done", flush=True)
    np.savez(OUT / f"constrained_{tag}{suffix}.npz", arms=np.asarray(ARMS), latlon=found_latlon, restricted=restricted,
             truth=np.asarray([e["truth"] for e in photos]), pinpoint=np.asarray([e["pool"][0] for e in photos]),
             pick=np.asarray([[picks[e["image_id"]]["lat"], picks[e["image_id"]]["lon"]] if e["image_id"] in picks else [np.nan, np.nan] for e in photos]))
    print(f"-> {OUT / f'constrained_{tag}{suffix}.npz'}")
    report(tag, picks_name)


def report(tag: str, picks_name: str = "") -> None:
    from .strategy_search import _bootstrap, _haversine_km

    z = np.load(OUT / f"constrained_{tag}{'_' + picks_name if picks_name else ''}.npz", allow_pickle=True)
    truth, base_ll = z["truth"], z["pinpoint"]
    km = lambda ll: np.asarray([_haversine_km(*t, np.asarray([p]))[0] if np.isfinite(p).all() else np.inf for t, p in zip(truth, ll)])
    base = km(base_ll)
    n = len(base)
    print(f"{tag}: {n} photos; Pinpoint (saved pool top-1) < 1 km {(base < 1).mean():.1%}, < 25 km {(base < 25).mean():.1%}")

    def line(label: str, d: np.ndarray, mask: np.ndarray) -> None:
        c = _bootstrap((d < 25) * 1.0 - (base < 25))
        fixes, breaks = ((d < 25) & (base >= 25)).sum(), ((d >= 25) & (base < 25)).sum()
        print(f"  {label:44s} applied {mask.sum():4d}: < 1 km {(d < 1).mean():5.1%}  < 25 km {(d < 25).mean():5.1%} "
              f"({100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}])  fixes {fixes:4d}  breaks {breaks:4d}")

    for i, arm in enumerate(z["arms"]):
        mask = z["restricted"][i]
        d = np.where(mask, km(z["latlon"][i]), base)
        line(f"{arm} (others keep Pinpoint)", d, mask)
        if arm == "pinpoint":
            same = (km(z["latlon"][i]) - base == 0).mean()
            print(f"    sanity: unrestricted rerun equals the saved top-1 on {same:.1%} of photos")
    pick = km(z["pick"])
    has = np.isfinite(pick)
    line("text pick's own coordinate", np.where(has, pick, base), has)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("labels", "run", "report"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--picks", default="", help="picks file suffix: text_search_picks_{tag}_{picks}.json (e.g. clue_judge picks)")
    args = parser.parse_args(argv)
    {"labels": lambda: labels(), "run": lambda: run(args.tag, args.picks), "report": lambda: report(args.tag, args.picks)}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
