"""Score geolocation predictions on commons26. Needs only numpy (geographiclib, if installed, gives WGS84 distances; else haversine).

Usage: python score.py --predictions P.csv [--split dev|test|all] [--benchmark benchmark.csv]
P.csv has columns IMG_ID, LAT, LON. Headline: % within 1 / 25 / 200 / 750 / 2500 km with the six continents weighted equally (the set is ~31%
Europe and ~7% Africa), then the plain mean, each continent, and the 27B's locatability tiers as slices only. Standard errors are clustered by
uploader (`group`). This file is shipped with the dataset as score.py and used by commons_bench.score.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

THRESHOLDS_KM = (1, 25, 200, 750, 2500)
GEOGUESSR_DECAY_KM = 1492.7


def geodesic_km(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    try:
        from geographiclib.geodesic import Geodesic
    except ImportError:
        a, b = np.radians(predictions), np.radians(labels)
        h = np.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return np.asarray([Geodesic.WGS84.Inverse(p[0], p[1], l[0], l[1])["s12"] / 1000.0 for p, l in zip(predictions, labels)])


def score(predictions: Path, split: str = "test", benchmark: Path = Path(__file__).with_name("benchmark.csv")) -> None:
    # csv, not pandas: pandas reads the continent code "NA" (North America) as missing
    rows = [r for r in csv.DictReader(benchmark.open(encoding="utf-8")) if split == "all" or r["split"] == split]
    pred = {r["IMG_ID"]: (float(r["LAT"]), float(r["LON"])) for r in csv.DictReader(predictions.open(encoding="utf-8"))}
    missing = sum(r["IMG_ID"] not in pred for r in rows)
    if missing:
        print(f"warning: {missing} photos have no prediction; scored as (0, 0)")
    km = geodesic_km(np.array([pred.get(r["IMG_ID"], (0.0, 0.0)) for r in rows]), np.array([[float(r["LAT"]), float(r["LON"])] for r in rows]))
    groups = np.array([r["group"] for r in rows])
    tier = np.array([r["tier"] for r in rows])
    continent = np.array([r["continent"] for r in rows])
    continents = sorted(set(continent))
    balanced = np.array([1 / (len(continents) * (continent == c).sum()) for c in continent])

    def line(mask: np.ndarray, weight: np.ndarray | None = None) -> str:
        """Weighted mean (weights normalised within each continent stratum) with an uploader-clustered SE: each photo adds w_i (y_i - stratum mean)."""
        w = np.where(mask, 1.0, 0.0) if weight is None else np.where(mask, weight, 0.0)
        w = w / w.sum()
        strata = continent if weight is not None else np.zeros(len(rows), dtype=str)
        cells = []
        for t in THRESHOLDS_KM:
            y = (km < t).astype(float)
            centred = y.copy()
            for s in set(strata[mask]):
                m = mask & (strata == s)
                centred[m] -= np.average(y[m], weights=w[m])
            sums: dict[str, float] = {}
            for v, g in zip(w * centred, groups):
                sums[g] = sums.get(g, 0.0) + v
            cells.append(f"{100 * (w * y).sum():5.1f}±{100 * np.sqrt(sum(v * v for v in sums.values())):3.1f}")
        geo = (w * np.round(5000 * np.exp(-km / GEOGUESSR_DECAY_KM))).sum()
        return f"{' '.join(cells)}  {geo:6.0f}  {np.median(km[mask]):8.1f}  n={mask.sum()}"

    everything = np.ones(len(rows), bool)
    print(f"{'':18s} {'  '.join(f'<{t} km'.rjust(8) for t in THRESHOLDS_KM)}  GeoScore  median km")
    print(f"{'continent-balanced':18s} {line(everything, balanced)}")
    print(f"{'all photos':18s} {line(everything)}")
    for c in continents:
        print(f"{'continent ' + c:18s} {line(continent == c)}")
    for t in ("landmark", "city", "region", "none"):
        print(f"{'27B tier ' + t:18s} {line(tier == t)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with IMG_ID, LAT, LON")
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).with_name("benchmark.csv"))
    args = parser.parse_args()
    score(args.predictions, args.split, args.benchmark)
