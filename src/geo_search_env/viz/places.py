"""Readable labels for coordinates: the nearest GeoNames populated place (population >= 1,000) and its country, cached once from
allCountries.txt -> artifacts/viz/cities.npz."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

GEONAMES = Path("data/pinpoint/geonames")
CITIES = Path("artifacts/viz/cities.npz")


def _build() -> None:
    countries = {}
    for line in (GEONAMES / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            p = line.split("\t")
            countries[p[0]] = p[4]
    lat, lon, name, cc = [], [], [], []
    with (GEONAMES / "allCountries.txt").open(encoding="utf-8") as stream:
        for line in stream:
            p = line.split("\t")
            if p[6] != "P" or not p[14] or int(p[14]) < 1000:
                continue
            lat.append(float(p[4]))
            lon.append(float(p[5]))
            name.append(p[1])
            cc.append(p[8])
    CITIES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CITIES, lat=np.asarray(lat), lon=np.asarray(lon), name=np.asarray(name), country=np.asarray([countries.get(c, c) for c in cc]))


@lru_cache(maxsize=1)
def _tree():
    from scipy.spatial import cKDTree

    if not CITIES.exists():
        _build()
    z = np.load(CITIES)
    xyz = _xyz(z["lat"], z["lon"])
    return cKDTree(xyz), z["name"], z["country"]


def _xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def label(lat: float, lon: float) -> str:
    """'Town, Country' of the nearest place; 'near Town, Country' beyond 30 km; '' if the coordinate is invalid."""

    if not (math.isfinite(lat) and math.isfinite(lon)):
        return ""
    tree, name, country = _tree()
    dist, i = tree.query(_xyz(lat, lon))
    km = 2 * 6371.0 * math.asin(min(1.0, dist / 2))
    return f"{'near ' if km > 30 else ''}{name[i]}, {country[i]}"


def labels(latlon) -> list[str]:
    latlon = np.asarray(latlon, np.float64).reshape(-1, 2)
    if not len(latlon):
        return []
    tree, name, country = _tree()
    ok = np.isfinite(latlon).all(1)
    dist, idx = tree.query(_xyz(np.where(ok, latlon[:, 0], 0), np.where(ok, latlon[:, 1], 0)))
    km = 2 * 6371.0 * np.arcsin(np.minimum(1.0, dist / 2))
    return [f"{'near ' if k > 30 else ''}{name[i]}, {country[i]}" if o else "" for k, i, o in zip(km, idx, ok)]
