# OpenStreetMap extract for the offline maps tools (maps_search, features_near; the offline stand-in for Google Maps, see the web-search
# replication plan): every named object plus unnamed landmark features, with a centre, a bounding box and its tags, from the planet file.
# Storage: Parquet parts under data/osm/extract/ (zstd); the node-location cache (~100 GB, dense) goes to the NVMe drive and is deleted at the end.
# Usage: .venvs/osm/bin/python src/geo_search_env/experiment/osm_extract.py data/downloads/osm/planet-latest.osm.pbf data/osm/extract \
#            --cache /opt/dlami/nvme/osm/nodes.cache          (run inside the memory-capped scope; a few hours on the planet)
#        .venvs/osm/bin/python src/geo_search_env/experiment/osm_extract.py --summary data/osm/extract

"""Kept: nodes, ways and multipolygon / boundary relations (as assembled areas) that carry a name-like tag (name, name:*, alt_name, brand, ...)
or one of the FEATURES tags (churches, towers, bridges, peaks, rivers, stations...: what features_near asks about, named or not). Closed ways
are kept as ways (their area copy is skipped). Columns: osm_type (n / w / r), osm_id, lat / lon (bbox centre for ways and areas), lat_min /
lat_max / lon_min / lon_max, name (the name tag, else the first name-like tag, else ''), tags (JSON of all tags minus DROP_PREFIXES).
Runs in the osm env (pyosmium, pyarrow), not the repo env."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import osmium
import pyarrow as pa
import pyarrow.parquet as pq

NAME_KEYS = ("name", "int_name", "official_name", "alt_name", "old_name", "short_name", "loc_name", "brand")
FEATURES = {  # key -> values kept without a name ("*" = any value)
    "historic": "*", "man_made": {"lighthouse", "tower", "water_tower", "windmill", "watermill", "chimney", "pier", "bridge", "obelisk", "cross"},
    "tourism": {"attraction", "viewpoint", "museum", "artwork", "theme_park", "zoo"},
    "amenity": {"place_of_worship", "university", "hospital", "ferry_terminal", "townhall", "fountain", "clock", "marketplace"},
    "natural": {"peak", "volcano", "cliff", "beach", "bay", "glacier", "cave_entrance", "arch", "waterfall"},
    "waterway": {"river", "canal", "waterfall", "dam", "weir", "lock_gate"}, "railway": {"station", "halt", "tram_stop", "funicular"},
    "aeroway": {"aerodrome"}, "leisure": {"stadium", "marina", "golf_course", "water_park"}, "bridge": "*", "tunnel": {"yes"},
    "building": {"cathedral", "church", "mosque", "temple", "synagogue", "castle", "stadium", "train_station"},
    "generator:source": {"wind"}, "aerialway": {"cable_car", "gondola", "chair_lift"},
}
DROP_PREFIXES = ("source", "note", "fixme", "FIXME", "created_by", "tiger:", "gnis:", "NHD:", "nhd:", "lacounty:", "osak:", "massgis:",
                 "yh:", "kms:", "naptan:", "at_bev:", "linz:", "canvec:", "geobase:", "check_date", "survey:", "mapillary", "attribution")
PART_ROWS = 2_000_000
SCHEMA = pa.schema([("osm_type", pa.string()), ("osm_id", pa.int64()), ("lat", pa.float32()), ("lon", pa.float32()),
                    ("lat_min", pa.float32()), ("lat_max", pa.float32()), ("lon_min", pa.float32()), ("lon_max", pa.float32()),
                    ("name", pa.string()), ("tags", pa.string())])


def _name(tags: dict[str, str]) -> str:
    if "name" in tags:
        return tags["name"]
    return next((v for k, v in tags.items() if k.startswith("name:") or k in NAME_KEYS), "")


def _wanted(tags: dict[str, str]) -> bool:
    if any(k in tags for k in NAME_KEYS) or any(k.startswith("name:") for k in tags):
        return True
    return any((v := tags.get(k)) is not None and (allowed == "*" or v in allowed) for k, allowed in FEATURES.items())


class _Parts:
    def __init__(self, out: Path) -> None:
        self.out, self.rows, self.part, self.count = out, {f: [] for f in SCHEMA.names}, 0, {"n": 0, "w": 0, "r": 0}

    def add(self, kind: str, osm_id: int, box: tuple[float, float, float, float], tags: dict[str, str]) -> None:
        lat_min, lat_max, lon_min, lon_max = box
        kept = {k: v for k, v in tags.items() if not k.startswith(DROP_PREFIXES)}
        for f, v in zip(SCHEMA.names, (kind, osm_id, (lat_min + lat_max) / 2, (lon_min + lon_max) / 2, lat_min, lat_max, lon_min, lon_max,
                                       _name(tags), json.dumps(kept, ensure_ascii=False, separators=(",", ":")))):
            self.rows[f].append(v)
        self.count[kind] += 1
        if len(self.rows["osm_id"]) >= PART_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.rows["osm_id"]:
            return
        pq.write_table(pa.table(self.rows, schema=SCHEMA), self.out / f"part-{self.part:05d}.parquet", compression="zstd")
        self.rows, self.part = {f: [] for f in SCHEMA.names}, self.part + 1


def _box(locations) -> tuple[float, float, float, float] | None:
    lats, lons = [], []
    for loc in locations:
        if loc.valid():
            lats.append(loc.lat)
            lons.append(loc.lon)
    return (min(lats), max(lats), min(lons), max(lons)) if lats else None


def extract(pbf: Path, out: Path, cache: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("part-*.parquet")):
        raise SystemExit(f"{out} already holds parts; remove them first")
    storage = "flex_mem"
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        storage = f"dense_file_array,{cache}"
    parts, start, seen = _Parts(out), time.time(), 0
    keys = osmium.filter.KeyFilter(*NAME_KEYS, *FEATURES)  # C++ pre-filter; node locations are still cached for every node
    processor = osmium.FileProcessor(str(pbf)).with_locations(storage).with_areas(osmium.filter.KeyFilter("name")).with_filter(keys)
    try:
        for obj in processor:
            seen += 1
            if seen % 50_000_000 == 0:
                print(f"  {seen / 1e6:.0f}M objects, kept {parts.count}, {(time.time() - start) / 60:.0f} min", flush=True)
            if obj.is_relation() or not obj.tags:
                continue
            tags = dict(obj.tags)
            if not _wanted(tags):
                continue
            if obj.is_node():
                if obj.location.valid():
                    lat, lon = obj.location.lat, obj.location.lon
                    parts.add("n", obj.id, (lat, lat, lon, lon), tags)
            elif obj.is_way():
                if (box := _box(n.location for n in obj.nodes)) is not None:
                    parts.add("w", obj.id, box, tags)
            elif obj.is_area() and not obj.from_way():
                box = _box(n.location for ring in obj.outer_rings() for n in ring)
                if box is not None:
                    parts.add("r", obj.orig_id(), box, tags)
    finally:
        parts.flush()
        if cache is not None and cache.exists():
            os.remove(cache)
    print(f"done: {seen / 1e6:.1f}M objects read, kept {parts.count} in {parts.part} parts, {(time.time() - start) / 60:.0f} min", flush=True)


def summary(out: Path) -> None:
    files = sorted(out.glob("part-*.parquet"))
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    size = sum(f.stat().st_size for f in files)
    print(f"{len(files)} parts, {rows:,} rows, {size / 1e9:.1f} GB")
    t = pq.read_table(files[0], columns=["osm_type", "name", "tags"]).slice(0, 200_000)
    named = sum(bool(n) for n in t["name"].to_pylist())
    print(f"first part: types {dict(zip(*[x.to_pylist() for x in t['osm_type'].value_counts().flatten()]))}, named {named / len(t):.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", type=Path, nargs="?")
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--cache", type=Path, help="dense on-disk node-location cache (needed for the planet)")
    parser.add_argument("--summary", type=Path, metavar="OUT", help="print row counts of an existing extract")
    args = parser.parse_args()
    if args.summary:
        summary(args.summary)
    else:
        extract(args.pbf, args.out, args.cache)


if __name__ == "__main__":
    main()
