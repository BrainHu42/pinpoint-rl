# OpenStreetMap extract for the offline maps tools (maps_search, features_near; the offline stand-in for Google Maps, see the web-search
# replication plan): every named object plus unnamed landmark features, with a centre, a bounding box and its tags, from the planet file.
# Storage: Parquet parts under data/osm/extract/ (zstd); the node-location cache (~115 GB, dense) goes to the NVMe drive.
# Usage: .venvs/osm/bin/python src/geo_search_env/experiment/osm_extract.py data/downloads/osm/planet-latest.osm.pbf data/osm/extract \
#            --cache /opt/dlami/nvme/osm/nodes.cache [--skip-nodes]   (memory-capped; --skip-nodes reuses the node rows and a complete cache)
#        .venvs/osm/bin/python src/geo_search_env/experiment/osm_extract.py --summary data/osm/extract

"""Kept: nodes, ways and multipolygon / boundary relations that carry a name-like tag (name, name:*, alt_name, brand, ...) or one of the
FEATURES tags (churches, towers, bridges, peaks, rivers, stations...: what features_near asks about, named or not). Columns: osm_type
(n / w / r), osm_id, lat / lon (bbox centre for ways and relations), lat_min / lat_max / lon_min / lon_max, name (the name tag, else the
first name-like tag, else ''), tags (JSON of all tags minus DROP_PREFIXES). Runs in the osm env (pyosmium, pyarrow, numpy).

Passes: (1) nodes: libosmium stores every node location in the dense cache, Python sees only the kept nodes; (2) relations: kept
multipolygon / boundary relations and their non-inner member ways; (3) kept ways and (4) member ways: node ids collected in Python (at
most MAX_REFS per way, evenly spaced), their locations read from the cache with numpy in sorted batches. Letting libosmium add locations
instead looks up every node of all ~1.1B planet ways (mostly unnamed buildings) and crawled at ~12 MB/min once the cache outgrew RAM.
A relation's box is the union of its member ways' boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import osmium
import pyarrow as pa
import pyarrow.compute as pc
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
AREA_TYPES = {"multipolygon", "boundary"}
MAX_REFS = 64  # node locations read per way (evenly spaced, ends included); enough for a bounding box
BATCH_WAYS = 1_000_000
PART_ROWS = 2_000_000
UNDEFINED = np.iinfo(np.int32).max  # libosmium's undefined Location coordinate
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


def _key_filter():
    return osmium.filter.KeyFilter(*NAME_KEYS, *FEATURES)  # C++ pre-filter; _wanted makes the exact decision


class _Parts:
    def __init__(self, out: Path, first: int = 0) -> None:
        self.out, self.rows, self.part, self.count = out, {f: [] for f in SCHEMA.names}, first, {"n": 0, "w": 0, "r": 0}

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


def _refs(way) -> list[int]:
    nodes = way.nodes
    n = len(nodes)
    if n <= MAX_REFS:
        return [r.ref for r in nodes]
    return [nodes[i].ref for i in np.linspace(0, n - 1, MAX_REFS).astype(int)]


def _boxes(cache: np.memmap, refs: list[list[int]]) -> np.ndarray:
    """[ways, 4] lat_min, lat_max, lon_min, lon_max (NaN when no node has a location), reading the cache in id order."""
    counts = np.fromiter((len(r) for r in refs), dtype=np.int64, count=len(refs))
    ids = np.fromiter((i for r in refs for i in r), dtype=np.int64, count=int(counts.sum()))
    order = np.argsort(ids, kind="stable")
    xy = np.empty((len(ids), 2), dtype=np.int32)
    inside = ids[order] < len(cache)
    xy[order[inside]] = cache[ids[order][inside]]
    xy[order[~inside]] = UNDEFINED
    lon, lat = xy[:, 0].astype(np.float64) / 1e7, xy[:, 1].astype(np.float64) / 1e7
    bad = xy[:, 0] == UNDEFINED
    lon[bad], lat[bad] = np.nan, np.nan
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.full((len(refs), 4), np.nan)
    nonempty = counts > 0
    with np.errstate(invalid="ignore"):
        for col, (values, reduce) in enumerate(((lat, np.fmin), (lat, np.fmax), (lon, np.fmin), (lon, np.fmax))):
            out[nonempty, col] = reduce.reduceat(values, starts[nonempty])
    return out


def _nodes(pbf: Path, parts: _Parts, cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    processor = osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_locations(f"dense_file_array,{cache}").with_filter(_key_filter())
    for node in processor:
        tags = dict(node.tags)
        if _wanted(tags) and node.location.valid():
            lat, lon = node.location.lat, node.location.lon
            parts.add("n", node.id, (lat, lat, lon, lon), tags)


def _relations(pbf: Path) -> tuple[list[tuple[int, dict[str, str]]], np.ndarray, np.ndarray]:
    """Kept area relations, and (relation index, member way id) pairs for their non-inner way members."""
    relations, rel_index, way_ids = [], [], []
    for rel in osmium.FileProcessor(str(pbf), osmium.osm.RELATION).with_filter(_key_filter()):
        tags = dict(rel.tags)
        if tags.get("type") not in AREA_TYPES or not _wanted(tags):
            continue
        members = [m.ref for m in rel.members if m.type == "w" and m.role != "inner"]
        if members:
            rel_index.extend([len(relations)] * len(members))
            way_ids.extend(members)
            relations.append((rel.id, tags))
    return relations, np.asarray(rel_index, dtype=np.int64), np.asarray(way_ids, dtype=np.int64)


def _ways(pbf: Path, parts: _Parts, cache: np.memmap) -> None:
    batch: list[tuple[int, dict[str, str]]] = []
    refs: list[list[int]] = []

    def flush() -> None:
        for (way_id, tags), box in zip(batch, _boxes(cache, refs)):
            if not np.isnan(box[0]):
                parts.add("w", way_id, tuple(float(v) for v in box), tags)
        batch.clear()
        refs.clear()

    for way in osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(_key_filter()):
        tags = dict(way.tags)
        if _wanted(tags):
            batch.append((way.id, tags))
            refs.append(_refs(way))
            if len(batch) >= BATCH_WAYS:
                flush()
    flush()


def _member_boxes(pbf: Path, cache: np.memmap, way_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted member way ids and their boxes."""
    ids: list[int] = []
    boxes: list[np.ndarray] = []
    refs: list[list[int]] = []
    for way in osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(osmium.filter.IdFilter(np.unique(way_ids).tolist())):
        ids.append(way.id)
        refs.append(_refs(way))
        if len(refs) >= BATCH_WAYS:
            boxes.append(_boxes(cache, refs))
            refs.clear()
    if refs:
        boxes.append(_boxes(cache, refs))
    found = np.asarray(ids, dtype=np.int64)
    box = np.concatenate(boxes) if boxes else np.empty((0, 4))
    order = np.argsort(found)
    return found[order], box[order]


def extract(pbf: Path, out: Path, cache_file: Path, skip_nodes: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if skip_nodes:  # node rows from an earlier run (they come first in every part that holds them); drop its way / relation rows
        files = sorted(out.glob("part-*.parquet"))
        last = 0
        for f in files:
            t = pq.read_table(f)
            nodes = t.filter(pc.equal(t["osm_type"], "n"))
            if nodes.num_rows == 0:
                f.unlink()
            else:
                if nodes.num_rows < t.num_rows:
                    pq.write_table(nodes, f, compression="zstd")
                last = int(f.stem.split("-")[1])
        parts = _Parts(out, first=last + 1)
        print(f"kept node rows in parts 0-{last}", flush=True)
    else:
        if any(out.glob("part-*.parquet")):
            raise SystemExit(f"{out} already holds parts; remove them first or pass --skip-nodes")
        parts = _Parts(out)
        _nodes(pbf, parts, cache_file)
        parts.flush()
        print(f"nodes: {parts.count['n']:,} kept, {(time.time() - start) / 60:.0f} min", flush=True)
    cache = np.memmap(cache_file, dtype=np.int32, mode="r").reshape(-1, 2)
    relations, rel_index, way_ids = _relations(pbf)
    print(f"relations: {len(relations):,} kept, {len(way_ids):,} member ways, {(time.time() - start) / 60:.0f} min", flush=True)
    _ways(pbf, parts, cache)
    print(f"ways: {parts.count['w']:,} kept, {(time.time() - start) / 60:.0f} min", flush=True)
    found, boxes = _member_boxes(pbf, cache, way_ids)
    pos = np.clip(np.searchsorted(found, way_ids), 0, max(len(found) - 1, 0))
    hit = (found[pos] == way_ids) if len(found) else np.zeros(len(way_ids), bool)
    union = np.full((len(relations), 4), np.nan)
    with np.errstate(invalid="ignore"):
        for col, reduce in enumerate((np.fmin, np.fmax, np.fmin, np.fmax)):
            reduce.at(union[:, col], rel_index[hit], boxes[pos[hit], col])
    for (rel_id, tags), box in zip(relations, union):
        if not np.isnan(box[0]):
            parts.add("r", rel_id, tuple(float(v) for v in box), tags)
    parts.flush()
    print(f"done: kept {parts.count} (nodes: earlier run if --skip-nodes), {parts.part} parts, {(time.time() - start) / 60:.0f} min", flush=True)


def summary(out: Path) -> None:
    files = sorted(out.glob("part-*.parquet"))
    counts: dict[str, int] = {}
    named = 0
    for f in files:
        t = pq.read_table(f, columns=["osm_type", "name"])
        for d in t["osm_type"].value_counts().to_pylist():
            counts[d["values"]] = counts.get(d["values"], 0) + d["counts"]
        named += int(pc.sum(pc.not_equal(t["name"], "")).as_py() or 0)
    rows = sum(counts.values())
    size = sum(f.stat().st_size for f in files)
    print(f"{len(files)} parts, {rows:,} rows {counts}, named {named / max(rows, 1):.0%}, {size / 1e9:.1f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", type=Path, nargs="?")
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--cache", type=Path, help="dense on-disk node-location cache (written by the node pass)")
    parser.add_argument("--skip-nodes", action="store_true", help="reuse the node rows in OUT and a complete --cache from an earlier run")
    parser.add_argument("--summary", type=Path, metavar="OUT", help="print row counts of an existing extract")
    args = parser.parse_args()
    if args.summary:
        summary(args.summary)
    else:
        extract(args.pbf, args.out, args.cache, args.skip_nodes)


if __name__ == "__main__":
    main()
