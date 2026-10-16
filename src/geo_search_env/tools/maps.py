# Offline stand-in for Google Maps (see memory / plan: offline replica of the online tools, so skills learned in RL transfer to the live
# tools): maps_search(query, near=) finds places by name like Maps search, features_near(lat, lon, km, features) lists which kinds of
# feature (church, river, rail bridge, ...) exist around a point. Data: the OSM planet extract (osm_extract.py) plus the GeoNames / Overture
# names already in text_search's places.db. Storage: data/maps_search/osm.db (SQLite: FTS5 over every name variant, an R*Tree over boxes);
# `stage` copies it to the NVMe drive for fast reads (wiped on restart, so redo after each one).
# Usage: PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.tools.maps build       (DuckDB; a few hours)
#        PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.tools.maps features-index   (features_near in batch: FeatureIndex)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.maps stage
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.maps query "City Island Avenue" [--near LAT LON KM]
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.maps query "+33 1 42 60 34 56"      (a phone number or website: the business)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.maps near 41.651 -0.889 2 church river "rail bridge"

"""build: osm.db: places (one row per OSM object: name, every name variant, a few tag pairs as `kinds`, an address from addr:* tags, box),
       names (FTS5 over the variants, unicode61 without diacritics, as text_search), boxes (R*Tree over the boxes; rows written in a
       coarse spatial order so the tree builds faster).
query: phrase match on all names in OSM and places.db (GeoNames, Overture); exact-name matches first, then (with near) by distance, then
       by how notable the place is; results sharing a name and kind are merged (within 0.5 km, 3 km for streets). Printed like a Maps result list.
       A phone number or a website / domain instead finds the Overture businesses listing it (places.db phones / domains tables).
near:  per requested feature, the matching objects whose box is within KM of the point: count, nearest distance and name. Feature words
       map to OSM tag pairs (FEATURES); a raw 'key=value' works too. Long features (rivers, rail lines) are boxes, so 'within' is generous."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Sequence

import numpy as np

from ..experiment.text_search import GENERIC_TLDS, PLACES_DB, _countries, _db_dir, domain_of, haversine_km, norm, phone_digits

ROOT = Path("data/maps_search")
NVME = Path("/opt/dlami/nvme/maps_search")
OSM_DB = ROOT / "osm.db"
EXTRACT = Path("data/osm/extract")
NAME_KEYS = ("name", "int_name", "official_name", "alt_name", "old_name", "short_name", "loc_name", "brand", "operator")
KIND_KEYS = ("amenity", "shop", "tourism", "historic", "leisure", "natural", "waterway", "railway", "aeroway", "man_made", "building", "place",
             "highway", "bridge", "tunnel", "generator:source", "aerialway", "religion", "water", "sport", "public_transport", "office", "craft",
             "landuse", "boundary", "admin_level", "cuisine")
NOTABLE = {"place=city": 6, "place=town": 5, "boundary=administrative": 4, "place=village": 3, "railway=station": 3, "aeroway=aerodrome": 3,
           "tourism=attraction": 2, "historic=castle": 2, "natural=peak": 2, "place=suburb": 2}
FEATURES: dict[str, list[tuple[str, ...]]] = {  # word -> tag pairs; a tuple of several pairs means all of them
    "church": [("amenity=place_of_worship", "religion=christian"), ("building=church",), ("building=cathedral",)],
    "mosque": [("amenity=place_of_worship", "religion=muslim"), ("building=mosque",)],
    "temple": [("amenity=place_of_worship", "religion=buddhist"), ("amenity=place_of_worship", "religion=hindu"), ("building=temple",)],
    "synagogue": [("amenity=place_of_worship", "religion=jewish"), ("building=synagogue",)],
    "place of worship": [("amenity=place_of_worship",)], "river": [("waterway=river",)], "canal": [("waterway=canal",)],
    "stream": [("waterway=stream",)], "waterfall": [("waterway=waterfall",), ("natural=waterfall",)], "dam": [("waterway=dam",)],
    "lake": [("water=lake",), ("natural=water",)], "beach": [("natural=beach",)], "cliff": [("natural=cliff",)], "glacier": [("natural=glacier",)],
    "peak": [("natural=peak",)], "volcano": [("natural=volcano",)], "bay": [("natural=bay",)], "cave": [("natural=cave_entrance",)],
    "bridge": [("bridge=yes",), ("man_made=bridge",)], "rail bridge": [("bridge=yes", "railway=rail"), ("bridge=viaduct", "railway=rail")],
    "road bridge": [("bridge=yes", "highway=*")], "tunnel": [("tunnel=yes",)], "station": [("railway=station",), ("railway=halt",)],
    "tram stop": [("railway=tram_stop",)], "railway": [("railway=rail",)], "airport": [("aeroway=aerodrome",)], "harbour": [("leisure=marina",),
    ("landuse=harbour",)], "marina": [("leisure=marina",)], "pier": [("man_made=pier",)], "ferry terminal": [("amenity=ferry_terminal",)],
    "lighthouse": [("man_made=lighthouse",)], "tower": [("man_made=tower",)], "water tower": [("man_made=water_tower",)],
    "windmill": [("man_made=windmill",)], "wind turbine": [("generator:source=wind",)], "chimney": [("man_made=chimney",)],
    "castle": [("historic=castle",), ("building=castle",)], "ruins": [("historic=ruins",)], "monument": [("historic=monument",)],
    "memorial": [("historic=memorial",)], "fort": [("historic=fort",)], "archaeological site": [("historic=archaeological_site",)],
    "museum": [("tourism=museum",)], "viewpoint": [("tourism=viewpoint",)], "zoo": [("tourism=zoo",)], "theme park": [("tourism=theme_park",)],
    "stadium": [("leisure=stadium",), ("building=stadium",)], "golf course": [("leisure=golf_course",)], "university": [("amenity=university",)],
    "hospital": [("amenity=hospital",)], "town hall": [("amenity=townhall",)], "fountain": [("amenity=fountain",)], "clock": [("amenity=clock",)],
    "market": [("amenity=marketplace",)], "cable car": [("aerialway=cable_car",), ("aerialway=gondola",)], "park": [("leisure=park",)],
    # the rest of observe.VOCAB (what the 4B clue reader reports), on tag keys osm.db keeps (KIND_KEYS)
    "railway station": [("railway=station",), ("railway=halt",)], "shrine": [("historic=wayside_shrine",), ("amenity=place_of_worship", "religion=shinto")],
    "palace": [("historic=palace",), ("building=palace",)], "square": [("place=square",)], "theatre": [("amenity=theatre",)],
    "cemetery": [("landuse=cemetery",), ("amenity=grave_yard",)], "garden": [("leisure=garden",)], "observatory": [("man_made=observatory",)],
    "island": [("place=island",), ("place=islet",)], "mountain": [("natural=peak",)], "hill": [("natural=hill",), ("natural=peak",)],
    "valley": [("natural=valley",)], "desert": [("natural=desert",), ("natural=sand",)], "mountain pass": [("natural=saddle",)],
    "forest": [("landuse=forest",), ("natural=wood",)], "vineyard": [("landuse=vineyard",)],
}
FEATURES_NPZ = ROOT / "features.npz"  # features_near in batch (training): every object matching a FEATURES word, as boxes, plus a density grid
DENSITY_DEG = 0.01  # density grid cell (~1 km): how well OSM maps a place, to tell 'none nearby' from 'not mapped'
MAX_MATCHES = 2000  # name matches scored per source; more and the name is too common to rank by anything but distance
RARE = 50_000  # with near: names matched at most this often are looked up by name first, commoner ones by scanning the box
TOP = 10
MERGE_KM = 0.5
STREET_MERGE_KM = 3.0


def _connect_new(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA cache_size = -2000000; PRAGMA temp_store = FILE;")
    return con


def _address(tags: dict[str, str]) -> str:
    street = " ".join(x for x in (tags.get("addr:housenumber"), tags.get("addr:street")) if x)
    town = tags.get("addr:city") or tags.get("addr:place") or tags.get("addr:suburb") or ""
    return ", ".join(x for x in (street, " ".join(x for x in (tags.get("addr:postcode"), town) if x), tags.get("addr:country")) if x)


def build(memory: str = "6GB") -> None:
    import duckdb

    con = _connect_new(OSM_DB)
    con.executescript("""
        CREATE TABLE places (rowid INTEGER PRIMARY KEY, osm TEXT, name TEXT, key TEXT, variants TEXT, kinds TEXT, address TEXT,
                             lat REAL, lon REAL, lat_min REAL, lat_max REAL, lon_min REAL, lon_max REAL);
        CREATE VIRTUAL TABLE boxes USING rtree (id, lat_min, lat_max, lon_min, lon_max);""")
    duck = duckdb.connect()
    spill = NVME.parent / "maps_search_tmp" if NVME.parent.exists() else ROOT / "duckdb_tmp"  # ~14 GB of sort spill; the NVMe drive is faster
    duck.execute(f"SET memory_limit = '{memory}'; SET threads = 4; SET temp_directory = '{spill}'")
    cursor = duck.execute(f"""
        SELECT osm_type, osm_id, name, tags, lat, lon, lat_min, lat_max, lon_min, lon_max FROM read_parquet('{EXTRACT}/*.parquet')
        ORDER BY floor(lat) * 1000 + floor(lon), floor(lat * 8) * 10000 + floor(lon * 8)""")
    rowid = 0
    while batch := cursor.fetchmany(200_000):
        places, boxes = [], []
        for osm_type, osm_id, name, tags_json, lat, lon, lat_min, lat_max, lon_min, lon_max in batch:
            tags = json.loads(tags_json)
            variants = {v for k, v in tags.items() if k in NAME_KEYS or k.startswith(("name:", "alt_name:", "official_name:", "old_name:"))}
            kinds = " ".join(f"{k}={v}" for k in KIND_KEYS if (v := tags.get(k)) and ";" not in v and len(v) < 40)
            rowid += 1
            places.append((rowid, f"{osm_type}{osm_id}", name, norm(name), " | ".join(sorted(variants)), kinds, _address(tags), lat, lon,
                           lat_min, lat_max, lon_min, lon_max))
            boxes.append((rowid, lat_min, lat_max, lon_min, lon_max))
        con.executemany("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", places)
        con.executemany("INSERT INTO boxes VALUES (?, ?, ?, ?, ?)", boxes)
        if rowid % 5_000_000 < 200_000:
            print(f"  {rowid / 1e6:.0f}M objects", flush=True)
    print(f"{rowid:,} objects; indexing names", flush=True)
    con.executescript("""
        CREATE INDEX places_key ON places (key);
        CREATE VIRTUAL TABLE names USING fts5 (variants, content = 'places', content_rowid = 'rowid', tokenize = 'unicode61 remove_diacritics 2');
        INSERT INTO names (names) VALUES ('rebuild');
        INSERT INTO names (names) VALUES ('optimize');""")
    con.commit()
    con.close()
    shutil.rmtree(spill, ignore_errors=True)
    print(f"-> {OSM_DB} ({OSM_DB.stat().st_size / 1e9:.1f} GB)", flush=True)


def features_index(memory: str = "3GB") -> None:
    """features.npz: one row per (object, FEATURES word it matches) with the object's box, and the number of OSM objects (any kind) per
    DENSITY_DEG cell; one extract part at a time (DuckDB parses the tag keys the rules use once, the rules run on the numpy columns)."""

    import duckdb

    keys = sorted({t.split("=")[0] for rules in FEATURES.values() for rule in rules for t in rule})
    words = list(FEATURES)
    duck = duckdb.connect()
    duck.execute(f"SET memory_limit = '{memory}'; SET threads = 3")
    paths = ", ".join(f"'$.\"{k}\"'" for k in keys)
    ncol = round(360 / DENSITY_DEG)
    feat, box, cells = [], [], [(np.empty(0, np.int64), np.empty(0, np.int64))]
    parts = sorted(EXTRACT.glob("*.parquet"))
    for n, part in enumerate(parts):
        t = duck.execute(f"""SELECT {", ".join(f"v[{i + 1}] AS k{i}" for i in range(len(keys)))}, lat_min, lat_max, lon_min, lon_max FROM (
                             SELECT json_extract_string(tags, [{paths}]) AS v, * FROM read_parquet('{part}'))""").fetchnumpy()
        col = {k: np.asarray(t[f"k{i}"], dtype=object) for i, k in enumerate(keys)}
        b = np.stack([np.asarray(t[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")], 1).astype(np.float32)
        for w, word in enumerate(words):
            hit = np.zeros(len(b), bool)
            for rule in FEATURES[word]:
                m = np.ones(len(b), bool)
                for pair in rule:
                    k, v = pair.split("=", 1)
                    m &= np.not_equal(col[k], None) if v == "*" else np.equal(col[k], v)
                hit |= m
            feat.append(np.full(hit.sum(), w, np.int16))
            box.append(b[hit])
        c = duck.execute(f"""SELECT floor((lat + 90) / {DENSITY_DEG})::BIGINT * {ncol} + floor((lon + 180) / {DENSITY_DEG})::BIGINT AS c, count(*) AS n
                             FROM read_parquet('{part}') GROUP BY c""").fetchnumpy()
        cells.append((c["c"].astype(np.int64), c["n"].astype(np.int64)))
        if len(cells) > 10 or n == len(parts) - 1:  # merge cells shared by parts
            cell, inv = np.unique(np.concatenate([x[0] for x in cells]), return_inverse=True)
            cells = [(cell, np.bincount(inv, np.concatenate([x[1] for x in cells])).astype(np.int64))]
        print(f"  part {n + 1}/{len(parts)}: {sum(map(len, feat)):,} feature rows", flush=True)
    feat, box = np.concatenate(feat), np.concatenate(box)
    cell, cell_n = cells[0]
    np.savez(FEATURES_NPZ, words=json.dumps(words), feat=feat, box=box, cell=cell, cell_n=cell_n.astype(np.int32))
    for w, word in enumerate(words):
        print(f"  {word}: {np.sum(feat == w):,}")
    print(f"-> {FEATURES_NPZ} ({len(feat):,} rows, {len(cell):,} density cells)", flush=True)


def _xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class FeatureIndex:
    """features_near for many points at once (training the agent): the same FEATURES words and box distances as MapsSearch.near, from
    features.npz. Objects are split by box size so each KD-tree (over box centres) is searched with radius km + its largest half-diagonal,
    then filtered by the exact point-to-box distance."""

    TIERS_KM = (0.25, 1.0, 4.0, 16.0, 64.0, math.inf)  # box half-diagonal tiers

    def __init__(self, path: Path = FEATURES_NPZ) -> None:
        z = np.load(path)
        self.words = json.loads(str(z["words"]))
        self._feat, self._box = z["feat"], z["box"]  # float32; hit rows are upcast
        self._cell, self._cell_n = z["cell"], z["cell_n"]
        self._trees: dict[int, list] = {}

    def _tiers(self, w: int) -> list:
        from scipy.spatial import cKDTree

        if w not in self._trees:
            box = self._box[self._feat == w]
            clat, clon = box[:, :2].astype(np.float64).mean(1), box[:, 2:].astype(np.float64).mean(1)
            half = 0.5 * np.hypot((box[:, 1] - box[:, 0]) * 111.2, (box[:, 3] - box[:, 2]) * 111.2 * np.cos(np.radians(clat)))
            tier = np.searchsorted(self.TIERS_KM, half)
            self._trees[w] = [(cKDTree(_xyz(clat[m], clon[m])), box[m], float(half[m].max())) for t in range(len(self.TIERS_KM))
                              if (m := tier == t).any()]
        return self._trees[w]

    def near(self, lat: np.ndarray, lon: np.ndarray, km: float, words: Sequence[str], chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
        """count [points, words] of matching objects within km, nearest km [points, words] (inf if none); points in chunks (hit lists)."""

        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        parts = [self._near(lat[i:i + chunk], lon[i:i + chunk], km, words) for i in range(0, len(lat), chunk)]
        if not parts:
            return np.zeros((0, len(words)), np.int32), np.zeros((0, len(words)), np.float32)
        return np.concatenate([c for c, _ in parts]), np.concatenate([d for _, d in parts])

    def _near(self, lat: np.ndarray, lon: np.ndarray, km: float, words: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        q = _xyz(lat, lon)
        count = np.zeros((len(lat), len(words)), np.int32)
        nearest = np.full((len(lat), len(words)), np.inf, np.float32)
        for j, word in enumerate(words):
            for tree, box, half in self._tiers(self.words.index(word)):
                reach = (km + half) * 1.01 + 0.01
                hits = tree.query_ball_point(q, 2 * math.sin(min(reach / 6371.0, math.pi) / 2), workers=-1)
                lens = np.fromiter(map(len, hits), np.int64, len(hits))
                if not lens.sum():
                    continue
                pt, obj = np.repeat(np.arange(len(lat)), lens), np.concatenate([np.asarray(h, np.int64) for h in hits if h])
                b = box[obj].astype(np.float64)
                p1, p2 = np.radians(lat[pt]), np.radians(np.clip(lat[pt], b[:, 0], b[:, 1]))
                dlon = np.radians(np.clip(lon[pt], b[:, 2], b[:, 3]) - lon[pt])
                a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
                d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                keep = d <= km
                np.add.at(count[:, j], pt[keep], 1)
                np.minimum.at(nearest[:, j], pt[keep], d[keep].astype(np.float32))
        return count, nearest

    def density(self, lat: np.ndarray, lon: np.ndarray, km: float) -> np.ndarray:
        """Number of OSM objects (any kind) in the DENSITY_DEG cells within about km of each point."""

        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        ncol = round(360 / DENSITY_DEG)
        row, col = np.floor((lat + 90) / DENSITY_DEG).astype(np.int64), np.floor((lon + 180) / DENSITY_DEG).astype(np.int64)
        ky = math.ceil(km / (111.2 * DENSITY_DEG))
        kx = np.ceil(km / (111.2 * DENSITY_DEG * np.maximum(np.cos(np.radians(lat)), 0.05))).astype(np.int64)
        out = np.zeros(len(lat), np.int64)
        for dy in range(-ky, ky + 1):
            for dx in range(-int(kx.max()), int(kx.max()) + 1):
                m = np.abs(dx) <= kx
                c = (row[m] + dy) * ncol + (col[m] + dx) % ncol
                i = np.clip(np.searchsorted(self._cell, c), 0, len(self._cell) - 1)
                out[m] += np.where(self._cell[i] == c, self._cell_n[i], 0)
        return out


def stage() -> None:
    NVME.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OSM_DB, NVME / OSM_DB.name)
    print(f"{OSM_DB} -> {NVME / OSM_DB.name}", flush=True)


def _osm_path() -> Path:
    if os.environ.get("MAPS_SEARCH_DIR"):
        return Path(os.environ["MAPS_SEARCH_DIR"]) / OSM_DB.name
    fast = NVME / OSM_DB.name
    return fast if fast.exists() and fast.stat().st_size == OSM_DB.stat().st_size else OSM_DB


@dataclass
class Place:
    name: str
    kind: str
    address: str
    lat: float
    lon: float
    source: str  # osm / overture / geonames
    exact: bool
    notable: int
    km: float | None = None  # from `near`
    label: str = ""  # nearest town, country


@dataclass
class Feature:
    word: str
    count: int
    nearest_km: float | None
    nearest_name: str


def _box_km(lat: float, lon: float, lat_min, lat_max, lon_min, lon_max) -> np.ndarray:
    """Distance from a point to boxes (0 inside), clamping the point into each box."""
    return haversine_km(lat, lon, np.clip(lat, lat_min, lat_max), np.clip(lon, lon_min, lon_max))


def _phrase(query: str) -> str | None:
    words = norm(query).split()
    return '"' + " ".join(words) + '"' if words else None


class MapsSearch:
    def __init__(self) -> None:
        self.osm = sqlite3.connect(f"file:{_osm_path()}?mode=ro", uri=True, check_same_thread=False)
        places_db = _db_dir() / PLACES_DB.name
        self.places = sqlite3.connect(f"file:{places_db}?mode=ro", uri=True, check_same_thread=False) if places_db.exists() else None
        if self.places is not None:
            towns = self.places.execute("SELECT lat, lon, name, cc FROM towns WHERE pop >= 5000").fetchall()
            self.towns = np.asarray([(t[0], t[1]) for t in towns]), [f"{t[2]}, {t[3]}" for t in towns]
        else:
            self.towns = np.empty((0, 2)), []

    def _label(self, lat: float, lon: float) -> str:
        coords, names = self.towns
        if not names:
            return ""
        near = np.flatnonzero((np.abs(coords[:, 0] - lat) < 1) & (np.abs(coords[:, 1] - lon) < 1.5))
        if len(near) == 0:
            return ""
        km = haversine_km(lat, lon, coords[near, 0], coords[near, 1])
        return names[near[int(np.argmin(km))]]

    def _listed(self, query: str) -> list[Place] | None:
        """Businesses listing a phone number or website read in the photo; None if the query is neither."""

        if self.places is None:
            return None
        if not hasattr(self, "_tlds"):
            self._tlds = GENERIC_TLDS | set(_countries()[0])  # country TLDs too ('St.Mary' is not a domain)
        digits, domain = phone_digits(query), domain_of(query, self._tlds)
        if digits:
            intl, d = digits.startswith("+"), digits.lstrip("+")
            if not intl:
                d = d.lstrip("0")  # national number: drop the trunk prefix, match the end of the international one
            ids = [p for p, full in self.places.execute("SELECT place, digits FROM phones WHERE tail = ? LIMIT ?", (d[-7:], MAX_MATCHES))
                   if full.endswith(d) or (intl and d.endswith(full))]
        elif domain:
            ids = [r[0] for r in self.places.execute("SELECT place FROM domains WHERE domain = ? LIMIT ?", (domain, MAX_MATCHES))]
        else:
            return None
        rows = []
        for i in range(0, len(ids), 900):
            chunk = ids[i: i + 900]
            rows += self.places.execute(f"SELECT name, kind, lat, lon, src FROM places WHERE rowid IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        return [Place(name, kind, "", lat, lon, "overture" if src == 1 else "geonames", True, 0) for name, kind, lat, lon, src in rows]

    def query(self, query: str, near: tuple[float, float, float] | None = None, top: int = TOP) -> list[Place]:
        listed = self._listed(query)
        if listed is not None:
            if near is not None:
                for p in listed:
                    p.km = float(haversine_km(near[0], near[1], np.asarray([p.lat]), np.asarray([p.lon]))[0])
                listed = sorted((p for p in listed if p.km <= near[2]), key=lambda p: p.km)
            for p in listed[:top]:
                p.label = self._label(p.lat, p.lon)
            return listed[:top]
        phrase, key = _phrase(query), norm(query)
        if phrase is None:
            return []
        found: list[Place] = []
        cols = "p.name, p.variants, p.kinds, p.address, p.lat, p.lon"
        if near is not None:
            lat, lon, km = near
            dlat, dlon = km / 111.0, km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
            box = (lat - dlat, lat + dlat, lon - dlon, lon + dlon)
            inside = "b.lat_max >= ? AND b.lat_min <= ? AND b.lon_max >= ? AND b.lon_min <= ?"
            rare = len(self.osm.execute("SELECT rowid FROM names WHERE names MATCH ? LIMIT ?", (phrase, RARE + 1)).fetchall()) <= RARE
            if rare:
                rows = self.osm.execute(f"SELECT {cols} FROM places p JOIN boxes b ON b.id = p.rowid WHERE {inside} "
                                        "AND p.rowid IN (SELECT rowid FROM names WHERE names MATCH ?)", (*box, phrase)).fetchall()
            else:  # a common name ('station', 'main street'): scan the box and match the phrase in Python
                words = f" {key} "
                rows = [r for r in self.osm.execute(f"SELECT {cols} FROM boxes b JOIN places p ON p.rowid = b.id WHERE {inside}", box)
                        if any(words in f" {norm(v)} " for v in r[1].split(" | "))]
        else:
            rows = self.osm.execute(f"SELECT {cols} FROM places p WHERE p.key = ? LIMIT ?", (key, MAX_MATCHES)).fetchall()
            rows += self.osm.execute(f"SELECT {cols} FROM names JOIN places p ON p.rowid = names.rowid WHERE names MATCH ? LIMIT ?",
                                     (phrase, MAX_MATCHES)).fetchall()
        for name, variants, kinds, address, plat, plon in rows:
            exact = any(norm(v) == key for v in variants.split(" | "))
            notable = max((NOTABLE.get(k, 0) for k in kinds.split()), default=0)
            found.append(Place(name or variants.split(" | ")[0], kinds.split()[0] if kinds else "", address, plat, plon, "osm", exact, notable))
        if self.places is not None:
            sql = "SELECT p.name, p.kind, p.lat, p.lon, p.src FROM names JOIN places p ON p.rowid = names.rowid WHERE names MATCH ?"
            args: list = [phrase]
            if near is not None:
                sql += " AND p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ?"
                args += list(box)
            for name, kind, plat, plon, src in self.places.execute(sql + " LIMIT ?", (*args, MAX_MATCHES)).fetchall():
                found.append(Place(name, kind, "", plat, plon, "overture" if src == 1 else "geonames", norm(name) == key, 0))
        if not found:
            return []
        lats, lons = np.asarray([p.lat for p in found]), np.asarray([p.lon for p in found])
        if near is not None:
            km_all = haversine_km(lat, lon, lats, lons)
            for p, d in zip(found, km_all):
                p.km = float(d)
            found = [p for p in found if p.km <= km]
        found.sort(key=lambda p: (not p.exact, p.km if p.km is not None else 0.0, -p.notable, p.source != "osm"))
        merged: list[Place] = []
        for p in found:
            reach = STREET_MERGE_KM if p.kind.startswith("highway=") else MERGE_KM  # a street is many segments
            same = next((m for m in merged if norm(m.name) == norm(p.name) and m.kind == p.kind and haversine_km(
                m.lat, m.lon, np.asarray([p.lat]), np.asarray([p.lon]))[0] < reach), None)
            if same is None:
                merged.append(p)
                if len(merged) == top:
                    break
            elif not same.address and p.address:
                same.address = p.address
        for p in merged:
            p.label = self._label(p.lat, p.lon)
        return merged

    def near(self, lat: float, lon: float, km: float, features: Sequence[str]) -> list[Feature]:
        dlat, dlon = km / 111.0, km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
        rows = self.osm.execute("SELECT p.name, p.kinds, b.lat_min, b.lat_max, b.lon_min, b.lon_max FROM boxes b JOIN places p ON p.rowid = b.id "
                                "WHERE b.lat_max >= ? AND b.lat_min <= ? AND b.lon_max >= ? AND b.lon_min <= ?",
                                (lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchall()
        if rows:
            box = np.asarray([r[2:] for r in rows], dtype=np.float64)
            dist = _box_km(lat, lon, box[:, 0], box[:, 1], box[:, 2], box[:, 3])
        out = []
        for word in features:
            rules = FEATURES.get(word.casefold().strip()) or ([tuple(word.split("+"))] if "=" in word else [])
            hits = []
            for i, (name, kinds, *_) in enumerate(rows):
                have = set(kinds.split())
                keys = {k.split("=")[0] for k in have}
                if any(all(t in have or (t.endswith("=*") and t[:-2] in keys) for t in rule) for rule in rules) and dist[i] <= km:
                    hits.append(i)
            if hits:
                best = min(hits, key=lambda i: dist[i])
                out.append(Feature(word, len(hits), float(dist[best]), rows[best][0]))
            else:
                out.append(Feature(word, 0, None, ""))
        return out


def format_places(query: str, places: list[Place], near: tuple[float, float, float] | None = None) -> str:
    where = f" near ({near[0]:.3f}, {near[1]:.3f}) within {near[2]:g} km" if near else ""
    lines = [f'maps_search("{query}"){where}: {len(places)} result{"s" * (len(places) != 1)}']
    for i, p in enumerate(places, 1):
        kind = p.kind.split("=")[-1].replace("_", " ") if p.kind else ""
        parts = [p.name] + [x for x in (kind, p.address, p.label) if x]
        dist = f", {p.km:.1f} km away" if p.km is not None else ""
        lines.append(f"  {i}. " + " · ".join(parts) + f" ({p.lat:.4f}, {p.lon:.4f}){dist}")
    return "\n".join(lines)


def format_features(lat: float, lon: float, km: float, features: list[Feature]) -> str:
    lines = [f"features_near({lat:.4f}, {lon:.4f}, {km:g} km):"]
    for f in features:
        if f.count:
            lines.append(f"  {f.word}: {f.count} (nearest {f.nearest_km:.1f} km{': ' + f.nearest_name if f.nearest_name else ''})")
        else:
            lines.append(f"  {f.word}: none")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("features-index")
    sub.add_parser("stage")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--near", nargs=3, type=float, metavar=("LAT", "LON", "KM"))
    n = sub.add_parser("near")
    n.add_argument("lat", type=float)
    n.add_argument("lon", type=float)
    n.add_argument("km", type=float)
    n.add_argument("features", nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        build()
    elif args.cmd == "features-index":
        features_index()
    elif args.cmd == "stage":
        stage()
    elif args.cmd == "query":
        near = tuple(args.near) if args.near else None
        print(format_places(args.text, MapsSearch().query(args.text, near), near))
    else:
        print(format_features(args.lat, args.lon, args.km, MapsSearch().near(args.lat, args.lon, args.km, args.features)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
