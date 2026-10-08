# Text search tool for the geolocation agent (AGENT_DIRECTION.md, GEOLOCATION_CLUES.md section 1): text read in a photo (a business or place name,
# a website, a phone number, an event name) -> the places it points to, from offline sources: GeoNames (~13M places and their alternate names),
# Overture places (~60M businesses / POIs with websites and phones) and YFCC100M geotagged Flickr titles + tags (built by yfcc_names extract / merge).
# Storage: SQLite FTS5 databases under data/text_search/ (on disk, stdlib only at query time, so it runs in any env, including GRPO rollouts).
# Usage: PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.text_search build-places     (GeoNames + Overture; needs DuckDB)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_search build-yfcc                  (needs data/yfcc100m/geo.npz)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_search stage   (copy to the NVMe drive after each restart; searches use it)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_search query "Galéria Nedbalka" [--near LAT LON KM]
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_search report --tag dev3k

"""build-places: places.db: places (one row per distinct normalised name of a GeoNames entry, one per named Overture POI: name, lat, lon, country,
              kind, chain flag) with an FTS5 index on the name; domains (registrable domain of each Overture website -> place); phones (digits of
              each Overture phone -> place, keyed by the last 7 digits); towns (GeoNames populated places >= 1,000 people, to label results).
build-yfcc:   yfcc.db: meta (photo id, user, lat, lon, GPS accuracy) and a contentless FTS5 index on the normalised title + tags.
query:        one search, printed as the agent would see it.
report:       every model-produced phrase of a photo set (yfcc_names phrases + observe queries) searched with the photo and its photographer excluded;
              the tool's own top place per photo (picked without the truth) vs the truth and Pinpoint's top-1: search-fixable photos (Pinpoint
              >= 25 km off, pick < 25 km), traps (Pinpoint right, pick >= 25 km off), and a shuffled-phrase control for lucky matches (dev sets only).

Search: the query is classified as a domain / URL, a phone number or a name. Names are matched as a phrase (all words, in order, whole words,
accents and case ignored; if nothing matches, all words in any order). Matches are grouped into 25 km clusters, weighting each Overture / GeoNames
place 1 and each Flickr user 1 in total (so one prolific photographer counts once); clusters are ranked by weight. A country hint comes from a
country-code domain or an international phone prefix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Sequence

import numpy as np

ROOT = Path("data/text_search")  # the durable copy (builds write here)
NVME = Path("/opt/dlami/nvme/text_search")  # instance-local NVMe: much faster reads, wiped on restart (`stage` copies the databases there)


def _db_dir() -> Path:
    """Where searches read: $TEXT_SEARCH_DIR, else the NVMe copy if it is complete, else the durable copy."""

    if os.environ.get("TEXT_SEARCH_DIR"):
        return Path(os.environ["TEXT_SEARCH_DIR"])
    durable = [p for p in (ROOT / "places.db", ROOT / "yfcc.db") if p.exists()]
    if durable and all((NVME / p.name).exists() and (NVME / p.name).stat().st_size == p.stat().st_size for p in durable):
        return NVME
    return ROOT


PLACES_DB, YFCC_DB = ROOT / "places.db", ROOT / "yfcc.db"
GEONAMES = Path("data/pinpoint/geonames")
OVERTURE = Path("data/pinpoint/overture/places")
YFCC_NPZ = Path("data/yfcc100m/geo.npz")
OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")

CLUSTER_KM = 25.0
MAX_MATCHES = 50_000  # matches fetched per source and query; more means the phrase is too common to place anything
SAMPLE = 2_000  # matches clustered per source (a fixed random sample beyond that)
TOP = 8  # clusters returned
MIN_ACCURACY = 11  # Flickr GPS accuracy (1 world .. 16 street); below city level a photo can't place a 25 km cluster
SECOND_LEVEL = {"co", "com", "net", "org", "gov", "ac", "edu", "ne", "or", "go", "gob", "nic", "mil"}  # co.uk, com.au, ...
GENERIC_TLDS = {"com", "net", "org", "info", "biz", "edu", "gov", "mil", "int", "eu", "asia", "cat", "berlin", "nyc", "bayern", "london", "paris",
                "tokyo", "wien", "swiss", "travel", "museum", "shop", "store", "online", "site", "app", "io", "tv", "me", "co", "xyz", "club"}
SHARED_PREFIXES = {"1": "United States / Canada / Caribbean (NANP)", "7": "Russia / Kazakhstan"}
TLD_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\d?\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,})(?:[:/?#].*)?$")


def norm(text: str) -> str:
    """Case folded, accents stripped, non-letters / digits -> one space (as yfcc_names.norm and FTS5 unicode61 with remove_diacritics 2)."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", text).strip()


def domain_of(text: str, tlds: set[str] | None = None) -> str | None:
    """'https://www.porsche.lu/fr/' -> 'porsche.lu'; 'shop.example.co.uk' -> 'example.co.uk'; None if it isn't a domain or URL (or, given tlds,
    its ending isn't one of them, so read text like 'St.Mary' isn't taken for a domain)."""

    m = TLD_RE.match(text.strip().casefold())
    if not m:
        return None
    labels = m.group(1).split(".")
    if tlds is not None and labels[-1] not in tlds:
        return None
    keep = 3 if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL else 2
    return ".".join(labels[-keep:])


def phone_digits(text: str) -> str | None:
    """Digits of a phone number with the international '00' written as '+'; None unless it looks like one (7-15 digits, phone characters only)."""

    t = text.strip()
    if not re.fullmatch(r"(?:tel\.?|phone|ph|t)?[:\s]*\+?[\d\s().\-/]{7,}", t, flags=re.I):
        return None
    digits = re.sub(r"\D", "", t)
    if not 7 <= len(digits) <= 15:
        return None
    if "+" in t:
        return "+" + digits
    return "+" + digits[2:] if digits.startswith("00") else digits


def haversine_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    p1, p2 = math.radians(lat), np.radians(lats)
    a = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _countries() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """GeoNames countryInfo: tld -> country code, phone prefix -> country code, country code -> name."""

    tld, phone, names = {}, {}, {}
    for line in (GEONAMES / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        names[f[0]] = f[4]
        if f[9]:
            tld.setdefault(f[9].lstrip(".").casefold(), f[0])
        for code in re.split(r"\s*and\s*|,", f[12]):
            code = re.sub(r"\D", "", code)
            if code:
                phone.setdefault(code, f[0])
    tld["uk"] = "GB"
    return tld, phone, names


# ---------------------------------------------------------------------------------------------------------------------------- build


def _connect_new(path: Path) -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA cache_size = -1000000; PRAGMA temp_store = FILE;")
    return con


def build_places(memory: str = "8GB") -> None:
    import duckdb

    kinds = {}
    for line in (GEONAMES / "featureCodes_en.txt").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            kinds[f[0]] = f[1]
    con = _connect_new(PLACES_DB)
    con.executescript("""
        CREATE TABLE places (rowid INTEGER PRIMARY KEY, src INTEGER, name TEXT, lat REAL, lon REAL, cc TEXT, kind TEXT, chain INTEGER);
        CREATE TABLE domains (domain TEXT, place INTEGER);
        CREATE TABLE phones (tail TEXT, digits TEXT, place INTEGER);
        CREATE TABLE towns (lat REAL, lon REAL, name TEXT, cc TEXT, pop INTEGER);""")
    rowid = 0

    rows, towns = [], []
    with (GEONAMES / "allCountries.txt").open("r", encoding="utf-8") as stream:
        for n, line in enumerate(stream):
            f = line.rstrip("\n").split("\t")
            lat, lon, code, cc = float(f[4]), float(f[5]), f[6] + "." + f[7], f[8]
            seen = set()
            for name in [f[1], f[2]] + (f[3].split(",") if f[3] else []):
                key = norm(name)
                if len(key) < 2 or key in seen:
                    continue
                seen.add(key)
                rowid += 1
                rows.append((rowid, 0, name, lat, lon, cc, kinds.get(code, code), 0))
            if f[6] == "P" and f[14].isdigit() and int(f[14]) >= 1000:
                towns.append((lat, lon, f[1], cc, int(f[14])))
            if len(rows) >= 200_000:
                con.executemany("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                rows = []
            if n % 2_000_000 == 0:
                print(f"  geonames: {n} entries, {rowid} names", flush=True)
    con.executemany("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.executemany("INSERT INTO towns VALUES (?, ?, ?, ?, ?)", towns)
    print(f"geonames: {rowid} names, {len(towns)} towns", flush=True)

    duck = duckdb.connect()
    duck.execute(f"SET memory_limit = '{memory}'")
    cursor = duck.execute(f"""
        SELECT names."primary", (bbox.ymin + bbox.ymax) / 2, (bbox.xmin + bbox.xmax) / 2, addresses[1].country, basic_category,
               brand IS NOT NULL, websites, phones
        FROM read_parquet('{OVERTURE}/*.parquet') WHERE names."primary" IS NOT NULL""")
    count = 0
    while batch := cursor.fetchmany(200_000):
        places, domains, phones = [], [], []
        for name, lat, lon, cc, kind, chain, websites, numbers in batch:
            rowid += 1
            places.append((rowid, 1, name, lat, lon, cc or "", kind or "", int(chain)))
            for d in {domain_of(w) for w in websites or ()} - {None}:
                domains.append((d, rowid))
            for p in {phone_digits(x) for x in numbers or ()} - {None}:
                digits = p.lstrip("+")
                phones.append((digits[-7:], digits, rowid))
        con.executemany("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)", places)
        con.executemany("INSERT INTO domains VALUES (?, ?)", domains)
        con.executemany("INSERT INTO phones VALUES (?, ?, ?)", phones)
        count += len(batch)
        if count % 5_000_000 < 200_000:
            print(f"  overture: {count} places", flush=True)
    print(f"overture: {count} places; indexing", flush=True)
    con.executescript("""
        CREATE INDEX domains_domain ON domains (domain);
        CREATE INDEX phones_tail ON phones (tail);
        CREATE VIRTUAL TABLE names USING fts5 (name, content = 'places', content_rowid = 'rowid', tokenize = 'unicode61 remove_diacritics 2');
        INSERT INTO names (names) VALUES ('rebuild');
        INSERT INTO names (names) VALUES ('optimize');""")
    con.commit()
    con.close()
    print(f"{rowid} names -> {PLACES_DB} ({PLACES_DB.stat().st_size / 1e9:.1f} GB)", flush=True)


def build_yfcc() -> None:
    z = np.load(YFCC_NPZ)
    vocab = json.loads((YFCC_NPZ.parent / "uid_vocab.json").read_text(encoding="utf-8"))
    photo_id, uid, lat, lon = z["photo_id"], z["uid"], z["lat"], z["lon"]
    accuracy = z["accuracy"] if "accuracy" in z.files else np.full(len(photo_id), 16, np.int8)
    blob, offsets = z["blob"], z["offsets"]
    con = _connect_new(YFCC_DB)
    con.executescript("""
        CREATE TABLE meta (rowid INTEGER PRIMARY KEY, photo_id INTEGER, uid TEXT, lat REAL, lon REAL, accuracy INTEGER);
        CREATE VIRTUAL TABLE titles USING fts5 (text, content = '', tokenize = 'unicode61 remove_diacritics 2');""")
    step = 500_000
    for start in range(0, len(photo_id), step):
        end = min(start + step, len(photo_id))
        raw = blob[offsets[start]:offsets[end]].tobytes()
        base = offsets[start]
        texts = [raw[offsets[i] - base:offsets[i + 1] - base].decode("utf-8", "replace").strip() for i in range(start, end)]
        con.executemany("INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?)",
                        ((i + 1, int(photo_id[i]), vocab[uid[i]], float(lat[i]), float(lon[i]), int(accuracy[i])) for i in range(start, end)))
        con.executemany("INSERT INTO titles (rowid, text) VALUES (?, ?)", ((i + 1, t) for i, t in zip(range(start, end), texts)))
        print(f"  yfcc: {end} of {len(photo_id)} photos", flush=True)
    con.execute("INSERT INTO titles (titles) VALUES ('optimize')")
    con.commit()
    con.close()
    print(f"{len(photo_id)} photos -> {YFCC_DB} ({YFCC_DB.stat().st_size / 1e9:.1f} GB)", flush=True)


def stage() -> None:
    """Copy the built databases to the NVMe drive (redo after every restart)."""

    import shutil

    NVME.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "places.db", ROOT / "yfcc.db"):
        if p.exists():
            shutil.copyfile(p, NVME / p.name)
            print(f"{p} -> {NVME / p.name}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------- search


@dataclass
class Cluster:
    lat: float
    lon: float
    label: str  # nearest town, country code
    weight: float  # places + Flickr users in the cluster
    share: float  # of all matches' weight
    places: int = 0
    photos: int = 0
    users: int = 0
    examples: list[str] = field(default_factory=list)  # matched place names with kind / source


@dataclass
class Result:
    query: str
    kind: str  # name | domain | phone
    matches: dict[str, int]  # per source, before sampling (capped at MAX_MATCHES)
    clusters: list[Cluster]
    hint: str = ""  # country from a domain / phone prefix
    loose: bool = False  # words matched in any order (no phrase match)

    def text(self) -> str:
        """What the agent reads."""

        head = f'text_search("{self.query}") [{self.kind}]: ' + ", ".join(f"{n}{'+' if n >= MAX_MATCHES else ''} {s}" for s, n in self.matches.items())
        lines = [head + (" (words in any order)" if self.loose else "")]
        if self.hint:
            lines.append(f"  hint: {self.hint}")
        if not self.clusters:
            lines.append("  no places found")
        for i, c in enumerate(self.clusters, 1):
            parts = []
            if c.places:
                parts.append(f"{c.places} place{'s' * (c.places > 1)}: " + "; ".join(c.examples))
            if c.photos:
                parts.append(f"{c.photos} Flickr photo{'s' * (c.photos > 1)} by {c.users} user{'s' * (c.users > 1)}")
            lines.append(f"  {i}. near {c.label} ({c.lat:.3f}, {c.lon:.3f}): {c.share:.0%} of matches; " + "; ".join(parts))
        return "\n".join(lines)


class TextSearch:
    def __init__(self, yfcc: bool = True, min_accuracy: int = MIN_ACCURACY) -> None:
        places_db, yfcc_db = PLACES_DB, YFCC_DB
        if places_db == ROOT / "places.db":  # not overridden (tests point these at temporary files)
            places_db, yfcc_db = _db_dir() / "places.db", _db_dir() / "yfcc.db"
        self.places = sqlite3.connect(f"file:{places_db}?mode=ro", uri=True, check_same_thread=False)
        self.yfcc = sqlite3.connect(f"file:{yfcc_db}?mode=ro", uri=True, check_same_thread=False) if yfcc and yfcc_db.exists() else None
        self.min_accuracy = min_accuracy
        self.tld, self.phone, self.country_names = _countries()
        t = np.asarray(self.places.execute("SELECT lat, lon, pop FROM towns").fetchall(), np.float64)
        self.town_lat, self.town_lon = t[:, 0], t[:, 1]
        self.town_name = [f"{n}, {c}" for n, c in self.places.execute("SELECT name, cc FROM towns")]
        self.cells: dict[tuple[int, int], list[int]] = {}
        for i, (la, lo) in enumerate(zip(self.town_lat, self.town_lon)):
            self.cells.setdefault((int(math.floor(la)), int(math.floor(lo))), []).append(i)

    def label(self, lat: float, lon: float) -> str:
        for r in (0, 1, 3):
            idx = [i for a in range(-r, r + 1) for b in range(-r, r + 1) for i in self.cells.get((int(math.floor(lat)) + a, int(math.floor(lon)) + b), [])]
            if idx:
                idx = np.asarray(idx)
                return self.town_name[int(idx[haversine_km(lat, lon, self.town_lat[idx], self.town_lon[idx]).argmin()])]
        return "?"

    # ---- sources

    def _place_rows(self, rowids: list[int]) -> list[tuple]:
        out = []
        for s in range(0, len(rowids), 900):
            chunk = rowids[s:s + 900]
            out += self.places.execute(f"SELECT name, lat, lon, cc, kind, src, chain FROM places WHERE rowid IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        return out

    @staticmethod
    def _sample(ids: list[int], key: str) -> list[int]:
        if len(ids) <= SAMPLE:
            return ids
        rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
        return [ids[i] for i in sorted(rng.choice(len(ids), SAMPLE, replace=False))]

    def _fts(self, con: sqlite3.Connection, table: str, match: str, box: tuple | None, meta: str) -> list[int]:
        if box is None:
            sql, args = f"SELECT rowid FROM {table} WHERE {table} MATCH ? LIMIT {MAX_MATCHES}", [match]
        else:
            sql = (f"SELECT m.rowid FROM {table} JOIN {meta} m ON m.rowid = {table}.rowid WHERE {table} MATCH ? "
                   f"AND m.lat BETWEEN ? AND ? AND m.lon BETWEEN ? AND ? LIMIT {MAX_MATCHES}")
            args = [match, *box]
        return [r[0] for r in con.execute(sql, args)]

    def _names(self, words: list[str], box: tuple | None, exclude: tuple[int, str] | None) -> tuple[list, list, dict[str, int], bool]:
        places, photos, counts = [], [], {}
        for loose in (False, True):
            if loose and len(words) < 2:
                break
            match = " ".join(words) if loose else '"' + " ".join(words) + '"'
            ids = self._fts(self.places, "names", match, box, "places")
            counts["places"] = len(ids)
            places = self._place_rows(self._sample(ids, match))
            if self.yfcc is not None:
                ids = self._fts(self.yfcc, "titles", match, box, "meta")
                rows = []
                for s in range(0, len(ids), 900):
                    chunk = ids[s:s + 900]
                    rows += self.yfcc.execute(f"SELECT photo_id, uid, lat, lon, accuracy FROM meta WHERE rowid IN ({','.join('?' * len(chunk))})", chunk).fetchall()
                rows = [r for r in rows if r[4] >= self.min_accuracy and (exclude is None or (r[0] != exclude[0] and r[1] != exclude[1]))]
                counts["Flickr photos"] = len(rows)
                photos = [rows[i] for i in self._sample(list(range(len(rows))), match)]
            if places or photos:
                return places, photos, counts, loose
        return places, photos, counts, False

    def _domain(self, domain: str) -> tuple[list, dict[str, int]]:
        ids = [r[0] for r in self.places.execute(f"SELECT place FROM domains WHERE domain = ? LIMIT {MAX_MATCHES}", [domain])]
        return self._place_rows(self._sample(ids, domain)), {"places": len(ids)}

    def _phone(self, digits: str) -> tuple[list, dict[str, int]]:
        intl, d = digits.startswith("+"), digits.lstrip("+")
        if not intl:
            d = d.lstrip("0")  # national number: drop the trunk prefix, match the end of the international one
        ids = [p for p, full in self.places.execute(f"SELECT place, digits FROM phones WHERE tail = ? LIMIT {MAX_MATCHES}", [d[-7:]])
               if full.endswith(d) or (intl and d.endswith(full))]
        return self._place_rows(self._sample(ids, d)), {"places": len(ids)}

    # ---- search

    def search(self, query: str, near: tuple[float, float, float] | None = None, exclude: tuple[int, str] | None = None) -> Result:
        """query: text as read in the photo. near: (lat, lon, km) to look only around a place. exclude: (Flickr photo id, user) of the photo being
        located, never matched (set by the environment, not the agent)."""

        hint, loose, photos = "", False, []
        domain, digits = domain_of(query, GENERIC_TLDS | set(self.tld)), phone_digits(query)
        if digits:
            kind = "phone"
            places, counts = self._phone(digits)
            if digits.startswith("+"):
                d = digits[1:]
                code = next((d[:k] for k in (4, 3, 2, 1) if d[:k] in self.phone), None)
                if code:
                    cc = self.phone[code]
                    hint = f"phone prefix +{code} = " + SHARED_PREFIXES.get(code, f"{self.country_names.get(cc, cc)} ({cc})")
        elif domain:
            kind = "domain"
            places, counts = self._domain(domain)
            cc = self.tld.get(domain.rsplit(".", 1)[-1])
            if cc:
                hint = f"domain .{domain.rsplit('.', 1)[-1]} = {self.country_names.get(cc, cc)} ({cc})"
            if not places:  # also search the name part as words ("porsche.lu" -> "porsche")
                name_places, photos, counts2, loose = self._names(norm(domain.rsplit(".", 1)[0]).split(), None, exclude)
                places, counts = name_places, {"places (domain not listed; name words)": counts2.get("places", 0)} | {
                    k: v for k, v in counts2.items() if k != "places"}
        else:
            kind = "name"
            words = norm(query).split()
            if not words:
                return Result(query, kind, {}, [])
            box = None
            if near:
                la, lo, km = near
                dlat, dlon = km / 111.0, km / (111.0 * max(math.cos(math.radians(la)), 0.05))
                box = (la - dlat, la + dlat, lo - dlon, lo + dlon)
            places, photos, counts, loose = self._names(words, box, exclude)
        clusters = self._cluster(places, photos, query, near)
        return Result(query, kind, counts, clusters, hint, loose)

    def _cluster(self, places: list[tuple], photos: list[tuple], query: str, near: tuple | None) -> list[Cluster]:
        lat = np.asarray([p[1] for p in places] + [p[2] for p in photos], np.float64)
        lon = np.asarray([p[2] for p in places] + [p[3] for p in photos], np.float64)
        if near and len(lat):
            keep = haversine_km(near[0], near[1], lat, lon) <= near[2]
            n = len(places)
            places = [p for p, k in zip(places, keep[:n]) if k]
            photos = [p for p, k in zip(photos, keep[n:]) if k]
            lat, lon = lat[keep], lon[keep]
        n_places = len(places)
        if not len(lat):
            return []
        per_user: dict[str, int] = {}
        for p in photos:
            per_user[p[1]] = per_user.get(p[1], 0) + 1
        w = np.asarray([1.0] * n_places + [1.0 / per_user[p[1]] for p in photos])
        rad = np.radians(np.stack([lat, lon], 1))
        alive = np.ones(len(lat), bool)
        total, qn, out = w.sum(), norm(query), []
        while alive.any() and len(out) < TOP:
            idx = np.flatnonzero(alive)
            la, lo = rad[idx, 0], rad[idx, 1]
            a = (np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2)
            near_m = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) < CLUSTER_KM
            j = int((near_m * w[idx][None]).sum(1).argmax())
            members = idx[near_m[j]]
            alive[members] = False
            pm = members[members < n_places]
            ph = members[members >= n_places] - n_places
            exact = sorted(pm, key=lambda i: (norm(places[i][0]) != qn, places[i][6], places[i][5]))  # exact names, non-chains, Overture first
            examples = [f"{places[i][0]} ({places[i][4] or '?'}, {'Overture' if places[i][5] else 'GeoNames'}{', chain' if places[i][6] else ''})"
                        for i in exact[:3]]
            c = idx[j]
            out.append(Cluster(float(lat[c]), float(lon[c]), self.label(float(lat[c]), float(lon[c])), float(w[members].sum()),
                               float(w[members].sum() / total), len(pm), len(ph), len({photos[i][1] for i in ph}), examples))
        return out


# ---------------------------------------------------------------------------------------------------------------------------- report


def _phrases(tag: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    path = OUT / f"yfcc_phrases_{tag}.json"
    if path.exists():
        for image_id, table in json.loads(path.read_text(encoding="utf-8")).items():
            out.setdefault(image_id, []).extend(table)
    path = OUT / f"queries_{tag}_9b.json"
    if path.exists():
        for image_id, queries in json.loads(path.read_text(encoding="utf-8")).items():
            out.setdefault(image_id, []).extend(queries)
    return {k: list(dict.fromkeys(v)) for k, v in out.items() if v}


def _pick(results: list[Result]) -> Cluster | None:
    """The tool's own answer for a photo, without the truth: the heaviest top cluster among its queries that holds most of its query's matches."""

    best = None
    for r in results:
        if r.clusters and r.clusters[0].share >= 0.5 and (best is None or r.clusters[0].weight > best.weight):
            best = r.clusters[0]
    return best


def report(tag: str, limit: int | None = None) -> None:
    import time

    from .yfcc_names import _mp16_authors

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))[:limit]
    phrases = _phrases(tag)
    authors = _mp16_authors({e["image_id"] for e in photos})
    tool = TextSearch()
    start, cache = time.time(), {}
    n_queries = 0

    def results(e: dict, queries: list[str]) -> list[Result]:
        nonlocal n_queries
        own = (int(e["image_id"].split("_")[-1].split(".")[0]), authors.get(e["image_id"], ""))
        out = []
        for q in queries:
            if (q, own) not in cache:
                cache[(q, own)] = tool.search(q, exclude=own)
                n_queries += 1
            out.append(cache[(q, own)])
        return out

    truth = np.asarray([e["truth"] for e in photos], np.float64)
    base = np.asarray([haversine_km(*e["truth"], np.asarray([e["pool"][0][0]]), np.asarray([e["pool"][0][1]]))[0] for e in photos])
    perm = np.random.default_rng(0).permutation(len(photos))
    rows = {"real": [], "shuffled": []}
    any_hit = []
    for p, e in enumerate(photos):
        for name, source in (("real", e), ("shuffled", photos[perm[p]])):
            res = results(e, phrases.get(source["image_id"], []))
            pick = _pick(res)
            rows[name].append(np.inf if pick is None else haversine_km(*truth[p], np.asarray([pick.lat]), np.asarray([pick.lon]))[0])
            if name == "real":
                any_hit.append(any((haversine_km(*truth[p], np.asarray([c.lat for c in r.clusters]), np.asarray([c.lon for c in r.clusters])) < 25).any()
                                   for r in res if r.clusters))
        if p % 200 == 0:
            print(f"  {p} photos, {n_queries} searches, {time.time() - start:.0f} s", flush=True)
    elapsed = time.time() - start
    mistakes = base >= 25
    print(f"{tag}: {len(photos)} photos, {sum(e['image_id'] in phrases for e in photos)} with phrases; {n_queries} searches, "
          f"{1000 * elapsed / max(n_queries, 1):.0f} ms each; Pinpoint top-1 < 25 km {(~mistakes).mean():.1%} ({mistakes.sum()} mistakes)")
    for name, d in rows.items():
        d = np.asarray(d)
        picked = np.isfinite(d)
        right = d < 25
        print(f"  {name:8s} phrases: picks {picked.sum():4d}; right < 25 km {right.sum():4d} ({right.sum() / max(picked.sum(), 1):.0%} of picks), "
              f"< 1 km {(d < 1).sum():4d}; search-fixable (Pinpoint wrong, pick right) {(right & mistakes).sum():3d}; "
              f"traps (Pinpoint right, pick wrong) {(picked & ~right & ~mistakes).sum():3d}")
    any_hit = np.asarray(any_hit)
    print(f"  any returned cluster < 25 km (not a blind pick): {any_hit.sum()} photos, {(any_hit & mistakes).sum()} of Pinpoint's mistakes")
    fixable = [photos[p]["image_id"] for p in np.flatnonzero((np.asarray(rows["real"]) < 25) & mistakes)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"text_search_fixable_{tag}.json").write_text(json.dumps(fixable) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-places")
    b.add_argument("--memory", default="8GB")
    sub.add_parser("build-yfcc")
    sub.add_parser("stage")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--near", type=float, nargs=3, metavar=("LAT", "LON", "KM"))
    q.add_argument("--no-yfcc", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--tag", default="dev3k")
    r.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.cmd == "build-places":
        build_places(args.memory)
    elif args.cmd == "build-yfcc":
        build_yfcc()
    elif args.cmd == "stage":
        stage()
    elif args.cmd == "query":
        print(TextSearch(yfcc=not args.no_yfcc).search(args.text, near=tuple(args.near) if args.near else None).text())
    else:
        report(args.tag, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
