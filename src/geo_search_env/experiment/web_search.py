# Offline stand-in for Google web search, encyclopedic part (see memory / plan: offline replica of the online tools): web_search(query)
# returns Google-style results (title, URL, snippet) from Wikipedia in WIKI_LANGS languages and Wikivoyage, with a knowledge panel
# (coordinates, first sentence) when the top result is a page with a primary coordinate. Results are text the agent has to read;
# coordinates appear only in the panel, as on Google.
# Data: data/wikipedia/hf (wikimedia/wikipedia 20231101 parquet), data/wikipedia/geo_tags (<lang>wiki-latest-geo_tags.sql.gz),
# data/wikipedia/wikivoyage (<lang>wikivoyage-latest-pages-articles.xml.bz2). Storage: one SQLite per language under data/web_search/
# (FTS5 over title + passages, passages zlib-compressed); `stage` copies them to the NVMe drive (wiped on restart).
# Usage: PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.web_search build --langs da sv ...   (DuckDB; one process
#            per language list; several can run side by side)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.web_search stage
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.web_search query "Wittenham Clumps"

"""build: per language, pages (title, url, source wikipedia / wikivoyage, primary coordinate and type from geo_tags) and passages (the
       article's first MAX_CHARS characters cut at paragraph ends into ~PASSAGE_CHARS pieces); fts: FTS5 over (title, body) without stored
       content (unicode61 without diacritics; CJK text is indexed by the trigram-free unicode61 tokenizer, so Chinese / Japanese match
       whole runs of characters, which is coarse).
query: the phrase in every language's index (all words in any order if no phrase matches), bm25 with the title weighted 5x, best
       passage per page; pages ranked across languages by bm25. Snippet: ~SNIPPET chars around the first matched word."""

from __future__ import annotations

import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Iterator, Sequence
from urllib.parse import quote
import xml.etree.ElementTree as ET
import zlib

from .text_search import norm

WIKI = Path("data/wikipedia")
ROOT = Path("data/web_search")
NVME = Path("/opt/dlami/nvme/web_search")
WIKI_LANGS = ("en de fr es it nl pl pt ru ja zh sv uk cs ko tr ar ca hu fi no da el he id vi fa ro th").split()
MAX_CHARS = 6000  # of each article: the lead and first sections, where places are described
PASSAGE_CHARS = 700
SNIPPET = 240
TOP = 10
TITLE_EXACT, TITLE_PREFIX, LENGTH_WEIGHT, POP_WEIGHT = 30.0, 8.0, 3.0, 3.0  # ranking (_rank); bm25 of a good match is about -10 to -30
LANG_BONUS = {"en": 5.0}  # Google for an English-speaking user; the same page in other languages is dropped (WebSearch.query)
PER_DB = 20  # pages fetched per language before merging
CAP = 4000  # FTS matches read per language and query (see WebSearch._one)
RESOLVE = 200  # best-scoring matches whose pages are looked up and re-ranked with _rank


def _connect_new(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA cache_size = -1000000; PRAGMA temp_store = FILE;")
    return con


def _passages(text: str) -> list[str]:
    out, cur = [], ""
    paras = re.split(r"\n+", text[:MAX_CHARS])
    if len(paras) == 1 and len(paras[0]) > PASSAGE_CHARS:  # CirrusSearch text has no paragraph breaks: cut at sentence ends
        paras = re.split(r"(?<=[.!?。])\s+", paras[0])
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) > PASSAGE_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def _sql_rows(line: str) -> Iterator[list[str | None]]:
    """Tuples in one line of a MySQL dump (an INSERT with all its rows, or one row per line in newer dumps), as strings (None for NULL)."""
    i, n = (line.index(" VALUES ") + 8 if " VALUES " in line else 0), len(line)
    while i < n:
        if line[i] != "(":
            i += 1
            continue
        i += 1
        row, field = [], []
        while True:
            c = line[i]
            if c == "'":
                i += 1
                while line[i] != "'":
                    if line[i] == "\\":
                        i += 1
                    field.append(line[i])
                    i += 1
                i += 1
                row.append("".join(field))
                field = []
            elif c in ",)":
                if field:
                    v = "".join(field)
                    row.append(None if v == "NULL" else v)
                    field = []
                i += 1
                if c == ")":
                    break
            else:
                field.append(c)
                i += 1
        yield row


def _geo_tags(lang: str) -> dict[int, tuple[float, float, str]]:
    """page id -> (lat, lon, type) of its primary Earth coordinate."""
    path = WIKI / "geo_tags" / f"{lang}wiki-latest-geo_tags.sql.gz"
    if not path.exists():
        return {}
    cols: list[str] = []
    out, in_create = {}, False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("CREATE TABLE"):
                in_create = True
            elif in_create and line.strip().startswith("`"):
                cols.append(line.split("`")[1])
            elif in_create and line.startswith(")"):
                in_create = False
            if not (line.startswith("INSERT INTO") or line.startswith("(")):
                continue
            at = {c: k for k, c in enumerate(cols)}
            for row in _sql_rows(line):
                if row[at["gt_globe"]] == "earth" and row[at["gt_primary"]] == "1":
                    try:
                        out[int(row[at["gt_page_id"]])] = (float(row[at["gt_lat"]]), float(row[at["gt_lon"]]), row[at["gt_type"]] or "")
                    except (TypeError, ValueError):
                        continue
    return out


WIKITEXT = [(re.compile(p, re.S), r) for p, r in [
    (r"<!--.*?-->", ""), (r"<ref[^>]*/>", ""), (r"<ref.*?</ref>", ""), (r"\{\|.*?\|\}", ""), (r"\[\[(?:File|Image|Datei|Fichier|Archivo)[^\]]*\]\]", ""),
    (r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1"), (r"\[https?://\S+ ([^\]]*)\]", r"\1"), (r"'{2,}", ""), (r"<[^>]+>", "")]]


def _strip_wikitext(text: str) -> str:
    for _ in range(3):  # nested templates
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    for pattern, repl in WIKITEXT:
        text = pattern.sub(repl, text)
    text = re.sub(r"(?m)^=+\s*(.*?)\s*=+\s*$", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _voyage_pages(lang: str) -> Iterator[tuple[int, str, str]]:
    path = WIKI / "wikivoyage" / f"{lang}wikivoyage-latest-pages-articles.xml.bz2"
    if not path.exists():
        return
    title = page_id = ns = None
    for _, el in ET.iterparse(bz2.open(path), events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "title":
            title = el.text
        elif tag == "ns":
            ns = el.text
        elif tag == "id" and page_id is None:
            page_id = int(el.text)
        elif tag == "text":
            text = el.text or ""
            if ns == "0" and title and not text.lstrip().casefold().startswith("#redirect"):
                yield page_id, title, _strip_wikitext(text)
        elif tag == "page":
            title = page_id = ns = None
            el.clear()


def _cirrus_worker(paths: list[str], queue) -> None:
    """Parse CirrusSearch dump files (bulk format: an index line, then the document) and put batches of article tuples on the queue."""
    import json

    batch = []
    for path in paths:
        with bz2.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith('{"index"'):
                    continue
                d = json.loads(line)
                if d.get("namespace") != 0 or not d.get("title"):
                    continue
                if d.get("page_type") == "redirect":  # a separate record per redirect: an alias of its target
                    target = (d.get("redirect_target") or {}).get("title")
                    if target:
                        batch.append(("R", d["title"], target))
                    continue
                coord = next(((c["coord"]["lat"], c["coord"]["lon"], c.get("type") or "") for c in d.get("coordinates") or ()
                              if c.get("primary") and c.get("globe") == "earth" and c.get("coord")), None)
                text = d.get("text") or ""
                redirects = [r["title"] for r in d.get("redirect") or () if r.get("namespace") == 0 and r.get("title")]
                batch.append(("A", int(d["page_id"]), d["title"], text[:MAX_CHARS], coord, d.get("popularity_score"), redirects, len(text)))
                if len(batch) >= 2000:
                    queue.put(batch)
                    batch = []
    queue.put(batch)
    queue.put(None)


def _cirrus_batches(lang: str, workers: int = 3) -> Iterator[list[tuple]]:
    import multiprocessing as mp

    files = sorted(str(p) for p in (WIKI / "cirrus" / f"{lang}wiki").glob("*.json.bz2"))
    if not files:
        raise SystemExit(f"no CirrusSearch dump files for {lang}")
    queue = mp.get_context("fork").Queue(maxsize=16)
    procs = [mp.get_context("fork").Process(target=_cirrus_worker, args=(files[k::workers], queue)) for k in range(workers)]
    for proc in procs:
        proc.start()
    done = 0
    while done < workers:
        batch = queue.get()
        if batch is None:
            done += 1
        elif batch:
            yield batch
    for proc in procs:
        proc.join()
        if proc.exitcode:
            raise SystemExit(f"a CirrusSearch reader failed (exit {proc.exitcode})")


def build(lang: str, source: str = "hf") -> None:
    """source 'hf': wikimedia/wikipedia 20231101 parquet + geo_tags; 'cirrus': Wikimedia's CirrusSearch index dump (current text,
    coordinates, popularity, redirects; data/wikipedia/cirrus/<lang>wiki/). The English HF dump lacks many of the best-known articles
    (Paris, London, Tokyo, Eiffel Tower, ...), so English uses cirrus. Writes to a temporary file and swaps it in at the end."""
    import duckdb

    path = ROOT / f"wiki_{lang}.db"
    tmp = path.with_suffix(".db.tmp")
    con = _connect_new(tmp)
    con.executescript("""
        CREATE TABLE pages (rowid INTEGER PRIMARY KEY, source TEXT, page_id INTEGER, title TEXT, url TEXT, lat REAL, lon REAL, type TEXT,
                            chars INTEGER, pop REAL);
        CREATE TABLE aliases (title TEXT, target TEXT);
        CREATE TABLE passages (rowid INTEGER PRIMARY KEY, page INTEGER, n INTEGER, body BLOB);
        CREATE VIRTUAL TABLE fts USING fts5 (title, body, content = '', tokenize = 'unicode61 remove_diacritics 2');""")
    geo = _geo_tags(lang) if source == "hf" else {}
    page_row = passage_row = with_coords = 0

    def add(source: str, page_id: int, title: str, url: str, text: str, coord: tuple[float, float, str] | None, chars: int | None = None,
            pop: float | None = None, redirects: Sequence[str] = ()) -> None:
        nonlocal page_row, passage_row, with_coords
        pieces = _passages(text)
        if not pieces:
            return
        page_row += 1
        lat, lon, kind = coord if coord else (None, None, None)
        with_coords += coord is not None
        pages.append((page_row, source, page_id, title, url, lat, lon, kind, chars or len(text), pop))
        aliases.extend((r, title) for r in redirects)
        for k, body in enumerate(pieces):
            passage_row += 1
            passages.append((passage_row, page_row, k, zlib.compress(body.encode("utf-8"), 6)))
            fts.append((passage_row, title, body))

    def flush() -> None:
        con.executemany("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pages)
        con.executemany("INSERT INTO aliases VALUES (?, ?)", aliases)
        con.executemany("INSERT INTO passages VALUES (?, ?, ?, ?)", passages)
        con.executemany("INSERT INTO fts (rowid, title, body) VALUES (?, ?, ?)", fts)
        pages.clear(), passages.clear(), fts.clear(), aliases.clear()

    pages, passages, fts, aliases = [], [], [], []
    count = 0
    if source == "cirrus":
        for batch in _cirrus_batches(lang):
            before = count
            for item in batch:
                if item[0] == "R":
                    aliases.append(item[1:])
                    continue
                _, page_id, title, text, coord, pop, redirects, chars = item
                add("wikipedia", page_id, title, f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}", text, coord, chars, pop,
                    redirects)
                count += 1
            if len(passages) >= 50_000:
                flush()
            if count // 500_000 != before // 500_000:
                print(f"  {lang}: {count:,} articles", flush=True)
    else:
        files = sorted((WIKI / "hf" / f"20231101.{lang}").glob("*.parquet"))
        duck = duckdb.connect()
        duck.execute("SET memory_limit = '1GB'; SET threads = 2")  # DuckDB sizes itself to the machine, not the job's cgroup (OOM at 4G)
        cursor = duck.execute(f"SELECT id, url, title, text FROM read_parquet({[str(f) for f in files]})")
        while batch := cursor.fetchmany(20_000):
            for page_id, url, title, text in batch:
                add("wikipedia", int(page_id), title, url, text, geo.get(int(page_id)))
            count += len(batch)
            flush()
            if count % 500_000 < 20_000:
                print(f"  {lang}: {count:,} articles", flush=True)
        duck.close()
    voyage = 0
    for page_id, title, text in _voyage_pages(lang):
        add("wikivoyage", page_id, title, f"https://{lang}.wikivoyage.org/wiki/{quote(title.replace(' ', '_'))}", text, None)
        voyage += 1
        if len(passages) >= 50_000:
            flush()
    flush()
    con.execute("CREATE INDEX passages_page ON passages (page, n)")
    con.execute("CREATE INDEX pages_title ON pages (title COLLATE NOCASE)")
    con.execute("CREATE INDEX aliases_title ON aliases (title COLLATE NOCASE)")
    con.execute("INSERT INTO fts (fts) VALUES ('optimize')")
    con.commit()
    con.close()
    os.replace(tmp, path)
    print(f"{lang} ({source}): {count:,} articles ({with_coords:,} with coordinates), {voyage:,} wikivoyage pages, {passage_row:,} passages -> "
          f"{path} ({path.stat().st_size / 1e9:.1f} GB)", flush=True)


def stage() -> None:
    NVME.mkdir(parents=True, exist_ok=True)
    for p in sorted(ROOT.glob("wiki_*.db")):
        shutil.copyfile(p, NVME / p.name)
        print(f"{p} -> {NVME / p.name}", flush=True)


def _db_paths() -> list[Path]:
    if os.environ.get("WEB_SEARCH_DIR"):
        return sorted(Path(os.environ["WEB_SEARCH_DIR"]).glob("wiki_*.db"))
    out = []
    for p in sorted(ROOT.glob("wiki_*.db")):
        fast = NVME / p.name
        out.append(fast if fast.exists() and fast.stat().st_size == p.stat().st_size else p)
    return out


@dataclass
class Hit:
    title: str
    url: str
    snippet: str
    score: float  # bm25 (lower is better)
    lang: str
    source: str
    lat: float | None
    lon: float | None
    lead: str  # first passage, for the knowledge panel


def _rank(bm25: float, key: str, title: str, chars: int, pop: float | None = None, alias: bool = False) -> float:
    """Lower is better: bm25 plus stand-ins for Google's notability: the page is named for the query (or a redirect to it is), and its
    page views (CirrusSearch popularity, English) or else its length."""
    name = norm(re.sub(r"\s*\(.*\)$", "", title))  # 'Nyhavn (film)' -> 'nyhavn'
    bonus = TITLE_EXACT if name == key or alias else TITLE_PREFIX if name.startswith(key) else 0.0
    if pop:
        return bm25 - bonus - POP_WEIGHT * min(max(math.log10(pop) + 10, 0.0), 8.0)
    return bm25 - bonus - LENGTH_WEIGHT * math.log1p(chars / 1000)


def _snippet(body: str, words: list[str]) -> str:
    folded = norm(body)
    at = min((i for w in words if (i := folded.find(w)) >= 0), default=0)
    ratio = len(body) / max(len(folded), 1)  # norm() shortens text a little; map back roughly
    start = max(0, int(at * ratio) - SNIPPET // 3)
    text = re.sub(r"\s+", " ", body[start:start + SNIPPET]).strip()
    return ("…" if start else "") + text + ("…" if start + SNIPPET < len(body) else "")


class WebSearch:
    def __init__(self, langs: Sequence[str] | None = None) -> None:
        paths = [p for p in _db_paths() if langs is None or p.stem.split("_", 1)[1] in langs]
        self.dbs = [(p.stem.split("_", 1)[1], sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)) for p in paths]
        self.columns = {}  # language -> page columns and tables (older builds lack pop / aliases)
        for lang, con in self.dbs:
            cols = {r[1] for r in con.execute("PRAGMA table_info(pages)")}
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            self.columns[lang] = cols | (tables & {"aliases"})
        self.pool = ThreadPoolExecutor(max_workers=max(len(self.dbs), 1))

    def _one(self, lang: str, con: sqlite3.Connection, match: str, words: list[str], query: str) -> list[Hit]:
        """At most CAP matches are read (in index order) and ranked here: a full ORDER BY rank scores every match, ~1 s for a word like
        '2006'. Under CAP matches this is the exact bm25 order; above it, a sample. Pages titled exactly as the query are always added."""
        key = " ".join(words)
        rows = con.execute("SELECT rowid, bm25(fts, 5.0, 1.0) FROM fts WHERE fts MATCH ? LIMIT ?", (match, CAP)).fetchall()
        rows.sort(key=lambda r: r[1])
        best = rows[0][1] if rows else -10.0
        titled = con.execute("SELECT rowid FROM pages WHERE title = ? COLLATE NOCASE LIMIT 5", (query.strip(),)).fetchall()
        aliased = set()
        if self.columns[lang] >= {"aliases"}:
            aliased = {r[0] for r in con.execute("SELECT p.rowid FROM aliases a JOIN pages p ON p.title = a.target COLLATE NOCASE WHERE a.title = ? COLLATE NOCASE "
                                                 "LIMIT 5", (query.strip(),))}
        candidates: dict[int, tuple[int, float]] = {}  # page -> (passage, bm25)
        for page in [r[0] for r in titled] + sorted(aliased):
            lead = con.execute("SELECT rowid FROM passages WHERE page = ? AND n = 0", (page,)).fetchone()
            if lead:
                candidates[page] = (lead[0], best)
        for passage, score in rows[:RESOLVE]:
            page = con.execute("SELECT page FROM passages WHERE rowid = ?", (passage,)).fetchone()[0]
            if page not in candidates:
                candidates[page] = (passage, score)
        hits = []
        pop = "pop" if "pop" in self.columns[lang] else "NULL"
        for page, (passage, score) in candidates.items():
            source, title, url, lat, lon, chars, views = con.execute(
                f"SELECT source, title, url, lat, lon, chars, {pop} FROM pages WHERE rowid = ?", (page,)).fetchone()
            hits.append((_rank(score, key, title, chars, views, page in aliased), page, passage, source, title, url, lat, lon))
        hits.sort()
        out = []
        for rank, page, passage, source, title, url, lat, lon in hits[:PER_DB]:
            n, body = con.execute("SELECT n, body FROM passages WHERE rowid = ?", (passage,)).fetchone()
            text = zlib.decompress(body).decode("utf-8")
            lead = text if n == 0 else zlib.decompress(con.execute("SELECT body FROM passages WHERE page = ? AND n = 0", (page,)).fetchone()[0]
                                                         ).decode("utf-8")
            out.append(Hit(title, url, _snippet(text, words), rank, lang, source, lat, lon, lead))
        return out

    def query(self, query: str, top: int = TOP) -> tuple[list[Hit], bool]:
        """Hits, and whether the words only matched in any order (no phrase match anywhere)."""
        words = norm(query).split()
        if not words:
            return [], False
        for match, loose in (('"' + " ".join(words) + '"', False), (" AND ".join(f'"{w}"' for w in words), True)):
            hits = [h for part in self.pool.map(lambda d: self._one(d[0], d[1], match, words, query), self.dbs) for h in part]
            if hits:
                for h in hits:
                    h.score -= LANG_BONUS.get(h.lang, 0.0)
                hits.sort(key=lambda h: h.score)
                first_lang, unique = {}, []
                for h in hits:  # drop translations: a title already shown from another language (same-language namesakes stay)
                    name = norm(re.sub(r"\s*\(.*\)$", "", h.title))
                    if first_lang.setdefault(name, h.lang) == h.lang:
                        unique.append(h)
                return unique[:top], loose
        return [], False


def format_hits(query: str, hits: list[Hit], loose: bool) -> str:
    lines = [f'web_search("{query}"): {len(hits)} result{"s" * (len(hits) != 1)}' + (" (words in any order)" if loose else "")]
    if hits and hits[0].lat is not None and norm(hits[0].title) == norm(query):
        lead = re.sub(r"\s+", " ", hits[0].lead.strip())
        first = next((m.group(0) for m in re.finditer(r".{60,}?[.!?。](?=\s|$)", lead)), lead[:300])
        lines.append(f"  [panel] {hits[0].title} ({hits[0].lat:.4f}, {hits[0].lon:.4f}): {first[:300]}")
    for i, h in enumerate(hits, 1):
        lines.append(f"  {i}. {h.title} — {h.url}\n     {h.snippet}")
    return "\n".join(lines)


def report(tag: str) -> None:
    """Coverage on Pinpoint's mistakes against the live English-Wikipedia search (web_geocode.py, same clues and metric): a clue's first
    result with a coordinate (of the top 5) within 25 km of the truth; offline over all languages and English only; shuffled-clue luck."""
    import json

    import numpy as np

    from .web_geocode import OUT, PHOTO_SETS, _mistakes
    from .text_search import haversine_km

    mistakes, phrases = _mistakes(tag)
    ids = [e["image_id"] for e in mistakes]
    perm = np.random.default_rng(0).permutation(len(ids))
    perm = np.where(perm == np.arange(len(ids)), (perm + 1) % len(ids), perm)
    shuffled = {ids[i]: phrases[ids[perm[i]]] for i in range(len(ids))}
    n_photos = len(json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8")))
    live = {}
    for line in (OUT / f"web_wikipedia_{tag}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        live[row["query"]] = [r[:2] for r in row["results"]]
    clues = sorted({q for i in ids for q in phrases[i]} | {q for i in ids for q in shuffled[i]})
    tables = {"live en Wikipedia": live}
    for name, langs in (("offline all languages", None), ("offline en only", ["en"])):
        search, table = WebSearch(langs), {}
        for q in clues:
            hits, _ = search.query(q, top=5)
            table[q] = [[h.lat, h.lon] for h in hits if h.lat is not None]
        tables[name] = table
    print(f"{tag}: {len(mistakes)} of Pinpoint's mistakes have 4B clues ({len(clues)} distinct clues incl. shuffled); {n_photos} photos")
    reached = {}
    for name, table in tables.items():
        for label, sets in (("real", {i: phrases[i] for i in ids}), ("shuffled", shuffled)):
            hit = set()
            for e in mistakes:
                firsts = [r[0] for r in (table.get(q) or [] for q in sets[e["image_id"]]) if r]
                if firsts and (haversine_km(*e["truth"], np.asarray([f[0] for f in firsts]), np.asarray([f[1] for f in firsts])) < 25).any():
                    hit.add(e["image_id"])
            reached[name, label] = hit
            print(f"  {name:22s} {label:8s} first result < 25 km: {len(hit):4d} (+{100 * len(hit) / n_photos:.1f} pts)")
    live_hit = reached["live en Wikipedia", "real"]
    for name in ("offline all languages", "offline en only"):
        mine = reached[name, "real"]
        print(f"  {name}: of live's {len(live_hit)}, also reached {len(mine & live_hit)}; reached that live misses: {len(mine - live_hit)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--langs", nargs="+", default=list(WIKI_LANGS))
    b.add_argument("--source", choices=["hf", "cirrus"], default="hf")
    sub.add_parser("stage")
    q = sub.add_parser("query")
    q.add_argument("text")
    r = sub.add_parser("report")
    r.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        for lang in args.langs:
            build(lang, args.source)
    elif args.cmd == "stage":
        stage()
    elif args.cmd == "report":
        report(args.tag)
    else:
        print(format_hits(args.text, *WebSearch().query(args.text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
