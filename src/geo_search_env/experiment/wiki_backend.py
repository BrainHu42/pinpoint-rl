# Offline geotagged-Wikipedia search backend (BM25 + dense), and a go/no-go probe of what it adds beyond the shown candidates.
# Usage: ~/.venvs/sft/bin/python -m geo_search_env.experiment.wiki_backend {build,embed,probe}   (PYTHONPATH=src)

"""English Wikipedia articles with a primary Earth coordinate, searchable by text.

build: join the geo_tags dump (coordinates, Sep 2026) with wikimedia/wikipedia 20231101.en (text) on page id; write a
       SQLite table of (title, lat, lon, type, lead) with an FTS5 index over title + lead.
embed: bge-base-en-v1.5 embeddings of "title. lead" (CLS, normalized, fp16) for dense search.
probe: run the dev run's generated queries (artifacts/query_evidence/<tag>) through both indexes and measure coverage
       beyond the shown candidates, next to the SigLIP2 evidence of the same queries.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Sequence

import numpy as np


WIKI = Path("/data/pinpoint/wikipedia")
GEO_TAGS = WIKI / "enwiki-latest-geo_tags.sql.gz"
TEXT = WIKI / "hf" / "20231101.en"
DB = WIKI / "enwiki_geo.sqlite"
EMBEDDINGS = WIKI / "enwiki_geo_bge-base.f16.npy"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LEAD_CHARS = 1500
EMBED_TOKENS = 256
EVIDENCE_ROOT = Path("artifacts/query_evidence")
THRESHOLDS = (1.0, 25.0)
ROW = re.compile(r"^\((\d+),(\d+),'([^']*)',(\d),(-?[\d.]+|NULL),(-?[\d.]+|NULL),(-?\d+|NULL),(NULL|'(?:[^'\\]|\\.)*')")


def _coordinates() -> dict[int, tuple[float, float, str | None]]:
    """Primary Earth coordinate (lat, lon, type) per page id."""

    out: dict[int, tuple[float, float, str | None]] = {}
    with gzip.open(GEO_TAGS, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            m = ROW.match(line)
            if not m or m.group(3) != "earth" or m.group(4) != "1" or "NULL" in (m.group(5), m.group(6)):
                continue
            lat, lon = float(m.group(5)), float(m.group(6))
            if abs(lat) <= 90 and abs(lon) <= 180:
                out[int(m.group(2))] = (lat, lon, None if m.group(8) == "NULL" else m.group(8)[1:-1] or None)
    return out


def build() -> None:
    import pyarrow.parquet as pq

    coords = _coordinates()
    print(f"{len(coords)} pages with a primary Earth coordinate", flush=True)
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, page_id INTEGER, title TEXT, lat REAL, lon REAL, type TEXT, lead TEXT)")
    kept = 0
    for shard in sorted(TEXT.glob("*.parquet")):
        table = pq.read_table(shard, columns=["id", "title", "text"]).to_pydict()
        rows = []
        for page, title, text in zip(table["id"], table["title"], table["text"]):
            hit = coords.get(int(page))
            if hit:
                rows.append((int(page), title, hit[0], hit[1], hit[2], text[:LEAD_CHARS]))
        db.executemany("INSERT INTO articles (page_id, title, lat, lon, type, lead) VALUES (?, ?, ?, ?, ?, ?)", rows)
        kept += len(rows)
        print(f"  {shard.name}: {kept} articles", flush=True)
    db.execute("CREATE VIRTUAL TABLE search USING fts5(title, lead, content='articles', content_rowid='id', tokenize='unicode61 remove_diacritics 2')")
    db.execute("INSERT INTO search (rowid, title, lead) SELECT id, title, lead FROM articles")
    db.commit()
    print(f"{kept} geotagged articles indexed in {DB}")


def _encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.float16).cuda().eval()

    def encode(texts: list[str]) -> np.ndarray:
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=EMBED_TOKENS, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            cls = model(**tokens).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls.float(), dim=-1).half().cpu().numpy()

    return encode


def embed(*, batch: int = 512) -> None:
    db = sqlite3.connect(DB)
    count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    encode = _encoder()
    out = np.lib.format.open_memmap(EMBEDDINGS, mode="w+", dtype=np.float16, shape=(count, 768))
    # Rows are written in id order (ids are 1..count), so row i of the matrix is article id i + 1.
    cursor = db.execute("SELECT id, title, lead FROM articles ORDER BY id")
    done = 0
    while rows := cursor.fetchmany(batch):
        out[done : done + len(rows)] = encode([f"{title}. {lead}" for _, title, lead in rows])
        done += len(rows)
        if done % (batch * 200) < batch:
            print(f"  embedded {done}/{count}", flush=True)
    out.flush()
    print(f"{done} embeddings in {EMBEDDINGS}")


class Wiki:
    """Top-k geotagged articles for a text query, by BM25 (title weighted 10x) or dense similarity."""

    def __init__(self, dense: bool = True) -> None:
        import torch

        self.local = threading.local()
        self.encode = _encoder() if dense else None
        self.matrix = torch.as_tensor(np.load(EMBEDDINGS, mmap_mode="r")[:], device="cuda") if dense else None

    @property
    def db(self) -> sqlite3.Connection:
        """One read-only connection per thread, so BM25 lookups can run in parallel."""

        if not hasattr(self.local, "db"):
            self.local.db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        return self.local.db

    def _rows(self, ids: Sequence[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        found = {r[0]: r for r in self.db.execute(f"SELECT id, title, lat, lon, type FROM articles WHERE id IN ({','.join('?' * len(ids))})", list(ids))}
        return [{"id": i, "title": found[i][1], "lat": found[i][2], "lon": found[i][3], "type": found[i][4]} for i in ids if i in found]

    def bm25(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        words = re.findall(r"\w+", query.lower())
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
        ids = [r[0] for r in self.db.execute("SELECT rowid FROM search WHERE search MATCH ? ORDER BY bm25(search, 10.0, 1.0) LIMIT ?", (match, k))]
        return self._rows(ids)

    def bm25_many(self, queries: Sequence[str], k: int = 10, workers: int = 16) -> list[list[dict[str, Any]]]:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(lambda q: self.bm25(q, k), queries))

    def dense(self, queries: list[str], k: int = 10) -> list[list[dict[str, Any]]]:
        import torch

        out = []
        for start in range(0, len(queries), 256):
            q = torch.as_tensor(self.encode([QUERY_PREFIX + t for t in queries[start : start + 256]]), device="cuda")
            top = torch.topk(q @ self.matrix.T, k, dim=1).indices.cpu().numpy() + 1  # matrix row i is article id i + 1
            out += [self._rows([int(i) for i in row]) for row in top]
        return out


def _km(points: np.ndarray, lat: float, lon: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lat1, lon1, lat2, lon2 = math.radians(lat), math.radians(lon), np.radians(points[:, 0]), np.radians(points[:, 1])
    h = np.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def probe(tag: str, *, per_query: int = 1) -> None:
    """Coverage of the truth by the top `per_query` article(s) of each generated query, vs the shown candidates and SigLIP2."""

    root = EVIDENCE_ROOT / tag
    dev = json.loads((root / "dev.json").read_text(encoding="utf-8"))
    generated = json.loads((root / "generated.json").read_text(encoding="utf-8"))
    retrieved = json.loads((root / "retrieved.json").read_text(encoding="utf-8"))
    wiki = Wiki()
    texts = [(m, arm, q) for m, e in enumerate(dev) for arm in ("visual", "geo", "caption") for q in generated[e["image_id"]][arm]["queries"]]
    dense = wiki.dense([q for _, _, q in texts], k=max(3, per_query))
    bm25 = wiki.bm25_many([q for _, _, q in texts], k=max(3, per_query))
    hits: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for (m, arm, q), d, b in zip(texts, dense, bm25):
        k = 3 if arm == "caption" else per_query  # caption: one query, so three results, as in the SigLIP arm
        hits.setdefault(("dense", arm, m), []).extend(d[:k])
        hits.setdefault(("bm25", arm, m), []).extend(b[:k])
    report: dict[str, Any] = {"n": len(dev), "per_query": per_query}
    shown_km = np.asarray([_km([c[1:] for c in e["candidates"]], *e["truth"]).min() for e in dev])
    reranker_km = np.asarray([_km([e["candidates"][0][1:]], *e["truth"])[0] for e in dev])  # candidates are in reranker order
    for name, d in (("reranker top-1", reranker_km), ("shown", shown_km)):
        for t in THRESHOLDS:
            report[f"{name} <{int(t)} km"] = float((d < t).mean())
    print(f"n={len(dev)}, top-{per_query} per query   reranker wrong, a hit right   reranker+hits   shown+hits   (all <25 km; NEW = no shown near)")
    print(f"  {'reranker top-1':26s} {'':28s} {report['reranker top-1 <25 km']:9.1%}")
    print(f"  {'shown candidates (oracle)':26s} {'':28s} {'':9s} {report['shown <25 km']:12.1%}")
    for backend in ("siglip", "bm25", "dense"):
        for arm in ("visual", "geo", "caption"):
            near = []
            for m, e in enumerate(dev):
                pts = ([[c["lat"], c["lon"]] for c in retrieved[e["image_id"]][arm]] if backend == "siglip"
                       else [[h["lat"], h["lon"]] for h in hits.get((backend, arm, m), [])])
                near.append(_km(pts, *e["truth"]).min() if pts else np.inf)
            near = np.asarray(near)
            row = {f"hit <{int(t)} km": float((near < t).mean()) for t in THRESHOLDS}
            row |= {f"shown+hits <{int(t)} km": float((np.minimum(near, shown_km) < t).mean()) for t in THRESHOLDS}
            row["new <25 km"] = float(((near < 25) & (shown_km >= 25)).mean())
            row["beats reranker <25 km"] = float(((near < 25) & (reranker_km >= 25)).mean())
            row["reranker+hits <25 km"] = float((np.minimum(near, reranker_km) < 25).mean())
            report[f"{backend}/{arm}"] = row
            print(f"  {backend + ' / ' + arm:26s} {row['beats reranker <25 km']:16.1%} {'':11s} {row['reranker+hits <25 km']:9.1%} "
                  f"{row['shown+hits <25 km']:12.1%}   NEW {row['new <25 km']:.1%}")
    (root / f"wiki_probe_top{per_query}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    samples = []
    for m, e in enumerate(dev[:40]):
        samples.append({"image_id": e["image_id"], "truth": e["truth"], "geo": generated[e["image_id"]]["geo"]["queries"],
                        "bm25": [(h["title"], round(float(_km([[h["lat"], h["lon"]]], *e["truth"])[0]))) for h in hits.get(("bm25", "geo", m), [])],
                        "dense": [(h["title"], round(float(_km([[h["lat"], h["lon"]]], *e["truth"])[0]))) for h in hits.get(("dense", "geo", m), [])],
                        "visual": generated[e["image_id"]]["visual"]["queries"],
                        "dense visual": [(h["title"], round(float(_km([[h["lat"], h["lon"]]], *e["truth"])[0]))) for h in hits.get(("dense", "visual", m), [])]})
    (root / f"wiki_probe_samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("build", "embed", "probe"))
    parser.add_argument("--tag", default="dev", help="probe: query_evidence run to read queries from")
    parser.add_argument("--per-query", type=int, default=1, help="probe: articles kept per query")
    args = parser.parse_args(argv)
    if args.node == "build":
        build()
    elif args.node == "embed":
        embed()
    else:
        probe(args.tag, per_query=args.per_query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
