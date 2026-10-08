# Landmark / object names looked up in YFCC100M's geotagged Flickr titles and tags (lesson 62 follow-up): a name the models produce for a photo
# ("City Island Lobster House", "Bagan", "Wittenham Clumps") is searched among ~48M geotagged photos' titles + tags; a rare name whose matches
# concentrate in one place proposes that place; it replaces Pinpoint's top-1 when Pinpoint's own confidence is low.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.yfcc_names extract            (SQLite -> data/yfcc100m/geo.npz; ~20 min)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.yfcc_names phrases --tag dev3k
#        PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.yfcc_names match --tag dev3k   (needs pyahocorasick)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.yfcc_names report --tag dev3k

"""extract: one photoid range of the geotagged rows of data/yfcc100m/yfcc100m_dataset.sql; merge -> data/yfcc100m/geo.npz: photo id, user (code +
         vocab), lat, lon, GPS accuracy (1-16), year taken, video flag, image URL parts (farm, server, secret), and three normalised text blobs with
         offsets: 'title tags' (blob / offsets), description (first 300 chars), machine tags (first 300 chars).
phrases: per photo of a set, the names the models produced: the 27B / 9B landmark names (observe.py landmarks) and the 9B's read text and place
         names (observe.py extract) -> artifacts/agent/yfcc_phrases_{tag}.json.
match:   all phrases (whole-word) in every title + tags -> artifacts/agent/yfcc_matches_{tag}.npz: per phrase, the number of matching photos and up to
         MAX_ROWS matching rows.
report:  per photo, per phrase: matches excluding the photo itself and its photographer; the densest 25 km cluster and its share of the matches;
         the best phrase proposes its cluster's centre. Switch rule: proposal >= 25 km from Pinpoint's top-1, cluster share / size thresholds, and
         Pinpoint's top-1 score below a quantile; change in top-1 < 25 km vs Pinpoint, fixes and breaks (dev set)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata
from typing import Sequence
from urllib.parse import unquote_plus

import numpy as np

YFCC = Path("data/yfcc100m")
OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
MIN_CHARS, MAX_ROWS = 5, 20_000


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", text).strip()


ID_RANGE = (28_605, 14_068_203_844)  # min / max photoid (the table's integer primary key)


TEXTS = ("text", "desc", "mtags")  # title + user tags; description (first DESC_CHARS); machine tags (first DESC_CHARS)
DESC_CHARS = 300


def extract(part: int, parts: int = 16) -> None:
    """One photoid range of the geotagged rows -> data/yfcc100m/geo_part{part}.npz (run all parts in parallel, then merge)."""

    from array import array
    import sqlite3

    con = sqlite3.connect(f"file:{YFCC / 'yfcc100m_dataset.sql'}?mode=ro", uri=True)
    con.execute("PRAGMA mmap_size = 80000000000")  # 1 KB pages: memory-map so the OS reads ahead (plain reads ran at ~3 MB/s)
    lo = ID_RANGE[0] + (ID_RANGE[1] - ID_RANGE[0]) * part // parts
    hi = ID_RANGE[0] + (ID_RANGE[1] - ID_RANGE[0]) * (part + 1) // parts
    query = ("SELECT photoid, uid, latitude, longitude, title, usertags, description, machinetags, accuracy, datetaken, marker, farmid, serverid, secret "
             f"FROM yfcc100m_dataset WHERE latitude IS NOT NULL AND latitude != '' AND photoid >= {lo} AND photoid < {hi}")
    ids, uids, lat, lon = array("q"), array("i"), array("f"), array("f")
    accuracy, year, video, farm, server = array("b"), array("h"), array("b"), array("h"), array("i")
    secrets, vocab = [], {}
    blobs = {k: bytearray() for k in TEXTS}
    offsets = {k: array("q", [0]) for k in TEXTS}
    for n, (pid, uid, la, lon_, title, tags, desc, mtags, acc, taken, marker, farm_, server_, secret) in enumerate(con.execute(query)):
        try:
            la, lon_ = float(la), float(lon_)
        except (TypeError, ValueError):
            continue
        texts = {"text": norm(unquote_plus(title or "") + " " + unquote_plus(tags or "").replace(",", " ")),
                 "desc": norm(unquote_plus(desc or "")[:DESC_CHARS * 2])[:DESC_CHARS],
                 "mtags": unquote_plus(mtags or "").casefold()[:DESC_CHARS].replace("\n", " ")}
        for k in TEXTS:
            blobs[k] += texts[k].encode("utf-8") + b"\n"
            offsets[k].append(len(blobs[k]))
        ids.append(int(pid)); uids.append(vocab.setdefault(uid, len(vocab))); lat.append(la); lon.append(lon_)
        accuracy.append(int(acc or 0)); year.append(int(str(taken)[:4]) if taken and str(taken)[:4].isdigit() else 0)
        video.append(int(marker or 0)); farm.append(int(farm_ or 0)); server.append(int(server_ or 0)); secrets.append(str(secret or ""))
        if n % 1_000_000 == 0:
            print(f"  part {part}: {n} geotagged rows", flush=True)
    np.savez(YFCC / f"geo_part{part:02d}.npz", photo_id=np.frombuffer(ids, np.int64), uid=np.frombuffer(uids, np.int32),
             lat=np.frombuffer(lat, np.float32), lon=np.frombuffer(lon, np.float32), accuracy=np.frombuffer(accuracy, np.int8),
             year=np.frombuffer(year, np.int16), video=np.frombuffer(video, np.int8), farm=np.frombuffer(farm, np.int16),
             server=np.frombuffer(server, np.int32), secret=np.asarray(secrets, dtype="S16"), vocab=np.asarray(sorted(vocab, key=vocab.get)),
             **{f"{k}_blob": np.frombuffer(bytes(blobs[k]), np.uint8) for k in TEXTS}, **{f"{k}_offsets": np.frombuffer(offsets[k], np.int64) for k in TEXTS})
    print(f"part {part}: {len(ids)} geotagged photos -> {YFCC / f'geo_part{part:02d}.npz'}", flush=True)


def merge(parts: int = 16) -> None:
    """The parts of a parallel extract -> geo.npz + uid_vocab.json (one user vocabulary). The title + tags blob keeps the names blob / offsets."""

    vocab: dict[str, int] = {}
    arrays = ("photo_id", "uid", "lat", "lon", "accuracy", "year", "video", "farm", "server", "secret")
    cols = {k: [] for k in arrays}
    blobs = {k: [] for k in TEXTS}
    offsets = {k: [np.zeros(1, np.int64)] for k in TEXTS}
    totals = {k: 0 for k in TEXTS}
    for part in range(parts):
        z = np.load(YFCC / f"geo_part{part:02d}.npz")
        remap = np.asarray([vocab.setdefault(u, len(vocab)) for u in z["vocab"]], np.int32)
        for k in arrays:
            cols[k].append(remap[z["uid"]] if k == "uid" and len(z["uid"]) else z[k])
        for k in TEXTS:
            blobs[k].append(z[f"{k}_blob"]); offsets[k].append(z[f"{k}_offsets"][1:] + totals[k]); totals[k] += len(z[f"{k}_blob"])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out |= {("blob" if k == "text" else f"{k}_blob"): np.concatenate(blobs[k]) for k in TEXTS}
    out |= {("offsets" if k == "text" else f"{k}_offsets"): np.concatenate(offsets[k]) for k in TEXTS}
    np.savez(YFCC / "geo.npz", **out)
    (YFCC / "uid_vocab.json").write_text(json.dumps(sorted(vocab, key=vocab.get)) + "\n", encoding="utf-8")
    print(f"merged {parts} parts: {len(out['photo_id'])} geotagged photos, {len(vocab)} users; text " + ", ".join(f"{k} {totals[k] / 1e9:.2f} GB" for k in TEXTS))


def phrases(tag: str) -> None:
    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    sources = {}
    for name in ("27b", "9b"):
        path = OUT / f"landmark_{tag}_{name}.json"
        if path.exists():
            sources[f"landmark {name}"] = {k: [v["landmark"]] for k, v in json.loads(path.read_text(encoding="utf-8")).items() if v and v.get("landmark")}
    searches = OUT / f"queries_{tag}_9b.json"  # observe.py queries: up to 5 searchable names per photo
    if searches.exists():
        sources["queries 9b"] = {k: v or [] for k, v in json.loads(searches.read_text(encoding="utf-8")).items()}
    clues = OUT / f"observe_{tag}_9bcrop.json"  # observe.py crops: full-image clues + text read on 4 tiles
    if not clues.exists():
        clues = OUT / f"observe_{tag}_9b.json"
    if clues.exists():
        sources["text 9b"] = {k: ((v.get("parsed") or {}).get("text") or []) + ((v.get("parsed") or {}).get("place_names") or [])
                              for k, v in json.loads(clues.read_text(encoding="utf-8")).items()}
    out = {}
    for e in photos:
        found = {}
        for src, table in sources.items():
            for raw in table.get(e["image_id"], []):
                p = norm(str(raw))
                if len(p) >= MIN_CHARS and not p.replace(" ", "").isdigit():
                    found.setdefault(p, src)
        if found:
            out[e["image_id"]] = found
    (OUT / f"yfcc_phrases_{tag}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{tag}: phrases for {len(out)} of {len(photos)} photos, {len({p for v in out.values() for p in v})} distinct -> {OUT / f'yfcc_phrases_{tag}.json'}")


def match(tag: str) -> None:
    import ahocorasick

    table = json.loads((OUT / f"yfcc_phrases_{tag}.json").read_text(encoding="utf-8"))
    phrase_list = sorted({p for v in table.values() for p in v})
    auto = ahocorasick.Automaton()
    for i, p in enumerate(phrase_list):
        auto.add_word(p, (i, len(p)))
    auto.make_automaton()
    z = np.load(YFCC / "geo.npz")
    blob, offsets = z["blob"].tobytes(), z["offsets"]
    counts = np.zeros(len(phrase_list), np.int64)
    rows: list[list[int]] = [[] for _ in phrase_list]
    for d in range(len(offsets) - 1):
        doc = " " + blob[offsets[d]:offsets[d + 1] - 1].decode("utf-8", "replace") + " "
        seen = set()
        for end, (i, length) in auto.iter(doc):
            if i in seen or doc[end - length] != " " or doc[end + 1] != " ":
                continue  # whole words only, once per photo
            seen.add(i)
            counts[i] += 1
            if len(rows[i]) < MAX_ROWS:
                rows[i].append(d)
        if d % 5_000_000 == 0:
            print(f"  {d} / {len(offsets) - 1} photos", flush=True)
    flat = np.asarray([r for rs in rows for r in rs], np.int64)
    starts = np.cumsum([0] + [len(rs) for rs in rows])
    np.savez(OUT / f"yfcc_matches_{tag}.npz", phrases=np.asarray(phrase_list), counts=counts, rows=flat, starts=starts)
    print(f"{tag}: {len(phrase_list)} phrases, {np.count_nonzero(counts)} found -> {OUT / f'yfcc_matches_{tag}.npz'}", flush=True)


def _mp16_authors(image_ids: set[str]) -> dict[str, str]:
    import csv

    from .strategy_search import MP16_CSV

    out = {}
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        i, a = header.index("IMG_ID"), header.index("AUTHOR")
        for row in reader:
            if row[i] in image_ids:
                out[row[i]] = row[a]
    return out


def report(tag: str) -> None:
    from .pinpoint_reference import load_pools
    from .strategy_search import _bootstrap, _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    table = json.loads((OUT / f"yfcc_phrases_{tag}.json").read_text(encoding="utf-8"))
    m = np.load(OUT / f"yfcc_matches_{tag}.npz")
    index = {p: i for i, p in enumerate(m["phrases"])}
    z = np.load(YFCC / "geo.npz")
    photo_id, uid, lat, lon = z["photo_id"], z["uid"], z["lat"], z["lon"]
    vocab = {u: i for i, u in enumerate(json.loads((YFCC / "uid_vocab.json").read_text(encoding="utf-8")))}
    authors = _mp16_authors({e["image_id"] for e in photos})
    pools = load_pools("mp16")
    score = np.asarray([np.where(pools["valid"][e["index"]], pools["scores"][e["index"]], -np.inf)[0] for e in photos])
    base = np.asarray([_haversine_km(*e["truth"], np.asarray(e["pool"][:1]))[0] for e in photos])
    proposals = []  # (photo, lat, lon, cluster size, cluster share, matches)
    for p, e in enumerate(photos):
        own_id = int(e["image_id"].split("_")[-1].split(".")[0])
        own_uid = vocab.get(authors.get(e["image_id"]), -1)
        best = None
        for phrase in table.get(e["image_id"], {}):
            i = index[phrase]
            r = m["rows"][m["starts"][i]:m["starts"][i + 1]]
            r = r[(photo_id[r] != own_id) & (uid[r] != own_uid)]  # never the photo itself or its photographer
            if len(r) < 2:
                continue
            sample = r if len(r) <= 400 else r[np.random.default_rng(0).choice(len(r), 400, replace=False)]
            pts = np.stack([lat[sample], lon[sample]], 1).astype(np.float64)
            near = np.stack([_haversine_km(*q, pts) < 25 for q in pts])
            j = int(near.sum(1).argmax())
            size, share = int(near[j].sum()), float(near[j].mean())
            centre = np.median(pts[near[j]], 0)
            if best is None or (share, size) > (best[4], best[3]):
                best = (p, centre[0], centre[1], size, share, len(r))
        if best:
            proposals.append(best)
    print(f"{tag}: n={len(photos)}, Pinpoint top-1 < 25 km {(base < 25).mean():.1%}; photos with a name proposal {len(proposals)}")
    right = [(_haversine_km(*photos[p]["truth"], np.asarray([[la, lo]]))[0] < 25) for p, la, lo, *_ in proposals]
    mist = [base[p] >= 25 for p, *_ in proposals]
    print(f"  proposals right (< 25 km): {np.mean(right):.1%}; on Pinpoint's mistakes {np.mean([r for r, w in zip(right, mist) if w]):.1%} "
          f"({sum(r and w for r, w in zip(right, mist))} photos)")
    for min_share in (0.3, 0.5, 0.7, 0.9):
        for min_size in (2, 5):
            for q in (0.2, 0.5, 1.0):
                T = np.quantile(score, q) if q < 1 else np.inf
                chosen, switched = base.copy(), 0
                for p, la, lo, size, share, _ in proposals:
                    if share < min_share or size < min_size or score[p] > T:
                        continue
                    e = photos[p]
                    if _haversine_km(la, lo, np.asarray(e["pool"][:1]))[0] < 25:
                        continue
                    chosen[p] = _haversine_km(*e["truth"], np.asarray([[la, lo]]))[0]
                    switched += 1
                c = _bootstrap((chosen < 25) * 1.0 - (base < 25))
                fixes, breaks = ((chosen < 25) & (base >= 25)).sum(), ((chosen >= 25) & (base < 25)).sum()
                print(f"  share >= {min_share:.1f}, size >= {min_size}, Pinpoint score <= q{q:.1f}: switched {switched:3d}; < 25 km {100 * c[0]:+.1f} "
                      f"[{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]; fixes {fixes}, breaks {breaks}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("extract", "merge", "phrases", "match", "report"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--part", type=int, default=None, help="extract: one of --parts photoid ranges (run in parallel, then merge)")
    parser.add_argument("--parts", type=int, default=16)
    args = parser.parse_args(argv)
    {"extract": lambda: extract(args.part if args.part is not None else 0, args.parts), "merge": lambda: merge(args.parts), "phrases": lambda: phrases(args.tag), "match": lambda: match(args.tag), "report": lambda: report(args.tag)}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
