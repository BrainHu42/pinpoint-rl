# Landmark / object names looked up in YFCC100M's geotagged Flickr titles and tags (lesson 62 follow-up): a name the models produce for a photo
# ("City Island Lobster House", "Bagan", "Wittenham Clumps") is searched among ~48M geotagged photos' titles + tags; a rare name whose matches
# concentrate in one place proposes that place; it replaces Pinpoint's top-1 when Pinpoint's own confidence is low.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.yfcc_names extract            (SQLite -> data/yfcc100m/geo.npz; ~20 min)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.yfcc_names phrases --tag dev3k
#        PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.yfcc_names match --tag dev3k   (needs pyahocorasick)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.yfcc_names report --tag dev3k

"""extract: every geotagged row of data/yfcc100m/yfcc100m_dataset.sql -> data/yfcc100m/geo.npz: photo id, user (code + vocab), lat, lon, and
         'title tags' normalised (URL-decoded, casefolded, accents stripped, non-letters -> space) as one blob with offsets.
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


def extract() -> None:
    import sqlite3

    con = sqlite3.connect(f"file:{YFCC / 'yfcc100m_dataset.sql'}?mode=ro", uri=True)
    table = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone()[0]
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    print(f"table {table}: {cols}", flush=True)
    pick = lambda *keys: next(c for c in cols if any(k == c.lower() for k in keys))
    c_id, c_uid = pick("photoid", "id", "photo_id"), pick("uid", "userid", "user_nsid", "usernsid")
    c_lat, c_lon = pick("latitude", "lat"), pick("longitude", "lon")
    c_title, c_tags = pick("title"), pick("usertags", "user_tags", "tags")
    ids, uids, lat, lon, offsets, blob, vocab = [], [], [], [], [0], bytearray(), {}
    query = f"SELECT {c_id}, {c_uid}, {c_lat}, {c_lon}, {c_title}, {c_tags} FROM {table} WHERE {c_lat} IS NOT NULL AND {c_lat} != ''"
    for n, (pid, uid, la, lo, title, tags) in enumerate(con.execute(query)):
        try:
            la, lo = float(la), float(lo)
        except (TypeError, ValueError):
            continue
        text = norm(unquote_plus(title or "") + " " + unquote_plus(tags or "").replace(",", " "))
        ids.append(int(pid)); uids.append(vocab.setdefault(uid, len(vocab))); lat.append(la); lon.append(lo)
        blob += text.encode("utf-8") + b"\n"
        offsets.append(len(blob))
        if n % 5_000_000 == 0:
            print(f"  {n} geotagged rows", flush=True)
    np.savez(YFCC / "geo.npz", photo_id=np.asarray(ids, np.int64), uid=np.asarray(uids, np.int32), lat=np.asarray(lat, np.float32),
             lon=np.asarray(lon, np.float32), offsets=np.asarray(offsets, np.int64), blob=np.frombuffer(bytes(blob), np.uint8))
    (YFCC / "uid_vocab.json").write_text(json.dumps(sorted(vocab, key=vocab.get)) + "\n", encoding="utf-8")
    print(f"{len(ids)} geotagged photos, {len(vocab)} users, {len(blob) / 1e9:.2f} GB of text -> {YFCC / 'geo.npz'}", flush=True)


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
    parser.add_argument("node", choices=("extract", "phrases", "match", "report"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    {"extract": lambda: extract(), "phrases": lambda: phrases(args.tag), "match": lambda: match(args.tag), "report": lambda: report(args.tag)}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
