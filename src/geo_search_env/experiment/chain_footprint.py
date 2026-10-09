# Regional chains as region constraints (GEOLOCATION_CLUES.md section 8): a chain name read in a photo (Wawa, Mercadona, Biedronka) rules out
# regions where the chain has no stores. The text search drops chain names (their matches don't cluster); this uses their footprint instead.
# Clue n-grams of the 4B clues are matched to Overture brand names (>= MIN_STORES stores); each brand's stores are labelled with regions; on
# Pinpoint's dev3k mistakes, does the footprint keep the true region and exclude Pinpoint's? Shuffled-clue control. dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.chain_footprint grams --tag dev3k
#        PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.chain_footprint stores --tag dev3k   (DuckDB over Overture places)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.chain_footprint report --tag dev3k

"""grams:  every 1-4 word n-gram of every clue (normalised) -> artifacts/agent/chain_grams_{tag}.json {photo: [n-grams]}.
stores: Overture brands whose normalised primary name equals one of those n-grams and has >= MIN_STORES places; their store coordinates (at most
        MAX_STORES per brand, a fixed sample) -> artifacts/agent/chain_stores_{tag}.npz.
report: per photo with a brand, the brand's region shares (stores labelled by the nearest index photo); the footprint = regions holding a share
        >= MIN_SHARE; whether it keeps the truth's region and excludes Pinpoint's top-1 region, on Pinpoint's mistakes and right photos; real vs
        shuffled brands."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import unicodedata
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
OVERTURE = Path("data/pinpoint/overture/places")
MIN_STORES, MAX_STORES = 20, 5000


def norm(text: str) -> str:  # as text_search.norm (no package imports here: `stores` runs in the DuckDB env)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", text).strip()


def grams(tag: str) -> None:
    from .region_evidence import _raw_phrases

    out = {}
    for image_id, table in _raw_phrases(tag, "4b").items():
        g = set()
        for q in table:
            w = norm(q).split()
            g |= {" ".join(w[i:i + n]) for n in range(1, 5) for i in range(len(w) - n + 1) if len(" ".join(w[i:i + n])) >= 3}
        out[image_id] = sorted(g)
    (OUT / f"chain_grams_{tag}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(out)} photos, {len({x for v in out.values() for x in v})} distinct n-grams")


def stores(tag: str, memory: str = "8GB") -> None:
    import duckdb

    wanted = {x for v in json.loads((OUT / f"chain_grams_{tag}.json").read_text(encoding="utf-8")).values() for x in v}
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory}'")
    con.create_function("norm", norm, ["VARCHAR"], "VARCHAR")
    brands = con.execute(f"""SELECT brand.names."primary" AS b, count(*) AS n FROM read_parquet('{OVERTURE}/*.parquet')
                             WHERE brand.names."primary" IS NOT NULL GROUP BY 1 HAVING count(*) >= {MIN_STORES}""").fetchall()
    keep = {}
    for b, n in brands:
        k = norm(b)
        if k in wanted and (k not in keep or n > keep[k][1]):
            keep[k] = (b, n)
    print(f"{len(brands)} brands with >= {MIN_STORES} places; {len(keep)} match clue n-grams", flush=True)
    con.execute("CREATE TABLE wanted (b VARCHAR)")
    con.executemany("INSERT INTO wanted VALUES (?)", [(b,) for b, _ in keep.values()])
    rows = con.execute(f"""SELECT p.brand.names."primary", (p.bbox.ymin + p.bbox.ymax) / 2, (p.bbox.xmin + p.bbox.xmax) / 2
                           FROM read_parquet('{OVERTURE}/*.parquet') p JOIN wanted w ON p.brand.names."primary" = w.b""").fetchall()  # one scan
    by_brand: dict[str, list] = {}
    for b, la, lo in rows:
        by_brand.setdefault(b, []).append((la, lo))
    rng = np.random.default_rng(7)
    names, lat, lon, offsets = [], [], [], [0]
    for k, (b, n) in sorted(keep.items()):
        pts = by_brand.get(b, [])
        if len(pts) > MAX_STORES:
            pts = [pts[i] for i in sorted(rng.choice(len(pts), MAX_STORES, replace=False))]
        names.append(k)
        lat += [p[0] for p in pts]
        lon += [p[1] for p in pts]
        offsets.append(len(lat))
    np.savez(OUT / f"chain_stores_{tag}.npz", brand=np.asarray(names), lat=np.asarray(lat, np.float32), lon=np.asarray(lon, np.float32),
             offsets=np.asarray(offsets), stores=np.asarray([keep[k][1] for k in names]))
    print(f"{len(names)} brands, {len(lat)} store coordinates -> {OUT / f'chain_stores_{tag}.npz'}")


def report(tag: str, min_share: float = 0.02) -> None:
    from .agent_targets import same_region
    from .region_evidence import _Labeller
    from .strategy_search import _haversine_km

    photos = {e["image_id"]: e for e in json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))}
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    grams_by_photo = json.loads((OUT / f"chain_grams_{tag}.json").read_text(encoding="utf-8"))
    z = dict(np.load(OUT / f"chain_stores_{tag}.npz"))
    label = _Labeller()
    share: dict[str, dict[tuple[str, str], float]] = {}
    for i, b in enumerate(z["brand"]):
        a, c = z["offsets"][i], z["offsets"][i + 1]
        regions = collections.Counter(label(float(la), float(lo)) for la, lo in zip(z["lat"][a:c], z["lon"][a:c]))
        total = sum(regions.values())
        share[str(b)] = {r: n / total for r, n in regions.items()}
    print(f"{tag}: {len(share)} brands matched in clues (>= {20} stores each); footprint = regions with >= {min_share:.0%} of a brand's stores")
    base = {k: _haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] for k, e in photos.items()}
    ids = [k for k in photos if any(g in share for g in grams_by_photo.get(k, []))]
    perm = np.random.default_rng(0).permutation(len(ids))
    perm = np.where(perm == np.arange(len(ids)), (perm + 1) % len(ids), perm)
    for name, brand_of in (("real", {k: k for k in ids}), ("shuffled (luck)", {ids[i]: ids[perm[i]] for i in range(len(ids))})):
        counts = collections.Counter()
        examples = []
        for k in ids:
            bs = [g for g in grams_by_photo[brand_of[k]] if g in share]
            # the most concentrated brand of the photo (fewest footprint regions) decides
            b = min(bs, key=lambda g: sum(v >= min_share for v in share[g].values()))
            foot = {r for r, v in share[b].items() if v >= min_share}
            truth, pin = labels[k]["truth"], (labels[k]["pool"][0] if labels[k]["pool"] else {})
            reg = lambda r: {"country": r[0], "region": r[1]}
            keeps_truth = any(same_region(reg(r), truth) for r in foot)
            keeps_pin = any(same_region(reg(r), pin) for r in foot)
            wrong = base[k] >= 25
            counts["photos"] += 1
            counts["mistakes"] += wrong
            counts["footprint keeps truth"] += keeps_truth
            counts["excludes Pinpoint, keeps truth, Pinpoint wrong (possible fix)"] += wrong and keeps_truth and not keeps_pin
            counts["excludes Pinpoint while Pinpoint right (would break)"] += (not wrong) and not keeps_pin
            counts["footprint <= 5 regions"] += len(foot) <= 5
            if name == "real" and wrong and keeps_truth and not keeps_pin and len(examples) < 8:
                examples.append(f"{b} ({len(foot)} regions)")
        print(f" {name}: " + "; ".join(f"{a} {n}" for a, n in counts.items()))
        if examples:
            print("   e.g. " + ", ".join(examples))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("grams", "stores", "report"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    {"grams": grams, "stores": stores, "report": report}[args.cmd](args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
