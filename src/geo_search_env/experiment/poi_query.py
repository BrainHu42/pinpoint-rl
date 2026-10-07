# POI name search for AGENT_DIRECTION.md: which Overture places (businesses, institutions, landmarks; ~60M named POIs) carry each n-gram read from the
# photos? Input: artifacts/agent/poi_grams_{name}.json (geo_tools poigrams); output: artifacts/agent/poi_{name}.json {n-gram: {"count", "points"}}.
# Needs DuckDB (env .venvs/overture) and the places parquet (data/pinpoint/overture/places, release 2026-09-23.1). No package imports, so it runs there.
# Usage: PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.poi_query --name 9b

"""Names are normalised in SQL like geo_tools.norm (lower case, accents stripped, non-letters / digits -> one space). For each matched n-gram: the number
of POIs with exactly that primary name and up to MAX_POINTS of their (lat, lon), sampled reproducibly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

PLACES = Path("data/pinpoint/overture/places")
OUT = Path("artifacts/agent")
MAX_POINTS = 500


def main(argv: Sequence[str] | None = None) -> int:
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="9b")
    parser.add_argument("--memory", default="10GB")
    args = parser.parse_args(argv)
    grams = json.loads((OUT / f"poi_grams_{args.name}.json").read_text(encoding="utf-8"))
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute("CREATE TABLE grams (g VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO grams VALUES (?)", [(g,) for g in grams])
    rows = con.execute(f"""
        WITH p AS (
            SELECT trim(regexp_replace(lower(strip_accents(names."primary")), '[^\\p{{L}}\\p{{N}}]+', ' ', 'g')) AS n,
                   (bbox.ymin + bbox.ymax) / 2 AS lat, (bbox.xmin + bbox.xmax) / 2 AS lon
            FROM read_parquet('{PLACES}/*.parquet')
            WHERE names."primary" IS NOT NULL
        ), m AS (
            SELECT n, lat, lon, row_number() OVER (PARTITION BY n ORDER BY hash(lat, lon)) AS r, count(*) OVER (PARTITION BY n) AS c
            FROM p JOIN grams ON p.n = grams.g
        )
        SELECT n, any_value(c), list([lat, lon]) FROM m WHERE r <= {MAX_POINTS} GROUP BY n
    """).fetchall()
    out = {n: {"count": int(c), "points": [[round(a, 5), round(b, 5)] for a, b in pts]} for n, c, pts in rows}
    (OUT / f"poi_{args.name}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(grams)} n-grams, {len(out)} found among Overture places -> {OUT / f'poi_{args.name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
