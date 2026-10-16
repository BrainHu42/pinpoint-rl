"""web(query): the agent's offline Google web search. Two parts, as a Google result page mixes them:

  pages   Wikipedia / Wikivoyage (experiment.web_search, owned by the osm-search line): titles, URLs, snippets, a knowledge panel with
          coordinates when the top page is the place itself
  photos  where the words were used: Flickr titles / tags (YFCC100M) and named places (GeoNames, Overture), clustered with a share per
          cluster (experiment.text_search; lessons 64-65, the clue evidence of the pipeline's read chain)

A phone number or website instead finds the businesses listing it (text_search; tools.maps.query does the same on the map). The photo being
located and its photographer's other photos are never matched (exclude, set by the environment, not the agent).

  PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.web "Wittenham Clumps"
"""

from __future__ import annotations

import argparse
from typing import Sequence


class Web:
    def __init__(self, pages: bool = True, photos: bool = True) -> None:
        from ..experiment.text_search import TextSearch
        from ..experiment.web_search import WebSearch

        self.pages = WebSearch() if pages else None
        if self.pages is not None and not self.pages.dbs:
            self.pages = None  # no Wikipedia indexes built / staged
        self.photos = TextSearch() if photos else None

    def query(self, query: str, exclude: tuple[int, str] | None = None, near: tuple[float, float, float] | None = None, top: int = 5) -> str:
        from ..experiment.web_search import format_hits

        parts = []
        if self.pages is not None:
            hits, loose = self.pages.query(query, top)
            parts.append(format_hits(query, hits, loose))
        if self.photos is not None:
            parts.append(self.photos.search(query, near=near, exclude=exclude).text())
        return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--near", type=float, nargs=3, metavar=("LAT", "LON", "KM"))
    args = parser.parse_args(argv)
    print(Web().query(args.query, near=tuple(args.near) if args.near else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
