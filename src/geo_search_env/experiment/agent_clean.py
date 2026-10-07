# Clean re-measurement after finding that the "val" photo set is 1,000 benchmark eval-half photos (query_evidence.py select --source bench), not MP16:
# tool combiners fitted on MP16 dev only, applied to the full eval halves and to the eval halves minus the val photos (never seen during development).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_clean --name 9b --groups text poi text+poi

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from .geo_tools import _cols, _load, _summary
from .query_evidence import ROOT


def main(argv: Sequence[str] | None = None) -> int:
    from .name_score import _fit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="9b")
    parser.add_argument("--groups", nargs="+", default=["text"])
    args = parser.parse_args(argv)
    Ff, Df, Vf, _, _ = _load("dev", args.name)
    F, D, V, fix, base = _load("full", args.name)
    C = max(Ff.shape[1], F.shape[1])
    pad = lambda a, fill: np.concatenate([a, np.full((a.shape[0], C - a.shape[1], *a.shape[2:]), fill, a.dtype)], 1)
    Ff, Df, Vf, F, D, V = pad(Ff, 0), pad(Df, 1e5), pad(Vf, False), pad(F, 0), pad(D, 1e5), pad(V, False)
    val_ids = {e["image_id"] for e in json.loads((ROOT / "val" / "dev.json").read_text(encoding="utf-8"))}
    unseen = np.asarray([e["image_id"] not in val_ids for e in json.loads((ROOT / "full" / "dev.json").read_text(encoding="utf-8"))])
    print(f"fit on MP16 dev only (n={len(Df)}); eval halves n={len(D)}, of which never in 'val' n={unseen.sum()}; reranker top-1 < 25 km "
          f"{(base < 25).mean():.1%} (unseen {(base[unseen] < 25).mean():.1%})")
    for group in args.groups:
        hits = {km: np.zeros(len(D)) for km in (1, 25)}
        for seed in range(3):
            pick = np.argmax(_fit(Ff[..., _cols(group)], Df, Vf, seed)(F[..., _cols(group)], V), 1)
            for km in hits:
                hits[km] += (D[np.arange(len(D)), pick] < km) / 3
        print(_summary(f"{group}, all eval halves", hits, base, fix))
        print(_summary(f"{group}, unseen only", {km: h[unseen] for km, h in hits.items()}, base[unseen], fix[unseen]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
