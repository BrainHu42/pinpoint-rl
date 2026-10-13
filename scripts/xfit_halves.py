"""Cross-fitting (lesson 75): split MP16's embedding-cache rows into two photographer halves (hashed author id), and the 135k trained-on
query photos (queries_more + queries_extra) by the same halves. Reranker xfit_a trains on half 0's photos and scores half 1's queries; xfit_b
the reverse, so every query is scored by a reranker that never trained on its photographer.

Writes artifacts/pinpoint_train/xfit_half{0,1}_rows.npy (bool mask over cache rows, for PINPOINT_TRAIN_ROW_MASK) and
artifacts/pinpoint_set/queries_xfit{0,1}.json (queries whose photographer is in that half).

  .venv/bin/python scripts/xfit_halves.py
"""

import json
from pathlib import Path

import numpy as np

author = np.load("artifacts/pinpoint_train/author_by_cache_row.npy").astype(np.int64)
half = ((author * 2654435761) % (1 << 32)) % 2  # hashed author id -> 0 / 1
for h in (0, 1):
    np.save(f"artifacts/pinpoint_train/xfit_half{h}_rows.npy", half == h)
queries = json.loads(Path("artifacts/pinpoint_set/queries_more.json").read_text()) + json.loads(Path("artifacts/sft/queries_extra.json").read_text())
for h in (0, 1):
    mine = [q for q in queries if half[q["row"]] == h]
    Path(f"artifacts/pinpoint_set/queries_xfit{h}.json").write_text(json.dumps(mine) + "\n")
    print(f"half {h}: {np.mean(half == h):.1%} of cache rows, {len(np.unique(author[half == h]))} photographers, {len(mine)} queries")
