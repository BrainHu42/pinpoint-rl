# Do photo attributes matched against map attributes of each candidate help the learned reranker choose?
# Usage: .venv/bin/python -m geo_search_env.experiment.attribute_ranker fit    (GPU, minutes; needs candidate_attributes build)

"""The pipeline's reranker (see evidence_ranker.py), refit on MP16 train photos with extra per-candidate features, evaluated on
`dev` and `val` over three seeds:

A  original features (baseline)
F  + candidate attributes from offline maps (climate, elevation, ruggedness, coast distance, urbanness): a prior on places,
   no photo information, so any gain here is not from reading the photo
G  + photo attributes (the VLM's description of the photo) and match features against the candidate's attributes
H  = G with the photo attributes shuffled across photos within each set (control: G's gain must come from the photo)

G and H need `photo_attributes.json` per tag (written by the VLM pass); without it only A and F run.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np

from .evidence_ranker import SEEDS, THRESHOLDS, _reward, fit_scores, load_set
from .query_evidence import ROOT
from .stage1_eval import _bootstrap

ATTRIBUTES = ROOT / "attributes"


def candidate_features(tag: str) -> np.ndarray:
    saved = np.load(ATTRIBUTES / f"candidates_{tag}.npz")
    return saved["num"]


def fit() -> None:
    sets = {name: load_set(name) for name in ("train", "dev", "val")}
    sizes = {name: len(s["photos"]) for name, s in sets.items()}
    join = lambda key: np.concatenate([sets[n][key] for n in sets])
    valid, distance, one_shot = join("valid"), join("distance"), join("one_shot")
    reward = _reward(distance, valid)
    bounds = np.cumsum([0, *sizes.values()])
    rows = {name: np.arange(bounds[i], bounds[i + 1]) for i, name in enumerate(sets)}
    train = np.zeros(len(valid), dtype=bool)
    train[rows["train"]] = True
    cand = np.concatenate([candidate_features(n) for n in sets])
    configs = {"A original features": one_shot, "F + candidate attributes": np.concatenate([one_shot, cand], axis=-1)}

    hits: dict[str, dict[float, np.ndarray]] = {}
    for name, features in configs.items():
        picked = [np.take_along_axis(distance, fit_scores(features, valid, reward, train, seed).argmax(1)[:, None], 1)[:, 0] for seed in SEEDS]
        hits[name] = {t: np.mean([p < t for p in picked], axis=0) for t in THRESHOLDS}
        print(f"fitted {name}", flush=True)

    report: dict[str, Any] = {"photos": sizes, "seeds": list(SEEDS)}
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))  # Flickr "photo no longer available" placeholders (audit_sets.py)
    keep = np.asarray([e["image_id"] not in excluded for e in sets["val"]["photos"]])
    val_rows = rows["val"][keep]
    benchmark = np.asarray([e["benchmark"] for e in sets["val"]["photos"]])[keep]
    groups = {"dev": rows["dev"], "val": val_rows}
    groups |= {f"val {b}": val_rows[benchmark == b] for b in ("im2gps3k", "yfcc4k")}
    base = "A original features"
    for g, idx in groups.items():
        print(f"\n{g} (n={len(idx)})   top-1 accuracy {'<1 km':>7s} {'<25 km':>7s} {'<200 km':>8s}   change vs A (<1 / <25 / <200 km); <25 km 95% CI")
        report[g] = {}
        for name in configs:
            entry = {f"<{int(t)} km": float(hits[name][t][idx].mean()) for t in THRESHOLDS}
            line = f"  {name:34s} " + " ".join(f"{entry[f'<{int(t)} km']:7.1%}" for t in THRESHOLDS)
            if name != base:
                delta = {t: float(hits[name][t][idx].mean() - hits[base][t][idx].mean()) for t in THRESHOLDS}
                ci = _bootstrap(hits[name][25.0][idx] - hits[base][25.0][idx])
                entry |= {"change vs A": {f"<{int(t)} km": v for t, v in delta.items()}, "<25 km CI": ci}
                line += "   " + " ".join(f"{delta[t]:+6.1%}" for t in THRESHOLDS) + f"   [{ci[1]:+.1%}, {ci[2]:+.1%}]"
            report[g][name] = entry
            print(line)
    (ROOT / "attribute_ranker.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("fit",))
    parser.parse_args(argv)
    fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
