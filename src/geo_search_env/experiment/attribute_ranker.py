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

from .attribute_check import LANGUAGES
from .evidence_ranker import SEEDS, THRESHOLDS, _reward, fit_scores, load_set
from .photo_attributes import CHOICES
from .query_evidence import ROOT
from .stage1_eval import _bootstrap

ATTRIBUTES = ROOT / "attributes"
INDICATORS = [(key, value) for key, values in CHOICES.items() for value in values]  # 21 photo indicators


def candidate_features(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """(photos, candidates, 10) map attributes and (photos, candidates) country codes."""

    saved = np.load(ATTRIBUTES / f"candidates_{tag}.npz")
    return saved["num"], saved["country"]


def photo_records(tag: str, photos: list[dict[str, Any]]) -> tuple[np.ndarray, list[str | None]]:
    """(photos, 21) one-hot of the VLM's choices (an invalid choice leaves its field all zero) and each photo's text language code or None."""

    described = json.loads((ATTRIBUTES / f"photo_{tag}.json").read_text(encoding="utf-8"))
    onehot = np.zeros((len(photos), len(INDICATORS)), dtype=np.float32)
    language: list[str | None] = []
    for m, e in enumerate(photos):
        attrs = described[e["image_id"]]["attrs"]
        for i, (key, value) in enumerate(INDICATORS):
            onehot[m, i] = attrs[key] == value
        language.append(LANGUAGES.get(attrs["text_language"] or ""))  # None when "none" or a language outside the table
    return onehot, language


def attribute_features(onehot: np.ndarray, language: list[str | None], num_z: np.ndarray, country: np.ndarray, languages: dict[str, list[str]]) -> np.ndarray:
    """Per candidate: the photo's indicators, their products with the candidate's standardized map attributes, a text-language match and
    whether the photo names a language."""

    n, k = country.shape
    products = (onehot[:, None, :, None] * num_z[:, :, None, :]).reshape(n, k, -1)
    match = np.zeros((n, k), dtype=np.float32)
    for m, lang in enumerate(language):
        if lang is not None:
            match[m] = [lang in languages.get(c, []) for c in country[m]]
    has_text = np.asarray([lang is not None for lang in language], dtype=np.float32)[:, None].repeat(k, 1)
    return np.concatenate([np.repeat(onehot[:, None, :], k, 1), products, match[..., None], has_text[..., None]], axis=-1)


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
    cand = np.concatenate([candidate_features(n)[0] for n in sets])
    country = np.concatenate([candidate_features(n)[1] for n in sets])
    mean, std = cand[train][valid[train]].mean(0), cand[train][valid[train]].std(0) + 1e-6
    cand_z = np.where(valid[..., None], (cand - mean) / std, 0.0).astype(np.float32)
    languages = json.loads((ATTRIBUTES / "country_languages.json").read_text(encoding="utf-8"))
    records = [photo_records(n, sets[n]["photos"]) for n in sets]
    onehot = np.concatenate([r[0] for r in records])
    language = [lang for r in records for lang in r[1]]
    rng = np.random.default_rng(0)
    order = np.concatenate([bounds[i] + rng.permutation(sizes[n]) for i, n in enumerate(sets)])  # shuffle photos within each set
    shuffled = attribute_features(onehot[order], [language[i] for i in order], cand_z, country, languages)
    configs = {
        "A original features": one_shot,
        "F + candidate attributes": np.concatenate([one_shot, cand], axis=-1),
        "G + photo attributes and matches": np.concatenate([one_shot, cand, attribute_features(onehot, language, cand_z, country, languages)], axis=-1),
        "H G with photo attributes shuffled": np.concatenate([one_shot, cand, shuffled], axis=-1),
    }

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
    base, comparisons = "A original features", {"G + photo attributes and matches": ("F + candidate attributes", "H G with photo attributes shuffled")}
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
        for name, others in comparisons.items():
            for other in others:
                ci = _bootstrap(hits[name][25.0][idx] - hits[other][25.0][idx])
                delta = {t: float(hits[name][t][idx].mean() - hits[other][t][idx].mean()) for t in THRESHOLDS}
                report[g][f"{name} minus {other}"] = {"change": {f"<{int(t)} km": v for t, v in delta.items()}, "<25 km CI": ci}
                print(f"  G minus {other[:1]}: " + " ".join(f"{delta[t]:+6.1%}" for t in THRESHOLDS) + f"   <25 km [{ci[1]:+.1%}, {ci[2]:+.1%}]")
    (ROOT / "attribute_ranker.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("fit",))
    parser.parse_args(argv)
    fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
