"""Why OSM names inside areas add nothing (lesson 78): a funnel over the baseline answer head's own mistakes. On held-out 10k train photos and
dev3k, the answer head (clues read, all areas zoomed, gated pick; trained on the other train photos) picks an area; 'choice errors' are photos
it gets wrong (> 25 km) although one of its areas is right. Names can only fix those. Funnel: has a name -> a name found in OSM -> a name
within 5 km of a right area's top-1 -> and not of the chosen area's -> whether the existing clue evidence (text-search clues near the area,
or the area being a clue place) already marks a right area. Also: the reverse (baseline right, a name near a wrong area but not the chosen
one), how often found names lie near the truth at all, and examples.

  PYTHONPATH=src .venv/bin/python scripts/diagnose_names.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from geo_search_env.experiment.hypothesis_agent import SOURCES, _paths, _sub, fit_answer, load
from geo_search_env.experiment.name_checks import KINDS, Hits, _area_points, _km, photo_names

R = 5.0


def slots(tag: str):
    """Padded (photo, slot) -> feature-file row, as hypothesis_agent.load builds them."""

    f = np.load(_paths(tag, "setft_a")[0])
    rows = np.flatnonzero(np.isin(f["source"], SOURCES))
    photo = f["photo"][rows]
    n = int(photo.max()) + 1
    h = int(np.bincount(photo).max())
    out = np.full((n, h), -1)
    seen = np.zeros(n, int)
    for r, p in zip(rows, photo):
        out[p, seen[p]] = r
        seen[p] += 1
    return out, f


def picks(net, d, photos):
    with torch.no_grad():
        b = _sub(d, torch.as_tensor(photos, device="cuda"))
        return net.answer_logits(b, b["mask"]).argmax(1).cpu().numpy(), (b["km_zoom"] < 25).cpu().numpy() & b["mask"].cpu().numpy()


def funnel(label, tag, d, net, photos, hits):
    slot_row, f = slots(tag)
    nc = np.load(Path("artifacts/agent") / f"namecheck_{tag}_setft_a.npz")
    col = {c: i for i, c in enumerate(nc["columns"].tolist())}
    near = (nc["x"][:, col["read_n5"]] > 0) | (nc["x"][:, col["recall_n5"]] > 0)  # per feature-file row
    clue = (f["read_n"] > 0) | (f["recall_n"] > 0) | (f["source"] == 1)
    names = photo_names(tag)
    truth = np.asarray([p["truth"] for p in json.loads((Path("artifacts/photo_sets") / f"{tag}_setft_a" / "photos.json").read_text())])
    _, _, point = _area_points(tag)
    pick, right = [], []
    for chunk in np.array_split(photos, max(1, len(photos) // 1024)):
        p, r = picks(net, d, chunk)
        pick.append(p)
        right.append(r)
    pick, right = np.concatenate(pick), np.concatenate(right)
    c = {k: 0 for k in ("errors", "name", "found", "near_right", "fix", "fix_new", "harm", "base_right")}
    fix_ex, harm_ex = [], []
    near_truth = {1: 0, 5: 0, 25: 0}
    with_found = 0
    for i, p in enumerate(photos):
        rows = slot_row[p][: right.shape[1]]
        ok = right[i]
        chosen = rows[pick[i]]
        found = [(k, x, *hits.get(x)) for k in KINDS for x in names[p][k]]
        found = [(k, x, n, xy) for k, x, n, xy in found if len(xy)]
        if found:
            with_found += 1
            dmin = min(_km(truth[[p]], xy).min() for *_, xy in found)
            for t in near_truth:
                near_truth[t] += dmin <= t
        if ok[pick[i]]:
            c["base_right"] += 1
            wrong_rows = rows[(rows >= 0) & ~ok]
            if near[wrong_rows].any() and not near[chosen]:
                c["harm"] += 1
                harm_ex.append((p, chosen, wrong_rows[near[wrong_rows]]))
            continue
        if not ok.any():
            continue
        c["errors"] += 1
        c["name"] += bool(names[p]["read"] or names[p]["recall"])
        c["found"] += bool(found)
        right_rows = rows[ok]
        if not near[right_rows].any():
            continue
        c["near_right"] += 1
        if near[chosen]:
            continue
        c["fix"] += 1
        new = not clue[right_rows[near[right_rows]]].any()
        c["fix_new"] += new
        fix_ex.append((p, chosen, right_rows[near[right_rows]], new))
    n = len(photos)
    pct = lambda x: f"{x} ({100 * x / n:.1f}%)"
    print(f"\n{label} ({n} photos): baseline right {pct(c['base_right'])}")
    print(f"  choice errors (wrong, but a right area exists): {pct(c['errors'])}")
    print(f"    with a name (read or recalled):              {pct(c['name'])}")
    print(f"    with a name found in OSM:                    {pct(c['found'])}")
    print(f"    a name within {R:g} km of a right area:          {pct(c['near_right'])}")
    print(f"    ... and not of the chosen area (fixable):    {pct(c['fix'])}")
    print(f"    ... of those, no existing clue evidence there: {pct(c['fix_new'])}")
    print(f"  reverse (baseline right, a name near a wrong area, not the chosen one): {pct(c['harm'])}")
    print(f"  photos with a found name: {with_found}; nearest found-name match within 1 / 5 / 25 km of the truth: "
          + " / ".join(f"{100 * near_truth[t] / max(with_found, 1):.0f}%" for t in near_truth))

    def show(p, rows_):
        out = []
        for k in KINDS:
            for x in names[p][k]:
                cnt, xy = hits.get(x)
                if len(xy) and (_km(point[rows_], xy).min(1) <= R).any():
                    out.append(f"{k}:'{x}' ({cnt} worldwide)")
        return ", ".join(out[:4])

    rng = np.random.default_rng(0)
    print("  examples, fixable:")
    for j in rng.permutation(len(fix_ex))[:6]:
        p, chosen, rr, new = fix_ex[j]
        print(f"    {'NEW ' if new else 'dup '} names near the right area: {show(p, rr)}")
    print("  examples, reverse:")
    for j in rng.permutation(len(harm_ex))[:6]:
        p, chosen, wr = harm_ex[j]
        print(f"    names near a wrong area: {show(p, wr)}")


tr = load("train", "setft_a")
dv = load("dev3k", "setft_a", stats=tr["stats"])
order = np.random.default_rng(0).permutation(tr["n"])
held, pool = order[:10_000], order[10_000:]
net = fit_answer(tr, pool, seed=0)
hits = Hits()
funnel("held-out train", "train", tr, net, held, hits)
funnel("dev3k", "dev3k", dv, net, np.arange(dv["n"]), hits)
