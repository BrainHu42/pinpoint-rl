# Clue ablation (dev3k, setft_own, MegaLoc on): answer heads trained on train with each view, every area zoomed, gated pick.
#   none:     no reading (Pinpoint areas, clue evidence zeroed)
#   areas:    clue-place tokens shown, clue evidence zeroed on every token
#   evidence: clue evidence on Pinpoint areas, clue-place tokens hidden
#   both:     the full read state
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from geo_search_env.experiment.action_agent import _released_own_dev3k_km  # noqa: E402
from geo_search_env.experiment.hypothesis_agent import CLUE_COLS, _answer, fit_answer, load  # noqa: E402

CKPT = "setft_own"
SEEDS = (0, 1, 2)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

tr = load("train", CKPT, extra=("megaloc",))
dv = load("dev3k", CKPT, stats=tr["stats"], extra=("megaloc",))
dv["base_km"] = _released_own_dev3k_km()


def view(d, name):
    v = dict(d)
    noev = d["pre"].clone()
    noev[..., CLUE_COLS] = d["pre_noclue"][..., CLUE_COLS]
    if name in ("none", "areas"):
        v["pre"] = noev
    if name in ("none", "evidence"):
        v["mask"] = d["mask"] & d["is_area"]
    v.pop("pre_noclue", None)  # fit_answer's _view only acts with read_mode; keep the view fixed
    return v


def boot(x, n=2000, seed=0):
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


base = np.asarray(dv["base_km"])
has_clue = ((dv["mask"] & ~dv["is_area"]).any(1)).cpu().numpy()
ev_any = (dv["pre_noclue"][..., CLUE_COLS] != dv["pre"][..., CLUE_COLS]).any(-1)
has_ev = ((ev_any & dv["mask"] & dv["is_area"]).any(1)).cpu().numpy()
lines = [f"dev3k {dv['n']} photos; released Pinpoint (original protocol) < 25 km {100 * (base < 25).mean():.1f}, < 1 km {100 * (base < 1).mean():.1f}; "
         f"photos with a clue-place area {100 * has_clue.mean():.1f}%, with clue evidence on some Pinpoint area {100 * has_ev.mean():.1f}%"]
print(lines[0], flush=True)
photos = np.arange(tr["n"])
res = {}
for name in ("none", "areas", "evidence", "both"):
    tv, dvv = view(tr, name), view(dv, name)
    logits = []
    per_seed = []
    for s in SEEDS:
        net = fit_answer(tv, photos, s)
        z = dvv["mask"]
        per_seed.append(_answer(net, dvv, z).cpu().numpy())
        with torch.no_grad():
            logits.append(torch.log_softmax(net.answer_logits(dvv, z), 1))
    pick = torch.stack(logits).mean(0).argmax(1)
    km = dvv["km_zoom"].gather(1, pick[:, None]).squeeze(1).cpu().numpy()
    res[name] = km
    seeds25 = [100 * (k < 25).mean() for k in per_seed]
    msg = (f"{name:9s} ensemble < 25 km {100 * (km < 25).mean():.1f}, < 1 km {100 * (km < 1).mean():.1f}; seeds < 25 km "
           f"{' '.join(f'{x:.1f}' for x in seeds25)}")
    print(msg, flush=True)
    lines.append(msg)

lines.append("\npaired differences (ensemble, % < 25 km; < 1 km), all photos | photos with a clue-place area | photos with clue evidence on a Pinpoint area:")
for a, b in (("areas", "none"), ("evidence", "none"), ("both", "none"), ("both", "areas"), ("both", "evidence"), ("none", "base")):
    ka = res[a]
    kb = base if b == "base" else res[b]
    parts = []
    for sub in (np.ones_like(has_clue), has_clue, has_ev):
        g = (ka[sub] < 25).astype(float) - (kb[sub] < 25).astype(float)
        g1 = (ka[sub] < 1).astype(float) - (kb[sub] < 1).astype(float)
        c, c1 = boot(g), boot(g1)
        parts.append(f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]; {c1[0]:+.1f} (n {int(sub.sum())})")
    lines.append(f"  {a:8s} - {b:8s} " + " | ".join(parts))
for l in lines[-7:]:
    print(l, flush=True)
(OUT / "clue_ablation.txt").write_text("\n".join(lines) + "\n")
