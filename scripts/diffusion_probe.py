# Does diffusion over the gallery graph add evidence? (graph_diffusion diffuse outputs; setft_own; dev3k, plus train for the cheap checks)
#   photo level: % < 25 km of the diffused / seeds-only mass peak vs the reranker's top-1
#   area level:  within-photo AUC (right area < 25 km vs wrong) of log mass: diffused, seeds-only, Pinpoint support, restricted mass
#   head:        answer head (MegaLoc on, clues read, all zoomed, 3-seed ensembles) with seeds-only / diffused / both columns vs base
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from geo_search_env.experiment import action_agent  # noqa: E402
from geo_search_env.experiment.area_model import _paths  # noqa: E402
from geo_search_env.experiment.hypothesis_agent import SOURCES, fit_answer  # noqa: E402

CKPT = "setft_own"
VARIANT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else ""
HEAD = "--no-head" not in sys.argv
D = Path("artifacts/diffusion")


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def load(tag):
    f = dict(np.load(_paths(tag, CKPT)[0]))
    z = dict(np.load(D / f"diffusion_{tag}_{CKPT}{'_' + VARIANT if VARIANT else ''}.npz"))
    assert np.array_equal(z["photo"], f["photo"]) and np.array_equal(z["source"], f["source"])
    return f, z


def auc_within(score, right, photo):
    """Mean over photos with a right and a wrong area of P(score right > score wrong) (ties half)."""
    acc, n = 0.0, 0
    for p in np.unique(photo):
        J = np.flatnonzero(photo == p)
        r, w = J[right[J]], J[~right[J]]
        if len(r) and len(w):
            s_r, s_w = score[r][:, None], score[w][None]
            acc += ((s_r > s_w).mean() + 0.5 * (s_r == s_w).mean())
            n += 1
    return acc / max(n, 1)


for tag in ("train", "dev3k"):
    import json

    if not (D / f"diffusion_{tag}_{CKPT}{'_' + VARIANT if VARIANT else ''}.npz").exists():
        print(f"\n== {tag}: no diffusion file yet, skipped")
        continue
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}_{CKPT}/photos.json").read_text())
    truth = np.asarray([e["truth"] for e in photos])
    f, z = load(tag)
    print(f"\n== {tag}: {len(photos)} photos; reranker top-1 < 25 km {100 * np.mean(f['base_km'] < 25):.1f}")
    for name in ("diff", "seed"):
        km = hav(truth[:, 0], truth[:, 1], z[f"pick_{name}"][:, 0], z[f"pick_{name}"][:, 1])
        print(f"  {name:4s} mass peak < 25 km {100 * np.mean(km < 25):.1f}")
    keep = np.isin(f["source"], SOURCES)
    ph, right = f["photo"][keep], (f["km_agree"] < 25)[keep]
    lg = lambda x: np.log(np.maximum(x, 1e-9))
    for name, s in (("diffused mass", lg(z["mass_diff"])), ("seeds-only mass", lg(z["mass_seed"])), ("Pinpoint support", lg(f["support"])),
                    ("restricted mass_0.2", lg(f["mass_0.2"]))):
        s = s[keep]
        a1 = np.mean([right[np.flatnonzero(ph == p)][np.argmax(s[ph == p])] for p in np.unique(ph)])
        print(f"  {name:20s} within-photo AUC {auc_within(s, right, ph):.3f}; area ranked first by it right {100 * a1:.1f}%")

if not HEAD:
    raise SystemExit(0)

tr = action_agent._data("train", ckpt=CKPT)
dv = action_agent._data("dev3k", stats=tr["stats"], ckpt=CKPT)


def cols(tag, d):
    f, z = load(tag)
    keep = np.flatnonzero(np.isin(f["source"], SOURCES))
    photo = f["photo"][keep]
    slot = np.zeros(len(keep), int)
    seen = {}
    for j, p in enumerate(photo):
        slot[j] = seen.get(p, 0)
        seen[p] = slot[j] + 1
    out = {}
    for name in ("diff", "seed"):
        x = np.log(np.maximum(z[f"mass_{name}"][keep], 1e-6))
        mx = np.full(photo.max() + 1, -np.inf)
        np.maximum.at(mx, photo, x)
        c = np.stack([x, x - mx[photo]], 1)
        pad = np.zeros(d["mask"].shape + (2,), np.float32)
        pad[photo, slot] = c
        out[name] = pad
    return out


ex = {"train": cols("train", tr), "dev3k": cols("dev3k", dv)}
m = tr["mask"].cpu().numpy()
for k in ("diff", "seed"):
    mu, sd = ex["train"][k][m].mean(0), ex["train"][k][m].std(0) + 1e-6
    for n in ex:
        ex[n][k] = (ex[n][k] - mu) / sd


def view(d, name, groups):
    v = dict(d)
    v.pop("pre_noclue", None)
    if groups:
        v["pre"] = torch.cat([d["pre"], torch.as_tensor(np.concatenate([ex[name][g] for g in groups], -1), device=d["pre"].device)], -1)
    return v


res = {}
for label, groups in (("base", ()), ("+seed", ("seed",)), ("+diff", ("diff",)), ("+both", ("seed", "diff"))):
    t, v = view(tr, "train", groups), view(dv, "dev3k", groups)
    logits = []
    for s in (0, 1, 2):
        net = fit_answer(t, np.arange(t["n"]), s)
        with torch.no_grad():
            logits.append(torch.log_softmax(net.answer_logits(v, v["mask"]), 1))
    pick = torch.stack(logits).mean(0).argmax(1)
    res[label] = v["km_zoom"].gather(1, pick[:, None]).squeeze(1).cpu().numpy()
    print(f"head {label:6s} dev3k < 25 km {100 * np.mean(res[label] < 25):.1f}, < 1 km {100 * np.mean(res[label] < 1):.1f}", flush=True)
rng = np.random.default_rng(0)
for label in ("+seed", "+diff", "+both"):
    g = (res[label] < 25).astype(float) - (res["base"] < 25)
    mm = g[rng.integers(0, len(g), (2000, len(g)))].mean(1)
    print(f"  {label:6s} vs base {100 * g.mean():+.1f} [{100 * np.percentile(mm, 2.5):+.1f}, {100 * np.percentile(mm, 97.5):+.1f}]; fixes {(g > 0).sum()}, breaks {(g < 0).sum()}")
