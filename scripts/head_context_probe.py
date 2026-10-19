# New per-area inputs for the answer head (setft_own, MegaLoc on, clues read, every area zoomed; 3-seed ensembles, trained on train, dev3k):
#   loc:    the area centre (unit xyz + sin / cos of lat, lon at 5 frequencies): learned regional calibration of Pinpoint's evidence
#   glance: the query's SigLIP2 embedding (PCA 32, fit on train) on every area token: what kind of photo is being judged
#   geo:    Flickr geotag accuracy (YFCC100M) of the area's restricted candidates: mean level, share <= 13, top candidate's level, share 16
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from geo_search_env.experiment import action_agent  # noqa: E402
from geo_search_env.experiment.area_model import _paths  # noqa: E402
from geo_search_env.experiment.hypothesis_agent import SOURCES, fit_answer  # noqa: E402

CKPT = "setft_own"
SEEDS = (0, 1, 2)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

# geotag accuracy per MP16 cache row, and index row -> cache row
emb = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
ids = np.asarray([int(Path(s.strip()).stem.split("_")[-1]) for s in open(emb / "image_ids.txt")], np.int64)
z = np.load("data/yfcc100m/geo.npz", mmap_mode="r")
pid, acc = np.asarray(z["photo_id"]), np.asarray(z["accuracy"])
o = np.argsort(pid)
pid, acc = pid[o], acc[o]
p = np.clip(np.searchsorted(pid, ids), 0, len(pid) - 1)
row_acc = np.where(pid[p] == ids, acc[p], 12).astype(np.float32)
del z, pid, acc, o, p
from geo_search_env.models.pinpoint_reranker import _discover_index  # noqa: E402

index_dir = _discover_index()
manifest = json.loads((index_dir / "manifest.json").read_text())
cache_rows = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
print("geotag accuracy table ready", flush=True)


def token_rows(tag):
    """The zoom file rows in load()'s token order: (photo, slot) per kept hypothesis."""
    _, zoom_path = _paths(tag, CKPT)
    zm = np.load(zoom_path)
    f = np.load(_paths(tag, CKPT)[0])
    keep = np.flatnonzero(np.isin(f["source"], SOURCES))
    photo = f["photo"][keep]
    slot = np.zeros(len(keep), int)
    seen = {}
    for j, ph in enumerate(photo):
        slot[j] = seen.get(ph, 0)
        seen[ph] = slot[j] + 1
    return photo, slot, zm["rows"][keep], zm["scores"][keep], zm["center"][keep]


def extra(tag, d):
    photo, slot, rows, scores, centre = token_rows(tag)
    valid = rows >= 0
    a = np.where(valid, row_acc[cache_rows[np.where(valid, rows, 0)]], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        top = np.where(valid, scores, -np.inf).argmax(1)
        geo = np.stack([np.nanmean(a, 1), np.nansum(a <= 13, 1) / np.maximum(valid.sum(1), 1),
                        a[np.arange(len(a)), top], np.nansum(a == 16, 1) / np.maximum(valid.sum(1), 1)], 1)
    geo = np.nan_to_num(geo, nan=12.0)
    lat, lon = np.radians(centre[:, 0]), np.radians(centre[:, 1])
    loc = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    for k in (1, 2, 4, 8, 16):
        loc += [np.sin(k * lat), np.cos(k * lat), np.sin(k * lon), np.cos(k * lon)]
    loc = np.stack(loc, 1)
    n, h = d["mask"].shape

    def pad(x):
        out = np.zeros((n, h, x.shape[1]), np.float32)
        out[photo, slot] = x
        return out

    return {"loc": pad(loc), "geo": pad(geo)}


tr = action_agent._data("train", ckpt=CKPT)
dv = action_agent._data("dev3k", stats=tr["stats"], ckpt=CKPT)
ex = {"train": extra("train", tr), "dev3k": extra("dev3k", dv)}
# glance: PCA 32 on train
g_tr, g_dv = tr["glance"].float(), dv["glance"].float()
mu = g_tr.mean(0)
_, _, V = torch.pca_lowrank(g_tr - mu, q=32, center=False)
for name, d, g in (("train", tr, g_tr), ("dev3k", dv, g_dv)):
    gl = ((g - mu) @ V).cpu().numpy()
    ex[name]["glance"] = np.repeat(gl[:, None, :], d["mask"].shape[1], 1).astype(np.float32)
# normalise each group with train statistics over real tokens
m_tr = tr["mask"].cpu().numpy()
for k in ("loc", "geo", "glance"):
    x = ex["train"][k][m_tr]
    mu_k, sd_k = x.mean(0), x.std(0) + 1e-6
    for name in ex:
        ex[name][k] = (ex[name][k] - mu_k) / sd_k


def with_groups(d, name, groups):
    v = dict(d)
    v.pop("pre_noclue", None)  # clues read: the full state
    if groups:
        add = np.concatenate([ex[name][k] for k in groups], -1)
        v["pre"] = torch.cat([d["pre"], torch.as_tensor(add, device=d["pre"].device)], -1)
    return v


base = np.asarray(dv["base_km"])
res = {}
lines = [f"dev3k {dv['n']} photos; released Pinpoint (original protocol) < 25 km {100 * (base < 25).mean():.1f}, < 1 km {100 * (base < 1).mean():.1f}"]
print(lines[0], flush=True)
for label, groups in (("base", ()), ("+loc", ("loc",)), ("+glance", ("glance",)), ("+geo", ("geo",)), ("+all", ("loc", "glance", "geo"))):
    t, v = with_groups(tr, "train", groups), with_groups(dv, "dev3k", groups)
    logits, seeds = [], []
    for s in SEEDS:
        net = fit_answer(t, np.arange(t["n"]), s)
        with torch.no_grad():
            lg = torch.log_softmax(net.answer_logits(v, v["mask"]), 1)
        logits.append(lg)
        seeds.append(100 * (v["km_zoom"].gather(1, lg.argmax(1)[:, None]).squeeze(1).cpu().numpy() < 25).mean())
    pick = torch.stack(logits).mean(0).argmax(1)
    km = v["km_zoom"].gather(1, pick[:, None]).squeeze(1).cpu().numpy()
    res[label] = km
    msg = f"{label:8s} ensemble < 25 km {100 * (km < 25).mean():.1f}, < 1 km {100 * (km < 1).mean():.1f}; seeds {' '.join(f'{x:.1f}' for x in seeds)}"
    print(msg, flush=True)
    lines.append(msg)

rng = np.random.default_rng(0)
lines.append("\npaired vs base (% < 25 km; < 1 km):")
for label in ("+loc", "+glance", "+geo", "+all"):
    g = (res[label] < 25).astype(float) - (res["base"] < 25)
    g1 = (res[label] < 1).astype(float) - (res["base"] < 1)
    m = g[rng.integers(0, len(g), (2000, len(g)))].mean(1)
    lines.append(f"  {label:8s} {100 * g.mean():+.1f} [{100 * np.percentile(m, 2.5):+.1f}, {100 * np.percentile(m, 97.5):+.1f}]; {100 * g1.mean():+.1f}; "
                 f"fixes {(g > 0).sum()}, breaks {(g < 0).sum()}")
for l in lines[-5:]:
    print(l, flush=True)
(OUT / "head_context_probe.txt").write_text("\n".join(lines) + "\n")
