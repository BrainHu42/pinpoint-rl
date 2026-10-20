# Cheap screen of area-forming rules on cached setft_own pools (CPU). Choose on train subsample, report on dev3k.
# Metrics per photo with K areas: cov = some area's point < 25 km of truth; first = area 1's point < 25 km; memb = some member of the
# K areas < 25 km (ceiling for refining inside the areas). Area point: founder (current), best-scored member, weighted medoid.
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from geo_search_env.experiment.pinpoint_reference import load_pools  # noqa: E402

K = 5
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
rng = np.random.default_rng(0)


def km_matrix(a, b):
    la, lo = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    lb, lob = np.radians(b[:, 0])[None], np.radians(b[:, 1])[None]
    h = np.sin((lb - la) / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin((lob - lo) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


print("loading pools", flush=True)
pools = load_pools("mp16_setft_own")
hid_all = np.load("artifacts/pinpoint_set/mp16_32_8_8_setft_own.npz")["hidden"]


def load_set(tag, n=None):
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text())
    if n and n < len(photos):
        photos = [photos[i] for i in sorted(rng.choice(len(photos), n, replace=False))]
    out = []
    for e in photos:
        i = e["index"]
        v = pools["valid"][i]
        s = pools["scores"][i][v].astype(np.float64)
        ll = pools["latlon"][i][v].astype(np.float64)
        h = hid_all[i][v].astype(np.float32)
        h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        out.append(dict(s=s, ll=ll, d=km_matrix(ll, ll), t=km_matrix(np.asarray([e["truth"]]), ll)[0], csv=pools["csv_rows"][i][v], h=h))
    return out


sets = {"train": load_set("train_setft_own", 12000), "dev3k": load_set("dev3k_setft_own")}

# index density: MP16 gallery counts on a 0.05 deg grid, summed over a ~+-10 km box -> photos per km^2
print("density grid", flush=True)
emb = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
gl = np.fromfile(emb / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2)
C = 0.05
NY, NX = int(180 / C), int(360 / C)
iy = np.clip(((gl[:, 0] + 90) / C).astype(int), 0, NY - 1)
ix = np.clip(((gl[:, 1] + 180) / C).astype(int), 0, NX - 1)
grid = np.zeros((NY, NX), np.int64)
np.add.at(grid, (iy, ix), 1)
sat = np.zeros((NY + 1, NX + 1), np.int64)
sat[1:, 1:] = grid.cumsum(0).cumsum(1)
del grid, gl, iy, ix


def density(ll, half_km=10.0):
    lat = ll[:, 0]
    dy = max(1, round(half_km / 111.0 / C))
    dx = np.maximum(1, np.round(half_km / (111.0 * np.maximum(np.cos(np.radians(lat)), 0.05)) / C)).astype(int)
    cy = np.clip(((lat + 90) / C).astype(int), 0, NY - 1)
    cx = np.clip(((ll[:, 1] + 180) / C).astype(int), 0, NX - 1)
    y0, y1 = np.clip(cy - dy, 0, NY), np.clip(cy + dy + 1, 0, NY)
    x0, x1 = np.clip(cx - dx, 0, NX), np.clip(cx + dx + 1, 0, NX)
    cnt = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
    area = ((y1 - y0) * C * 111.0) * ((x1 - x0) * C * 111.0 * np.maximum(np.cos(np.radians(lat)), 0.05))
    return np.maximum(cnt, 1) / area


for name, ps in sets.items():
    for p in ps:
        p["rho"] = density(p["ll"])

# admin labels from MP16-Pro (csv rows)
print("admin labels", flush=True)
need = sorted({int(r) for ps in sets.values() for p in ps for r in p["csv"]})
import csv  # noqa: E402

needset = set(need)
lab = {}
with open("data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ci_, st_, co_, cy_ = (hdr.index(k) for k in ("country", "state", "county", "city"))
    for r, row in enumerate(rd):
        if r in needset:
            lab[r] = (row[ci_], row[st_], row[co_], row[cy_])
for ps in sets.values():
    for p in ps:
        L = [lab[int(r)] for r in p["csv"]]
        p["country"] = [l[0] or "?" for l in L]
        p["state"] = [(l[0], l[1] or l[2] or "?") for l in L]
        p["city"] = [(l[0], l[1], l[3] or l[2] or "?") for l in L]
print("ready", flush=True)


def weights(s, T):
    if np.isinf(T):
        return np.full(len(s), 1.0 / len(s))
    w = np.exp((s - s.max()) / T)
    return w / w.sum()


def greedy(w, near):
    alive = np.ones(len(w), bool)
    areas = []
    while alive.any() and len(areas) < K:
        sup = (near & alive[None]) @ w
        sup[~alive] = -1
        f = int(np.argmax(sup))
        m = np.flatnonzero(near[f] & alive)
        if f not in m:
            m = np.append(m, f)
        areas.append((f, m))
        alive[m] = False
    return areas


def groups(w, d, key, radius=25.0):
    by = {}
    for j, k in enumerate(key):
        by.setdefault(k, []).append(j)
    gs = sorted(by.values(), key=lambda m: -w[m].sum())[:K]
    out = []
    for m in gs:
        m = np.asarray(m)
        sup = (d[np.ix_(m, m)] < radius) @ w[m]
        out.append((int(m[np.argmax(sup)]), m))
    return out


def score(p, areas, w):
    t = p["t"]
    res = {}
    for kind in ("founder", "best", "medoid"):
        pts = []
        for f, m in areas:
            if kind == "founder":
                pts.append(f)
            elif kind == "best":
                pts.append(int(m[np.argmax(p["s"][m])]))
            else:
                pts.append(int(m[np.argmin(p["d"][np.ix_(m, m)] @ w[m])]))
        hit = t[pts] < 25
        res[kind] = (hit.any(), hit[0])
    memb = np.concatenate([m for _, m in areas])
    res["memb"] = (t[memb] < 25).any()
    return res


def run(ps, method, **kw):
    cov, first, memb, cov_b, first_b, cov_m, first_m, nar = [], [], [], [], [], [], [], []
    for p in ps:
        s, d = p["s"], p["d"]
        top = kw.get("top")
        if top:
            o = np.argsort(-s)[:top]
            q = {k: (p[k][o] if isinstance(p[k], np.ndarray) and len(p[k]) == len(s) else p[k]) for k in ("s", "ll", "t", "rho", "h")}
            q["d"] = d[np.ix_(o, o)]
            q.update({k: [p[k][i] for i in o] for k in ("country", "state", "city")})
            p, s, d = q, q["s"], q["d"]
        w = weights(s, kw.get("T", 0.2))
        if method == "greedy":
            near = d < kw.get("R", 25.0)
        elif method == "density":
            r = np.clip(np.sqrt(kw["N"] / (np.pi * p["rho"])), kw.get("rmin", 3.0), kw.get("rmax", 150.0))
            near = d < r[:, None]
        elif method == "spread":
            dm = np.average(d[np.argmax(s)], weights=w)
            near = d < np.clip(kw["c"] * dm, 3.0, 150.0)
        elif method == "merge":
            cos = p["h"] @ p["h"].T
            near = (d < 25) | ((d < kw["Rbig"]) & (cos > kw["tau"]))
        elif method == "split":
            cos = p["h"] @ p["h"].T
            near = (d < 25) & (cos > kw["tau"])
            np.fill_diagonal(near, True)
        elif method == "admin":
            areas = groups(w, d, p[kw["level"]])
        if method != "admin":
            areas = greedy(w, near)
        r = score(p, areas, w)
        cov.append(r["founder"][0]); first.append(r["founder"][1]); memb.append(r["memb"])
        cov_b.append(r["best"][0]); first_b.append(r["best"][1]); cov_m.append(r["medoid"][0]); first_m.append(r["medoid"][1])
        nar.append(len(areas))
    a = lambda x: np.asarray(x, float)
    return dict(cov=a(cov), first=a(first), memb=a(memb), cov_b=a(cov_b), first_b=a(first_b), cov_m=a(cov_m), first_m=a(first_m), n=a(nar))


def boot(x, n=2000):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return 100 * x.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


FAMILIES = {
    "radius": [("greedy", dict(R=R)) for R in (5, 10, 15, 25, 40, 60, 100)],
    "temperature": [("greedy", dict(T=T)) for T in (0.05, 0.2, 0.5, 1.0, 2.0, np.inf)],
    "pool width": [("greedy", dict(top=n)) for n in (12, 24, 36)] + [("greedy", {})],
    "density radius": [("density", dict(N=N, rmin=rmin)) for N in (30, 100, 300, 1000, 3000, 10000, 30000) for rmin in (3.0, 10.0)],
    "spread radius": [("spread", dict(c=c)) for c in (0.05, 0.1, 0.2, 0.35, 0.5)],
    "same-place merge": [("merge", dict(Rbig=Rb, tau=tau)) for Rb in (100, 300) for tau in (0.7, 0.8, 0.9, 0.95)],
    "same-place split": [("split", dict(tau=tau)) for tau in (0.3, 0.5, 0.7, 0.8)],
    "admin": [("admin", dict(level=l)) for l in ("country", "state", "city")],
}

lines = []


def show(msg):
    print(msg, flush=True)
    lines.append(msg)


for name, ps in sets.items():
    po = np.mean([(p["t"] < 25).any() for p in ps])
    t1 = np.mean([p["t"][np.argmax(p["s"])] < 25 for p in ps])
    show(f"{name}: {len(ps)} photos; reranker top-1 < 25 km {100 * t1:.1f}; pool oracle {100 * po:.1f}; mean pool {np.mean([len(p['s']) for p in ps]):.1f}")

# calibration: real restricted-search oracle of the current areas on dev3k
h = np.load("artifacts/agent/hypotheses_dev3k_setft_own.npz")
_ph = json.loads(Path("artifacts/photo_sets/dev3k_setft_own/photos.json").read_text())
truth = {i: np.asarray(_ph[i]["truth"]) for i in range(len(_ph))}
ok = np.zeros(len(_ph), bool)
for j in np.flatnonzero(h["source"] == 0):
    sc = np.where(h["rows"][j] >= 0, h["scores"][j], -np.inf)
    if np.isfinite(sc).any():
        ok[h["photo"][j]] |= km_matrix(truth[int(h["photo"][j])][None], h["latlon"][j][[int(np.argmax(sc))]])[0, 0] < 25
show(f"dev3k calibration: restricted-search oracle of the current 5 areas {100 * ok.mean():.1f}")

base = {n: run(ps, "greedy") for n, ps in sets.items()}
for n in sets:
    b = base[n]
    show(f"{n} current (R 25, T 0.2): cov {100 * b['cov'].mean():.1f} first {100 * b['first'].mean():.1f} memb {100 * b['memb'].mean():.1f} | "
         f"best-member point cov {100 * b['cov_b'].mean():.1f} first {100 * b['first_b'].mean():.1f} | medoid cov {100 * b['cov_m'].mean():.1f} "
         f"first {100 * b['first_m'].mean():.1f} | areas {b['n'].mean():.2f}")

summary = []
for fam, cfgs in FAMILIES.items():
    t0 = time.time()
    show(f"\n== {fam} (train cov / first / memb / areas | dev3k cov / first / memb)")
    best = None
    for method, kw in cfgs:
        tr = run(sets["train"], method, **kw)
        dv = run(sets["dev3k"], method, **kw)
        show(f"  {method:8s} {str({k: v for k, v in kw.items()}):38s} train {100 * tr['cov'].mean():5.1f} {100 * tr['first'].mean():5.1f} "
             f"{100 * tr['memb'].mean():5.1f} {tr['n'].mean():4.2f} | dev3k {100 * dv['cov'].mean():5.1f} {100 * dv['first'].mean():5.1f} {100 * dv['memb'].mean():5.1f}")
        if best is None or tr["cov"].mean() > best[0]:
            best = (tr["cov"].mean(), method, kw, dv)
    _, method, kw, dv = best
    c = boot(dv["cov"] - base["dev3k"]["cov"])
    f = boot(dv["first"] - base["dev3k"]["first"])
    summary.append(f"  {fam:17s} {method} {kw}: dev3k cov {c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}], first {f[0]:+.1f} [{f[1]:+.1f}, {f[2]:+.1f}]")
    print(f"  ({time.time() - t0:.0f}s)", flush=True)

show("\nTrain-chosen (by cov) per family, dev3k vs current:")
for s_ in summary:
    show(s_)
(OUT / "cluster_screen.txt").write_text("\n".join(lines) + "\n")
