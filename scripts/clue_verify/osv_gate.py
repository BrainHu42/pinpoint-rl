"""Where does OSV beat the pipeline? Per photo: street-likeness (OSV top-1 sim - MP16 top-1 sim, raw SigLIP2), OSV agreement (similarity
mass of the best 25 km cluster of the OSV top 50), pipeline confidence; OSV answer = the best cluster's centre member. Bins, and a 5-fold
CV switch rule (OSV if street-like, agreeing and the pipeline unsure).

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_gate.py [contrastive|raw]
"""
import sys

import numpy as np

from geo_search_env.experiment.clue_verify import _haversine

space = sys.argv[1] if len(sys.argv) > 1 else "contrastive"
d = np.load(f"artifacts/osv_propose/search{'_contrastive' if space == 'contrastive' else ''}.npz")
raw = np.load("artifacts/osv_propose/search.npz")
ans = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26.npz")


def best_cluster(ll, s, r=25.0):
    D = _haversine(ll[:, None, 0], ll[:, None, 1], ll[None, :, 0], ll[None, :, 1])
    w = np.exp((s - s[0]) / 0.02)
    mass = (w[None] * (D < r)).sum(1)
    j = int(np.argmax(mass))
    return ll[j], mass[j] / w.sum()


for st, key in (("dev3k", "dev3k|RL policy, lambda 0"), ("commons26", "commons26:setft_own|RL policy, lambda 0")):
    m = np.flatnonzero(d["set"] == st)
    truth, ph = d["truth"][m], d["photo"][m]
    pipe = ans[key][ph]
    conf = ans[key + "|conf"][ph]
    street = raw["osv_sim"][m, 0] - raw["mp16_sim"][m, 0]
    agree, osv_err = np.zeros(len(m)), np.zeros(len(m))
    for j, q in enumerate(m):
        c, agree[j] = best_cluster(d["osv_ll"][q], d["osv_sim"][q])
        osv_err[j] = _haversine(truth[j, 0], truth[j, 1], c[0], c[1])
    print(f"== {st} ({len(m)}), {space}: < 25 km pipeline {100 * np.mean(pipe < 25):.1f}%, OSV best cluster {100 * np.mean(osv_err < 25):.1f}%; "
          f"< 1 km {100 * np.mean(pipe < 1):.1f} / {100 * np.mean(osv_err < 1):.1f}")
    for name, f in (("street-likeness", street), ("OSV agreement", agree), ("pipeline conf", conf)):
        qs = np.quantile(f, [0, .5, .75, .9, .95, 1])
        print(f"  by {name}:")
        for lo, hi in zip(qs[:-1], qs[1:]):
            b = (f >= lo) & (f <= hi)
            print(f"    [{lo:+.3f},{hi:+.3f}] n={b.sum():4d}  pipeline {100 * np.mean(pipe[b] < 25):5.1f}  OSV {100 * np.mean(osv_err[b] < 25):5.1f}  "
                  f"OSV right & pipeline wrong {100 * np.mean((osv_err[b] < 25) & (pipe[b] >= 25)):4.1f}  "
                  f"reverse {100 * np.mean((osv_err[b] >= 25) & (pipe[b] < 25)):4.1f}")
    fold = np.random.default_rng(0).integers(0, 5, len(m))
    grid = [(a, b, c) for a in np.quantile(street, [0, .5, .75, .9]) for b in np.quantile(agree, [0, .5, .75, .9])
            for c in np.quantile(conf, [.25, .5, .75]).tolist() + [np.inf]]

    def gain(sw, idx):
        return np.mean(np.where(sw, osv_err[idx] < 25, pipe[idx] < 25)) - np.mean(pipe[idx] < 25)

    tot, switched = 0.0, 0
    for k in range(5):
        tr, te = fold != k, fold == k
        g = max(grid, key=lambda g: gain((street[tr] >= g[0]) & (agree[tr] >= g[1]) & (conf[tr] <= g[2]), tr))
        sw = (street[te] >= g[0]) & (agree[te] >= g[1]) & (conf[te] <= g[2])
        tot += gain(sw, te) * te.sum()
        switched += sw.sum()
    print(f"  CV switch to OSV: {100 * tot / len(m):+.2f} pts < 25 km, switched {100 * switched / len(m):.1f}%")
