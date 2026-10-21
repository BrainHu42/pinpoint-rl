# GLDv2 full index (lesson R10): the 1 km refinement and the 25 km switch on commons26 dev and dev3k, thresholds chosen on the other set.
# Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_eval.py [near-duplicate cut, default 0.9]   (needs gldv2_index search --set commons26dev / dev3k)
#   refine: among the query's top landmarks (best-photo MegaLoc similarity over the whole index), the best one within 25 km of the pipeline's
#           answer; move the answer there when its similarity >= t.
#   switch: the overall top landmark when it lies > 25 km from the answer and its similarity >= t.
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_pilot as p  # noqa: E402

OUT = p.OUT


def _answers(name: str):
    """truth [n, 2], the pipeline's answer [n, 2], its km [n] (commons26 dev: action_agent apply; dev3k: the viewer's traces, lesson 109 head)."""

    if name == "commons26dev":
        _, truth, centre, km = p._photos()
        return truth, centre, km
    rows = []
    for f in sorted((f for f in Path("artifacts/viz/traces/dev3k").glob("*.json") if f.stem.isdigit()), key=lambda x: int(x.stem)):
        t = json.loads(f.read_text())
        fin = t["agent"]["0"]["final"]
        rows.append((t["truth"]["lat"], t["truth"]["lon"], fin["lat"], fin["lon"], fin["km"]))
    a = np.asarray(rows, np.float64)
    return a[:, :2], a[:, 2:4], a[:, 4]


def _landmark_lists(z, cut: float, k: int = 64):
    """Per query, the top-k landmarks by best photo after dropping hits more similar than `cut` (near-duplicates of the query)."""

    S, L = z["photo_sim"].astype(np.float32), z["photo_lm"]
    n = len(S)
    lm = np.zeros((n, k), np.int64)
    sim = np.full((n, k), -2.0, np.float32)
    for q in range(n):
        keep = S[q] <= cut
        u, first = np.unique(L[q][keep], return_index=True)
        o = np.argsort(first)[:k]
        lm[q, : len(o)], sim[q, : len(o)] = u[o], S[q][keep][first[o]]
    return lm, sim


def _load(name: str, cut: float) -> dict:
    z = np.load(OUT / f"search_{name}.npz")
    truth, ans, km = _answers(name)
    lm, sim = _landmark_lists(z, cut)
    tab = _table()
    ll = tab[lm]
    n, k = sim.shape
    d_ans = p._hav(np.repeat(ans[:, :1], k, 1), np.repeat(ans[:, 1:], k, 1), ll[..., 0], ll[..., 1])
    d_truth = p._hav(np.repeat(truth[:, :1], k, 1), np.repeat(truth[:, 1:], k, 1), ll[..., 0], ll[..., 1])
    sim = np.where(np.isfinite(d_ans), sim, -2.0)
    # refinement candidate: best landmark within 25 km of the answer
    near = np.where(d_ans < 25, sim, -2.0)
    j = near.argmax(1)
    r_sim, r_dt = near[np.arange(n), j], d_truth[np.arange(n), j]
    # switch candidate: the overall top landmark if > 25 km from the answer
    s_sim = np.where(d_ans[:, 0] > 25, sim[:, 0], -2.0)
    return {"n": n, "km": km, "r_sim": r_sim, "r_dt": r_dt, "s_sim": s_sim, "s_dt": d_truth[:, 0]}


def _apply(d: dict, kind: str, t: float) -> tuple[int, int, int, np.ndarray]:
    km = d["km"]
    if kind == "refine":
        go = d["r_sim"] >= t
        new = np.where(go, d["r_dt"], km)
        r = 1.0
    else:
        go = d["s_sim"] >= t
        new = np.where(go, d["s_dt"], km)
        r = 25.0
    return int(go.sum()), int(((new < r) & (km >= r)).sum()), int(((new >= r) & (km < r)).sum()), new


def _table() -> np.ndarray:
    ids, ll = p._landmarks()
    tab = np.full((ids.max() + 1, 2), np.nan)
    tab[ids] = ll
    return tab


def main() -> None:
    cut = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    print(f"near-duplicate cut {cut}")
    for name in ("dev3k", "commons26dev"):  # the top-1 photo similarity: commons26 (2026 photos) can't be in GLDv2 (2019), dev3k can
        top = np.load(OUT / f"search_{name}.npz")["photo_sim"][:, 0].astype(np.float32)
        print(f"  {name} top-1 photo similarity quantiles 50/90/99/99.9/max: " + " / ".join(f"{x:.3f}" for x in np.quantile(top, [0.5, 0.9, 0.99, 0.999, 1])))
    sets = {s: _load(s, cut) for s in ("dev3k", "commons26dev")}
    for name, d in sets.items():
        print(f"{name}: {d['n']} photos, pipeline < 1 km {100 * np.mean(d['km'] < 1):.1f}%, < 25 km {100 * np.mean(d['km'] < 25):.1f}%")
    for kind, r in (("refine", 1), ("switch", 25)):
        print(f"\n{kind} (scored at {r} km): threshold chosen on one set (best net fixes), applied to the other")
        for src, dst in (("dev3k", "commons26dev"), ("commons26dev", "dev3k")):
            s = sets[src]
            key = "r_sim" if kind == "refine" else "s_sim"
            grid = np.unique(np.quantile(s[key][s[key] > -2], np.linspace(0, 0.995, 200)))
            nets = [(_apply(s, kind, t)[1] - _apply(s, kind, t)[2], t) for t in grid]
            best_net, t = max(nets)
            go, f, b, _ = _apply(sets[dst], kind, t)
            print(f"  t {t:.3f} from {src} (there {100 * best_net / s['n']:+.2f} pts) -> {dst}: moves {go}, fixes {f}, breaks {b} = "
                  f"{100 * (f - b) / sets[dst]['n']:+.2f} pts at {r} km")
        for name, d in sets.items():
            key = "r_sim" if kind == "refine" else "s_sim"
            line = []
            for q in (0.5, 0.8, 0.9, 0.95):
                t = np.quantile(d[key][d[key] > -2], q)
                go, f, b, _ = _apply(d, kind, t)
                line.append(f"t {t:.3f}: {go} moves {f}/{b} ({100 * (f - b) / d['n']:+.2f})")
            print(f"  {name} sweep: " + " | ".join(line))


if __name__ == "__main__":
    main()
