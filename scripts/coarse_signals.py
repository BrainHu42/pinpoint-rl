"""VALUE_PLAN regime idea R3 (user, 2026-10-24): signals closed at 25 km, judged at coarse scales. Each was tested as a 25 km signal and
found useless where retrieval fails; lesson V12 showed a region question needs a coarse judge independent of retrieval. Cached on dev3k:
the 4B's country guess (`observe_dev3k_4b.json`), OSV-5M's 50 nearest street views (`osv_propose/osv_dev3k_setft_own.npz`), the read
chain's text-search hits with coordinates (`region_evidence_dev3k_4b.json`; its `km` field is the distance to the truth, a label, never
read), next to the PE-G region probe (and R4's region experts, --experts).

alone: per signal its own coarse guess (probe top cell, OSV top-1 / mass point, the heaviest text hit, the 4B's country) right within 200 /
       750 / 2500 km (country: same country as the truth), on all photos and on area-oracle misses, with coverage.
boxes: as lesson V12, each signal's support inside each of V9's 7 hedge boxes picks one box; region / place on area-oracle misses, net of
       nearby-truth luck; also each signal as a tie-break on the probe (mean of the two ranks).

  PYTHONPATH=src .venv/bin/python scripts/coarse_signals.py [--experts artifacts/value/region_expert_b2_8192.npz ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hedge_search as hs  # noqa: E402

ALIASES = {"usa": "united states", "us": "united states", "united states of america": "united states", "uk": "united kingdom",
           "england": "united kingdom", "scotland": "united kingdom", "wales": "united kingdom", "northern ireland": "united kingdom",
           "great britain": "united kingdom", "czech republic": "czechia", "korea": "south korea", "republic of korea": "south korea",
           "holland": "netherlands", "the netherlands": "netherlands", "vatican city": "vatican", "burma": "myanmar", "ivory coast": "cote d'ivoire"}


def _norm(c) -> str:
    c = (c or "").strip().lower()
    return ALIASES.get(c, c)


def _country(latlon: np.ndarray) -> np.ndarray:
    """Country name (lower case) of the nearest GeoNames place (population >= 1,000) per coordinate."""
    from geo_search_env.viz.places import _tree, _xyz

    tree, _, country = _tree()
    ll = np.asarray(latlon, np.float64).reshape(-1, 2)
    _, i = tree.query(_xyz(ll[:, 0], ll[:, 1]))
    return np.asarray([_norm(c) for c in country[i]])


def run(experts: list[str]) -> None:
    tag = "dev3k"
    photos = hs._photos(tag)
    ids = [e["image_id"] for e in photos]
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    n = len(truth)
    area = hs._area_oracle(tag)
    miss = ~area
    z = np.load(hs.OUT / f"hedge_search_{tag}.npz")
    pts, ll = z["points"].astype(np.float64), z["latlon"].astype(np.float64)
    K = pts.shape[1]

    # signals
    probes = {"region probe": (np.exp(np.load(hs.OUT / f"probe_logp_{tag}.npy").astype(np.float64)),
                               np.load("artifacts/retrieval/probe/cells.npy").astype(np.float64))}
    for f in experts:
        e = np.load(f)
        probes[Path(f).stem.replace("region_expert_", "expert ")] = (np.exp(e["logp"].astype(np.float64)), e["cells"].astype(np.float64))
    osv = np.load("artifacts/osv_propose/osv_dev3k_setft_own.npz")
    o_ll, o_sim = osv["latlon"].astype(np.float64), osv["sim"].astype(np.float64)
    o_w = np.exp(20 * (o_sim - o_sim.max(1, keepdims=True)))
    o_w /= o_w.sum(1, keepdims=True)
    text = defaultdict(list)
    for r in json.loads(Path("artifacts/agent/region_evidence_dev3k_4b.json").read_text(encoding="utf-8")):
        if r.get("lat") is not None:
            text[r["image_id"]].append((r["lat"], r["lon"], float(r.get("weight") or 0.0)))
    t_has = np.asarray([i in text for i in ids])
    T = max(len(v) for v in text.values())
    t_ll = np.full((n, T, 2), np.nan)
    t_w = np.zeros((n, T))
    for p, i in enumerate(ids):
        for j, (a, b, w) in enumerate(text.get(i, [])):
            t_ll[p, j], t_w[p, j] = (a, b), w
    obs = json.loads(Path("artifacts/agent/observe_dev3k_4b.json").read_text(encoding="utf-8"))
    guess = np.asarray([_norm((obs.get(i, {}).get("parsed") or {}).get("country")) for i in ids])
    g_has = (guess != "") & (guess != "unknown")
    t_country = _country(truth)

    def acc(guess_ll, has):
        d = hs._km(truth, guess_ll)
        return [(100 * np.mean((d < r) & has), 100 * np.mean(((d < r) & has)[miss])) for r in (200, 750, 2500)]

    print(f"{tag}: {n} photos, area-oracle misses {miss.mean():.1%}; coverage: OSV 100%, text hits {t_has.mean():.1%} "
          f"({t_has[miss].mean():.1%} of misses), 4B country {g_has.mean():.1%} ({g_has[miss].mean():.1%} of misses)")
    print("\n alone: own coarse guess right within 200 / 750 / 2500 km, % of all photos | % of area-oracle misses")
    rows = {}
    for name, (prob, cent) in probes.items():
        rows[f"{name}, top cell"] = acc(cent[prob.argmax(1)], np.ones(n, bool))
    rows["OSV top-1"] = acc(o_ll[:, 0], np.ones(n, bool))
    d_oo = 6371.0 * np.arccos(np.clip(np.einsum("nid,njd->nij", hs._xyz(o_ll), hs._xyz(o_ll)), -1, 1))
    rows["OSV mass point (500 km)"] = acc(o_ll[np.arange(n), ((d_oo < 500) * o_w[:, None, :]).sum(-1).argmax(1)], np.ones(n, bool))
    rows["text, heaviest hit"] = acc(t_ll[np.arange(n), t_w.argmax(1)], t_has)
    for name, a in rows.items():
        print(f"   {name:28s} " + "  ".join(f"{x:5.1f} | {y:5.1f}" for x, y in a))
    right_c = g_has & (guess == t_country)
    print(f"   {'4B country = true country':28s} {100 * right_c.mean():5.1f} | {100 * right_c[miss].mean():5.1f}   "
          f"(when it names one: {right_c[g_has].mean():.1%} | {right_c[g_has & miss].mean():.1%})")

    # box judges
    dt = hs._km(truth[:, None, :], truth[None, :, :])
    rng = np.random.default_rng(1)
    near = np.asarray([rng.choice(c) if len(c := np.flatnonzero((dt[i] > 100) & (dt[i] < 750))) else i for i in range(n)])
    box_country = _country(pts.reshape(-1, 2)).reshape(n, K)
    for r_i, r in enumerate(hs.RADII):
        reg = hs._km(truth[:, None, :], pts) < r
        place = hs._km(truth[:, None, :], ll[:, :, r_i, 0]) < 25
        lucky = hs._km(truth[near][:, None, :], ll[:, :, r_i, 0]) < 25
        judges = {}
        for name, (prob, cent) in probes.items():
            dc = 6371.0 * np.arccos(np.clip(np.einsum("nkd,cd->nkc", hs._xyz(pts), hs._xyz(cent)), -1, 1))
            judges[f"{name} mass"] = np.einsum("nkc,nc->nk", (dc < r).astype(np.float64), prob)
        d_ob = 6371.0 * np.arccos(np.clip(np.einsum("nkd,nmd->nkm", hs._xyz(pts), hs._xyz(o_ll)), -1, 1))
        judges["OSV mass"] = np.einsum("nkm,nm->nk", (d_ob < r).astype(np.float64), o_w)
        tl = np.where(np.isfinite(t_ll), t_ll, 0.0)
        d_tb = 6371.0 * np.arccos(np.clip(np.einsum("nkd,nmd->nkm", hs._xyz(pts), hs._xyz(tl)), -1, 1))
        judges["text weight"] = np.einsum("nkm,nm->nk", ((d_tb < r) & np.isfinite(t_ll[:, None, :, 0])).astype(np.float64), t_w)
        judges["4B country match"] = (box_country == guess[:, None]).astype(np.float64)
        base = judges["region probe mass"]
        rk = lambda s: np.argsort(np.argsort(s + 1e-9 * np.random.default_rng(2).random(s.shape), 1), 1)
        for sname in ("OSV mass", "text weight", "4B country match") + tuple(f"{e} mass" for e in probes if e != "region probe"):
            judges[f"probe + {sname}"] = rk(base) + rk(judges[sname])
        judges["random box"] = np.random.default_rng(0).random((n, K))
        print(f"\n boxes +-{r:.0f} km, area-oracle misses: region / place [nearby luck] net; and on misses where the signal exists")
        for name, s in judges.items():
            k = s.argmax(1)
            rr, pp, lk = reg[np.arange(n), k], place[np.arange(n), k], lucky[np.arange(n), k]
            has = t_has if "text" in name else g_has if "country" in name else np.ones(n, bool)
            m2 = miss & has
            kb = base.argmax(1)  # the probe's pick on the same covered photos
            print(f"   {name:34s} {100 * rr[miss].mean():5.1f} / {100 * pp[miss].mean():4.1f} [{100 * lk[miss].mean():3.1f}] "
                  f"{100 * (pp[miss].mean() - lk[miss].mean()):+5.1f}   | covered {m2.mean() / miss.mean():4.0%}: "
                  f"{100 * rr[m2].mean():5.1f} / {100 * pp[m2].mean():4.1f} [{100 * lk[m2].mean():3.1f}]   probe there "
                  f"{100 * reg[np.arange(n), kb][m2].mean():5.1f} / {100 * place[np.arange(n), kb][m2].mean():4.1f} "
                  f"[{100 * lucky[np.arange(n), kb][m2].mean():3.1f}]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", nargs="*", default=[])
    args = ap.parse_args()
    run(args.experts)


if __name__ == "__main__":
    main()
