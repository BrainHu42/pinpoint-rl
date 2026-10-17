import json, numpy as np
rows = json.load(open("/home/brianhu/pinpoint-rl/artifacts/clue_verify/commons26_dev_osv.json"))
for kind in ("text", "area"):
    pl = [p for r in rows for p in r["places"] if p["kind"].startswith(kind)]
    n1 = np.asarray([p["n_1"] for p in pl]); right = np.asarray([p["right"] for p in pl])
    print(f"{kind}: OSV images within 1 km per place: median {np.median(n1):.0f}, mean {n1.mean():.1f}; >5: {100*np.mean(n1>5):.1f}%; "
          f"right places median {np.median(n1[right]):.0f}, >5 {100*np.mean(n1[right]>5):.1f}%")
    ml = np.asarray([p["ml"] for p in pl])
    for lo, hi in ((1, 5), (6, 20), (21, 10**6)):
        m = (n1 >= lo) & (n1 <= hi) & (ml > -1)
        if m.sum():
            print(f"   n_1 {lo}-{hi}: {m.sum()} places, MegaLoc best-of-5 p90 right {np.quantile(ml[m & right], .9) if (m & right).sum() else float('nan'):.3f} / wrong {np.quantile(ml[m & ~right], .9):.3f}")
