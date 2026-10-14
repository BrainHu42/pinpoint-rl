"""Lesson 84: is the 4B's region guess calibrated? Majority region over 16 samples vs truth, where it disagrees with Pinpoint's top-1 region."""
import json
from collections import Counter

import numpy as np

from geo_search_env.tools.pinpoint import Pinpoint

rows = json.loads(open("artifacts/agent/nested_passk_dev3k.json").read())
z = np.load("artifacts/agent/nested_passk_dev3k_answers.npz")
tool = Pinpoint(targets=None)
pc, pr = tool.labels_at(np.asarray([r["top1"] for r in rows]))
group = np.asarray([r["group"] for r in rows])
w = np.where(group == "fixable", 302 / 3000 / (group == "fixable").sum(), 2698 / 3000 / (group == "control").sum())
for level, S, T, P in (("country", z["sample_country"], z["truth_country"], pc), ("region", z["sample_region"], z["truth_region"], pr)):
    maj, share = [], []
    for i in range(len(rows)):
        c = Counter(x for x in S[i] if x)
        m, n = c.most_common(1)[0] if c else ("", 0)
        maj.append(m)
        share.append(n / S.shape[1])
    maj, share = np.asarray(maj), np.asarray(share)
    print(f"\n{level}: Pinpoint's {level} right {np.average(P == T, weights=w):.1%} (dev3k-weighted)")
    for t in (0.25, 0.5, 0.625, 0.75, 0.875):
        o = (share >= t) & (maj != P) & (maj != "")
        fix = o & (maj == T)
        brk = o & (P == T)
        print(f"  majority share >= {t}: overrides {np.average(o, weights=w):.1%} of photos; precision {np.average(maj[o] == T[o], weights=w[o]) if o.any() else 0:.1%}; "
              f"fixes {np.average(fix, weights=w):.2%}, breaks {np.average(brk, weights=w):.2%} (n={o.sum()})")
