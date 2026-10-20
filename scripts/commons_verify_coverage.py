# Commons coverage at right vs wrong candidate places (retrieval track, lesson R7); needs commons_verify pairs + fetch (refs.json)
# Usage: PYTHONPATH=src .venv/bin/python scripts/commons_verify_coverage.py
import json
import numpy as np
O = "artifacts/retrieval/commons_verify/"
ps = json.load(open(O + "pairs.json")); refs = json.load(open(O + "refs.json"))
rows = []
for p in ps:
    for j, r in enumerate(p["right"]):
        rs = refs.get(f"{p['image_id']}|{j}")
        if rs is None:
            continue
        rows.append((j, r, len(rs), min([x["dist"] for x in rs], default=np.inf)))
a = np.array(rows, dtype=float)
for lab, m in (("pipeline answer, right", (a[:, 0] == 0) & (a[:, 1] == 1)), ("pipeline answer, wrong", (a[:, 0] == 0) & (a[:, 1] == 0)),
               ("text place, right", (a[:, 0] > 0) & (a[:, 1] == 1)), ("text place, wrong", (a[:, 0] > 0) & (a[:, 1] == 0))):
    print(f"{lab:24s} {m.sum():5d}: 10 refs {100 * np.mean(a[m, 2] == 10):5.1f}%, none {100 * np.mean(a[m, 2] == 0):5.1f}%, nearest ref median "
          f"{np.median(a[m, 3][np.isfinite(a[m, 3])]):.0f} m")
# fixable photos: pipeline wrong, a text place right
fix = [p for p in ps if not p["right"][0] and any(p["right"][1:])]
print("fixable photos", len(fix), "with >=1 ref at a right text place:",
      sum(any(p["right"][j] and refs.get(f"{p['image_id']}|{j}") for j in range(1, len(p["right"]))) for p in fix))
