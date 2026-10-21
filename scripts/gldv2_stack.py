# Do the sub-threshold head inputs stack on dev3k? base vs + diffusion + crop views (lessons 127, 89; combined_head stack) vs + GLDv2 columns too
# (lesson R10). Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_stack.py [--seeds 2] [--cut 0.7]
import argparse
import collections
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_head as gh  # noqa: E402
import gldv2_pilot as p  # noqa: E402
from geo_search_env.experiment import action_agent as aa  # noqa: E402
from geo_search_env.experiment import combined_head as ch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cut", type=float, default=0.7)
    a = ap.parse_args()
    aa.CKPT_TAG, aa.ONE_KM = ch.CKPT, 0.5
    tr = aa._data("train")
    dv = aa._data("dev3k", stats=tr["stats"])
    m_tr, m_dv = tr["mask"].cpu().numpy(), dv["mask"].cpu().numpy()
    groups = {}
    for g in ("diffusion", "crops"):
        x_tr, ref = gh._norm(ch._slot_pad("train", ch._row_cols("train", g), tr), m_tr)
        groups[g] = (x_tr, gh._norm(ch._slot_pad("dev3k", ch._row_cols("dev3k", g), dv), m_dv, ref)[0])
    tab = gh._lm_table()
    x_tr, ref = gh._norm(gh.area_cols("train", tr, a.cut, tab), m_tr)
    groups["gldv2"] = (x_tr, gh._norm(gh.area_cols("dev3k", dv, a.cut, tab), m_dv, ref)[0])
    variants = {"base": (), "+diffusion+crops": ("diffusion", "crops"), "+diffusion+crops+gldv2": ("diffusion", "crops", "gldv2")}
    res = collections.defaultdict(dict)
    for name, gs in variants.items():
        tr_v = ch._with_cols(tr, np.concatenate([groups[g][0] for g in gs], -1)) if gs else tr
        dv_v = ch._with_cols(dv, np.concatenate([groups[g][1] for g in gs], -1)) if gs else dv
        for s in range(a.seeds):
            net, pols = ch._fit(tr_v, np.arange(tr["n"]), 7 + 100 * s)
            for key, v in ch._keep(aa.evaluate(net, pols, dv_v, np.arange(dv["n"]), dv["base_km"])).items():
                res[name].setdefault(key, []).append(v)
        print(f"{name} done", flush=True)
    lines = [f"stacked sub-threshold head inputs, dev3k ({a.seeds} seeds, GLDv2 near-duplicate cut {a.cut})"]
    for name in res:
        for key, vs in res[name].items():
            lines.append(gh._line(name, key, vs, np.mean([v < 25 for v in res["base"][key]], 0)))
    text = "\n".join(lines)
    print(text, flush=True)
    (p.OUT / "stack_dev3k.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
