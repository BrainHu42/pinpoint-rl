"""Read-chain funnel on commons26 dev: read -> searched -> a search result < 25 km -> judged yes -> final answer < 25 km."""
import collections
import csv
import json
import random

import numpy as np

rows = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
photos = json.load(open("artifacts/photo_sets/commons26_setft_own/photos.json"))
z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26.npz")
ids_all = [e["image_id"] for e in photos]
final = dict(zip(ids_all, z["commons26:setft_own|RL policy, lambda 0"]))
rt = dict(zip(ids_all, z["commons26:setft_own|retrained"]))
obs = json.load(open("artifacts/agent/observe_commons26_4b.json"))
qs = dict(json.load(open("artifacts/agent/queries_commons26_4b.json")))
ev = collections.defaultdict(list)
for r in json.load(open("artifacts/agent/region_evidence_commons26_4b.json")):
    ev[r["image_id"]].append(r)
judge = {(j["image_id"], j["query"]): j["p_yes"] for j in json.load(open("artifacts/agent/clue_judge_commons26_4b_4b.json"))}
dev = [i for i, r in rows.items() if r["split"] == "dev"]


def stats(ids, label):
    n = len(ids)
    has_text = sum(bool(obs[i].get("parsed", {}).get("text") or obs[i].get("parsed", {}).get("place_names")) for i in ids)
    has_q = sum(bool(qs.get(i)) for i in ids)
    found = [i for i in ids if any(e.get("km", 1e9) < 25 for e in ev[i])]
    found_top = [i for i in ids if any(e.get("km", 1e9) < 25 and e.get("share", 0) >= 0.5 for e in ev[i])]
    judged = [i for i in found if any(e.get("km", 1e9) < 25 and (judge.get((i, e["query"])) or 0) >= 0.5 for e in ev[i])]
    fin = sum(final[i] < 25 for i in ids)
    base = sum(rt[i] < 25 for i in ids)
    ff = np.mean([final[i] < 25 for i in found]) if found else 0
    fr = np.mean([rt[i] < 25 for i in found]) if found else 0
    print(f"{label:16s} n={n:4d} read text/names {has_text / n:5.1%} queries {has_q / n:5.1%} | result<25km {len(found) / n:5.1%} "
          f"(share>=.5 {len(found_top) / n:5.1%}) judged-yes {len(judged) / n:5.1%} | final<25 {fin / n:5.1%} (retrained {base / n:5.1%}) "
          f"| among found: final {ff:5.1%} retrained {fr:5.1%}")


stats(dev, "dev all")
for t in ("landmark", "city", "region", "none"):
    stats([i for i in dev if rows[i]["tier"] == t], f"tier {t}")
for v in ("1", "0"):
    stats([i for i in dev if rows[i]["place_text"] == v], f"place_text={v}")
src = collections.Counter(e["source"] for i in dev for e in ev[i] if e.get("km", 1e9) < 25)
allsrc = collections.Counter(e["source"] for i in dev for e in ev[i])
print("results < 25 km by source:", [(k, v, allsrc[k]) for k, v in src.most_common()])
miss = [i for i in dev if rows[i]["place_text"] == "1" and final[i] >= 25]
print("place_text=1 misses:", len(miss))
random.seed(0)
for i in random.sample(miss, 15):
    p = obs[i].get("parsed", {})
    print(" ", i, rows[i]["country"], rows[i]["tier"], "| text", (p.get("text") or [])[:4], "| names", (p.get("place_names") or [])[:3],
          "| q", (qs.get(i) or [])[:2], "| best km", round(min([e["km"] for e in ev[i] if "km" in e], default=-1)))
