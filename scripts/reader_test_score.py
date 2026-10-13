"""Score the blind reader test: my notes vs the 4B's observe output vs Pinpoint (retrained reranker top-1), against the truth."""
import json
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np

tmp = Path("artifacts/agent/reader_test")
key = {k["p"]: k for k in json.loads((tmp / "reader_key" / "key.json").read_text())}
notes = json.loads(Path("fixtures/reader_test_dev3k.json").read_text())  # blind readings (lesson 80)
obs = json.loads(Path("artifacts/agent/observe_dev3k_4b.json").read_text())
queries = json.loads(Path("artifacts/agent/queries_dev3k_4b.json").read_text())
photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_a/photos.json").read_text())
con = sqlite3.connect("file:/opt/dlami/nvme/text_search/places.db?mode=ro", uri=True)
towns = con.execute("SELECT lat, lon, name, cc, pop FROM towns WHERE pop >= 5000").fetchall()
T = np.asarray([(t[0], t[1]) for t in towns])
CC = {"Australia": "AU", "France": "FR", "Denmark": "DK", "Malaysia": "MY", "United States": "US", "Botswana": "BW", "Ireland": "IE",
      "Bahamas": "BS", "Sweden": "SE", "Pakistan": "PK", "Germany": "DE", "Mexico": "MX", "United Kingdom": "GB", "Spain": "ES",
      "Slovakia": "SK", "USA": "US", "UK": "GB", "England": "GB", "Canada": "CA", "Italy": "IT", "Netherlands": "NL", "Japan": "JP",
      "China": "CN", "India": "IN", "Brazil": "BR", "Norway": "NO", "Finland": "FI", "Poland": "PL", "Portugal": "PT", "Greece": "GR",
      "Belgium": "BE", "Switzerland": "CH", "Austria": "AT", "Czech Republic": "CZ", "Turkey": "TR", "New Zealand": "NZ", "Thailand": "TH",
      "South Africa": "ZA", "Iceland": "IS", "Croatia": "HR", "Hungary": "HU", "Russia": "RU", "Argentina": "AR", "Chile": "CL",
      "Peru": "PE", "Kenya": "KE", "Egypt": "EG", "Morocco": "MA", "Indonesia": "ID", "Philippines": "PH", "Vietnam": "VN",
      "South Korea": "KR", "Taiwan": "TW", "Singapore": "SG", "Scotland": "GB", "Wales": "GB", "Israel": "IL", "Jordan": "JO"}


def km(a, b):
    p1, p2 = np.radians(a[0]), np.radians(b[:, 0])
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(b[:, 1] - a[1]) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def country_at(ll):
    return towns[int(np.argmin(km(ll, T)))][3]


def fold(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


by_name = {}
for t in towns:
    by_name.setdefault(fold(t[2]), []).append(t)


def geocode(place):
    for part in [x.strip() for x in place.split(",")]:
        hits = by_name.get(fold(part))
        if hits:
            t = max(hits, key=lambda t: t[4])
            return np.asarray([t[0], t[1]])
    return None


rows = []
for n in notes:
    k = key[n["p"]]
    truth = np.asarray(k["truth"], np.float64)
    true_cc = country_at(truth)
    pin = np.asarray(photos[k["index"]]["candidates"][0][1:], np.float64)
    parsed = (obs.get(k["image_id"]) or {}).get("parsed") or {}
    my_cc = CC.get(n["country"]) if n["country"] else None
    b4_cc = CC.get(parsed.get("country")) if parsed.get("country") else None
    g = geocode(n["guess_place"]) if n["guess_place"] else None
    rows.append({"p": n["p"], "tier": n["tier"], "conf": n["conf"], "explicit": n["explicit"], "true_cc": true_cc,
                 "near_town": towns[int(np.argmin(km(truth, T)))][2], "pin_cc": country_at(pin), "pin_km": float(km(truth, pin[None])[0]),
                 "my_cc": my_cc, "b4_cc": b4_cc, "b4_country_raw": parsed.get("country"), "my_km": None if g is None else float(km(truth, g[None])[0]),
                 "my_guess": n["guess_place"] or n["guess"], "b4_text": parsed.get("text") or [], "b4_names": parsed.get("place_names") or [],
                 "b4_queries": queries.get(k["image_id"]) or []})

print(f"{'p':4s} {'tier':4s} {'truth':18s} {'Pinpoint':12s} {'me':18s} {'4B country':12s} my guess / 4B text, names, queries")
for r in rows:
    me = f"{r['my_cc'] or '-'}" + (f" {r['my_km']:.0f} km" if r["my_km"] is not None else "")
    print(f"{r['p']:4s} {r['tier']:4s} {r['true_cc'] + ' ' + r['near_town'][:14]:18s} {r['pin_cc'] + ' ' + str(int(r['pin_km'])) + ' km':12s} {me:18s} "
          f"{(r['b4_cc'] or str(r['b4_country_raw'])[:10]):12s} {str(r['my_guess'])[:40]} | 4B: {r['b4_text'][:3]} {r['b4_names'][:2]} {r['b4_queries'][:2]}")

n = len(rows)


def acc(key_):
    ans = [r for r in rows if r[key_]]
    return len(ans), sum(r[key_] == r["true_cc"] for r in ans)


for label, k_ in (("me", "my_cc"), ("4B", "b4_cc"), ("Pinpoint top-1", "pin_cc")):
    a, c = acc(k_)
    print(f"country: {label:14s} answers {a}/{n}, right {c} ({c / max(a, 1):.0%} of answers, {c / n:.0%} of all)")
mine = [r for r in rows if r["my_km"] is not None]
print(f"my place guesses: {len(mine)}; within 25 km {sum(r['my_km'] < 25 for r in mine)}, 200 km {sum(r['my_km'] < 200 for r in mine)}, "
      f"750 km {sum(r['my_km'] < 750 for r in mine)}")
print(f"explicit place-specific clues (my reading): {sum(r['explicit'] for r in rows)}; tiers: " +
      ", ".join(f"{t} {sum(r['tier'] == t for r in rows)}" for t in "PCRNKX"))
print(f"Pinpoint top-1 within 200 km {sum(r['pin_km'] < 200 for r in rows)}, 750 km {sum(r['pin_km'] < 750 for r in rows)}")
