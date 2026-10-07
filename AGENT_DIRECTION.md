# Direction: an agent that geolocates like a professional (agreed 2026-10-07)

**Goal.** Train an RL agent that does what professional geolocators do: eliminate places with rule-like clues and
structured map queries, then verify, instead of asking "what does this look like?". Retrieval stays as one tool (the
candidate generator); the agent's job is to win on the photos where visual similarity fails.

## Why (evidence, full im2gps3k + yfcc4k eval halves, n = 3,795)
- Every chooser, evidence source and comparator so far lands at -0.5 to +1.3 pts over the reranker top-1 (16.7 / 38.2 %
  within 1 / 25 km), because each re-encodes what visual retrieval already knows (LEARNINGS summary, lessons 1-3, 17-19, 51).
- The target is concrete: in **792 photos** (20.9 %) the reranker is wrong but a pool candidate is within 25 km. Picking the
  best-ranked candidate in the **true region** would fix **413** of them: **+10.8 pts** at 25 km [+9.8, +11.8] (true city
  +8.4, true country +3.3; lesson 52). Knowing the region is enough; the hard part is knowing it on these photos.
- Knowledge in the weights doesn't get there. The knowledge SFT (250k photos) names the right region on 52 % of photos
  the reranker gets right but **10.9 %** of the 413 (base 4B 9.0 %); as a chooser it adds -0.2 to -0.1 (lesson 52).
- It isn't a coverage problem (lesson 53). The 413 come from well-covered regions (median 872 training photos in the true
  region vs 1,073 for reranker-right photos). With 300-1,500 training photos in the region, the model names it 55 %
  on reranker-right photos but **6 %** on the 413. These photos don't look like their region, for the VLM or for retrieval.
- So the next gain has to come from clues that aren't overall appearance. That is what professionals rely on.

## The target photos (the 413, eval halves; measure only, never develop on them)
- Definition: reranker top-1 >= 25 km off, and the best-ranked pool candidate in the photo's true (country, region) is < 25 km.
- 280 from yfcc4k (12.1 % of it), 133 from im2gps3k (9.0 %): personal, often generic photos more than landmarks.
- Far misses, not near-misses: top-1 error > 200 km for 77 %, > 750 km for 45 %. But the top-1 is in the true country for
  62 %: the typical failure is the wrong region inside the right country (United States 140, UK 25, Spain 19, China 17,
  Taiwan 17, Japan 16). 23 % are > 2,500 km off (wrong continent).
- The fix is often deep: the first true-region candidate is rank 2 for 20 %, ranks 3-6 for 31 %, rank 7+ for 49 %;
  pools span a median of 8 regions.
- So the clues that matter most are region-level within a country (plates, route markers and road signs, place names in
  text, feature co-occurrence); country rules (driving side, script) cover the wrong-continent quarter.
- An upper bound: it assumes perfect region knowledge and right labels, and in large regions the 25 km hit partly relies on
  retrieval's order within the region.

## How professionals work
Hypothesise, eliminate, verify. Each clue rules places out; the shortlist is then checked against maps or imagery.

| Clue | What it rules out | Tool / data the agent needs | Offline here? | Tried before |
|---|---|---|---|---|
| Script, language, spelling; phone / postcode / domain formats; prices, currency | countries, sometimes regions | read text (crop / zoom + VLM), format rules | yes (VLM) | text helps where present (lesson 4: 64.9 vs 60.9 % < 25 km on text photos); transcribed text as stage-1 evidence added little *as retrieval queries* (11-19) |
| Infrastructure conventions: driving side, road lines, bollards, poles, plates, sign styles | countries / regions | VLM attribute reading + a rule table | partly (VLM; no rule table yet) | photo attributes were accurate but redundant *as reranker features* (21) |
| Named places in text (shop, street, station, town) | to the street | fuzzy geocoder | yes: GeoNames `allCountries` (fuzzy matching needed, lesson 6) | GeoNames landmark evidence +1.3 zero-shot (8) |
| **Features that co-occur** ("church tower + river + rail bridge") | to a neighbourhood | feature query near each candidate (Overpass-style) | GeoNames feature codes now; Overture places / buildings downloadable (free S3) | **no**. Only visual matching was tried (map search, 47-49; concept similarity, 15 / 51) |
| Sun direction, shadow length | hemisphere, latitude band | geometry; needs a capture time (rarely present) | no time | no |
| Terrain: ridgelines, coastline | to a valley | elevation model matching | no DEM | no |
| Vegetation, climate, soil | regional prior | VLM | yes | redundant with retrieval (21) |
| Verification against satellite / Street View | confirms a place | imagery | no | exemplar photos instead (22-26, comparator +0.7-0.9) |
| Outside context: EXIF, reverse image search, photographer's other photos | anything | web | n/a | **banned** here (contamination, same-photographer rule); fine for a deployed agent |

The difference from what we tried: we kept asking tools for **similarity** ("photos like this one",
"places that look like this"). Professionals ask for **constraints** ("places where X and Y both exist", "countries
that write like this"). A constraint can be decisive on a photo that looks generic.

## Agent design (analogous to the professional loop)
1. **Observe:** describe the photo, read text, list checkable features. Tools: crop / zoom (small details are lost at 448 px).
2. **Hypothesise:** the retrieval candidates (Pinpoint pool, ~17) and the agent's own country / region guess.
3. **Constrain:** call tools, e.g. `geocode(text)`; `features_near(candidate, ["church", "river", "rail bridge"], km)`
   -> which listed features exist within reach of each candidate; `rule(script / driving side / plate)` -> allowed countries.
4. **Eliminate and verify:** drop inconsistent candidates; answer with the best survivor (or a geocoded place).
- Reward: distance to the truth (as in `grpo_train.py`), ground truth only, no Gemini traces (CLAUDE.md).
- Train on MP16 bucket 99 (34.5k), multi-turn tool calls, Qwen3.5-4B thinking off, LoRA.

## How to proceed (one real-task check, then the agent)
0. **Development set:** the same "fixable" definition on MP16 dev / val (bucket 99), so prompts, tools and rules are
   developed there; the 413 are measured once per method.
1. **Tool ceiling on the real photos (no training, ~1 day):** for the 792 choosing photos (and all 3,795), base 4B lists
   features and reads text; the tools score each candidate; measure fixes / breaks vs the reranker top-1, choosing any
   rule on MP16 dev. Key number: of the 413 region-fixable photos, how many the constraints point to the right region
   **without breaking** reranker-right photos. If that is far below the current 10.9 % right-region rate, the clues aren't there.
2. **Agent:** if the tools have signal there, GRPO the agent with them (bucket 99), compare to the reranker top-1 by fixes / breaks.

## Rules that carry over
- Measure what a change adds **beyond** the reranker top-1 and the pool oracle, on the photos retrieval fails on.
  Check where a new signal is right: lessons 17-19 and 52 found every signal right mostly where retrieval is already right.
- Choose rules and scorers on MP16 dev; benchmarks are measured once. Full eval halves for decisions.
- Expect a ceiling: about a third of the hard photos have nothing place-specific (lesson 35, by eye). No method fixes those.

## Open questions
- Can the 4B list features reliably enough, and read small text at 448 px with crops?
- Map data coverage: GeoNames has churches, bridges, stations, peaks, lakes, but not every shop or street feature;
  Overture places adds ~60M POIs with categories. Which one covers the features in hard photos?
- How to stop the agent over-trusting a constraint that is often wrong (the breaks problem from lesson 52)?
