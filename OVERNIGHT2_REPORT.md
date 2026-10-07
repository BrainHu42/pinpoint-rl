# Overnight 2 report (2026-10-07; plan `OVERNIGHT.md`, journal `artifacts/overnight2/JOURNAL.md`)

Ran 05:06-05:27 and 14:30-19:30, with two stalls (05:30-14:23: the wakeup never fired; 15:40-17:37: my error, no wakeup
scheduled). Question: can constraint-style evidence through tools fix the photos where visual retrieval fails?

**Answer so far: a little, precisely, and with a low ceiling.** Text read from the photo and matched to a GeoNames name near
each candidate is the one tool that holds up: **+0.4 pts within 25 km on the eval halves [+0.2, +0.6], 18 fixes, 3 breaks**.
That was the single test-set measurement, of a method chosen on MP16 dev + val. The other tools add nothing.
The reason: most photos retrieval fails on carry no place-identifying clue that 4-9B models and open map data can reach.

## Built (branch `worktree-agent-tools`)
`agent_targets.py` (fixable photos: dev 102, val 106, full 413), `observe.py` + `scripts/observe.sh` (VLM -> JSON clues), `geo_tools.py`
(GeoNames index, features near a candidate, text near a candidate, text -> region, country rules, Overture POI names, CV combiner, transfer),
`poi_query.py` (DuckDB over Overture places), `agent_capacity.py` + `scripts/capacity.sh`, `agent_vote.py`.

## Results (change in top-1 within 25 km vs the reranker; dev + val CV n = 2,000 unless noted)
| Tool (clues read by Qwen3.5-9B) | Change | Fixes / breaks |
|---|---|---|
| **Text matched to GeoNames near the candidate, eval halves (fit on dev + val)** | **+0.4 [+0.2, +0.6]** | **18 / 3** |
| Text near candidate, dev + val | +0.4 [+0.1, +0.9] | 14 / 5 |
| Overture POI (business) names near candidate | +0.2 [-0.1, +0.7] | 13 / 8 |
| Text -> region (GeoNames admin1) | +0.1 | 5 / 4 |
| Visible features near candidate (GeoNames H/T/S/R codes) | -0.4 | 3 / 10 |
| Country rules (language, script, driving side), VLM country guess | 0.0 | 0 / 0 |
| All tools together | -0.1 | 24 / 26 |

- **Text ceiling:** read text names a place in the true region specifically on only 4 / 102 dev and 7 / 106 val fixable photos.
- **Better reading doesn't help:** 9B text from 4 image crops per photo raises text coverage on fixable photos 43% -> 63-73%, but rank + text stays
  +0.4 [-0.0, +0.8] (14 / 6): the extra text isn't geographic.
- **Capacity:** right region on fixable photos is 4B 13.9%, 9B 16.8% (+2.9 [-1.9, +7.7]). For the 4B, 448 -> 786k px gives +3.9 [+1.0, +7.8] on dev
  fixable photos; 1.6M px adds nothing.
- **Agreement:** switching only when 4B and 9B agree halves the breaks but also halves the fixes (best +0.4 [-0.1, +0.9]).
- **By-eye audit, 24 dev fixable photos:** 58% have nothing place-specific; ~15% show a business name (POI search, better reading); 12% a landmark
  (landmark matching); 12% country-level cultural objects (e.g. Strandkorb beach chairs).

## What it means
- Constraint tools are precise when they fire (6 fixes per break on the test set), unlike every earlier chooser, but they fire rarely.
- The remaining headroom on these photos is in landmark recognition (~12%) and in reading business names correctly (~15%); most of the rest is unfixable from the image.
- An RL agent over these tools would have a weak signal to learn from (lesson 3). Build the signal first.

## Recommended next steps
1. Landmark matching as a tool: the comparator (lessons 24-41) restricted to photos with a detected landmark, scored on the dev fixable set.
2. POI search with fuzzy names (the audit's misread "Tepper"); crops alone were tried and add nothing.
3. Keep "rank + text" as the first agent tool. It's the one confirmed positive: `geo_tools transfer --fit dev val --tag full --name 9b --group text`.
