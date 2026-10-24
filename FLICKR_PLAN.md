# Live Flickr as an agent tool: research plan (draft 2026-10-24, for the user's review)

**Idea.** A geolocation agent that spends a real, rate-limited budget of live Flickr calls. It decides which read clue to search and which
candidate place to check against photos taken there, and it learns that policy with RL, where the per-call cost is a real constraint rather
than a simulated one. Flickr is the right source for our main test sets: im2gps3k, yfcc4k and MP16 are Flickr photos from 2004-2014, so
Flickr photos at a candidate place come from the same era and show the same kinds of photo spots. That's what lesson 135 (2004-14 photos vs
2026 queries) and lesson 138 (sparse Commons and Mapillary) lacked. Tool: `src/geo_search_env/tools/flickr.py` (cache, limiter, leak
filters, harvest, report).

Evaluation follows CLAUDE.md:
- choose and tune on dev3k
- test on all of im2gps3k + yfcc4k (7,533 photos, original protocol), only when the user asks
- commons26 test once at the end as a final validation, with the "uploaded before capture" filter
- go/no-go measures what Flickr adds *beyond the current pipeline and the offline YFCC copy*, at +1 pt at 25 km (dev3k SE ~0.26)

## What we already know (why this might fail)
- On MP16 photos, text sits where retrieval already succeeds. A text-search result within 25 km exists on 17.5% of dev3k photos, but on
  only 1.3% of photos with no right Pinpoint area (lesson 123). Commons-trained text trust hurts dev3k (lesson 141).
- Coverage within 25 km is not the gap. Pinpoint's index has a photo within 1 km of the truth for ~85% of queries. Adding the 50 nearest
  YFCC photos within 1 km would fix at most 1.2-2.8% at 1 km (lesson 134).
- Look-alike verification saturates at AUC ~0.6-0.79 (main point 2), but exact-view matching works when the view is covered: MegaLoc 0.47-0.58
  on a covered view (lesson 138).
- Where dev3k loses (47.3% < 25 km): retrieval miss 26.5 pts, right candidate never an area 11.8, wrong area chosen 14.4 (cross-region,
  median 381 km).

So each use below has to escape one of these, and each phase starts with a cheap ceiling check that can stop it.

## What live Flickr has that our offline copies don't
1. **Volume and fields.** All public geotagged Flickr photos, not YFCC's Creative Commons subset of 48M; text search covers descriptions
   too; results are ranked by relevance.
2. **Photos at any point.** A box search (up to 250 photos per call, with thumbnails) at a candidate spot: dense same-era photos where
   MP16 has few.
3. **Recency.** Uploads from 2015 on. This matters for live use and commons26, not for the main test sets.

Leak rules (in the tool, applied when results are read back):
- Hide the query photo and its photographer, always. Pinpoint's index keeps the photographer's photos under the primary protocol, but on
  Flickr the same trip's photos carry the same titles, so the tool is stricter.
- yfcc4k photos come from YFCC itself, so their ids are always dropped.
- Re-uploads by other accounts can slip through. Phase 1b checks MegaLoc ≥ 0.9 near-duplicates of the query and drops them.

## Phases

**Phase 0: does live text search find right places the offline copy misses? (running)**
- Inputs: every 4B query of dev3k (9,726 distinct searches, ~5.5 h at the safe pace; the first run was interrupted by Flickr throttling).
- `flickr report --set dev3k_setft_own`, cache only. It reports, live vs YFCC copy, % of photos with a match / the tool's own pick within
  25 km, on all photos and where the pipeline is wrong.
- Add a third subset: photos with no right area, the 26.5 + 11.8 pts the pipeline can't reach.
- **Go:** the live pick is right where the pipeline is wrong *and the YFCC copy's pick isn't* on ≥ 2% of dev3k. That's the headroom for
  +1 pt after the answer head's losses. Otherwise text is closed as a Flickr use and only Phase 1b remains.

**Phase 1: ceilings for the two uses, offline on cached calls**
- *1a. Text evidence per area.* For each of the head's areas: guarded Flickr matches within 25 km, their share of the query's matches, and
  distinct owners. These become answer-head columns, compared against the same columns from the YFCC copy.
  - Lesson 141 says trust in text must be learned on MP16 training photos, so it needs a train harvest.
  - 10k train photos ≈ 25k searches ≈ 14 h at the safe pace, over two nights.
  - CV on dev3k. **Go:** Flickr columns beat YFCC columns by ≥ +0.5 and the total gain is ≥ +1 pt.
- *1b. Photos at the spot.* For a check worked back from the truth (like lesson 138's test), 300 dev3k photos where the pipeline is wrong:
  - Flickr photos within ±0.5 km of the truth and of the head's wrong pick (2 box searches per photo, 20 nearest thumbnails each).
  - Best MegaLoc match vs the query.
  - ~600 calls, ~12k thumbnails (~1 h at a polite 3-4 per second).
  - **Go:** truth over wrong pick on well over half (AUC ≥ 0.7). Lesson 138 got 0.49 with Commons, and 0.64 even at the true spot.
  - If go: the same per-area column for the head's top-3 areas on all of dev3k and the train subset (≈ 9k + 30k calls, multi-night), CV
    on dev3k, bar +1 pt.

**Phase 2: the RL agent (only for the uses that passed Phase 1)**
- Actions, over the existing area tokens (`action_agent.py`):
  - `flickr_text(query)`, worldwide
  - `flickr_text(query, area)`
  - `flickr_spot(area)`
  - answer
- Each action has its live cost in calls.
- Training replays cached responses only (`FlickrSearch(offline=True)`). Every action that can be listed for a training photo is
  harvested beforehand, so rollouts are unlimited and cost no API calls.
- Reward: 25 km hit (and 1 km), minus λ × calls.
  - Add potential-based shaping (Φ = head probability within 25 km of the truth) only if a ~1 h offline check shows the change in Φ predicts
    which actions improve the final answer.
- **Judged against** fixed recipes (call everything; text only; spot only) and heuristic gates (call when the head is unsure), at matched
  call budgets of 1, 3 and 5 calls per photo (memory: offline actions simulate costly tools). RL earns its place only if it beats the best
  gate at the same budget. Lesson 94/98: RL ties "take every action" when actions are cheap and independent; here they are neither.
- Training on the YFCC copy (free, unlimited) with live-cache calibration is the fallback if train harvests are too slow.

**Phase 3: test (only when the user asks)**
- One harvest of the chosen actions on im2gps3k + yfcc4k: ≤ 5 calls × 7,533 photos ≈ 38k calls ≈ 21 h at the safe pace.
- Original-protocol table: benchmark × 1 / 25 / 200 / 750 / 2500 km vs released Pinpoint and the current pipeline.
- Then commons26 test once, as the final validation.

## Budget and logistics
- **Pace.** Flickr throttled us (HTTP 429) after ~600 calls in 13 min at ~50 per minute, below its nominal 3,600 per hour. The tool now
  runs one call every 2 s, at most 1,800 per hour, 2 at a time, and all processes pause after any 429. That's about 40k calls per day if a
  harvest runs all day; the machine stops at 7 PM Pacific and harvests resume from the cache.
- **Thumbnails** come from the static CDN, which doesn't count against the API limit. We keep only the MegaLoc / SigLIP embeddings
  long-term; the images are deleted after embedding (Flickr's terms allow storing photos only for "reasonable periods").
- **Calls by phase:** 0: ~10k (dev3k). 1a: ~25k (train). 1b: ~1k for the check, ~40k if it passes. 2: none new beyond the chosen actions'
  harvest. 3: ~38k (test, when asked).

## Order and stop points
Phase 0 report → stop text if it fails the 2% bar → Phase 1b check (cheap, ~1 h) → Phase 1a / full 1b columns only for what passed →
Phase 2 RL → Phase 3 test when asked. Each phase gets a lesson in `LESSONS.md` and a line in `TOOL_USE_DESIGN.md`.
