# Live Flickr as an agent tool: research plan (revised 2026-10-24 with the user)

## Goal (refined)
First find where the Flickr API can actually help, then build the agent around that one use. A screen of each API capability against
dev3k's losses (pipeline 47.3% < 25 km) left one use worth testing first:

**Photos at a point.** Dense, same-era Flickr photos at a candidate place, compared with the query by exact-view matching (MegaLoc). The
question is whether they separate the right candidate from the look-alike the pipeline chose: the 14.4 pts of "wrong area chosen" and part
of the 11.8 pts of "right candidate ranked low".

If this works, the research goal becomes: **an agent that verifies candidate places with a limited budget of live Flickr photos at the
place**, learning with RL which candidates are worth a call.

Evaluation follows CLAUDE.md:
- choose and tune on dev3k
- test on all of im2gps3k + yfcc4k when the user asks
- commons26 test once at the end, as final validation only (no development on commons26 dev)
- bar: +1 pt at 25 km beyond the current pipeline (which already has MegaLoc against Pinpoint's own photos per area, lesson 93)

## The screen: Flickr capability against dev3k loss

| Capability | Targets | Prior evidence | Verdict |
|---|---|---|---|
| Text → places (titles, tags, descriptions) | Retrieval misses / never an area, photos with readable text | On MP16, text helps where retrieval already works: a right result on 1.3% of photos with no right area (lesson 123); the limit is the 4B's queries, not the backend (lesson 115) | weak; Phase 0 harvest measures live vs YFCC when the throttle allows |
| **Photos at a point** + exact-view matching | Wrong area chosen (14.4), ranked low (part of 11.8) | Look-alike checks saturate (main point 2), but an exact covered view is matched cleanly (lesson 138, sparse Commons, 2026 queries); never tried with dense same-era photos | **test first** |
| Photos at a point, 1 km tier | 1 km ranking | Up to 50 YFCC photos near the truth fix ≤ 1.2-2.8% at 1 km (lesson 134) | small; revisit if the above works |
| More gallery photos (sparse regions) | Retrieval misses (26.5) | Track R's plan with YFCC images (CDN, not API) | belongs to track R |
| Tag statistics per area | Wrong area chosen | Area profiles closed (main point 4) | weak |

## Step 1 (running): coverage audit worked back from the truth, offline, YFCC as a lower bound for Flickr
`experiment/spot_audit.py`, no API calls (images from Flickr's static CDN).
- **Photos.** dev3k close calls: error tiers 1 policy, 2 coin flip, 4 ranked low, 5 in pool (`viz/errors.py`). That's 725 photos where a
  right place is on the table and the pipeline picked another.
- **Points per photo.**
  - truth: ceiling
  - right: the right area's centre, or the nearest right candidate (what an agent could check)
  - wrong: the pipeline's answer
- **References per point.** The 50 nearest YFCC100M photos within 1 km (accuracy ≥ 13). The query photo and its photographer are excluded,
  and near-duplicates of the query (MegaLoc ≥ 0.9) are dropped and counted. Each reference is flagged as in Pinpoint's index or new. That
  gives 85k references (79k unique); only 13% are in the index.
- **Measure.** Best MegaLoc cosine at each point:
  - P(right > wrong), paired
  - clean wins (≥ 0.3 and above the other point) vs clean losses
  - broken down by number of references (5 / 10 / 25 / 50), by source (all / index only / new only) and by tier
- **Go:**
  - P(right > wrong) ≥ 0.7 with all references.
  - Clean wins well above clean losses.
  - New photos add clearly over index-only photos, since the pipeline already has MegaLoc against index photos.
  - If the curve still rises at 50, live Flickr's extra density should push further.
- **No-go:** P(right > wrong) < 0.6, or no gain from new photos. Then photos at a point are closed as a Flickr use (with a note on what
  coverage would be needed), and we're left with the narrow uses (text for landmark photos, 1 km refinement) or a different research goal.

**Preliminary (2026-10-24 01:40 UTC, 48.7k of 79k references fetched; heading to no-go).**
- Best MegaLoc at the right point beats the wrong pick on only 43% of photos (truth point 40%), the same for index-only (44%) and new-only
  (44%) references. The curve is flat in the number of references (k 5 → 50: 0.42 → 0.43).
- The wrong pick sits in denser places (median 22 kept references vs 18 at the right point), and that tilts best-of-k toward it. Compared
  at equal reference counts (both points ≥ k), it's still chance: k 5 / 10 / 25 → 0.48 / 0.45 / 0.46 (n 437 / 350 / 149).
- Scores are at noise level: median best 0.06-0.10 at both points; a clean match (≥ 0.3) at 2-5% of points either way. Clean wins 2.6% vs
  clean losses 5.0% at k 50.
- Only where the right point has ≥ 30 references does it reach 0.56.
- Reading: the nearest same-era photos within 1 km rarely show the query's view, even for MP16 queries, as lessons 135 and 138 found for
  2026 queries. More of the same density doesn't change it.
- To finish when the fetch completes: the full report, then a check on spots before closing (a 0.25 km radius, and references chosen by
  SigLIP similarity to the query instead of distance; the latter is a look-alike and needs care).

## Step 2 (if step 1 is go): how much more live Flickr returns than YFCC
- ~300 close-call photos × 2 box searches (right and wrong point, ±0.5-1 km) ≈ 600 API calls, plus thumbnails.
- Compare live photo counts per point and the same MegaLoc measures against step 1's YFCC numbers. This scales the offline lower bound to
  the live tool, which also has photos from 2015 on and non-Creative-Commons photos.
- Leak rules: `tools/flickr.py` hides the query photo and its photographer; near-duplicates are dropped as in step 1.

## Step 3: a per-area spot column in the answer head
- Best spot match per area as an answer-head column (like lesson 93's MegaLoc columns), for the head's top-3 areas.
- Built on dev3k (CV) and on a train subset so trust is learned on MP16 (lesson 141).
- Built from YFCC first (offline, free); then from live Flickr if step 2 shows a clear density gain.
- Bar: +1 pt at 25 km over the current head.

## Step 4: the RL agent with a call budget
- Action `flickr_spot(area)` over the existing area tokens (`action_agent.py`), each call with its live cost.
- Trained by replaying cached responses (`FlickrSearch(offline=True)`; every listed action for a training photo is harvested beforehand).
- Reward: hit within 25 km (and 1 km) − λ × calls; potential-based shaping only if an offline check supports it.
- Judged against fixed recipes (check the top 1 / 2 / 3 areas) and heuristic gates (check when the head is unsure) at matched budgets.
  RL earns its place only by beating the best gate at the same budget.

## Step 5: test (only when the user asks)
im2gps3k + yfcc4k (7,533 photos), original-protocol table vs released Pinpoint and the current pipeline; then commons26 test once.

## Budget and logistics
- **API:** Flickr throttled us (HTTP 429) after ~600 calls in 13 min at ~50 per minute. The tool now runs ≤ 1,800 calls per hour,
  2 at a time, with a shared pause after any 429.
- **Image CDN:** it answered 403 / 429 for a while after a burst (and 403 to a fake browser user-agent). Downloads now identify the client,
  run 8 at a time (~21 per second), and pause on 403 / 429 instead of counting the photo as gone. Only MegaLoc embeddings are kept, never
  the images.
- **Phase 0, the text harvest on dev3k** (9,726 searches, `tools.flickr harvest`), resumes when the API throttle lifts. It's a side
  measurement of the weak text use, not on the critical path.
