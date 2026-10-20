# Lessons

New lessons, one entry each (2-4 lines: the claim, the deciding numbers, the module). Lessons 1-137 are archived in
`docs/archive/LESSONS_1-137.md` (anchor `#lN` per lesson); their main points are in `LEARNINGS.md`. When a new lesson changes a main point,
update `LEARNINGS.md` too.

The two tracks run in parallel sessions (2026-10-20), so each numbers its own lessons and appends only to its own section: R1, R2, ...
for `RETRIEVAL_PLAN.md`, S1, S2, ... for `SEARCH_PLAN.md`. Later single-track work continues from 138.

## Retrieval track (R)

## Search-agent track (S)

S1. **Search-tree ceiling (step 1): +14.0 pts beyond the area oracle with ~37 searches per photo, but about half is spread, not evidence**
    (`search_tree.py`; dev3k, setft_own, original protocol). Vocabulary from each photo's own pool: its candidates' countries (6.3 per photo),
    regions (14.5) and +-25 km boxes around candidates outside every area (16.0). Some restricted top-1 < 25 km: 70.3% (area oracle 61.4, pool
    oracle 73.1); beyond the area oracle country +3.6, region +9.0, box +11.2, all +14.0, of which +3.3 beyond the pool oracle too (new places).
    Luck (lesson 120, same vocabulary vs another photo's truth): a random photo +3.3; a same-country photo > 100 km away +7.7 (area oracle there
    6.0), so net ~+6 pts, and the extras are only 1.8x chance where the areas are 10x. By budget (constraints in pool-mass order) real / same-
    country luck: 8 searches +4.0 / +1.6, 16 +8.0 / +3.6, 64 +13.7 / +7.7. By bucket: never an area 89.9% reachable (boxes: they restate pool
    candidates as areas), wrong area 75.0% (regions), retrieval miss only 12.5% (3.3 pts). True region as a reference search: 71.6% (+18.1).
    Speed of `tools/pinpoint.py`: a streamed pass (none / box / country / region) costs ~1.2-1.7 s whatever the batch, 2.4-3.4 ms / query at
    batch 1,024 (~1.5 ms at 4,096; GPU 4 GB); rows / near (box-sized, ~15k rows) 2.6 / 6.5 ms per query on the 10.5 GB resident index.

S2. **A search's result is not evidence the head can act on (step 2, no-go)** (`search_tree.py evidence`, `evidence-megaloc`; 1.31M cached
    searches on all 35,359 train photos, evaluated on dev3k). P(restricted top-1 < 25 km) from prior features A (the constraint's pool mass,
    candidates inside, best score / rank inside) vs A + result features B (restricted top-1 score and drop vs the unconstrained top, margin,
    agreement inside the restricted list, novelty, agreement with the photo's other searches): AUC among alternatives (top-1 >= 25 km from
    the pipeline's answer) 0.686 -> 0.748 / 0.707 -> 0.764 / 0.783 -> 0.829 at 8 / 16 / all searches. But the best-scored alternative is right
    on 6% of photos while some alternative is right on 14 / 19 / 25%, and no switch threshold nets > 0 even in hindsight (cross-fitted -0.01 to
    -0.10 pts; the 30 most confident switches: 3-6 fixes / 10-17 breaks). Step 2b, MegaLoc (query vs restricted top-1; 78k photos, 7 min):
    alone AUC 0.68; added to A+B (dev3k 5-fold) AUC among alternatives +0.002 to +0.006, switches -0.07 to -0.04 pts. (Numbers after the
    `_fit` fix of S3; before it, the same conclusions.) As lessons 111-112 and main point 2: searching finds right regions the
    evidence can't pick. A sequential agent over pool-derived constraints has nothing to condition on; closed in this form.

S3. **Text-proposed places verify far better than pool ones, but not where text could fix the pipeline** (`text_verify.py`; commons26
    dev, 977 photos). Lesson 119's text-search candidates (<= 64 per photo; some right 35.0%, luck 3.7%) and, on the same photos, boxes
    around the pool candidates; Pinpoint restricted to +-25 km around each (77k searches). The restricted result alone separates right from
    wrong text candidates at AUC 0.944 (0.914 away from the free top-1) vs 0.847 / 0.788 for pool boxes: independence from retrieval
    matters (lesson S2's selection effect). But it adds nothing to lesson 119's ranker (top-1 20.7 -> 20.5 linear, 19.1 -> 19.2 MLP),
    whose text features already hold "Pinpoint points here", and the switch against the pipeline stays below the bar (text only +0.41 /
    +0.72, with verification +0.41 / +0.20; SE 0.5-0.7; a blank MegaLoc column had shown +1.1 / +1.2, so these switches move ~1 pt on noise).
    Why: on the 116 photos with a right text candidate and a wrong pipeline, the right place verifies worse than the ranker's wrong top
    (score drop vs the free search -0.24 vs -0.03, right ahead on 31%; index photos within 25 km e^7.2 vs e^8.9, ahead on 16%; MegaLoc ahead
    on 37%). Fresh photos at the right place have few MP16 references; the wrong tops sit in dense, look-alike-rich places. The limit is
    reference coverage at the candidate, which a live imagery tool would supply and the offline gallery can't.
    Found on the way: `megaloc_queries_commons26_setft_own.npy` and `megaloc_commons26_setft_own.npz` (Oct 16, before the bd2d12d path fix)
    are blank (200 photos -> 4 distinct vectors, cosine 0.99999); only the commons26dev files were redone, so the commons26 test run of
    lesson 113 had no MegaLoc signal. Also, `search_tree._fit` ran only 16-250 Adam steps on small sets (now >= 2,000); S2b re-run with it.

## Single-track lessons (138-)

138. **Live imagery near a text-proposed place doesn't verify it: the view is rarely covered, though a covered view is matched cleanly**
    (`tools/imagery.py`, `live_verify.py`; `TOOL_USE_DESIGN.md` first test, the 116 commons26 dev photos of S3). Guarded live imagery (made
    before the query was taken; Commons also not by its uploader), the N nearest per place, best MegaLoc match vs the query, and z = that
    best against the same imagery's best matches to the other 115 queries (cancels generic or plentiful imagery). N = 20: right text place
    above the ranker's wrong pick on 50% (AUC 0.49; z 54%) with Commons (10 km), 39% (0.40; z 45%) with Mapillary street level (1 km, no
    panoramas, <= 2 frames per sequence); the pre-declared go ("well over half") fails, so the 977-photo switch was not run. Control, imagery
    at the photo's own location: Commons 64% (z AUC 0.66; best >= 0.2 on 15%, z >= 3 on 25% vs 12% at wrong picks), Mapillary 54% (0.54;
    z >= 3 on 10%). Mapillary at N = 200 changes nothing (truth z >= 3 on 12%; a median 34 frames exist within 1 km of commons26 spots).
    Where Commons covers the view, MegaLoc is unambiguous (same church 0.58 at 18 m, same plaque 0.54, same valley 0.47); typically the
    nearest files show other things. Two limits: right text places are a median 8.3 km from the spot (20% within 1 km), and the view's
    coverage. Optimistic yield as tested (20% of the 116 near the spot x a ~30% hit rate) ~7 photos, ~0.7 pts of 977: below the bar.
    Untested: dense Commons (N >= 60, or a search over the candidate's whole area); upload.wikimedia.org's bot limit ("Your bot is making
    too many requests", 429) throttled cold thumbnails even at 1 / s after the parallel runs, so a dense or area search needs a multi-day
    crawl or a higher limit (bot-traffic@wikimedia.org). Tooling: Commons geosearch also throttles ("search too busy"), so the tool searches
    a local index of the `geo_tags` dump (40.6M primary coordinates, `imagery index`; agrees with the API) and looks up file info by page id;
    Mapillary's bbox API returns HTTP 500 in dense areas, so the tool reads its z14 coverage tiles (positions match the API to ~1 m).
