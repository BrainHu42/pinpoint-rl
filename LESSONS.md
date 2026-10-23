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

139. **Map attributes read from the photo's embedding (SigLIP2 or MegaLoc) don't separate close calls (no-go), though the attributes at the truth would**
    (`scripts/geodata_grid.py`, `scripts/geodata_check.py`; dev3k; close calls = tiers 1, 2, 4 of `viz/errors.py`, 397 photos). 21 attributes
    on a 2.5' grid (WorldClim elevation / relief / 6 bioclim, EarthEnv 12 land-cover classes, coast distance); an area = its 25 km disc;
    score = Gaussian log-likelihood of the photo's attributes under the disc. Ceiling (map at the truth): AUC right over chosen 0.91 (climate
    alone 0.98), re-ranking +8.5 pts. Realistic (MLP probe on SigLIP2, 400k MP16 photos, dev3k photographers out; held-out R^2 0.33-0.70):
    AUC 0.47 (tier 1 0.52, tier 2 0.57, tier 4 0.41), 0.59 on correct photos; re-ranking cross-fitted +0.07 pts (14 fixes / 12 breaks).
    A second encoder doesn't change it: MegaLoc (DINOv2 place recognition; lesson 112's cached embeddings, same 420k training photos) has
    held-out R^2 0.13-0.36 and close-call AUC 0.47 (tier 4 0.44; SigLIP2 on the same photos 0.46 / 0.38; both 0.47 / 0.42), re-ranking
    +0.23 pts (15 / 8). Why: a close-call photo looks like the chosen area's environment, so any appearance-based reading of it points there
    (below chance on tier 4 with both encoders). What's left is a reader that brings knowledge rather than look-alike appearance (a VLM
    naming species, crops, building materials), or text.

140. **One answer head over areas and text places works on commons26 dev when the Commons rows are lightly weighted: +1.9 pts (SE 0.7)**
    (`combined_head.py` cv / adapter / stack; step A of the combined-head plan; setft_own, MegaLoc on, `--one-km 0.5`, RL lambda 0, 2 seeds;
    commons26 dev in 5 uploader folds, each scored by a head trained on MP16 train + the other 4 folds; dev3k for regressions). Text evidence
    enters as 8 columns per area token (OSM / GeoNames names within 25 km matched <= 5 / 20 / 100 times, lesson 119's listwise ranker's mass,
    max and top-1 within 25 km, nested out of fold) and up to 3 text-place tokens (ranker places away from every area, answered at the
    nearest pool candidate); columns and tokens are visible only after the read action. Base head 25.0% < 25 km (released Pinpoint 20.2).
    Commons rows x10: no text +0.10 (SE 0.49), text + tokens -0.46 (0.88; 64 fixes / 66 breaks). Commons rows x3: text columns only +1.07
    (0.46; 27 / 11), text + tokens **+1.89 (0.66; 48 / 26)**, continent-balanced 27.8 -> 29.8, < 1 km 7.5 -> 7.9; dev3k +0.25 (0.17).
    Frozen MP16 head + a 10-parameter text adapter (score = a logit + w . columns; tokens b + w . columns; fit on the other folds): +0.92
    (0.56; 27 / 14); the adapter leans on names matched <= 20 / <= 100 times (+0.42 / +0.22) and ranker top here (+0.28), not <= 5. So weight 10
    on ~780 training photos overfits them; at weight 3 the full head uses the text and the tokens add ~0.8 on top of the columns. Weight was
    picked from two settings on these folds, so step B (a separate commons_train pool, all 977 commons26 dev photos held out) is the check.
    Sub-threshold stacking on dev3k (3 seeds, base 47.3): MP16 kNN-graph diffusion columns +0.38 (0.24), crop-look columns +0.50 (0.23),
    both +0.34; not adopted.

141. **Step B: the combined head trained on a separate Commons pool keeps the gain on all of commons26 dev, +1.8 (SE 0.8) RL / +2.2 fixed recipe**
    (`combined_head.py transfer --weight 3`, `scripts/commons_train_pipeline.sh`; setft_own, MegaLoc on, `--one-km 0.5`, 2 seeds; all 977
    commons26 dev photos held out, no CV). Training pool `commons_train` (`commons_bench` with `COMMONS_OUT`): geotagged Commons JPEGs uploaded
    2026-01 to 06 (every day read; commons26 is uploaded from 07), taken in 2026, commons26's 2,345 uploaders excluded (they own 77% of the
    candidates), <= 20 per uploader, >= 50 m apart: 11,282 photos, 3,556 uploaders, 155 countries; retrained top-1 21.1% / pool oracle 50.9%
    (commons26 dev 19.7 / 48.2), so the same difficulty. Whole pipeline on it (~3.5 h after a 3 h paced download). Text ranker trained on
    commons_train: top-1 right on commons26 dev 20.8% (lesson 119's in-domain CV 20.7). Results on commons26 dev, % < 25 km (base 25.0):
    MP16 + commons_train x3 without text +0.97 (0.52; 39 fixes / 25 breaks); with text columns + text-place tokens **+1.84 (0.76; 54 / 33)**
    under the lambda-0 policy, +2.20 (0.77) with the fixed read-everything recipe, < 1 km 7.5 -> 8.2; frozen MP16 head + text adapter fitted
    on commons_train +1.38 (0.67; 37 / 20). dev3k flat (+0.07, SE 0.24). So lesson 140's +1.9 was not a product of training on commons26 dev
    folds or of the weight choice: in-domain photos give ~+1, the text evidence another ~+0.9-1.2 on top. Fitted on 11k photos the adapter
    now uses names matched <= 5 times too (+0.25; ~0 when fitted on commons26 dev), ranker top here +0.37.
    On dev3k with its own text inputs (same searches, ranker trained on commons_train; a right candidate 33.1%, ranker top-1 right 24.0%
    vs the reranker's 44.3%) the text hurts: full head -0.30 (SE 0.35; 103 fixes / 120 breaks), adapter -0.78 (0.30; 38 / 70); Commons rows
    alone +0.20. The head learned how far to trust text only from Commons rows (MP16 training rows carry zero text columns), where the
    reranker is weak (~20%); on Flickr photos the reranker is strong and the same trust overrides right answers. So no gain is expected on
    im2gps3k / yfcc4k as trained; the fix is text features on MP16 training photos (gather on train, ~35k photos) so trust is learned per domain.
