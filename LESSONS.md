# Lessons

New lessons, one entry each (2-4 lines: the claim, the deciding numbers, the module). Lessons 1-137 are archived in
`docs/archive/LESSONS_1-137.md` (anchor `#lN` per lesson); their main points are in `LEARNINGS.md`. When a new lesson changes a main point,
update `LEARNINGS.md` too.

The two tracks run in parallel sessions (2026-10-20), so each numbers its own lessons and appends only to its own section: R1, R2, ...
for `RETRIEVAL_PLAN.md`, S1, S2, ... for `SEARCH_PLAN.md`. Later single-track work continues from 138.

## Retrieval track (R)
R1. **Retrieval misses are depth, not absence** (`retrieval_recall.py`; dev3k, released retriever, original protocol; exact rank of the best right
    index photo per search). Within 25 km: union 32 / 8 / 8 73.3 (= pool oracle 73.1), 8 / 2 / 2 59.8; per modality K = 64 / 256 / 1000:
    82.2 / 90.5 / 95.7; only 0.2% have no right index photo. Of the 26.7-pt miss, 8.9 / 17.2 / 22.4 pts return at K = 64 / 256 / 1000. No own photo
    in the pool (56%): 56.0 -> 70.4 / 84.1 / 92.8, best-modality rank median 11, p75 95 (with one: 0 / 2). Contrastive image search is first on
    63%, raw SigLIP2 adds 7.7% alone at top 32, GPS 0.9% (recall flat after K = 1: 32.2 -> 40.5 at 32, its hits pile onto one place). 1 km: 32 / 8 / 8
    52.3, K = 1000 80.6 (95.6% have a photo within 1 km). Own photos: 92% of photographers have one within 25 km of the truth in the index (76%
    within 1 km), 44% get one into the pool; removing them drops 32 / 8 / 8 to 63.6. Sparse gallery (< 100 photos within 25 km, 8%): 50.2, K = 1000 80.3.
R2. **Fine-tuning the retriever's image tower doesn't move recall** (`retriever_train.py`, `retrieval_recall compare`; dev3k, 5k steps from the
    released checkpoint, lr 1e-4, batch 4096, 4-8 min each). Control (same losses as released) reproduces it: image@32 within 25 km 67.5 vs 67.7,
    32 / 8 / 8 72.8 vs 73.3 (GPS search drops 40.5 -> 36.2 in every fine-tune). A distance-only image-image term (positive = another photo in the
    anchor's 1 / 5 / 25 km cube): 67.9 (weight 0.2), 67.4 / 66.3 / 65.7 (weight 1, larger cubes worse; image@128 80.7 -> 79.5 / 78.6 / 77.7); plus
    mined far look-alikes as negatives (> 25 / > 250 km, 8 per anchor): 66.6 / 66.7. Best 1 km no-own-photo pool +1.7 (30.9 -> 32.6), within noise.
    The tower beats raw SigLIP2 by only 2.7 at @32 (67.7 vs 65.0): an MLP over a frozen global embedding has little left to learn; new signal needs
    another embedding or more data. Index encoding reproduces the shipped index (cosine >= 0.9998).
R3. **A deeper pool isn't converted by the reranker, even trained on it** (`retrieval_recall top1`, `pinpoint_train.sh POOL=deep`; dev3k, original
    protocol, reranker's own pick, % < 1 / 25 km). setft_own over 32 / 8 / 8: 28.3 / 44.3 (pool oracle 52.1 / 73.1); over 128 / 32 / 8 (159
    candidates): 28.4 / 44.0 (oracle 63.1 / 84.1). Fine-tuned on 128 / 32 / 8 pools (`ret_deep_own`: setft_own's recipe, no photographer exclusion,
    batch 64 x 16k steps = 1.0M photos vs 1.5M, ~26 GB GPU at batch 64; 39 GB at 256): 28.1 / 43.8 deep, 28.0 / 43.9 on 32 / 8 / 8. The +11-pt
    oracle headroom is unreachable with features from the same embeddings (as lesson 130 for ranks 9-29).
R4. **In-distribution reranker data (bucket 99) doesn't help** (`PINPOINT_TRAIN_IDS`, `scripts/reranker_b99.sh`; setft_own continued on the 33,006
    bucket-99 train photos without dev3k photographers vs the same steps on buckets 0-98; dev3k top-1 < 1 / 25 km, 32 / 8 / 8). 600 steps, lr 2e-5:
    28.8 / 44.3 vs control 28.5 / 44.2 (setft_own 28.3 / 44.3). 2,000 steps, lr 5e-5: 27.1 / 42.4 (overfits) vs control 28.4 / 44.2.
R5. **The pipeline's misses are photos with no geography in the pixels; what locates them is the photographer's own upload batch**
    (`scripts/scene_zeroshot.py`, `scripts/event_ceiling.py`; dev3k, pipeline 47.3% < 25 km). SigLIP2 zero-shot scene tags: indoor / people
    close-up / object close-up are 77% of photos and 43.5 of the 52.7 wrong pts (pipeline 41.3 / 53.0 / 30.7%); terrain skylines (mountains, hills)
    8.2% of photos, 3.8 wrong pts (a DEM skyline check can't matter). "Same event" (index photo < 25 km, Flickr photo id within 5e5 ~ 2.5 h of
    upload): 69.6% vs 26% with the query's id shifted (luck), but by other photographers only 33.3% vs 27.8%: the +44-pt excess is the
    photographer's own batch, other people's same-day photos ~+5. A pixel-only "same-trip" retriever would chase the original protocol's
    own-photo effect, not geography.
R6. **On commons26's geolocatable photos the room is large and reachable** (`scripts/commons26_room.py`; commons26 dev, tier landmark / city /
    region, 719 photos; no own-photo effect: Commons uploaders are not MP16 authors). Pipeline 31.6% < 25 km (1 km 9.7), retrained top-1 24.3,
    area / pool oracle 48.0 / 56.6, a right photo in the top 1,000 of some search 89.3, some text-search cluster right 58.1. Tier region is the
    weakest (pipeline 16.0, pool 45.2). Wrong photos (68.4% of them): right place in both the text candidates and the visual top 1,000 28.1,
    text only 3.3, visual only 29.9, neither 7.1. Retrieval misses 41.0, of which 30.6 have a right photo deeper (top 1,000). Two independent
    sources agree on the right place for over a quarter of the photos; S3 found MP16 too sparse there to verify it.
R7. **Pre-2026 Commons imagery can't pick among text places at scale: coverage doesn't separate right from wrong, and the fixable photos are
    few** (`commons_verify.py`; the 719 geolocatable commons26 dev photos). Candidate set: the pipeline's answer + the top-5 distinct text places
    (> 25 km apart, lesson 119's ordering): pipeline 32.3%, a text place right where the pipeline is wrong 8.1 pts (oracle 40.3; top-8 / 16 / 64:
    9.7 / 12.5 / 19.7). Commons files within 2 km, pre-2026, not by the uploader (geosearch, 4,285 places, 38 min; `scripts/commons_verify_coverage.py`): no file at 36.5% of right
    and 39.7% of wrong text places, 36.2% / 32.0% of right / wrong pipeline answers (nearest median 47 vs 98 m), so coverage is no signal.
    Only 38 photos (5.3 pts geo, 3.9 of 977) have both a fixable right text place and any reference there; at lesson 138's rates (~30% of
    covered views matched, right place ahead 50%) that's ~1 pt before breaks. The thumbnail download (~43k at upload.wikimedia.org's ~1 / s bot
    limit, ~12 h) was stopped after 16 files; embed / report are in place but never ran.
R8. **The answer head is not data-limited** (`scripts/head_learning_curve.py`; setft_own, MegaLoc on, all areas zoomed, clues read, 3 seeds per
    point, a different random subset per seed). dev3k < 25 km by training photos: 4.4k 47.2, 8.8k 46.9, 17.7k 47.4, all 35.4k 47.8: about +0.5
    for 8x the data, so more MP16-style training photos won't buy 1 pt (lesson 125 said as much for near-misses). commons26 needs new evidence
    or in-domain rows (lesson 140), not more of the same.
R9. **GLDv2 as a landmark gallery, ceiling without images: it passes the pre-declared bar, but it is densest where the wrong answers are**
    (`scripts/gldv2_ceiling.py`, `scripts/gldv2_ceiling_control.py`; Google Landmarks v2 train metadata from its public S3 bucket, 203k landmarks / 4.13M Commons photos;
    coordinates via Wikidata, P373 or the category's Commons sitelink, 13 min of SPARQL: 184k landmarks / 3.77M photos, median 8 photos per
    landmark; commons26 dev). A landmark within 0.3 / 1 / 5 km of the truth: landmark tier 29.7 / 49.7 / 77.3%, the tier's pipeline misses
    (173) 20.2 / 41.0 / 71.1% (bar: >= 30% within 1 km), region tier 10.9 / 24.7 / 52.9. Pipeline misses with a landmark within 1 km: 161
    photos = 16.5 pts of all dev (0.3 km: 70 = 7.2). But GLDv2 doesn't fill MP16's gaps: wrong geolocatable photos with no MP16 photo within
    1 km (240) have a GLDv2 landmark that close on 7.9%. And landmarks lie near the pipeline's wrong answer more often than near the truth
    (1 km, landmark-tier misses: 61.8 vs 41.0%; a landmark near the truth but none near the answer 6.9%): wrong picks sit in landmark-rich
    places, so proximity is no signal and only same-landmark image matching (instance retrieval, the task GLDv2 was built for, many views:
    median 30-47 photos per nearby landmark) can help. Step 2 (download ~500 GB, embed, search) tests that.
R10. **Matching commons26 photos against GLDv2 landmarks rarely finds the right place: the photos show things the gallery doesn't**
    (`scripts/gldv2_pilot.py`, `gldv2_recall.py`, `gldv2_selfcheck.py`; commons26 dev, 977 photos; a mini-gallery of the 8,448 landmarks within
    1 km of a truth or of the pipeline's answer + 15,000 random distractor landmarks, <= 50 photos each, 376k photos streamed from the S3 tars
    in 32 min; best photo per landmark). Top landmark within 25 km of the truth (pipeline 25.2%): MegaLoc 6.7% (no distractors 8.3), SigLIP2
    12.3 (14.8), both averaged 11.4; landmark tier 14.0 / 22.3 / 21.7% (pipeline 42.3). Switching to the top landmark when > 25 km from the
    answer never clears the bar: best +0.51 pts (both, 6 fixes / 1 break, threshold picked on these photos), MegaLoc +0.2, SigLIP2 0. A 1 km
    refinement (move to the top landmark within 25 km of the answer when its match is confident) gives +1.0-1.5 pts at 1 km in-sample (MegaLoc
    13 fixes / 1 break; both 17 / 2). Recall is the limit, not ranking: 37.7% have a gallery landmark within 1 km of the truth, but it's in the
    top 10 for 7.8% (MegaLoc) / 12.7% (SigLIP2); a right landmark within 25 km is in SigLIP2's top 10 for 29.0% (16.7% of the pipeline's misses,
    MegaLoc 7.0). The matcher works on GLDv2 itself (a held-out GLDv2 photo's own landmark is MegaLoc's top-1 photo 70.0%, top 10 80.2%), so
    the commons26 photos show other things: a gym inside a listed hall, a procession statue, details, not the landmark's canonical views, and
    GLDv2 categories are noisy (colour tables, door knobs). Confidence is informative (right vs wrong top landmark AUC 0.83 MegaLoc, 0.69 SigLIP2).
    Full index (`scripts/gldv2_index.py`, `gldv2_eval.py`, `gldv2_head.py`, `gldv2_after_index.sh`): all 184k landmarks with coordinates, <= 50
    photos each, 2.71M photos (stream 49 min, MegaLoc 4.1 h at ~210 / s with the CPUs shared, 46 GB fp16 in 28 shards; search ~6 min per set).
    dev3k holds Flickr photos copied to Commons (top-1 similarity 0.987 at the 99.9th percentile, commons26 dev max 0.667): hits > 0.7 dropped.
    Rules with the threshold from the other set: the 1 km refinement is commons26-only (in-sample +0.8-1.2 at 1 km; on dev3k never above +0.07,
    the commons26 threshold -1.10 there: dev3k answers already within 1 km get moved off), the 25 km switch loses everywhere. As answer-head
    columns (per area: best match within 25 km, log hits, any, best minus top; visible in every state; 2 seeds): dev3k +0.62 fixed recipe / +0.70
    RL (SE 0.26; 80 fixes / 49 breaks), commons26 dev (uploader folds, +commons weight 3) +0.92 / +0.67 (SE 0.43): consistent but sub-threshold,
    the size of MegaLoc per area (lesson 93). On top of lesson 140's text columns + tokens (`gldv2_head.py --text`; that cell 27.0% vs +commons
    25.4) they add +0.00 / +0.26 (SE 0.64; 42 fixes / 39 breaks): on commons26 the text already holds what the landmark matches know. Stacked
    on dev3k with diffusion + crop views (`scripts/gldv2_stack.py`, 2 seeds): diffusion + crops +0.33 / +0.37, all three +0.77 / +0.92 (SE 0.31;
    107 fixes / 69 breaks): mostly overlapping, still under 1 pt. The test
    sets were not run (the pre-agreed condition, >= 1 pt on commons26 dev, wasn't met).

## Search-agent track (S)
