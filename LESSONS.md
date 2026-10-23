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
    RL (SE 0.26; 80 fixes / 49 breaks; seeds 2-3 again +0.63 / +0.70), commons26 dev (uploader folds, +commons weight 3) +0.92 / +0.67 (SE 0.43; a third seed +0.82 / +1.02, three-seed mean
    +0.89 / +0.79): consistent but sub-threshold,
    the size of MegaLoc per area (lesson 93). On top of lesson 140's text columns + tokens (`gldv2_head.py --text`; that cell 27.0% vs +commons
    25.4) they add +0.00 / +0.26 (SE 0.64; 42 fixes / 39 breaks): on commons26 the text already holds what the landmark matches know. Stacked
    on dev3k with diffusion + crop views (`scripts/gldv2_stack.py`, 2 seeds): diffusion + crops +0.33 / +0.37, all three +0.77 / +0.92 (SE 0.31;
    107 fixes / 69 breaks): mostly overlapping, still under 1 pt. The test
    sets were not run (the pre-agreed condition, >= 1 pt on commons26 dev, wasn't met).

R11. **The answer head's own structure is not the lever, but it can be simpler** (`scripts/structure_probe.py`; dev3k, setft_own, read + recognise
    + zoom all, 3 seeds, vs the current `--one-km 0.5` head at 47.4 / 30.6). Trained on the state it is used in instead of random reveals and
    zooms: 47.6 / 30.7 (+0.26, SE 0.22), so the random-state training and the RL policy (which ties "take everything", lesson 94) can go at no
    cost. A 25 km-only head 48.0 / 28.5 (+0.62 / -2.1): the joint 1 km loss costs ~0.6 at 25 km. Factorized (area from the 25 km head, point
    from the 1 km head's two options in that area) 47.7 / 30.5 (+0.28 / -0.1): half the 25 km cost comes back, because the point choice inside
    the area also moves answers across 25 km. With an oracle choice of the area's two points 48.3 / 32.4 (+0.9 / +1.8): the point picker is
    the weak part, not the area choice. 3-seed ensembles add nothing (+0.0-0.3). All within +-1: like its inputs (lessons 122-127), the head's
    form is saturated.

R12. **Refining text searches with what the 4B already read doesn't rank the right place first more often: no room for an RL query agent
    on today's reads** (`scripts/text_refine_check.py`; commons26 dev, 977 photos, 884 with phrases; every match of the 4B's 6,740 phrases and
    their spans fetched once, 27,705 queries, 10 min on the NVMe databases). Top-1 right < 25 km of one committed place (all / geolocatable
    tiers), with the candidate set's ceiling and lesson 120's luck (random other photo / same-country photo > 100 km away): one-shot phrases
    (top cluster, share / log2(2 + matches)) 12.8 / 17.2%, ceiling 18.9 (luck 0.5 / 2.5; lesson 116's 19.0 reproduced); + spans 9.1%, ceiling
    24.6 (8.9 same-country); + the 4B's country 12.8%. Conditional search ("phrase B near A's cluster", other phrases sharing no specific word,
    support = how many match within 25 km): 11.6%; support only from phrases with <= 500 / <= 50 matches 12.4 / 13.6% (+0.8 at best). The
    candidate sets grow to a 38.4% ceiling but same-country luck grows to 24.6, so the net ceiling falls (16.4 -> 13.8). Against the pipeline
    (25.2%) every switch "commit when support >= k" loses or ties (best: <= 50 matches, k 3, 39 switches, 59% right, 4 fixes / 4 breaks).
    Right text answers sit where the pipeline is already right (top-1 right on pipeline misses 4.4-4.7%). The limit is what the 4B reads,
    not how it searches; an agent that only recombines its reads has nothing to learn.

R13. **Dense Commons imagery doesn't verify text places either: even the true location loses to the wrong pick** (`scripts/gldv2_verify_proxy.py`;
    lesson 138's 116 commons26 dev photos with a right text place and a wrong pipeline answer; GLDv2's 2.71M Commons landmark photos stand in
    for a dense Commons crawl, which upload.wikimedia.org's bot limit blocks; MegaLoc, hits > 0.7 dropped). Best match among gallery photos
    within 1 / 10 / 25 km: the right text place beats the ranker's wrong pick on 42 / 36 / 34% where both are covered, and the truth itself
    on 42 / 36 / 34% (coverage within 1 km: right 41%, truth 44%, wrong pick 72%). Median best similarity 0.09-0.14 everywhere (a covered view
    matches at 0.47-0.58, lesson 138): these photos' views aren't in the gallery, and denser wrong places win on chance maxima. With lessons
    135 (YFCC Flickr), 138 (live Commons, Mapillary) and R7 (Commons coverage), imagery near a proposed place is closed as a verifier for
    commons26. Done while choosing a live tool for an RL agent: a Flickr API key now needs a Pro subscription.

R14. **Mapillary on dev3k: above chance, far below the head, adds nothing to it** (`scripts/mapillary_dev3k.py`, `mapillary_dev3k_combine.py`;
    `tools/imagery.py` from the tool-use branch). 1,334 dev3k photos whose head top two areas (25 km head on the full state, R11) hold exactly
    one right area; the head picks it on 86.6% (179 fixable, 1,155 breakable). Up to 20 Mapillary frames within 1 km of each area's point
    (restricted top-1) and of the truth (control), no panoramas, <= 2 per sequence: 57,751 frames, any frame at 85-89% of places, median 20.
    Best MegaLoc match median 0.07 everywhere (>= 0.3 on 1.6% at the head's top, 1.5% at the truth); a clear match (z >= 3 against the same
    frames' best matches to 300 other queries) on 15.0% at the truth, 12.8% at the top, 5.7% at the second. Right area above the wrong one
    58.3% (AUC 0.566; truth above the wrong one 60.6%) vs the head's 86.6%. Every switch rule loses (lead > 0: 76 fixes / 369 breaks; z lead
    > 2: 17 / 61); a logistic model of the head's margin + Mapillary (5-fold CV) has AUC 0.842 vs 0.845 for the margin alone, and its 28
    switches give 11 fixes / 17 breaks. Unlike commons26 (lesson 138) the candidates are precise (both 1 km checks around real gallery
    photos), so the limit is the view again: old Flickr photos (people, interiors, details, 2004-14) rarely share a street-level frame's
    view, and the few clear matches sit where the head is already right. With R12-R13: no live or offline tool tested gives an RL agent an
    informative, costly observation. Engineering: Mapillary's coverage tiles hit a daily quota ("Not Logged In" with HTTP 200) after ~10k
    tiles; `imagery.py` now falls back to the graph API's bbox search (small box first, split on errors), paces Mapillary at 0.02 s, and finds
    the repo-root `.env` from a worktree.

R15. **A dedicated OCR model reads less than the 4B and adds ~1 pt of ceiling, nothing at top-1** (`scripts/ocr_commons26.py`, PaddleOCR 3 in
    `.venvs/paddle`: PP-OCRv6 for CJK + Latin scripts, PP-OCRv5 for Cyrillic, Korean, Arabic, Devanagari, Thai, Greek; lines >= 0.5 confidence,
    per-script lines kept only if they hold that script; 15 min on the 977 commons26 dev photos; `text_refine_check.py --source ocr|both`).
    OCR finds text on 611 photos (4B 491) but searches worse: one-shot top-1 6.8% / ceiling 9.9% (4B 12.8 / 18.9). 4B + OCR: top-1 12.4%,
    ceiling 20.6% (same-country luck 2.5 -> 3.2, so ~+1 net), the same 34 right answers on pipeline misses, every switch loses. Title-word
    recall (`scripts/ocr_name_recall.py`; the 282 photos flagged as showing place text, reference = the Commons file title): a specific title
    word in the 4B's text 44.6% (with its place names and queries 54.2%), OCR 35.8%, union 56.5% (approximate matching +3-6 for every reader).
    Striking single cases (the 4B's "Bouche-Saint-Martin" where OCR reads Biache-Saint-Vaast, "Cabanillan" for Cabanatuan) are not the rule:
    the 4B transcribes as well or better, and it adds recognised names and queries. With R12, the gap on commons26 sits between reading the
    right words (~55-60% of place-text photos) and placing them (one-shot top-1 17% on geolocatable tiers): disambiguation and ranking of
    what is read, the combined head's ground (lessons 140-141), not transcription.

R16. **PE-core-G's gain is the model, not the column form, the resolution or a middle layer: per-layer probes, the paper's protocol, peak at the output; a region probe on the pooled output adds to the head**
    (`scripts/siglip_gate.py`, `scripts/pe_layers.py`, `PE_PLAN.md`; dev3k, the same 3,000 queries + 140,871 area photos as track V's gate,
    per-area columns stacked on the answer head's option logits, 2-fold x 3 seeds, head 47.20 / 30.77). Track V found PE-G's columns add
    +0.79 (gate) and +1.37 / +0.97 in the retrained head, +2.04 / +1.28 with SigLIP2's columns (V5). Controls here: SigLIP2's own pooled
    embedding as the same columns -0.03 (from the MP16 cache, no GPU), so the column form adds nothing; alone, the best-matching area is right
    for PE-G 67.5% vs SigLIP2 63.4% of photos with a right area (30.9 vs 24.6% of the head's misses). Resolution: PE-G at 378 px (27 x 27
    patches, ~SigLIP2's 384) +0.87 vs +0.79 at 448 px (with SigLIP2's columns +0.98 vs +1.18, SE ~0.33), solo 67.1%: not more pixels, and a
    re-embed can run at 378 px (~1.4x fewer tokens). Intermediate layers (Perception Encoder paper's point, arXiv 2504.13181): mean-pooled
    residual stream of blocks 32 / 38 / 44 / 47, centred and unit-normed, +0.34 / +0.19 / +0.39 / +0.37 (SE ~0.2), solo 52.9 / 54.1 / 62.5 /
    65.4%, rising towards the output: no-go *for this readout*. It is not the paper's test: every layer result there (Fig. 8) comes from a
    readout trained per layer on frozen features (attention probe, LLM projector, detection / depth decoders) or from patch-level matching
    (tracking), never from the cosine of a mean-pooled vector. Only the pooled output is trained to be compared by cosine, so a cosine test
    favours it by construction; mean pooling also discards the local detail middle layers are claimed to hold, and their large shared
    residual component (the global tokens from block ~33) dominates the mean. The on-claim test, a geolocation probe trained per layer on
    patch tokens, is `PE_PLAN.md` step 5b.
    **Step 5b, the paper's protocol** (`scripts/pe_probe.py`): the same attention probe (one cross-attention block, the paper's classification
    probe) trained per layer on PE-G 378 px patch tokens (27 x 27 average-pooled to 7 x 7) to predict 2,048 k-means GPS cells; 93,106 MP16
    training photos (train split + 65k random, no dev3k photo or photographer), 4,900 held out for the epoch, 15 epochs; the pooled output
    gets an MLP. dev3k, top cell within 25 / 200 / 750 / 2500 km: block 32 5.3 / 9.0 / 20.4 / 41.5, block 38 8.0 / 14.0 / 29.9 / 51.9, block 44
    13.9 / 25.1 / 47.6 / 69.0, block 47 19.8 / 34.9 / 58.4 / 77.7, last block 25.5 / 45.6 / 68.5 / 83.2, pooled 26.4 / 46.5 / 68.1 / 83.4.
    Geolocation behaves like the paper's ImageNet probe (best layer 50 of 50), not like its OCR / VQA tasks: with a trained readout too, the
    middle layers hold less, by 10-35 pts at 200 km. As head columns (each area's cell log-prob, and the mass within 100 km, raw and relative to
    the photo's other areas; gate as above): blocks 32-47 +0.06 to +0.71, last block +0.94, pooled +1.38 / +0.36. Closed for PE-G's middle
    layers (caveats: 7 x 7 pooling and ~100k probe photos; full-resolution tokens with ImageNet-scale data were not tried). A by-product to
    keep: the pooled output's region probe stacks on the cosine columns (PE-G + SigLIP2 +1.18 -> +1.86 / +1.03 in the gate).
    **Step 5c, trained head** (`pe_probe.py crossfit` / `head`): train-side probe columns cross-fitted (5 folds grouped by photographer, every
    probe photo of a fold's photographers left out; cross-fitted train top cell < 200 km 44.9% vs dev3k 46.3%, commons26 dev 42.9%: no leak
    visible), the answer head trained on train, 3 seeds, track V's V5 loop (reproduces V5 exactly). dev3k 47.17 / 30.63 -> PE-G + SigLIP2
    49.21 / 31.91 -> + probe **49.82 / 32.02** (+0.61 / +0.11 over PE-G + SigLIP2, SE 0.26; +2.66 / +1.39 over the head); commons26 dev 25.01 /
    7.68 -> 26.89 / 7.78 -> **27.70 / 7.92** (+0.82 / +0.14, SE 0.48; +2.70 over the head). Track V's bar (declared before the run: >= +0.5 on
    dev3k, commons26 dev not down) passes. One image model (PE-G pooled output, 378 px for the probe) feeds two kinds of head column:
    similarity to each area's photos and a learned region prior.

- **R17. PE-G as a search: raw cosine adds ~0 to Pinpoint's contrastive search, but the same small trained projection on PE-G beats it by
    +5.5 at 25 km (K = 32), and +5.0 over SigLIP2 with the same recipe** (`scripts/pe_retrieval.py`, PE_PLAN step 7 = B1; Oct 22-23).
    Gallery: a random 10% of the released index (408,092 photos, seed 0; no dev3k photos in it), PE-G 378 px pooled output; queries dev3k;
    % with a gallery photo within 1 / 25 / 200 km among the top K. **Raw** (`recall.log`, original protocol, K = 32): contrastive 35.4 / 61.1 /
    81.2, raw SigLIP2 29.2 / 52.6 / 76.3, raw PE-G 31.9 / 54.7 / 76.5; contrastive + raw PE-G at K/2 each 61.3 (+0.2; photographer excluded
    -1.2). PE-G finds a photo within 25 km for 18.6% of contrastive's 1,168 misses, but the half of K it takes costs about as much. **Trained**
    (`trained.log`): an MLP 1280/1536 -> 2048 -> 256, InfoNCE, positives another photo of the same ~22 km cell (distance only), batch pairs
    within 25 km masked as negatives, 20 epochs, trained on half the gallery (photographers split by hash, dev3k photographers out; 191k
    photos), evaluated on the other half (203k). K = 32 at 25 km, original / photographer excluded: contrastive 57.2 / 53.7, SigLIP2 trained
    57.7 / 53.7, **PE-G trained 62.7 / 58.2**; at K = 8 43.8 -> 49.2, K = 256 76.5 -> 80.3; union with contrastive at K/2 each 62.7 / 58.5
    (contrastive adds nothing once PE-G is trained). The recipe on SigLIP2 matches Pinpoint's contrastive search (trained on all of MP16), so
    it is a fair stand-in, and swapping the backbone to PE-G is worth ~5 pts of recall at every K. B1's declared bar (vs contrastive: union
    +2 at K = 32, or PE-G recovering >= 3% of contrastive's misses) passed only on its weak second clause; the trained result is the case for
    B2. Caveats: recall, not the reranker's top-1; half a 10% gallery (density differs from the full index); one seed.
    **Follow-ups (Oct 23, before B2; `b1_scaling.sh`, logs in `artifacts/retrieval/b1*/`).** Gallery density (nested shares of the eval half,
    K = 32 at 25 km, contrastive -> PE-G trained): 1.25% of the index 53.5 -> 58.7, 2.5% 55.7 -> 60.8, 5% 57.2 -> 62.7 (gap +5.2 / +5.1 /
    +5.5; photographer excluded +5.4 / +5.0 / +4.5, drifting down ~0.4 per doubling, so maybe +2.5-3 at the full index). Training data
    (photos of the train half, same steps): 48k 60.7, 96k 62.0, 191k 62.7 (+1.3, +0.7 per doubling): all of MP16 would add ~+1, so B2's
    value is the searchable index, not more training data. Both models as input (SigLIP2 + PE-G concatenated): 63.3 vs 62.7 (+0.6; excluded
    +1.1), small. **224 px is a no-go**: PE-G trained 40.3, raw 33.4 (vs 62.7 / 49.3 at 378 px); queries and gallery shift alike (same-photo
    cosine 378 vs 224 median 0.786 for both, so no bug) but the embeddings crowd (mean pairwise cosine 0.59 vs 0.43); 224 px ran ~100
    photos/s vs ~28, but B2 stays at 378 px. **Pool oracle** (dev3k, today's setft_own pool of ~46 candidates: 73.1% with one within 25 km,
    52.1% within 1 km), adding each search's top K from the 10% gallery, photographer kept / excluded: contrastive top 8 +1.1 / +1.0,
    SigLIP2 trained +1.7 / +1.3, **PE-G trained +3.8 / +3.0** (1 km +3.0 / +1.5); top 16 +2.3 / +1.9 vs **+5.5 / +4.5**; top 32 +3.9 / +3.3
    vs **+7.2 / +6.3**. Net of the contrastive control, ~+2-3 pts more photos with a right candidate from a gallery a tenth of the index.

- **R18. The reranker can't use PE-G's extra candidates, but adding a trained PE-G similarity to its score lifts its top-1 by +3.2 at
    25 km on dev3k (+3.9 with PE-G candidates added); the same projection on SigLIP2 adds +0.3** (`pe_retrieval.py rerank [--combo]`, Oct 23;
    `artifacts/retrieval/b1/{rerank,combo,combo_control}.log`). dev3k's setft_own pool (the pipeline's, photographer kept: 41% of pools hold a
    photo by the query's photographer; rescoring it reproduces the stored top-1 for 99.9%), top-1 44.30 / 28.33 (25 / 1 km). (1) Each search's
    top K from the 10% gallery added, scored by the setft_own reranker with rank 1 / not-in-shortlist features (as its wide lists' extra hits):
    contrastive top 8 +0.03, SigLIP2 trained -0.07, PE-G trained +0.10 / +0.13 (an added photo is the top-1 for 1.7%), although the pool oracle
    rises +3.8 (R17): the reranker sees only SigLIP2-based features. (2) Score = reranker score + lambda x the query's cosine to each candidate
    (lambda by 2-fold CV on dev3k, folds by parity; candidate PE-G embeddings from B2, the dev3k pools' 133k rows embedded first), today's
    pool: raw PE-G +1.23 / +0.97, **trained PE-G (R17's projection) +3.20 +- 0.55 / +2.77 (47.50 / 31.10)**, SigLIP2 trained (same recipe)
    +0.27 +- 0.33, PE-G + SigLIP2 trained summed +2.20. With PE-G trained top 16 added: **48.23 / 31.20 (+3.93 / +2.87)**; contrastive top 16
    added instead: +3.20 (no extra). So most of the gain is picking within today's pool, and PE-G's own candidates add ~+0.7 more. For scale,
    the answer head is 47.17 and 49.82 with V5 + probe columns; whether the trained PE-G similarity adds to that head is the next test.
    **commons26 dev transfers** (`--set commons26dev`, `combo_commons26dev.log`; its pools' 41k rows embedded into B2; queries at 378 px
    from step 5c): reranker top-1 19.65 / 6.04; + trained PE-G cosine **21.80 (+2.15 +- 0.91) / 6.45**; with PE-G trained top 16 added
    **22.62 (+2.97 +- 0.99) / 7.27 (+1.23)**; SigLIP2 trained -0.31 +- 0.45; raw PE-G cosine adds nothing (lambda 0 chosen in both folds).
    Caveats: one projection seed; lambda cross-fitted within each set (one parameter).

- **R19. The trained PE-G similarity as per-area columns adds to the best head: dev3k 49.82 -> 50.87 (+1.05 / +0.67 at 25 / 1 km),
    commons26 dev 27.70 -> 28.18 (+0.48 / +0.13); total over today's head +3.70 / +2.06 and +3.17 / +0.38** (`scripts/pe_trained_cols.py`,
    `artifacts/retrieval/pe_gtr/head.log`, Oct 23). R17's projection (trained on B1's gallery at 378 px) applied to track V's 448 px PE-G
    embeddings of each set's queries and area photos (same photos at both sizes: projected vectors agree at median cosine 0.977,
    query-photo similarities correlate 0.95), written as a 4th column model "pe_gtr" (best / top-3 over each area's 12 photos, both relative
    to the photo's other areas; train columns for the informative photos only, as V5), then track V's trained-head loop (3 seeds, every tool
    revealed). Configs: head 47.17 / 30.63, 25.01 / 7.68; pe_g + siglip_final + pe_probe 49.82 / 32.02, 27.70 / 7.92 (reproduces R16);
    **+ pe_gtr 50.87 / 32.69 (seeds 50.9 / 50.8 / 50.9), 28.18 / 8.05**; paired SE vs head 0.44 / 0.69. dev3k clears PE_PLAN's +1.0 bar over
    what we have; commons26 dev is positive but within noise. One PE-G embedding now feeds three kinds of head column: raw similarity to each
    area's photos, the trained (location-tuned) similarity, and the region prior. Caveats: one projection seed, trained on 191k photos (R17:
    more data adds ~+1 recall); 448 px inputs to a 378 px projection.
    **Each model alone** (same head and seeds, `pe_trained_cols.py single`, `single.log`), dev3k / commons26 dev at 25 km: SigLIP2 columns
    47.46 (+0.29) / 25.01 (+0.00); raw PE-G 48.53 (+1.36) / 26.17 (+1.16); **trained PE-G alone 50.38 (+3.21) / 27.43 (+2.42)**; all PE-G
    columns (raw + trained + probe) 50.66 (+3.49) / 28.08 (+3.07) vs + SigLIP2 50.87 / 28.18: once PE-G is in, SigLIP2's columns add
    +0.21 / +0.10 (noise). PE-G only beats SigLIP2 only by ~+3 in the head; the rest of the pipeline (retrieval, reranker, head base
    features) is still SigLIP2.

## Search-agent track (S)
