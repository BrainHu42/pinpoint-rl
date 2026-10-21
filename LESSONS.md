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

## Search-agent track (S)
