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

## Search-agent track (S)
