# PE-core-G as the image model (track R with track V, 2026-10-22)

**Question.** Every Pinpoint score comes from one frozen SigLIP2-g embedding, and track R found it is the ceiling (R2-R4). Does Meta's
Perception Encoder (PE-core-G, arXiv 2504.13181) carry location signal SigLIP2 lacks, and where in the pipeline does it add most? Using both
SigLIP2 and PE-G is fine when both pull their weight (user, 2026-10-22); the evidence so far says they do (PE-G + SigLIP2 columns +2.04 vs
PE-G alone +1.37).

**Measure.** dev3k, original protocol, % < 25 / 1 km over the current answer head (47.2 / 30.8). Bar: >= +1.0 pt at 25 km, 1 km not down
by more than 0.3. Test set untouched.

## Status (update as steps finish)
| # | Step | Owner | When (Pacific) | Status | Result |
|---|---|---|---|---|---|
| 1 | PE-G / PE-L / DINOv3 as per-area columns on the head (gate, CV on dev3k) | track V | done | done | PE-G +0.79 / +0.63, PE-L +0.23, DINOv3-7B +0.23, PE-G + DINOv3 +1.04 |
| 2 | Control: SigLIP2's final embedding as the same columns (no GPU, MP16 cache) | track R | done | done | SigLIP2 -0.03; SigLIP2 + PE-G +1.19 (SE 0.35): the column form adds nothing, the gain is PE-G's |
| 3 | Each model alone: is the best-matching area right? | track R | done | done | PE-G 67.5% vs SigLIP2 63.4% (head's misses: 30.9% vs 24.6%) |
| 4 | V5: answer head retrained on train with the columns, 3 seeds (head alone 47.17 / 30.63) | track V | done | **done, passes** | PE-G +1.37 / +0.97; DINOv3 +0.78; SigLIP2 +0.29; PE-G + DINOv3 +1.59 (pre-declared bar); **PE-G + SigLIP2 +2.04 / +1.28 (SE 0.38)**; all three +2.12 (lesson V5, worktree-value-agent 3f7d1f5) |
| 5 | One PE-G pass at 378 px (~SigLIP2's 384): pooled output (resolution) + mean-pooled layers 32 / 38 / 44 / 47 (intermediate layers), same gate and solo check | track R | done | done (lesson R16) | 378 px pooled +0.87 (448 px +0.79; solo 67.1 vs 67.5%): resolution isn't the source, use 378 px. Mean-pooled layers 32 / 38 / 44 / 47 by cosine +0.34 / +0.19 / +0.39 / +0.37: this readout fails, but it is not the paper's test (5b) |
| 5b | On-claim layer test: a geolocation probe trained per layer on PE-G patch tokens (the paper's Fig. 8 protocol) | track R | done | **done, no-go (R16)** | dev3k top cell within 200 km: blocks 32 / 38 / 44 / 47 9.0 / 14.0 / 25.1 / 34.9, last 45.6, pooled 46.5: the output is best, as for the paper's ImageNet probe. By-product: the pooled region probe as head columns +1.38, on top of PE-G + SigLIP2 cosine +1.18 -> +1.86 / +1.03 (gate) |
| 5c | Region-probe columns in the trained head (V5 setup + probe columns; train-side columns cross-fitted, the probe must not see the photo it scores). Bar (track V, declared): probe adds >= +0.5 at 25 km over pe_g + siglip_final on dev3k, commons26 dev not down | files and head test: track R | ~5:00-5:45 PM | head test running (track R, `pe_probe.py head`, log `artifacts/retrieval/probe/head.log`) | cross-fitted train columns 44.9% top cell < 200 km vs dev3k 46.3%, commons26 dev 42.9%: no leak visible |
| 6 | Commit PE-G next to SigLIP2 in the answer head; confirm on commons26 dev (below, A) | track V (`scripts/second_model_commons.sh`, ac43ec4) | done | **done, passes** | commons26 dev (head 25.01 / 7.68): PE-G +1.16 / -0.03 (SE 0.45), **PE-G + SigLIP2 +1.88 / +0.10 (SE 0.54)**; dev3k reproduced (+1.37, +2.04); lesson V5 |
| 7 | Retrieval pre-check on a 10% MP16 gallery (below, B1) | track R | next night | | |
| 8 | Full PE-G index as a retrieval modality next to SigLIP2 (below, B2) | track R | if 7 passes, with the user | | |

Step 4 settled that the gain is real, so PE-G stays. Step 5 says whether it is PE's knowledge or just
more pixels (448 vs 378 px), sets the resolution for any re-embed, and tried the cheapest middle-layer readout (cosine of mean-pooled tokens; it fails). Lower priority, run only if asked: SigLIP2 intermediate layers (~40 min GPU); step 2
and the paper make a gain unlikely.

## Step 5b: the paper's claim, tested as the paper tests it
The paper reads every layer with a readout *trained* on that layer's frozen features (an attention probe for classification; for OCR / VQA a
projector into an LLM; detection and depth decoders) or by patch-to-patch matching (tracking). Step 5 compared mean-pooled vectors by cosine,
which only the pooled output is trained for, and which averages away the local detail (signs, text, architecture) middle layers are said to
hold. Geolocation is a fine-grained, long-tail task, closer to the paper's OCR / VQA tasks (best layers 38 and 47 of 50) than to ImageNet
(best layer 50).
- Photos: the 35,359 train photos plus ~65k random MP16 photos (never dev3k or benchmark photos), and the 3,000 dev3k queries. PE-G at 378 px
  (~30 photos/s: ~1 h).
- Features: patch tokens of blocks 32 / 38 / 44 / 47 and of the last block, average-pooled 27 x 27 -> 7 x 7 (49 tokens; ~75 GB on EBS),
  plus the pooled output.
- Readout: the same attention probe per layer (one cross-attention block + linear layer), trained to predict geographic cells (~2-4k MP16
  cells); for the pooled output a linear probe and an MLP. Same recipe and budget for every layer.
- Measure on dev3k: % of queries whose predicted cell is within 25 / 200 / 750 km. Go: a middle layer beats both the last block's tokens and
  the pooled output by >= 2 pts at 200 or 750 km (SE ~0.9). Then the probe's probability of each candidate area's cell becomes head columns,
  gated as in step 2 against the pooled-output probe.
- Optional, lower: patch-level matching between the query and an area's photos at block ~32 (the paper's tracking protocol); lesson 101's
  patch matcher (MegaLoc / DINOv2 patches, fetched references) saturated, so it goes after the probe.

## Next steps (PE-G next to SigLIP2)
V5 passed the bar, so PE-G stays. Each step keeps SigLIP2 and asks what PE-G adds on top of it.

- **A. Answer head: PE-G per-area columns next to SigLIP2's** (V5's +2.04 / +1.28 configuration), pooled output; 448 px as in V5, or 378 px (step 5: equal within noise, cheaper).
  - Confirm on commons26 dev (977 photos; the PE-G columns need its queries + area photos embedded, ~30 min). Go: not below the current
    head there.
  - Optional simplification: the same head without MegaLoc; drop MegaLoc only if 25 km and 1 km hold.
  - Inference cost: PE-G on the query + ~12 photos per candidate area. This re-ranks existing candidates only.
- **B. Retrieval: PE-G as a search next to SigLIP2's**, the only lever on retrieval misses (26.5 pts on dev3k).
  - **B1 (pre-check, ~4-5 h GPU, one night):** embed a random 10% of MP16 (~410k photos) with PE-G. On that gallery, compare recall@K
    (within 25 km, K = 8 / 32 / 256) for dev3k queries: SigLIP2 alone, PE-G alone, and their union at equal total K. Go: the union beats
    SigLIP2 at equal K by >= 2 pts at K = 32, or PE-G finds a right photo for >= 3% of the photos SigLIP2 misses.
  - **B2 (if B1 passes):** full re-embed of MP16 with PE-G (4.12M photos; ~40-50 h at 448 px, less at 378, resumable over 2-3 nights
    because of the 02:00 shutdown) as a fourth search in the reranker's pool, then re-fine-tune the reranker with PE-G similarity features.
    Long run: confirm with the user first.
- Both A and B keep every SigLIP2 component; no backbone swap is planned unless B shows PE-G retrieval makes SigLIP2's redundant.

## Where things are
- Track R: `scripts/siglip_gate.py` (branch `worktree-retrieval-track`), outputs `artifacts/retrieval/gate/` (`gate_siglip_final.log`, `solo.log`).
- Track V: `scripts/second_model_gate.py` (`embed` / `gate` / `head` / `siglip`), `scripts/second_model_where.py` (branch `worktree-value-agent`),
  embeddings and logs `artifacts/value/`; V5 lands in its `LESSONS.md`.
- Paper notes: best layers are intermediate for local tasks (OCR, detection), the last layer for global ones; PE-G beats SigLIP2-g by 17.5
  pts on zero-shot Countries211.
