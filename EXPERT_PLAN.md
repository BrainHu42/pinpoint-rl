# Plan: choose among candidate areas the way an expert does (2026-10-15)

Replaces the earlier plans (agent direction, hypothesis testing, look-zoom-rerank, text search, knowledge SFT); they are at git tag
`plans-2026-10-15`. Results and lessons: `LEARNINGS.md`, `LESSONS.md`.

## Goal
An RL agent that, given retrieval's candidate areas, works like a professional geolocator: find what distinguishes *these* candidates,
fetch reference imagery and map data of each, eliminate, verify, and stop once one is confirmed. Retrieval (Pinpoint + our pipeline) stays
the candidate generator. The RL policy decides which check to run on which candidates; the checks cost real time or money, so that decision
matters.

## Where we are
- Best system (lesson 94 action agent): dev3k 39.7% < 25 km. Test set, primary protocol (photographer's photos kept): 48.2% vs Pinpoint
  46.4% at 25 km, but 25.5% vs 28.9% at 1 km (yfcc4k 27.3 vs 33.5); im2gps3k +4.7 at 25 km (lesson 98).
- The gap is choosing, not proposing: some Pinpoint area is right for 51.5% of dev3k, we pick right ~40%. Every per-area signal we compute
  is saturated (lessons 85-97: AUC 0.6-0.74); new-place proposers fail (lessons 83, 87, 95, 96).
- The RL policy ties "take every action" (43.5 vs 43.4% on the test set): today's actions are cheap, independent and reveal fixed columns,
  so there is nothing to decide.

## What experts do that we don't
| Expert step | Us |
|---|---|
| Ask what differs between *these* candidates, then check exactly that | Each area scored alone with fixed features |
| Look at reference imagery of each candidate (Street View, other people's photos, satellite) and match specific structures | MegaLoc against Pinpoint's own index photos only; the right ones show other scenes of the area (lessons 95, 97) |
| Check the distinguishing feature on a map near each candidate | Map checks of generic features (lesson 77: nothing, the wrong areas look alike) |
| Next check depends on the last result; stop when confirmed | One-shot reveal of fixed columns |

## The agent
- **State:** the photo and its candidate areas with the current pipeline's features (the free glance).
- **Actions** (tool signatures like the live services; offline backends for training, live ones at deployment, as the tools in `tools/`):
  - `references(area, point, k)`: photos taken near a point of an area. Offline: YFCC100M snapshot (48.5M geotagged; metadata in
    `data/yfcc100m/geo.npz`, images in the public `multimedia-commons` S3 bucket; ~10x the index's density near dev3k truths). Live: Flickr,
    Mapillary, Wikimedia Commons. Matched with MegaLoc and shown to the reader.
  - `contrast(A, B)`: a reader sees the photo and the references of two areas, names the feature that distinguishes them and picks one.
    Readers: local open models only (the 4B, larger open VLMs on our GPU, our own trained models); no paid LLMs (user, 2026-10-15).
  - `features_near(area, feature)`: map check of the distinguishing feature only (`tools/maps.py`).
  - `read`, `recognise`, `zoom` (existing), `stop`.
- **Reward:** [answer < 25 km] (and < 1 km) minus lambda x the call's real cost (seconds or dollars). Ground truth only, no distillation.
- **Training without paying per step:** each tool's output depends only on (photo, area) or (photo, pair), so it is fetched once and cached
  for a train subset (~10k photos) and dev3k; RL replays the cache. First as a small policy over cached outcomes (as `action_agent.py`),
  then the 4B itself as the multi-turn policy if the new actions carry signal.

## First experiment (real task, dev3k)
500 dev3k photos, top 3 areas each. Fetch YFCC references near each area's top candidates; run `contrast` on the top pair with the 4B (and
a larger local VLM). Measure net fixes / breaks of following the contrast's pick over the answer head's, at 25 and 1 km.
Same-recipe control: the same reader shown each area's existing 12 restricted-search photos instead of fetched references, which isolates
what the reference imagery adds. Worth building the agent if the fetched-reference contrast is clearly ahead of the control and of the
answer head on the photos where they disagree.

## Benchmark and training protocol
- Primary benchmark: the original Pinpoint protocol, photographer's photos kept (`PINPOINT_KEEP_OWN=1`); a table per benchmark x
  1 / 25 / 200 / 750 / 2500 km vs the released Pinpoint. Photographer-excluded numbers are secondary. Test set only when the user asks.
- In parallel: retrain the pipeline without the photographer exclusion (train / dev3k pools, reranker fine-tune, heads, policy) so it can
  use the photographer's photos as Pinpoint does; lesson 74 found the exclusion was the fine-tune's key ingredient, so re-test that first.
- Leak guards for `references`: always drop the photo itself and near-duplicates (yfcc4k photos are YFCC photos); by default also the
  photographer, so gains come from place matching.
- Develop and decide on dev3k; measure beyond the current best pipeline, not against a weaker baseline.

## Open questions
- Should `references` return the photographer's other photos under the primary protocol (consistent with it, but it is trip retrieval)?
- Can a local reader do what Sonnet did with named candidates (lesson 105: +0.94 pts at 25 km; the 4B is at chance), or must we train one?
- Flickr link rot and the S3 image key mapping for YFCC; Mapillary needs a token.
- About a third of hard photos show nothing place-specific (lessons 35, 80); no check fixes those, and the policy should learn to stop.
