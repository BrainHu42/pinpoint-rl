# Mapillary Baseline — Implementation Plan

The complete experiment design is in
[`docs/mapillary_baseline_plan.md`](docs/mapillary_baseline_plan.md). Work must
pass each gate before the next phase begins.

## Current foundation

- Strict public/private contracts.
- Four actions: `inspect_coverage`, `search_near`, `open_results`, and `finish`.
- Budgeted episode state machine and terminal-only private scoring.
- Lazy `CorpusStore` and `ImageResolver` boundaries.
- Deterministic in-memory fixtures and an API-shaped `OSVSimulatorTools`.
- Bounded `MapillaryApiClient` and `LiveMapillaryTools`.
- Baseline-only, fixed round-robin, and adaptive scripted policies.
- Reproducible runner with 1 km headline metrics.

The repository foundation is not evidence that the experiment is feasible or
RL-ready.

## Phase 0 — Stabilize the baseline contract

- Keep simulator and live capabilities identical.
- Test budgets, pagination, invalid actions, fallback, and private-label
  isolation.
- Keep 25 m and 100 m metrics diagnostic-only.
- Maintain a green test suite and fixture smoke run.

Gate: the same public trace schema runs unchanged against snapshot and live
backends.

## Phase 1 — Audit OSV-5M

- Implement the source-specific OSV-5M catalog and image resolver in the
  Pinpoint repository, reusing its existing loader.
- Pin dataset revisions and checksums.
- Review camera-coordinate provenance, rights, attribution, missing images,
  and sequence metadata.
- Create location-separated splits and reject same-sequence, same-capture, and
  near-duplicate query/reference pairs.
- Run the 20–50-query feasibility sample specified in the baseline plan.

Gate: independent 1 km evidence, matcher recall, and relaxed-budget acquisition
are sufficient for policy decisions to affect outcomes.

## Phase 2 — Characterize live Mapillary

- Run bounded credentialed probes over correct and wrong anchors, several
  radii, coverage tiles, multiple pages, sparse regions, deleted assets, and
  provider errors.
- Record public normalized responses, limits, cursor behavior, latency, and
  failures without credentials or private labels.
- Complete and review one scripted episode end to end.

Gate: live geographic search and coverage inspection support the frozen
four-action contract.

## Phase 3 — Calibrate the simulator

- Back `OSVSimulatorTools` with the audited disk catalog.
- Model measured Mapillary coverage summaries, page density, ordering,
  failures, stale assets, radius filtering, and costs.
- Compare snapshot and live behavior.
- Run baseline-only, round-robin, and adaptive policies across matched budgets.

Gate: simulator mismatch is quantified and scripted acquisition changes held-
out outcomes.

## Phase 4 — Add the model policy

- Implement replay and injected-client model adapters.
- Serialize public observations and images only.
- Strictly parse one of the four allowed actions.
- Record raw generations separately from parsed actions.
- Add prompt-isolation and trace-replay tests.
- Supervise on scripted and curated trajectories.

Gate: a small model completes held-out simulator and controlled live episodes
with a low invalid-action rate.

## Phase 5 — Offline RL and frozen live transfer

- Initialize from supervised training and optimize only in the simulator.
- Run multiple seeds and report learning curves.
- Select checkpoints without examining final live-test outcomes.
- Compare all policies at matched budgets on held-out snapshot and predeclared
  live sets.
- Report all-query 1 km accuracy, acquisition cost, fallback, harmful
  refinement, evidence availability, acquisition success, selection success,
  provider failures, and simulator-to-live deltas.

Gate: conclude successful transfer, narrow the claim, or attribute failure to
availability, acquisition, recognition, selection, or transfer.
