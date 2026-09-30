# OSV-5M Feasibility Report

## Decision

**Stop Or Narrow**

This is a diagnostic pilot. Audit limitations prevent treating it as a final held-out result.

## Primary results

- Independent 1 km availability: **0/40 (0.0%)**
- Relaxed acquisition conditional on availability: **0/0 (0.0%)**
- Matcher top-1 conditional on acquisition: **0.0%**
- Matcher precision / recall at frozen threshold `1.000000`: **0.0% / 0.0%**
- Pinpoint baseline 1 km accuracy: **0.0%**
- Fixed-search 1 km accuracy: **0.0%** (+0.0 pp)
- Fixed-search harmful refinement: **0.0%**
- End-to-end rescue opportunities: **0/40**

## Gates

- FAIL — `audit_integrity`
- FAIL — `independent_availability`
- FAIL — `relaxed_acquisition`
- FAIL — `matcher_ranking`
- FAIL — `matcher_safety`
- FAIL — `consequentiality`
- PASS — `fixed_policy_safety`

## Audit limitations

- Exact and perceptual duplicate checks cover references within 100 m of pilot queries, not all 4.9M train images.
- Checkpoint metadata proves OSV training but does not by itself prove that OSV-5M test images were excluded.
- The pilot uses an in-memory cKDTree over a disk-backed coordinate cache rather than the planned persistent SQLite catalog.
- Country-frequency weighted sensitivity estimates are not implemented in this diagnostic run.
