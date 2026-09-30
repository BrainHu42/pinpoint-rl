# Candidate-Location Recall Report

This offline diagnostic exposes geographic location proposals only; retrieved corpus images are not experiment outputs.

Independent evidence is available for **20/40** report queries.

| Source | Locations | Truth ≤1 km | Truth ≤5 km | Truth ≤25 km | Evidence reachable ≤5 km |
| --- | ---: | ---: | ---: | ---: | ---: |
| mp16 | 10 | 0.0% | 2.5% | 10.0% | 0/20 (0.0%) |
| mp16 | 25 | 0.0% | 5.0% | 20.0% | 1/20 (5.0%) |
| mp16 | 50 | 2.5% | 10.0% | 30.0% | 3/20 (15.0%) |
| osv5m | 10 | 5.0% | 15.0% | 30.0% | 3/20 (15.0%) |
| osv5m | 25 | 7.5% | 22.5% | 40.0% | 6/20 (30.0%) |
| osv5m | 50 | 22.5% | 32.5% | 47.5% | 10/20 (50.0%) |
| union | 10 | 2.5% | 10.0% | 25.0% | 2/20 (10.0%) |
| union | 25 | 5.0% | 15.0% | 35.0% | 3/20 (15.0%) |
| union | 50 | 7.5% | 22.5% | 45.0% | 6/20 (30.0%) |

**Gate: PASS.** union top-50 conditional evidence recall >= 30% and >= 10 percentage points above MP16.

MP16-Pro and OSV-5M scores are not compared during fusion. Candidate locations are alternated and geographically deduplicated.
