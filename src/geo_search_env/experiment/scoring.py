"""Terminal-only private scoring for the 1 km Mapillary experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..core.contracts import Coordinate, EpisodeTrace, GroundTruthRecord
from ..core.geography import SnapshotWorld, distance_m
from ..data.corpus import CorpusStore


@dataclass(frozen=True, slots=True)
class RewardConfig:
    threshold_m: float = 1_000.0
    success_reward: float = 1.0
    progress_weight: float = 0.2
    cost_per_credit: float = 0.01
    failure_penalty: float = 0.25

    def __post_init__(self) -> None:
        for name in ("threshold_m", "success_reward", "progress_weight", "cost_per_credit", "failure_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name == "threshold_m" and value == 0:
                raise ValueError("threshold_m must be positive")
            object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ScoreResult:
    episode_id: str
    reward: float
    final_candidate_id: str
    final_error_m: float
    baseline_error_m: float
    success_1000m: bool
    baseline_success_1000m: bool
    success_25m: bool
    success_100m: bool
    acquisition_cost: int
    refinement_improved: bool
    harmful_refinement: bool
    final_answer_failure: bool
    missing_final_output: bool
    available_reference_count_1000m: int | None
    acquired_reference_count_1000m: int | None
    acquired_reference_coverage_1000m: float | None

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def score_episode(
    trace: EpisodeTrace,
    ground_truth: GroundTruthRecord,
    reward_config: RewardConfig | None = None,
    *,
    eligible_world: SnapshotWorld | CorpusStore | None = None,
) -> ScoreResult:
    config = reward_config or RewardConfig()
    if trace.public_episode.episode_id != ground_truth.episode_id:
        raise ValueError("trace and ground truth episode IDs differ")
    episode = trace.public_episode
    initial = {item.candidate_id: item.coordinate for item in episode.initial_candidates}
    candidates: dict[str, Coordinate] = dict(initial)
    for step in trace.steps:
        for asset in step.observation.opened_assets:
            candidates[asset.asset_id] = asset.published_coordinate

    missing = trace.final_selection is None
    selection = trace.final_selection
    if missing:
        final_id = episode.baseline_candidate_id
        final_coordinate = initial[final_id]
    else:
        final_id = selection.candidate_id
        if final_id not in candidates or selection.coordinate != candidates[final_id]:
            raise ValueError("final selection is not supported by public evidence")
        final_coordinate = candidates[final_id]

    truth = ground_truth.query_coordinate
    baseline_error = distance_m(initial[episode.baseline_candidate_id], truth)
    final_error = distance_m(final_coordinate, truth)
    failure = missing or selection.used_fallback
    reward = (
        config.success_reward * (final_error <= config.threshold_m)
        + config.progress_weight * (math.log1p(baseline_error) - math.log1p(final_error))
        - config.cost_per_credit * trace.total_cost
        - config.failure_penalty * failure
    )

    available_count = acquired_count = None
    coverage = None
    if eligible_world is not None:
        nearby = (
            tuple(item for item in eligible_world.references if distance_m(item.published_coordinate, truth) <= 1_000)
            if isinstance(eligible_world, SnapshotWorld)
            else tuple(eligible_world.eligible_references_near(truth, 1_000))
        )
        eligible_ids = {item.asset_id for item in nearby}
        acquired_ids = {item.asset_id for step in trace.steps for item in step.observation.opened_assets}
        available_count = len(eligible_ids)
        acquired_count = len(eligible_ids & acquired_ids)
        coverage = acquired_count / available_count if available_count else None

    return ScoreResult(
        episode.episode_id,
        reward,
        final_id,
        final_error,
        baseline_error,
        final_error <= 1_000,
        baseline_error <= 1_000,
        final_error <= 25,
        final_error <= 100,
        trace.total_cost,
        final_error < baseline_error,
        baseline_error <= 1_000 < final_error,
        failure,
        missing,
        available_count,
        acquired_count,
        coverage,
    )
