# Run Mapillary-baseline policies and score completed traces privately.
# Usage: python -m geo_search_env.runner --public-episodes fixtures/public/episodes.jsonl --private-labels fixtures/private/labels.jsonl --policy adaptive_search --output /tmp/mapillary-run

"""Reproducible public rollouts followed by separate private evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.osv_simulator import OSVSimulatorTools
from ..core.backend import ToolBackend
from ..core.contracts import EpisodeStatus, EpisodeTrace, GroundTruthRecord, PublicEpisode
from ..core.environment import SearchEnvironment
from ..core.geography import SnapshotWorld
from ..data.corpus import CorpusStore
from ..models.matching import Matcher, SyntheticFixtureMatcher
from .policies import POLICY_VERSION, make_policy
from .scoring import RewardConfig, ScoreResult, score_episode


def _jsonl_records(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_public_episodes(path: Path) -> tuple[PublicEpisode, ...]:
    episodes = tuple(PublicEpisode.from_dict(item) for item in _jsonl_records(path))
    ids = [item.episode_id for item in episodes]
    if not episodes or len(ids) != len(set(ids)):
        raise ValueError("public episodes must be nonempty with unique IDs")
    return episodes


def load_private_labels(path: Path) -> dict[str, GroundTruthRecord]:
    records = [GroundTruthRecord.from_dict(item) for item in _jsonl_records(path)]
    result = {item.episode_id: item for item in records}
    if len(result) != len(records):
        raise ValueError("private label episode IDs must be unique")
    return result


def run_episodes(
    episodes: Sequence[PublicEpisode],
    backend: ToolBackend,
    policy_name: str,
    *,
    seed: int = 0,
    budget_override: int | None = None,
) -> tuple[EpisodeTrace, ...]:
    if budget_override is not None and (type(budget_override) is not int or budget_override < 1):
        raise ValueError("budget_override must be a positive integer")
    environment = SearchEnvironment(backend)
    traces = []
    for original in episodes:
        episode = replace(original, budget=replace(original.budget, credits=budget_override)) if budget_override is not None else original
        policy = make_policy(policy_name, episode)
        observation = environment.reset(episode, seed=seed)
        for _ in range(episode.budget.max_nonterminal_actions + 2):
            transition = environment.step(policy.next_action(observation))
            observation = transition.observation
            if transition.termination_state is EpisodeStatus.FINISHED:
                break
        else:
            raise RuntimeError(f"policy did not finish episode {episode.episode_id}")
        traces.append(environment.get_trace())
    return tuple(traces)


def run_corpus_episodes(
    episodes: Sequence[PublicEpisode],
    corpus: CorpusStore,
    matcher: Matcher,
    policy_name: str,
    *,
    seed: int = 0,
    budget_override: int | None = None,
) -> tuple[EpisodeTrace, ...]:
    return run_episodes(episodes, OSVSimulatorTools(corpus, matcher), policy_name, seed=seed, budget_override=budget_override)


def score_traces(
    traces: Sequence[EpisodeTrace],
    private_labels: dict[str, GroundTruthRecord],
    world: SnapshotWorld | CorpusStore,
    reward_config: RewardConfig,
) -> tuple[ScoreResult, ...]:
    trace_ids = {item.public_episode.episode_id for item in traces}
    if trace_ids != set(private_labels) or len(trace_ids) != len(traces):
        raise ValueError("public trace IDs and private label IDs must match exactly")
    return tuple(score_episode(trace, private_labels[trace.public_episode.episode_id], reward_config, eligible_world=world) for trace in traces)


def summarize(metrics: Sequence[ScoreResult]) -> dict[str, float | int | None]:
    if not metrics:
        raise ValueError("cannot summarize an empty run")
    count = len(metrics)
    available_queries = sum((item.available_reference_count_1000m or 0) > 0 for item in metrics)
    available_assets = sum(item.available_reference_count_1000m or 0 for item in metrics)
    acquired_assets = sum(item.acquired_reference_count_1000m or 0 for item in metrics)
    return {
        "episode_count": count,
        "accuracy_1000m": sum(item.success_1000m for item in metrics) / count,
        "baseline_accuracy_1000m": sum(item.baseline_success_1000m for item in metrics) / count,
        "accuracy_delta_1000m": (sum(item.success_1000m for item in metrics) - sum(item.baseline_success_1000m for item in metrics)) / count,
        "accuracy_100m": sum(item.success_100m for item in metrics) / count,
        "accuracy_25m": sum(item.success_25m for item in metrics) / count,
        "mean_reward": sum(item.reward for item in metrics) / count,
        "mean_credits_used": sum(item.acquisition_cost for item in metrics) / count,
        "fallback_rate": sum(item.final_answer_failure for item in metrics) / count,
        "harmful_refinement_rate": sum(item.harmful_refinement for item in metrics) / count,
        "reference_available_query_count_1000m": available_queries,
        "acquired_reference_query_coverage_1000m": (sum((item.acquired_reference_count_1000m or 0) > 0 for item in metrics) / available_queries if available_queries else None),
        "acquired_reference_asset_coverage_1000m": acquired_assets / available_assets if available_assets else None,
        "final_answer_failure_count": sum(item.final_answer_failure for item in metrics),
    }


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def write_run(
    output: Path,
    traces: Sequence[EpisodeTrace],
    metrics: Sequence[ScoreResult],
    *,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    if len(traces) != len(metrics):
        raise ValueError("traces and metrics must have equal length")
    paths = [output / name for name in ("traces.jsonl", "metrics.jsonl", "summary.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("run output already exists")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(paths[0], [item.to_dict() for item in traces])
    _write_jsonl(paths[1], [item.to_dict() for item in metrics])
    summary = {"configuration": dict(configuration), "results": summarize(metrics)}
    with paths[2].open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-episodes", required=True, type=Path)
    parser.add_argument("--private-labels", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, default=Path("fixtures/public/snapshot.json"))
    parser.add_argument("--matcher-fixture", type=Path, default=Path("fixtures/public/matcher.json"))
    parser.add_argument("--policy", choices=("baseline_only", "round_robin_search", "adaptive_search"), required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    episodes = load_public_episodes(args.public_episodes)
    world = SnapshotWorld.from_dict(json.loads(args.snapshot.read_text(encoding="utf-8")))
    matcher = SyntheticFixtureMatcher.from_dict(json.loads(args.matcher_fixture.read_text(encoding="utf-8")))
    traces = run_episodes(episodes, OSVSimulatorTools(world, matcher), args.policy, seed=args.seed, budget_override=args.budget)
    labels = load_private_labels(args.private_labels)  # Loaded only after public rollouts finish.
    reward_config = RewardConfig()
    metrics = score_traces(traces, labels, world, reward_config)
    summary = write_run(args.output, traces, metrics, configuration={"corpus_version": world.corpus_version, "matcher_version": matcher.version, "policy": args.policy, "policy_version": POLICY_VERSION, "budget_override": args.budget, "seed": args.seed, "reward_config": reward_config.to_dict()})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
