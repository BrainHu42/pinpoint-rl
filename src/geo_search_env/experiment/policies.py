"""Deterministic baseline policies over public observations only."""

from __future__ import annotations

from typing import Protocol

from ..core.contracts import Action, EpisodeStatus, Observation, PublicEpisode

POLICY_VERSION = "mapillary-scripted-v1"


class Policy(Protocol):
    def next_action(self, observation: Observation) -> Action: ...


class BaselineOnlyPolicy:
    def __init__(self, episode: PublicEpisode) -> None:
        self.episode = episode

    def next_action(self, observation: Observation) -> Action:
        return Action.finish(self.episode.baseline_candidate_id)


class _SearchPolicy:
    radius_m = 1_000.0
    minimum_select_score = 0.8

    def __init__(self, episode: PublicEpisode) -> None:
        self.episode = episode
        self._searched: set[str] = set()
        self._expanded: set[str] = set()

    def _can_spend(self, observation: Observation, amount: int) -> bool:
        return observation.status is EpisodeStatus.ACTIVE and observation.remaining_nonterminal_actions > 0 and observation.remaining_credits >= amount

    def _selected_id(self, observation: Observation) -> str:
        scores = {item.asset_id: item.score for item in observation.match_scores}
        eligible = sorted(
            (-scores[item.asset_id], item.asset_id)
            for item in observation.opened_assets
            if scores.get(item.asset_id, 0) >= self.minimum_select_score
        )
        return eligible[0][1] if eligible else self.episode.baseline_candidate_id

    def _search_initial(self, observation: Observation, *, first_only: bool = False) -> Action | None:
        candidates = sorted(self.episode.initial_candidates, key=lambda item: item.rank)
        if first_only:
            candidates = candidates[:1]
        for candidate in candidates:
            if candidate.candidate_id not in self._searched:
                if self._can_spend(observation, self.episode.budget.search_cost):
                    self._searched.add(candidate.candidate_id)
                    return Action.search_near(candidate.candidate_id, self.radius_m)
                return None
        return None

    def _open_next(self, observation: Observation) -> Action | None:
        opened = {item.asset_id for item in observation.opened_assets}
        if self._can_spend(observation, self.episode.budget.open_result_cost):
            for card in observation.discovered_assets:
                if card.asset_id not in opened:
                    return Action.open_results((card.asset_id,))
        return None

    def _finish(self, observation: Observation) -> Action:
        return Action.finish(self._selected_id(observation))


class RoundRobinSearchPolicy(_SearchPolicy):
    """Search every upstream hypothesis, then open results in returned order."""

    def next_action(self, observation: Observation) -> Action:
        if observation.status is not EpisodeStatus.ACTIVE:
            return self._finish(observation)
        return self._search_initial(observation) or self._open_next(observation) or self._finish(observation)


class AdaptiveSearchPolicy(_SearchPolicy):
    """Inspect the best hypothesis first, then expand or explore as evidence warrants."""

    def _strong_match(self, observation: Observation) -> bool:
        return self._selected_id(observation) != self.episode.baseline_candidate_id and any(
            item.score >= 0.9 for item in observation.match_scores
        )

    def _expand(self, observation: Observation) -> Action | None:
        scores = {item.asset_id: item.score for item in observation.match_scores}
        if not self._can_spend(observation, self.episode.budget.search_cost):
            return None
        for asset in observation.opened_assets:
            if asset.asset_id not in self._expanded and scores.get(asset.asset_id, 0) >= 0.6:
                self._expanded.add(asset.asset_id)
                return Action.search_near(asset.asset_id, self.radius_m)
        return None

    def next_action(self, observation: Observation) -> Action:
        if observation.status is not EpisodeStatus.ACTIVE or self._strong_match(observation):
            return self._finish(observation)
        return (
            self._search_initial(observation, first_only=True)
            or self._open_next(observation)
            or self._expand(observation)
            or self._search_initial(observation)
            or self._finish(observation)
        )


def make_policy(name: str, episode: PublicEpisode) -> Policy:
    policies = {
        "baseline_only": BaselineOnlyPolicy,
        "round_robin_search": RoundRobinSearchPolicy,
        "adaptive_search": AdaptiveSearchPolicy,
    }
    try:
        return policies[name](episode)
    except KeyError as error:
        raise ValueError(f"unknown policy: {name}") from error
