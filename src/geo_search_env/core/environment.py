"""Budget accounting and state transitions for the baseline experiment."""

from __future__ import annotations

from copy import deepcopy

from .contracts import (
    Action,
    ActionKind,
    AssetIdsArgs,
    EpisodeStatus,
    EpisodeTrace,
    FinalSelection,
    FinishArgs,
    InspectCoverageArgs,
    Observation,
    PublicEpisode,
    SearchNearArgs,
    StructuredError,
    ToolResponse,
    TraceStep,
    Transition,
)
from .backend import ToolBackend, ToolBackendError


class SearchEnvironment:
    """Run isolated Mapillary search episodes without consulting private labels."""

    def __init__(self, tools: ToolBackend) -> None:
        self._tools = tools
        self._episode: PublicEpisode | None = None

    def reset(self, episode: PublicEpisode, *, seed: int = 0) -> Observation:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if not self._tools.supports_episode(episode.episode_id):
            raise ValueError(f"backend does not support episode {episode.episode_id!r}")
        self._episode = episode
        self._capabilities = self._tools.capabilities()
        self._status = EpisodeStatus.ACTIVE
        self._remaining_credits = episode.budget.credits
        self._action_count = 0
        self._discovered = {}
        self._opened = {}
        self._scores = {}
        self._candidate_coordinates = {item.candidate_id: item.coordinate for item in episode.initial_candidates}
        self._latest_response: ToolResponse | None = None
        self._steps: list[TraceStep] = []
        self._final_selection: FinalSelection | None = None
        self._update_exhaustion()
        self._initial_observation = self._observation()
        return deepcopy(self._initial_observation)

    def _require_episode(self) -> PublicEpisode:
        if self._episode is None:
            raise RuntimeError("reset must be called first")
        return self._episode

    def _observation(self) -> Observation:
        episode = self._require_episode()
        return Observation(
            episode.episode_id,
            self._status,
            self._remaining_credits,
            max(0, episode.budget.max_nonterminal_actions - self._action_count),
            tuple(self._candidate_coordinates),
            self._capabilities,
            tuple(self._discovered.values()),
            tuple(self._opened.values()),
            tuple(self._scores.values()),
            self._latest_response,
        )

    def _update_exhaustion(self) -> None:
        episode = self._require_episode()
        if self._status is EpisodeStatus.ACTIVE and (
            self._action_count >= episode.budget.max_nonterminal_actions
            or self._remaining_credits < episode.budget.minimum_acquisition_cost
        ):
            self._status = EpisodeStatus.FINAL_ANSWER_ONLY

    def _fallback(self, code: str) -> None:
        episode = self._require_episode()
        candidate_id = episode.baseline_candidate_id
        self._final_selection = FinalSelection(candidate_id, self._candidate_coordinates[candidate_id], True, code)
        self._status = EpisodeStatus.FINISHED
        self._latest_response = None

    def _finish(self, action: Action) -> StructuredError | None:
        assert isinstance(action.arguments, FinishArgs)
        candidate_id = action.arguments.candidate_id
        if candidate_id not in self._candidate_coordinates:
            error = StructuredError("invalid_final_candidate", "finish must select a valid candidate ID")
            self._fallback(error.code)
            return error
        self._final_selection = FinalSelection(candidate_id, self._candidate_coordinates[candidate_id])
        self._status = EpisodeStatus.FINISHED
        self._latest_response = None
        return None

    def _validate(self, action: Action) -> StructuredError | None:
        if action.kind in (ActionKind.SEARCH_NEAR, ActionKind.INSPECT_COVERAGE):
            args = action.arguments
            assert isinstance(args, (SearchNearArgs, InspectCoverageArgs))
            if (
                args.anchor_id not in self._candidate_coordinates
                and args.anchor_id not in self._discovered
            ):
                return StructuredError("invalid_anchor", "anchor must be an initial candidate or discovered Mapillary card")
            cap = self._capabilities.capability(action.kind)
            if cap.max_radius_m is not None and args.radius_m > cap.max_radius_m:
                return StructuredError("invalid_radius", "radius exceeds the backend limit")
        elif action.kind is ActionKind.OPEN_RESULTS:
            assert isinstance(action.arguments, AssetIdsArgs)
            if any(asset_id not in self._discovered for asset_id in action.arguments.asset_ids):
                return StructuredError("asset_not_discovered", "all assets must be discovered before opening")
            cap = self._capabilities.capability(ActionKind.OPEN_RESULTS)
            if cap.max_batch_size is not None and len(action.arguments.asset_ids) > cap.max_batch_size:
                return StructuredError("batch_too_large", "open batch exceeds the backend limit")
        return self._tools.validate_action(action)

    def _cost(self, action: Action) -> int:
        budget = self._require_episode().budget
        if action.kind is ActionKind.SEARCH_NEAR:
            return budget.search_cost
        if action.kind is ActionKind.INSPECT_COVERAGE:
            return budget.coverage_cost
        assert isinstance(action.arguments, AssetIdsArgs)
        return sum(asset_id not in self._opened for asset_id in action.arguments.asset_ids) * budget.open_result_cost

    def _execute(self, action: Action) -> ToolResponse:
        episode = self._require_episode()
        if action.kind is ActionKind.SEARCH_NEAR:
            assert isinstance(action.arguments, SearchNearArgs)
            return self._tools.search_near(episode, action.arguments.anchor_id, action.arguments.radius_m, action.arguments.cursor)
        if action.kind is ActionKind.INSPECT_COVERAGE:
            assert isinstance(action.arguments, InspectCoverageArgs)
            return self._tools.inspect_coverage(episode, action.arguments.anchor_id, action.arguments.radius_m)
        assert isinstance(action.arguments, AssetIdsArgs)
        new_ids = tuple(asset_id for asset_id in action.arguments.asset_ids if asset_id not in self._opened)
        return self._tools.open_results(episode, new_ids) if new_ids else ToolResponse(ActionKind.OPEN_RESULTS)

    def _apply(self, response: ToolResponse) -> None:
        for card in response.search_results:
            self._discovered.setdefault(card.asset_id, card)
        for asset in response.opened_assets:
            self._opened.setdefault(asset.asset_id, asset)
            self._candidate_coordinates.setdefault(asset.asset_id, asset.published_coordinate)
        for score in response.match_scores:
            self._scores.setdefault(score.asset_id, score)

    def _record(self, action: Action, cost: int, error: StructuredError | None) -> Transition:
        observation = self._observation()
        self._steps.append(TraceStep(action, observation, cost, error))
        return Transition(deepcopy(observation), cost, error, self._status)

    def step(self, action: Action) -> Transition:
        self._require_episode()
        if self._status is EpisodeStatus.FINISHED:
            raise RuntimeError("cannot step a finished episode")
        if self._status is EpisodeStatus.FINAL_ANSWER_ONLY:
            if action.kind is ActionKind.FINISH:
                error = self._finish(action)
            else:
                error = StructuredError("final_answer_required", "only finish is permitted after budget exhaustion")
                self._fallback(error.code)
            return self._record(action, 0, error)
        if action.kind is ActionKind.FINISH:
            return self._record(action, 0, self._finish(action))

        self._action_count += 1
        error = self._validate(action)
        if error is not None:
            self._latest_response = None
            self._update_exhaustion()
            return self._record(action, 0, error)
        cost = self._cost(action)
        if cost > self._remaining_credits:
            error = StructuredError("unaffordable_action", "action exceeds the remaining credit budget")
            self._latest_response = None
            self._update_exhaustion()
            return self._record(action, 0, error)
        try:
            response = self._execute(action)
        except ToolBackendError as backend_error:
            charged = cost if backend_error.charge_attempt else 0
            self._remaining_credits -= charged
            self._latest_response = None
            self._update_exhaustion()
            return self._record(action, charged, StructuredError(backend_error.code, str(backend_error)))
        self._remaining_credits -= cost
        self._apply(response)
        self._latest_response = response
        self._update_exhaustion()
        return self._record(action, cost, None)

    def get_trace(self) -> EpisodeTrace:
        self._require_episode()
        return deepcopy(EpisodeTrace(self._episode, self._initial_observation, tuple(self._steps), self._final_selection))
