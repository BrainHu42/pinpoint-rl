"""Backend boundary shared by the environment and provider adapters."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .contracts import (
    Action,
    BackendCapabilities,
    PublicEpisode,
    StructuredError,
    ToolResponse,
)


@runtime_checkable
class ToolBackend(Protocol):
    def supports_episode(self, episode_id: str) -> bool: ...

    def capabilities(self) -> BackendCapabilities: ...

    def validate_action(self, action: Action) -> StructuredError | None: ...

    def search_near(
        self, episode: PublicEpisode, anchor_id: str, radius_m: float, cursor: str | None
    ) -> ToolResponse: ...

    def inspect_coverage(
        self, episode: PublicEpisode, anchor_id: str, radius_m: float
    ) -> ToolResponse: ...

    def open_results(self, episode: PublicEpisode, asset_ids: Sequence[str]) -> ToolResponse: ...


class ToolBackendError(ValueError):
    """Provider error with an environment-visible code and charging policy."""

    def __init__(self, code: str, message: str, *, charge_attempt: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.charge_attempt = charge_attempt
