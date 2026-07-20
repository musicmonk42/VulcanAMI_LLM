"""Authoritative CognitiveEpisode lifecycle state machine."""
from __future__ import annotations

from enum import Enum


class EpisodeState(str, Enum):
    PERCEIVED = "perceived"
    INTERPRETED = "interpreted"
    GROUNDED = "grounded"
    DELIBERATING = "deliberating"
    EPISTEMICALLY_COMMITTED = "epistemically_committed"
    NORMATIVELY_AUTHORIZED = "normatively_authorized"
    EXECUTED = "executed"
    OBSERVED = "observed"
    COMMUNICATED = "communicated"
    CONSOLIDATED = "consolidated"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES


TERMINAL_STATES = frozenset({
    EpisodeState.CONSOLIDATED,
    EpisodeState.ABSTAINED,
    EpisodeState.BLOCKED,
    EpisodeState.FAILED,
    EpisodeState.CANCELLED,
})

ALLOWED_TRANSITIONS: dict[EpisodeState, frozenset[EpisodeState]] = {
    EpisodeState.PERCEIVED: frozenset({EpisodeState.INTERPRETED, EpisodeState.ABSTAINED, EpisodeState.BLOCKED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.INTERPRETED: frozenset({EpisodeState.GROUNDED, EpisodeState.ABSTAINED, EpisodeState.BLOCKED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.GROUNDED: frozenset({EpisodeState.DELIBERATING, EpisodeState.ABSTAINED, EpisodeState.BLOCKED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.DELIBERATING: frozenset({EpisodeState.EPISTEMICALLY_COMMITTED, EpisodeState.ABSTAINED, EpisodeState.BLOCKED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.EPISTEMICALLY_COMMITTED: frozenset({EpisodeState.NORMATIVELY_AUTHORIZED, EpisodeState.ABSTAINED, EpisodeState.BLOCKED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.NORMATIVELY_AUTHORIZED: frozenset({EpisodeState.EXECUTED, EpisodeState.ABSTAINED, EpisodeState.BLOCKED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.EXECUTED: frozenset({EpisodeState.OBSERVED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.OBSERVED: frozenset({EpisodeState.COMMUNICATED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.COMMUNICATED: frozenset({EpisodeState.CONSOLIDATED, EpisodeState.FAILED, EpisodeState.CANCELLED}),
    EpisodeState.CONSOLIDATED: frozenset(),
    EpisodeState.ABSTAINED: frozenset(),
    EpisodeState.BLOCKED: frozenset(),
    EpisodeState.FAILED: frozenset(),
    EpisodeState.CANCELLED: frozenset(),
}


class EpisodeTransitionError(ValueError):
    """Raised when an episode transition violates the authoritative lifecycle."""


def ensure_transition(current: EpisodeState, target: EpisodeState) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise EpisodeTransitionError(f"invalid CognitiveEpisode transition {current.value} -> {target.value}")
