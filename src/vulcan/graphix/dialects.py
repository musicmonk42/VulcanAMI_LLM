"""Code-owned dialect names for the canonical cognitive Graphix chain."""

from __future__ import annotations

LANGUAGE_PROPOSAL = "graphix.language.proposal"
INTERPRETATION_CANDIDATE = "graphix.language.candidate"
PLAN_CANDIDATE = "graphix.plan.candidate"
EPISTEMIC_COMMIT_CANDIDATE = "graphix.epistemic.candidate"
RESPONSE_PROJECTION = "graphix.response.projection"

CANONICAL_DIALECTS = frozenset(
    {
        LANGUAGE_PROPOSAL,
        INTERPRETATION_CANDIDATE,
        PLAN_CANDIDATE,
        EPISTEMIC_COMMIT_CANDIDATE,
        RESPONSE_PROJECTION,
    }
)
