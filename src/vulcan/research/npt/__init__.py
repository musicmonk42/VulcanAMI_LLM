"""External, non-authoritative instrumentation for NPT experiments.

This package is deliberately beneath :mod:`vulcan.research`, which the serving
import policy denies.  Its values are measurements and evidence classifications,
never beliefs, plans, effects, or claims about consciousness.
"""

from .instrumentor import (
    BlindInstrumentor,
    ClosureVector,
    EvidenceClassification,
    GroundTruthInstrumentor,
    PreregisteredExperiment,
    ResearchReport,
    SyntheticCyclicWorld,
    TelemetryRow,
    classify_evidence,
    run_flagship_experiment,
)

__all__ = (
    "BlindInstrumentor",
    "ClosureVector",
    "EvidenceClassification",
    "GroundTruthInstrumentor",
    "PreregisteredExperiment",
    "ResearchReport",
    "SyntheticCyclicWorld",
    "TelemetryRow",
    "classify_evidence",
    "run_flagship_experiment",
)
