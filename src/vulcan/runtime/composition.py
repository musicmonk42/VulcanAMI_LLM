"""Typed, fail-closed composition root for the canonical serving graph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from vulcan.learning_owner import LearningOwner
from vulcan.memory.composition import compose_governed_memory
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.transactions import ConstitutionalTransactionService
from vulcan.platform import require_canonical_serving_platform
from vulcan.safety.response_adapter import EnhancedSafetyResponseAdapter
from vulcan.safety.safety_types import ResponseSafetyDecision, ResponseSafetyStatus

from .alignment import AlignmentRegistry
from .audit import CanonicalAudit
from .capabilities import CapabilityManifestAuthority
from .constitutional_kernel import ConstitutionalCognitiveKernel
from .container import RuntimeContainer, RuntimeOwnerInputs
from .domain_registry import PersistentDomainRegistry
from .errors import StartupErrorCategory, StartupFailure
from .finalization import SafetyResponseFinalizer
from .output import DeterministicLanguageOutput
from vulcan.graphix.runtime import DeterministicLanguageInput
from .settings import RuntimeSettings, VulcanEnvironment
from .state_authorities import StateAuthoritySet


class DevelopmentStubWorld:
    production_ready = False
    snapshot_id = "development-stub-world"

    def readiness(self) -> bool:
        return False


class DevelopmentStubSafety:
    production_ready = False

    def readiness(self) -> bool:
        return False


class DevelopmentUnavailableSafetyPort:
    """Typed deny-only finalization port for explicit development stub mode."""

    def readiness(self) -> bool:
        return True

    async def evaluate_response(self, response_text, context):
        return ResponseSafetyDecision(
            ResponseSafetyStatus.UNAVAILABLE,
            "development stub has no response safety authority",
            0.0,
            {},
        )


class DevelopmentStubDeployment:
    """Compatibility name for the explicitly unavailable development ports."""

    production_ready = False

    def close(self) -> None:
        return None


class LegacyWorldReadOnlyAdapter:
    """Narrow legacy adapter; exposes no reasoning or mutation operation."""

    def __init__(self, legacy_world: Any):
        if legacy_world is None:
            raise RuntimeError("legacy world proposal component is unavailable")
        self._legacy_world = legacy_world

    @property
    def snapshot_id(self) -> str:
        value = getattr(self._legacy_world, "snapshot_id", None)
        return str(value) if value else "legacy-world:unversioned"

    def readiness(self) -> bool:
        check = getattr(self._legacy_world, "readiness", None)
        return check() is not False if callable(check) else True

    def close(self) -> None:
        close = getattr(self._legacy_world, "close", None) or getattr(
            self._legacy_world, "shutdown", None
        )
        if close is not None:
            close()


class CanonicalWorldProposalPort:
    """Dependency-light world proposal port; owns no state or reasoning authority."""

    production_ready = True
    snapshot_id = "canonical-world-proposal:v1"

    def readiness(self) -> bool:
        return True

    def close(self) -> None:
        return None


def _production_world() -> CanonicalWorldProposalPort:
    return CanonicalWorldProposalPort()


def _production_safety() -> Any:
    from .response_safety import CanonicalResponseSafetyValidator

    return CanonicalResponseSafetyValidator()


@dataclass(frozen=True)
class CompositionSpecification:
    """Factories for every externally selected canonical graph edge.

    Durable authorities are deliberately constructed by ``RuntimeContainer``
    from these factories so no partially initialized object can be published.
    """

    world_proposal_factory: Callable[[], Any] = _production_world
    safety_validator_factory: Callable[[], Any] = _production_safety
    state_authority_factory: Callable[..., Any] = StateAuthoritySet
    transaction_service_factory: Callable[..., Any] = ConstitutionalTransactionService
    episode_store_factory: Callable[..., Any] = EpisodeStore
    epistemic_store_factory: Callable[..., Any] = EpistemicStore
    audit_projector_factory: Callable[..., Any] = CanonicalAudit
    governed_memory_factory: Callable[..., Any] = compose_governed_memory
    alignment_factory: Callable[..., Any] = AlignmentRegistry
    capability_manifest_factory: Callable[..., Any] = CapabilityManifestAuthority
    language_proposal_factory: Callable[..., Any] = DeterministicLanguageInput
    language_output_factory: Callable[..., Any] = DeterministicLanguageOutput
    safety_port_factory: Callable[..., Any] = EnhancedSafetyResponseAdapter
    safety_finalizer_factory: Callable[..., Any] = SafetyResponseFinalizer
    domain_lookup_factory: Callable[..., Any] = PersistentDomainRegistry
    learning_factory: Callable[..., Any] = LearningOwner


def compose_runtime(
    settings: RuntimeSettings, specification: CompositionSpecification | None = None
) -> RuntimeContainer:
    """Construct and publish exactly one typed production owner graph."""
    require_canonical_serving_platform()
    spec = specification or CompositionSpecification()
    if (
        settings.development_stub_mode
        and settings.environment is VulcanEnvironment.production
    ):
        raise StartupFailure(
            StartupErrorCategory.SETTINGS_INVALID,
            "development stub mode is forbidden in production",
        )
    try:
        try:
            world = (
                DevelopmentStubWorld()
                if settings.development_stub_mode
                else spec.world_proposal_factory()
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise StartupFailure(
                StartupErrorCategory.WORLD_MISSING,
                "required world proposal port construction failed",
                exc,
            ) from exc
        try:
            safety = (
                DevelopmentStubSafety()
                if settings.development_stub_mode
                else spec.safety_validator_factory()
            )
        except asyncio.CancelledError as exc:
            close = getattr(world, "close", None) or getattr(world, "shutdown", None)
            if close is not None:
                try:
                    close()
                except BaseException as cleanup_error:
                    exc.add_note(f"cleanup failed for world_proposal: {cleanup_error}")
            raise
        except Exception as exc:
            close = getattr(world, "close", None) or getattr(world, "shutdown", None)
            if close is not None:
                try:
                    close()
                except BaseException as cleanup_error:
                    exc.add_note(f"cleanup failed for world_proposal: {cleanup_error}")
            raise StartupFailure(
                StartupErrorCategory.SAFETY_MISSING,
                "required safety validator construction failed",
                exc,
            ) from exc
        inputs = RuntimeOwnerInputs(
            world_proposal=world,
            safety_validator=safety,
            audit_factory=spec.audit_projector_factory,
            memory_factory=spec.governed_memory_factory,
            alignment_factory=spec.alignment_factory,
            domain_factory=spec.domain_lookup_factory,
            learning_factory=spec.learning_factory,
            safety_port_factory=(
                (lambda unused: DevelopmentUnavailableSafetyPort())
                if settings.development_stub_mode
                else spec.safety_port_factory
            ),
            finalizer_factory=spec.safety_finalizer_factory,
            state_authority_factory=spec.state_authority_factory,
            capability_manifest_factory=spec.capability_manifest_factory,
            language_proposal_factory=spec.language_proposal_factory,
            language_output_factory=spec.language_output_factory,
            episode_store_factory=spec.episode_store_factory,
            epistemic_store_factory=spec.epistemic_store_factory,
            transaction_service_factory=spec.transaction_service_factory,
        )
        return RuntimeContainer.from_owner_inputs(inputs=inputs, settings=settings)
    except StartupFailure:
        raise
    except OSError as exc:
        raise StartupFailure(
            StartupErrorCategory.FILESYSTEM_UNAVAILABLE,
            "runtime durable filesystem unavailable",
            exc,
        ) from exc
    except Exception as exc:
        text = str(exc).lower()
        category = (
            StartupErrorCategory.WORLD_MISSING
            if "world" in text
            else (
                StartupErrorCategory.SAFETY_MISSING
                if "safety" in text
                else StartupErrorCategory.RUNTIME_UNHEALTHY
            )
        )
        raise StartupFailure(category, str(exc), exc) from exc
