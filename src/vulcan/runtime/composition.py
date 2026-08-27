"""Single composition function for the production runtime."""
from __future__ import annotations

from importlib import import_module, util
from types import SimpleNamespace

from .container import RuntimeContainer
from .constitutional_kernel import ConstitutionalCognitiveKernel
from .errors import StartupErrorCategory, StartupFailure
from .settings import RuntimeSettings, VulcanEnvironment


class DevelopmentStubDeployment:
    """Deliberately named non-production stub; cognitive routes must remain unavailable."""

    production_ready = False

    def __init__(self) -> None:
        self.collective = SimpleNamespace(
            deps=SimpleNamespace(
                world_model=DevelopmentStubWorld(),
                safety_validator=DevelopmentStubSafety(),
            )
        )

    def readiness(self) -> bool:
        return False

    def close(self) -> None:
        return None


class DevelopmentStubWorld:
    production_ready = False
    snapshot_id = "development-stub-world"

    def readiness(self) -> bool:
        return False


class DevelopmentStubSafety:
    production_ready = False

    def readiness(self) -> bool:
        return False


def _startup_failure(
    category: StartupErrorCategory,
    message: str,
    exc: BaseException | None = None,
) -> StartupFailure:
    return StartupFailure(category, message, exc)


def _module_available(name: str) -> bool:
    try:
        return util.find_spec(name) is not None
    except ValueError:
        return True


def _bind_constitutional_admission(container: RuntimeContainer) -> RuntimeContainer:
    """Wrap the compatibility kernel in the composed snapshot authority."""
    container.kernel = ConstitutionalCognitiveKernel.from_kernel(
        container.kernel,
        snapshot_admitter=container.admit_snapshot_bundle,
    )
    return container


def compose_runtime(settings: RuntimeSettings) -> RuntimeContainer:
    """Construct the deployment graph from the already-parsed settings authority."""
    if settings.development_stub_mode:
        if settings.environment is VulcanEnvironment.production:
            raise _startup_failure(
                StartupErrorCategory.SETTINGS_INVALID,
                "development stub mode is forbidden in production",
            )
        return _bind_constitutional_admission(
            RuntimeContainer.new(
                deployment=DevelopmentStubDeployment(),
                settings=settings,
            )
        )
    if not _module_available("vulcan.config") or not _module_available(
        "vulcan.orchestrator.deployment"
    ):
        raise _startup_failure(
            StartupErrorCategory.DEPLOYMENT_IMPORT_FAILED,
            "production deployment dependency import failed",
        )
    try:
        config_module = import_module("vulcan.config")
        deployment_module = import_module("vulcan.orchestrator.deployment")
        deployment = deployment_module.ProductionDeployment(config_module.get_config())
    except BaseException as exc:
        raise _startup_failure(
            StartupErrorCategory.DEPLOYMENT_CONSTRUCTION_FAILED,
            "production deployment construction failed",
            exc,
        ) from exc
    try:
        return _bind_constitutional_admission(
            RuntimeContainer.new(deployment=deployment, settings=settings)
        )
    except StartupFailure:
        raise
    except OSError as exc:
        raise _startup_failure(
            StartupErrorCategory.FILESYSTEM_UNAVAILABLE,
            "runtime durable filesystem unavailable",
            exc,
        ) from exc
    except RuntimeError as exc:
        text = str(exc).lower()
        category = StartupErrorCategory.RUNTIME_UNHEALTHY
        if "world" in text:
            category = StartupErrorCategory.WORLD_MISSING
        elif "safety" in text:
            category = StartupErrorCategory.SAFETY_MISSING
        elif "kernel" in text or "reason" in text:
            category = StartupErrorCategory.REASONER_MISSING
        raise _startup_failure(category, str(exc), exc) from exc
