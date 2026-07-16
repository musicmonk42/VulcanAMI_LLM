"""Single composition function for the production runtime."""

from __future__ import annotations

from typing import Any

from .container import RuntimeContainer
from .kernel import KernelRequest


async def legacy_chat_adapter(command: KernelRequest, _case: Any) -> dict[str, Any]:
    """Compatibility adapter; transport is kept out of the kernel contract."""
    from vulcan.endpoints.unified_chat import legacy_unified_chat
    return await legacy_unified_chat(command.payload, command.payload.body)


def compose_runtime() -> RuntimeContainer:
    """Construct the only permitted production deployment and World State."""
    from vulcan.config import get_config
    from vulcan.orchestrator.deployment import ProductionDeployment

    deployment = ProductionDeployment(get_config())
    return RuntimeContainer.new(deployment=deployment, executor=legacy_chat_adapter)
