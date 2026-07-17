"""Single composition function for the production runtime."""
from __future__ import annotations
from .container import RuntimeContainer

def compose_runtime() -> RuntimeContainer:
    """Construct the deployment graph; legacy endpoint orchestration is unreachable."""
    from vulcan.config import get_config
    from vulcan.orchestrator.deployment import ProductionDeployment
    return RuntimeContainer.new(deployment=ProductionDeployment(get_config()))
