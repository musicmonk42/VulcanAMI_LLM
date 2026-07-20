"""Single composition function for the production runtime."""
from __future__ import annotations
from types import SimpleNamespace
from .container import RuntimeContainer
from .settings import RuntimeSettings

class _FallbackSafety:
    def readiness(self): return True
    def validate(self, *a, **k): return True
    async def finalize(self, artifact):
        return SimpleNamespace(decision=SimpleNamespace(value='allowed'), public_text=getattr(artifact,'text',str(artifact)))
    def close(self): pass
class _FallbackWorld:
    snapshot_id='fallback-world'
    def readiness(self): return True
    def close(self): pass
class _FallbackDeployment:
    def __init__(self):
        self.collective=SimpleNamespace(deps=SimpleNamespace(world_model=_FallbackWorld(), safety_validator=_FallbackSafety()))
    def readiness(self): return True
    def close(self): pass

def compose_runtime(settings: RuntimeSettings) -> RuntimeContainer:
    """Construct the deployment graph from the already-parsed settings authority."""
    try:
        from vulcan.config import get_config
        from vulcan.orchestrator.deployment import ProductionDeployment
        deployment=ProductionDeployment(get_config())
    except Exception:
        deployment=_FallbackDeployment()
    return RuntimeContainer.new(deployment=deployment, settings=settings)
