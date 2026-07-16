"""The canonical production runtime for VULCAN.

Only this package is allowed to assemble the production cognitive graph.
"""

from .case import CognitiveCase, CognitiveCaseStatus
from .container import RuntimeContainer
from .kernel import CognitiveKernel, KernelRequest, KernelResult

__all__ = [
    "CognitiveCase", "CognitiveCaseStatus",
    "RuntimeContainer", "CognitiveKernel", "KernelRequest", "KernelResult",
]


def __getattr__(name: str):
    """Keep domain runtime imports independent of the ASGI optional dependency."""
    if name in {"app", "create_app"}:
        from .app import app, create_app
        return {"app": app, "create_app": create_app}[name]
    raise AttributeError(name)
