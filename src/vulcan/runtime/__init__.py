"""The canonical production runtime for VULCAN."""
from __future__ import annotations
__all__=["CognitiveCase","CognitiveCaseStatus","RuntimeContainer","CognitiveKernel","KernelRequest","KernelResult"]
def __getattr__(name: str):
    if name in {"CognitiveCase","CognitiveCaseStatus"}:
        from .case import CognitiveCase, CognitiveCaseStatus
        return {"CognitiveCase":CognitiveCase,"CognitiveCaseStatus":CognitiveCaseStatus}[name]
    if name == "RuntimeContainer":
        from .container import RuntimeContainer
        return RuntimeContainer
    if name in {"CognitiveKernel","KernelRequest","KernelResult"}:
        from .kernel import CognitiveKernel, KernelRequest, KernelResult
        return {"CognitiveKernel":CognitiveKernel,"KernelRequest":KernelRequest,"KernelResult":KernelResult}[name]
    if name in {"app","create_app"}:
        from .app import app, create_app
        return {"app":app,"create_app":create_app}[name]
    raise AttributeError(name)
