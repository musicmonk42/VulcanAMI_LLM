"""
Vulcan API Endpoints Module

This module contains all FastAPI endpoint handlers extracted from main.py.
Each endpoint category is in its own file for better organization.
"""

from vulcan.endpoints.agents import router as agents_router
from vulcan.endpoints.chat import router as chat_router
from vulcan.endpoints.config import router as config_router
from vulcan.endpoints.distillation import router as distillation_router
from vulcan.endpoints.execution import router as execution_router
from vulcan.endpoints.feedback import router as feedback_router
from vulcan.endpoints.health import router as health_router
from vulcan.endpoints.memory import router as memory_router
from vulcan.endpoints.monitoring import router as monitoring_router
from vulcan.endpoints.planning import router as planning_router
from vulcan.endpoints.reasoning import router as reasoning_router
from vulcan.endpoints.safety import router as safety_router
from vulcan.endpoints.self_improvement import router as self_improvement_router
from vulcan.endpoints.status import router as status_router
from vulcan.endpoints.unified_chat import router as unified_chat_router
from vulcan.endpoints.world_model import router as world_model_router

__all__ = [
    "agents_router",
    "chat_router",
    "config_router",
    "distillation_router",
    "execution_router",
    "feedback_router",
    "health_router",
    "memory_router",
    "monitoring_router",
    "planning_router",
    "reasoning_router",
    "safety_router",
    "self_improvement_router",
    "status_router",
    "unified_chat_router",
    "world_model_router",
]
