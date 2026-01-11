"""
Vulcan API Endpoints Module

This module contains all FastAPI endpoint handlers extracted from main.py.
Each endpoint category is in its own file for better organization.
"""

from vulcan.endpoints.config import router as config_router
from vulcan.endpoints.execution import router as execution_router
from vulcan.endpoints.health import router as health_router
from vulcan.endpoints.memory import router as memory_router
from vulcan.endpoints.monitoring import router as monitoring_router
from vulcan.endpoints.planning import router as planning_router
from vulcan.endpoints.status import router as status_router

__all__ = [
    "config_router",
    "execution_router",
    "health_router",
    "memory_router",
    "monitoring_router",
    "planning_router",
    "status_router",
]
