"""
Vulcan API Endpoints Module

This module contains all FastAPI endpoint handlers extracted from main.py.
Each endpoint category is in its own file for better organization.
"""

from vulcan.endpoints.health import router as health_router
from vulcan.endpoints.monitoring import router as monitoring_router

__all__ = [
    "health_router",
    "monitoring_router",
]
