"""
Vulcan Server Module

FastAPI application creation and lifecycle management.
"""

from vulcan.server.app import create_app, lifespan
from vulcan.server import state

__all__ = ["create_app", "lifespan", "state"]
