"""
Vulcan Server Module

FastAPI application creation and lifecycle management.
"""

from vulcan.server.app import create_app, lifespan

__all__ = ["create_app", "lifespan"]
