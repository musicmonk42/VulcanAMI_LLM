"""
safety_status_endpoint.py - FastAPI endpoint for safety system status
Provides monitoring and demo capabilities for safety initialization state
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter()


@router.get("/status")
async def get_safety_status() -> Dict[str, Any]:
    """
    Get comprehensive safety system status.

    Returns:
        JSON with initialization state, counts, validator ID, and domain info
    """
    try:
        # Import here to avoid circular dependencies
        from .safety_validator import (
            _SAFETY_SINGLETON_READY,
            _SAFETY_SINGLETON_BUNDLE,
            initialize_all_safety_components,
        )
        from .domain_validators import _DOMAIN_VALIDATORS_INIT_DONE, validator_registry

        # Get singleton validator
        validator = _SAFETY_SINGLETON_BUNDLE

        # Build status response
        status = {
            "initialized": _SAFETY_SINGLETON_READY and validator is not None,
            "validator_id": hex(id(validator)) if validator else None,
            "domain_validators_initialized": _DOMAIN_VALIDATORS_INIT_DONE,
        }

        # Add counts if validator is available
        if validator:
            status["constraints_count"] = len(validator._dedup_constraints)
            status["properties_count"] = len(validator._dedup_properties)
            status["invariants_count"] = len(validator._dedup_invariants)

            # Add constraint names
            status["constraints"] = list(validator._dedup_constraints)
            status["properties"] = list(validator._dedup_properties)
            status["invariants"] = list(validator._dedup_invariants)
        else:
            status["constraints_count"] = 0
            status["properties_count"] = 0
            status["invariants_count"] = 0
            status["constraints"] = []
            status["properties"] = []
            status["invariants"] = []

        # Add domain validator info
        try:
            domains_registered = validator_registry.list_domains()
            status["domains_registered"] = domains_registered
            status["domains_count"] = len(domains_registered)
        except Exception as e:
            logger.warning(f"Could not get domain validators: {e}")
            status["domains_registered"] = []
            status["domains_count"] = 0

        return status

    except Exception as e:
        logger.error(f"Error getting safety status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get safety status: {str(e)}"
        )


@router.post("/initialize")
async def initialize_safety() -> Dict[str, Any]:
    """
    Manually trigger safety system initialization.

    Returns:
        JSON with initialization result
    """
    try:
        from .safety_validator import initialize_all_safety_components

        # Initialize with default config
        validator = initialize_all_safety_components(config=None, reuse_existing=False)

        return {
            "status": "initialized",
            "validator_id": hex(id(validator)),
            "message": "Safety system initialized successfully",
        }

    except Exception as e:
        logger.error(f"Error initializing safety: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize safety: {str(e)}"
        )
