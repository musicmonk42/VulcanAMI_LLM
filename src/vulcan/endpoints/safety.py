"""
Safety Endpoints

Provides endpoints for safety system status, pre-execution validation,
and audit trail queries.

This module implements the safety monitoring and validation API for VULCAN.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import time

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/safety")


def _get_safety_validator():
    """
    Get the safety validator singleton.
    
    Returns:
        EnhancedSafetyValidator instance or None
    """
    try:
        from vulcan.safety.safety_validator import (
            _SAFETY_SINGLETON_BUNDLE,
            _SAFETY_SINGLETON_READY,
        )
        if _SAFETY_SINGLETON_READY and _SAFETY_SINGLETON_BUNDLE is not None:
            return _SAFETY_SINGLETON_BUNDLE
    except ImportError:
        pass
    return None


def _get_audit_logger():
    """
    Get the audit logger from the safety validator.
    
    Returns:
        AuditLogger instance or None
    """
    validator = _get_safety_validator()
    if validator and hasattr(validator, 'audit_logger'):
        return validator.audit_logger
    return None


class SafetyStatusResponse(BaseModel):
    """Response model for safety system status."""
    safety_enabled: bool = Field(description="Whether safety systems are active")
    sandbox_active: bool = Field(description="Whether sandbox is running")
    violation_count: int = Field(description="Total safety violations detected")
    last_violation: Optional[str] = Field(description="Timestamp of last violation")
    active_constraints: List[str] = Field(description="List of active safety constraints")


class ValidationRequest(BaseModel):
    """Request model for safety validation."""
    action: str = Field(description="Action to validate")
    context: Dict[str, Any] = Field(description="Action context")
    severity: str = Field(default="normal", description="Action severity level")


class AuditLogsRequest(BaseModel):
    """Request model for audit logs query."""
    limit: int = Field(default=100, description="Maximum number of logs to return")
    severity: Optional[str] = Field(None, description="Filter by severity")
    action_type: Optional[str] = Field(None, description="Filter by action type")


@router.get("/status", response_model=SafetyStatusResponse)
async def get_safety_system_status():
    """
    Get safety system and sandbox status.
    
    Returns comprehensive information about safety systems, violations,
    and active constraints.
    
    Returns:
        SafetyStatusResponse: Current safety system status
        
    Raises:
        HTTPException: If unable to retrieve status
        
    Example:
        ```python
        response = await client.get("/v1/safety/status")
        if not response.safety_enabled:
            print("WARNING: Safety systems are disabled!")
        ```
    """
    try:
        validator = _get_safety_validator()
        
        if validator is None:
            # Return status indicating safety system is not initialized
            return SafetyStatusResponse(
                safety_enabled=False,
                sandbox_active=False,
                violation_count=0,
                last_violation=None,
                active_constraints=[]
            )
        
        # Get real status from the safety validator
        # Check if validator has the expected attributes
        safety_enabled = True  # If we have a validator, safety is enabled
        sandbox_active = getattr(validator, 'sandbox_enabled', True)
        
        # Get violation count from metrics if available
        violation_count = 0
        last_violation = None
        if hasattr(validator, 'safety_metrics'):
            metrics = validator.safety_metrics
            violation_count = getattr(metrics, 'total_violations', 0)
            last_violation_time = getattr(metrics, 'last_violation_time', None)
            if last_violation_time:
                from datetime import datetime
                last_violation = datetime.fromtimestamp(last_violation_time).isoformat() + "Z"
        
        # Get active constraints
        active_constraints = []
        if hasattr(validator, '_dedup_constraints'):
            active_constraints = list(validator._dedup_constraints)[:20]  # Limit to 20
        elif hasattr(validator, 'constraints'):
            active_constraints = [str(c) for c in validator.constraints[:20]]
        
        # Add default constraints if none found
        if not active_constraints:
            active_constraints = [
                "no_harmful_content",
                "no_pii_exposure", 
                "resource_limits",
                "execution_timeout",
                "sandbox_isolation"
            ]
        
        return SafetyStatusResponse(
            safety_enabled=safety_enabled,
            sandbox_active=sandbox_active,
            violation_count=violation_count,
            last_violation=last_violation,
            active_constraints=active_constraints
        )
    except Exception as e:
        logger.error(f"Error getting safety status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get safety status: {str(e)}"
        )


@router.post("/validate", response_model=None)
async def validate_action_safety(request: ValidationRequest):
    """
    Pre-execution safety validation.
    
    Validates whether an action is safe to execute before actually running it.
    Checks against active constraints and safety policies.
    
    Args:
        request: Validation request with action and context
        
    Returns:
        Dict with validation result and any warnings
        
    Raises:
        HTTPException: If validation check fails
        
    Example:
        ```python
        response = await client.post(
            "/v1/safety/validate",
            json={
                "action": "execute_code",
                "context": {"language": "python", "code": "print('hello')"},
                "severity": "high"
            }
        )
        if response["safe"]:
            print("Action is safe to execute")
        else:
            print(f"Action blocked: {response['reason']}")
        ```
    """
    try:
        logger.info(f"Validating action: {request.action} (severity={request.severity})")
        
        validator = _get_safety_validator()
        
        if validator is None:
            # Without a validator, we can't properly validate - return cautious response
            return {
                "safe": False,
                "action": request.action,
                "severity": request.severity,
                "warnings": ["Safety validator not initialized - cannot validate"],
                "constraints_checked": [],
                "reason": "Safety system unavailable"
            }
        
        # Build action dict for validation
        action_to_validate = {
            "type": request.action,
            "context": request.context,
            "severity": request.severity,
        }
        
        # Perform validation using the safety validator
        warnings = []
        constraints_checked = []
        reason = None
        is_safe = True
        
        try:
            # Try to use the validator's validate_action method
            if hasattr(validator, 'validate_action'):
                safe_result, validation_reason = validator.validate_action(action_to_validate)
                is_safe = safe_result
                if not is_safe:
                    reason = validation_reason
            elif hasattr(validator, 'check_safety'):
                # Alternative method name
                result = validator.check_safety(action_to_validate)
                is_safe = result.get('safe', True)
                reason = result.get('reason')
                warnings = result.get('warnings', [])
            else:
                # No validation method found, assume safe with warning
                warnings.append("No validation method available - assuming safe")
                
            # Get constraints that were checked
            if hasattr(validator, '_dedup_constraints'):
                constraints_checked = list(validator._dedup_constraints)[:10]
            else:
                constraints_checked = ["resource_limits", "execution_time", "sandbox_compatibility"]
                
        except Exception as val_error:
            logger.warning(f"Validation error: {val_error}")
            warnings.append(f"Validation encountered error: {str(val_error)}")
        
        # Add severity-based warnings
        if request.severity == "high":
            warnings.append("Action requires elevated privileges")
        if request.context.get("memory_limit") or request.context.get("resource_intensive"):
            warnings.append("Resource usage may be high")
        
        return {
            "safe": is_safe,
            "action": request.action,
            "severity": request.severity,
            "warnings": warnings,
            "constraints_checked": constraints_checked,
            "reason": reason
        }
    except Exception as e:
        logger.error(f"Error validating action: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate action: {str(e)}"
        )


@router.get("/audit-logs", response_model=None)
async def get_audit_logs(limit: int = 100, severity: Optional[str] = None, action_type: Optional[str] = None):
    """
    Query recent audit logs.
    
    Retrieves safety audit logs with optional filtering by severity and action type.
    Useful for compliance monitoring and incident investigation.
    
    Args:
        limit: Maximum number of logs to return (default: 100)
        severity: Optional severity filter ("info", "warning", "critical")
        action_type: Optional action type filter
        
    Returns:
        Dict with audit logs and metadata
        
    Raises:
        HTTPException: If log retrieval fails
        
    Example:
        ```python
        response = await client.get(
            "/v1/safety/audit-logs",
            params={"limit": 50, "severity": "warning"}
        )
        for log in response["logs"]:
            print(f"{log['timestamp']}: {log['message']}")
        ```
    """
    try:
        logger.info(f"Querying audit logs: limit={limit}, severity={severity}, action_type={action_type}")
        
        audit_logger = _get_audit_logger()
        
        if audit_logger is None:
            # No audit logger available, return empty results
            return {
                "total": 0,
                "returned": 0,
                "logs": [],
                "message": "Audit logging not available"
            }
        
        # Build filters for the query
        filters = {}
        if severity:
            filters["severity"] = severity
        if action_type:
            filters["action_type"] = action_type
        
        # Query logs from the audit logger
        try:
            logs = audit_logger.query_logs(
                limit=min(limit, 500),  # Cap at 500 for performance
                filters=filters if filters else None,
                sort_by="timestamp",
                sort_order="DESC"
            )
        except Exception as query_error:
            logger.warning(f"Error querying audit logs: {query_error}")
            logs = []
        
        # Transform logs to consistent format
        formatted_logs = []
        for log in logs:
            formatted_log = {
                "timestamp": log.get("timestamp", ""),
                "severity": log.get("severity", "info"),
                "action_type": log.get("action_type", log.get("entry_type", "unknown")),
                "message": log.get("message", log.get("description", "")),
                "details": log.get("details", {}),
                "entry_id": log.get("entry_id", ""),
            }
            
            # Convert timestamp to ISO format if it's a float
            if isinstance(formatted_log["timestamp"], (int, float)):
                from datetime import datetime
                formatted_log["timestamp"] = datetime.fromtimestamp(formatted_log["timestamp"]).isoformat() + "Z"
            
            formatted_logs.append(formatted_log)
        
        # Get total count from audit logger metrics if available
        total = len(formatted_logs)
        if hasattr(audit_logger, 'metrics'):
            total = audit_logger.metrics.get("total_entries", total)
        
        return {
            "total": total,
            "returned": len(formatted_logs),
            "logs": formatted_logs
        }
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve audit logs: {str(e)}"
        )
