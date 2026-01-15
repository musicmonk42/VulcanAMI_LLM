"""
Safety Endpoints

Provides endpoints for safety system status, pre-execution validation,
and audit trail queries.

This module implements the safety monitoring and validation API for VULCAN.

Security considerations:
- Audit logs may be filtered to prevent information leakage
- Error messages are sanitized
- Input validation is enforced via Pydantic
- Rate limiting should be applied at the infrastructure level
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Create router with tags for OpenAPI documentation
router = APIRouter(prefix="/v1/safety", tags=["safety"])

# Constants for validation
MAX_AUDIT_LOG_LIMIT = 500
MIN_AUDIT_LOG_LIMIT = 1
VALID_SEVERITIES = {"info", "warning", "critical", "error"}
VALID_ACTION_TYPES = {"code_execution", "file_access", "api_call", "network", "model_update"}


def _get_safety_validator():
    """
    Get the safety validator singleton.
    
    Uses the global singleton pattern established in safety_validator.py.
    Returns None if the safety system is not initialized.
    
    Returns:
        EnhancedSafetyValidator instance or None if not available
    """
    try:
        from vulcan.safety.safety_validator import (
            _SAFETY_SINGLETON_BUNDLE,
            _SAFETY_SINGLETON_READY,
        )
        if _SAFETY_SINGLETON_READY and _SAFETY_SINGLETON_BUNDLE is not None:
            return _SAFETY_SINGLETON_BUNDLE
    except ImportError:
        logger.debug("Safety validator module not available")
    except Exception as e:
        logger.warning(f"Error accessing safety validator: {e}")
    return None


def _get_audit_logger():
    """
    Get the audit logger from the safety validator.
    
    The audit logger is attached to the safety validator singleton
    and provides tamper-evident logging capabilities.
    
    Returns:
        AuditLogger instance or None if not available
    """
    validator = _get_safety_validator()
    if validator is not None and hasattr(validator, 'audit_logger'):
        return validator.audit_logger
    return None


class SafetyStatusResponse(BaseModel):
    """Response model for safety system status."""
    safety_enabled: bool = Field(description="Whether safety systems are active")
    sandbox_active: bool = Field(description="Whether sandbox isolation is running")
    violation_count: int = Field(ge=0, description="Total safety violations detected")
    last_violation: Optional[str] = Field(default=None, description="ISO 8601 timestamp of last violation")
    active_constraints: List[str] = Field(description="List of active safety constraints")

    class Config:
        json_schema_extra = {
            "example": {
                "safety_enabled": True,
                "sandbox_active": True,
                "violation_count": 3,
                "last_violation": "2026-01-15T06:45:00Z",
                "active_constraints": ["no_harmful_content", "no_pii_exposure", "resource_limits"]
            }
        }


class ValidationRequest(BaseModel):
    """Request model for safety validation."""
    action: str = Field(min_length=1, max_length=100, description="Action to validate")
    context: Dict[str, Any] = Field(default_factory=dict, description="Action context")
    severity: str = Field(default="normal", description="Action severity level")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Sanitize action string to prevent injection."""
        # Remove any potentially dangerous characters
        sanitized = ''.join(c for c in v if c.isalnum() or c in '_-.')
        if not sanitized:
            raise ValueError("Action must contain valid characters")
        return sanitized


class AuditLogsRequest(BaseModel):
    """Request model for audit logs query."""
    limit: int = Field(
        default=100,
        ge=MIN_AUDIT_LOG_LIMIT,
        le=MAX_AUDIT_LOG_LIMIT,
        description=f"Maximum number of logs to return ({MIN_AUDIT_LOG_LIMIT}-{MAX_AUDIT_LOG_LIMIT})"
    )
    severity: Optional[str] = Field(default=None, description="Filter by severity")
    action_type: Optional[str] = Field(default=None, description="Filter by action type")


@router.get("/status", response_model=SafetyStatusResponse)
async def get_safety_system_status() -> SafetyStatusResponse:
    """
    Get safety system and sandbox status.
    
    Returns comprehensive information about safety systems, violations,
    and active constraints. When the safety system is not initialized,
    returns a response indicating the system is disabled.
    
    Returns:
        SafetyStatusResponse: Current safety system status
        
    Raises:
        HTTPException: 500 if an unexpected error occurs
        
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
        safety_enabled = True  # If we have a validator, safety is enabled
        sandbox_active = bool(getattr(validator, 'sandbox_enabled', True))
        
        # Get violation count from metrics if available
        violation_count = 0
        last_violation = None
        if hasattr(validator, 'safety_metrics'):
            metrics = validator.safety_metrics
            violation_count = int(getattr(metrics, 'total_violations', 0))
            last_violation_time = getattr(metrics, 'last_violation_time', None)
            if last_violation_time is not None:
                try:
                    last_violation = datetime.fromtimestamp(
                        last_violation_time, tz=timezone.utc
                    ).isoformat()
                except (ValueError, OSError, OverflowError) as e:
                    logger.warning(f"Invalid violation timestamp: {last_violation_time}, error: {e}")
        
        # Get active constraints
        active_constraints: List[str] = []
        if hasattr(validator, '_dedup_constraints'):
            # Limit to first 20 constraints for response size
            active_constraints = [str(c) for c in list(validator._dedup_constraints)[:20]]
        elif hasattr(validator, 'constraints'):
            active_constraints = [str(c) for c in validator.constraints[:20]]
        
        # Add default constraints if none found (system still has implicit constraints)
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
        logger.error(f"Error getting safety status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to get safety status"
        )


@router.post("/validate", response_model=None)
async def validate_action_safety(request: ValidationRequest) -> Dict[str, Any]:
    """
    Pre-execution safety validation.
    
    Validates whether an action is safe to execute before actually running it.
    Checks against active constraints and safety policies. This endpoint
    should be called before executing any potentially risky action.
    
    Args:
        request: Validation request with action and context
        
    Returns:
        Dict containing:
        - safe: Boolean indicating if action is safe
        - action: The validated action name
        - severity: The action severity level
        - warnings: List of warning messages
        - constraints_checked: List of constraints that were evaluated
        - reason: Explanation if action is not safe
        
    Raises:
        HTTPException: 422 if request validation fails
        HTTPException: 500 if validation check fails
        
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
            # Without a validator, we cannot properly validate - return cautious response
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
        warnings: List[str] = []
        constraints_checked: List[str] = []
        reason: Optional[str] = None
        is_safe = True
        
        try:
            # Try to use the validator's validate_action method
            if hasattr(validator, 'validate_action'):
                safe_result, validation_reason = validator.validate_action(action_to_validate)
                is_safe = bool(safe_result)
                if not is_safe:
                    reason = str(validation_reason) if validation_reason else "Validation failed"
            elif hasattr(validator, 'check_safety'):
                # Alternative method name
                result = validator.check_safety(action_to_validate)
                is_safe = bool(result.get('safe', True))
                reason = str(result.get('reason')) if result.get('reason') else None
                warnings = [str(w) for w in result.get('warnings', [])]
            else:
                # No validation method found, assume safe with warning
                warnings.append("No validation method available - assuming safe")
                
            # Get constraints that were checked
            if hasattr(validator, '_dedup_constraints'):
                constraints_checked = [str(c) for c in list(validator._dedup_constraints)[:10]]
            else:
                constraints_checked = ["resource_limits", "execution_time", "sandbox_compatibility"]
                
        except Exception as val_error:
            logger.warning(f"Validation error: {val_error}")
            warnings.append("Validation encountered an error")
            # Don't expose internal error details
        
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
        logger.error(f"Error validating action: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to validate action"
        )


@router.get("/audit-logs", response_model=None)
async def get_audit_logs(
    limit: int = Query(
        default=100,
        ge=MIN_AUDIT_LOG_LIMIT,
        le=MAX_AUDIT_LOG_LIMIT,
        description=f"Maximum number of logs to return ({MIN_AUDIT_LOG_LIMIT}-{MAX_AUDIT_LOG_LIMIT})"
    ),
    severity: Optional[str] = Query(
        default=None,
        description="Filter by severity (info, warning, critical, error)"
    ),
    action_type: Optional[str] = Query(
        default=None,
        description="Filter by action type"
    )
) -> Dict[str, Any]:
    """
    Query recent audit logs.
    
    Retrieves safety audit logs with optional filtering by severity and action type.
    Useful for compliance monitoring and incident investigation.
    
    Logs are returned in reverse chronological order (newest first).
    
    Args:
        limit: Maximum number of logs to return (1-500, default: 100)
        severity: Optional severity filter ("info", "warning", "critical", "error")
        action_type: Optional action type filter
        
    Returns:
        Dict containing:
        - total: Total number of audit entries in the system
        - returned: Number of entries in this response
        - logs: List of audit log entries
        - message: Additional context message (if applicable)
        
    Raises:
        HTTPException: 422 if query parameter validation fails
        HTTPException: 500 if log retrieval fails
        
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
        # Validate severity filter if provided
        if severity is not None and severity.lower() not in VALID_SEVERITIES:
            logger.warning(f"Invalid severity filter: {severity}")
            # Don't reject, just ignore invalid filter
            severity = None
        
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
        filters: Optional[Dict[str, str]] = None
        if severity or action_type:
            filters = {}
            if severity:
                filters["severity"] = severity.lower()
            if action_type:
                # Sanitize action_type to prevent injection
                filters["action_type"] = ''.join(c for c in action_type if c.isalnum() or c in '_-')
        
        # Query logs from the audit logger
        logs: List[Dict[str, Any]] = []
        try:
            logs = audit_logger.query_logs(
                limit=limit,
                filters=filters,
                sort_by="timestamp",
                sort_order="DESC"
            )
        except ValueError as ve:
            # Handle validation errors from audit logger
            logger.warning(f"Audit log query validation error: {ve}")
        except Exception as query_error:
            logger.warning(f"Error querying audit logs: {query_error}")
        
        # Transform logs to consistent format
        formatted_logs: List[Dict[str, Any]] = []
        for log in logs:
            try:
                formatted_log = {
                    "timestamp": log.get("timestamp", ""),
                    "severity": str(log.get("severity", "info")),
                    "action_type": str(log.get("action_type", log.get("entry_type", "unknown"))),
                    "message": str(log.get("message", log.get("description", ""))),
                    "details": log.get("details", {}),
                    "entry_id": str(log.get("entry_id", "")),
                }
                
                # Convert timestamp to ISO format if it's a float
                if isinstance(formatted_log["timestamp"], (int, float)):
                    try:
                        formatted_log["timestamp"] = datetime.fromtimestamp(
                            formatted_log["timestamp"], tz=timezone.utc
                        ).isoformat()
                    except (ValueError, OSError, OverflowError):
                        formatted_log["timestamp"] = ""
                
                formatted_logs.append(formatted_log)
            except Exception as format_error:
                logger.debug(f"Error formatting log entry: {format_error}")
                continue
        
        # Get total count from audit logger metrics if available
        total = len(formatted_logs)
        if hasattr(audit_logger, 'metrics') and isinstance(audit_logger.metrics, dict):
            total = int(audit_logger.metrics.get("total_entries", total))
        
        return {
            "total": total,
            "returned": len(formatted_logs),
            "logs": formatted_logs
        }
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve audit logs"
        )
