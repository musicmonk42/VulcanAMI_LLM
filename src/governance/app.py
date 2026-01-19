"""
Graphix IR Registry Flask App Wrapper
=====================================

This module provides a Flask-based HTTP API wrapper around the RegistryAPI
for integration with the unified platform as a WSGI sub-application.

The Registry service provides:
- Proposal submission and management
- Consensus voting workflows  
- Validation and deployment of grammar versions
- Audit logging with cryptographic integrity
"""

import hmac
import json
import logging
import os
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .registry_api import RegistryAPI, InMemoryBackend, SimpleKMS, DatabaseBackendAdapter

# Configure logging - use module-level logger
logger = logging.getLogger(__name__)


def _check_security_configuration():
    """
    Check security configuration at startup.
    
    Industry standard practice to fail fast if security requirements aren't met.
    """
    issues = []
    
    # Check if running in production mode
    is_production = os.environ.get("REGISTRY_PRODUCTION_MODE", "").lower() == "true"
    
    if is_production:
        # In production, require API key to be set
        if not os.environ.get("REGISTRY_API_KEY"):
            issues.append("REGISTRY_API_KEY environment variable must be set in production mode")
        
        # Warn about HTTPS
        if not os.environ.get("FORCE_HTTPS", "").lower() == "true":
            logger.warning("FORCE_HTTPS is not enabled - API should be served over HTTPS in production")
        
        # Check database path is set
        if not os.environ.get("REGISTRY_DB_PATH"):
            logger.warning("REGISTRY_DB_PATH not set - using default 'registry.db'")
    
    if issues:
        error_msg = "Security configuration errors:\n" + "\n".join(f"  - {issue}" for issue in issues)
        logger.error(error_msg)
        if is_production:
            raise RuntimeError(error_msg)
        else:
            logger.warning("Running in development mode with configuration issues")
    else:
        logger.info(f"Security configuration check passed (production={is_production})")


# Check security configuration at module load
_check_security_configuration()

# Create Flask app
app = Flask(__name__)

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """
    Add security headers to all responses.
    
    Industry standard security headers:
    - X-Content-Type-Options: Prevent MIME type sniffing
    - X-Frame-Options: Prevent clickjacking
    - X-XSS-Protection: Enable browser XSS protection
    - Strict-Transport-Security: Force HTTPS
    - Content-Security-Policy: Restrict resource loading
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Only add HSTS if running on HTTPS
    if request.is_secure:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Restrictive CSP for API endpoints
    response.headers['Content-Security-Policy'] = "default-src 'none'; frame-ancestors 'none'"
    
    # Remove server identification header
    response.headers.pop('Server', None)
    
    return response

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per minute"],
    storage_uri="memory://",
)

# Initialize the Registry API
_registry: Optional[RegistryAPI] = None
_db_manager = None


def get_db_manager():
    """Get or create the DatabaseManager singleton."""
    global _db_manager
    if _db_manager is None:
        db_path = os.environ.get("REGISTRY_DB_PATH", "registry.db")
        # Import DatabaseManager from registry_api_server
        try:
            from .registry_api_server import DatabaseManager
            _db_manager = DatabaseManager(db_path)
            logger.info(f"DatabaseManager initialized with path: {db_path}")
        except ImportError as e:
            logger.warning(f"Could not import DatabaseManager: {e}. Using InMemoryBackend.")
            _db_manager = None
    return _db_manager


def get_registry() -> RegistryAPI:
    """Get or create the Registry API singleton with persistent storage."""
    global _registry
    if _registry is None:
        db_manager = get_db_manager()
        if db_manager:
            # Use DatabaseManager for persistent storage
            _registry = RegistryAPI(backend=DatabaseBackendAdapter(db_manager), kms=SimpleKMS())
            logger.info("Registry API initialized with persistent storage")
        else:
            # Fallback to InMemoryBackend for development/testing
            _registry = RegistryAPI(backend=InMemoryBackend(), kms=SimpleKMS())
            logger.warning("Registry API initialized with InMemoryBackend (data will not persist)")
    return _registry


def require_auth(f):
    """
    Authentication decorator for Flask endpoints.
    
    Validates requests using either:
    1. API key in X-API-Key header
    2. Agent signature in Authorization header with replay attack prevention
    
    Industry standard implementation with:
    - Constant-time comparison for API keys
    - RSA-PSS signature verification
    - Replay attack prevention via timestamp validation
    - Security audit logging
    - No information leakage in error messages
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        authenticated = False
        auth_method = None
        agent_id = None
        
        # Check for API key authentication
        api_key = request.headers.get('X-API-Key')
        if api_key:
            expected_key = os.environ.get('REGISTRY_API_KEY')
            if expected_key:
                try:
                    # Use constant-time comparison to prevent timing attacks
                    if hmac.compare_digest(api_key, expected_key):
                        authenticated = True
                        auth_method = "api_key"
                except Exception as e:
                    logger.error(f"API key comparison error: {e}")
        
        # Check for agent signature authentication
        if not authenticated:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Signature '):
                try:
                    # Parse: "Signature agent_id:timestamp:signature_hex"
                    # Format includes timestamp for replay attack prevention
                    parts = auth_header[10:].split(':', 2)
                    if len(parts) == 3:
                        agent_id, timestamp_str, signature_hex = parts
                        
                        # Validate agent_id format (alphanumeric, dash, underscore only)
                        if not all(c.isalnum() or c in '-_' for c in agent_id):
                            logger.warning(f"Invalid agent_id format from {request.remote_addr}")
                            return jsonify({"error": "Unauthorized"}), 401
                        
                        # Validate timestamp is recent (within 5 minutes)
                        try:
                            timestamp = int(timestamp_str)
                            current_time = int(datetime.utcnow().timestamp())
                            time_diff = abs(current_time - timestamp)
                            if time_diff > 300:  # 5 minutes
                                logger.warning(f"Stale timestamp from agent {agent_id}: {time_diff}s old")
                                return jsonify({"error": "Unauthorized"}), 401
                        except ValueError:
                            logger.warning(f"Invalid timestamp format from {request.remote_addr}")
                            return jsonify({"error": "Unauthorized"}), 401
                        
                        # Validate signature is hex format
                        if not all(c in '0123456789abcdefABCDEF' for c in signature_hex):
                            logger.warning(f"Invalid signature format from agent {agent_id}")
                            return jsonify({"error": "Unauthorized"}), 401
                        
                        registry = get_registry()
                        # Create message from request data including timestamp
                        request_body = request.get_data(as_text=True)
                        message = f"{request.method}:{request.path}:{timestamp_str}:{request_body}"
                        
                        if registry.agent_registry.verify_agent_signature(agent_id, message.encode(), signature_hex):
                            authenticated = True
                            auth_method = "agent_signature"
                            request.authenticated_agent_id = agent_id
                        else:
                            logger.warning(f"Signature verification failed for agent {agent_id}")
                    
                    # Also support legacy format without timestamp for backwards compatibility
                    # This should be removed in production after migration period
                    elif len(parts) == 2 and os.environ.get('ALLOW_LEGACY_AUTH', '').lower() == 'true':
                        agent_id, signature_hex = parts
                        if not all(c.isalnum() or c in '-_' for c in agent_id):
                            logger.warning(f"Invalid agent_id format from {request.remote_addr}")
                            return jsonify({"error": "Unauthorized"}), 401
                        
                        registry = get_registry()
                        message = f"{request.method}:{request.path}:{request.get_data(as_text=True)}"
                        
                        if registry.agent_registry.verify_agent_signature(agent_id, message.encode(), signature_hex):
                            authenticated = True
                            auth_method = "agent_signature_legacy"
                            request.authenticated_agent_id = agent_id
                            logger.warning(f"Agent {agent_id} used legacy auth - should upgrade to timestamped format")
                        
                except Exception as e:
                    # Don't leak internal error details in response
                    logger.error(f"Signature authentication error from {request.remote_addr}: {e}", exc_info=True)
        
        if authenticated:
            # Log successful authentication for security audit
            logger.info(f"Authenticated request: method={auth_method}, agent={agent_id or 'api_key'}, "
                       f"path={request.path}, remote_addr={request.remote_addr}")
            return f(*args, **kwargs)
        
        # Log failed authentication attempt for security monitoring
        logger.warning(f"Unauthorized access attempt: path={request.path}, "
                      f"remote_addr={request.remote_addr}, headers={dict(request.headers)}")
        
        # Return generic error message to prevent information leakage
        return jsonify({"error": "Unauthorized"}), 401
    
    return decorated


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        registry = get_registry()
        version = registry.get_active_grammar_version()
        return jsonify({
            "status": "healthy",
            "service": "registry",
            "grammar_version": version,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "service": "registry",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 500


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with service info."""
    return jsonify({
        "service": "Graphix IR Registry",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "proposals": "/proposals",
            "grammar_version": "/grammar/version",
            "audit_log": "/audit/log"
        }
    })


@app.route("/proposals", methods=["GET"])
def list_proposals():
    """List all proposals with optional filtering."""
    try:
        registry = get_registry()
        status = request.args.get("status")
        proposed_by = request.args.get("proposed_by")
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", 0, type=int)
        
        proposals = registry.query_proposals(
            status=status,
            proposed_by=proposed_by,
            limit=limit,
            offset=offset
        )
        return jsonify({
            "proposals": proposals,
            "count": len(proposals)
        })
    except Exception as e:
        logger.error(f"Error listing proposals: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/proposals", methods=["POST"])
@limiter.limit("30 per minute")
@require_auth
def submit_proposal():
    """Submit a new proposal."""
    try:
        registry = get_registry()
        proposal_data = request.get_json()
        
        if not proposal_data:
            return jsonify({"error": "Request body is required"}), 400
        
        proposal_id = registry.submit_proposal(proposal_data)
        return jsonify({
            "proposal_id": proposal_id,
            "status": "pending",
            "message": "Proposal submitted successfully"
        }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting proposal: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/proposals/<proposal_id>", methods=["GET"])
def get_proposal(proposal_id: str):
    """
    Get a specific proposal by ID.
    
    Industry standard implementation with:
    - Input validation for proposal_id
    - Proper error handling
    - No information leakage
    """
    # Validate proposal_id format
    if not proposal_id or not isinstance(proposal_id, str):
        return jsonify({"error": "Invalid proposal ID"}), 400
    
    # Sanitize proposal_id (alphanumeric, dash, underscore only)
    if not all(c.isalnum() or c in '-_' for c in proposal_id):
        return jsonify({"error": "Invalid proposal ID format"}), 400
    
    if len(proposal_id) > 128:
        return jsonify({"error": "Proposal ID too long"}), 400
    
    try:
        registry = get_registry()
        proposal = registry.get_proposal(proposal_id)
        
        if not proposal:
            return jsonify({"error": "Proposal not found"}), 404
        
        return jsonify(proposal)
    except Exception as e:
        logger.error(f"Error getting proposal {proposal_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/proposals/<proposal_id>/vote", methods=["POST"])
@limiter.limit("30 per minute")
@require_auth
def vote_on_proposal(proposal_id: str):
    """
    Record votes for a proposal.
    
    Industry standard implementation with:
    - Authentication required
    - Rate limiting (30/min)
    - Input validation
    - Comprehensive error handling
    """
    # Validate proposal_id format
    if not proposal_id or not isinstance(proposal_id, str):
        return jsonify({"error": "Invalid proposal ID"}), 400
    
    if not all(c.isalnum() or c in '-_' for c in proposal_id):
        return jsonify({"error": "Invalid proposal ID format"}), 400
    
    if len(proposal_id) > 128:
        return jsonify({"error": "Proposal ID too long"}), 400
    
    try:
        registry = get_registry()
        vote_data = request.get_json()
        
        if not vote_data:
            return jsonify({"error": "Request body is required"}), 400
        
        # Validate vote_data structure
        if not isinstance(vote_data, dict):
            return jsonify({"error": "Invalid request body format"}), 400
        
        # Ensure proposal_id is in the vote data
        vote_data["proposal_id"] = proposal_id
        
        consensus_reached = registry.record_vote(vote_data)
        
        # Log the vote for audit trail
        logger.info(f"Vote recorded on proposal {proposal_id} by {getattr(request, 'authenticated_agent_id', 'unknown')}")
        
        return jsonify({
            "proposal_id": proposal_id,
            "consensus_reached": consensus_reached,
            "message": "Votes recorded successfully"
        })
    except ValueError as e:
        logger.warning(f"Invalid vote data for proposal {proposal_id}: {e}")
        return jsonify({"error": "Invalid vote data"}), 400
    except Exception as e:
        logger.error(f"Error recording votes on proposal {proposal_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/proposals/<proposal_id>/validate", methods=["POST"])
@limiter.limit("30 per minute")
@require_auth
def validate_proposal(proposal_id: str):
    """
    Record validation results for a proposal.
    
    Industry standard implementation with:
    - Authentication required
    - Rate limiting (30/min)
    - Input validation
    - Comprehensive error handling
    """
    # Validate proposal_id format
    if not proposal_id or not isinstance(proposal_id, str):
        return jsonify({"error": "Invalid proposal ID"}), 400
    
    if not all(c.isalnum() or c in '-_' for c in proposal_id):
        return jsonify({"error": "Invalid proposal ID format"}), 400
    
    if len(proposal_id) > 128:
        return jsonify({"error": "Proposal ID too long"}), 400
    
    try:
        registry = get_registry()
        validation_data = request.get_json()
        
        if not validation_data:
            return jsonify({"error": "Request body is required"}), 400
        
        # Validate validation_data structure
        if not isinstance(validation_data, dict):
            return jsonify({"error": "Invalid request body format"}), 400
        
        # Ensure target is set
        validation_data["target"] = proposal_id
        
        validation_result = registry.record_validation(validation_data)
        
        # Log validation for audit trail
        logger.info(f"Validation recorded on proposal {proposal_id} by {getattr(request, 'authenticated_agent_id', 'unknown')}")
        
        return jsonify({
            "proposal_id": proposal_id,
            "validation_passed": validation_result,
            "message": "Validation recorded"
        })
    except ValueError as e:
        logger.warning(f"Invalid validation data for proposal {proposal_id}: {e}")
        return jsonify({"error": "Invalid validation data"}), 400
    except Exception as e:
        logger.error(f"Error recording validation on proposal {proposal_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/proposals/<proposal_id>/deploy", methods=["POST"])
@limiter.limit("5 per minute")
@require_auth
def deploy_proposal(proposal_id: str):
    """
    Deploy a grammar version from a proposal.
    
    Industry standard implementation with:
    - Authentication required
    - Strict rate limiting (5/min)
    - Input validation
    - Comprehensive audit logging
    - No information leakage
    """
    # Validate proposal_id format
    if not proposal_id or not isinstance(proposal_id, str):
        return jsonify({"error": "Invalid proposal ID"}), 400
    
    if not all(c.isalnum() or c in '-_' for c in proposal_id):
        return jsonify({"error": "Invalid proposal ID format"}), 400
    
    if len(proposal_id) > 128:
        return jsonify({"error": "Proposal ID too long"}), 400
    
    try:
        registry = get_registry()
        deploy_data = request.get_json()
        
        if not deploy_data or "new_version" not in deploy_data:
            return jsonify({"error": "new_version is required"}), 400
        
        new_version = deploy_data["new_version"]
        
        # Validate version format (semantic versioning)
        if not isinstance(new_version, str):
            return jsonify({"error": "Invalid version format"}), 400
        
        # Basic semver validation: X.Y.Z where X, Y, Z are numbers
        version_parts = new_version.split('.')
        if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
            return jsonify({"error": "Version must be in semantic versioning format (X.Y.Z)"}), 400
        
        success = registry.deploy_grammar_version(proposal_id, new_version)
        
        if success:
            # Log deployment for audit trail
            logger.info(f"Grammar version {new_version} deployed from proposal {proposal_id} "
                       f"by {getattr(request, 'authenticated_agent_id', 'unknown')}")
            
            return jsonify({
                "proposal_id": proposal_id,
                "new_version": new_version,
                "deployed": True,
                "message": f"Grammar version {new_version} deployed successfully"
            })
        else:
            logger.warning(f"Deployment failed for proposal {proposal_id}: not in approved/validated state")
            return jsonify({"error": "Deployment failed - proposal not in approved/validated state"}), 400
            
    except ValueError as e:
        logger.warning(f"Invalid deployment data for proposal {proposal_id}: {e}")
        return jsonify({"error": "Invalid deployment data"}), 400
    except Exception as e:
        logger.error(f"Error deploying grammar version for proposal {proposal_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/grammar/version", methods=["GET"])
def get_grammar_version():
    """Get the currently active grammar version."""
    try:
        registry = get_registry()
        version = registry.get_active_grammar_version()
        return jsonify({
            "active_version": version
        })
    except Exception as e:
        logger.error(f"Error getting grammar version: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/audit/log", methods=["GET"])
def get_audit_log():
    """Get the audit log."""
    try:
        registry = get_registry()
        audit_log = registry.get_full_audit_log()
        return jsonify({
            "audit_log": audit_log,
            "count": len(audit_log)
        })
    except Exception as e:
        logger.error(f"Error getting audit log: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/audit/verify", methods=["GET"])
def verify_audit_log():
    """Verify the integrity of the audit log."""
    try:
        registry = get_registry()
        is_valid = registry.verify_audit_log_integrity()
        return jsonify({
            "integrity_valid": is_valid,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Error verifying audit log: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/agents", methods=["POST"])
@require_auth
def register_agent():
    """Register a new agent."""
    try:
        registry = get_registry()
        agent_data = request.get_json()
        
        if not agent_data:
            return jsonify({"error": "Request body is required"}), 400
        
        agent_id = agent_data.get("agent_id")
        public_key_pem = agent_data.get("public_key_pem")
        trust_level = agent_data.get("trust_level", 0.5)
        
        if not agent_id or not public_key_pem:
            return jsonify({"error": "agent_id and public_key_pem are required"}), 400
        
        registry.agent_registry.register_agent(agent_id, public_key_pem, trust_level)
        return jsonify({
            "agent_id": agent_id,
            "trust_level": trust_level,
            "message": "Agent registered successfully"
        }), 201
    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/agents/<agent_id>", methods=["GET"])
def get_agent(agent_id: str):
    """Get information about a registered agent."""
    try:
        registry = get_registry()
        agent_info = registry.agent_registry.get_agent_info(agent_id)
        
        if not agent_info:
            return jsonify({"error": f"Agent '{agent_id}' not found"}), 404
        
        return jsonify(agent_info)
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get registry metrics."""
    try:
        registry = get_registry()
        return jsonify({
            "metrics": registry.registry.get("metrics", {}),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({"error": str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # SECURITY WARNING: This is for development/testing only
    # Production deployments MUST use a production WSGI server (gunicorn, uwsgi)
    # with proper process management, worker count, and timeout settings
    
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    if debug_mode:
        logger.warning(
            "Running Flask app with debug=True. "
            "This exposes the Werkzeug debugger and allows arbitrary code execution. "
            "NEVER use debug=True in production!"
        )
    
    # Use localhost by default for security; override with FLASK_HOST if needed
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    if host == "0.0.0.0":
        logger.warning(
            "Binding to 0.0.0.0 exposes the service to all network interfaces. "
            "Ensure proper firewall rules are in place."
        )
    
    port = int(os.environ.get("FLASK_PORT", "5000"))
    
    logger.info(f"Starting Flask development server on {host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)
