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

# Create Flask app
app = Flask(__name__)

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
    2. Agent signature in Authorization header
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for API key authentication
        api_key = request.headers.get('X-API-Key')
        if api_key:
            expected_key = os.environ.get('REGISTRY_API_KEY')
            if expected_key and hmac.compare_digest(api_key, expected_key):
                return f(*args, **kwargs)
        
        # Check for agent signature authentication
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Signature '):
            try:
                # Parse: "Signature agent_id:signature_hex"
                parts = auth_header[10:].split(':')
                if len(parts) == 2:
                    agent_id, signature_hex = parts
                    registry = get_registry()
                    # Create message from request data
                    message = f"{request.method}:{request.path}:{request.get_data(as_text=True)}"
                    if registry.agent_registry.verify_agent_signature(agent_id, message.encode(), signature_hex):
                        request.authenticated_agent_id = agent_id
                        return f(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Signature authentication failed: {e}")
        
        return jsonify({"error": "Unauthorized", "message": "Valid API key or agent signature required"}), 401
    
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
    """Get a specific proposal by ID."""
    try:
        registry = get_registry()
        proposal = registry.get_proposal(proposal_id)
        
        if not proposal:
            return jsonify({"error": f"Proposal '{proposal_id}' not found"}), 404
        
        return jsonify(proposal)
    except Exception as e:
        logger.error(f"Error getting proposal: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/proposals/<proposal_id>/vote", methods=["POST"])
@limiter.limit("30 per minute")
@require_auth
def vote_on_proposal(proposal_id: str):
    """Record votes for a proposal."""
    try:
        registry = get_registry()
        vote_data = request.get_json()
        
        if not vote_data:
            return jsonify({"error": "Request body is required"}), 400
        
        # Ensure proposal_id is in the vote data
        vote_data["proposal_id"] = proposal_id
        
        consensus_reached = registry.record_vote(vote_data)
        return jsonify({
            "proposal_id": proposal_id,
            "consensus_reached": consensus_reached,
            "message": "Votes recorded successfully"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error recording votes: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/proposals/<proposal_id>/validate", methods=["POST"])
def validate_proposal(proposal_id: str):
    """Record validation results for a proposal."""
    try:
        registry = get_registry()
        validation_data = request.get_json()
        
        if not validation_data:
            return jsonify({"error": "Request body is required"}), 400
        
        # Ensure target is set
        validation_data["target"] = proposal_id
        
        validation_result = registry.record_validation(validation_data)
        return jsonify({
            "proposal_id": proposal_id,
            "validation_passed": validation_result,
            "message": "Validation recorded"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error recording validation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/proposals/<proposal_id>/deploy", methods=["POST"])
@limiter.limit("5 per minute")
@require_auth
def deploy_proposal(proposal_id: str):
    """Deploy a grammar version from a proposal."""
    try:
        registry = get_registry()
        deploy_data = request.get_json()
        
        if not deploy_data or "new_version" not in deploy_data:
            return jsonify({"error": "new_version is required"}), 400
        
        new_version = deploy_data["new_version"]
        success = registry.deploy_grammar_version(proposal_id, new_version)
        
        if success:
            return jsonify({
                "proposal_id": proposal_id,
                "new_version": new_version,
                "deployed": True,
                "message": f"Grammar version {new_version} deployed successfully"
            })
        else:
            return jsonify({
                "proposal_id": proposal_id,
                "deployed": False,
                "message": "Deployment failed - proposal not in approved/validated state"
            }), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error deploying grammar version: {e}")
        return jsonify({"error": str(e)}), 500


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
    # For standalone testing
    app.run(host="0.0.0.0", port=5000, debug=True)
