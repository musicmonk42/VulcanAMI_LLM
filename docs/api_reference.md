# API Reference (Registry & Arena)

**Last Updated:** December 2024 
**API Version:** v1.0

This document provides comprehensive API reference for both the Registry API (Flask) and Arena API (FastAPI).

---

## Table of Contents
1. [Authentication](#1-authentication)
2. [Common JSON Conventions](#2-common-json-conventions)
3. [Registry Endpoints](#3-registry-endpoints)
4. [Arena Endpoints](#4-arena-endpoints)
5. [Error Schema](#5-error-schema)
6. [Rate Limiting](#6-rate-limiting)
7. [Metrics Endpoint](#7-metrics-endpoint-snippet)
8. [Security Headers](#8-security-headers)
9. [Versioning & Deprecation](#9-versioning--deprecation)
10. [Future Extensions](#10-future-extensions)

---

## 1. Authentication

### JWT Authentication (Registry API)

**Header Format:**
```
Authorization: Bearer <token>
```

**JWT Claims:**
- `sub`: Subject (agent ID)
- `trust_level`: Trust level score (0.0 to 1.0)
- `scopes`: Array of permission scopes
- `iat`: Issued at timestamp (Unix epoch)
- `exp`: Expiration timestamp (Unix epoch)

**Token Expiration:** 30 minutes (configurable via JWT_EXP_MINUTES)

**Example JWT Payload:**
```json
{
 "sub": "agent_alpha_001",
 "trust_level": 0.85,
 "scopes": ["read", "write", "propose"],
 "iat": 1701234567,
 "exp": 1701236367
}
```

### API Key Authentication (Arena API)

**Header Format:**
```
X-API-Key: <key>
```

**Key Format:** 32-character alphanumeric string
**Storage:** Keys are stored hashed with bcrypt
**Revocation:** Supported via management endpoints

**Example Request:**
```bash
curl -H "X-API-Key: abc123def456ghi789jkl012" \
 https://api.example.com/execute/graph
```

### Bootstrap Authentication

**Purpose:** One-time initialization to create the first admin/agent
**Header Format:**
```
X-Bootstrap-Key: <bootstrap_secret>
```

**Security Notes:**
- Only works when no agents exist in the system
- Should be disabled after initial setup
- HTTPS enforcement recommended for bootstrap endpoint 

## 2. Common JSON Conventions

### Timestamps
All timestamps use **RFC3339 format** with timezone:
```json
"created_at": "2025-11-11T03:35:00Z"
```

### Pagination
Cursor-based pagination (future implementation):
```json
{
 "limit": 50,
 "next_cursor": "eyJpZCI6MTIzNDU2fQ=="
}
```

### Risk Scores
Normalized to the range `[0.0, 1.0]`:
- `0.0` = No risk
- `0.5` = Medium risk
- `1.0` = Maximum risk

```json
{
 "risk": {
 "score": 0.11,
 "factors": {
 "node_count": 1,
 "complexity": 0.05,
 "external_calls": 0
 }
 }
}
```

### Response Status Codes
- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## 3. Registry Endpoints

The Registry API (Flask) handles authentication, agent onboarding, proposal submission, and audit log queries.

**Base URL:** `http://localhost:5000` (development) or your configured HOST

### Endpoint Summary

| Method | Path | Purpose | Auth | Rate Limit |
|--------|------|---------|------|------------|
| POST | `/registry/bootstrap` | Initialize first agent | Bootstrap Key | N/A |
| POST | `/auth/login` | Issue JWT token | None | 50/hour |
| POST | `/registry/onboard` | Add agent with trust level | JWT | 200/day |
| POST | `/ir/propose` | Submit graph/ontology proposal | JWT | 100/day |
| GET | `/audit/logs` | Retrieve audit entries | JWT | 1000/hour |
| GET | `/health` | Liveness check | None | Unlimited |
| GET | `/metrics` | Prometheus metrics | Optional | Unlimited |

### POST /registry/bootstrap

**Purpose:** Create the first admin/agent in an empty system

**Authentication:** Bootstrap Key (X-Bootstrap-Key header)

**Request:**
```json
{
 "agent_id": "admin_001",
 "trust_level": 1.0,
 "scopes": ["admin", "read", "write", "propose"],
 "metadata": {
 "description": "System administrator",
 "created_by": "deployment_script"
 }
}
```

**Response (201 Created):**
```json
{
 "agent_id": "admin_001",
 "trust_level": 1.0,
 "created_at": "2025-11-11T03:35:00Z",
 "message": "Bootstrap agent created successfully"
}
```

**Security Notes:**
- Only works when agent database is empty
- HTTPS enforcement recommended in production
- Disable bootstrap key after initial setup

### POST /auth/login

**Purpose:** Authenticate and receive JWT token

**Authentication:** None (public endpoint)

**Request:**
```json
{
 "agent_id": "agent_alpha_001",
 "password": "secure_password_here",
 "requested_scopes": ["read", "write"]
}
```

**Response (200 OK):**
```json
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
 "token_type": "Bearer",
 "expires_in": 1800,
 "scopes": ["read", "write"]
}
```

**Error Response (401 Unauthorized):**
```json
{
 "error": {
 "type": "AuthenticationError",
 "message": "Invalid credentials",
 "retryable": false
 }
}
```

### POST /registry/onboard

**Purpose:** Register a new agent with specified trust level

**Authentication:** JWT with 'admin' scope

**Request:**
```json
{
 "agent_id": "agent_beta_002",
 "trust_level": 0.75,
 "scopes": ["read", "write"],
 "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
 "metadata": {
 "team": "data_science",
 "description": "ML experiment runner"
 }
}
```

**Response (201 Created):**
```json
{
 "agent_id": "agent_beta_002",
 "trust_level": 0.75,
 "scopes": ["read", "write"],
 "created_at": "2025-11-11T03:40:00Z",
 "status": "active"
}
```

### POST /ir/propose

**Purpose:** Submit a graph or ontology proposal for governance review

**Authentication:** JWT with 'propose' scope

**Request:**
```json
{
 "proposal_type": "graph",
 "nodes": [
 {
 "id": "c1",
 "type": "CONST",
 "params": { "value": 10 }
 },
 {
 "id": "add1",
 "type": "ADD",
 "params": {}
 }
 ],
 "edges": [
 {
 "from": "c1",
 "to": "add1",
 "port": "input_a"
 }
 ],
 "metadata": {
 "authors": ["agent_alpha_001"],
 "version": "1.0.0",
 "description": "Basic arithmetic graph",
 "tags": ["arithmetic", "example"]
 }
}
```

**Response (201 Created):**
```json
{
 "proposal_id": "prop_2025_00123",
 "status": "draft",
 "hash": "sha256:de...ad",
 "risk": {
 "score": 0.11,
 "factors": {
 "node_count": 2,
 "complexity": 0.05,
 "external_calls": 0
 }
 },
 "created_at": "2025-11-11T03:35:00Z",
 "voting_opens_at": "2025-11-11T04:00:00Z",
 "expires_at": "2025-11-18T03:35:00Z"
}
```

### GET /audit/logs

**Purpose:** Query audit trail entries

**Authentication:** JWT with 'audit' scope

**Query Parameters:**
- `start_time` (optional): RFC3339 timestamp
- `end_time` (optional): RFC3339 timestamp
- `agent_id` (optional): Filter by agent
- `severity` (optional): critical, high, medium, low
- `limit` (optional): Max results (default: 100, max: 1000)

**Request:**
```bash
GET /audit/logs?start_time=2025-11-11T00:00:00Z&severity=high&limit=50
```

**Response (200 OK):**
```json
{
 "logs": [
 {
 "id": "audit_001",
 "timestamp": "2025-11-11T03:35:00Z",
 "agent_id": "agent_alpha_001",
 "action": "proposal_submitted",
 "severity": "medium",
 "details": {
 "proposal_id": "prop_2025_00123",
 "risk_score": 0.11
 }
 }
 ],
 "total_count": 42,
 "next_cursor": null
}
```

### GET /health

**Purpose:** Liveness and readiness check

**Authentication:** None

**Response (200 OK):**
```json
{
 "status": "healthy",
 "timestamp": "2025-11-11T03:35:00Z",
 "version": "1.0.0",
 "components": {
 "database": "healthy",
 "redis": "healthy",
 "audit_log": "healthy"
 }
}
```

### GET /metrics

**Purpose:** Prometheus metrics exposition

**Authentication:** Optional (configurable)

**Response (200 OK):**
```
# HELP graphix_nodes_executed Total nodes executed
# TYPE graphix_nodes_executed counter
graphix_nodes_executed 508

# HELP graphix_cache_hit_rate Cache hit rate
# TYPE graphix_cache_hit_rate gauge
graphix_cache_hit_rate 0.67
...
```

## 4. Arena Endpoints

The Arena API (FastAPI) handles graph execution, monitoring, and operational tasks.

**Base URL:** `http://localhost:8000` (development) or your configured API_HOST

### Endpoint Summary

| Method | Path | Purpose | Auth | Rate Limit |
|--------|------|---------|------|------------|
| GET | `/health` | Liveness check | Optional | Unlimited |
| GET | `/ready` | Readiness check | Optional | Unlimited |
| POST | `/execute/graph` | Execute validated graph | API Key | 1000/hour |
| GET | `/execution/{id}/status` | Check execution status | API Key | 5000/hour |
| POST | `/execution/{id}/cancel` | Cancel running execution | API Key | 100/hour |
| GET | `/metrics` | Prometheus metrics | API Key | Unlimited |

### GET /health

**Purpose:** Check if the service is alive

**Authentication:** Optional

**Response (200 OK):**
```json
{
 "status": "healthy",
 "timestamp": "2025-11-11T03:35:00Z",
 "version": "1.0.0"
}
```

### GET /ready

**Purpose:** Check if service and dependencies are ready to serve requests

**Authentication:** Optional

**Response (200 OK):**
```json
{
 "ready": true,
 "dependencies": {
 "vulcan_world_model": "ready",
 "graph_compiler": "ready",
 "llm_core": "ready",
 "persistent_memory": "ready"
 },
 "timestamp": "2025-11-11T03:35:00Z"
}
```

**Response (503 Service Unavailable):**
```json
{
 "ready": false,
 "dependencies": {
 "vulcan_world_model": "ready",
 "graph_compiler": "initializing",
 "llm_core": "error",
 "persistent_memory": "ready"
 },
 "timestamp": "2025-11-11T03:35:00Z"
}
```

### POST /execute/graph

**Purpose:** Execute a validated graph artifact with specified execution profile

**Authentication:** API Key (X-API-Key header)

**Request:**
```json
{
 "graph_id": "graph_abc123",
 "artifact": {
 "nodes": [
 {
 "id": "input_1",
 "type": "INPUT",
 "params": { "value": [1, 2, 3, 4, 5] }
 },
 {
 "id": "transform_1",
 "type": "TRANSFORM",
 "params": { "operation": "multiply", "factor": 2 }
 },
 {
 "id": "output_1",
 "type": "OUTPUT",
 "params": {}
 }
 ],
 "edges": [
 { "from": "input_1", "to": "transform_1", "port": "data" },
 { "from": "transform_1", "to": "output_1", "port": "result" }
 ],
 "metadata": {
 "version": "1.0.2",
 "description": "Simple transformation pipeline"
 }
 },
 "execution_profile": {
 "mode": "parallel",
 "max_concurrency": 8,
 "timeout_seconds": 120,
 "retry_policy": {
 "max_retries": 2,
 "backoff_factor": 2.0
 },
 "resource_limits": {
 "max_memory_mb": 4096,
 "max_cpu_cores": 4
 }
 }
}
```

**Response (202 Accepted):**
```json
{
 "execution_id": "exec_2025_00456",
 "status": "queued",
 "graph_id": "graph_abc123",
 "estimated_duration_seconds": 45,
 "created_at": "2025-11-11T03:35:00Z",
 "status_url": "/execution/exec_2025_00456/status"
}
```

**Response (400 Bad Request):**
```json
{
 "error": {
 "type": "ValidationError",
 "message": "Graph contains cycle",
 "details": {
 "cycle_path": ["node_1", "node_2", "node_3", "node_1"]
 },
 "retryable": false
 }
}
```

### GET /execution/{id}/status

**Purpose:** Check the status of a running or completed execution

**Authentication:** API Key

**Response (200 OK) - Running:**
```json
{
 "execution_id": "exec_2025_00456",
 "status": "running",
 "progress": {
 "nodes_completed": 5,
 "nodes_total": 10,
 "percentage": 50.0
 },
 "started_at": "2025-11-11T03:35:00Z",
 "estimated_completion": "2025-11-11T03:36:00Z"
}
```

**Response (200 OK) - Completed:**
```json
{
 "execution_id": "exec_2025_00456",
 "status": "completed",
 "result": {
 "output_1": [2, 4, 6, 8, 10]
 },
 "metrics": {
 "duration_seconds": 42.5,
 "nodes_executed": 10,
 "nodes_cached": 2,
 "resource_usage": {
 "peak_memory_mb": 1024,
 "cpu_time_seconds": 38.2
 }
 },
 "started_at": "2025-11-11T03:35:00Z",
 "completed_at": "2025-11-11T03:36:42Z"
}
```

### POST /execution/{id}/cancel

**Purpose:** Cancel a running execution

**Authentication:** API Key

**Response (200 OK):**
```json
{
 "execution_id": "exec_2025_00456",
 "status": "cancelled",
 "cancelled_at": "2025-11-11T03:35:30Z",
 "partial_results": {
 "nodes_completed": 3
 }
}
```

### GET /metrics

**Purpose:** Prometheus metrics for monitoring

**Authentication:** API Key

**Response (200 OK):**
```
# HELP arena_executions_total Total executions
# TYPE arena_executions_total counter
arena_executions_total{status="completed"} 1542
arena_executions_total{status="failed"} 23

# HELP arena_execution_duration_seconds Execution duration
# TYPE arena_execution_duration_seconds histogram
arena_execution_duration_seconds_bucket{le="1.0"} 245
arena_execution_duration_seconds_bucket{le="5.0"} 892
arena_execution_duration_seconds_bucket{le="30.0"} 1432
...
```

## 5. Error Schema

All error responses follow a consistent schema for easy parsing and handling.

### Standard Error Format

```json
{
 "error": {
 "type": "ValidationError",
 "message": "Missing 'nodes' key in graph definition",
 "details": {
 "path": "$.artifact",
 "required_field": "nodes"
 },
 "severity": "error",
 "retryable": false,
 "request_id": "req_123abc",
 "timestamp": "2025-11-11T03:40:00Z",
 "remediation": "Add 'nodes' array to your graph artifact"
 }
}
```

### Error Types

| Error Type | HTTP Status | Description | Retryable |
|-----------|-------------|-------------|-----------|
| `ValidationError` | 400 | Invalid request data | No |
| `AuthenticationError` | 401 | Missing/invalid authentication | No |
| `AuthorizationError` | 403 | Insufficient permissions | No |
| `NotFoundError` | 404 | Resource not found | No |
| `RateLimitError` | 429 | Too many requests | Yes (after delay) |
| `TimeoutError` | 504 | Request timeout | Yes |
| `InternalError` | 500 | Server error | Yes |
| `ServiceUnavailableError` | 503 | Service temporarily down | Yes |

### Extended Error Fields

**Optional fields that may be present:**

- `correlation_id`: Trace ID for distributed tracing
- `risk_context`: Additional risk assessment information
- `affected_resources`: List of resources affected by the error
- `suggested_actions`: Array of recommended next steps

**Example with extended fields:**
```json
{
 "error": {
 "type": "RateLimitError",
 "message": "Rate limit exceeded for API endpoint",
 "severity": "warning",
 "retryable": true,
 "request_id": "req_789xyz",
 "timestamp": "2025-11-11T03:40:00Z",
 "details": {
 "limit": "1000 requests per hour",
 "current_usage": 1000,
 "reset_at": "2025-11-11T04:00:00Z"
 },
 "remediation": "Wait until rate limit resets or upgrade your plan",
 "retry_after_seconds": 1200
 }
}
```

## 6. Rate Limiting

Rate limits protect the API from abuse and ensure fair usage across all clients.

### Rate Limit Headers

All responses include rate limit information in headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 742
X-RateLimit-Reset: 1701234567
```

**Header Definitions:**
- `X-RateLimit-Limit`: Maximum requests allowed in the current window
- `X-RateLimit-Remaining`: Requests remaining in the current window
- `X-RateLimit-Reset`: Unix timestamp when the rate limit resets

### Default Rate Limits

| Endpoint Category | Limit | Window | Scope |
|------------------|-------|--------|-------|
| Authentication | 50/hour | 1 hour | Per IP |
| Onboarding | 200/day | 24 hours | Per agent |
| Proposals | 100/day | 24 hours | Per agent |
| Execution | 1000/hour | 1 hour | Per API key |
| Audit Logs | 1000/hour | 1 hour | Per agent |
| Health/Metrics | Unlimited | N/A | N/A |

### Rate Limit Response

When rate limit is exceeded, the API returns HTTP 429:

```json
{
 "error": {
 "type": "RateLimitError",
 "message": "Rate limit exceeded",
 "details": {
 "limit": "1000 requests per hour",
 "window_reset_at": "2025-11-11T04:00:00Z",
 "retry_after_seconds": 1200
 },
 "retryable": true
 }
}
```

### Governance Throttling

Additional throttling applies to proposals to prevent spam:

- **MAX_PROPOSALS_PER_AGENT_PER_HOUR**: 10 proposals
- **Similarity Damping**: Duplicate or near-duplicate proposals are rejected
- **Trust Level Factor**: Higher trust levels may have higher limits

### Best Practices

1. **Respect Rate Limits**: Monitor headers and adjust request rate accordingly
2. **Implement Exponential Backoff**: Wait longer between retries after failures
3. **Cache Responses**: Cache frequently accessed data to reduce API calls
4. **Batch Operations**: Use batch endpoints when available
5. **Use Webhooks**: Subscribe to webhooks instead of polling

## 7. Metrics Endpoint Snippet
```
graphix_nodes_executed 508
graphix_cache_hit_rate 0.67
graphix_success_rate 0.94
graphix_total_latency_ms 1234
graphix_rss_mb 210.5
```

## 8. Security Headers
Recommended:
- Strict-Transport-Security
- X-Frame-Options: deny
- X-Content-Type-Options: nosniff
- Content-Security-Policy: script-src 'none' (if UI absent)

## 9. Versioning & Deprecation
- Semantic version for IR vs grammar separately.
- Deprecation flow: announce → grace period → enforced rejection with DeprecationError.

## 10. Future Extensions
Planned endpoints:
- /governance/proposals/{id}/vote
- /graphs/{id}/status
- /graphs/{id}/cancel
- /provenance/{artifact_id}
- /alignment/score (integration with VULCAN)
