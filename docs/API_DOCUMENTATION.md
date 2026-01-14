# VulcanAMI API Documentation

**Version:** 1.0.0 
**Last Updated:** December 2024 
**Copyright © 2024 Novatrax Labs LTD. All rights reserved.**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Authentication](#3-authentication)
4. [Gateway API Endpoints](#4-gateway-api-endpoints)
5. [Core API Endpoints](#5-core-api-endpoints)
6. [Common Workflows](#6-common-workflows)
7. [Error Handling](#7-error-handling)
8. [Rate Limiting](#8-rate-limiting)
9. [Security Headers](#9-security-headers)
10. [GraphQL API](#10-graphql-api)
11. [WebSocket API](#11-websocket-api)
12. [Appendix](#12-appendix)

---

## 1. System Overview

### What is VulcanAMI?

VulcanAMI (Vulcan Advanced Machine Intelligence) is Novatrax's enterprise-grade AI-native graph execution and governance platform. It provides a complete cognitive architecture that enables:

- **AI Agent Orchestration**: Coordinate multiple AI agents with different capabilities
- **Graph-Based Workflow Execution**: Define and execute complex workflows as typed, JSON-based graphs
- **Cognitive Reasoning**: Multiple reasoning modalities (symbolic, probabilistic, causal, analogical)
- **Trust-Weighted Governance**: Decentralized decision-making with consensus mechanisms
- **Persistent Memory**: Long-term and associative memory with intelligent retrieval
- **Safety & Ethics Boundaries**: CSIU framework (Curiosity, Safety, Impact, Uncertainty) for responsible AI

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Multi-Modal Processing** | Process text, embeddings, and structured data |
| **Hierarchical Planning** | Generate and execute complex plans with resource awareness |
| **Continual Learning** | Learn from experiences without catastrophic forgetting |
| **Causal Reasoning** | Estimate causal effects and perform interventions |
| **Knowledge Crystallization** | Extract and consolidate knowledge over time |
| **Semantic Bridge** | Cross-domain concept transfer and reasoning |
| **Graph Compilation** | LLVM-based optimization for 10-100x speedup |

### Use Cases

- Safety-governed agentic systems
- Provenance-aware ML operations
- Orchestrating LLM and tool pipelines with control and visibility
- Policy-driven evolution of workflows across teams
- Multi-stakeholder AI governance

*Developed by Novatrax Labs LTD*

---

## 2. Architecture

VulcanAMI follows a layered microservices architecture with two primary API layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLIENT APPLICATIONS │
└─────────────────────────────────────────────────────────────────────────────┘
 │
 ┌────────────────┴────────────────┐
 ▼ ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ GATEWAY API (aiohttp) │ │ CORE API (HTTP Server) │
│ Port 8080 (Default) │ │ Port 8000 (Default) │
├───────────────────────────────┤ ├───────────────────────────────┤
│ • Authentication/Authorization│ │ • Graph Submission │
│ • Rate Limiting │ │ • Proposal Management │
│ • Circuit Breakers │ │ • Voting System │
│ • Caching │ │ • Reasoning Engine │
│ • Service Discovery │ │ • Direct DB Access │
│ • WebSocket Support │ │ • Audit Logging │
│ • GraphQL Interface │ │ • Agent Management │
└───────────────────────────────┘ └───────────────────────────────┘
 │ │
 └────────────────┬────────────────┘
 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ VULCAN-AMI CORE (285,000+ LOC) │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Reasoning │ │ World Model │ │ Memory │ │ Safety │ │
│ │ Systems │ │ (Causal) │ │ Hierarchy │ │ Validator │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Planning │ │ Learning │ │ Semantic │ │ Consensus │ │
│ │ Engine │ │ Systems │ │ Bridge │ │ Engine │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PERSISTENCE & OBSERVABILITY │
├─────────────────────────────────────────────────────────────────────────────┤
│ SQLite/PostgreSQL │ Redis │ Prometheus │ Grafana │ Audit Logs │ S3/CDN │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Gateway vs Core Components

| Aspect | Gateway API | Core API |
|--------|-------------|----------|
| **Framework** | aiohttp (async) | Custom HTTP Server (threaded) |
| **Port** | 8080 | 8000 |
| **Purpose** | External-facing, high-level API | Internal services, graph execution |
| **Authentication** | JWT with refresh tokens | JWT or API Key |
| **Features** | WebSocket, GraphQL, Caching | Direct DB access, Agent CRUD |
| **Rate Limiting** | Redis-backed with fallback | Token bucket per-IP/user |
| **Scaling** | Horizontal with Redis | Vertical with thread pool |

---

## 3. Authentication

VulcanAMI supports multiple authentication methods depending on the API layer.

### 3.1 Gateway Authentication (JWT)

The Gateway API uses JWT tokens with access/refresh token rotation.

#### Login Flow

```
POST /auth/login-gateway
```

**Request:**
```json
{
 "username": "admin",
 "password": "your-secure-password"
}
```

**Response (200 OK):**
```json
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "token_type": "Bearer",
 "expires_in": 1800,
 "user_id": "admin",
 "roles": ["admin"],
 "scopes": ["read", "write", "admin"]
}
```

#### Using the Access Token

Include the token in the Authorization header:
```
Authorization: Bearer <access_token>
```

#### Token Refresh

```
POST /auth/refresh
```

**Request:**
```json
{
 "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9..."
}
```

**Response (200 OK):**
```json
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "token_type": "Bearer",
 "expires_in": 1800
}
```

### 3.2 Core Authentication (JWT/API Key)

The Core API supports both JWT and API Key authentication.

#### API Key Authentication

Include in the header:
```
X-API-Key: <your-api-key>
```

API keys are 32-128 character lowercase hexadecimal strings (e.g., `a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4`).

#### JWT Login (Core)

```
POST /auth/login-core
```

**Request:**
```json
{
 "api_key": "your-api-key",
 "password": "your-password"
}
```

Or with mutual proof (challenge-response):
```json
{
 "api_key": "your-api-key",
 "nonce": "random-nonce",
 "timestamp": "2024-12-14T10:30:00Z",
 "proof": "base64-encoded-hmac-signature"
}
```

**Response (200 OK):**
```json
{
 "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
 "agent_id": "agent_abc12345",
 "expires_in": 86400,
 "issuer": "graphix-api",
 "audience": "graphix-clients",
 "kid": "abc123def456"
}
```

### 3.3 JWT Token Structure

**JWT Claims:**
```json
{
 "sub": "user_id",
 "user_id": "user_id",
 "roles": ["user", "admin"],
 "scopes": ["read", "write"],
 "type": "access",
 "jti": "unique-token-id",
 "iss": "vulcan-ami-gateway",
 "aud": "vulcan-clients",
 "iat": 1702540800,
 "exp": 1702542600
}
```

### 3.4 Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| `user` | Basic read/write operations |
| `admin` | System configuration, user management |
| `govern` | Proposal creation, voting |
| `developer` | Extended API access |

| Scope | Description |
|-------|-------------|
| `read` | Read operations (GET requests) |
| `write` | Write operations (POST, PUT, DELETE) |
| `admin` | Administrative operations |

---

## 4. Gateway API Endpoints

Base URL: `https://gateway.example.com` (or `http://localhost:8080` for development)

### 4.1 Health & Monitoring

#### GET /health/components

**Purpose:** Get detailed health status of all gateway components.

**Authentication:** None required

**Response (200 OK):**
```json
{
 "timestamp": 1702540800.123,
 "service": "vulcan-api-gateway",
 "version": "1.0.0",
 "components": {
 "api_gateway": true,
 "service_registry": true,
 "auth_manager": true,
 "cache_manager": true,
 "rate_limiter": true,
 "redis_client": true,
 "websocket_support": true,
 "graphql_support": true
 },
 "degraded_mode": {
 "auth": false,
 "cache": false,
 "rate_limiter": false
 },
 "registered_services": 5,
 "statistics": {
 "total": 8,
 "available": 8,
 "missing": 0
 },
 "health_summary": {
 "status": "healthy",
 "components_health": "8/8 available"
 }
}
```

**Use Case:** Kubernetes/Docker health probes, monitoring dashboards.

---

#### GET /metrics-gateway

**Purpose:** Prometheus metrics for monitoring.

**Authentication:** None required

**Response (200 OK):**
```
# HELP api_gateway_requests_total Total API requests
# TYPE api_gateway_requests_total counter
api_gateway_requests_total{method="GET",endpoint="/health",status="200"} 1542

# HELP api_gateway_request_duration_seconds Request duration
# TYPE api_gateway_request_duration_seconds histogram
api_gateway_request_duration_seconds_bucket{method="POST",endpoint="/v1/reason",le="0.1"} 234

# HELP api_gateway_active_connections Active connections
# TYPE api_gateway_active_connections gauge
api_gateway_active_connections 42

# HELP api_gateway_cache_hits_total Cache hits
# TYPE api_gateway_cache_hits_total counter
api_gateway_cache_hits_total{cache_type="memory"} 8923
```

**Use Case:** Prometheus scraping, Grafana dashboards.

---

### 4.2 Authentication Endpoints

#### POST /auth/login-gateway

**Purpose:** Authenticate and receive access/refresh tokens.

**Authentication:** None (public endpoint)

**Request:**
```json
{
 "username": "alice",
 "password": "secure-password-123"
}
```

**Response (200 OK):**
```json
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "token_type": "Bearer",
 "expires_in": 1800,
 "user_id": "alice",
 "roles": ["user"],
 "scopes": ["read", "write"]
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid credentials
- `429 Too Many Requests`: Rate limit exceeded

**Use Case:** Initial authentication for all Gateway API operations.

---

#### POST /auth/refresh

**Purpose:** Refresh expired access token using refresh token.

**Authentication:** None (public endpoint)

**Request:**
```json
{
 "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9..."
}
```

**Response (200 OK):**
```json
{
 "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9...",
 "token_type": "Bearer",
 "expires_in": 1800
}
```

**Note:** Refresh tokens are rotated on each use. Old refresh tokens become invalid.

**Use Case:** Maintain long-lived sessions without re-authentication.

---

#### POST /auth/logout-gateway

**Purpose:** Revoke access and refresh tokens.

**Authentication:** Bearer token required

**Request:**
```json
{
 "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InZ1bGNhbi1rZXktMSJ9..."
}
```

**Response (200 OK):**
```json
{
 "message": "Logged out successfully"
}
```

**Use Case:** Secure session termination, token invalidation.

---

### 4.3 Processing Endpoints

#### POST /v1/process

**Purpose:** Process multimodal input (text, embeddings, structured data).

**Authentication:** Bearer token with `write` scope

**Request:**
```json
{
 "input": "What is the capital of France?"
}
```

Or with structured input:
```json
{
 "input": {
 "text": "Analyze this data",
 "data": [1, 2, 3, 4, 5],
 "type": "numerical_analysis"
 }
}
```

**Response (200 OK):**
```json
{
 "embedding": [0.123, -0.456, 0.789, ...],
 "modality": "text",
 "uncertainty": 0.15,
 "metadata": {
 "token_count": 7,
 "processing_time_ms": 45
 }
}
```

**Use Case:** Convert raw input into embeddings for further processing.

---

#### POST /v1/plan

**Purpose:** Generate an execution plan for a goal.

**Authentication:** Bearer token with `write` scope

**Request:**
```json
{
 "goal": "Deploy a machine learning model to production",
 "context": {
 "model_type": "classification",
 "infrastructure": "kubernetes",
 "constraints": {
 "max_latency_ms": 100,
 "min_accuracy": 0.95
 }
 }
}
```

**Response (200 OK):**
```json
{
 "plan_id": "plan_abc123",
 "steps": [
 {
 "step_id": "step_1",
 "action": "validate_model",
 "dependencies": [],
 "estimated_duration_seconds": 60
 },
 {
 "step_id": "step_2",
 "action": "create_container",
 "dependencies": ["step_1"],
 "estimated_duration_seconds": 120
 },
 {
 "step_id": "step_3",
 "action": "deploy_to_cluster",
 "dependencies": ["step_2"],
 "estimated_duration_seconds": 180
 }
 ],
 "total_estimated_duration_seconds": 360,
 "risk_assessment": {
 "score": 0.25,
 "factors": ["infrastructure_complexity"]
 }
}
```

**Use Case:** Strategic planning for complex multi-step operations.

---

#### POST /v1/execute

**Purpose:** Execute a generated plan.

**Authentication:** Bearer token with `write` scope

**Request:**
```json
{
 "plan": {
 "plan_id": "plan_abc123",
 "steps": [
 {
 "step_id": "step_1",
 "action": "validate_model"
 }
 ]
 }
}
```

**Response (200 OK):**
```json
{
 "execution_id": "exec_xyz789",
 "status": "completed",
 "results": {
 "step_1": {
 "status": "success",
 "output": {"validation_passed": true},
 "duration_ms": 1234
 }
 },
 "metrics": {
 "total_duration_ms": 1234,
 "steps_completed": 1,
 "steps_failed": 0
 }
}
```

**Use Case:** Execute automated workflows, run validated plans.

---

#### POST /v1/learn

**Purpose:** Submit learning experiences for continual learning.

**Authentication:** Bearer token with `write` scope

**Request:**
```json
{
 "experience": {
 "input": "User query about machine learning",
 "output": "Generated response about ML concepts",
 "feedback": {
 "rating": 4.5,
 "helpful": true,
 "corrections": null
 }
 }
}
```

**Response (200 OK):**
```json
{
 "adapted": true,
 "loss": 0.023,
 "metadata": {
 "experience_id": "exp_123",
 "learning_rate": 0.001,
 "samples_processed": 1
 }
}
```

**Use Case:** Improve model performance through feedback loops.

---

#### POST /v1/reason

**Purpose:** Perform reasoning operations (symbolic, probabilistic, causal).

**Authentication:** Bearer token with `write` scope

**Request (Probabilistic Reasoning):**
```json
{
 "query": "What is the probability of success?",
 "type": "probabilistic",
 "input": [0.8, 0.6, 0.9]
}
```

**Request (Symbolic Reasoning):**
```json
{
 "query": "IF weather = sunny AND temperature > 70 THEN activity = outdoor",
 "type": "symbolic"
}
```

**Request (Causal Reasoning):**
```json
{
 "query": "estimate_effect",
 "type": "causal",
 "treatment": "marketing_campaign",
 "outcome": "sales_increase"
}
```

**Response (200 OK):**
```json
{
 "result": {
 "mean": 0.85,
 "std": 0.12,
 "confidence_interval": [0.73, 0.97],
 "reasoning_trace": [
 "Analyzed input probabilities",
 "Applied Bayesian inference",
 "Computed posterior distribution"
 ]
 }
}
```

**Use Case:** Complex decision-making, what-if analysis, logical inference.

---

### 4.4 Memory Endpoints

#### GET /v1/memory/search

**Purpose:** Search the memory store for relevant information.

**Authentication:** Bearer token with `read` scope

**Query Parameters:**
- `q` (required): Search query string
- `k` (optional): Number of results to return (default: 10)

**Request:**
```
GET /v1/memory/search?q=machine+learning+deployment&k=5
```

**Response (200 OK):**
```json
{
 "results": [
 {
 "id": "mem_001",
 "score": 0.95,
 "metadata": {
 "content_type": "documentation",
 "created_at": "2024-12-01T10:00:00Z"
 }
 },
 {
 "id": "mem_002",
 "score": 0.87,
 "metadata": {
 "content_type": "conversation",
 "created_at": "2024-12-10T15:30:00Z"
 }
 }
 ]
}
```

**Use Case:** Retrieve relevant context for decision-making, RAG applications.

---

#### POST /v1/memory/store

**Purpose:** Store new content in the memory system.

**Authentication:** Bearer token with `write` scope

**Request:**
```json
{
 "content": "Best practices for deploying ML models include containerization, CI/CD pipelines, and monitoring.",
 "metadata": {
 "content_type": "documentation",
 "source": "internal_wiki",
 "tags": ["ml", "deployment", "best-practices"]
 }
}
```

**Response (200 OK):**
```json
{
 "id": "mem_003",
 "stored": true
}
```

**Use Case:** Build knowledge base, store conversation history.

---

### 4.5 System Management

#### GET /v1/status

**Purpose:** Get comprehensive system status.

**Authentication:** Bearer token with `read` scope

**Response (200 OK):**
```json
{
 "status": "active",
 "version": "2.1.0",
 "uptime_seconds": 86400,
 "components": {
 "world_model": "active",
 "reasoning_engines": {
 "symbolic": "active",
 "probabilistic": "active",
 "causal": "active"
 },
 "memory": "active",
 "planner": "active",
 "learner": "active"
 },
 "metrics": {
 "requests_processed": 15432,
 "avg_latency_ms": 45,
 "error_rate": 0.002
 }
}
```

**Use Case:** System monitoring, health dashboards.

---

#### POST /v1/configure

**Purpose:** Update system configuration (admin only).

**Authentication:** Bearer token with `admin` scope

**Request:**
```json
{
 "enable_learning": true,
 "max_workers": 8,
 "cache_ttl_seconds": 3600
}
```

**Response (200 OK):**
```json
{
 "configured": true,
 "settings": {
 "enable_learning": true,
 "max_workers": 8,
 "cache_ttl_seconds": 3600
 }
}
```

**Use Case:** Runtime configuration changes without restart.

---

#### POST /graphql-gateway

**Purpose:** Execute GraphQL queries.

**Authentication:** Bearer token with `read` scope

**Request:**
```json
{
 "query": "query { status memorySearch(query: \"deployment\", k: 5) }",
 "variables": {}
}
```

**Response (200 OK):**
```json
{
 "data": {
 "status": "healthy",
 "memorySearch": ["mem_001", "mem_002", "mem_003"]
 }
}
```

**Use Case:** Flexible queries, frontend applications.

---

#### GET /status

**Purpose:** Basic status check for load balancers.

**Authentication:** None

**Response (200 OK):**
```json
{
 "status": "healthy",
 "timestamp": "2024-12-14T10:30:00Z"
}
```

---

## 5. Core API Endpoints

Base URL: `https://core-api.example.com` (or `http://localhost:8000` for development)

### 5.1 Authentication

#### POST /auth/login-core

**Purpose:** Authenticate to the Core API.

**Authentication:** None (public endpoint)

**Request:**
```json
{
 "api_key": "your-api-key-here",
 "password": "your-password"
}
```

**Response (200 OK):**
```json
{
 "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
 "agent_id": "agent_abc12345",
 "expires_in": 86400,
 "issuer": "graphix-api",
 "audience": "graphix-clients",
 "kid": "abc123def456"
}
```

---

#### POST /auth/logout-core

**Purpose:** Logout and revoke token.

**Authentication:** Bearer token or API Key

**Response (200 OK):**
```json
{
 "status": "revoked",
 "jti": "token-id"
}
```

---

### 5.2 Graph Operations

#### POST /graphs/submit

**Purpose:** Submit a graph for execution.

**Authentication:** Bearer token or API Key (requires `user` role)

**Request:**
```json
{
 "graph": {
 "id": "my-workflow",
 "type": "Graph",
 "nodes": [
 {
 "id": "input_1",
 "type": "INPUT",
 "params": {"value": [1, 2, 3, 4, 5]}
 },
 {
 "id": "transform_1",
 "type": "TRANSFORM",
 "params": {"operation": "multiply", "factor": 2}
 },
 {
 "id": "output_1",
 "type": "OUTPUT",
 "params": {}
 }
 ],
 "edges": [
 {"from": "input_1", "to": "transform_1"},
 {"from": "transform_1", "to": "output_1"}
 ]
 },
 "priority": 5,
 "timeout": 120,
 "callback": "https://my-service.example.com/webhook"
}
```

**Response (200 OK):**
```json
{
 "status": "submitted",
 "graph_id": "graph_xyz789",
 "queue_position": 3
}
```

**Validation Errors (400 Bad Request):**
```json
{
 "error": "Invalid graph: Missing required field: id",
 "code": 400,
 "timestamp": "2024-12-14T10:30:00Z"
}
```

**Use Case:** Submit complex workflows for background execution.

---

#### GET /graphs/{graph_id}

**Purpose:** Retrieve graph execution status and results.

**Authentication:** None (can be configured)

**Response (200 OK):**
```json
{
 "id": "graph_xyz789",
 "agent_id": "agent_abc12345",
 "status": "completed",
 "submitted_at": "2024-12-14T10:30:00Z",
 "started_at": "2024-12-14T10:30:05Z",
 "completed_at": "2024-12-14T10:30:15Z",
 "result": {
 "nodes_processed": 3,
 "edges_processed": 2,
 "output": "Processed graph graph_xyz789",
 "metrics": {
 "execution_time_ms": 10000,
 "memory_used_mb": 50
 }
 },
 "error": null,
 "metadata": {
 "version": "1.0.0",
 "priority": 5,
 "timeout": 120
 }
}
```

**Use Case:** Poll for execution completion, retrieve results.

---

### 5.3 Governance Endpoints

#### POST /proposals/create

**Purpose:** Create a governance proposal for graph changes.

**Authentication:** Bearer token or API Key (requires `govern` role)

**Request:**
```json
{
 "title": "Add new ML pipeline node",
 "description": "Proposal to add a new feature extraction node to the ML pipeline",
 "graph": {
 "id": "ml-pipeline-v2",
 "type": "Graph",
 "nodes": [
 {"id": "feature_extractor", "type": "TRANSFORM"}
 ],
 "edges": []
 }
}
```

**Response (200 OK):**
```json
{
 "id": "prop_2024_00123",
 "status": "created"
}
```

**Use Case:** Propose changes that require governance approval.

---

#### POST /proposals/{proposal_id}/vote

**Purpose:** Vote on a governance proposal.

**Authentication:** Bearer token or API Key

**Request:**
```json
{
 "vote": "for"
}
```

Valid votes: `for`, `against`

**Response (200 OK):**
```json
{
 "success": true
}
```

**Use Case:** Participate in decentralized governance.

---

### 5.4 Reasoning Endpoint

#### POST /api/reason

**Purpose:** Perform reasoning using the unified reasoning engine.

**Authentication:** Bearer token or API Key

**Request:**
```json
{
 "query": "What factors contribute to customer churn?",
 "reasoning_type": "causal"
}
```

Valid reasoning types: `symbolic`, `probabilistic`, `causal`, `analogical`, `unified`

**Response (200 OK):**
```json
{
 "conclusion": "Customer churn is primarily driven by price sensitivity and service quality",
 "confidence": 0.87,
 "reasoning_type": "causal",
 "explanation": "Causal analysis identified three main pathways...",
 "uncertainty": 0.13,
 "metadata": {
 "variables_analyzed": 15,
 "data_points": 10000,
 "method": "do-calculus"
 },
 "safety_status": "approved"
}
```

**Use Case:** Complex analytical queries, decision support.

---

### 5.5 GraphQL Endpoint

#### POST /graphql-core

**Purpose:** Execute GraphQL queries on the Core API.

**Authentication:** Bearer token or API Key

**Request:**
```json
{
 "query": "query { status }"
}
```

**Response (200 OK):**
```json
{
 "message": "GraphQL endpoint (integration pending)",
 "query_preview": "query { status }"
}
```

---

### 5.6 Monitoring

#### GET /metrics-core

**Purpose:** Prometheus metrics for the Core API.

**Authentication:** None

**Response (200 OK):**
```
graphix_nodes_executed 508
graphix_cache_hit_rate 0.67
graphix_success_rate 0.94
graphix_total_latency_ms 1234
graphix_rss_mb 210.5
```

---

#### GET /vulcan/insights

**Purpose:** Get VULCAN integration insights.

**Authentication:** Bearer token or API Key

**Response (200 OK):**
```json
{
 "vulcan_enabled": true,
 "world_model_active": true,
 "reasoning_enabled": true,
 "capabilities": [
 "temporal_reasoning",
 "safety_validation",
 "goal_alignment",
 "proposal_evaluation"
 ]
}
```

---

#### GET /status

**Purpose:** Get comprehensive system status.

**Authentication:** None

**Response (200 OK):**
```json
{
 "status": "active",
 "version": "2.2.0",
 "uptime_seconds": 86400,
 "start_time": "2024-12-13T10:30:00Z",
 "graphs": {
 "submitted": 1500,
 "completed": 1450,
 "failed": 50,
 "executing": 5
 },
 "proposals": {
 "total": 25,
 "open": 3,
 "approved": 20
 },
 "agents": {
 "registered": 15
 },
 "auth": {
 "failures": 12,
 "success": 5000,
 "revoked_tokens": 50
 },
 "requests": {
 "total": 50000,
 "errors": 100
 }
}
```

---

## 6. Common Workflows

### 6.1 Getting Started (Authentication)

```bash
# Step 1: Login to get access token
curl -X POST https://gateway.example.com/auth/login-gateway \
 -H "Content-Type: application/json" \
 -d '{"username": "your-username", "password": "your-password"}'

# Step 2: Use the access token in subsequent requests
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X GET https://gateway.example.com/v1/status \
 -H "Authorization: Bearer $TOKEN"

# Step 3: Refresh token when it expires
curl -X POST https://gateway.example.com/auth/refresh \
 -H "Content-Type: application/json" \
 -d '{"refresh_token": "your-refresh-token"}'
```

### 6.2 Submitting a Graph for Processing

```bash
# Step 1: Authenticate
TOKEN=$(curl -s -X POST https://core-api.example.com/auth/login-core \
 -H "Content-Type: application/json" \
 -d '{"api_key": "your-api-key", "password": "your-password"}' \
 | jq -r '.token')

# Step 2: Submit the graph
RESPONSE=$(curl -s -X POST https://core-api.example.com/graphs/submit \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{
 "graph": {
 "id": "my-graph",
 "type": "Graph",
 "nodes": [
 {"id": "n1", "type": "INPUT", "params": {"value": 10}},
 {"id": "n2", "type": "TRANSFORM", "params": {"operation": "double"}},
 {"id": "n3", "type": "OUTPUT"}
 ],
 "edges": [
 {"from": "n1", "to": "n2"},
 {"from": "n2", "to": "n3"}
 ]
 },
 "priority": 5,
 "timeout": 60
 }')

GRAPH_ID=$(echo $RESPONSE | jq -r '.graph_id')
echo "Submitted graph: $GRAPH_ID"

# Step 3: Poll for completion
while true; do
 STATUS=$(curl -s -X GET "https://core-api.example.com/graphs/$GRAPH_ID" \
 -H "Authorization: Bearer $TOKEN" | jq -r '.status')
 
 echo "Status: $STATUS"
 
 if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
 break
 fi
 
 sleep 5
done

# Step 4: Get results
curl -s -X GET "https://core-api.example.com/graphs/$GRAPH_ID" \
 -H "Authorization: Bearer $TOKEN"
```

### 6.3 Using the Reasoning API

```bash
# Symbolic reasoning
curl -X POST https://gateway.example.com/v1/reason \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{
 "query": "IF temperature > 30 AND humidity > 80 THEN comfort = low",
 "type": "symbolic"
 }'

# Causal reasoning
curl -X POST https://gateway.example.com/v1/reason \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{
 "query": "estimate_effect",
 "type": "causal",
 "treatment": "price_reduction",
 "outcome": "sales_volume"
 }'

# Probabilistic reasoning
curl -X POST https://gateway.example.com/v1/reason \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{
 "query": "predict_with_uncertainty",
 "type": "probabilistic",
 "input": [0.8, 0.6, 0.9, 0.7]
 }'
```

### 6.4 Querying Memory/Insights

```bash
# Store knowledge
curl -X POST https://gateway.example.com/v1/memory/store \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{
 "content": "Machine learning models should be validated before deployment",
 "metadata": {"category": "best-practices", "source": "internal"}
 }'

# Search knowledge
curl -X GET "https://gateway.example.com/v1/memory/search?q=deployment+validation&k=5" \
 -H "Authorization: Bearer $TOKEN"
```

### 6.5 Creating and Voting on Proposals

```bash
# Create a proposal (requires 'govern' role)
curl -X POST https://core-api.example.com/proposals/create \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{
 "title": "Add data validation node",
 "description": "Proposal to add input validation to the ML pipeline",
 "graph": {
 "id": "validation-node",
 "type": "Graph",
 "nodes": [{"id": "validator", "type": "VALIDATE"}],
 "edges": []
 }
 }'

# Vote on a proposal
curl -X POST https://core-api.example.com/proposals/prop_2024_00123/vote \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{"vote": "for"}'
```

---

## 7. Error Handling

### 7.1 Standard Error Format

All API endpoints return errors in a consistent format:

```json
{
 "error": "Human-readable error message",
 "code": 400,
 "timestamp": "2024-12-14T10:30:00.000Z",
 "request_id": "req_abc123xyz"
}
```

### 7.2 HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data or parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions/scope |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | Request body exceeds size limit |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable |

### 7.3 Error Types

| Error Type | Description | Resolution |
|------------|-------------|------------|
| `ValidationError` | Invalid request data | Check request format and required fields |
| `AuthenticationError` | Auth failure | Check credentials, refresh token |
| `AuthorizationError` | Permission denied | Check user roles and scopes |
| `RateLimitError` | Too many requests | Wait and retry with backoff |
| `TimeoutError` | Request timeout | Increase timeout or optimize request |
| `GraphValidationError` | Invalid graph structure | Check node IDs, edges, cycles |

### 7.4 Handling Errors (Best Practices)

```python
import requests
import time

def make_request_with_retry(url, headers, data, max_retries=3):
 for attempt in range(max_retries):
 response = requests.post(url, headers=headers, json=data)
 
 if response.status_code == 200:
 return response.json()
 
 if response.status_code == 429:
 # Rate limited - wait and retry
 retry_after = int(response.headers.get('Retry-After', 60))
 time.sleep(retry_after)
 continue
 
 if response.status_code == 401:
 # Token expired - refresh and retry
 headers['Authorization'] = f"Bearer {refresh_token()}"
 continue
 
 if response.status_code >= 500:
 # Server error - exponential backoff
 time.sleep(2 ** attempt)
 continue
 
 # Client error - don't retry
 raise Exception(f"Error {response.status_code}: {response.json()}")
 
 raise Exception("Max retries exceeded")
```

---

## 8. Rate Limiting

### 8.1 Default Limits

| Endpoint Category | Limit | Window | Scope |
|------------------|-------|--------|-------|
| Authentication | 10/minute | Per IP | Global |
| General API | 100/minute | Per user | User |
| Memory Operations | 50/minute | Per endpoint | Endpoint |
| Heavy Processing | 20/minute | Per user | User |
| Health/Metrics | Unlimited | - | - |

### 8.2 Rate Limit Headers

Responses include rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702540860
```

### 8.3 Rate Limit Response

```json
{
 "error": "Rate limit exceeded",
 "retry_after_seconds": 30
}
```

---

## 9. Security Headers

All responses include security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Content-Security-Policy: default-src 'none'
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Cache-Control: no-store
X-Request-ID: req_abc123xyz
```

---

## 10. GraphQL API

### 10.1 Schema

```graphql
type Query {
 status: String!
 memorySearch(query: String!, k: Int = 10): [String!]!
}

type Mutation {
 process(input: String!): String!
}
```

### 10.2 Example Queries

```graphql
# Query status
query {
 status
}

# Search memory
query SearchMemory($q: String!, $limit: Int) {
 memorySearch(query: $q, k: $limit)
}

# Process input
mutation ProcessInput($input: String!) {
 process(input: $input)
}
```

---

## 11. WebSocket API

### 11.1 Connection

```javascript
const ws = new WebSocket('wss://gateway.example.com/ws?token=YOUR_ACCESS_TOKEN');

ws.onopen = () => {
 console.log('Connected');
 
 // Subscribe to topics
 ws.send(JSON.stringify({
 type: 'subscribe',
 topics: ['execution', 'proposals']
 }));
};

ws.onmessage = (event) => {
 const data = JSON.parse(event.data);
 console.log('Received:', data);
};
```

### 11.2 Message Types

**Subscribe:**
```json
{
 "type": "subscribe",
 "topics": ["execution", "proposals", "system"]
}
```

**Process:**
```json
{
 "type": "process",
 "input": "Your input data"
}
```

**Response:**
```json
{
 "type": "result",
 "data": {
 "embedding": [...],
 "modality": "text"
 }
}
```

### 11.3 Rate Limits

- Maximum connections: 10,000
- Messages per connection: 20/second burst, 10/second sustained

---

## 12. Appendix

### 12.1 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Secret key for JWT signing | Auto-generated |
| `JWT_PRIVATE_KEY` | Private key for RS256 | None |
| `JWT_PUBLIC_KEY` | Public key for RS256 | None |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379 |
| `ACCESS_TOKEN_TTL_MIN` | Access token lifetime | 30 |
| `REFRESH_TOKEN_TTL_DAYS` | Refresh token lifetime | 7 |
| `ALLOWED_ORIGINS` | CORS allowed origins | localhost:3000 |
| `GRAPHIX_JWT_SECRET` | Core API JWT secret | Required |
| `GRAPHIX_API_PORT` | Core API port | 8000 |

### 12.2 SDKs and Libraries

- **Python SDK**: `pip install vulcanami-sdk` (coming soon)
- **JavaScript SDK**: `npm install vulcanami-sdk` (coming soon)
- **OpenAPI Spec**: Available at `/api/swagger.yml`

### 12.3 Support

- Enterprise support: Contact your Novatrax account team
- Documentation: https://docs.novatrax.io
- API Status: https://status.novatrax.io

---

**Copyright © 2024 Novatrax Labs LTD. All rights reserved.**
