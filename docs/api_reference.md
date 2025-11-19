# API Reference (Registry & Arena)

## 1. Authentication
JWT (Registry)  
Header: `Authorization: Bearer <token>`  
Claims: sub, trust_level, scopes, iat, exp  

API Key (Arena)  
Header: `X-API-Key: <key>`  

## 2. Common JSON Conventions
- RFC3339 timestamps
- Pagination: limit, next_cursor (future)
- Risk scores normalized [0.0, 1.0]

## 3. Registry Endpoints
| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| POST | /registry/bootstrap | Initialize first agent | Bootstrap Key |
| POST | /auth/login | Issue JWT | None |
| POST | /registry/onboard | Add agent w/trust | JWT |
| POST | /ir/propose | Submit graph/ontology proposal | JWT |
| GET | /audit/logs | Retrieve audit entries | JWT |
| GET | /health | Liveness check | None |
| GET | /metrics | Prometheus metrics | Optional |

### Proposal Submission Example
```json
{
  "nodes": [
    { "id": "c1", "type": "CONST", "params": { "value": 10 } }
  ],
  "edges": [],
  "metadata": {
    "authors": ["agent_alpha"],
    "version": "1.0.0",
    "description": "Basic constant graph"
  }
}
```

### Response
```json
{
  "proposal_id": "prop_2025_00123",
  "status": "draft",
  "hash": "sha256:de...ad",
  "risk": { "score": 0.11, "factors": { "node_count": 1 } },
  "created_at": "2025-11-11T03:35:00Z"
}
```

## 4. Arena Endpoints
| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| GET | /health | Liveness | Optional |
| GET | /ready | Dependency readiness | Optional |
| POST | /execute/graph | Execute validated graph artifact | API Key |
| GET | /metrics | Metrics exposure | API Key |

Execution Request (illustrative):
```json
{
  "graph_id": "graph_abc123",
  "artifact": {
    "nodes": [...],
    "edges": [...],
    "metadata": { "version": "1.0.2" }
  },
  "execution_profile": {
    "max_concurrency": 8,
    "timeout_seconds": 120,
    "retry_policy": { "max_retries": 2, "backoff_factor": 2.0 }
  }
}
```

## 5. Error Schema
```json
{
  "error": {
    "type": "ValidationError",
    "message": "Missing 'nodes' key",
    "details": { "path": "$" },
    "severity": "warning",
    "retryable": false,
    "request_id": "req_123",
    "timestamp": "2025-11-11T03:40:00Z"
  }
}
```

Extended fields: remediation, correlation_id (trace), risk_context.

## 6. Rate Limiting
Headers (future):
- X-RateLimit-Limit
- X-RateLimit-Remaining
- X-RateLimit-Reset

Governance throttle: MAX_PROPOSALS_PER_AGENT_PER_HOUR, similarity damping.

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
