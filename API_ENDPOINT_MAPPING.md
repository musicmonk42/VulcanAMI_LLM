# API Endpoint Mapping - Frontend to Backend

This document maps the API endpoints called by the frontend (`index.html`) to the backend implementations.

## Platform Architecture

- **Full Platform**: `src/full_platform.py` (runs on port 8080 by default)
- **Vulcan Sub-App**: `src/vulcan/main.py` (mounted at `/vulcan`)
- **Arena Sub-App**: Arena components (mounted at `/arena`)
- **Registry Sub-App**: Registry components (mounted at `/registry`)

## Endpoint Mappings

### Arena Endpoints (Full Platform)

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `POST /api/arena/run/{agentId}` | `POST /api/arena/run/{agent_id}` | `src/full_platform.py:1547` | ✅ Fixed |

**Note**: Frontend was calling `/arena/run/` but has been updated to `/api/arena/run/` to match the backend.

### VULCAN Endpoints (Vulcan Sub-App)

All VULCAN endpoints are prefixed with `/vulcan` when accessed from the frontend because the Vulcan app is mounted at that path in `full_platform.py`.

#### Orchestrator Endpoints

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `GET /vulcan/orchestrator/agents/status` | `GET /orchestrator/agents/status` | `src/vulcan/main.py:1520` | ✅ Added |
| `POST /vulcan/orchestrator/agents/spawn` | `POST /orchestrator/agents/spawn` | `src/vulcan/main.py:1559` | ✅ Added |

#### World Model Endpoints

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `GET /vulcan/world-model/status` | `GET /world-model/status` | `src/vulcan/main.py:1591` | ✅ Added |
| `POST /vulcan/world-model/intervene` | `POST /world-model/intervene` | `src/vulcan/main.py:1626` | ✅ Added |
| `POST /vulcan/world-model/predict` | `POST /world-model/predict` | `src/vulcan/main.py:1654` | ✅ Added |

#### Safety Endpoints

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `GET /vulcan/safety/status` | `GET /safety/status` | `src/vulcan/main.py:1686` | ✅ Added |
| `POST /vulcan/safety/validate` | `POST /safety/validate` | `src/vulcan/main.py:1719` | ✅ Added |
| `GET /vulcan/safety/audit/recent?limit=20` | `GET /safety/audit/recent` | `src/vulcan/main.py:1746` | ✅ Added |

#### Self-Improvement Endpoints

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `GET /vulcan/self-improvement/objectives` | `GET /self-improvement/objectives` | `src/vulcan/main.py:1787` | ✅ Added |

#### Transparency Endpoints

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `POST /vulcan/transparency/query` | `POST /transparency/query` | `src/vulcan/main.py:1828` | ✅ Added |

#### Memory Endpoints

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `GET /vulcan/memory/status` | `GET /memory/status` | `src/vulcan/main.py:1872` | ✅ Added |
| `POST /vulcan/memory/search` | `POST /v1/memory/search` | `src/vulcan/main.py:967` | ✅ Exists |
| `POST /vulcan/memory/store` | `POST /memory/store` | `src/vulcan/main.py:1904` | ✅ Added |

**Note**: Memory search endpoint uses `/v1/memory/search` path which is the existing endpoint.

#### Hardware Endpoints

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `GET /vulcan/hardware/status` | `GET /hardware/status` | `src/vulcan/main.py:1936` | ✅ Added |

#### LLM Endpoints (Already Existing)

| Frontend Call | Backend Endpoint | File | Status |
|--------------|------------------|------|--------|
| `POST /vulcan/llm/chat` | `POST /llm/chat` | `src/vulcan/main.py:1306` | ✅ Exists |
| `POST /vulcan/llm/reason` | `POST /llm/reason` | `src/vulcan/main.py:1327` | ✅ Exists |

## CORS Configuration

Updated CORS configuration in `src/full_platform.py:179-188` to allow requests from:

- `http://localhost:3000` (default React dev server)
- `http://127.0.0.1:3000`
- `http://localhost:8000` (platform server)
- `http://127.0.0.1:8000`
- `null` (required for file:// URLs when opening HTML files directly)

## Request/Response Schemas

### Common Response Format

Most endpoints return JSON with the following general structure:

```json
{
  "status": "success",
  "data": { ... },
  "timestamp": 1234567890
}
```

Error responses follow FastAPI standard:

```json
{
  "detail": "Error message"
}
```

### Specific Endpoint Schemas

#### GET /orchestrator/agents/status

**Response:**
```json
{
  "total_agents": 0,
  "state_distribution": {
    "idle": 0,
    "busy": 0,
    "error": 0
  },
  "pending_tasks": 0,
  "capability_distribution": {},
  "statistics": {
    "total_jobs_submitted": 0,
    "total_jobs_completed": 0,
    "total_jobs_failed": 0,
    "total_recoveries_successful": 0
  }
}
```

#### POST /orchestrator/agents/spawn

**Request:**
```json
{
  "capability": "general"
}
```

**Response:**
```json
{
  "status": "spawned",
  "agent_id": "agent_123",
  "capability": "general"
}
```

#### GET /world-model/status

**Response:**
```json
{
  "active": true,
  "entities_tracked": 0,
  "relationships_tracked": 0,
  "prediction_accuracy": 0.85,
  "last_update": 1234567890
}
```

#### POST /world-model/predict

**Request:**
```json
{
  "query": "What will happen if X?",
  "evidence": {}
}
```

**Response:**
```json
{
  "prediction": "...",
  "confidence": 0.85
}
```

#### GET /safety/status

**Response:**
```json
{
  "safety_score": 0.95,
  "violations_detected": 0,
  "violations_prevented": 0,
  "audit_entries": 0,
  "monitoring_active": true
}
```

#### POST /safety/validate

**Request:**
```json
{
  "action": {
    "type": "...",
    "parameters": {}
  }
}
```

**Response:**
```json
{
  "safe": true,
  "reason": "Action meets safety requirements",
  "action": { ... }
}
```

#### GET /memory/status

**Response:**
```json
{
  "total_memories": 0,
  "memory_usage_mb": 0,
  "retrieval_latency_ms": 0,
  "storage_backend": "in-memory"
}
```

#### POST /memory/store

**Request:**
```json
{
  "content": "...",
  "metadata": {}
}
```

**Response:**
```json
{
  "status": "stored",
  "memory_id": "mem_123"
}
```

#### GET /hardware/status

**Response:**
```json
{
  "cpu_usage_percent": 0,
  "memory_usage_mb": 0,
  "disk_usage_percent": 0,
  "gpu_available": false,
  "gpu_usage_percent": 0,
  "energy_budget_left_nJ": 0
}
```

## Testing the Integration

### Start the Platform

```bash
cd /home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM
python3 src/full_platform.py
```

The platform will start on `http://localhost:8080` by default.

### Open the Frontend

Open `index.html` in a web browser. The frontend will attempt to connect to the backend at the configured `baseUrl`.

### Test Each Feature

1. **Connect**: Click "Connect to Server" button
2. **Health Check**: Should succeed if backend is running
3. **Orchestrator**: Load agent pool status, spawn agents
4. **World Model**: Check status, make predictions
5. **Safety**: Validate actions, view audit logs
6. **Memory**: Search, store, check status
7. **Hardware**: View system status
8. **LLM**: Chat and reasoning features

## Known Limitations

1. Some endpoints provide simulated/mock data when the underlying subsystem is not fully initialized
2. The deployment must be properly initialized in `full_platform.py` for all features to work
3. Error handling returns graceful degradation messages when subsystems are unavailable

## Troubleshooting

### CORS Errors

If you see CORS errors in the browser console:
- Ensure `cors_enabled: true` in settings
- Verify the frontend is accessing from an allowed origin
- Check browser console for the actual origin being used

### 503 Service Unavailable

If endpoints return 503:
- Check that `full_platform.py` successfully mounted the Vulcan sub-app
- Look for startup errors in the server logs
- Verify the deployment initialized successfully

### Connection Refused

If the frontend cannot connect:
- Ensure the backend is running
- Check the `baseUrl` in `index.html` matches the server address
- Verify the server port (default 8080)
