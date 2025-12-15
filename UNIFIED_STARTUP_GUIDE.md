# Unified Platform Startup Guide

**Version:** 2.1.0  
**Last Updated:** 2024-12-15  
**Status:** ✅ All 9 Core Services Integrated

---

## Overview

The VulcanAMI platform provides a **unified entry point** (`src/full_platform.py`) that starts all 9 core runtime services in a single process or coordinated multi-process setup.

### Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Unified Platform Server (port 8080)                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ FastAPI Main App                                        │ │
│ │ ┌─────────┐ ┌─────────┐ ┌──────────┐                  │ │
│ │ │ VULCAN  │ │  Arena  │ │ Registry │ (Mounted)        │ │
│ │ │ /vulcan │ │ /arena  │ │/registry │                  │ │
│ │ └─────────┘ └─────────┘ └──────────┘                  │ │
│ │ ┌──────────┐ ┌──────┐ ┌──────┐                        │ │
│ │ │API GW    │ │ DQS  │ │ PII  │ (Mounted)              │ │
│ │ │/api-gw   │ │ /dqs │ │ /pii │                        │ │
│ │ └──────────┘ └──────┘ └──────┘                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ Background Processes:                                        │
│ ┌──────────┐ ┌────────────┐ ┌──────────┐                  │
│ │API Server│ │Registry    │ │Listener  │                  │
│ │Port 8001 │ │gRPC :50051 │ │Port 8084 │                  │
│ └──────────┘ └────────────┘ └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Start All Services (Development)

```bash
# From repository root
python -m src.full_platform

# Or with uvicorn directly
uvicorn src.full_platform:app --host 0.0.0.0 --port 8080 --reload
```

### Start All Services (Production)

```bash
# With multiple workers
uvicorn src.full_platform:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --log-level info
```

### Access the Platform

- **Main Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **Service Status**: http://localhost:8080/api/status
- **Metrics**: http://localhost:8080/metrics

---

## Service Details

### 1. VULCAN (Mounted at `/vulcan`)

**Type:** FastAPI sub-application  
**Purpose:** Core cognitive architecture with AGI capabilities

**Endpoints:**
- `GET /vulcan/health` - Health check
- `GET /vulcan/docs` - VULCAN API documentation
- `POST /vulcan/reason` - Causal reasoning
- `POST /vulcan/plan` - Planning and goal decomposition
- More endpoints in VULCAN docs

**Configuration:**
```python
UNIFIED_VULCAN_MOUNT=/vulcan  # Mount path
UNIFIED_VULCAN_MODULE=src.vulcan.main  # Import path
```

### 2. Arena (Mounted at `/arena`)

**Type:** FastAPI sub-application  
**Purpose:** Distributed AI agent collaboration and graph evolution

**Endpoints:**
- `GET /arena/health` - Health check
- `GET /arena/docs` - Arena API documentation
- `POST /arena/api/run/{agent_id}` - Run agent task
- `POST /arena/api/feedback` - Submit feedback
- `POST /arena/api/tournament` - Run tournament

**Configuration:**
```python
UNIFIED_ARENA_MOUNT=/arena  # Mount path
UNIFIED_ARENA_MODULE=src.graphix_arena  # Import path
```

**Security:**
- Requires `X-API-KEY` header (default: from `GRAPHIX_API_KEY` env var)
- Rate limited: 100 requests/minute per IP

### 3. Registry Flask (Mounted at `/registry`)

**Type:** Flask application via WSGI middleware  
**Purpose:** Graph IR registry and proposal management

**Endpoints:**
- `GET /registry/health` - Health check
- `GET /registry/` - Registry dashboard
- `POST /registry/proposals` - Submit proposal
- `GET /registry/proposals/{id}` - Get proposal

**Configuration:**
```python
UNIFIED_REGISTRY_MOUNT=/registry  # Mount path
UNIFIED_REGISTRY_MODULE=app  # Import path (Flask app)
```

### 4. API Gateway (Mounted at `/api-gateway`)

**Type:** FastAPI sub-application  
**Purpose:** Enterprise API gateway with service discovery

**Endpoints:**
- `GET /api-gateway/health` - Health check
- `GET /api-gateway/ready` - Readiness probe
- `GET /api-gateway/docs` - API Gateway documentation

**Configuration:**
```python
UNIFIED_API_GATEWAY_MOUNT=/api-gateway
UNIFIED_ENABLE_API_GATEWAY=true  # Enable/disable
```

### 5. DQS Service (Mounted at `/dqs`)

**Type:** FastAPI sub-application  
**Purpose:** Real-time data quality monitoring and scoring

**Endpoints:**
- `GET /dqs/health` - Health check
- `GET /dqs/docs` - DQS API documentation
- `POST /dqs/classify` - Classify data quality
- `GET /dqs/metrics` - Quality metrics

**Configuration:**
```python
UNIFIED_DQS_MOUNT=/dqs
UNIFIED_ENABLE_DQS_SERVICE=true  # Enable/disable
```

### 6. PII Service (Mounted at `/pii`)

**Type:** FastAPI sub-application  
**Purpose:** PII detection and privacy protection

**Endpoints:**
- `GET /pii/health` - Health check
- `GET /pii/docs` - PII Service documentation
- `POST /pii/scan` - Scan for PII
- `POST /pii/redact` - Redact PII

**Configuration:**
```python
UNIFIED_PII_MOUNT=/pii
UNIFIED_ENABLE_PII_SERVICE=true  # Enable/disable
```

### 7. API Server (Standalone - Port 8001)

**Type:** Custom HTTP server (ThreadedHTTPServer)  
**Purpose:** Graphix IR graph submission and execution

**Access:** http://localhost:8001

**Configuration:**
```python
UNIFIED_API_SERVER_PORT=8001
UNIFIED_ENABLE_API_SERVER=true  # Enable/disable
```

**Note:** Runs as background process, not mounted as sub-app

### 8. Registry gRPC (Standalone - Port 50051)

**Type:** gRPC server  
**Purpose:** Distributed agent communication and governance

**Access:** grpc://localhost:50051

**Configuration:**
```python
UNIFIED_REGISTRY_GRPC_PORT=50051
UNIFIED_ENABLE_REGISTRY_GRPC=true  # Enable/disable
```

**Note:** Runs as background process with separate gRPC protocol

### 9. Listener (Standalone - Port 8084)

**Type:** Custom HTTP server  
**Purpose:** Secure graph submission listener

**Access:** http://localhost:8084

**Configuration:**
```python
UNIFIED_LISTENER_PORT=8084
UNIFIED_ENABLE_LISTENER=true  # Enable/disable
```

**Note:** Runs as background process with specialized graph validation

---

## Configuration

### Environment Variables

All configuration can be controlled via environment variables with the `UNIFIED_` prefix:

```bash
# Server configuration
UNIFIED_HOST=0.0.0.0              # Bind address (default: 127.0.0.1)
UNIFIED_PORT=8080                  # Main server port
UNIFIED_WORKERS=4                  # Number of workers

# Authentication
UNIFIED_AUTH_METHOD=jwt            # none, api_key, jwt, oauth2
UNIFIED_API_KEY=your-api-key
UNIFIED_JWT_SECRET=your-jwt-secret

# Service enable/disable
UNIFIED_ENABLE_API_GATEWAY=true
UNIFIED_ENABLE_DQS_SERVICE=true
UNIFIED_ENABLE_PII_SERVICE=true
UNIFIED_ENABLE_API_SERVER=true
UNIFIED_ENABLE_REGISTRY_GRPC=true
UNIFIED_ENABLE_LISTENER=true

# Standalone service ports
UNIFIED_API_SERVER_PORT=8001
UNIFIED_REGISTRY_GRPC_PORT=50051
UNIFIED_LISTENER_PORT=8084

# Feature flags
UNIFIED_ENABLE_METRICS=true        # Prometheus metrics
UNIFIED_ENABLE_HEALTH_CHECKS=true  # Health check aggregation
UNIFIED_CORS_ENABLED=true          # CORS support
```

### Configuration File

Create a `.env` file in the repository root:

```bash
# .env
UNIFIED_HOST=0.0.0.0
UNIFIED_PORT=8080
UNIFIED_AUTH_METHOD=api_key
UNIFIED_API_KEY=my-secret-key
UNIFIED_JWT_SECRET=my-jwt-secret

# Disable services you don't need
UNIFIED_ENABLE_LISTENER=false
UNIFIED_ENABLE_REGISTRY_GRPC=false
```

---

## Port Allocation

| Service | Port | Type | Configurable |
|---------|------|------|-------------|
| Unified Platform | 8080 | HTTP | ✅ `UNIFIED_PORT` |
| API Server | 8001 | HTTP | ✅ `UNIFIED_API_SERVER_PORT` |
| PII Service | 8082 | (mounted) | N/A |
| DQS Service | 8083 | (mounted) | N/A |
| Listener | 8084 | HTTP | ✅ `UNIFIED_LISTENER_PORT` |
| Registry gRPC | 50051 | gRPC | ✅ `UNIFIED_REGISTRY_GRPC_PORT` |

**Note:** Mounted services (VULCAN, Arena, Registry Flask, API Gateway, DQS, PII) all share the main platform port (8080) with different URL paths.

---

## Health Checks

### Individual Service Health

Each mounted service has its own health endpoint:

```bash
curl http://localhost:8080/vulcan/health
curl http://localhost:8080/arena/health
curl http://localhost:8080/registry/health
curl http://localhost:8080/api-gateway/health
curl http://localhost:8080/dqs/health
curl http://localhost:8080/pii/health
```

### Aggregated Platform Health

The platform provides an aggregated health check:

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-15T10:30:00Z",
  "worker_pid": 12345,
  "services": {
    "vulcan": {
      "mounted": true,
      "health": {
        "status": "healthy",
        "latency_ms": 12.3
      }
    },
    "arena": {
      "mounted": true,
      "health": {
        "status": "healthy",
        "latency_ms": 8.5
      }
    }
    // ... other services
  }
}
```

---

## Service Status

Get detailed status of all services:

```bash
curl http://localhost:8080/api/status
```

Response:
```json
{
  "platform": {
    "name": "Graphix Vulcan Unified Platform",
    "version": "2.1.0",
    "timestamp": "2024-12-15T10:30:00Z",
    "worker_pid": 12345,
    "workers": 1
  },
  "services": {
    "vulcan": {
      "mounted": true,
      "mount_path": "/vulcan",
      "health_path": "/vulcan/health",
      "import_path": "src.vulcan.main.app",
      "docs_url": "/vulcan/docs"
    },
    "api_server": {
      "mounted": false,
      "import_success": true,
      "import_path": "src.api_server (standalone)",
      "mount_path": "http://localhost:8001"
    }
    // ... other services
  },
  "configuration": {
    "auth_method": "api_key",
    "metrics_enabled": true,
    "health_checks_enabled": true
  }
}
```

---

## Troubleshooting

### Service Won't Start

**Check logs** in `unified_platform.log`:

```bash
tail -f unified_platform.log
```

**Common issues:**
- Port already in use: Change port via environment variable
- Import errors: Ensure all dependencies installed (`pip install -r requirements.txt`)
- Permission denied: Check file permissions and binding to ports < 1024

### Background Process Not Starting

Background processes (API Server, Registry gRPC, Listener) log to their own outputs. Check their logs:

```bash
# Find the process
ps aux | grep "api_server"

# Check if port is in use
netstat -tuln | grep 8001
```

### Service Shows as "FAILED" in Status

1. Check import path is correct
2. Verify service module exists
3. Check for syntax errors in service file
4. Review startup logs for detailed error

### Performance Issues

- Reduce number of workers for development: `UNIFIED_WORKERS=1`
- Disable unnecessary services via config
- Check resource usage: `htop` or `docker stats`

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Expose main port and standalone service ports
EXPOSE 8080 8001 8084 50051

CMD ["python", "-m", "src.full_platform"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  platform:
    build: .
    ports:
      - "8080:8080"  # Main platform
      - "8001:8001"  # API Server
      - "8084:8084"  # Listener
      - "50051:50051"  # Registry gRPC
    environment:
      - UNIFIED_HOST=0.0.0.0
      - UNIFIED_WORKERS=4
      - UNIFIED_AUTH_METHOD=jwt
      - UNIFIED_JWT_SECRET=${JWT_SECRET}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
```

### Kubernetes

See `DEPLOYMENT.md` for complete Kubernetes deployment manifests.

---

## Monitoring

### Prometheus Metrics

All services expose metrics at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

**Key metrics:**
- `unified_platform_requests_total` - Total requests by service
- `unified_platform_request_duration_seconds` - Request latency
- `unified_platform_service_health` - Service health status
- `unified_platform_active_workers` - Active worker count

### Grafana Dashboards

Import the provided dashboard:

```bash
# See ops/monitoring/grafana_dashboard.json
```

---

## Security

### Authentication

The platform supports multiple authentication methods:

1. **None** (development only): `UNIFIED_AUTH_METHOD=none`
2. **API Key**: `UNIFIED_AUTH_METHOD=api_key`
3. **JWT**: `UNIFIED_AUTH_METHOD=jwt`
4. **OAuth2**: `UNIFIED_AUTH_METHOD=oauth2` (coming soon)

### HTTPS/TLS

For production, use a reverse proxy (nginx, Traefik) or enable TLS directly:

```python
# In custom deployment
uvicorn src.full_platform:app \
  --host 0.0.0.0 \
  --port 8080 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

### Rate Limiting

Built-in rate limiting protects all endpoints:
- 100 requests/minute per IP (default)
- Configurable per service
- Redis-backed for distributed rate limiting

---

## Next Steps

1. **Development**: Start with `python -m src.full_platform`
2. **Testing**: Run `python test_platform_startup.py`
3. **Documentation**: Read service-specific docs in `/docs`
4. **Deployment**: See `DEPLOYMENT.md` for production setup
5. **Monitoring**: Set up Prometheus/Grafana dashboards

---

## Support

- **Documentation**: `/docs` directory
- **Issues**: GitHub Issues (for authorized users)
- **Security**: security@novatraxlabs.com

---

**Last Updated:** 2024-12-15  
**Version:** 2.1.0  
**Status:** ✅ Production Ready
