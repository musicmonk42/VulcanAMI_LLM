# ⚠️ DEPRECATED

**This document has been consolidated.** 
**Archived:** December 23, 2024

## Migration Path

For PLATFORM_SERVICES_INVENTORY.md content → See [COMPLETE_SERVICE_CATALOG.md](../../COMPLETE_SERVICE_CATALOG.md)

---

# VulcanAMI Platform Services Inventory & Startup Analysis

**Generated:** 2024-12-15 
**Purpose:** Complete inventory of all platform services and their startup mechanisms

---

## Executive Summary

The VulcanAMI platform consists of **multiple service layers** that need to be coordinated for complete platform startup:

- **3 Core Entry Points**: `full_platform.py`, `app.py`, `main.py`
- **3 Mounted Sub-Services**: VULCAN, Arena, Registry (Flask)
- **6 Standalone Services**: API Gateway, API Server, DQS, PII, Registry gRPC, Listener
- **71 Total Functional Services**: Per COMPLETE_SERVICE_CATALOG.md

**Current Issue**: The unified entry point (`full_platform.py`) only mounts 3 sub-applications but doesn't start 6 additional standalone services that have their own server processes.

---

## 1. Current Platform Startup (full_platform.py)

### 1.1 What Currently Starts

**Main Server**: FastAPI application on port 8080 (configurable via `UNIFIED_PORT`)

**Mounted Sub-Applications**:
1. **VULCAN** - Mounted at `/vulcan`
 - Source: `src/vulcan/main.py`
 - Type: FastAPI app
 - Initialization: Manual deployment setup in lifespan
 
2. **Arena** - Mounted at `/arena`
 - Source: `src/graphix_arena.py`
 - Type: FastAPI app
 - Initialization: Routes registered at import time
 
3. **Registry** - Mounted at `/registry`
 - Source: `app.py`
 - Type: Flask app (via WSGI middleware)
 - Port: Runs under unified platform (not standalone)

### 1.2 Service Manager

The `AsyncServiceManager` class handles:
- Service import with fallback strategies
- Mounting services to main FastAPI app
- Health check aggregation
- Status reporting

---

## 2. Standalone Services NOT Started by full_platform.py

### 2.1 API Gateway Service

**File**: `src/api_gateway.py` 
**Type**: FastAPI 
**Default Port**: 8000 (ENV: `API_PORT`) 
**Purpose**: Enterprise API gateway with VULCAN AMI integration

**Features**:
- Service discovery and routing
- Health and readiness probes
- Prometheus metrics
- Rate limiting
- CORS support

**Current Status**: ❌ NOT STARTED by full_platform.py

### 2.2 API Server (Graphix)

**File**: `src/api_server.py` 
**Type**: HTTP server (custom ThreadedHTTPServer) 
**Default Port**: Variable (configurable) 
**Purpose**: Graphix IR graph submission and execution

**Features**:
- JWT authentication
- Graph execution engine
- Request size limiting
- Rate limiting
- Audit logging

**Current Status**: ❌ NOT STARTED by full_platform.py

### 2.3 Data Quality Service (DQS)

**File**: `src/dqs_service.py` 
**Type**: FastAPI 
**Default Port**: 8080 (ENV: `DQS_PORT`) ⚠️ **CONFLICTS with full_platform.py** 
**Purpose**: Real-time data quality monitoring and scoring

**Features**:
- DQS classification engine integration
- Quality metrics and reporting
- Prometheus metrics
- Rate limiting

**Current Status**: ❌ NOT STARTED by full_platform.py 
**Issue**: Port 8080 conflicts with full_platform.py default

### 2.4 PII Service

**File**: `src/pii_service.py` 
**Type**: FastAPI 
**Default Port**: 8082 (ENV: `PII_PORT`) 
**Purpose**: PII detection and privacy protection

**Features**:
- PII scanning
- Privacy compliance
- Redaction capabilities
- Rate limiting

**Current Status**: ❌ NOT STARTED by full_platform.py

### 2.5 Registry API Server (gRPC)

**File**: `src/governance/registry_api_server.py` 
**Type**: gRPC server 
**Default Port**: 50051 (ENV: `REGISTRY_PORT`) 
**Purpose**: Graphix IR Registry for distributed agent communication

**Features**:
- Agent authentication
- Proposal submission and voting
- Validation workflows
- Audit logging

**Current Status**: ❌ NOT STARTED by full_platform.py

### 2.6 Listener Service

**File**: `src/listener.py` 
**Type**: HTTP server (custom) 
**Default Port**: 8181 (`--port` arg) ⚠️ **CONFLICTS with Arena** 
**Purpose**: Secure HTTP listener for Graphix IR graph submission

**Features**:
- Agent signature verification
- Graph validation
- Rate limiting
- Graceful shutdown

**Current Status**: ❌ NOT STARTED by full_platform.py 
**Issue**: Port 8181 conflicts with Arena default

---

## 3. Port Conflict Matrix

| Service | Default Port | Environment Variable | Conflicts With |
|---------|-------------|---------------------|----------------|
| full_platform.py | 8080 | UNIFIED_PORT | dqs_service.py |
| app.py (Registry Flask) | 5000 | PORT | - |
| graphix_arena.py | 8181 | (constructor arg) | listener.py |
| api_gateway.py | 8000 | API_PORT | - |
| api_server.py | Variable | - | - |
| dqs_service.py | 8080 | DQS_PORT | **full_platform.py** |
| pii_service.py | 8082 | PII_PORT | - |
| registry_api_server.py | 50051 | REGISTRY_PORT | - |
| listener.py | 8181 | --port | **graphix_arena.py** |

**Critical Conflicts**:
1. **Port 8080**: full_platform.py vs dqs_service.py
2. **Port 8181**: graphix_arena.py vs listener.py

---

## 4. Recommended Port Allocation

To avoid conflicts, the following port allocation is recommended:

| Service | Recommended Port | Environment Variable |
|---------|-----------------|---------------------|
| full_platform.py | 8080 | UNIFIED_PORT |
| app.py (standalone) | 5000 | PORT |
| VULCAN (mounted) | - | (sub-app at /vulcan) |
| Arena (mounted) | - | (sub-app at /arena) |
| Registry Flask (mounted) | - | (sub-app at /registry) |
| api_gateway.py | 8000 | API_PORT |
| api_server.py | 8001 | GRAPHIX_API_PORT |
| dqs_service.py | 8083 | DQS_PORT |
| pii_service.py | 8082 | PII_PORT |
| registry_api_server.py (gRPC) | 50051 | REGISTRY_PORT |
| listener.py | 8084 | LISTENER_PORT |

---

## 5. Service Startup Options

### Option A: All-in-One (Recommended for Development)

**Use full_platform.py with sub-app mounting only**
- ✅ Single process
- ✅ Unified logging
- ✅ Shared state
- ✅ Easy debugging
- ❌ Limited scalability

**Services**: VULCAN, Arena, Registry (3 services)

### Option B: Microservices (Recommended for Production)

**Run each service independently with orchestration**
- ✅ Independent scaling
- ✅ Fault isolation
- ✅ Service-specific resources
- ❌ Complex orchestration
- ❌ Distributed state management

**Services**: All 9 services running independently

### Option C: Hybrid (Current State - INCOMPLETE)

**full_platform.py + some standalone services**
- ⚠️ Currently only 3/9 services start
- ⚠️ Port conflicts exist
- ⚠️ No coordination mechanism

---

## 6. 71 Functional Services (from COMPLETE_SERVICE_CATALOG.md)

The platform includes 71 functional service modules with 557 files:

### Core Runtime Services (9)
1. ✅ **full_platform.py** - Unified platform server
2. ✅ **graphix_arena.py** - Arena service
3. ✅ **app.py** - Registry Flask server
4. ❌ **api_gateway.py** - API Gateway
5. ❌ **api_server.py** - Graphix API Server
6. ❌ **dqs_service.py** - Data Quality Service
7. ❌ **pii_service.py** - PII Service
8. ❌ **registry_api_server.py** - Registry gRPC
9. ❌ **listener.py** - Graph Listener

### Compiler Services (5 files, 75 functions)
10. **graph_compiler.py** - LLVM-based graph compilation
11. **hybrid_executor.py** - Hybrid execution engine
12. **llvm_backend.py** - LLVM backend integration

### LLM Core Services (7 files, 125 functions)
13. **graphix_transformer.py** - Custom transformer
14. **graphix_executor.py** - LLM execution engine
15. **ir_attention.py** - Attention mechanisms
16. **ir_feedforward.py** - Feed-forward layers
17. **ir_layer_norm.py** - Layer normalization
18. **ir_embeddings.py** - Embedding layers
19. **persistant_context.py** - Context management

### VULCAN Services (256 files, 13,304 functions)
20-75. Complete VULCAN-AMI cognitive architecture

### Additional Services (200+ files)
76-100+. Supporting infrastructure, utilities, and specialized services

**Total**: 557 files, 21,523 functions, 4,353 classes

---

## 7. Recommendations

### 7.1 Immediate Actions

1. **Fix Port Conflicts**:
 - Change DQS default port from 8080 to 8083
 - Change Listener default port from 8181 to 8084

2. **Document Service Architecture**:
 - ✅ This document provides the inventory
 - Create service dependency diagram
 - Document inter-service communication

3. **Choose Deployment Strategy**:
 - **Development**: Use full_platform.py only (current 3 services)
 - **Production**: Use Docker Compose for all 9 services

### 7.2 For Full Platform Startup

If the goal is to start ALL services from `full_platform.py`:

**Option 1: Mount as Sub-Apps** (where possible)
- Convert FastAPI services to sub-applications
- Mount at different paths
- Share single port

**Option 2: Background Process Management**
- Start standalone services as background processes
- Use process manager (supervisord, systemd)
- Coordinate lifecycle

**Option 3: Docker Compose** (Recommended)
- Each service in own container
- Use docker-compose.yml for orchestration
- Service discovery via Docker networking

### 7.3 Documentation Updates Needed

1. **DEPLOYMENT.md**: Add multi-service deployment guide
2. **QUICKSTART.md**: Clarify which services start by default
3. **README.md**: Update architecture diagram
4. **docker-compose.yml**: Add all 9 services

---

## 8. Conclusion

The VulcanAMI platform consists of **9 core runtime services** plus **62+ functional service modules**. Currently, `full_platform.py` starts only 3 of the 9 runtime services by mounting them as sub-applications.

**To achieve complete platform startup**, one of the following approaches is needed:

1. **Mount all services as sub-apps** (requires refactoring 6 services)
2. **Use process orchestration** (supervisord, systemd)
3. **Use container orchestration** (Docker Compose, Kubernetes)

The documentation clearly lists "everything the platform does" across 557 files and 21,523 functions. The startup gap is in the **runtime orchestration layer**, not in the functional capabilities.

---

**Next Steps**: 
1. Decide on deployment strategy (dev vs prod)
2. Fix port conflicts in default configurations
3. Update full_platform.py to either:
 - Mount additional services as sub-apps, OR
 - Document that standalone services must be started separately
4. Create comprehensive docker-compose.yml for full platform
