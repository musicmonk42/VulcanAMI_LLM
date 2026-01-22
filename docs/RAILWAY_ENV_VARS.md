# Railway Environment Variables - Complete Configuration Guide

This document provides a comprehensive list of all environment variables for deploying VulcanAMI on Railway cloud platform.

## Priority Classification

### 🔴 CRITICAL - Required for Railway Deployment

```bash
# ============================================================================
# CORE AUTHENTICATION (CRITICAL)
# ============================================================================
JWT_SECRET_KEY="<generate with: openssl rand -base64 48>"
GRAPHIX_JWT_SECRET="${JWT_SECRET_KEY}"  # Can use same value
BOOTSTRAP_KEY="<generate with: openssl rand -base64 32>"
SECRET_KEY="<generate with: openssl rand -base64 32>"

# ============================================================================
# LLM API KEYS (CRITICAL - Required for core functionality)
# ============================================================================
OPENAI_API_KEY="sk-..."  # Primary LLM for language formatting
VULCAN_LLM_API_KEY="${OPENAI_API_KEY}"  # For self-improvement

# ============================================================================
# RAILWAY PLATFORM (CRITICAL)
# ============================================================================
PORT="${PORT}"  # Railway auto-sets this
HOST="0.0.0.0"  # MUST be 0.0.0.0 for Railway (not 127.0.0.1)
API_HOST="0.0.0.0"
UNIFIED_HOST="0.0.0.0"
UNIFIED_PORT="${PORT}"
ENVIRONMENT="production"
LOG_LEVEL="INFO"

# ============================================================================
# DATABASE (CRITICAL - Use Railway's PostgreSQL)
# ============================================================================
DATABASE_URL="${DATABASE_URL}"  # Railway provides this
POSTGRES_DB="vulcanami"
POSTGRES_USER="${PGUSER}"  # Railway provides
POSTGRES_PASSWORD="${PGPASSWORD}"  # Railway provides
PGDATA="${PGDATA}"  # Railway provides
PGPORT="${PGPORT}"  # Railway provides

# ============================================================================
# REDIS (CRITICAL - Use Railway's Redis)
# ============================================================================
REDIS_URL="${REDIS_URL}"  # Railway provides this
REDIS_PUBLIC_URL="${REDIS_PUBLIC_URL}"  # Railway provides
REDIS_HOST="${REDIS_HOST}"  # Railway provides
REDIS_PORT="${REDIS_PORT}"  # Railway provides
REDIS_PASSWORD="${REDIS_PASSWORD}"  # Railway provides
```

### 🟡 HIGH PRIORITY - Strongly Recommended

```bash
# ============================================================================
# ADDITIONAL LLM PROVIDERS (RECOMMENDED)
# ============================================================================
ANTHROPIC_API_KEY="sk-ant-..."  # For Claude models
GEMINI_API_KEY="..."  # For Google Gemini
COHERE_API_KEY="..."  # For Cohere models
HUGGINGFACE_API_KEY="..."  # For HuggingFace models

# ============================================================================
# GRAPHIX SERVICE (RECOMMENDED)
# ============================================================================
GRAPHIX_API_KEY="..."
GRAPHIX_JWT_SECRET="${JWT_SECRET_KEY}"
ENABLE_GRAPHIX_VULCAN_LLM="true"

# ============================================================================
# OBJECT STORAGE (RECOMMENDED - for persistent memory)
# ============================================================================
# Option 1: Use Railway's MinIO or external S3-compatible service
MINIO_ENDPOINT="minio.railway.internal:9000"
MINIO_ROOT_USER="minioadmin"
MINIO_ROOT_PASSWORD="<generate with: openssl rand -base64 24>"
MINIO_SECURE="false"

# Option 2: Use AWS S3
# AWS_ACCESS_KEY_ID="..."
# AWS_SECRET_ACCESS_KEY="..."
# AWS_REGION="us-east-1"
# S3_BUCKET="vulcanami-memory"

# ============================================================================
# MEMORY ENCRYPTION (RECOMMENDED)
# ============================================================================
MEMORY_ENCRYPT_KEY="<generate with: openssl rand -base64 32>"

# ============================================================================
# GRAFANA MONITORING (RECOMMENDED)
# ============================================================================
GRAFANA_PASSWORD="<generate with: openssl rand -base64 16>"
```

### 🟢 OPTIONAL - Performance & Feature Flags

```bash
# ============================================================================
# NEW VARIABLES FROM PRODUCTION FIXES (✨ Added in this PR)
# ============================================================================
# Phantom Resolution Circuit Breaker Configuration
VULCAN_PHANTOM_THRESHOLD="5"  # Increased from 3 to 5
VULCAN_PHANTOM_WINDOW="3600"  # 1 hour window
VULCAN_PHANTOM_COOLDOWN="3600"  # 1 hour cooldown

# Warning Suppression (useful in containers)
VULCAN_SUPPRESS_CPU_PRIORITY_WARNING="1"  # ✨ NEW
VULCAN_SUPPRESS_DEFAULT_OBJECTIVES_WARNING="1"  # ✨ NEW

# Resolution Configuration
VULCAN_RESOLUTION_TTL="1800"  # 30 minutes
VULCAN_GAP_GIVEUP_THRESHOLD="10"  # Increased from 3

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================
# OpenAI Configuration
OPENAI_LANGUAGE_ONLY="true"
OPENAI_LANGUAGE_FORMATTING="true"
OPENAI_LANGUAGE_POLISH="false"
HYBRID_EXECUTOR_TIMEOUT="30.0"

# Query Routing
TRUST_ROUTER_TOOL_SELECTION="true"
SINGLE_REASONING_PATH="true"
QUERY_ROUTING_TIMEOUT="30.0"

# LLM Timeouts
GRAPHIX_VULCAN_TIMEOUT="120.0"
VULCAN_EMBEDDING_TIMEOUT="5.0"
VULCAN_LLM_TIMEOUT="120.0"
VULCAN_LLM_FAST_TIMEOUT="30.0"
VULCAN_LLM_HARD_TIMEOUT="180.0"
VULCAN_LLM_PER_TOKEN_TIMEOUT="1.0"

# Caching
OPENAI_CACHE_ENABLED="true"
OPENAI_CACHE_MAX_SIZE="1000"
OPENAI_CACHE_TTL_SECONDS="3600"

# CPU Thread Management (CRITICAL for Railway)
OMP_NUM_THREADS="2"  # Reduced for Railway (2-4 vCPU typical)
MKL_NUM_THREADS="2"
TORCH_NUM_THREADS="2"
OPENBLAS_NUM_THREADS="2"
VECLIB_MAXIMUM_THREADS="2"
NUMEXPR_NUM_THREADS="2"
TOKENIZERS_PARALLELISM="false"

# Ray Configuration (DISABLE on Railway - limited /dev/shm)
RAY_DISABLE_DOCKER_CPU_WARNING="1"
# DO NOT SET: VULCAN_ENABLE_RAY (let it auto-detect and disable)

# ============================================================================
# CURIOSITY ENGINE CONFIGURATION
# ============================================================================
CURIOSITY_HEARTBEAT_INTERVAL="60.0"
CURIOSITY_MIN_BUDGET="10.0"
CURIOSITY_MAX_EXPERIMENTS="5"
CURIOSITY_LOW_BUDGET_SLEEP="120.0"
CURIOSITY_CYCLE_TIMEOUT="300.0"

# ============================================================================
# SELF-IMPROVEMENT CONFIGURATION
# ============================================================================
VULCAN_ENABLE_SELF_IMPROVEMENT="1"
SELF_IMPROVEMENT_INTERVAL="86400"  # 24 hours
SELF_IMPROVEMENT_MIN_INTERVAL="3600"  # 1 hour
VULCAN_SELF_IMPROVEMENT_AUTO_COMMIT="false"  # Don't auto-commit on Railway

# Intrinsic Drives
INTRINSIC_DRIVES_ENABLED="true"
INTRINSIC_DRIVES_APPROVAL_REQUIRED="false"
INTRINSIC_DRIVES_CHECK_INTERVAL="120"
INTRINSIC_DRIVES_MAX_COST_SESSION="2.0"
INTRINSIC_DRIVES_MAX_COST_DAY="10.0"

# ============================================================================
# AGENT POOL CONFIGURATION
# ============================================================================
MIN_AGENTS="5"  # Reduced for Railway resource constraints
MAX_AGENTS="20"  # Reduced from 100
AGENT_CHECK_INTERVAL="60"  # Reduced polling frequency

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================
REJECT_INSECURE_JWT="true"
ALLOW_EPHEMERAL_KEY="false"
ALLOW_LEGACY_AUTH="false"
VULCAN_ENFORCE_SAFE_PICKLE="true"  # Enable in production
FORCE_HTTPS="true"  # Railway provides HTTPS

# JWT Configuration
JWT_ALGORITHM="HS256"
GRAPHIX_JWT_ALGO="HS256"
GRAPHIX_JWT_EXPIRY_HOURS="24"
TOKEN_EXPIRY_HOURS="24"

# CORS Configuration
ALLOWED_ORIGINS="https://your-domain.railway.app"
CORS_ORIGINS="https://your-domain.railway.app"

# ============================================================================
# AUDIT & LOGGING
# ============================================================================
AUDIT_LOG_ENCRYPT="true"
AUDIT_LOG_ENCRYPTION_KEY="<generate with: openssl rand -base64 32>"
AUDIT_LOG_PATH="/app/logs/audit"
AUDIT_LOG_MAX_SIZE="100"  # MB
AUDIT_LOG_RETENTION="30"  # days

# ============================================================================
# FEATURE FLAGS
# ============================================================================
# Disable resource-intensive features on Railway
SKIP_BERT_EMBEDDINGS="false"
DISABLE_WORLD_MODEL="false"
DISABLE_META_REASONING="false"
ARENA_ENABLED="false"  # Arena can be resource-intensive

# Enable performance features
LLM_FIRST_CLASSIFICATION="true"
CLASSIFICATION_LLM_TIMEOUT="3.0"
CLASSIFICATION_LLM_MODEL="gpt-4o-mini"
TOOL_SELECTOR_USE_LLM_CLASSIFICATION="true"
TOOL_SELECTOR_LLM_CONFIDENCE_THRESHOLD="0.8"

# ============================================================================
# STORAGE PATHS (Railway ephemeral filesystem)
# ============================================================================
VULCAN_STORAGE_PATH="/app/storage"
VULCAN_STATE_DIR="/app/state"
MEMORY_STORE_PATH="/app/memory_store"
CACHE_ROOT="/tmp/vulcan_cache"
CONFIG_PATH="/app/configs"
DB_PATH="/app/data/vulcan.db"
VULCAN_RESOLUTION_DB_PATH="/app/data/resolutions.db"

# ============================================================================
# GRAPHIX SPECIFIC CONFIGURATION
# ============================================================================
GRAPHIX_API_PORT="${PORT}"
GRAPHIX_CACHE_TTL="3600"
GRAPHIX_CACHE_MAX="1000"
GRAPHIX_MAX_NODES="10000"
GRAPHIX_MAX_EDGES="50000"
GRAPHIX_REQUEST_TIMEOUT="30.0"
GRAPHIX_RATE_MAX="100"
GRAPHIX_RATE_WINDOW="60"
GRAPHIX_DB_PATH="/app/data/graphix.db"
GRAPHIX_DB_POOL="10"

# ============================================================================
# DQS & PII SERVICES (if deployed separately)
# ============================================================================
DQS_HOST="dqs.railway.internal"
DQS_PORT="8083"
PII_HOST="pii.railway.internal"
PII_PORT="8082"

# ============================================================================
# TESTING & DEBUG (DO NOT USE IN PRODUCTION)
# ============================================================================
# DEBUG="false"
# VULCAN_TEST_MODE="false"
# LOAD_TEST_TESTING_MODE="false"
```

## Railway-Specific Considerations

### 1. **Network Binding**
```bash
# ❌ WRONG for Railway (will not receive traffic)
HOST="127.0.0.1"

# ✅ CORRECT for Railway
HOST="0.0.0.0"
API_HOST="0.0.0.0"
UNIFIED_HOST="0.0.0.0"
```

### 2. **Port Configuration**
```bash
# Railway provides the PORT variable - use it
PORT="${PORT}"
UNIFIED_PORT="${PORT}"
```

### 3. **Resource Constraints**
Railway typically provides:
- 2-4 vCPU
- 2-8 GB RAM
- Limited /dev/shm (64MB)

Adjust accordingly:
```bash
# Thread limits
OMP_NUM_THREADS="2"
MKL_NUM_THREADS="2"
TORCH_NUM_THREADS="2"

# Agent limits
MIN_AGENTS="5"
MAX_AGENTS="20"

# Disable Ray (requires large /dev/shm)
# Don't set VULCAN_ENABLE_RAY - let it auto-detect
```

### 4. **Ephemeral Filesystem**
Railway uses ephemeral storage - all local files are lost on restart.

**Solutions:**
- Use Railway's PostgreSQL for persistent data
- Use Railway's Redis for caching
- Use external object storage (MinIO/S3) for large files
- Consider Railway's Volumes for critical local state

```bash
# Store paths that will be wiped on restart
VULCAN_STORAGE_PATH="/app/storage"  # Ephemeral
CACHE_ROOT="/tmp/vulcan_cache"  # Ephemeral

# Use database for persistence
DATABASE_URL="${DATABASE_URL}"  # Persistent
REDIS_URL="${REDIS_URL}"  # Persistent
```

### 5. **Git Persistence** (Important for Self-Improvement)
```bash
# To prevent "Groundhog Day" loop where improvements are lost:
VULCAN_GIT_PUSH_ENABLED="1"  # Enable if Railway Git Integration is configured
```

## Environment Variables Comparison

### Already Set ✅
Your current Railway environment has these covered:
- ✅ API_HOST, API_PORT, HOST, PORT
- ✅ UNIFIED_HOST, UNIFIED_PORT, UNIFIED_API_KEY, UNIFIED_JWT_SECRET
- ✅ DATABASE_URL, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, PGDATA, PGPORT
- ✅ REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_URL, REDIS_PUBLIC_URL
- ✅ OPENAI_API_KEY, GEMINI_API_KEY, LLM_API_KEY, VULCAN_LLM_API_KEY
- ✅ GRAPHIX_API_KEY, GRAPHIX_JWT_SECRET
- ✅ JWT_SECRET_KEY, SECRET_KEY, BOOTSTRAP_KEY
- ✅ MINIO_ENDPOINT, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, MINIO_SECURE
- ✅ MEMORY_ENCRYPT_KEY
- ✅ GRAFANA_PASSWORD
- ✅ ENABLE_GRAPHIX_VULCAN_LLM
- ✅ ENVIRONMENT, LOG_LEVEL
- ✅ REJECT_INSECURE_JWT

### ⚠️ MISSING - Should Add

#### Critical Performance (✨ From This PR)
```bash
VULCAN_SUPPRESS_CPU_PRIORITY_WARNING="1"
VULCAN_SUPPRESS_DEFAULT_OBJECTIVES_WARNING="1"
VULCAN_PHANTOM_THRESHOLD="5"
```

#### CPU Thread Management (Critical for Railway)
```bash
OMP_NUM_THREADS="2"
MKL_NUM_THREADS="2"
TORCH_NUM_THREADS="2"
OPENBLAS_NUM_THREADS="2"
VECLIB_MAXIMUM_THREADS="2"
NUMEXPR_NUM_THREADS="2"
TOKENIZERS_PARALLELISM="false"
```

#### Railway-Specific
```bash
RAY_DISABLE_DOCKER_CPU_WARNING="1"
RAILWAY_ENVIRONMENT="${RAILWAY_ENVIRONMENT}"  # Auto-provided by Railway
RAILWAY_SERVICE_NAME="${RAILWAY_SERVICE_NAME}"  # Auto-provided by Railway
```

#### Performance Optimization
```bash
OPENAI_LANGUAGE_FORMATTING="true"
QUERY_ROUTING_TIMEOUT="30.0"
HYBRID_EXECUTOR_TIMEOUT="30.0"
GRAPHIX_VULCAN_TIMEOUT="120.0"
OPENAI_CACHE_ENABLED="true"
```

### 🔍 Optional Additions

```bash
# Additional API Keys (if using these services)
ANTHROPIC_API_KEY="sk-ant-..."
COHERE_API_KEY="..."
HUGGINGFACE_API_KEY="..."
HF_TOKEN="..."

# Agent Pool Limits (recommended for Railway)
MIN_AGENTS="5"
MAX_AGENTS="20"
AGENT_CHECK_INTERVAL="60"

# Curiosity Engine
CURIOSITY_HEARTBEAT_INTERVAL="60.0"
CURIOSITY_MAX_EXPERIMENTS="5"

# Security
VULCAN_ENFORCE_SAFE_PICKLE="true"
AUDIT_LOG_ENCRYPT="true"
```

## Quick Setup Commands

Add these to your Railway dashboard (Settings > Variables):

```bash
# Performance & Optimization
VULCAN_SUPPRESS_CPU_PRIORITY_WARNING=1
VULCAN_SUPPRESS_DEFAULT_OBJECTIVES_WARNING=1
VULCAN_PHANTOM_THRESHOLD=5
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
TORCH_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
TOKENIZERS_PARALLELISM=false
RAY_DISABLE_DOCKER_CPU_WARNING=1

# Feature Optimization
OPENAI_LANGUAGE_FORMATTING=true
QUERY_ROUTING_TIMEOUT=30.0
HYBRID_EXECUTOR_TIMEOUT=30.0
OPENAI_CACHE_ENABLED=true
MIN_AGENTS=5
MAX_AGENTS=20

# Security
VULCAN_ENFORCE_SAFE_PICKLE=true
FORCE_HTTPS=true
```

## Testing Your Configuration

1. **Health Check**
   ```bash
   curl https://your-app.railway.app/health/live
   ```

2. **LLM Status**
   ```bash
   curl https://your-app.railway.app/vulcan/v1/llm/status
   ```

3. **Test Query**
   ```bash
   curl -X POST https://your-app.railway.app/vulcan/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, test the system"}'
   ```

## Troubleshooting

### Issue: Application not receiving traffic
**Solution:** Ensure `HOST="0.0.0.0"` (not `127.0.0.1`)

### Issue: Database connection failed
**Solution:** Verify `DATABASE_URL` is set by Railway PostgreSQL service

### Issue: High CPU usage
**Solution:** Check thread limits are set (OMP_NUM_THREADS, etc.)

### Issue: Ray initialization warnings
**Solution:** Add `RAY_DISABLE_DOCKER_CPU_WARNING="1"`

### Issue: Out of memory
**Solution:** Reduce agent pool size (MIN_AGENTS=5, MAX_AGENTS=20)

## Security Checklist

- [ ] All secrets generated with proper entropy (openssl rand)
- [ ] JWT_SECRET_KEY is strong (min 32 characters)
- [ ] REJECT_INSECURE_JWT=true
- [ ] FORCE_HTTPS=true on Railway
- [ ] VULCAN_ENFORCE_SAFE_PICKLE=true in production
- [ ] Audit logging encryption enabled
- [ ] No secrets in logs (LOG_LEVEL=INFO, not DEBUG)
- [ ] CORS origins properly configured
- [ ] MinIO/S3 credentials secured

## See Also

- `.env.example` - Complete environment variable reference
- `docs/PRODUCTION_FIXES.md` - Production issue fixes
- `railway.toml` - Railway deployment configuration
- `docs/COMPLETE_SERVICE_CATALOG.md` - Service port allocation
