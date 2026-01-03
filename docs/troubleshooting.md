# VulcanAMI Platform Troubleshooting Guide

**Version:** 2.4.0  
**Last Updated:** January 1, 2026

This guide provides solutions for common issues encountered when developing, deploying, and operating the VulcanAMI/GraphixVulcan platform.

---

## Table of Contents

1. [Quick Diagnostic Commands](#quick-diagnostic-commands)
2. [Docker Issues](#1-docker-issues)
3. [Kubernetes Issues](#2-kubernetes-issues)
4. [Python/Dependency Issues](#3-pythondependency-issues)
5. [Service Startup Issues](#4-service-startup-issues)
6. [Performance Issues](#5-performance-issues)
7. [Security Issues](#6-security-issues)
8. [Network Issues](#7-network-issues)
9. [Diagnostic Workflows](#diagnostic-workflows)
10. [Related Documentation](#related-documentation)

---

## Quick Diagnostic Commands

Run these commands to quickly assess system health:

```bash
# ===== Validation Scripts =====

# Quick pre-deployment check (30 seconds) - validates core configs only
./quick_test.sh quick

# Full comprehensive validation (42+ checks) - runs all validation tests
./test_full_cicd.sh

# Validate all configurations
./scripts/validate-all.sh

# Docker configuration validation
./quick_docker_validation.sh

# ===== Service Health Checks =====

# Docker Compose status
docker compose -f docker-compose.dev.yml ps

# Check Docker logs
docker compose -f docker-compose.dev.yml logs -f

# Kubernetes pod status
kubectl get pods -n vulcanami

# Check Kubernetes events
kubectl get events -n vulcanami --sort-by='.lastTimestamp'

# ===== System Health =====

# Check Python environment
python --version
pip list | grep -E "torch|flask|fastapi"

# Check port usage
sudo lsof -i :8000
sudo lsof -i :8080
sudo lsof -i :5432
sudo lsof -i :6379

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:5000/health
```

---

## 1. Docker Issues

### Issue: REJECT_INSECURE_JWT build error

**Symptoms:**
- Docker build fails with message: "Refusing to build: set --build-arg REJECT_INSECURE_JWT=ack"

**Cause:**
All Dockerfiles require explicit acknowledgment that you're not embedding secrets in the image.

**Solution:**

For local Docker builds:
```bash
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:latest .
```

For docker-compose:
```yaml
services:
  your-service:
    build:
      context: .
      args:
        REJECT_INSECURE_JWT: ack
```

For Railway deployment:
Set `REJECT_INSECURE_JWT=ack` as an environment variable in your Railway service's Variables tab.

For Google Cloud Build:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

---

### Issue: "required variable X is missing" in Docker Compose

**Symptoms:**
- Docker Compose fails with errors about missing environment variables
- `docker-compose.prod.yml` validation fails

**Cause:**
Production Docker Compose requires all secrets to be explicitly set.

**Solution:**

Create a `.env` file with all required values:
```bash
# Create .env from template
cp .env.example .env

# Generate secure secrets
export JWT_SECRET_KEY=$(openssl rand -base64 48)
export BOOTSTRAP_KEY=$(openssl rand -base64 32)
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export MINIO_ROOT_PASSWORD=$(openssl rand -base64 24)
export GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Update .env with your secrets (or use generate-secrets script)
./scripts/generate-secrets.sh >> .env
```

For development, use `docker-compose.dev.yml` which has sensible defaults:
```bash
docker compose -f docker-compose.dev.yml up -d
```

---

### Issue: Port conflicts

**Symptoms:**
- Services fail to start with "address already in use" errors
- Container exits immediately after starting

**Cause:**
Multiple services or processes are attempting to use the same port.

**Solution:**

Check what's using the ports:
```bash
sudo lsof -i :8000
sudo lsof -i :8080
sudo lsof -i :5432
sudo lsof -i :6379
```

Kill conflicting processes or change ports:
```bash
# Option 1: Kill the process
kill -9 <PID>

# Option 2: Change ports in docker-compose.yml
# Edit the ports mapping in your compose file
```

**Port Allocation Reference:**

> See [COMPLETE_SERVICE_CATALOG.md](COMPLETE_SERVICE_CATALOG.md#port-allocation-and-conflicts) for the complete port allocation matrix.

| Service | Default Port | Environment Variable |
|---------|--------------|---------------------|
| full_platform.py | 8080 | UNIFIED_PORT |
| dqs_service.py | 8083 | DQS_PORT |
| pii_service.py | 8082 | PII_PORT |
| Arena | 8181 | ARENA_PORT |
| listener.py | 8084 | LISTENER_PORT |
| API Gateway | 8000 | API_PORT |

---

### Issue: Docker Compose validation failures

**Symptoms:**
- `docker compose config` fails
- YAML parsing errors

**Solution:**

Ensure you're using Docker Compose v2:
```bash
# Check version (should be v2.x)
docker compose version

# Correct command (with space, not hyphen)
docker compose -f docker-compose.dev.yml up -d

# Wrong (old syntax)
docker-compose up -d
```

---

### Issue: Image build failures (SSL/network errors)

**Symptoms:**
- pip install fails during Docker build
- Certificate verification errors

**Cause:**
Network restrictions or proxy configuration issues in the build environment.

**Solution:**

This typically happens in CI environments with self-signed certificates. Build in a standard environment with proper SSL certificates, or configure your proxy:

```dockerfile
# Add to Dockerfile if needed
ENV http_proxy=http://proxy:port
ENV https_proxy=http://proxy:port
```

---

### Issue: Permission denied errors

**Symptoms:**
- Files owned by root
- Container cannot write to mounted volumes

**Cause:**
Docker containers run as non-root users (uid 1001) for security.

**Solution:**

Fix volume permissions:
```bash
# Change ownership to match container user
chown -R 1001:1001 /path/to/volume

# Or run with the appropriate user
docker run --user $(id -u):$(id -g) ...
```

---

## 2. Kubernetes Issues

### Issue: Connection refused to kubectl

**Symptoms:**
- `kubectl` commands fail with "connection refused"
- Cannot connect to cluster

**Cause:**
kubectl is not configured or cluster is not running.

**Solution:**

```bash
# Check if kubectl is configured
kubectl cluster-info

# If not configured, set up a local cluster
# Option 1: minikube
minikube start

# Option 2: kind
kind create cluster

# Option 3: Docker Desktop Kubernetes
# Enable Kubernetes in Docker Desktop settings

# Get credentials for cloud clusters
aws eks update-kubeconfig --region us-east-1 --name vulcanami-prod  # AWS
az aks get-credentials --resource-group vulcanami-prod --name vulcanami-cluster  # Azure
gcloud container clusters get-credentials vulcanami-prod --zone us-central1-a  # GCP
```

---

### Issue: Pods in CrashLoopBackOff

**Symptoms:**
- Pods repeatedly restart
- `kubectl get pods` shows CrashLoopBackOff status

**Cause:**
Application is failing to start, usually due to missing configuration or secrets.

**Solution:**

```bash
# Check pod status
kubectl describe pod <pod-name> -n vulcanami

# Check pod logs
kubectl logs <pod-name> -n vulcanami

# Check events
kubectl get events -n vulcanami --sort-by='.lastTimestamp'

# Common fixes:
# 1. Missing secrets
kubectl create secret generic vulcanami-secrets \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
  --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
  -n vulcanami

# 2. Resource constraints - check limits in deployment
kubectl edit deployment <deployment-name> -n vulcanami
```

---

### Issue: Secret management

**Symptoms:**
- Application fails with "missing secret" errors
- Secrets not available to pods

**Solution:**

Create Kubernetes secrets:
```bash
# From literal values
kubectl create secret generic vulcanami-secrets \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
  --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
  -n vulcanami

# From env file
kubectl create secret generic vulcanami-secrets \
  --from-env-file=.env \
  -n vulcanami

# Verify secrets exist
kubectl get secrets -n vulcanami
```

---

### Issue: Helm validation errors

**Symptoms:**
- `helm lint` shows "secrets.X is required" errors
- Helm install fails

**Cause:**
Helm values for secrets are not provided.

**Solution:**

These warnings are informational. Provide secrets during install:
```bash
helm install vulcanami ./helm/vulcanami \
  --namespace vulcanami \
  --set secrets.jwtSecretKey=$(openssl rand -base64 48 | tr -d '\n') \
  --set secrets.bootstrapKey=$(openssl rand -base64 32 | tr -d '\n') \
  --set secrets.postgresPassword=$(openssl rand -base64 32 | tr -d '\n') \
  --set secrets.redisPassword=$(openssl rand -base64 32 | tr -d '\n')
```

---

## 3. Python/Dependency Issues

### Issue: ModuleNotFoundError

**Symptoms:**
- Python imports fail with `ModuleNotFoundError`
- Dependencies not found

**Solution:**

```bash
# Ensure you're in the virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Set PYTHONPATH
export PYTHONPATH=.
```

---

### Issue: Dependency conflicts

**Symptoms:**
- pip install fails with version conflicts
- Package compatibility errors

**Solution:**

```bash
# Create a fresh virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Install with exact versions
pip install -r requirements-hashed.txt

# If conflicts persist, use pip-tools
pip install pip-tools
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
```

---

### Issue: Virtual environment issues

**Symptoms:**
- Wrong Python version
- Packages installed globally instead of in venv

**Solution:**

```bash
# Create venv with specific Python version
python3.11 -m venv .venv

# Verify activation
which python  # Should point to .venv/bin/python

# Verify Python version
python --version  # Should be 3.11+
```

---

## 4. Service Startup Issues

### Issue: GraphixVulcanLLM not available

**Symptoms:**
- API endpoints return 500 errors
- Service not responding

**Solution:**

Check service status:
```bash
# Docker Compose
docker compose -f docker-compose.dev.yml ps
docker compose -f docker-compose.dev.yml logs api-gateway

# Local development
python app.py  # Registry API on port 5000
# or
uvicorn src.graphix_arena:app --host 127.0.0.1 --port 8000  # Arena API
```

---

### Issue: Service mount failures

**Symptoms:**
- Unified platform fails to start
- Sub-services not mounting correctly

**Solution:**

```bash
# Start the unified platform
python src/full_platform.py

# Check logs for mount status
# Look for: "✅ vulcan: MOUNTED (path /vulcan)"

# Verify mounted services
curl http://localhost:8080/health
curl http://localhost:8080/vulcan/health
curl http://localhost:8080/arena/health
```

---

### Issue: Entrypoint fails with JWT error

**Symptoms:**
- Container exits with "ERROR: No valid JWT secret provided"
- Application won't start

**Solution:**

Set a strong JWT secret:
```bash
# Generate and export secret
export JWT_SECRET_KEY=$(openssl rand -base64 48 | tr -d '+/')

# Run with secret
docker run -e JWT_SECRET_KEY="$JWT_SECRET_KEY" vulcanami:latest
```

Requirements for JWT secret:
- Minimum 32 characters
- No weak patterns (password, secret, default, etc.)

---

### Issue: Reasoning Engine Returns Internal Metrics Instead of Answers

**Symptoms:**
- Query "What is 2+2?" returns `"mean prediction 0.500 with uncertainty 0.500"` instead of `"4"`
- Mathematical questions get routed to probabilistic reasoning
- Bayesian queries return GP metrics instead of computed probabilities
- Log shows: `"Unrecognized task type 'mathematical_task' - falling back to SYMBOLIC"`

**Root Cause:**
1. Task type mappings missing `_task` suffix variants (e.g., `mathematical_task` not in mapping)
2. LLM client not registered with singletons module during initialization
3. Missing arithmetic fallback when SymPy unavailable

**Solution:**

These issues were fixed in the reasoning engine update. Verify you have the latest code:

```bash
git pull origin main
```

**Configuration Check:**

Ensure the LLM client is properly registered by checking startup logs for:
```
✓ LLM client registered with singletons module
```

For Helm deployments, verify `values.yaml` has:
```yaml
llm:
  openai:
    enabled: true
```

For test environments, set `SKIP_OPENAI=false` to enable OpenAI fallback:
```yaml
# In scalability_test.yml or your CI workflow
env:
  SKIP_OPENAI: 'false'
  OPENAI_LANGUAGE_FORMATTING: 'true'
```

---

### Issue: No Language Generation Backend Available

**Symptoms:**
- Error: `"[HybridExecutor] ❌ Both internal LLM AND OpenAI fallback failed. No language generation backend available."`
- Scalability tests failing with timeout errors
- Log shows: `"[TIMEOUT] Async generation exceeded 120.0s limit"`

**Root Cause:**
1. Internal LLM (GraphixVulcanLLM) not properly initialized or models not loaded
2. OpenAI fallback disabled (`SKIP_OPENAI=true`)
3. Both backends failing simultaneously

**Solution:**

1. Enable OpenAI fallback in your CI/CD workflow:
```yaml
env:
  SKIP_OPENAI: 'false'
  OPENAI_LANGUAGE_FORMATTING: 'true'
```

2. Ensure OpenAI API key is set:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. For local development, check internal LLM status:
```bash
python -c "from graphix_vulcan_llm import GraphixVulcanLLM; llm = GraphixVulcanLLM(); print('LLM ready')"
```

---

## 5. Performance Issues

### Issue: GraphixVulcanLLM Silent Failure / Generation Hangs

**Symptoms:**
- Chat requests timeout after 60 seconds with no response
- Logs show "Generating up to X tokens..." but nothing after that
- CPU usage stays high (95-99%) during generation
- No error messages, just silence

**Root Cause (Fixed in v2.0.3):**
This was caused by the async CognitiveLoop being called from within an event loop context (via `run_in_executor`). The code couldn't create a new event loop when one was already running, causing a silent hang.

**Solution:**

This issue was fixed in GraphixVulcanLLM v2.0.3. If you're experiencing this issue, ensure you have the latest version:

```bash
# Check your version
python -c "from graphix_vulcan_llm import GraphixVulcanLLM; print(GraphixVulcanLLM.VERSION)"
# Should output: 2.0.3

# If older, update the repository
git pull origin main
```

**Key Fixes in v2.0.3:**
1. **Timeout wrapper**: Generation now has a configurable timeout (default 60s) to prevent infinite hangs
2. **Threaded async execution**: When called from an async context, generation runs in a separate thread with its own event loop
3. **Improved error logging**: Silent failures now log detailed error messages

**Diagnostic Logging (Added in v2.0.4):**

When debugging generation hangs, look for checkpoint logs at INFO level:

```
13:43:56.143 - [INFO] [DIAG] About to call CognitiveLoop.generate()...
13:43:56.144 - [INFO] [DIAG] _consume_async_result: type=<class 'coroutine'>
13:43:56.144 - [INFO] [DIAG] DETECTED COROUTINE - AWAITING...
13:43:56.145 - [INFO] [DIAG-STEP-0] Starting _step for FIRST token
13:43:56.145 - [INFO] [DIAG-STEP-0] Calling bridge.before_execution...
[LAST LOG BEFORE HANG IDENTIFIES BLOCKING OPERATION]
```

The LAST checkpoint log before the silence identifies the exact blocking operation. Common hang locations:
- `bridge.before_execution` - Bridge initialization hang
- `transformer.encode` - Transformer encoding hang
- `obtain_logits` - Logits computation hang

**Hard Timeout Fix (Added in v2.0.5):**

`asyncio.wait_for()` only checks timeouts between await points. If code blocks synchronously, 
the timeout never fires. v2.0.5 adds `ThreadPoolExecutor` hard timeouts that WILL fire even 
if the underlying code blocks:

```
# Expected log pattern with hard timeout firing:
[HybridExecutor] Calling generate(prompt_len=2146, max_tokens=500)...
[HybridExecutor] Using HARD timeout: 15.0s
[HybridExecutor] ❌ HARD TIMEOUT after 15.0s!
```

The `_Checkpoint` helper class in `cognitive_loop.py` provides elapsed-time tracking:
```
[CHECKPOINT 0.000s] generate() START: prompt_len=2146, max_tokens=500
[CHECKPOINT 0.001s] Calling _tokenize()...
[CHECKPOINT 0.050s] _tokenize() done: 523 tokens
[CHECKPOINT 0.051s] _step(0) START (FIRST TOKEN)
[CHECKPOINT 0.052s] Calling bridge.before_execution...
[HANGS HERE - LAST CHECKPOINT IDENTIFIES BLOCKING OPERATION]
```

**Configuration (Optional):**
```bash
# Set generation timeout (default: 60 seconds)
# In your .env or environment:
VULCAN_LLM_GENERATION_TIMEOUT=60.0

# Set VULCAN hard timeout for HybridExecutor (default: 15 seconds)
VULCAN_LLM_TIMEOUT=15.0
```

**For Helm deployments:**
```yaml
# In values.yaml under the llm section
llm:
  # ... other llm settings ...
  graphixVulcan:
    generationTimeout: 60.0
    vulcanTimeout: 15.0
    verboseLogging: false
```

---

### Issue: Slow responses

**Symptoms:**
- API responses take >5 seconds
- High latency in chat interface

**Solution:**

Enable Simple Mode for faster responses:
```bash
# Set in .env
VULCAN_SIMPLE_MODE=true
SKIP_BERT_EMBEDDINGS=true
OPENAI_ONLY_MODE=true
```

Or tune specific parameters:
```bash
# Reduce agent pool
MIN_AGENTS=1
MAX_AGENTS=5

# Increase check interval
AGENT_CHECK_INTERVAL=300

# Limit provenance records
MAX_PROVENANCE_RECORDS=50
```

---

### Issue: High memory usage

**Symptoms:**
- OOM (Out of Memory) errors
- Container restarts due to memory limits

**Solution:**

```bash
# Check memory usage
docker stats

# Reduce memory footprint
# In .env:
MAX_PROVENANCE_RECORDS=50
PROVENANCE_TTL_SECONDS=1800

# Increase container limits if needed
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 8G
```

---

### Issue: FAISS/LLVM warnings on startup

**Symptoms:**
- Log shows "FAISS AVX512 fallback" warning
- Log shows "LLVM execution engine" warning

**Cause:**
These are informational messages, not errors.

**Explanation:**
- **FAISS AVX512 warning**: System automatically uses best available instruction set (typically AVX2). This is expected and optimal for your CPU.
- **LLVM execution engine**: If creation fails, IR generation and compilation still work. Only JIT execution is unavailable.

**Solution:**
No action needed. For production optimization:
```bash
# Optional: Install FAISS for better vector search
pip install faiss-cpu

# Check CPU capabilities
cat /proc/cpuinfo | grep -E 'avx|avx2|avx512'
```

---

## 6. Security Issues

### Issue: JWT validation failures

**Symptoms:**
- API returns 401 Unauthorized
- Token validation errors

**Solution:**

```bash
# Check JWT secret is set correctly
echo $JWT_SECRET_KEY

# Ensure secret is strong enough (32+ chars)
export JWT_SECRET_KEY=$(openssl rand -base64 48 | tr -d '+/')

# Check token format
# JWT should have 3 parts separated by dots: header.payload.signature
```

---

### Issue: Secret management in production

**Symptoms:**
- Secrets exposed in logs
- Secrets committed to git

**Solution:**

Never commit secrets to version control:
```bash
# Verify .gitignore
cat .gitignore | grep -E "\.env|secrets"

# Use secret managers
# AWS: aws secretsmanager get-secret-value
# Azure: az keyvault secret show
# HashiCorp Vault: vault kv get
```

---

### Issue: Port binding security warnings

**Symptoms:**
- Security scan warns about 0.0.0.0 binding
- Exposed ports on network interfaces

**Solution:**

For development (local only):
```bash
# Use localhost binding
export HOST=127.0.0.1
export API_HOST=127.0.0.1
```

For Docker/containers (required):
```bash
# Must use 0.0.0.0 for container networking
export HOST=0.0.0.0
export API_HOST=0.0.0.0
# But ensure proper firewall/ingress rules
```

---

## 7. Network Issues

### Issue: CORS errors

**Symptoms:**
- Browser console shows CORS errors
- Frontend cannot reach backend

**Solution:**

CORS is typically handled by nginx in production:
```nginx
# In nginx configuration
add_header 'Access-Control-Allow-Origin' '*';
add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
```

For development:
```python
# In FastAPI
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

---

### Issue: Connection timeouts

**Symptoms:**
- API calls timeout
- Services cannot reach each other

**Solution:**

```bash
# Test connectivity from inside container
docker exec -it <container> sh
curl http://service-name:port/health

# Check network configuration
docker network ls
docker network inspect vulcanami-network

# In Kubernetes
kubectl exec -it <pod> -n vulcanami -- sh
curl http://service.vulcanami.svc.cluster.local:8000/health
```

---

### Issue: Offline status in chat UI

**Symptoms:**
- Chat interface shows "offline"
- WebSocket connection fails

**Solution:**

```bash
# Check if backend is running
curl http://localhost:8000/health

# Check WebSocket support (if applicable)
# Ensure nginx/ingress supports WebSocket:
# proxy_http_version 1.1;
# proxy_set_header Upgrade $http_upgrade;
# proxy_set_header Connection "upgrade";

# Start the unified platform
python src/full_platform.py
```

---

## Diagnostic Workflows

### Workflow 1: New Installation Not Working

```bash
# Step 1: Validate configuration
./scripts/validate-all.sh

# Step 2: Check prerequisites
python --version  # Should be 3.11+
docker --version  # Should be 20.10+
docker compose version  # Should be v2+

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Set environment
cp .env.example .env
# Edit .env with your values

# Step 5: Start services
docker compose -f docker-compose.dev.yml up -d

# Step 6: Check health
curl http://localhost:8000/health
```

### Workflow 2: Production Deployment Failing

```bash
# Step 1: Validate all configurations
./test_full_cicd.sh

# Step 2: Check secrets are set
env | grep -E "JWT|BOOTSTRAP|POSTGRES|REDIS"

# Step 3: Build and test locally
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:test .
docker run -e JWT_SECRET_KEY="test-key-32-characters-minimum" vulcanami:test

# Step 4: Check Kubernetes/Helm
kubectl get events -n vulcanami --sort-by='.lastTimestamp'
helm status vulcanami -n vulcanami

# Step 5: Check pod logs
kubectl logs -f deployment/vulcanami-api -n vulcanami
```

### Workflow 3: Performance Debugging

```bash
# Step 1: Enable metrics
curl http://localhost:8000/metrics

# Step 2: Check resource usage
docker stats
# or
kubectl top pods -n vulcanami

# Step 3: Enable simple mode for testing
export VULCAN_SIMPLE_MODE=true

# Step 4: Profile specific request
time curl http://localhost:8000/api/chat -d '{"message":"test"}'

# Step 5: Check latency metrics
grep "latency" logs/app.log
```

---

## Related Documentation

For more detailed information, see:

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment guide
- **[DOCKER_BUILD_GUIDE.md](DOCKER_BUILD_GUIDE.md)** - Docker build details
- **[NEW_ENGINEER_SETUP.md](NEW_ENGINEER_SETUP.md)** - Onboarding guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[INFRASTRUCTURE_SECURITY_GUIDE.md](INFRASTRUCTURE_SECURITY_GUIDE.md)** - Security best practices
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration reference
- **[AI_OPS.md](AI_OPS.md)** - AI operations guide
- **[OBSERVABILITY.md](OBSERVABILITY.md)** - Monitoring and metrics

---

**Document Version:** 2.4.0  
**Last Updated:** January 1, 2026
