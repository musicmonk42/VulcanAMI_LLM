# Production Readiness Migration Guide

This guide provides step-by-step instructions to fix critical security and reliability issues identified in the security audit.

---

## Phase 1: Critical Security Fixes (Week 1)

### 1.1 Fix Bare Except Clauses (240+ instances)

**Priority:** P0 - Critical  
**Effort:** 2-3 days  
**Risk:** High - Silent failures, debugging impossible

#### Automated Fix Script:

```bash
# Find all bare except clauses
grep -rn "except:" --include="*.py" src/ > bare_excepts.txt

# Review each one and apply appropriate fix
```

#### Fix Pattern:

```python
# BEFORE (UNSAFE):
try:
    risky_operation()
except:
    pass

# AFTER (SAFE) - Option 1: Log and continue
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # Decide: return default, retry, or re-raise

# AFTER (SAFE) - Option 2: Be specific
try:
    with open(file_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Failed to load {file_path}: {e}")
    return None
except Exception as e:
    logger.critical(f"Unexpected error loading {file_path}: {e}", exc_info=True)
    raise
```

#### Files to Fix (High Priority):

1. `src/unified_runtime/graph_validator.py:714`
2. `src/unified_runtime/execution_engine.py:1162`
3. `src/unified_runtime/hardware_dispatcher_integration.py:873`
4. `src/vulcan/processing.py:243, 290, 336, 405, 730, 1012, 2042`
5. All files in `src/vulcan/world_model/`

### 1.2 Fix Unsafe Pickle Loading (15+ instances)

**Priority:** P0 - Critical  
**Effort:** 1-2 days  
**Risk:** Critical - Remote code execution

#### Migration Strategy:

**Option A: Replace with JSON (Preferred)**
```python
# BEFORE:
import pickle
with open('checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)

# AFTER:
import json
with open('checkpoint.json', 'r') as f:
    data = json.load(f)
```

**Option B: Use SafeTensors (for ML models)**
```python
# BEFORE:
torch.save(model.state_dict(), 'model.pkl')
model.load_state_dict(torch.load('model.pkl'))

# AFTER:
from safetensors.torch import save_file, load_file
save_file(model.state_dict(), 'model.safetensors')
model.load_state_dict(load_file('model.safetensors'))
```

**Option C: Use RestrictedUnpickler (if pickle required)**
```python
# Use the RestrictedUnpickler from src/security_fixes.py
from security_fixes import safe_pickle_load

# BEFORE:
with open('checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)

# AFTER:
data = safe_pickle_load('checkpoint.pkl')
```

#### Files to Fix:

1. `src/vulcan/world_model/world_model_router.py:1832`
2. `src/vulcan/processing.py:346`
3. `src/vulcan/orchestrator/deployment.py:1208`
4. `src/vulcan/knowledge_crystallizer/*.py` (multiple files)
5. `src/vulcan/reasoning/*.py` (2 files)

### 1.3 Fix Subprocess Command Injection (15+ instances)

**Priority:** P0 - Critical  
**Effort:** 1 day  
**Risk:** High - Command injection

#### Fix Pattern:

```python
# BEFORE (POTENTIALLY UNSAFE):
subprocess.run(['git', 'add', file_path], check=True)

# AFTER (SAFE):
from security_fixes import safe_git_add, validate_file_path

# Validate file path first
validated_path = validate_file_path(file_path, allowed_base=repo_root)
result = safe_git_add(str(validated_path), repo_root)
```

#### Files to Fix:

1. `src/vulcan/world_model/meta_reasoning/self_improvement_drive.py:2058, 2062, 2067`
2. `src/vulcan/world_model/meta_reasoning/auto_apply_policy.py:331`
3. `src/vulcan/world_model/world_model_core.py:1819, 1821, 1830`
4. `src/compiler/graph_compiler.py:631`

---

## Phase 2: Production Configuration (Week 2)

### 2.1 Remove In-Memory Fallbacks

**File:** `app.py:74-112`

```python
# BEFORE: Falls back to memory storage
try:
    redis_client.ping()
    redis_storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"
except Exception as e:
    print(f"⚠️  Redis not available for rate limiting: {e}")
    redis_storage_uri = "memory://"

# AFTER: Fail fast in production
import os

IS_PRODUCTION = os.environ.get('FLASK_ENV') == 'production'

try:
    redis_client.ping()
    redis_storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"
    print(f"✅ Rate limiter connected to Redis: {redis_storage_uri}")
except Exception as e:
    if IS_PRODUCTION:
        raise RuntimeError(f"Redis is required in production: {e}")
    else:
        print(f"⚠️  Redis not available, using in-memory (DEV ONLY): {e}")
        redis_storage_uri = "memory://"
```

### 2.2 Add Configuration Validation

Add to `app.py` startup:

```python
from security_fixes import validate_production_config

if __name__ == '__main__':
    # Validate configuration before starting
    if os.environ.get('FLASK_ENV') == 'production':
        validate_production_config()
    
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False  # Never True in production
    )
```

### 2.3 Remove Debug Output

**Create script to find and remove:**

```bash
# Find all print statements (2,167 found)
grep -rn "print(" --include="*.py" src/ > print_statements.txt

# Replace with proper logging
# Manual review required - not all prints are bad
```

**Pattern:**

```python
# BEFORE:
print(f"Processing {item}")

# AFTER:
logger.debug(f"Processing {item}")  # Use debug level
# OR
logger.info(f"Processing {item}")   # Use info for important events
```

---

## Phase 3: Monitoring & Observability (Week 3)

### 3.1 Add Health Checks

```python
# Add to app.py
@app.route('/health')
def health():
    """Comprehensive health check with dependency validation."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
    
    # Check database
    try:
        db.session.execute('SELECT 1')
        health_status['checks']['database'] = 'ok'
    except Exception as e:
        health_status['checks']['database'] = f'error: {e}'
        health_status['status'] = 'unhealthy'
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status['checks']['redis'] = 'ok'
    except Exception as e:
        health_status['checks']['redis'] = f'error: {e}'
        health_status['status'] = 'unhealthy'
    
    # Return appropriate status code
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code
```

### 3.2 Add Readiness Check

```python
@app.route('/ready')
def readiness():
    """Readiness check for load balancers."""
    # Check if app is ready to serve traffic
    # Could include: migrations complete, cache warmed, etc.
    return jsonify({'status': 'ready'}), 200
```

### 3.3 Enable Structured Logging

```python
import logging.config
import json

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'stream': 'ext://sys.stdout'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

## Phase 4: Testing & Validation (Week 4)

### 4.1 Run Security Scans

```bash
# Install security tools
pip install bandit safety pip-audit

# Run Bandit for security issues
bandit -r src/ -f json -o bandit_report.json

# Check for vulnerable dependencies
safety check --json > safety_report.json
pip-audit --format json > pip_audit_report.json

# Review reports and fix issues
```

### 4.2 Add Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
        
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        
  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=120']
```

Install:
```bash
pip install pre-commit
pre-commit install
```

### 4.3 Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Aim for >80% coverage
```

---

## Phase 5: Deployment Checklist

### Pre-Deployment Checklist:

- [ ] All bare except clauses fixed
- [ ] Pickle loading replaced or secured
- [ ] Subprocess calls validated
- [ ] Debug output removed or disabled
- [ ] Configuration validation implemented
- [ ] Redis required in production
- [ ] Health checks added
- [ ] Monitoring enabled (Prometheus/Grafana)
- [ ] Logging aggregation configured
- [ ] Security scans passed
- [ ] Test coverage >80%
- [ ] Load testing completed
- [ ] Disaster recovery plan documented
- [ ] Rollback procedure tested

### Environment Variables Required:

```bash
# Create .env.production file (DO NOT COMMIT)
export FLASK_ENV=production
export DEBUG=false
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export DB_URI=postgresql://user:pass@host:5432/dbname
export REDIS_HOST=redis.prod.internal
export REDIS_PORT=6379
export CORS_ORIGINS=https://app.example.com,https://www.example.com
export AUDIT_LOG_PATH=/var/log/graphix/audit.jsonl
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
```

### Deployment Steps:

1. **Backup existing system**
   ```bash
   pg_dump database > backup_$(date +%Y%m%d).sql
   ```

2. **Deploy new code**
   ```bash
   git pull origin main
   pip install -r requirements.txt
   ```

3. **Run migrations** (if any)
   ```bash
   flask db upgrade
   ```

4. **Restart services with zero downtime**
   ```bash
   # Blue-green deployment or rolling restart
   systemctl reload gunicorn
   ```

5. **Verify health checks**
   ```bash
   curl https://api.example.com/health
   curl https://api.example.com/ready
   ```

6. **Monitor for errors**
   - Check error rates in Grafana
   - Review logs in ELK/Splunk
   - Monitor Slack alerts

---

## Rollback Procedure

If deployment fails:

1. **Revert code**
   ```bash
   git revert <commit-sha>
   git push origin main
   ```

2. **Restore database** (if needed)
   ```bash
   psql database < backup_YYYYMMDD.sql
   ```

3. **Restart services**
   ```bash
   systemctl restart gunicorn
   ```

4. **Verify rollback**
   ```bash
   curl https://api.example.com/health
   ```

---

## Post-Deployment Monitoring

Monitor these metrics for 24-48 hours:

- **Error rates:** Should not increase
- **Response times:** Should remain stable
- **Memory usage:** Watch for leaks
- **CPU usage:** Should be normal
- **Database connections:** Should not leak
- **Redis connection pool:** Should be stable

---

## Support & Escalation

If issues occur:

1. **Check logs first**
   ```bash
   tail -f /var/log/graphix/app.log
   ```

2. **Check Grafana dashboards**
   - System health dashboard
   - Application metrics dashboard

3. **Escalation path**
   - Level 1: On-call engineer
   - Level 2: Senior engineer
   - Level 3: Engineering manager

---

*End of Migration Guide*
