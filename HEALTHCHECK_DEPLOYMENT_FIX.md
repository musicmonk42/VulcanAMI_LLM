# Healthcheck Deployment Failure - Root Cause Analysis and Fix

## Problem Statement

Railway deployment is failing with healthcheck timeout:
```
Attempt #1-11 failed with service unavailable. Continuing to retry...
1/2 replicas never became healthy!
Healthcheck failed!
```

## Root Cause

The healthcheck is failing at `/health/live` endpoint because the FastAPI application never starts. The issue is in the startup sequence:

1. **Entrypoint validation blocks startup**: `entrypoint.sh` validates JWT secrets BEFORE starting the application
2. **Missing or invalid JWT secret**: If Railway doesn't have a valid JWT secret configured, entrypoint.sh exits with error code 1
3. **Application never starts**: FastAPI never launches, so `/health/live` is never available
4. **Healthcheck times out**: Railway waits 5 minutes for healthcheck to succeed, but it never does

## Evidence

From `entrypoint.sh` (lines 83-95):
```sh
if [ "$SECRET_OK" -ne 1 ]; then
  cat >&2 <<'EOF'
ERROR: No valid JWT secret provided.
...
EOF
  exit 1  # <-- Application never starts
fi
```

From `Dockerfile` (line 291):
```dockerfile
CMD ["sh", "-c", "uvicorn src.full_platform:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
```

The CMD never executes if entrypoint.sh exits with error.

## Solution Options

### Option 1: Make JWT Secret Optional for Healthcheck (RECOMMENDED)

Modify `entrypoint.sh` to allow the application to start even without a JWT secret, but disable authentication features:

```sh
if [ "$SECRET_OK" -ne 1 ]; then
  echo "WARNING: No valid JWT secret provided." >&2
  echo "Application will start in LIMITED MODE:" >&2
  echo "  - Health endpoints (/health/live, /health/ready) will work" >&2
  echo "  - JWT authentication will be DISABLED" >&2
  echo "  - Protected endpoints will return 401 Unauthorized" >&2
  echo "Provide a valid JWT secret to enable full functionality." >&2
  export JWT_VALIDATION_MODE="disabled"
else
  echo "Verified JWT secret in variable: $SELECTED $EXPIRY_NOTE"
  export JWT_VALIDATION_MODE="enabled"
fi
```

Then modify `src/full_platform.py` to check `JWT_VALIDATION_MODE` and gracefully degrade.

### Option 2: Add Pre-healthcheck Script

Create a lightweight health server that starts BEFORE the main application:

```dockerfile
# Add a pre-health script
COPY pre-health.py /app/pre-health.py

# Start pre-health in background, then run main app
CMD ["sh", "-c", "python /app/pre-health.py & sleep 2 && /app/entrypoint.sh uvicorn src.full_platform:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
```

The `pre-health.py` would be a minimal HTTP server that only responds to `/health/live`.

### Option 3: Configure JWT Secret in Railway (IMMEDIATE FIX)

Set the required environment variable in Railway dashboard:

```
JWT_SECRET=<generate with: openssl rand -base64 48 | tr -d '+/'>
```

or

```
GRAPHIX_JWT_SECRET=<generate with: openssl rand -base64 48 | tr -d '+/'>
```

or

```
JWT_SECRET_KEY=<generate with: openssl rand -base64 48 | tr -d '+/'>
```

Minimum requirements:
- At least 32 characters
- Not matching known weak patterns
- URL-safe characters recommended

## Recommended Immediate Actions

1. **Short-term (Deploy Now)**:
   - Set a valid JWT secret in Railway environment variables
   - Redeploy to verify healthcheck passes

2. **Medium-term (Next PR)**:
   - Implement Option 1: Make JWT optional with graceful degradation
   - Add clear logging about JWT status during startup
   - Update documentation about JWT configuration

3. **Long-term (Architecture)**:
   - Consider separating health endpoints from main application
   - Implement JWT secret rotation mechanism
   - Add startup diagnostics endpoint that doesn't require auth

## Testing the Fix

After implementing Option 3 (immediate fix):

1. Set Railway environment variable:
   ```
   JWT_SECRET=$(openssl rand -base64 48 | tr -d '+/')
   ```

2. Redeploy:
   ```
   git push origin copilot/integrate-schema-auto-generator
   ```

3. Monitor Railway logs for:
   ```
   Verified JWT secret in variable: JWT_SECRET (rotate secrets periodically)
   Starting Unified Platform (Worker XXXXX)
   Server accepting connections - scheduling background initialization...
   ```

4. Verify healthcheck succeeds:
   ```
   curl https://your-railway-url.railway.app/health/live
   # Should return: {"status": "alive", "timestamp": "2026-01-12T..."}
   ```

## Files to Modify (for Option 1)

1. `entrypoint.sh` - Add graceful degradation
2. `src/full_platform.py` - Check JWT_VALIDATION_MODE env var
3. `railway.toml` - Update documentation about JWT requirement
4. `README.md` - Document JWT configuration clearly
5. `Dockerfile` - Add comments about JWT requirements

## Prevention for Future

- Add deployment tests that verify health endpoints respond
- Add Railway deployment validation to CI/CD
- Create deployment checklist that includes JWT secret verification
- Add startup diagnostics that don't require authentication
