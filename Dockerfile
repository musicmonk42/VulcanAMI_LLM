# =============================================================================
# Graphix / Vulcan Unified Platform Secure Container Build
# =============================================================================
# Hardened Dockerfile with:
# - Multi-stage build (builder + runtime)
# - Non-root execution (graphix user uid 1001)
# - Mandatory acknowledgement of NOT embedding insecure JWT secret
# - Runtime secret validation via entrypoint.sh (length, strength)
# - Hash-verified dependency installation (--require-hashes)
# - Optional SBOM generation (CycloneDX) for dependency transparency
# - Locale, timezone, buffering, bytecode settings
# - OS package updates & cleanup to reduce CVE footprint
# - Healthcheck endpoint probing
# - Reduced final attack surface (only necessary runtime artifacts copied)
#
# EXPECTED INPUTS AT RUNTIME (REQUIRED - NOT BAKED INTO IMAGE):
#   - One of the following secure JWT secret env vars:
#       GRAPHIX_JWT_SECRET   (Graphix API Server / src/api_server.py)
#       JWT_SECRET_KEY       (Flask registry / app.py)
#       JWT_SECRET           (Unified Platform / full_platform.py)
#
# RECOMMENDED: Provide a hashed requirements lock file named:
#       requirements-hashed.txt
# containing lines formatted as:
#       package==version --hash=sha256:<hash> [--hash=sha256:<hash2> ...]
#
# If requirements-hashed.txt is absent, the build will FALL BACK to
# standard 'pip install' (less secure). For production, ALWAYS supply
# the hashed lock file and remove the insecure fallback logic.
#
# To acknowledge that you intentionally are not baking a JWT secret
# into the image, pass:
#   --build-arg REJECT_INSECURE_JWT=ack
#
# =============================================================================

# -----------------------------
# Stage 1: Builder
# -----------------------------
FROM python:3.11-slim AS builder

ARG REJECT_INSECURE_JWT="default-super-secret-key-change-me"

# Fail build unless acknowledgement arg changed from default
RUN test "$REJECT_INSECURE_JWT" != "default-super-secret-key-change-me" || \
    (echo "Refusing to build: set --build-arg REJECT_INSECURE_JWT=ack (or any non-default value) to acknowledge no JWT secrets are embedded." >&2; exit 1)

# Set build environment related variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=UTC

WORKDIR /app

# System updates & minimal utilities (curl for healthcheck, ca-certificates)
# NOTE: Remove packages you do not strictly need to minimize surface.
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirement files
# requirements.txt is the human-friendly file
# requirements-hashed.txt (optional) should contain --require-hashes enforced entries
COPY requirements.txt ./requirements.txt

# (Optional) If you include a requirements-hashed.txt in the build context, it will be copied.
# Do NOT use shell redirection in COPY instruction; Dockerfile does not support it.
COPY requirements-hashed.txt ./requirements-hashed.txt

# Create virtual environment (optional; here we use system site-packages directly)
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with hash verification if lock file present and non-empty
# SECURITY: No fallback to --trusted-host. Build fails if verification fails.
# For production, always provide requirements-hashed.txt with pip-compile --generate-hashes
RUN if [ -f requirements-hashed.txt ] && [ -s requirements-hashed.txt ]; then \
        echo "Using hashed dependency verification (requirements-hashed.txt)"; \
        pip install --no-cache-dir --require-hashes -r requirements-hashed.txt; \
    else \
        echo "WARNING: requirements-hashed.txt not found or empty - using unhashed install (NOT RECOMMENDED FOR PRODUCTION)"; \
        echo "For production builds, generate requirements-hashed.txt with: pip-compile --generate-hashes requirements.txt"; \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Optional: Generate CycloneDX SBOM (can be skipped by removing lines)
# This gives you an sbom.json artifact for compliance / scanning.
RUN pip install --no-cache-dir cyclonedx-bom && \
    cyclonedx-py -o sbom.json || (echo "CycloneDX generation failed (continuing)"; true)

# Copy application source (builder keeps full code to run compile step)
COPY src/ ./src

# Pre-compile Python bytecode (optional performance / tamper evidence)
RUN python -m compileall -q src

# -----------------------------
# Stage 2: Runtime (slim)
# -----------------------------
FROM python:3.11-slim AS runtime

# Runtime environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=UTC \
    # Optionally run with Python optimization (-O) by setting below:
    PYTHONOPTIMIZE=1

WORKDIR /app

# OS hardening: minimal updates; remove apt caches immediately
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (uid 1001) and group
RUN useradd -r -u 1001 -d /app -s /usr/sbin/nologin graphix && \
    mkdir -p /app && chown -R graphix:graphix /app

# Copy only necessary Python site-packages and application code from builder
# This reduces image size and avoids build tools presence.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src
# Copy generated SBOM (optional)
COPY --from=builder /app/sbom.json ./sbom.json

# Add hardened entrypoint script
# This updated script enforces:
# - JWT secret presence
# - Minimum length >= 32 chars
# - Rejects known weak patterns
# - Ensures urlsafe compatibility (basic check)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod 0555 /app/entrypoint.sh

# Application database location environment variable example (SQLite default)
ENV SQLALCHEMY_DATABASE_URI="sqlite:///graphix_api.db"
ENV PYTHONPATH=/app

# Expose application port (Flask / FastAPI / Graphix API Server)
EXPOSE 5000

# Healthcheck using curl (depends on app exposing /health endpoint)
# If your health endpoint differs, modify accordingly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

# Switch to non-root user
USER graphix

# Entrypoint ensures runtime secrets are provided securely
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden by docker run arguments)
# Adjust if you want to run another service (e.g., full_platform.py)
CMD ["python", "src/api_server.py"]