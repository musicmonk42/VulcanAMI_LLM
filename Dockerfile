# =============================================================================
# VulcanAMI Full Platform - Unified Secure Container Build
# =============================================================================
# This is the MAIN Dockerfile for deploying the complete VulcanAMI platform.
# Used by Railway, and recommended for single-container deployments.
#
# WHAT THIS BUILDS:
# - Complete VulcanAMI platform via src/full_platform.py
# - VULCAN cognitive platform with /vulcan/v1/chat endpoint
# - Graphix Registry API
# - All 71+ integrated services behind a unified interface
#
# FOR MICROSERVICE DEPLOYMENTS:
# Use the service-specific Dockerfiles in docker/ directory:
# - docker/api/Dockerfile    - API Gateway microservice
# - docker/dqs/Dockerfile    - Data Quality Service
# - docker/pii/Dockerfile    - PII Detection Service
#
# SECURITY FEATURES:
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
FROM python:3.10.11-slim AS builder

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
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        git && \
    update-ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/*

# Upgrade pip and setuptools to latest versions
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirement files
# requirements.txt is the human-friendly file
# requirements-hashed.txt (optional) should contain --require-hashes enforced entries
COPY requirements.txt ./requirements.txt

# (Optional) If you include a requirements-hashed.txt in the build context, it will be copied.
# Do NOT use shell redirection in COPY instruction; Dockerfile does not support it.
COPY requirements-hashed.txt ./requirements-hashed.txt

# Copy setup.py and source for local package installation
COPY setup.py ./setup.py

# Create virtual environment (optional; here we use system site-packages directly)
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with hash verification if lock file present and non-empty
# SECURITY: No fallback to --trusted-host. Build fails if verification fails.
# For production, always provide requirements-hashed.txt with pip-compile --generate-hashes
# Check if the file exists, is non-empty, and contains actual package entries (not just comments)
RUN if [ -f requirements-hashed.txt ] && grep -qE '^[^#]' requirements-hashed.txt; then \
        echo "Using hashed dependency verification (requirements-hashed.txt)"; \
        pip install --no-cache-dir --require-hashes -r requirements-hashed.txt; \
    else \
        echo "WARNING: requirements-hashed.txt not found or empty - using unhashed install (NOT RECOMMENDED FOR PRODUCTION)"; \
        echo "For production builds, generate requirements-hashed.txt with: pip-compile --generate-hashes requirements.txt"; \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Optional: Generate CycloneDX SBOM (can be skipped by removing lines)
# This gives you an sbom.json artifact for compliance / scanning.
# hadolint ignore=DL3013,SC2015
RUN pip install --no-cache-dir cyclonedx-bom && \
    cyclonedx-py requirements requirements.txt -o sbom.json || (echo "CycloneDX generation failed (continuing)"; touch sbom.json)

# Copy application source (builder keeps full code to run compile step)
COPY src/ ./src

# Copy configuration files (required by application)
COPY configs/ ./configs/

# Copy demo files (including vulcan_chat.html)
COPY demos/ ./demos/

# Install local package (graphix) if setup.py exists
RUN if [ -f setup.py ]; then \
        echo "Installing local package from setup.py"; \
        pip install --no-cache-dir -e .; \
    fi

# Download spacy language model if spacy is installed
RUN python -m spacy download en_core_web_sm || echo "Spacy model download failed (non-critical)"

# Pre-compile Python bytecode (optional performance / tamper evidence)
RUN python -m compileall -q src

# -----------------------------
# Stage 2: Runtime (slim)
# -----------------------------
FROM python:3.10.11-slim AS runtime

# Runtime environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=UTC \
    # Optionally run with Python optimization (-O) by setting below:
    PYTHONOPTIMIZE=1

WORKDIR /app

# OS hardening: minimal updates; remove apt caches immediately
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/*

# Create non-root user (uid 1001) and group
RUN useradd -r -u 1001 -d /app -s /usr/sbin/nologin graphix && \
    mkdir -p /app && chown -R graphix:graphix /app

# Copy only necessary Python site-packages and application code from builder
# This reduces image size and avoids build tools presence.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs
# Copy demo files (including vulcan_chat.html)
COPY --from=builder /app/demos ./demos
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

# Default port for containerized deployments (can be overridden via PORT env var)
ENV PORT=8000

# Expose application port (Flask / FastAPI / Graphix API Server)
EXPOSE 8000

# Switch to non-root user
USER graphix

# Healthcheck using curl (depends on app exposing /health endpoint)
# Uses PORT env var with default of 8000 if not set
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

# Entrypoint ensures runtime secrets are provided securely
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - runs the full platform which includes:
# - Graphix Registry API
# - VULCAN cognitive platform with /vulcan/v1/chat endpoint
# - All 71+ services integrated behind the chat interface
CMD ["python", "src/full_platform.py"]