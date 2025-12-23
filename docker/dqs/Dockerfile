# =============================================================================
# DQS (Data Quality System) Service Dockerfile
# =============================================================================

FROM python:3.10.11-slim AS builder

ARG REJECT_INSECURE_JWT="default-super-secret-key-change-me"

RUN test "$REJECT_INSECURE_JWT" != "default-super-secret-key-change-me" || \
    (echo "Refusing to build: set --build-arg REJECT_INSECURE_JWT=ack" >&2; exit 1)

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        git \
        libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools to latest versions
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy setup.py for local package installation
COPY setup.py ./

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/

# Install local package if setup.py exists
RUN if [ -f setup.py ]; then \
        echo "Installing local package from setup.py"; \
        pip install --no-cache-dir -e .; \
    fi

# Download spacy language model if spacy is installed
RUN python -m spacy download en_core_web_sm || echo "Spacy model download failed (non-critical)"

# Pre-compile Python bytecode
RUN python -m compileall -q src

# =============================================================================
# Runtime Stage
# =============================================================================

FROM python:3.10.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=UTC \
    PYTHONOPTIMIZE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libpq5 \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -u 1001 -d /app -s /usr/sbin/nologin dqs && \
    mkdir -p /app && chown -R dqs:dqs /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod 0555 /app/entrypoint.sh

EXPOSE 8080 9145

USER dqs

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
# PRODUCTION MODE: --workers 1 ensures singleton process (no split-brain/ghost processes)
# DO NOT use --reload in production as it spawns a parent watcher + child worker
CMD ["python", "-m", "uvicorn", "src.dqs_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
