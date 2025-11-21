# =============================================================================
# API Gateway Service Dockerfile
# =============================================================================

FROM python:3.11-slim AS builder

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
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY app.py ./
COPY configs/ ./configs/

# Pre-compile Python bytecode
RUN python -m compileall -q src app.py

# =============================================================================
# Runtime Stage
# =============================================================================

FROM python:3.11-slim AS runtime

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
        libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -u 1001 -d /app -s /usr/sbin/nologin apiuser && \
    mkdir -p /app && chown -R apiuser:apiuser /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src
COPY --from=builder /app/app.py ./
COPY --from=builder /app/configs ./configs

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod 0555 /app/entrypoint.sh

EXPOSE 8000 9148

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

USER apiuser

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]
