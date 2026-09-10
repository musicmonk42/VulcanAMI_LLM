# Canonical Linux serving artifact. Research, cloud, distributed, and development
# extras are deliberately absent from both build resolution and runtime imports.
FROM python:3.11.13-slim-bookworm AS builder
WORKDIR /build
COPY pyproject.toml setup.py README.md LICENSE ./
COPY src/vulcan ./src/vulcan
COPY requirements-runtime.lock ./
RUN python -m pip install --no-cache-dir --require-hashes --target /install -r requirements-runtime.lock \
 && python -m pip wheel --no-deps --no-build-isolation --no-cache-dir --wheel-dir /wheels . \
 && python -m pip install --no-deps --no-index --target /install /wheels/*.whl

FROM python:3.11.13-slim-bookworm AS runtime
ARG SOURCE_COMMIT=unknown
ARG DEPENDENCY_LOCK_DIGEST=unknown
ARG ARCHITECTURE_STATUS_DIGEST=unknown
ARG REJECT_INSECURE_JWT=ack
ARG REQUIRE_HASHES=1
LABEL org.opencontainers.image.revision=$SOURCE_COMMIT \
      org.vulcan.dependency-lock-digest=$DEPENDENCY_LOCK_DIGEST \
      org.vulcan.architecture-status-digest=$ARCHITECTURE_STATUS_DIGEST \
      org.vulcan.qualification-gate="E"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 VULCAN_ENV=production \
    VULCAN_RUNTIME_DURABLE_ROOT=/var/lib/vulcan VULCAN_ENABLE_SELF_IMPROVEMENT=false \
    VULCAN_MEMORY_ENABLED=false VULCAN_CSIU_ENABLED=false VULCAN_LEARNING_ENABLED=false \
    VULCAN_RELEASE_EVIDENCE_ROOT=/app PORT=8000
WORKDIR /app
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY config/capabilities.yaml config/architecture-status.json /app/config/
COPY docs/architecture/ami-invariants.yaml docs/architecture/adr-006-local-language-interface.md /app/docs/architecture/
COPY docs/governance/controls.yaml docs/governance/impact-assessment.yaml /app/docs/governance/
COPY tests/security/test_language_contracts.py /app/tests/security/test_language_contracts.py
RUN useradd -r -u 1001 -d /app -s /usr/sbin/nologin vulcan \
 && install -d -o vulcan -g vulcan -m 0700 /var/lib/vulcan /tmp/vulcan-cache
RUN rm -rf \
      /usr/local/lib/python3.11/site-packages/vulcan/arena \
      /usr/local/lib/python3.11/site-packages/vulcan/orchestrator \
      /usr/local/lib/python3.11/site-packages/vulcan/deployment \
      /usr/local/lib/python3.11/site-packages/vulcan/semantic_bridge \
      /usr/local/lib/python3.11/site-packages/vulcan/reasoning \
      /usr/local/lib/python3.11/site-packages/vulcan/curiosity_engine \
      /usr/local/lib/python3.11/site-packages/vulcan/world_model \
      /usr/local/lib/python3.11/site-packages/vulcan/runtime/semantic.py \
      /usr/local/lib/python3.11/site-packages/vulcan/improvement/offline.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chown root:root /app/entrypoint.sh /usr/local/lib/python3.11/site-packages \
 && chmod 0555 /app/entrypoint.sh \
 && chown -R root:root /app/config /app/docs /app/tests \
 && chmod -R a-w /usr/local/lib/python3.11/site-packages /app/config /app/docs /app/tests
VOLUME ["/var/lib/vulcan"]
EXPOSE 8000
USER vulcan
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=2)" || exit 1
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sh", "-c", "python -m uvicorn vulcan.runtime.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
