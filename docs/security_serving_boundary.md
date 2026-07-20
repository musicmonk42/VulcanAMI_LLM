# Secure serving boundary baseline

This repository's canonical container command is:

```sh
uvicorn src.full_platform:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
```

The production serving boundary is deny-by-default. Public routes are limited to `/`, `/status`, `/health`, `/health/live`, `/health/ready`, `/health/startup`, `/auth/token`, `/vulcan_chat.html`, and static/demo asset prefixes. All other routes require the configured API key or JWT mechanism. Mutation-capable routes require an admin/operator role or a `mutation:write`/`admin` capability, and selected research/service mutation routes are unavailable in the production profile.

Production profile invariants are fail-closed: authentication must be configured, safety level must not be disabled/minimal, and runtime self-improvement/mutation must remain disabled. The entrypoint refuses the previous limited/no-auth downgrade.

Chat request safety is mandatory. Client `enable_safety=false` is ignored, the typed `safety_validator` dependency is used for input and final output decisions, safety exceptions/timeouts fail closed, and all scoped chat responses pass through the finalization seam before serialization.

Health semantics follow Kubernetes probe intent: liveness is a cheap process response, startup reports initialization state, and readiness returns HTTP 503 until mandatory serving-boundary conditions and service initialization are satisfied. Health output avoids secrets and traces.

The production image runs as a non-root user and makes `/app/src`, `/app/configs`, `/app/config`, and `/app/models` read-only. Writable durable authority state is limited to `/var/lib/vulcan`; `/app/data`, `/dev/shm`, and `/tmp/vulcan-cache` are ephemeral runtime scratch locations.

## Transformer authority boundary (sequence item 2)

Production language providers may submit enum-bounded routing proposals or format an existing structured engine result. Provider confidence is diagnostic only. Provider output cannot skip applicability checks, select a direct-answer path, invoke tools, approve policy, or replace absent/low-confidence cognition with an answer. Malformed proposals, unavailable providers, and inapplicable engines produce deterministic routing or an explicit unknown response.

Known limitations: this is a serving-boundary remediation only. Later remediation sequences still need the canonical runtime container, cognitive kernel, typed semantic ingress/egress, memory reconstruction, model replacement, provenance, and formal compliance assessment.
