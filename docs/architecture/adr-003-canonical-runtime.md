# ADR 003: Canonical runtime composition

## Decision

Docker runs `vulcan.runtime.app:app` with `PYTHONPATH=/app/src`.  `vulcan` is
the sole production package root.  Its static ASGI route table is complete
before lifespan starts.  Lifespan calls `compose_runtime()` once and attaches
only `RuntimeContainer` plus readiness metadata to application state.

`RuntimeContainer` owns the existing `ProductionDeployment`, its one injected
World Model (the canonical World State for this migration), safety validator,
legacy memory facade, and `CognitiveKernel`.  The kernel accepts framework-free
`KernelRequest` values and one request-scoped `CognitiveCase`; its compatibility
adapter is the only route to the retained legacy executor.

`/v1/chat`, `/v1/chat/orchestrated`, and `/vulcan/v1/chat` are aliases for one
handler and one kernel call.  The old `src.full_platform` parent/mounted child
composition and Graphix language-layer coordinators are not in Docker's
production import closure.  They remain research/legacy code pending deletion;
rollback is selecting the prior container image/commit, not mutating a running
container.

## Limits

The legacy executor is a bounded compatibility adapter while semantic ingress,
evidence, memory reconstruction, and output firewall work are deferred.  It is
not a claim of production readiness or correctness.
