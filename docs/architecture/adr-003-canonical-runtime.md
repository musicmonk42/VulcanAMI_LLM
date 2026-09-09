# ADR 003: Canonical runtime composition

## Decision

Docker runs `vulcan.runtime.app:app` with `PYTHONPATH=/app/src`.  `vulcan` is
the sole production package root.  Its static ASGI route table is complete
before lifespan starts.  Lifespan calls `compose_runtime()` once and attaches
only `RuntimeContainer` plus readiness metadata to application state.

`CompositionSpecification` and `compose_runtime()` now form the sole typed
production composition root. They construct the world proposal edge, safety
validator, state authorities, Constitutional Transaction Service, durable
episode and epistemic stores, audit projector, governed memory, alignment,
capability manifest, language ports, safety finalizer, domain lookup, and kernel
exactly once. `RuntimeContainer` records an explicit reverse close-order graph;
startup rejects missing or multiply owned nodes before publishing the runtime.

The old boundary constructed `ProductionDeployment`, reflectively discovered
its collective dependencies, and used `setattr` to inject learning and domain
owners after construction. The new boundary passes typed `RuntimeOwnerInputs`
to the container and performs no post-construction authority injection. The
canonical production module neither imports nor instantiates
`vulcan.orchestrator.deployment.ProductionDeployment`.

`RuntimeContainer.new(deployment=...)` remains the named
`legacy-deployment-input` test/migration adapter. It may only project a legacy
world proposal, safety validator, and continual-learning proposal into typed
inputs; it owns no authority and is removed when downstream tests and research
entrypoints use `CompositionSpecification`. `LegacyWorldReadOnlyAdapter` is the
only production-facing legacy World Model edge; it exposes readiness and a
snapshot identity, but no reasoning, domain mutation, policy, or effect method.
Remove it when the registered Graphix proposal path owns all supported semantic
proposals and Wave 2.3 closes the serving import graph. The kernel accepts framework-free
`KernelRequest` values and one request-scoped `CognitiveCase`; its compatibility
adapter is the only route to the retained legacy executor.

`/v1/chat`, `/v1/chat/orchestrated`, and `/vulcan/v1/chat` are aliases for one
handler and one kernel call.  The old `src.full_platform` parent/mounted child
composition and Graphix language-layer coordinators are not in Docker's
production import closure.  They remain research/legacy code pending deletion;
rollback is selecting the prior container image/commit, not mutating a running
container.

## Limits

Typed composition is M3 on the source-level canonical runtime path. Gate E is
not complete: production import closure and exact built-image restart
qualification remain outstanding. The legacy executor is a bounded compatibility adapter while semantic ingress,
evidence, memory reconstruction, and output firewall work are deferred.  It is
not a claim of production readiness or correctness.
