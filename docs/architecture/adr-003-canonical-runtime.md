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

`/v1/chat` is the sole canonical cognitive ingress; the orchestrated and `/vulcan`
compatibility routes are not deployed. The old `src.full_platform` parent/mounted child
composition and Graphix language-layer coordinators are not in Docker's
production import closure.  They remain research/legacy code pending deletion;
rollback is selecting the prior container image/commit, not mutating a running
container.

## Limits

Typed composition and production import closure are M3 on the canonical runtime path. Gate E remains pending only on exact built-image restart qualification in the hosted workflow. Compatibility adapters retained for research are outside the serving closure.

## Gate E qualification (Prompt 16)

The old packaging boundary installed the repository root as `src.*`, resolved a
broad research dependency graph, and copied that graph plus source into the
serving image. Source-level composition evidence therefore did not qualify the
artifact. The new boundary is the non-editable `vulcan` wheel installed with the
hash-locked server set into a Linux-only, non-root image. A transitive import
policy rooted at `vulcan.runtime.app` denies legacy deployment/orchestration,
retired runtime semantic, Arena, cloud/distributed, experimental memory, and NPT
research namespaces. Startup rejects non-Linux serving before composition.
The image carries the small, read-only release-evidence set required to validate
capability attestations; installed code resolves it through the explicit
`VULCAN_RELEASE_EVIDENCE_ROOT` rather than assuming a repository checkout.
Denied packages and offline source-application modules are physically removed
from the serving image in addition to being excluded from the import graph.

`LegacyWorldReadOnlyAdapter` remains solely for direct research/test composition
and is removed when those callers supply `CanonicalWorldProposalPort` directly.
The `runtime-semantic-import` adapter remains on disk for unconverted research
callers but is outside the serving import closure; remove it when those callers
use `vulcan.graphix.runtime`. The `setup.py` build shim remains for old build
frontends and is removed when supported tooling universally uses PEP 517.

CI builds the exact image and publishes its image digest, source commit,
dependency-lock digest, architecture-status digest, and qualification result.
The same durable volume is used to verify bounded arithmetic, episode/epistemic
audit state, and identical audit retrieval after restart. Gate E becomes complete only when the committed hosted workflow records passing
exact-image evidence for this commit. Native Windows serving is explicitly unsupported.
