# Runtime convergence reachability audit (before changes)

Baseline fetch: `git fetch origin` was attempted but the environment returned HTTP 403 for GitHub. The local HEAD is `c2e53c76d8a933c7dee701ba742a99650eede646`, matching the audited reference SHA.

## Owner graph before editing

Docker imported `vulcan.runtime.app:app`, whose lifespan called `compose_runtime()`. `compose_runtime()` constructed `ProductionDeployment(get_config())`, then `RuntimeContainer.new()`. The container derived `world_state` and `safety` from `deployment.collective.deps`, constructed `compose_governed_memory()` without passing the same audit owner, created a random `/tmp/vulcan-runtime-*` root when no deployment root existed, then created `CanonicalAudit`, `AlignmentRegistry`, and `PersistentDomainRegistry`.

Self-improvement was not owned by `RuntimeContainer`. `SelfImprovementDrive` could be directly constructed by tests, world-model code, or legacy endpoints and could construct a process-global CSIU singleton via `get_csiu_enforcer()`.

## Construction order before editing

1. Docker/ASGI loads `vulcan.runtime.app:app`.
2. FastAPI lifespan calls `compose_runtime()`.
3. `ProductionDeployment` builds collective dependencies.
4. `RuntimeContainer.new()` locates world model and safety from deployment dependencies.
5. Memory is composed separately.
6. Audit/alignment/domain are placed under deployment root or a random `/tmp` root.
7. `CognitiveKernel` is composed.
8. Any self-improvement drive could later be constructed independently.

## Readiness graph before editing

`RuntimeContainer.readiness()` checked deployment, world_state, kernel, safety, memory, language ports, audit, alignment, and domain registry. It did not verify durable root identity, CSIU accounting, approval store, journal reconciliation, drive readiness, policy digests, or scheduler uniqueness.

## Shutdown graph before editing

`RuntimeContainer.close()` deduplicated the fixed tuple of language ports, memory, alignment, audit, domain registry, kernel, safety, world_state, and deployment. It did not traverse a runtime-owned self-improvement graph.

## Self-improvement schedulers before editing

`SelfImprovementDrive.step()` called `should_trigger()` internally; other orchestration/WorldModel paths could also call drive methods directly. No runtime owner proved exactly one scheduler.

## Source mutation primitives before editing

`SelfImprovementDrive._apply_file_modification()` wrote source directly. `SelfImprovementDrive.apply_improvement()` staged a temp write path. `GovernedSelfImprovementTransaction._atomic_write()` wrote source through policy/approval/audit gates.

## Git add/commit/push paths before editing

`SelfImprovementDrive._commit_to_version_control()` ran `git add`, `git commit`, and `git rev-parse`, and called `_push_to_remote()`. `_push_to_remote()` ran `git push`. Governed transactions intentionally did not commit or push.

## Approval paths before editing

Legacy `request_approval()`, `approve_pending()`, and `reject_pending()` mutated drive state. Governed transactions used `ApprovalStore` records directly. No canonical authenticated approval API existed in `vulcan.runtime.app`.

## CSIU telemetry producers and consumers before editing

`SelfImprovementDrive` accepted a dotted-key metrics provider and cached values. `CSIUEnforcement` accepted typed snapshots but legacy drive code could assemble metrics. CSIU lifecycle events were held in an in-memory audit trail rather than the shared `CanonicalAudit`.

## Persistence paths before editing

Audit, alignment, and domain registry used the container root. Memory used its own composition. CSIU could use an optional durable store. Approval records used whichever `ApprovalStore` a caller supplied. Transaction unresolved state was an in-memory boolean.

## Docker-to-attempted-improvement call graph before editing

`Dockerfile` -> ASGI command importing `vulcan.runtime.app:app` -> lifespan -> `compose_runtime()` -> `ProductionDeployment` -> `RuntimeContainer.new()` -> `CognitiveKernel`/`WorldModel`. A later improvement attempt could come from `WorldModel`/collective/endpoint code constructing or accessing `SelfImprovementDrive`, calling `step()`, `generate_improvement_action()`, then legacy `_execute_improvement()`/`_apply_file_modification()`/`_commit_to_version_control()` or `apply_improvement()`, bypassing a single runtime-owned transaction owner.
