# Shadow Genome — Failure Record

---

## Failure Entry #1

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-2026-04-05
**Failure Mode**: SECURITY_STUB

### What Failed
PII detection endpoint (`src/pii_service.py:404-422`) — a live API endpoint that returns `pii_found=False` with `confidence=0.95` for all inputs.

### Why It Failed
Stub logic ships as a functional endpoint. Callers receive a high-confidence "no PII" response when zero detection was performed. In an L3 GDPR-compliant system, this creates false compliance — data subjects' PII flows through undetected while the system reports it was checked.

### Pattern to Avoid
Never ship stub detection/validation endpoints that return success. If a capability is not implemented, the endpoint must return an error or 501 Not Implemented — not a fake positive result.

### Remediation Attempted
Not yet. Blocked pending fix.

---

## Failure Entry #2

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-2026-04-05
**Failure Mode**: SECURITY_STUB

### What Failed
Authentication defaults to `AuthMethod.NONE` (`src/full_platform.py:456,940-941`). Missing env vars silently disable all auth.

### Why It Failed
Fail-open security design. The system should fail-closed: if credentials are not configured, refuse to start rather than running unauthenticated.

### Pattern to Avoid
Never default security-critical configuration to the permissive state. Auth, encryption, and access control must require explicit opt-out, not implicit opt-in.

### Remediation Attempted
Not yet. Blocked pending fix.

---

## Failure Entry #3

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-2026-04-05
**Failure Mode**: COMPLEXITY_VIOLATION

### What Failed
Section 4 Razor compliance — all four checks failed. Worst offenders: `world_model_core.py` (9,577 lines, 38x over), `_execute_agent_task` (~1,587 lines, 40x over), `audit_log.py` (9 nesting levels, 3x over).

### Why It Failed
Organic growth without decomposition discipline. God files and God functions accumulate when there is no automated enforcement of size limits.

### Pattern to Avoid
Functions exceeding 40 lines and files exceeding 250 lines signal missing abstractions. Enforce limits in CI (linting rules) to prevent regression.

### Remediation Attempted
Not yet. Blocked pending refactoring.

---

## Failure Entry #4

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-2026-04-05
**Failure Mode**: DUPLICATION

### What Failed
Triplicated domain logic: consensus (3 implementations), audit logging (5+ implementations), safety validation (3 implementations). No shared interfaces or base classes.

### Why It Failed
Independent development of the same capability without coordination. Each subsystem created its own version rather than importing from a canonical source.

### Pattern to Avoid
Before creating a new class for a cross-cutting concern (logging, auth, validation, consensus), search for existing implementations. Establish canonical modules for shared capabilities early.

### Remediation Attempted
Not yet. Blocked pending consolidation.

---

## Failure Entry #5

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-PLAN-2026-04-05
**Failure Mode**: SECURITY_STUB

### What Failed
Remediation plan security fixes (PV-1, PV-2, PV-3) use denylist-style environment variable checks (`!= "production"`, `!= "development"`) and empty-string defaults for credentials.

### Why It Failed
Denylists fail open on unexpected values. `VULCAN_ENV=staging`, `VULCAN_ENV=prod`, or any typo bypasses the check. Empty-string api_key default combined with `_safe_compare("", "")` creates a concrete auth bypass via `hmac.compare_digest`.

### Pattern to Avoid
Security gates must use allowlists (`in ("development", "test")`), never denylists (`!= "production"`). Credential defaults must be `None`, never empty string. A single `is_dev_env()` function should be the single source of truth for environment classification.

### Remediation Attempted
Plan revision required. VETO issued on plan.

---

## Failure Entry #6

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-PLAN-2026-04-05
**Failure Mode**: COMPLEXITY_VIOLATION

### What Failed
Plan estimates for God file decomposition are mathematically infeasible. Claims "~250 lines per module" but actual line counts range from 177 to 3,520 per grouping. `reasoning_modes.py` (2,075 lines), `introspection.py` (1,720 lines), and remaining WorldModel orchestrator (3,520 lines) each need further decomposition into 7-14 sub-modules.

### Why It Failed
Estimates were not based on actual line counts. The plan assumed equal distribution across extracted modules without measuring the real size of each logical grouping.

### Pattern to Avoid
Always measure actual line counts before proposing decomposition targets. Use `grep -n "def \|class "` and line gaps to estimate grouping sizes. Never claim "~250 lines" without verifying the math adds up to the total.

### Remediation Attempted
Plan revision required. VETO issued on plan.

---

## Failure Entry #7

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-PLAN-R2-2026-04-05
**Failure Mode**: COMPLEXITY_VIOLATION

### What Failed
Phase 4 `full_platform.py` decomposition plan claims "5,235 -> 5 modules + entry point" but destination modules sum to only ~1,030 lines (20% of actual). The "slimmed entry point (~150 lines)" claim is infeasible -- the file contains 75 top-level definitions including ~40 route handlers, ~10 proxy functions, middleware, background tasks, and lifespan management that occupy ~4,200 lines with no assigned destination.

### Why It Failed
The plan extracted only the obviously named classes (JWTAuth, SecretsManager, FlashMessage, AsyncServiceManager, Settings) but did not inventory the bulk of the file: route handlers, proxy endpoints, middleware stack, background initialization, and utility functions. The "slimmed entry point" line estimate appears to have been set by aspiration rather than measurement.

### Pattern to Avoid
When decomposing a God file, inventory ALL content -- not just the extractable classes. Route handlers, middleware, and utility functions are often the majority of a large entry-point file. Use `grep -c "^def \|^async def \|^class "` to count definitions, then estimate lines per definition to verify the total adds up. The destination module estimates must sum to within 5% of the actual file size.

### Remediation Attempted
Plan revision required. VETO issued on Revision 2.

---

## Failure Entry #8

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-REWIRING-2026-04-05
**Failure Mode**: SECURITY_STUB

### What Failed
Import rewiring plan proposes `settings = get_settings()` at module level in the thinned `full_platform.py` shell and in every route module. This triggers `UnifiedPlatformSettings.__init__()` at import time, which raises `ValueError` when no auth credentials are configured and `VULCAN_ENV` is not in the dev allowlist.

### Why It Failed
The original code used lazy function-scoped imports (`from src.full_platform import settings` inside route handler bodies) to defer Settings construction until request time. The plan converts these to eager module-level calls without accounting for the fail-closed auth validation in the constructor. The security behavior is correct (fail-closed), but the trigger point moves from request-time to import-time, breaking all test imports.

### Pattern to Avoid
When refactoring lazy imports to eager module-level imports, always check whether the imported object's constructor has side effects (network calls, validation, env checks). If it does, keep the lazy pattern or separate construction from validation. Specifically: Pydantic Settings classes that perform auth validation in `__init__` must not be instantiated at module level.

### Remediation Attempted
Plan revision required. VETO issued.

---

## Failure Entry #9

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-REWIRING-2026-04-05
**Failure Mode**: GHOST_PATH

### What Failed
Import rewiring plan claims to break circular imports between `full_platform.py` and `src/platform/routes_*.py`, but only provides a concrete rewiring pattern for `settings`. Route modules also lazy-import `app`, `service_manager`, and `flash_manager` from `src.full_platform` (12 occurrences across 5 route files). No accessor pattern is specified for these three globals.

### Why It Failed
The plan focused on the most obvious singleton (`settings`) and used "Same pattern" / "etc." for the remaining globals without working through the concrete import paths. `app`, `service_manager`, and `flash_manager` are runtime-constructed objects in `full_platform.py` that have no equivalent `get_*()` accessor in `src/platform/`. Without one, route modules must still import from `full_platform`, leaving the circular dependency intact.

### Pattern to Avoid
When claiming to break circular imports, enumerate ALL symbols that flow across the circular boundary and provide a concrete resolution for each one. Do not use "same pattern" when the pattern does not apply (there is no `get_service_manager()` function).

### Remediation Attempted
Plan revision required. VETO issued.

---

## Failure Entry #10

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-REWIRING-2026-04-05
**Failure Mode**: COMPLEXITY_VIOLATION

### What Failed
Plan proposes adding re-exports to `src/vulcan/orchestrator/__init__.py` and `src/vulcan/reasoning/selection/__init__.py` that conflict with existing re-exports. For example, `AgentPoolProxy` and `is_main_process` are already imported from `.agent_pool` in the existing `__init__.py`, but the plan proposes importing them from `.agent_pool_proxy`. Similarly, `SelectionMode` is already imported from `.tool_selector` but the plan proposes it from `.selection_types`.

### Why It Failed
The plan did not audit the existing `__init__.py` files before proposing additions. The Phase 2 re-export instructions were drafted against the architecture plan's expected state rather than the actual codebase state after implementation.

### Pattern to Avoid
Before proposing `__init__.py` modifications, always read the current file contents. Verify that proposed imports do not shadow or conflict with existing imports. When extracted modules re-export symbols that the God file also exports, specify explicitly whether the new import replaces or supplements the old one.

### Remediation Attempted
Plan revision required. VETO issued.

---

## Failure Entry #11

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-REWIRING-R2-2026-04-05
**Failure Mode**: SECURITY_STUB

### What Failed
Import rewiring plan Revision 2 proposes a `__getattr__` handler in the `full_platform.py` shell that calls `create_app()` on every attribute access of `full_platform.app`, with no singleton caching.

### Why It Failed
Python's `__getattr__` at module level is called every time an attribute is not found in the module's namespace. Unlike `get_settings()` in `globals.py` which uses a double-checked lock pattern, the `__getattr__` handler returns `create_app()` directly without caching the result as a module attribute. Each `from src.full_platform import app` in a different route handler triggers a fresh `create_app()`, constructing duplicate `FastAPI` instances, `UnifiedPlatformSettings` instances, and `AsyncServiceManager` instances. The `init_app()` call inside `create_app()` overwrites globals on each invocation, creating a race condition.

### Pattern to Avoid
When using module-level `__getattr__` for lazy initialization, always cache the result by assigning it to the module namespace (e.g., `globals()["app"] = _app`) so that subsequent accesses find the attribute directly and `__getattr__` is not invoked again. Module-level `__getattr__` is a fallback mechanism, not a factory -- treat it like a `@property` with caching.

### Remediation Attempted
Plan revision required. VETO issued.

---

## Failure Entry #12

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-REWIRING-R2-2026-04-05
**Failure Mode**: GHOST_PATH

### What Failed
Import rewiring plan Revision 2 claims to rewire all `routes_health.py` imports from `full_platform` to `globals`, but `routes_health.py:251-256` imports `_services_init_complete` and `_services_init_failed` -- mutable boolean flags with no corresponding accessor in `globals.py`.

### Why It Failed
The plan inventoried the 4 main singletons (`app`, `settings`, `service_manager`, `flash_manager`) but missed the 2 mutable state flags used by the `/health/ready` endpoint. These flags are set by background initialization tasks and read by the readiness check. They are not singletons with constructors -- they are runtime booleans that change state during the application lifecycle.

### Pattern to Avoid
When rewiring imports away from a God file, grep for ALL symbols imported from that file across the entire codebase -- not just the obvious singletons. Module-level boolean flags, constants, and utility functions are easy to overlook but equally critical. Use `grep -r "from src.full_platform import" src/` and enumerate every unique symbol.

### Remediation Attempted
Plan revision required. VETO issued.

---

## Failure Entry #13

**Date**: 2026-04-05T00:00:00Z
**Verdict ID**: GATE-TRIBUNAL-CLEANUP-2026-04-05
**Failure Mode**: COMPLEXITY_VIOLATION

### What Failed
Phase 2 of cleanup plan proposes splitting `src/platform/services.py` (565 lines) into two files, but the resulting `services.py` containing `AsyncServiceManager` would be ~350 lines -- exceeding the 250-line Section 4 Razor limit by 40%.

### Why It Failed
The plan estimated "~300 lines" for the post-split `services.py` without measuring the actual line range. `AsyncServiceManager` starts at line 229 and runs to line 565 (337 lines of class body). Adding an import preamble for symbols extracted to `service_imports.py` brings the total to ~350 lines. The estimate was aspirational rather than measured -- the same failure pattern documented in Shadow Genome Entry #6 and Entry #7.

### Pattern to Avoid
When splitting a file, measure the actual line count of each resulting segment BEFORE committing to the split strategy. If a single class exceeds 250 lines, the class itself must be further decomposed -- a file split alone is insufficient. Use `grep -n "class ClassName"` and `wc -l` to verify the math adds up.

### Remediation Attempted
Plan revision required. VETO issued.

---
_Shadow Genome tracks failure patterns to prevent repetition._
