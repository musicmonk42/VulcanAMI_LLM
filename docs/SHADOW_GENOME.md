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
_Shadow Genome tracks failure patterns to prevent repetition._
