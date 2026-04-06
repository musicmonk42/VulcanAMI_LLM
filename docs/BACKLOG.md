# Project Backlog

## Blockers (Must Fix Before Progress)

### Security Blockers
- [x] [S1] Full /qor-audit pass required — PASSED (Entry #5 in META_LEDGER)
- [x] [S2] Secret scanning pre-commit hook — detect-secrets configured in .pre-commit-config.yaml. Closes #973.

### Development Blockers
- [x] [D1] Section 4 Razor compliance audit needed — Remediation plan approved and implementation in progress

## Backlog (Planned Work)
- [x] [B1] Establish /qor-audit baseline for all security-touching modules — Phase 1 security fixes complete (V1-V5)
- [x] [B2] Map test coverage gaps on safety paths — 8 test files written: 4 safety module tests, 3 edge case tests, 1 decomposition smoke test. Closes #974.
- [ ] [B3] Validate governance consensus thresholds against production requirements
- [x] [B4] Remove logging.basicConfig() from 89 files — 68 files modified, only logging_config.py retains it. Closes #970.
- [x] [B5] Rewire full_platform.py imports to use src/platform/ modules — Route modules use globals.py, God file callers redirected
- [x] [B6] Split services.py (565→216 lines) into 3 files: services.py, service_imports.py, service_lifecycle.py. Closes #971.
- [x] [B7] Remove duplicate class bodies from God files — 3,172 lines removed across 3 files. Closes #972.

## Wishlist (Nice to Have)
- [ ] [W1] Automated QoreLogic chain validation in CI pipeline
- [ ] [W2] Risk grade auto-detection from commit diff analysis

---
_Updated by /qor-* commands automatically_
