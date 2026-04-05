# Project Backlog

## Blockers (Must Fix Before Progress)

### Security Blockers
- [x] [S1] Full /qor-audit pass required — PASSED (Entry #5 in META_LEDGER)
- [ ] [S2] Secret scanning pre-commit hook not yet configured in QoreLogic governance

### Development Blockers
- [x] [D1] Section 4 Razor compliance audit needed — Remediation plan approved and implementation in progress

## Backlog (Planned Work)
- [x] [B1] Establish /qor-audit baseline for all security-touching modules — Phase 1 security fixes complete (V1-V5)
- [ ] [B2] Map existing test coverage gaps against critical safety paths
- [ ] [B3] Validate governance consensus thresholds against production requirements
- [ ] [B4] Remove logging.basicConfig() from 77 files (centralized logging created, mechanical removal pending)
- [x] [B5] Rewire full_platform.py imports to use src/platform/ modules — Route modules use globals.py, God file callers redirected
- [ ] [B6] Further decompose services.py (565 lines) to ≤250 lines
- [ ] [B7] Remove duplicate class bodies from God files (deferred from import rewiring)

## Wishlist (Nice to Have)
- [ ] [W1] Automated QoreLogic chain validation in CI pipeline
- [ ] [W2] Risk grade auto-detection from commit diff analysis

---
_Updated by /qor-* commands automatically_
