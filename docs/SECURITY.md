# Security & Compliance Framework

## 1. Objectives
Integrity, confidentiality, safe autonomy, forensic-grade auditability.

## 2. Threat Matrix
| Threat | Vector | Mitigation |
|--------|--------|-----------|
| Injection | Node params (eval/exec) | Regex pattern block; dangerous type gating |
| Replay | Artifact repeat | Replay window hash gating |
| Privilege Escalation | Trust inflation | Audit anomaly detection; threshold caps |
| Data Exfiltration | Generative outputs | Output whitelisting & redaction filters |
| Resource DoS | Oversized graphs/timeouts | Caps & adaptive timeouts |
| Supply Chain | Dependency compromise | Version pinning, SBOM, SCA scanning |
| Side-channel | Timing inference | Constant-time for crypto operations |
| Autonomous Mutation Abuse | Self-modifying nodes | NSO gates + risk scoring + allowlists |

## 3. Secrets & Key Management
Rotation cadence; external KMS; ephemeral dev vs production separation; no logging of secret material.

## 4. Authentication & Authorization
JWT short TTL + scope; API key for Arena fallback; trust-level stored server-side; rate limiting on login & propose endpoints.

## 5. NSO Gates & Risk Pipeline
Sequence: lint → type → tests → security → performance → smoke → risk score evaluation → adversarial check → apply or manual review.

## 6. Cryptographic Integrity
Audit chain: hash(prev_hash + event_json). Periodic integrity sweeps raising alerts on mismatch.

## 7. Audit & Forensics
Event schema: timestamp, actor, event_type, payload_hash, severity. Replay reconstruction + diff bundling for proposals.

## 8. Privacy Controls
Secret redaction (pattern-based), retention policies (TTL prune), sanitized export for compliance audits.

## 9. Hardening Checklist (Extended)
| Item | Status |
|------|--------|
| Secret rotation implemented | Yes |
| TLS + HSTS enforced | Yes |
| Replay guard active | Yes |
| NSO gates integrated CI | Yes |
| Audit chain integrity sweep | Scheduled |
| Dependency scanning | Enabled |
| Self-modification gating | Enforced |
| Config integrity hashing | Roadmap |

## 10. Incident Response
Detect → Classify → Contain → Eradicate → Recover → Postmortem → Preventive policy update.

## 11. Supply Chain Security
Pinned versions; SBOM generation; tamper detection via hash compare; periodic CVE triage.

## 12. Future Enhancements
Secure enclaves, zero-trust workload identity, ML anomaly classification, policy DSL for declarative safety conditions.
