# AMI standards crosswalk

This crosswalk is an internal engineering map. It does not claim certification, legal compliance, production readiness, safety certification, erasure, or cryptographic proof.

| Control | Repository purpose | Standards mapped |
| --- | --- | --- |
| `AMI-AUTH-001` | One cognitive authority and authority-lattice enforcement. | NIST AI RMF 1.0 GOVERN; NIST AI RMF GenAI Profile Govern; ISO/IEC 42001; ISO/IEC 23894; MITRE ATLAS; OWASP GenAI Top 10. |
| `AMI-EVID-001` | Immutable evidence records with canonical JSON, SHA-256 digests, UTC timestamps, exact schemas, and no secret-bearing fields. | NIST AI RMF MAP; NIST SSDF 1.1 PO.1; NIST SP 800-218A; ISO/IEC 42005; SPDX; CycloneDX. |
| `AMI-PROMPT-001` | Prompt-injection and model-boundary containment. | OWASP GenAI Top 10; OWASP Top 10; MITRE ATLAS; NIST AI RMF GenAI Profile. |
| `AMI-PRIV-001` | Consent, privacy, right of exit, and cross-person isolation. | NIST AI RMF MAP/MANAGE; ISO/IEC 42001; ISO/IEC 23894; OWASP GenAI Top 10. |
| `AMI-SDLC-001` | Secure development, review, rollback, and draft SSDF tracking. | NIST SSDF 1.1; NIST SP 800-218A; NIST SSDF 1.2 draft tracking only; OWASP Top 10. |
| `AMI-SUPPLY-001` | Supply-chain integrity, SBOMs, signing, and provenance. | SLSA; Sigstore; SPDX; CycloneDX; NIST SSDF 1.1 PO.3; OWASP Top 10. |
| `AMI-LEARN-001` | Learning, model/data updates, CSIU, and governed promotion. | NIST AI RMF MEASURE/MANAGE; ISO/IEC 5338; ISO/IEC 23894; MITRE ATLAS; OWASP GenAI Top 10. |
| `AMI-OPS-001` | Operational resilience, audit capacity, split-brain prevention, and rollback. | NIST AI RMF MANAGE; ISO/IEC 42001; ISO/IEC 23894; NIST SSDF 1.1 RV.3; OWASP Top 10. |

## Standards tracking notes

- NIST SSDF 1.2 is tracked as draft-only until formally adopted by repository governance.
- SPDX, CycloneDX, SLSA, and Sigstore entries define evidence artifact expectations; their presence in this table is not a claim that every current artifact has that evidence.
- Repository constitution IDs remain the controlling local authority for AMI-specific invariants.
