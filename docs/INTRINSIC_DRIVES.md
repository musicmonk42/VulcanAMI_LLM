# Intrinsic Drives (Self-Improvement Policy)

## 1. Purpose
Controlled self-improvement targeting performance, maintainability, safety augmentation.

## 2. Triggers
startup, error surge, latency degradation, periodic, low_activity.

## 3. Objectives & Metrics
- optimize_performance (p95 latency delta)
- improve_test_coverage (coverage delta)
- enhance_safety_checks (new gates added)
- fix_known_bugs (closed issue list)
- reduce_energy_consumption (energy_nj total slope)

## 4. Constraints
rate limits, session duration cap, protected file globs (never_modify), rollback_on_failure enforcement.

## 5. Validation Pipeline
dry-run diff → lint → unit/integration tests → security scan → performance micro-benchmark → risk score → apply/abort.

## 6. Risk & Blast Radius
Factors: LOC changed, critical file touch count, complexity shift, test coverage delta.
Blast categories: low (isolated), medium (cross-module), high (core subsystem).

## 7. Adaptive Loop
Weight rebalancing based on success/failure (increase safety weight after failure, performance weight after success).

## 8. Rollback
Trigger: regression metrics, safety violation, unpredictable latency spike. Revert patch bundle + audit event.

## 9. Metrics
intrinsic_sessions_total, intrinsic_session_duration_seconds, rollback_events_total, risk_score_session, objective_weight_change_total.

## 10. Governance Interaction
High-risk sessions escalate to proposal gating; safe small improvements auto-apply under risk threshold.

## 11. Future
Reinforcement learning for weight vector; ML predictive impact modeling; cross-instance cooperative improvements.
