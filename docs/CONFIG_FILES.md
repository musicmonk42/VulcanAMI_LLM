# Configuration Files Reference

## 1. auto_apply_policy.yaml
Controls gate pipeline: lint → tests → security → performance → smoke; budgets (max_files, max_total_loc); adversarial flags (risk_score_max, adversarial_detected must be false).

## 2. crew_config.yaml
Defines multi-agent participants:
- agent manifest (id, traits, trust baseline)
- compliance_controls (control_id → status)
- escalation_paths (fallback resolution strategies)

## 3. graphix_core_ontology.json (Development)
Prototyping ontology; flexible, includes classes Node, Edge, Concept; fewer lifecycle constraints.

## 4. specs/formal_grammar/graphix_core_ontology.json (Canonical)
Versioned URIs; lifecycle states: active, deprecated, experimental, superseded; node.config schema enforcement.

## 5. type_system_manifest.json
Primitive bounds (string length, numeric ranges), dataclass field definitions, enum listings (ValidationOutcome, ObjectiveType, etc.).

## 6. hardware_profiles.json
Backend profile metrics:
- latency_ms
- throughput_tops
- energy_per_op_nj
- serialization_cost_mb_per_s
- health_score baseline

## 7. intrinsic_drives.json
Triggers, objectives, constraints, risk scoring weight matrix, rollback triggers list, never_modify globs.

## 8. profile_development.json / profile_testing.json
Environment-specific toggles (learning, tracing verbosity, concurrency caps).

## 9. tool_selection.yaml
Weighted arbitration for external API/tool invocation (cost vs latency vs reliability vs energy).

## 10. helm_chart.yaml
Kubernetes deployment: liveness/readiness probes (/healthz,/readyz), resource requests, secret mounts, replica scaling hints.

## 11. language_evolution_registry.py
Governance constants: grammar versions, voting thresholds range, replay window seconds, proposal size depth constraints.

## 12. Integrity & Change Control
Hash snapshotting per config file; periodic integrity sweeps; governance proposals required for canonical ontology modifications; planned signed bundle distribution.
