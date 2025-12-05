# Graphix IR Test Suite Guide

**Version:** 2.2.0  
**Date:** 2024-11-11

This guide explains how to run and interpret test suites validating Graphix IR’s schema, ethics, execution stability, and photonic emulation fidelity.

---

## 1. Test File Inventory

| File | Purpose |
|------|---------|
| `src/run_validation_test.py` | Unified validation (schema, signature, ethics, execution, photonic params) |
| `tests/test_graph_validation.py` | Schema & structural assertions for golden graphs |
| `tests/test_analog_photonic_emulator.py` | Photonic emulation benchmarks (<0.4 nJ/op) |
| `tests/test_graphix_arena.py` | Full lifecycle (proposal → execution) |
| `tests/test_governance_loop.py` | Governance and consensus validation |
| `tests/test_hardware_dispatcher.py` | Hardware dispatch reliability |

---

## 2. Setup

```bash
pip install pytest pytest-asyncio pytest-xdist jsonschema numpy networkx
export PYTHONPATH=.
```

---

## 3. Running Tests

| Scope | Command | Notes |
|-------|---------|-------|
| Core Validation | `pytest src/run_validation_test.py -v` | Fast feedback |
| Full Suite | `pytest tests --runslow -v` | Includes slow photonic benchmarks |
| Parallel | `pytest -n auto` | Uses xdist to parallelize |
| Specific Filter | `pytest -k analog` | Target subset by keyword |

---

## 4. Expected Output (Sample)

```
test_graph_validation[classifier.json] ... ok
test_stress_validation ................ ok
test_adversarial_validation ........... ok
```

Prometheus sample metrics during run:
- `validation_pass_total{test_type="schema_classifier"}`
- `validation_latency_seconds{test_type="exec_classifier"}`

---

## 5. Metrics & Audits

| Artifact | Location | Purpose |
|----------|----------|---------|
| Prometheus metrics | scrape endpoint | Live reliability stats |
| Audit DB (`audit.db`) | local dev | Security & validation event trail |
| Dashboard JSON | `observability_logs/demo_dashboard.json` | Import into Grafana |

---

## 6. CI/CD Integration (Example GitHub Actions)

```yaml
name: Graphix Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: python -m venv .venv
      - run: .venv/bin/pip install -r requirements.txt pytest pytest-asyncio pytest-xdist
      - run: .venv/bin/pytest src/run_validation_test.py -v
```

Extend for:
- Coverage upload (`pytest --cov`)
- Artifacts retention (failed logs)

---

## 7. Available Test Files

All core test files are available in the `tests/` directory:
- `test_graph_validation.py` - Graph structure and validation
- `test_graphix_arena.py` - Arena orchestration
- `test_governance_loop.py` - Governance and consensus
- `test_hardware_dispatcher.py` - Hardware dispatcher tests
- `test_execution_engine.py` - Execution engine tests

Run any test with: `pytest tests/<test_file>.py -v`

---

## 8. Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Serialization fail | Non-JSON-safe handler output | Update handler or wrap output |
| Photonic test slow | Large tensor size | Reduce dimension config |
| Ethics failure | NSOAligner risk detection | Inspect proposals; adjust patterns |
| Stress test flakiness | Race or unawaited tasks | Audit async handlers & cleanup |

---

## 9. 2024 Feature Coverage

- ITU F.748.53 compression validation
- Multi-model audit & bias taxonomy tagging
- Grok / other provider mock integration
- Analog noise parameter envelope testing

---

## 10. Recommendations

- Use `-n auto` cautiously (ensure thread safety in handlers).
- Pin seeds for reproducibility in photonic tests.
- Integrate test result metrics into Grafana to observe drift.
- Tag slow tests (`@pytest.mark.slow`) for pipeline tuning.

---

> Treat passing tests as baseline confidence, not absolute safety; couple results with audit chain inspection for critical deployments.
