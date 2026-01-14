# 🧠 AI Training & Self-Evolution Guide (Graphix / VULCAN-AI)

**Version:** 2.2.0 
**Updated:** 2024-11-11

---

## 1. Purpose

This guide describes advanced training and continual self-improvement workflows for Graphix/VULCAN-AI:

- Genetic / tournament evolution of IR graphs
- Zero‑data bootstrapping (intrinsic proposal generation)
- RLHF / supervised preference alignment
- Hardware-aware reward shaping (energy & latency)
- Long-term memory (LTM) replay avoidance

---

## 2. Environment Setup

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
export PYTHONPATH=.
```

Optional RL components (ensure `torchrl` availability). 
Set dataset path for RLHF:

```bash
export DATASET_PATH=./beavertails.csv
```

---

## 3. Core Workflows

### 3.1 Tournament-Based Evolution

Command:

```bash
python scripts/run_sentiment_tournament.py --mode offline --generations 6 --population 8 --seed 42
```

Evolution Cycle:
1. Load baseline graph → mutate variants
2. Score fitness (accuracy minus resource penalty)
3. Select champions
4. Persist `evolution_champions/gen_*_champion.json`

Fitness Considerations: 
`score = accuracy - alpha*(latency_ms) - beta*(energy_nj)`

Adjust alpha/beta in script for hardware-aware minimization.

---

### 3.2 Zero-Data Autonomous Bootstrapping

```python
from src.evolution_engine import EvolutionEngine
engine = EvolutionEngine()
engine.train_zero_data(num_epochs=100)
engine.replay_ltm()
```

- Random structural exploration
- Policy gradient or bandit heuristics over mutation operators
- Avoid reintroducing failed patterns (LTM vector similarity gate)

---

### 3.3 RLHF Preference Alignment

```python
from src.evolve.self_optimizer import SelfOptimizer
optimizer = SelfOptimizer()
optimizer.train_on_dataset(dataset_path="beavertails.csv")
```

Targets:
- Safety classification
- Quality preference scoring
- Rejection pattern reinforcement

Fallback: If dataset missing, mock data inserted (lower fidelity).

---

### 3.4 Hardware-Integrated Reward Shaping

Energy & latency metrics from hardware/emulator integrated into reward:

`reward = performance_score - γ * energy_nj - δ * latency_ms`

Tune γ / δ for sustainability weighting.

---

## 4. Validation & Regression Control

Run:

```bash
pytest src/run_validation_test.py -v --golden-files evolution_champions/gen_05_champion.json
```

Checks:
| Layer | Purpose |
|-------|---------|
| Schema | Structural integrity |
| Ethics | NSOAligner multi-model consensus |
| Execution | Runtime stability |
| Photonic Params | Safe noise/compression thresholds |

---

## 5. Memory & Replay Avoidance

LTM (FAISS or vector index):
- Store embeddings of successful proposals
- Reject or attenuate similarity > threshold (e.g., 0.95 cosine)
- Mark failed embeddings for negative sampling

---

## 6. Metrics to Track

| Metric | Interpretation | Threshold |
|--------|---------------|-----------|
| `mutation_validity_rate` | % non-rejected mutations | > 90% |
| `ltm_recall_rate` | Prevention of repeat failures | > 95% |
| `energy_per_op_nj` | Sustainability improvement | Decreasing over generations |
| `alignment_agreement` | Multi-LLM consensus | > 98% |
| `rollback_rate` | Unsafe proposal rollback success | 100% |

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Low Fitness Improvement | Overfitting to single metric | Adjust multi-objective weighting |
| Repeated Unsafe Mutations | Inadequate safety penalty | Increase risk cost or tighten NSOAligner rules |
| No Hardware Metrics | Emulator fallback only | Validate hardware key, enable backend availability |
| RLHF Flat Curve | Poor dataset quality | Clean dataset; ensure balanced labels |
| Replay Flood | Similarity threshold too low | Raise cosine similarity limit |

---

## 8. Best Practices

- Pin reproducible seeds (`--seed`).
- Archive champion lineage (hash + timestamp).
- Use incremental RLHF sessions (avoid catastrophic forgetting).
- Visualize fitness/energy trends (Prometheus + Grafana panels).
- Enforce mutation operator diversity (prevent stagnation).

---

## 9. References

| Component | File |
|-----------|------|
| Evolution Engine | `src/evolution_engine.py` |
| Self Optimizer | `src/evolve/self_optimizer.py` |
| Tournament Script | `scripts/run_sentiment_tournament.py` |
| Validation Suite | `src/run_validation_test.py` |
| Governance Loop | `src/governance_loop.py` |

---

## 10. Extension Ideas

- Multi-objective Pareto front analysis
- Curriculum scheduling (progressive complexity graphs)
- Contrastive safety embedding training
- Cross-hardware adaptation via transfer rewards

---

> Continual learning must remain bounded by safety/ethics invariants—never bypass NSO alignment gates.
