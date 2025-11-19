# Ontology & Semantic Contracts

## 1. Purpose
Stable semantic vocabulary for node/edge types enabling consistent validation, governance evolution, and provenance clarity.

## 2. Canonical vs Development
| Aspect | Development | Canonical |
|--------|------------|-----------|
| Stability | Mutable | Versioned |
| Experimental Use | Permitted freely | Gated & marked |
| Update Path | Direct edit | Governance proposal |
| Enforcement | Soft warnings | Hard validation + lifecycle tracking |

## 3. Lifecycle States
active | deprecated | experimental | superseded

## 4. Node Type Semantics (Sample)
| Type | Purpose | Risks | Controls |
|------|---------|-------|----------|
| CONST | Static value | Large memory params | Param size heuristic |
| ADD / MULTIPLY | Arithmetic | Input type mismatch | Type validator |
| EMBED | External embedding | Provider latency, cost | SLA contract + caching |
| GenerativeAINode | Text generation | Prompt injection | Budget + safety gating |
| PHOTONIC_MVM | Analog simulation | Noise drift | Noise std threshold |
| MEMRISTOR_MVM | CIM emulation | Accuracy variance | Post-check heuristics |
| ProposalNode | Governance proposal creation | Spam duplicates | Similarity dampening |
| ConsensusNode | Vote aggregation | Manipulated weights | Trust weighting + anomaly detection |

## 5. Edge Semantics
dependency (acyclic), soft_dependency (non-critical ordering recommendation), provenance (lineage link), conflict (mutual exclusion sets), consensus (governance modeling flows).

## 6. Validation Integration
Ontology ensures node type whitelisting; unknown types produce warnings unless marked forbidden or dangerous (ExecNode etc.).

## 7. Evolution Workflow
Submit → validate ontology diff → vote → apply → version bump (SemVer rules: major on breaking semantics, minor additive, patch metadata).

## 8. Deprecation Strategy
Grace period; superseded_by hint; enforcement after retirement date.

## 9. Consistency Rules
- Unique URIs
- No alias collisions
- Experimental types carry gating flag
- Edge types cannot impersonate dependency semantics improperly.

## 10. Expansion Patterns
CompositeNode, ConstraintNode, EthicsNode, QuantumNode (simulation pipeline).
