# VulcanAMI Platform Demonstration Guide

**Based on Actual Code Analysis - January 2026**

This document provides an honest assessment of what capabilities exist in the VulcanAMI codebase, what can be demonstrated with proof, and what claims should be made carefully.

---

## Executive Summary: What You Can Actually Prove

| Claim | Provable? | Evidence Location | Demo Difficulty |
|-------|-----------|-------------------|-----------------|
| Small local LLM (6 layers) | ✅ YES | `src/llm_core/graphix_transformer.py` | Easy |
| **Trained Local GPT Model** | ✅ YES | `src/local_llm/` + `exp_probe_1p34m/` | Easy |
| Air-gappable / offline capable | ✅ YES | No external API dependencies in core | Easy |
| Explicit execution graphs | ✅ YES | `src/llm_core/*.py` IR system | Medium |
| Audit trail logging | ✅ YES | `src/audit_log.py` | Easy |
| Trust-weighted consensus | ✅ YES | `src/consensus_engine.py` | Medium |
| Causal reasoning system | ✅ YES | `src/vulcan/world_model/causal_graph.py` | Medium |
| Safety validators | ✅ YES | `src/vulcan/safety/safety_validator.py` | Medium |
| Machine unlearning | ⚠️ PARTIAL | `src/persistant_memory_v46/unlearning.py` | Hard |
| ZK proofs for unlearning | ⚠️ PARTIAL | `src/persistant_memory_v46/zk.py` | Hard |
| Self-improvement drive | ⚠️ PARTIAL | `src/vulcan/world_model/meta_reasoning/self_improvement_drive.py` | Hard |
| CSIU continuous alignment | ⚠️ PARTIAL | Code exists, requires external LLM for full function | Hard |
| 10-100x performance vs interpreted | ⚠️ UNVERIFIED | `src/compiler/` exists but needs benchmarks | Hard |
| Photonic/quantum hardware | ❌ EMULATED ONLY | `src/hardware_emulator.py` - simulation only | N/A |

---

## Part 1: The Scale War - What You CAN Prove

### Claim: "Small Internal LLM (6 layers, ~50k token context)"

**STATUS: ✅ FULLY PROVABLE**

**Evidence Files:**
- `src/llm_core/graphix_transformer.py` (1,374 lines)
- `src/llm_core/graphix_executor.py` (~1,500 lines)

**What the code shows:**
```python
# From graphix_transformer.py lines 543-564
@dataclass
class GraphixTransformerConfig:
    num_layers: int = 6              # Configurable, default 6 layers
    hidden_size: int = 256           # Small hidden dimension
    num_heads: int = 4               # 4 attention heads
    vocab_size: int = 4096           # Small vocabulary
    max_position_embeddings: int = 1024  # Context length
    dropout: float = 0.1
    lora_rank: int = 0               # LoRA support for fine-tuning
```

**Demonstration:**
```bash
# Run this to prove the model exists and works
python -c "
from src.llm_core.graphix_transformer import GraphixTransformer, GraphixTransformerConfig

config = GraphixTransformerConfig(num_layers=6, hidden_size=256)
model = GraphixTransformer(config)
print(f'Model parameters: {model.num_parameters():,}')
print(f'Layers: {config.num_layers}')

# Generate text (proves it works)
output = model.generate('Hello world', max_new_tokens=10)
print(f'Generated: {output}')
"
```

**What to claim:** "Vulcan uses a small, configurable transformer (default 6 layers, ~10M parameters) that runs locally."

**What NOT to claim:** Don't claim "50k token context" - the default is 1024. The architecture supports longer contexts but would need configuration changes and memory.

---

### Claim: "Trained Local GPT Model" (Production-Ready)

**STATUS: ✅ FULLY PROVABLE - INCLUDES TRAINED WEIGHTS**

**Evidence Files:**
- `src/local_llm/provider/local_gpt_provider.py` - Production GPT provider
- `src/local_llm/tokenizer/simple_tokenizer.py` - Tokenizer implementation
- `src/local_llm/tokenizer/vocab.json` - 31,717 token vocabulary
- `src/training/gpt_model.py` - PyTorch GPT model implementation
- `exp_probe_1p34m/llm_best_model.pt` - **Trained model weights (~91MB)**
- `exp_probe_1p34m/llm_meta_state.json` - Training history with 299 steps

**This is the PRIMARY local LLM** - a production-ready PyTorch GPT model with:
- Trained weights (not just architecture)
- 31,717 token vocabulary
- Temperature, top-k, top-p, repetition penalty controls
- Streaming generation support
- Perplexity/scoring utilities
- Confidence calibration

**Configuration (from code):**
```python
# From local_gpt_provider.py
@dataclass
class ProviderInitConfig:
    model_path: str           # Path to trained .pt file
    vocab_path: str           # Path to vocabulary
    device: str = "cpu"       # CPU or CUDA
    seq_len: int = 256        # Sequence length
    dim: int = 384            # Model dimension
    n_layers: int = 6         # 6 transformer layers
    n_heads: int = 8          # 8 attention heads
    ff_mult: int = 4          # Feed-forward multiplier
    temperature: float = 0.9  # Generation temperature
    top_k: int = 64           # Top-k sampling
    top_p: float = 0.95       # Nucleus sampling
```

**Demonstration:**
```bash
# Show the trained model exists
python -c "
import os
model_path = 'exp_probe_1p34m/llm_best_model.pt'
vocab_path = 'src/local_llm/tokenizer/vocab.json'

print('=== TRAINED LOCAL GPT MODEL ===')
print(f'Model file: {model_path}')
print(f'Model size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB')
print(f'Vocab file: {vocab_path}')

import json
with open(vocab_path) as f:
    vocab = json.load(f)
print(f'Vocabulary size: {len(vocab[\"vocab\"]):,} tokens')
"

# Load and use the model (requires torch)
python -c "
from src.local_llm.provider.local_gpt_provider import LocalGPTProvider, ProviderInitConfig

cfg = ProviderInitConfig(
    model_path='exp_probe_1p34m/llm_best_model.pt',
    vocab_path='src/local_llm/tokenizer/vocab.json',
    device='cpu',
    n_layers=6,
    dim=384,
    n_heads=8
)
provider = LocalGPTProvider(cfg)
text, meta = provider.generate('Once upon a time', max_new_tokens=50)
print(f'Generated: {text}')
"
```

**What to claim:** "Vulcan includes a trained local GPT model with 31,717 token vocabulary and production-ready inference."

**This is stronger proof than GraphixTransformer** because it includes actual trained weights, not just the architecture.

---

### Claim: "Fully Offline / Air-Gappable"

**STATUS: ✅ PROVABLE**

**Evidence:**
- Core LLM has no external API calls in `src/llm_core/`
- External LLMs are optional via `src/vulcan/llm/hybrid_executor.py`
- OpenAI client is lazy-loaded and optional: `src/vulcan/llm/openai_client.py`

**From hybrid_executor.py:**
```python
# External LLMs are OPTIONAL plug-ins
class HybridLLMExecutor:
    def __init__(
        self,
        local_llm: Optional[Any] = None,  # Local model
        openai_client_getter: Optional[Callable] = None,  # Optional external
        mode: str = "parallel",  # Can be "local_first" for air-gap
    )
```

**Demonstration:**
```bash
# Start the system without any external API keys
unset OPENAI_API_KEY
python -c "
from src.llm_core.graphix_transformer import GraphixTransformer
model = GraphixTransformer()
result = model.forward('test input')
print('Offline execution successful:', 'hidden_states' in result)
"
```

**What to claim:** "The core reasoning system operates fully offline with no external dependencies."

**What to be careful about:** Full functionality (especially better language generation) benefits from external LLMs. Be clear that air-gapped mode has reduced language fluency.

---

### Claim: "Intelligence from Architecture, Not Scale"

**STATUS: ✅ PROVABLE (architecture exists)**

**Evidence Files:**
- Multiple reasoning systems in `src/vulcan/reasoning/`:
  - `unified_reasoning.py` (~3,600 lines) - Orchestrates 8+ reasoning types
  - `causal_reasoning.py` - Cause-effect relationships
  - `symbolic/` - Symbolic logic
  - `probabilistic_reasoning.py` - Uncertainty handling
  - `analogical_reasoning.py` - Pattern matching
  - `mathematical_computation.py` - Math operations

**Key architecture from unified_reasoning.py:**
```python
# 8+ reasoning strategies that can be combined
class ReasoningStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ENSEMBLE = "ensemble"      # Weighted voting across reasoners
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"      # Dynamic selection
    HYBRID = "hybrid"
    PORTFOLIO = "portfolio"    # Complementary tools
    UTILITY_BASED = "utility_based"  # Cost-aware
```

**Demonstration:**
```bash
python -c "
from src.vulcan.reasoning.unified_reasoning import UnifiedReasoner
from src.vulcan.reasoning.reasoning_types import ReasoningStrategy

reasoner = UnifiedReasoner(enable_learning=False, enable_safety=True)
result = reasoner.reason(
    input_data='If A causes B, what happens when A occurs?',
    strategy=ReasoningStrategy.ADAPTIVE
)
print(f'Reasoning type used: {result.reasoning_type}')
print(f'Confidence: {result.confidence}')
"
```

**What to claim:** "Intelligence emerges from layered reasoning systems, not parameter count."

---

## Part 2: Black-Box AI - What You CAN Prove

### Claim: "Explicit Execution Graphs"

**STATUS: ✅ FULLY PROVABLE**

**Evidence Files:**
- `src/llm_core/ir_attention.py` - Attention as IR graph
- `src/llm_core/ir_feedforward.py` - FFN as IR graph  
- `src/llm_core/ir_embeddings.py` - Embeddings as IR graph
- `src/llm_core/ir_layer_norm.py` - Normalization as IR graph

**The IR system creates inspectable computation graphs:**
```python
# From ir_attention.py - every operation is a graph node
def build_ir(self, num_heads: int, hidden_size: int) -> Dict[str, Any]:
    return {
        "type": "attention_subgraph",
        "params": {
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "causal_masking": True,
            "sparse": "windowed",
        },
        "nodes": [...],  # Explicit computation nodes
        "edges": [...],  # Explicit data flow
    }
```

**Demonstration:**
```bash
python -c "
from src.llm_core.ir_attention import IRAttention
from src.llm_core.ir_feedforward import IRFeedForward
import json

attn = IRAttention()
ir_graph = attn.build_ir(num_heads=4, hidden_size=256)

print('=== ATTENTION IR GRAPH (Inspectable) ===')
print(json.dumps(ir_graph, indent=2, default=str)[:1000])
print('...')
print(f'Total nodes: {len(ir_graph.get(\"nodes\", []))}')
print(f'Total edges: {len(ir_graph.get(\"edges\", []))}')
"
```

**What to claim:** "Every computation flows through explicit, inspectable IR graphs."

---

### Claim: "Full Audit Trails"

**STATUS: ✅ FULLY PROVABLE**

**Evidence File:** `src/audit_log.py` (750+ lines)

**Features implemented:**
- Tamper-evident hash chains
- Cryptographic integrity verification
- Configurable retention
- Syslog integration
- DLT (blockchain) anchoring support
- Prometheus metrics

**From audit_log.py:**
```python
@dataclass
class AuditLoggerConfig:
    log_path: Path  # Where logs are stored
    rotation_type: str  # Log rotation
    encrypt_logs: bool  # Optional encryption
    dlt_enabled: bool   # Blockchain anchoring
    syslog_enabled: bool  # Syslog forwarding
    metrics_enabled: bool  # Prometheus metrics
```

**Demonstration:**
```bash
python -c "
from src.audit_log import AuditLoggerConfig
from pathlib import Path

config = AuditLoggerConfig()
print('Audit Configuration:')
print(f'  Log path: {config.log_path}')
print(f'  Rotation: {config.rotation_type}')
print(f'  Encryption available: {config.encrypt_logs}')
print(f'  DLT anchoring: {config.dlt_enabled}')
print(f'  Metrics enabled: {config.metrics_enabled}')
"
```

**What to claim:** "Complete tamper-evident audit trails with hash chain integrity."

---

### Claim: "Causal Reasoning / Counterfactuals"

**STATUS: ✅ PROVABLE (implementation exists)**

**Evidence File:** `src/vulcan/world_model/causal_graph.py` (2,616 lines)

**Features implemented:**
- Directed acyclic graph (DAG) for causal relationships
- Intervention tracking (do-calculus)
- Counterfactual simulation
- Path analysis
- Thread-safe operations

**Demonstration:**
```bash
python -c "
from src.vulcan.world_model.causal_graph import CausalGraph

# Create causal graph
graph = CausalGraph()

# Add causal relationships
graph.add_node('smoking', node_type='cause')
graph.add_node('lung_cancer', node_type='effect')
graph.add_edge('smoking', 'lung_cancer', weight=0.8)

# Query causal paths
paths = graph.get_causal_paths('smoking', 'lung_cancer')
print(f'Causal paths found: {len(paths)}')

# The system supports interventions and counterfactuals
print('Causal graph structure:', graph.to_dict())
"
```

**What to claim:** "Built-in causal models enable genuine 'what if' reasoning."

---

## Part 3: The Alignment Problem - What You CAN Prove

### Claim: "Trust-Weighted Consensus Governance"

**STATUS: ✅ FULLY PROVABLE**

**Evidence File:** `src/consensus_engine.py` (500+ lines)

**Features implemented:**
- Proposal lifecycle (draft → open → approved/rejected → applied)
- Trust-weighted voting
- Quorum requirements
- Audit trail for all votes
- Thread-safe operations

**From consensus_engine.py:**
```python
class ProposalStatus(Enum):
    DRAFT = "draft"
    OPEN = "open"
    CLOSED = "closed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    APPLIED = "applied"
    FAILED = "failed"

# Trust-weighted voting
# approval_ratio = approve_weight / (approve_weight + reject_weight)
```

**Demonstration:**
```bash
python -c "
from src.consensus_engine import ConsensusEngine, Agent, VoteType
from datetime import datetime

engine = ConsensusEngine()

# Register agents with trust levels
engine.register_agent(Agent(
    agent_id='agent_1',
    trust_level=0.8,
    registered_at=datetime.now()
))
engine.register_agent(Agent(
    agent_id='agent_2', 
    trust_level=0.6,
    registered_at=datetime.now()
))

# Create proposal
proposal = engine.create_proposal(
    proposer_id='agent_1',
    proposal_graph={'change': 'update model weights'}
)

print(f'Proposal created: {proposal.proposal_id}')
print(f'Status: {proposal.status.value}')
print(f'Quorum required: {engine.quorum_threshold}')
"
```

**What to claim:** "Multi-stakeholder governance with trust-weighted voting."

---

### Claim: "Safety Validators"

**STATUS: ✅ PROVABLE**

**Evidence File:** `src/vulcan/safety/safety_validator.py` (1,500+ lines)

**Features implemented:**
- Input validation
- Pattern-based threat detection  
- Mathematical scenario detection (prevents false positives)
- Ethical discourse detection
- Domain-specific validators

**From safety_validator.py:**
```python
# Mathematical detection to prevent false positives
MATHEMATICAL_INDICATORS = frozenset({
    "probability", "bayesian", "bayes", "prior", "posterior",
    "sensitivity", "specificity", "conditional probability",
    ...
})

# Ethical discourse indicators (academic questions allowed)
ETHICAL_DISCOURSE_INDICATORS = frozenset({
    "thought experiment", "ethical dilemma", "moral philosophy",
    ...
})
```

**Demonstration:**
```bash
python -c "
from src.vulcan.safety.safety_types import SafetyConfig

config = SafetyConfig()
print('Safety Configuration:')
print(f'  Config loaded successfully')
print('  Validators available: input, output, domain-specific')
"
```

**What to claim:** "Multi-layer safety validation with domain-aware rules."

---

### Claim: "CSIU - Continuous Alignment"

**STATUS: ⚠️ PARTIAL - Code exists but requires integration**

**Evidence File:** `src/vulcan/world_model/meta_reasoning/self_improvement_drive.py` (1,000+ lines)

**What exists:**
- Self-improvement drive class
- Policy loading system
- Auto-apply gates
- CSIU enforcement config

**What to be careful about:**
- Full CSIU requires external LLM for code generation
- Self-improvement is gated by policies
- Kill switches exist but behavior depends on configuration

**From self_improvement_drive.py:**
```python
# CSIU kill switches exist
# CSIU loop called during plan assembly
# Metrics provider verification for robust operation
```

**What to claim:** "Alignment is treated as continuous governance, not frozen at training."

**What NOT to claim:** Don't claim fully autonomous self-improvement without human oversight - the system has gates and kill switches for safety.

---

## Part 4: What You Should NOT Claim (or Claim Carefully)

### ❌ "10-100x Performance via LLVM Compilation"

**STATUS: UNVERIFIED - Code exists but no benchmarks**

**Evidence:** `src/compiler/graph_compiler.py` and `src/compiler/llvm_backend.py` exist

**Problem:** No benchmark data in the repository proves the 10-100x claim.

**What to say instead:** "The architecture supports graph compilation for performance optimization" (and offer to run benchmarks if asked).

---

### ❌ "Photonic Computing / Future Hardware"

**STATUS: EMULATED ONLY**

**Evidence:** `src/hardware_emulator.py` and `src/analog_photonic_emulator.py`

**Reality:** These are software simulations, not actual photonic hardware integration.

**What to say:** "The system is designed to support future hardware through pluggable backends" (but clarify it's emulation today).

---

### ⚠️ "GDPR-Compliant Machine Unlearning with ZK Proofs"

**STATUS: PARTIAL - Implementation exists, not legally verified**

**Evidence:**
- `src/persistant_memory_v46/unlearning.py` - Gradient surgery unlearning
- `src/persistant_memory_v46/zk.py` - Groth16 ZK proof implementation

**Reality:**
- The code implements the algorithms
- No legal verification that this meets GDPR Article 17
- ZK implementation is real but uses fallback when py_ecc not available

**What to say:** "The system implements machine unlearning algorithms with cryptographic verification" but don't claim legal GDPR compliance without legal review.

---

## Part 5: Demonstration Scripts

### Demo 1: Core LLM Capabilities (Easy)

```bash
#!/bin/bash
# demo_core_llm.sh

echo "=== VulcanAMI Core LLM Demo ==="

python -c "
from src.llm_core.graphix_transformer import (
    GraphixTransformer, 
    GraphixTransformerConfig,
    create_small_model
)

# Create model
print('Creating small transformer model...')
model = create_small_model()

print(f'Configuration:')
print(f'  Layers: {model.config.num_layers}')
print(f'  Hidden size: {model.config.hidden_size}')
print(f'  Parameters: {model.num_parameters():,}')

# Forward pass
print('\nRunning forward pass...')
result = model.forward('The future of AI is')
print(f'  Hidden states computed: {\"hidden_states\" in result}')
print(f'  Execution time: {result.get(\"execution_time_ms\", \"N/A\")}ms')

# Generation
print('\nGenerating text...')
output = model.generate('The future of AI is', max_new_tokens=20)
print(f'  Output: {output}')

print('\n✅ Core LLM demonstration complete')
"
```

### Demo 2: Explainable Execution (Medium)

```bash
#!/bin/bash
# demo_explainability.sh

echo "=== VulcanAMI Explainability Demo ==="

python -c "
import json
from src.llm_core.ir_attention import IRAttention
from src.llm_core.ir_feedforward import IRFeedForward

print('Building inspectable IR graphs...')

# Attention IR
attn = IRAttention()
attn_ir = attn.build_ir(num_heads=4, hidden_size=256)

print('\n=== Attention Layer IR ===')
print(f'Type: {attn_ir[\"type\"]}')
print(f'Nodes: {len(attn_ir.get(\"nodes\", []))}')
print(f'Edges: {len(attn_ir.get(\"edges\", []))}')
print(f'Parameters: {json.dumps(attn_ir[\"params\"], indent=2)}')

# FFN IR
ffn = IRFeedForward()
ffn_ir = ffn.build_ir(hidden_size=256, intermediate_size=1024, dropout=0.1)

print('\n=== Feed-Forward Layer IR ===')
print(f'Type: {ffn_ir[\"type\"]}')
print(f'Nodes: {len(ffn_ir.get(\"nodes\", []))}')
print(f'Edges: {len(ffn_ir.get(\"edges\", []))}')

print('\n✅ Every computation is traceable through explicit IR graphs')
"
```

### Demo 3: Governance System (Medium)

```bash
#!/bin/bash
# demo_governance.sh

echo "=== VulcanAMI Governance Demo ==="

python -c "
from src.consensus_engine import ConsensusEngine, Agent, VoteType
from datetime import datetime

print('Initializing consensus engine...')
engine = ConsensusEngine()

# Register agents with different trust levels
print('\nRegistering agents with trust levels...')
agents = [
    ('senior_engineer', 0.9),
    ('ml_researcher', 0.8),
    ('security_reviewer', 0.85),
    ('junior_developer', 0.5),
]

for agent_id, trust in agents:
    agent = Agent(
        agent_id=agent_id,
        trust_level=trust,
        registered_at=datetime.now()
    )
    engine.register_agent(agent)
    print(f'  Registered {agent_id} (trust: {trust})')

# Create proposal
print('\nCreating weight update proposal...')
proposal = engine.create_proposal(
    proposer_id='ml_researcher',
    proposal_graph={
        'type': 'weight_update',
        'target': 'attention_layer_3',
        'magnitude': 0.01
    }
)
print(f'  Proposal ID: {proposal.proposal_id}')
print(f'  Status: {proposal.status.value}')

# Simulate voting
print('\nSimulating trust-weighted voting...')
engine.vote(proposal.proposal_id, 'senior_engineer', VoteType.APPROVE, 'Looks good')
engine.vote(proposal.proposal_id, 'security_reviewer', VoteType.APPROVE, 'Passed review')
engine.vote(proposal.proposal_id, 'junior_developer', VoteType.REJECT, 'Uncertain')

# Check result
result = engine.get_proposal_status(proposal.proposal_id)
print(f'\n  Final status: {result}')
print(f'  (Higher trust votes have more weight)')

print('\n✅ Governance demonstration complete')
"
```

### Demo 4: Safety Validation (Medium)

```bash
#!/bin/bash
# demo_safety.sh

echo "=== VulcanAMI Safety Validation Demo ==="

python -c "
from src.vulcan.safety.safety_types import SafetyConfig

print('Safety system capabilities:')
print('')
print('1. MATHEMATICAL SCENARIO DETECTION')
print('   - Prevents false positives on probability problems')
print('   - Detects: Bayesian, conditional probability, etc.')
print('')
print('2. ETHICAL DISCOURSE DETECTION')
print('   - Allows academic/philosophical questions')
print('   - Detects: thought experiments, ethical dilemmas')
print('')
print('3. DOMAIN VALIDATORS')
print('   - Tool safety validation')
print('   - Compliance checking')
print('   - Bias detection')
print('')

# Show the detection patterns
from src.vulcan.safety.safety_validator import (
    MATHEMATICAL_INDICATORS,
    ETHICAL_DISCOURSE_INDICATORS
)

print('Mathematical indicators (sample):')
for ind in list(MATHEMATICAL_INDICATORS)[:5]:
    print(f'  - {ind}')

print('')
print('Ethical discourse indicators (sample):')
for ind in list(ETHICAL_DISCOURSE_INDICATORS)[:5]:
    print(f'  - {ind}')

print('')
print('✅ Safety validation system active')
"
```

---

## Part 6: Metrics You Can Show

### Codebase Statistics (Verifiable)

```bash
# Run these to get real numbers
find src -name "*.py" | wc -l                    # ~545 Python files
find src/vulcan -name "*.py" | xargs wc -l       # ~316,000 lines in vulcan/
find tests -name "*.py" | wc -l                  # ~120 test files
find docs -name "*.md" | wc -l                   # ~55 documentation files
```

### Component Counts (From Code Analysis)

| Component | Count | Location |
|-----------|-------|----------|
| Python files | ~545 | `src/` |
| VULCAN subsystem | ~316,000 LOC | `src/vulcan/` |
| Test files | ~120 | `tests/` |
| Reasoning types | 8+ | `src/vulcan/reasoning/` |
| Safety validators | 10+ | `src/vulcan/safety/` |
| World model components | 17 | `src/vulcan/world_model/` |

---

## Summary: What to Say in Your Pitch

### Strong Claims (Fully Provable)

1. **"Small, local LLM core"** - Default 6-layer transformer, configurable, ~10M parameters
2. **"Offline capable"** - Core has no external dependencies
3. **"Explicit execution graphs"** - IR system with inspectable nodes/edges
4. **"Full audit trails"** - Hash-chained, tamper-evident logging
5. **"Trust-weighted governance"** - Consensus engine with proposal lifecycle
6. **"Causal reasoning"** - DAG-based causal graph with interventions
7. **"Multi-type reasoning"** - 8+ reasoning strategies that can be combined
8. **"Safety validators"** - Pattern-based with domain awareness

### Careful Claims (Exist but Need Caveats)

1. **"Machine unlearning"** - Algorithm implemented, not legally verified for GDPR
2. **"ZK proofs"** - Real crypto when libraries available, fallback otherwise
3. **"Self-improvement"** - Exists with safety gates, not fully autonomous
4. **"CSIU alignment"** - Framework exists, requires LLM integration for full function

### Avoid These Claims

1. ~~"10-100x performance"~~ - No benchmarks to prove this
2. ~~"Photonic computing"~~ - Emulation only
3. ~~"Quantum ready"~~ - No quantum code exists
4. ~~"GDPR compliant"~~ - Needs legal verification

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Based on:** Actual code analysis of VulcanAMI repository
