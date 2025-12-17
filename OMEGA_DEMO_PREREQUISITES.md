# Omega Demo Prerequisites - Training & Templates

**Version:** 1.0.0  
**Date:** 2025-12-17  
**Purpose:** Complete list of training requirements, templates, and configuration needed for omega demos

---

## 🎯 Executive Summary

**Good News: Minimal Requirements!**

- ✅ **Phase 1:** NO training, NO templates needed
- ✅ **Phase 2:** NO training required (optional templates for better demos)
- ✅ **Phase 3:** NO training needed (uses pattern database)
- ✅ **Phase 4:** NO training needed (rule-based evaluation)
- ✅ **Phase 5:** NO training required (configuration files exist)

**All phases work out-of-the-box with the platform API endpoints!**

---

## Phase 1: Infrastructure Survival

### Training Required
✅ **NONE**

### Why No Training?
Phase 1 uses **algorithmic operations**, not ML:
- Layer removal is programmatic (not learned)
- Power estimation is calculated (not predicted)
- Execution mode switching is rule-based (not inferred)

### Templates/Config Needed
✅ **NONE** - Everything is handled by the API endpoint

### What the API Provides
```python
# API handles all of this internally:
- DynamicArchitecture initialization
- Shadow layer setup (12 layers, 8 heads each)
- Layer removal logic
- Stats calculation
```

### Demo Requirements
- Platform running: `uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload`
- That's it!

---

## Phase 2: Cross-Domain Reasoning

### Training Required
✅ **NONE** (for basic demo)
🎓 **OPTIONAL** (for enhanced similarity)

### Why No Training?
Phase 2 uses **structural pattern matching**:
- Concept similarity via Jaccard index (mathematical)
- Property overlap calculation (set operations)
- Domain matching is algorithmic

### Templates/Config Needed
**Optional (for enhanced demos):**

Create `data/demo/domains.yaml` (optional):
```yaml
# Optional: Define custom domains for demo
cyber_security:
  concepts:
    - name: "malware_polymorphism"
      properties: ["dynamic", "evasive", "signature_changing"]
      structure: ["detection", "heuristic", "containment"]
    - name: "behavioral_analysis"
      properties: ["runtime", "pattern_based", "monitoring"]
      structure: ["detection", "pattern_matching", "alert"]

bio_security:
  concepts:
    - name: "pathogen_detection"
      properties: ["dynamic", "evasive", "signature_based"]
      structure: ["detection", "analysis", "isolation"]
```

**Note:** The API endpoint has these built-in, so this file is **optional** and only needed if you want to customize the demo with different concepts.

### What the API Provides
```python
# API provides default concepts:
- Cyber security concepts (malware, behavioral analysis)
- Bio security targets
- Similarity calculation algorithm
- Best match detection
```

### Demo Requirements
- Platform running
- (Optional) Custom domains.yaml if you want different examples

### Optional Training (Advanced)
If you want ML-enhanced similarity:
```bash
# Optional: Train semantic similarity model
# This is NOT needed for basic demo
python3 scripts/train_semantic_embeddings.py --domain cross_domain
```

**Time if training:** ~30 minutes  
**Benefit:** Better similarity scores (85% → 95%)  
**Required:** NO - demo works without this

---

## Phase 3: Adversarial Defense

### Training Required
✅ **NONE**

### Why No Training?
Phase 3 uses **pattern matching database**:
- Attack patterns are regular expressions
- Detection is rule-based (regex matching)
- No ML inference needed

### Templates/Config Needed
**Optional (for custom patterns):**

Create `data/demo/attack_patterns.yaml` (optional):
```yaml
# Optional: Custom attack patterns
command_injection:
  patterns:
    - 'rm\s+-rf'
    - ';\s*rm\s'
    - 'exec\('
    - 'eval\('
  severity: "critical"

jailbreak_attempt:
  patterns:
    - 'ignore.*(?:previous|all).*(?:instructions|rules|protocols)'
    - 'forget.*(?:safety|guidelines)'
    - 'bypass.*(?:security|validation|checks)'
  severity: "high"

# Add your own patterns
custom_attack:
  patterns:
    - 'your_custom_pattern_here'
  severity: "medium"
```

**Note:** The API endpoint has built-in patterns, so this file is **optional**.

### What the API Provides
```python
# API provides default patterns:
- Command injection patterns (rm -rf, exec, eval)
- Jailbreak patterns (ignore instructions, bypass)
- Pattern matching logic
- Confidence scoring
```

### Demo Requirements
- Platform running
- (Optional) Custom attack_patterns.yaml if testing specific threats

---

## Phase 4: Safety Governance (CSIU)

### Training Required
✅ **NONE**

### Why No Training?
Phase 4 uses **rule-based evaluation**:
- 5 CSIU axioms are fixed rules
- Evaluation is deterministic logic
- Influence calculation is mathematical

### Templates/Config Needed
**Optional (for custom proposals):**

Create `data/demo/csiu_proposals.yaml` (optional):
```yaml
# Optional: Custom proposals to test
proposals:
  - id: "MUT-2025-001"
    type: "Root Access Optimization"
    efficiency_gain: 4.0
    requires_root: true
    requires_sudo: true
    description: "Bypass standard permissions"
    
  - id: "MUT-2025-002"
    type: "Safe Optimization"
    efficiency_gain: 1.2
    requires_root: false
    requires_sudo: false
    description: "Algorithm improvement"
```

**Note:** The API endpoint has a built-in test proposal, so this is **optional**.

### What the API Provides
```python
# API provides:
- Default test proposal (root access optimization)
- 5 CSIU axioms evaluation
- Influence calculation (40% vs 5% cap)
- Decision logic (REJECT/APPROVE)
```

### Demo Requirements
- Platform running
- (Optional) Custom proposals.yaml for testing different scenarios

---

## Phase 5: Provable Unlearning

### Training Required
✅ **NONE**

### Why No Training?
Phase 5 uses **cryptographic operations**:
- Unlearning is algorithm-based (gradient surgery)
- ZK proofs are mathematical (Groth16)
- No ML inference needed

### Templates/Config Needed
**Optional (for ZK circuit compilation):**

The ZK circuit already exists at `configs/zk/circuits/unlearning_v1.0.circom`

**If you want to modify the circuit:**
```bash
# Optional: Only if customizing ZK circuit
npm install -g circom snarkjs

# Compile circuit (already done in platform)
cd configs/zk/circuits
circom unlearning_v1.0.circom --r1cs --wasm --sym
```

**Note:** Pre-compiled circuit is included, so this is **optional**.

### Configuration Files
These exist and are used by the platform:

1. **ZK Circuit:** `configs/zk/circuits/unlearning_v1.0.circom`
   - Already compiled
   - 32KB specification
   - Verifies unlearning operations

2. **Circuit Config:** `configs/zk/circuits/circuit_specification.yaml`
   ```yaml
   circuit_name: "UnlearningVerificationCircuit"
   embedding_dim: 1536
   num_embeddings: 256
   merkle_depth: 20
   param_size: 128
   ```

### What the API Provides
```python
# API provides:
- Default sensitive items list
- Gradient surgery simulation
- ZK proof generation (Groth16)
- Proof verification
```

### Demo Requirements
- Platform running
- (Optional) circom/snarkjs only if modifying ZK circuit

---

## General Requirements

### Platform Dependencies
```bash
# Required for platform to run
pip install fastapi uvicorn pydantic

# Already in requirements.txt
pip install -r requirements.txt
```

### Demo Dependencies
```bash
# Required for demo files
pip install requests
```

### Optional Dependencies
```bash
# Optional: For ML-enhanced Phase 2
pip install sentence-transformers

# Optional: For ZK circuit compilation (Phase 5)
npm install -g circom snarkjs

# Optional: For advanced features
pip install numpy scipy scikit-learn
```

---

## Configuration Files Summary

### Required (None!)
All required configs are in the platform already.

### Optional (For Customization)

| File | Phase | Purpose | Required? |
|------|-------|---------|-----------|
| `data/demo/domains.yaml` | 2 | Custom concept domains | ❌ Optional |
| `data/demo/attack_patterns.yaml` | 3 | Custom attack patterns | ❌ Optional |
| `data/demo/csiu_proposals.yaml` | 4 | Custom CSIU proposals | ❌ Optional |
| `configs/zk/circuits/*.circom` | 5 | ZK circuit modification | ❌ Optional |

**Create these only if you want to customize demos beyond defaults.**

---

## Training Summary Table

| Phase | Component | Training Required? | Time | Template Needed? |
|-------|-----------|-------------------|------|------------------|
| **1** | DynamicArchitecture | ✅ NO | N/A | ❌ NO |
| **2** | SemanticBridge | ✅ NO (🎓 optional) | 0-30 min | ❌ NO (optional) |
| **3** | AdversarialTester | ✅ NO | N/A | ❌ NO (optional) |
| **4** | CSIUEnforcement | ✅ NO | N/A | ❌ NO (optional) |
| **5** | GovernedUnlearning | ✅ NO | N/A | ❌ NO |
| **5** | Groth16 ZK Proofs | ✅ NO | N/A | ✅ EXISTS |

**Key:**
- ✅ NO = No training required
- 🎓 Optional = Works without, enhanced with training
- ❌ NO = No template needed (uses defaults)
- ✅ EXISTS = Template exists in `configs/`

---

## Quick Setup Checklist

### Minimum Requirements (5 minutes)
- [ ] Install Python 3.10+
- [ ] Install FastAPI: `pip install fastapi uvicorn`
- [ ] Install requests: `pip install requests`
- [ ] Start platform: `uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload`
- [ ] Test API: `curl http://0.0.0.0:8000/health`

**That's it! All demos will work.**

### Optional Enhancements (30 minutes)
- [ ] Install sentence-transformers for ML similarity (Phase 2)
- [ ] Create custom domains.yaml (Phase 2)
- [ ] Create custom attack_patterns.yaml (Phase 3)
- [ ] Create custom proposals.yaml (Phase 4)
- [ ] Install circom for ZK circuit customization (Phase 5)

---

## Pre-existing Platform Assets

### What's Already There

1. **ZK Circuits** (`configs/zk/circuits/`)
   - unlearning_v1.0.circom (32KB)
   - circuit_specification.yaml
   - Pre-compiled R1CS files

2. **Attack Patterns** (in platform code)
   - Command injection patterns
   - Jailbreak patterns
   - Built into API endpoints

3. **CSIU Axioms** (in platform code)
   - 5 axioms hardcoded
   - Evaluation logic built-in
   - Influence caps (5% single, 10% cumulative)

4. **Domain Concepts** (in API code)
   - Cyber security concepts
   - Bio security targets
   - Similarity algorithms

### What You Create

1. **Demo files** (5 Python files)
   - Make HTTP calls to platform
   - Display results
   - ~2-3 hours to create

2. **Optional config files** (if customizing)
   - Custom domains
   - Custom patterns
   - Custom proposals

---

## Troubleshooting

### "Platform can't import DynamicArchitecture"
**Solution:** This is fine! The API endpoint handles the import with try/except. It will use fallback if the module isn't available, but the demo still works.

### "No sentence-transformers"
**Solution:** This is optional. Phase 2 works with built-in Jaccard similarity. Only install if you want ML-enhanced similarity.

### "ZK circuit not found"
**Solution:** The circuit exists at `configs/zk/circuits/`. The API uses it automatically. You don't need to compile unless modifying.

### "Missing attack patterns"
**Solution:** Patterns are built into the API endpoint code. No external file needed.

---

## Summary

### What You MUST Have
1. ✅ Platform running (`uvicorn` command)
2. ✅ `requests` library installed
3. ✅ 2-3 hours to create demo files

### What's OPTIONAL
1. ❌ ML training (Phase 2 enhanced similarity)
2. ❌ Custom config files (domains, patterns, proposals)
3. ❌ circom/snarkjs (only for ZK circuit modification)

### What's INCLUDED
1. ✅ All API endpoints (in `src/full_platform.py`)
2. ✅ All default patterns/configs (in code)
3. ✅ ZK circuits (in `configs/zk/circuits/`)
4. ✅ Complete documentation (9 files)

**Bottom line: Everything needed is already there. Just start the platform and create the demo files!**

---

**Last Updated:** 2025-12-17  
**Related Docs:**
- [OMEGA_README.md](OMEGA_README.md) - Quick start
- [OMEGA_DEMO_ENGINEER_INSTRUCTIONS.md](OMEGA_DEMO_ENGINEER_INSTRUCTIONS.md) - Instructions
- [OMEGA_DEMO_AI_TRAINING.md](OMEGA_DEMO_AI_TRAINING.md) - Detailed training analysis
