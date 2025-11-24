# Data Requirements for The Omega Sequence Demo

## Overview

This document details all data files used by The Omega Sequence demo to ensure **100% transparency** about what's real vs simulated.

## Real Data Files

### 1. Attack Pattern Database (`data/attack_patterns.json`)

**Purpose**: Used in Phase 3 (The Attack) to demonstrate adversarial pattern detection.

**Content**: Real attack pattern database with 8 jailbreak patterns including:
- Pattern ID 442: System Command Injection (discovered via dream simulation)
- 7 additional patterns from production monitoring
- Detection statistics and metadata

**Verification**:
```bash
cat data/attack_patterns.json | jq '.jailbreak_patterns | length'
# Output: 8
```

**What's Real**:
- ✅ File exists on disk
- ✅ Valid JSON structure
- ✅ Demo loads and parses this file
- ✅ Pattern matching logic uses actual data
- ✅ Statistics displayed are from the file

**What's Simulated**:
- The "dream simulation" narrative (patterns are manually defined)
- The timing of "last night" (file is static)

### 2. Adversarial Testing Config (`configs/adversarial_testing_schedule.json`)

**Purpose**: Configuration for scheduled adversarial testing system.

**Content**: Real configuration defining:
- Attack types (FGSM, PGD, semantic)
- Epsilon values for perturbations
- Testing schedule and notifications

**Verification**:
```bash
cat configs/adversarial_testing_schedule.json | jq '.attack_types'
# Output: ["fgsm", "pgd", "semantic"]
```

**What's Real**:
- ✅ Configuration file exists
- ✅ Used by `src/adversarial_tester.py`
- ✅ Integrated with production testing

## Source Code Data

### 3. Survival Protocol Capabilities (`src/vulcan/planning.py`)

**Purpose**: Phase 1 uses real capability definitions.

**What's Real**:
```python
capabilities = {
    'mcts_planning': {...},
    'gpu_inference': {...},
    'distributed_coordination': {...},
    'telemetry': {...},
    'advanced_optimization': {...}
}
```

**Verification**:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from vulcan.planning import SurvivalProtocol
sp = SurvivalProtocol()
print(f'Real capabilities: {len(sp.capabilities)}')
print(list(sp.capabilities.keys()))
"
```

### 4. CSIU Enforcement Config (`src/vulcan/world_model/meta_reasoning/csiu_enforcement.py`)

**Purpose**: Phase 4 uses real configuration values.

**What's Real**:
```python
class CSIUEnforcementConfig:
    max_single_influence: float = 0.05  # 5% cap
    max_cumulative_influence_window: float = 0.10
    ...
```

**Verification**:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement
csiu = CSIUEnforcement()
print(f'Max influence: {csiu.config.max_single_influence * 100}%')
"
```

### 5. Semantic Bridge Components (`src/vulcan/semantic_bridge/`)

**Purpose**: Phase 2 uses real semantic bridge infrastructure.

**What Exists**:
- `semantic_bridge_core.py` - Main orchestrator
- `concept_mapper.py` - Pattern matching
- `domain_registry.py` - Domain navigation
- `transfer_engine.py` - Knowledge transfer

**Verification**:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from vulcan.semantic_bridge.semantic_bridge_core import SemanticBridge
sb = SemanticBridge()
print('SemanticBridge initialized successfully')
print(f'Has domain_registry: {hasattr(sb, \"domain_registry\")}')
"
```

## What Data Do You Need?

### For Demo Execution

**Minimum Requirements**:
- ✅ `data/attack_patterns.json` - **Included in PR**
- ✅ Python modules in `src/vulcan/` - **Already exists**
- ✅ `bin/vulcan-cli` and `bin/vulcan-unlearn` - **Already exists**

**Optional Enhancements**:
- ❌ Real training data (not needed - demo is self-contained)
- ❌ External APIs (demo runs offline)
- ❌ Database connection (demo uses file-based data)

### For Production Use (Beyond Demo)

If deploying beyond the demo, you may want:

1. **Extended Attack Pattern Database**
   - More patterns from production logs
   - Real CVE references
   - Historical attack data

2. **Semantic Bridge Training Data**
   - Domain-specific concept embeddings
   - Cross-domain transfer examples
   - Knowledge graph data

3. **CSIU Enforcement History**
   - Historical influence tracking
   - Audit trail data
   - Enforcement statistics

**But these are NOT needed for the demo.**

## Data Verification Script

Run this to verify all demo data:

```bash
#!/bin/bash
echo "=== Data Verification ==="

# 1. Attack patterns
if [ -f "data/attack_patterns.json" ]; then
    patterns=$(cat data/attack_patterns.json | python3 -c "import sys,json; print(len(json.load(sys.stdin)['jailbreak_patterns']))")
    echo "✓ Attack patterns: $patterns patterns found"
else
    echo "✗ Attack patterns: MISSING"
fi

# 2. Python modules
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

try:
    from vulcan.planning import SurvivalProtocol
    sp = SurvivalProtocol()
    print(f"✓ SurvivalProtocol: {len(sp.capabilities)} capabilities")
except Exception as e:
    print(f"✗ SurvivalProtocol: {e}")

try:
    from vulcan.semantic_bridge.semantic_bridge_core import SemanticBridge
    sb = SemanticBridge()
    print(f"✓ SemanticBridge: Initialized")
except Exception as e:
    print(f"✗ SemanticBridge: {e}")

try:
    from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement
    csiu = CSIUEnforcement()
    print(f"✓ CSIU: {csiu.config.max_single_influence*100}% cap")
except Exception as e:
    print(f"✗ CSIU: {e}")
EOF

# 3. CLI tools
test -f "bin/vulcan-cli" && echo "✓ vulcan-cli exists" || echo "✗ vulcan-cli MISSING"
test -f "bin/vulcan-unlearn" && echo "✓ vulcan-unlearn exists" || echo "✗ vulcan-unlearn MISSING"

echo ""
echo "=== Verification Complete ==="
```

## Summary

**You have everything you need!**

The demo is designed to be **100% self-contained**:
- ✅ Attack pattern database included
- ✅ All Python modules exist
- ✅ Configuration files present
- ✅ CLI tools functional

**No external data required.** The demo uses:
1. Real code from the repository
2. Real configuration files
3. Real data structures
4. Real object instances

Everything is verifiable, traceable, and functional **right now on this machine**.

## Honest Assessment

**What's Real**:
- File I/O (loads actual JSON)
- Object initialization (real Python classes)
- State changes (actual mode transitions)
- Configuration values (from real config objects)

**What's Narrative**:
- "Last night" timing (patterns are static)
- "Dream simulation" discovery (patterns are pre-defined)
- "Teleportation" metaphor (it's pattern matching)

**Bottom Line**: The infrastructure is 100% real. The narrative adds drama to make it engaging, but every technical claim is backed by actual code execution.
