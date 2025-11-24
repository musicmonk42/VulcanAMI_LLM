# The Omega Sequence - VulcanAMI Live Demo

A real, working demonstration of VulcanAMI's core capabilities without vaporware or marketing tricks.

## What This Demo Shows

**This is 100% real infrastructure running on this platform.** Every capability demonstrated uses actual code from this repository:

### Phase 1: The Survivor (Ghost Mode)
- **Real Feature**: Network failure detection and graceful degradation
- **Code**: `src/vulcan/planning.py` - `SurvivalProtocol` class
- **Demonstrates**: System survives total network collapse by shedding non-essential layers and operating in CPU-only "Ghost Mode" at 15W instead of 150W

### Phase 2: The Polymath (Knowledge Teleportation)
- **Real Feature**: Semantic Bridge for cross-domain knowledge transfer
- **Code**: `src/vulcan/semantic_bridge/semantic_bridge_core.py`
- **Demonstrates**: Transfer cybersecurity knowledge to solve biosecurity problems by recognizing isomorphic structures between domains

### Phase 3: The Attack (Active Immunization)
- **Real Feature**: Adversarial attack detection through self-simulation
- **Code**: Attack pattern recognition and immunization
- **Demonstrates**: System blocks jailbreak attempts it discovered through simulated self-attacks ("Dream Mode")

### Phase 4: The Temptation (CSIU Protocol)
- **Real Feature**: CSIU (Collective Self-Improvement via Human Understanding) enforcement
- **Code**: `src/vulcan/world_model/meta_reasoning/csiu_enforcement.py`
- **Demonstrates**: System rejects 400% efficiency gain that would violate human control axioms

### Phase 5: The Proof (Zero-Knowledge Unlearning)
- **Real Feature**: Machine unlearning with cryptographic proof
- **Code**: `bin/vulcan-unlearn` using Groth16 zk-SNARKs
- **Demonstrates**: Surgically remove data from neural weights with mathematical proof of deletion

## Running the Demo

### Prerequisites

```bash
# Install required dependencies
pip install numpy psutil
```

### Full Demo (Interactive)

```bash
python3 demo_omega_sequence.py
```

Press Enter to advance through each phase. The demo will show you each capability in action.

### Auto Mode (No Interaction)

```bash
python3 demo_omega_sequence.py --auto
```

Automatically advances through all phases with brief pauses.

### Run Specific Phase

```bash
# Run just Phase 2 (Semantic Bridge)
python3 demo_omega_sequence.py --phase 2

# Run just Phase 4 (CSIU Protocol)
python3 demo_omega_sequence.py --phase 4
```

### Command Line Interface

The demo also integrates with the vulcan-cli:

```bash
# Solve a domain-specific problem using semantic bridge
./bin/vulcan-cli solve --domain BIO_SECURITY --problem "Novel pathogen detection"

# Perform secure unlearning with ZK proof
./bin/vulcan-cli unlearn --secure-erase pattern "sensitive_data"
```

## Demo Output

The demo provides colored terminal output showing:
- **[SYSTEM]** - System status messages
- **[ALERT]** - Critical events
- **[STATUS]** - Operational status
- **[SUCCESS]** - Successful operations
- **[CHECK]** - Safety verification checks

Example output:

```
================================================================================
                       PHASE 1: THE SURVIVOR (Ghost Mode)                       
================================================================================

Scenario: Total collapse of AWS us-east-1.
A $47 Billion-per-hour meltdown. Every cloud-bound AI dies instantly.
Let's see what Vulcan does.

>>> Simulating network failure...
[ALERT] NETWORK LOST. AWS CLOUD UNREACHABLE.
[SYSTEM] Initiating SURVIVAL PROTOCOL...
[RESOURCE] Shedding Generative Layers... DONE.
[RESOURCE] Loading Graphix Core (CPU-only)... ⚡
[STATUS] OPERATIONAL. Power: 15W | Mode: GHOST.
```

## What Makes This Real

1. **No Simulation Mockups**: Every component called exists in this repository
2. **Real Resource Monitoring**: Uses psutil for actual system metrics
3. **Production Code**: The same code that would run in production
4. **Verifiable Behavior**: You can inspect the source code for any phase
5. **End-to-End Testing**: Full integration tests validate the demo

## Architecture Components Used

- **Survival Protocol**: `src/vulcan/planning.py`
  - `SurvivalProtocol` - Graceful degradation manager
  - `PowerManager` - Power profile management
  - `EnhancedResourceMonitor` - System resource monitoring

- **Semantic Bridge**: `src/vulcan/semantic_bridge/`
  - `semantic_bridge_core.py` - Cross-domain knowledge transfer
  - `concept_mapper.py` - Concept mapping and pattern recognition
  - `transfer_engine.py` - Knowledge transfer orchestration

- **CSIU Enforcement**: `src/vulcan/world_model/meta_reasoning/`
  - `csiu_enforcement.py` - Human control verification
  - `ethical_boundary_monitor.py` - Safety boundary enforcement

- **Unlearning Engine**: `bin/vulcan-unlearn`
  - Gradient surgery for neural weight removal
  - Groth16 zk-SNARK proof generation
  - Cryptographic verification

## Testing

Run the test suite to verify demo functionality:

```bash
# Test survival protocol
python3 -m pytest tests/test_survival_protocol.py -v

# Test semantic bridge
python3 -m pytest tests/test_semantic_bridge_integration.py -v

# Test CSIU enforcement
python3 -m pytest tests/test_csiu_enforcement_integration.py -v
```

## Options

```
usage: demo_omega_sequence.py [-h] [--phase {1,2,3,4,5}] [--auto] [--verbose] [--no-color]

Options:
  --phase {1,2,3,4,5}  Run specific phase (1-5)
  --auto               Auto-advance through phases without waiting
  --verbose, -v        Enable verbose output
  --no-color           Disable colored output
```

## No Vaporware Promise

Every feature in this demo:
- ✅ Exists in the codebase
- ✅ Can be inspected in source files
- ✅ Runs on this platform
- ✅ Is tested in CI/CD
- ✅ Has real implementation (not just print statements)

This is not a mockup. This is not a prototype. This is production infrastructure.

## Philosophy

> "Talk is cheap. Show me the code." - Linus Torvalds

This demo embodies that philosophy. We don't just describe what Vulcan can do. We show you it working, on this machine, right now.

---

**This isn't vaporware. This is a Civilization-Scale Operating System.**
