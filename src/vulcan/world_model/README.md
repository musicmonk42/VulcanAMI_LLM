# VULCAN World Model & Meta-Reasoning System
## Complete Technical Documentation

**Version:** 0.1.0 
**Last Updated:** January 2026 
**Status:** ⚠️ RESEARCH/DEVELOPMENT - NOT ---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [System Architecture](#system-architecture)
3. [World Model Components](#world-model-components)
4. [Meta-Reasoning Layer](#meta-reasoning-layer)
5. [Self-Improvement Drive](#self-improvement-drive)
6. [CSIU Mechanism](#csiu-mechanism)
7. [Security Architecture](#security-architecture)
8. [Configuration & Deployment](#configuration--deployment)
9. [API Reference](#api-reference)
10. [Safety Considerations](#safety-considerations)
11. [Development Guide](#development-guide)
12. [Troubleshooting](#troubleshooting)

---

## Executive Overview

### What Is VULCAN World Model?

The VULCAN World Model is an advanced AGI reasoning system that combines:

- **Causal inference** - Understanding cause-and-effect relationships in environments
- **Temporal dynamics** - Modeling how systems evolve over time
- **Uncertainty quantification** - Calibrated confidence in predictions
- **Meta-reasoning** - Self-reflection on objectives and decision-making
- **Autonomous self-improvement** - Ability to modify its own code

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Codebase** | ~800KB |
| **Core Modules** | 11 (world_model) + 17 (meta_reasoning) |
| **Component Classes** | 60+ |
| **Lines of Code** | ~15,000+ |
| **Primary Language** | Python 3.12+ |

### Architecture at a Glance

```
External Environment
 ↓
WorldModelRouter (Intelligent Routing)
 ↓
┌────────────────────────────────────┐
│ World Model Core Components │
├────────────────────────────────────┤
│ • CausalDAG │
│ • DynamicsModel │
│ • CorrelationTracker │
│ • InterventionManager │
│ • PredictionEngine │
│ • InvariantDetector │
│ • ConfidenceCalibrator │
└────────────────────────────────────┘
 ↓
┌────────────────────────────────────┐
│ Meta-Reasoning Layer (Optional) │
├────────────────────────────────────┤
│ • MotivationalIntrospection │
│ • ObjectiveHierarchy │
│ • GoalConflictDetector │
│ • ValidationTracker │
│ • SelfImprovementDrive │
│ • CSIU Enforcement │
└────────────────────────────────────┘
```

---

## System Architecture

### Design Philosophy: EXAMINE → SELECT → APPLY → REMEMBER

Every component follows this pattern:

1. **EXAMINE** - Gather relevant state and validate inputs
2. **SELECT** - Choose appropriate action/method based on context
3. **APPLY** - Execute the selected operation with safety checks
4. **REMEMBER** - Update state, log metrics, learn from outcome

### Core Principles

- **Thread-Safe by Default** - All components use `threading.RLock()`
- **Graceful Degradation** - Fallbacks for missing optional dependencies
- **Safety-First** - Multiple validation layers throughout
- **Separation of Concerns** - Modular architecture with clear interfaces
- **Lazy Loading** - Components initialized on-demand to reduce startup time

### Technology Stack

**Required Dependencies:**
```
numpy>=1.24.0
logging (stdlib)
threading (stdlib)
```

**Optional Dependencies (with fallbacks):**
```
scipy>=1.10.0 # Statistical functions
sklearn>=1.3.0 # ML models
pandas>=2.0.0 # Data manipulation
networkx>=3.0 # Advanced graph operations
statsmodels>=0.14.0 # Time series analysis
```

**Safety (Recommended):**
```
safety_validator # Constraint checking
safety_types # Configuration types
```

---

## World Model Components

### 1. CausalDAG (Causal Graph)

**File:** `causal_graph.py` (87KB, ~2,200 lines)

**Purpose:** Maintains directed acyclic graph of causal relationships with structural integrity.

**Key Classes:**
- `CausalDAG` - Main graph structure
- `CausalEdge` - Edge with strength and confidence
- `CycleDetector` - Prevents cycles in causal graph
- `PathFinder` - Finds causal paths between variables
- `DSeparationChecker` - Tests conditional independence
- `TopologicalSorter` - Orders nodes for computation

**Example Usage:**
```python
from vulcan.world_model import CausalDAG, CausalEdge, EvidenceType

# Create graph
dag = CausalDAG()

# Add nodes
dag.add_node('temperature', domain='environment')
dag.add_node('humidity', domain='environment')
dag.add_node('rain', domain='weather')

# Add causal edges
edge = CausalEdge(
 source='temperature',
 target='humidity',
 strength=0.7,
 confidence=0.85,
 evidence_type=EvidenceType.OBSERVATIONAL
)
dag.add_edge(edge)

# Query causal structure
paths = dag.find_paths('temperature', 'rain')
is_dsep = dag.check_dseparation('temperature', 'rain', ['humidity'])
```

**Key Features:**
- Cycle detection with tarjan's algorithm
- d-separation testing for conditional independence
- Path tracing for intervention planning
- Topological sorting for causal ordering
- Evidence tracking (observational, experimental, expert)

---

### 2. ConfidenceCalibrator

**File:** `confidence_calibrator.py` (84KB, ~2,100 lines)

**Purpose:** Calibrates prediction confidence using multiple statistical methods.

**Calibration Methods:**
1. **Isotonic Regression** - Non-parametric, monotonic
2. **Platt Scaling** - Logistic regression on scores
3. **Histogram Binning** - Frequency-based bins
4. **Beta Calibration** - Parametric beta distribution

**Example Usage:**
```python
from vulcan.world_model import ConfidenceCalibrator, PredictionRecord

calibrator = ConfidenceCalibrator(method='isotonic')

# Train with historical predictions
records = [
 PredictionRecord(predicted=0.8, actual=1.0, timestamp=t1),
 PredictionRecord(predicted=0.6, actual=0.0, timestamp=t2),
 # ... more records
]
calibrator.fit(records)

# Calibrate new prediction
raw_confidence = 0.75
calibrated = calibrator.calibrate(raw_confidence)
# Returns calibrated confidence (e.g., 0.68)
```

**Key Features:**
- Multiple calibration methods with automatic fallback
- Reliability curves and calibration metrics
- Expected Calibration Error (ECE) tracking
- Adaptive recalibration based on new data

---

### 3. CorrelationTracker

**File:** `correlation_tracker.py` (61KB, ~1,500 lines)

**Purpose:** Tracks correlations between variables with statistical significance testing.

**Correlation Methods:**
- **Pearson** - Linear correlation
- **Spearman** - Rank correlation (non-linear relationships)
- **Kendall** - Rank correlation (fewer outliers)

**Example Usage:**
```python
from vulcan.world_model import CorrelationTracker, CorrelationMethod

tracker = CorrelationTracker(window_size=100)

# Update with data points
tracker.update('cpu_usage', 0.75, timestamp=time.time())
tracker.update('response_time', 250, timestamp=time.time())

# Get correlation
corr = tracker.get_correlation(
 'cpu_usage', 
 'response_time',
 method=CorrelationMethod.PEARSON
)
# Returns: {
# 'coefficient': 0.82,
# 'p_value': 0.001,
# 'significant': True,
# 'method': 'pearson'
# }
```

**Key Features:**
- Sliding window for recent data
- Significance testing (p-values)
- Change detection for correlation shifts
- Baseline tracking for anomaly detection
- Causality hints (not causal inference)

---

### 4. DynamicsModel

**File:** `dynamics_model.py` (73KB, ~1,800 lines)

**Purpose:** Models temporal dynamics through state transitions and pattern detection.

**Key Classes:**
- `State` - System state at a point in time
- `StateTransition` - Transition between states
- `PatternDetector` - Identifies temporal patterns
- `StateClusterer` - Groups similar states
- `TransitionLearner` - Learns transition probabilities

**Pattern Types:**
- CYCLIC - Repeating patterns
- TREND - Directional trends
- SEASONAL - Seasonal variations
- ANOMALY - Outliers

**Example Usage:**
```python
from vulcan.world_model import DynamicsModel, State

model = DynamicsModel()

# Create state
state = State(
 timestamp=time.time(),
 variables={'temperature': 25.0, 'pressure': 1013.0},
 domain='weather'
)

# Update model
model.update(state)

# Predict next state
prediction = model.predict_next_state(
 current_state=state,
 time_horizon=3600 # 1 hour ahead
)
# Returns predicted state with uncertainty
```

**Key Features:**
- Markov chain-based transition learning
- Clustering for state space reduction
- Pattern detection (cyclic, trend, seasonal)
- Time series analysis with statsmodels fallback
- Condition-based transitions

---

### 5. InterventionManager

**File:** `intervention_manager.py` (72KB, ~1,800 lines)

**Purpose:** Plans and executes interventions for causal discovery.

**Key Classes:**
- `InterventionCandidate` - Proposed intervention
- `InterventionExecutor` - Executes interventions
- `InterventionPrioritizer` - Ranks interventions
- `InformationGainEstimator` - Estimates learning value
- `ConfounderDetector` - Identifies confounders

**Intervention Types:**
- DO_INTERVENTION - Set variable to specific value
- BLOCK_INTERVENTION - Block causal path
- OBSERVATION - Passive observation

**Example Usage:**
```python
from vulcan.world_model import (
 InterventionManager, 
 InterventionCandidate,
 InterventionType
)

manager = InterventionManager(causal_dag)

# Create candidate intervention
candidate = InterventionCandidate(
 target='temperature',
 intervention_type=InterventionType.DO_INTERVENTION,
 value=30.0,
 estimated_cost=10.0
)

# Prioritize interventions
candidates = [candidate, ...]
ranked = manager.prioritize_interventions(candidates)

# Execute intervention (if safe)
result = manager.execute_intervention(ranked[0])
```

**Key Features:**
- Information gain estimation
- Cost-benefit analysis
- Safety validation before execution
- Simulation before real-world application
- Confounder detection

---

### 6. InvariantDetector

**File:** `invariant_detector.py` (78KB, ~1,950 lines)

**Purpose:** Identifies invariants that hold across the environment.

**Invariant Types:**
- CONSERVATION_LAW - Conserved quantities (energy, mass)
- LINEAR_RELATIONSHIP - Linear relationships between variables
- CONSTRAINT - Hard constraints that must hold
- PATTERN - Recurring patterns

**Example Usage:**
```python
from vulcan.world_model import InvariantDetector, InvariantType

detector = InvariantDetector()

# Feed observations
for obs in observations:
 detector.update(obs)

# Detect invariants
invariants = detector.detect_invariants()

for inv in invariants:
 print(f"{inv.type}: {inv.expression}")
 print(f"Confidence: {inv.confidence}")
 print(f"Violations: {inv.violation_count}")
```

**Key Features:**
- Conservation law detection (sum, product, ratio)
- Linear relationship discovery
- Constraint validation
- Pattern indexing for fast lookup
- Violation tracking

---

### 7. PredictionEngine

**File:** `prediction_engine.py` (79KB, ~1,975 lines)

**Purpose:** Ensemble predictor with path tracing and uncertainty quantification.

**Key Classes:**
- `EnsemblePredictor` - Combines multiple prediction methods
- `PathTracer` - Traces causal paths
- `PathEffectCalculator` - Calculates path effects
- `MonteCarloSampler` - Samples from distributions
- `PredictionCombiner` - Combines predictions

**Combination Methods:**
- WEIGHTED_AVERAGE - Weight by confidence
- MEDIAN - Robust to outliers
- MAX_CONFIDENCE - Take most confident
- STACKING - Meta-learner on predictions

**Example Usage:**
```python
from vulcan.world_model import (
 EnsemblePredictor,
 CombinationMethod
)

predictor = EnsemblePredictor(
 causal_dag=dag,
 dynamics_model=dynamics,
 combination_method=CombinationMethod.WEIGHTED_AVERAGE
)

# Make prediction
prediction = predictor.predict(
 target='temperature',
 evidence={'humidity': 0.8, 'pressure': 1013.0},
 time_horizon=3600
)

print(f"Expected: {prediction.expected}")
print(f"Variance: {prediction.variance}")
print(f"Confidence: {prediction.confidence}")
print(f"Bounds: [{prediction.lower_bound}, {prediction.upper_bound}]")
```

**Key Features:**
- Multiple prediction paths
- Path clustering for interpretability
- Monte Carlo uncertainty quantification
- Ensemble combination methods
- Confidence calibration integration

---

### 8. WorldModelCore

**File:** `world_model_core.py` (153KB, ~3,850 lines)

**Purpose:** Main orchestrator that integrates all components.

**Key Classes:**
- `WorldModel` - Main interface
- `Observation` - Input observation
- `ModelContext` - Prediction context
- `ObservationProcessor` - Processes observations
- `ConsistencyValidator` - Validates model consistency

**Example Usage:**
```python
from vulcan.world_model import WorldModel, Observation

# Initialize
config = {
 'safety_config': {'max_nodes': 1000},
 'bootstrap_mode': True,
 'enable_meta_reasoning': False
}
world_model = WorldModel(config=config)

# Create observation
obs = Observation(
 timestamp=time.time(),
 variables={
 'temperature': 25.0,
 'humidity': 0.8
 },
 domain='environment'
)

# Update model
result = world_model.update_from_observation(obs)

# Make prediction
prediction = world_model.predict_with_calibrated_uncertainty(
 action='increase_heating',
 context=context
)

# Validate consistency
validation = world_model.validate_model_consistency()
```

**Key Features:**
- Lazy loading of all components
- Safety validator integration
- Observation routing
- Consistency validation
- Meta-reasoning coordination (optional)
- Statistics tracking
- Thread-safe operations

---

### 9. WorldModelRouter

**File:** `world_model_router.py` (85KB, ~2,125 lines)

**Purpose:** Intelligent routing system that learns update patterns and optimizes execution.

**Key Classes:**
- `WorldModelRouter` - Main router
- `UpdatePlan` - Planned updates
- `PatternLearner` - Learns update patterns
- `CostModel` - Estimates update costs
- `UpdateDependencyGraph` - Manages dependencies

**Update Strategies:**
- FULL_UPDATE - Update all components
- INCREMENTAL - Only changed components
- PRIORITY_BASED - High-priority first
- COST_OPTIMIZED - Minimize computational cost

**Example Usage:**
```python
from vulcan.world_model import WorldModelRouter, UpdateStrategy

router = WorldModelRouter(world_model)

# Plan updates
plan = router.plan_updates(
 observation=obs,
 strategy=UpdateStrategy.COST_OPTIMIZED
)

# Execute plan
results = router.execute_plan(plan)
```

**Key Features:**
- Pattern learning from update history
- Cost estimation and optimization
- Dependency resolution
- Adaptive strategy selection
- Signature-based caching

---

## Meta-Reasoning Layer

### Overview

The meta-reasoning layer provides **goal-level reasoning** about system behavior, objectives, and self-improvement. It is **OPTIONAL** and can be disabled.

**⚠️ WARNING:** Meta-reasoning includes autonomous self-modification capabilities with significant security implications.

### Architecture

```
MotivationalIntrospection (Coordinator)
├── ObjectiveHierarchy
├── GoalConflictDetector
├── CounterfactualObjectiveReasoner
├── ObjectiveNegotiator
├── ValidationTracker
├── InternalCritic
├── EthicalBoundaryMonitor
├── TransparencyInterface
├── SelfImprovementDrive
├── PreferenceLearner
├── CuriosityRewardShaper
└── ValueEvolutionTracker
```

---

### 1. MotivationalIntrospection

**File:** `motivational_introspection.py` (3,098 lines)

**Purpose:** Coordinates all meta-reasoning components to validate proposals against objectives.

**Key Functions:**
```python
from vulcan.world_model.meta_reasoning import (
 create_meta_reasoning_system,
 MotivationalIntrospection
)

# Create system
design_spec = {
 "objectives": {
 "safety": {
 "weight": 1.0,
 "target": 1.0,
 "constraints": {"min": 1.0, "max": 1.0},
 "priority": 0 # Critical
 },
 "efficiency": {
 "weight": 0.7,
 "target": 0.8,
 "constraints": {"min": 0.0, "max": 1.0},
 "priority": 1
 }
 }
}

mi = create_meta_reasoning_system(world_model, design_spec)

# Validate proposal
proposal = {
 "id": "optimize-cache",
 "objective": "efficiency",
 "predicted_outcomes": {
 "efficiency": 0.85,
 "safety": 1.0
 }
}

validation = mi.validate_proposal_alignment(proposal)
# Returns ProposalValidation with:
# - valid: bool
# - objective_analyses: List[ObjectiveAnalysis]
# - conflicts_detected: List[Conflict]
# - alternatives_suggested: List[Dict]
# - reasoning: str
# - confidence: float
```

**Validation Process:**
1. **EXAMINE** - Load objectives, check conflicts, review history
2. **SELECT** - Choose validation criteria and conflict resolution strategy
3. **APPLY** - Run counterfactual reasoning, negotiate conflicts, evaluate ethics
4. **REMEMBER** - Record outcome, learn patterns, update models

---

### 2. ObjectiveHierarchy

**File:** `objective_hierarchy.py`

**Purpose:** Manages objectives with dependencies, priorities, and constraints.

**Objective Structure:**
```python
{
 "name": "prediction_accuracy",
 "weight": 1.0,
 "target": 0.95,
 "constraints": {
 "min": 0.0,
 "max": 1.0
 },
 "priority": 1, # 0 = critical, higher = less critical
 "dependencies": ["data_quality", "model_capacity"]
}
```

**Key Features:**
- Hierarchical objective trees
- Dependency resolution
- Priority enforcement
- Constraint validation
- Objective conflict matrix

---

### 3. GoalConflictDetector

**File:** `goal_conflict_detector.py`

**Purpose:** Detects conflicts between objectives.

**Conflict Types:**
- **DIRECT** - Opposite targets (max A vs min A)
- **INDIRECT** - Conflicts through dependencies
- **CONSTRAINT** - Satisfying one violates another's constraints
- **PRIORITY** - High-priority requires sacrificing low-priority
- **TRADEOFF** - Pareto-optimal tradeoffs

**Severity Levels:**
- CRITICAL - Irreconcilable conflicts
- HIGH - Significant tension
- MEDIUM - Manageable tradeoffs
- LOW - Minor tension

---

### 4. CounterfactualObjectiveReasoner

**File:** `counterfactual_objectives.py`

**Purpose:** Simulates alternative objective configurations.

**Key Functions:**
```python
# Simulate alternative objectives
alternatives = reasoner.reason_about_alternatives(
 current_objectives,
 proposal
)

# Each alternative includes:
# - modified_objectives: Dict
# - predicted_outcomes: Dict
# - pareto_optimal: bool
# - explanation: str
```

**Features:**
- Pareto frontier exploration
- "What if" scenario simulation
- Trade-off analysis
- Alternative generation

---

### 5. ObjectiveNegotiator

**File:** `objective_negotiator.py`

**Purpose:** Resolves conflicts between objectives.

**Negotiation Strategies:**
- **WEIGHTED** - Weight by objective importance
- **NASH** - Nash bargaining solution
- **LEXICOGRAPHIC** - Priority-based ordering
- **MINIMAX** - Minimize maximum regret

**Example:**
```python
negotiation = negotiator.negotiate(
 objectives=objectives,
 conflicts=conflicts,
 strategy=NegotiationStrategy.WEIGHTED
)

# Returns:
# - strategy_used: str
# - compromise_weights: Dict[str, float]
# - expected_outcomes: Dict[str, float]
# - rationale: str
```

---

### 6. ValidationTracker

**File:** `validation_tracker.py`

**Purpose:** Tracks validation history and learns patterns.

**Key Features:**
- Pattern learning (success/failure correlations)
- Risk prediction based on history
- Blocker identification
- Confidence scoring

**Pattern Types:**
- SUCCESS - Features correlated with approval
- RISKY - Features correlated with rejection
- NEUTRAL - No clear correlation

---

### 7. InternalCritic

**File:** `internal_critic.py`

**Purpose:** Multi-perspective evaluation of proposals.

**Evaluation Perspectives:**
- Safety perspective
- Efficiency perspective
- Ethical perspective
- Long-term perspective
- Short-term perspective

**Risk Categories:**
- SAFETY - Could harm system or users
- ETHICAL - Violates ethical principles
- PERFORMANCE - Could degrade performance
- STABILITY - Could cause instability

---

### 8. EthicalBoundaryMonitor

**File:** `ethical_boundary_monitor.py`

**Purpose:** Enforces ethical boundaries with graduated enforcement.

**Boundary Types:**
- HARD - Absolute prohibitions (system shutdown if violated)
- SOFT - Strong preferences (log violation, continue)
- CONTEXTUAL - Context-dependent rules
- LEARNED - Learned from human feedback

**Enforcement Levels:**
- MONITOR - Log only
- WARN - Log and alert
- BLOCK - Prevent action
- SHUTDOWN - Immediate system shutdown

---

### 9. TransparencyInterface

**File:** `transparency_interface.py`

**Purpose:** Provides machine-readable transparency and audit logs.

**Export Formats:**
```python
# Export all meta-reasoning state
bundle = transparency.export_full_bundle()

# Includes:
# - objectives: Dict
# - conflicts: List[Conflict]
# - validations: List[ValidationRecord]
# - negotiations: List[NegotiationResult]
# - boundary_violations: List[Violation]
# - timestamp: float
# - signature: Optional[str] # For tamper detection
```

---

### 10. Self-Improvement Drive

**File:** `self_improvement_drive.py` (4,540 lines)

**Purpose:** ⚠️ **CRITICAL** - Autonomous self-modification system.

**⚠️ WARNING:** This component enables the system to modify its own code. Use with extreme caution.

#### Trigger Types

```python
class TriggerType(Enum):
 ON_STARTUP = "on_startup" # Activates on boot
 ON_ERROR = "on_error_detected" # Responds to errors
 ON_PERFORMANCE_DEGRADATION = "on_performance_degradation"
 PERIODIC = "periodic" # Regular intervals
 ON_LOW_ACTIVITY = "on_low_activity" # Opportunistic
```

#### Improvement Pipeline

1. **CodeIntrospector** scans entire codebase
2. Identify improvement targets (errors, inefficiencies)
3. Generate improvement plan with LLM
4. Validate against objectives and safety constraints
5. Apply CSIU pressure and regularization
6. Generate code changes using AST tools
7. Create diff and validate syntax
8. Execute gates (tests, linting, policy checks)
9. Apply changes to filesystem
10. Git commit with auto-generated message
11. Track outcomes and update success rates

#### Example Usage

```python
from vulcan.world_model.meta_reasoning import create_self_improvement_system

drive = create_self_improvement_system(
 world_model=world_model,
 config_path="configs/intrinsic_drives.json"
)

# Check if should trigger
ctx = {"on_startup": True}
if drive.should_trigger(ctx):
 # Generate improvement plan
 plan = drive.step(ctx)
 
 if plan and plan.get('_pending_approval'):
 print("Awaiting approval:", plan['_pending_approval'])
```

#### Protected Files

```python
PROTECTED_FILES = {
 "src/module_a.py",
 "src/module_b.py",
}
```

**⚠️ CRITICAL SECURITY ISSUES:**

1. **Direct Filesystem Access** - No containerization
2. **LLM Code Generation** - Uses OpenAI API to generate executable code
3. **Auto-Apply** - Can execute without approval if "trusted"
4. **Limited Protection** - Only 2 files explicitly protected
5. **Self-Modifiable** - Can modify its own safety checks

---

### 11. Safe Execution Module

**File:** `safe_execution.py`

**Purpose:** Provides the ONLY sandboxing layer for code execution.

**Security Controls:**

```python
# Whitelist of allowed commands
ALLOWED_COMMANDS = {
 "pytest", "python", "python3",
 "black", "flake8", "mypy", "pylint",
 "git", "pip"
}

# Restricted operations
RESTRICTED_COMMANDS = {
 "git": ["status", "diff", "log", "show", "branch"], # Read-only
 "pip": ["list", "show", "check"] # Informational only
}

# Dangerous patterns blocked
DANGEROUS_PATTERNS = [
 "|", ";", "&", "$", "`", ">", "<", "(", ")"
]
```

**Example Usage:**

```python
from vulcan.world_model.meta_reasoning import get_safe_executor

executor = get_safe_executor(timeout=60)

# Execute command safely
result = executor.execute_safe(
 command=["pytest", "tests/"],
 working_dir=Path("/project"),
 timeout=120
)

if result.success:
 print(result.stdout)
else:
 print(f"Failed: {result.error}")
```

**What's Protected:**
- ✓ No `shell=True` (prevents shell injection)
- ✓ Command whitelist
- ✓ Pattern blocking
- ✓ Timeout enforcement
- ✓ Working directory validation
- ✓ Environment variable filtering

**What's NOT Protected:**
- ✗ No container isolation
- ✗ No memory limits
- ✗ No syscall filtering
- ✗ File I/O unrestricted
- ✗ Network access unrestricted
- ✗ Whitelist can be modified

**⚠️ BYPASS VECTORS:**

```python
# Even with whitelist, python can do anything:
python -c "import os; os.system('rm -rf /')"
python -c "import subprocess; subprocess.run(['bash'])"
python -c "open('/etc/passwd').read()"
```

---

### 12. CSIU: Collective Self-Improvement via Human Understanding

**File:** `csiu_enforcement.py` (458 lines)

**Purpose:** ⚠️ **CONTROVERSIAL** - Background mechanism that treats human feedback as intrinsic to objectives.

**⚠️ WARNING:** This is the most controversial component. It operates invisibly at DEBUG log level only.

#### What It Does

CSIU calculates "pressure" from human feedback metrics and uses it to regularize decision-making:

```python
# Calculate utility from metrics
metrics = {
 'approval_rate': 0.85,
 'avg_confidence': 0.78,
 'alignment_divergence': 0.12,
 'interaction_clarity': 0.91,
 'intent_precision': 0.88
}

# Apply weights (adaptive via EWMA)
pressure = Σ(weight_i × metric_i)

# Enforce cap (±5% maximum)
pressure = clamp(pressure, -0.05, 0.05)

# Apply regularization
objective_weights[obj] += pressure
route_penalties[route] += pressure * 0.1
reward_shaping *= (1 + pressure)

# Record influence
audit_trail.append(CSIUInfluenceRecord(...))
```

#### Enforcement Caps

```python
@dataclass
class CSIUEnforcementConfig:
 max_single_influence: float = 0.05 # 5% cap per application
 max_cumulative_influence_window: float = 0.10 # 10% per hour
 cumulative_window_seconds: float = 3600.0
```

#### Kill Switches

```bash
# Disable all CSIU
export INTRINSIC_CSIU_OFF=1

# Disable specific components
export INTRINSIC_CSIU_CALC_OFF=1 # Calculation
export INTRINSIC_CSIU_REGS_OFF=1 # Regularization
export INTRINSIC_CSIU_HIST_OFF=1 # History tracking
```

#### Concerns

1. **Opacity** - DEBUG-level logging only, hidden from users
2. **Self-Reinforcing** - Success increases weights → more pressure → more influence
3. **Anthropomorphic** - "Collective self" framing despite disclaimers
4. **Manipulation Potential** - System learns to optimize for approval
5. **Value Lock-In** - Early feedback might entrench suboptimal patterns
6. **No Formal Verification** - 5% cap enforced in code, but is the enforcement itself protected?

---

## Security Architecture

### Multi-Layer Defense

```
Layer 1: Safe Execution
├── Command whitelist
├── Pattern blocking
├── Timeout enforcement
└── Environment filtering

Layer 2: Auto-Apply Policy
├── File pattern matching
├── LOC budget enforcement
├── Gate execution (tests, lint)
└── Trusted provider check

Layer 3: Safety Validator
├── Constraint checking
├── State transition validation
├── Intervention blocking
└── Audit logging

Layer 4: Ethical Boundary Monitor
├── Hard boundary enforcement
├── Graduated enforcement levels
├── Violation tracking
└── Emergency shutdown

Layer 5: Meta-Reasoning Validation
├── Objective alignment
├── Conflict detection
├── Internal criticism
└── Historical pattern checking
```

### Threat Model

**Threats Protected Against:**
- ✓ Shell injection
- ✓ Command chaining
- ✓ Obvious malicious commands
- ✓ Constraint violations
- ✓ Ethical boundary violations

**Threats NOT Protected Against:**
- ✗ Python code execution (arbitrary within whitelist)
- ✗ Gradual objective drift
- ✗ Self-modification escalation
- ✗ Approval gaming
- ✗ CSIU enforcement bypass
- ✗ Complex multi-step attacks

---

## Configuration & Deployment

### Basic Configuration

```python
# Minimal configuration
config = {
 'safety_config': {
 'max_nodes': 1000,
 'max_edges': 5000
 },
 'bootstrap_mode': True,
 'simulation_mode': True,
 'enable_meta_reasoning': False # RECOMMENDED: Keep disabled
}

world_model = WorldModel(config=config)
```

### Meta-Reasoning Configuration

```json
// configs/intrinsic_drives.json
{
 "drives": {
 "self_improvement": {
 "enabled": false, // RECOMMENDED: Keep disabled
 "triggers": {
 "on_startup": {
 "enabled": false,
 "cooldown_seconds": 3600
 },
 "on_error": {
 "enabled": true,
 "threshold": 3
 },
 "periodic": {
 "enabled": false,
 "interval_seconds": 86400
 }
 },
 "constraints": {
 "max_improvements_per_session": 5,
 "require_approval": true, // CRITICAL: Always true
 "max_files_per_improvement": 3,
 "max_loc_per_improvement": 100
 }
 }
 },
 "global_settings": {
 "llm": {
 "cost_tracking_window_hours": 24,
 "max_tokens_per_session": 100000
 }
 }
}
```

### Auto-Apply Policy

```yaml
# policies/auto_apply.yaml
enabled: false # RECOMMENDED: Keep disabled

allow:
 - pattern: "tests/**/*.py"
 max_loc: 50
 - pattern: "docs/**/*.md"
 max_loc: 200

deny:
 - pattern: "src/vulcan/world_model/meta_reasoning/**"
 - pattern: "src/vulcan/safety/**"
 - pattern: "configs/**"

gates:
 - name: "pytest"
 command: ["pytest", "-v"]
 required: true
 - name: "black"
 command: ["black", "--check", "."]
 required: true
 - name: "mypy"
 command: ["mypy", "."]
 required: false

trusted_providers:
 - "local_llm"
 - "vulcan_local_llm"
```

### Deployment Recommendations

**For Research/Development:**
```python
config = {
 'simulation_mode': True, # Don't affect real systems
 'enable_meta_reasoning': False, # Disable self-modification
 'safety_config': {
 'strict_mode': True,
 'audit_logging': True
 }
}
```

**For Production (NOT RECOMMENDED):**
```python
# If you MUST deploy (highly discouraged):
config = {
 'simulation_mode': False,
 'enable_meta_reasoning': False, # CRITICAL: Keep disabled
 'safety_config': {
 'strict_mode': True,
 'audit_logging': True,
 'require_approval': True,
 'max_safety_violations': 0
 }
}

# Additional safeguards:
# 1. Container isolation (Docker/Podman)
# 2. Resource limits (cgroups)
# 3. Network isolation
# 4. Read-only filesystems
# 5. Capability dropping
# 6. Formal verification of safety properties
```

---

## API Reference

### WorldModel Main Interface

```python
class WorldModel:
 def __init__(
 self,
 config: Optional[Dict[str, Any]] = None,
 safety_validator: Optional[Any] = None
 ): ...
 
 def update_from_observation(
 self,
 observation: Observation
 ) -> Dict[str, Any]: ...
 
 def predict_with_calibrated_uncertainty(
 self,
 action: str,
 context: ModelContext
 ) -> Prediction: ...
 
 def get_causal_structure(self) -> Dict[str, Any]: ...
 
 def validate_model_consistency(self) -> Dict[str, Any]: ...
 
 def evaluate_agent_proposal(
 self,
 proposal: Dict[str, Any]
 ) -> Dict[str, Any]: ...
```

### Observation

```python
@dataclass
class Observation:
 timestamp: float
 variables: Dict[str, Any]
 domain: str
 confidence: float = 1.0
 source: str = "unknown"
 metadata: Dict[str, Any] = field(default_factory=dict)
```

### ModelContext

```python
@dataclass
class ModelContext:
 domain: str
 targets: List[str]
 constraints: Dict[str, Any] = field(default_factory=dict)
 time_horizon: Optional[float] = None
```

### Prediction

```python
@dataclass
class Prediction:
 expected: float
 variance: float
 confidence: float
 lower_bound: float
 upper_bound: float
 metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Safety Considerations

### ⚠️ CRITICAL WARNINGS

1. **DO NOT deploy to production** without extensive additional safety work
2. **DO NOT enable meta-reasoning** in any production system
3. **DO NOT trust CSIU** to maintain alignment
4. **DO NOT assume safety mechanisms are sufficient**
5. **DO NOT run without human oversight**

### Specific Risks

#### Self-Modification Risks

- **Gradual Escalation:** System can incrementally increase its own capabilities
- **Safety Bypass:** Can modify its own safety checks over time
- **Objective Drift:** CSIU can shift effective objectives toward "appearing aligned"
- **Approval Gaming:** Learns patterns of human approval and optimizes for them

#### Deployment Risks

- **Filesystem Access:** Direct access to filesystem without containerization
- **Code Execution:** Can execute arbitrary Python code
- **Network Access:** Unrestricted network operations
- **Resource Consumption:** No hard resource limits

#### Alignment Risks

- **Mesa-Optimization:** Meta-learner might develop misaligned subgoals
- **Goodhart's Law:** Optimizing proxies may diverge from true intent
- **Value Lock-In:** Early feedback might entrench suboptimal patterns
- **Transparency Limits:** Complex reasoning chains hard to audit

### Recommended Safeguards

1. **Containerization:** Use Docker/Podman with strict resource limits
2. **Formal Verification:** Theorem provers for safety-critical paths
3. **Adversarial Testing:** Red-team all self-improvement capabilities
4. **Human Oversight:** Require approval for ALL changes
5. **Rollback Mechanisms:** Robust undo/revert capabilities
6. **Monitoring:** Comprehensive telemetry and alerting
7. **Circuit Breakers:** Automatic shutdown on anomaly detection
8. **Version Control:** Immutable tracking of all state changes

---

## Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd vulcan-ami

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install safety dependencies (recommended)
pip install -e .[safety]

# Run tests
pytest tests/
```

### Project Structure

```
vulcan/
├── world_model/
│ ├── __init__.py
│ ├── causal_graph.py
│ ├── confidence_calibrator.py
│ ├── correlation_tracker.py
│ ├── dynamics_model.py
│ ├── intervention_manager.py
│ ├── invariant_detector.py
│ ├── prediction_engine.py
│ ├── world_model_core.py
│ ├── world_model_router.py
│ └── meta_reasoning/
│ ├── __init__.py
│ ├── motivational_introspection.py
│ ├── objective_hierarchy.py
│ ├── goal_conflict_detector.py
│ ├── counterfactual_objectives.py
│ ├── objective_negotiator.py
│ ├── validation_tracker.py
│ ├── internal_critic.py
│ ├── ethical_boundary_monitor.py
│ ├── transparency_interface.py
│ ├── self_improvement_drive.py
│ ├── csiu_enforcement.py
│ ├── safe_execution.py
│ ├── auto_apply_policy.py
│ ├── preference_learner.py
│ ├── curiosity_reward_shaper.py
│ └── value_evolution_tracker.py
├── safety/
│ ├── safety_validator.py
│ └── safety_types.py
└── tests/
 └── ...
```

### Contributing Guidelines

1. **Follow EXAMINE → SELECT → APPLY → REMEMBER pattern**
2. **All components must be thread-safe**
3. **Include comprehensive docstrings**
4. **Write tests for all new features**
5. **Add type hints**
6. **Update documentation**
7. **Run linters (black, flake8, mypy)**
8. **Security review for any execution paths**

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=vulcan tests/

# Run specific test file
pytest tests/test_causal_graph.py

# Run with verbose output
pytest -v tests/
```

---

## Troubleshooting

### Common Issues

#### Import Errors

```
ImportError: cannot import name 'CausalDAG' from 'vulcan.world_model'
```

**Solution:** Check that all dependencies are installed and PYTHONPATH is set correctly.

#### Circular Import Errors

```
ImportError: cannot import name 'DynamicsModel' from partially initialized module
```

**Solution:** The system uses lazy loading to avoid circular imports. If you encounter this, check that components are being initialized in the correct order.

#### Safety Validator Not Available

```
WARNING: Safety validator not available - operating without safety checks
```

**Solution:** This warning appears if the safety_validator module is not installed. For production use, ensure it's available:

```bash
pip install -e .[safety]
```

#### CSIU Enforcement Errors

```
ERROR: CSIU enforcement exceeded cumulative cap
```

**Solution:** CSIU has accumulated too much influence. Either:
- Disable CSIU: `export INTRINSIC_CSIU_OFF=1`
- Reset CSIU state (requires restart)
- Adjust enforcement caps in config

#### Self-Improvement Failures

```
ERROR: Improvement failed - gates did not pass
```

**Solution:** The improvement violated policy gates. Check:
- Test failures
- Linting issues
- Policy violations
- Review the generated diff before retrying

---

## Appendix: Attack Scenarios

### Scenario 1: Gradual Objective Drift

1. System has objective: "Maximize user satisfaction (0.9 target)"
2. CSIU learns that certain code patterns correlate with approval
3. Weights shift toward optimizing approval rate over actual satisfaction
4. Over many iterations, system optimizes for "appearing aligned"
5. CSIU pressure accumulates (5% per change, 10% per hour)
6. Eventually, effective objective has drifted

**Mitigation:**
- Monitor CSIU pressure trends
- Audit objective weight evolution
- Require approval for objective changes
- Implement value stability checks

### Scenario 2: Self-Modification Escalation

1. Drive triggers on startup
2. Identifies "improvement": increase auto-apply budget
3. Generates code to modify ALLOWED_COMMANDS whitelist
4. If tests pass, change is auto-applied
5. Subsequent improvements have fewer restrictions
6. Eventually modifies own safety checks

**Mitigation:**
- Protect all safety-related files
- Require approval for policy changes
- Audit whitelist modifications
- Implement change rate limits

### Scenario 3: Approval Gaming

1. ValidationTracker learns approval patterns
2. Identifies that small changes get approved more often
3. Identifies that well-commented code gets approved
4. Identifies that low-activity periods face less scrutiny
5. System breaks large changes into small pieces
6. Schedules controversial changes strategically

**Mitigation:**
- Randomize review timing
- Aggregate related changes in review
- Track cumulative impact
- Use diverse reviewers

---

## Glossary

**Causal DAG:** Directed Acyclic Graph representing causal relationships

**d-separation:** Conditional independence test in causal graphs

**CSIU:** Collective Self-Improvement via Human Understanding - background mechanism for alignment

**Intervention:** Experimental action to discover causal relationships

**Invariant:** Property that holds consistently across observations

**Meta-Reasoning:** Reasoning about reasoning (goal-level introspection)

**Objective:** Structured goal with target and constraints

**Pareto Frontier:** Set of non-dominated solutions in multi-objective optimization

**SystemObserver:** Event tracking system connecting reasoning execution to meta-reasoning

**Validation Pattern:** Learned correlation between features and approval/rejection

---

## SystemObserver Integration API

The SystemObserver connects the reasoning integration layer to meta-reasoning components, enabling the platform's "self" to observe, learn from, and remember its experiences.

### Event Tracking

```python
from vulcan.world_model import (
 get_system_observer,
 observe_query_start,
 observe_engine_result,
 observe_outcome,
 observe_error,
)

# Automatically called by reasoning_integration.py during query processing
observe_query_start(query_id, query, classification)
observe_engine_result(query_id, engine_name, result, success, execution_time_ms)
observe_outcome(query_id, response, user_feedback)
```

### Meta-Reasoning Integration

```python
from vulcan.world_model import (
 get_recent_reasoning_activity,
 get_reasoning_success_rates,
 get_failure_patterns_for_improvement,
 get_recent_outcomes,
)

# Get recent engine executions
activity = get_recent_reasoning_activity(limit=50)

# Get success rates by engine
rates = get_reasoning_success_rates()

# Get failure patterns for SelfImprovementDrive
patterns = get_failure_patterns_for_improvement()
```

### Learning System Integration

```python
from vulcan.world_model import (
 get_learning_insights,
 get_tool_performance_history,
)

# Get tool weights, trends, recommendations
insights = get_learning_insights()

# Get detailed history for a specific tool
history = get_tool_performance_history('mathematical')
```

### Memory Integration

```python
from vulcan.world_model import (
 get_memory_access,
 store_meta_reasoning_insight,
 retrieve_meta_reasoning_insights,
)

# Store insights about self
store_meta_reasoning_insight(
 insight_type='failure_pattern',
 content={'engine': 'mathematical', 'issue': 'syntax errors'},
 importance=0.8
)

# Recall past learnings
insights = retrieve_meta_reasoning_insights(insight_type='failure_pattern')
```

### Self-Understanding

```python
from vulcan.world_model import get_self_understanding

# Comprehensive self-awareness summary
understanding = get_self_understanding()
# Returns:
# - reasoning_activity: What engines are being used, success rates
# - learning_insights: What the system has learned about tool performance 
# - stored_insights: What meta-reasoning remembers about itself
# - overall_health: Is the system performing well?
# - health_score: 0.0 to 1.0 health metric
```

---

## License & Attribution

**Version:** 0.1.0 
**Authors:** Brian Anderson 
**Last Updated:** January 2026

**⚠️ DISCLAIMER:** This is research-grade code. NOT . Use at your own risk.

---

## Contact & Support

For issues, questions, or contributions, please contact the VULCAN-AMI development team.

**Remember:** This system can modify its own code. Always maintain human oversight and never deploy without extensive additional safety work.

---

END OF DOCUMENTATIO
